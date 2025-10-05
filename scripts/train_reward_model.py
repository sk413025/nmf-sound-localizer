#!/usr/bin/env python3
"""Train a small Reward Model (RM) to approximate physics rewards for K-step completions.

RM = AutoModelForCausalLMWithValueHead with frozen LM and trainable v_head.
Input: prompt + K-token completion; Target: scalar reward R (proxyA or deltaIS).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import set_seed

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model
from doa_rl.hf.logits_mask import NoRepeatDirectionLogitsProcessor
from doa_rl.assets import load_H, load_W
from nmf_localizer.core.usm_trainer import USMTrainer


def _discover_angles(root: str) -> List[int]:
    base = Path(root)
    angles = sorted(
        {
            int(p.name.split("_")[1])
            for p in base.glob("angle_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        }
    )
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _prepare_samples(args, direction_angles: List[int]) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
    ds = DoADataset(
        args.data_root,
        direction_angles,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(ds, batch_size=1, shuffle=False)
    tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
    prompts: List[str] = []
    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for batch in dl:
        Y_t: torch.Tensor = batch["Y"].squeeze(0)
        Y_np = Y_t.numpy()
        prompt = " ".join(tok(Y_np))
        prompts.append(prompt)
        cache[prompt] = {"Y": Y_t.clone()}
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts, cache


def _resample_F(HF: torch.Tensor, target_F: int) -> torch.Tensor:
    F0, D = HF.shape
    if F0 == target_F:
        return HF
    if F0 > target_F:
        step = F0 / target_F
        idxs = torch.tensor([int(round(i * step)) for i in range(target_F)], dtype=torch.long)
        idxs = torch.clamp(idxs, 0, F0 - 1)
        return HF[idxs, :].clone()
    else:
        step = target_F / max(F0, 1)
        idxs = torch.tensor([int(round(i / step)) for i in range(target_F)], dtype=torch.long)
        idxs = torch.clamp(idxs, 0, F0 - 1)
        return HF[idxs, :].clone()


def _parse_completion(text: str, K: int, direction_angles: List[int]) -> List[int]:
    dirs: List[int] = []
    for m in re.finditer(r"<D_(\d{3})>", text):
        angle = int(m.group(1))
        if angle in direction_angles:
            dirs.append(direction_angles.index(angle))
        if len(dirs) >= K:
            break
    return dirs


def _compute_deltaIS_reward(Y: torch.Tensor, s_hat: torch.Tensor, H: torch.Tensor, selected: List[int]) -> float:
    eps = 1e-12
    Y = Y.clone().float()
    F, N = Y.shape
    H_res = H.float()
    if H_res.shape[0] != F:
        H_res = _resample_F(H_res, F)
    Hs = (H_res * s_hat.view(-1, 1).float())
    def is_div(Ytrue: torch.Tensor, Ymix: torch.Tensor) -> torch.Tensor:
        Yhat = torch.clamp(Ymix.view(-1, 1).expand(F, N), min=eps)
        ratio = torch.clamp(Ytrue, min=eps) / Yhat
        return torch.sum(ratio - torch.log(ratio) - 1.0)
    Y_mix = torch.full((F,), eps, dtype=torch.float32)
    prev = is_div(Y, Y_mix)
    total = 0.0
    for d in selected:
        Y_mix = torch.clamp(Y_mix + Hs[:, d], min=eps)
        cur = is_div(Y, Y_mix)
        total += float(-(cur - prev).item())
        prev = cur
    return total


def main():
    ap = argparse.ArgumentParser(description="Train Reward Model (RM) for PPO K>1 without modifying TRL")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--reward-mode", type=str, choices=["deltaIS"], default="deltaIS")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--out", type=str, default="rm_ckpt.pt")
    args = ap.parse_args()

    set_seed(args.seed)

    direction_angles = _discover_angles(args.data_root)
    prompts, cache = _prepare_samples(args, direction_angles)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Load assets
    H_full, anglesT = load_H(args.tf_path)
    H_full = H_full.float().cpu()
    anglesT = anglesT.cpu().numpy().astype(int).tolist()
    col_idx: List[int] = []
    for a in direction_angles:
        try:
            j = anglesT.index(int(a))
        except ValueError:
            diffs = [abs(a - at) for at in anglesT]
            j = int(np.argmin(diffs))
        col_idx.append(j)
    H = H_full[:, col_idx].contiguous()

    # Precompute ŝ
    if args.w_path.endswith('.npz'):
        W_np = np.load(args.w_path)["W"]
    else:
        W_t = load_W(args.w_path)
        W_np = W_t.cpu().numpy()
    for p in prompts:
        Y_t = cache[p]["Y"]
        try:
            s_hat_np = USMTrainer.compute_content_s_hat(Y=Y_t.numpy(), W=W_np, mode="S1", n_iter=50, l1=0.0)
        except Exception:
            s_hat_np = Y_t.numpy().mean(axis=1)
        cache[p]["s_hat"] = torch.from_numpy(s_hat_np.astype(np.float32))

    # Build RM model (freeze LM, train v_head)
    rm_model, _ = build_value_head_model(tokenizer)
    for p in rm_model.pretrained_model.parameters():
        p.requires_grad = False
    rm_model.score = rm_model.v_head  # TRL/Utils expect .score for reward
    rm_model.train()

    # Construct small dataset by sampling random K-token sets per prompt
    dir_tokens = list(tokenizer.direction_tokens)
    texts: List[str] = []
    rewards: List[float] = []
    for p in prompts:
        # Random no-repeat K selection
        idxs = np.random.choice(len(dir_tokens), size=args.K, replace=False)
        comp = " ".join(dir_tokens[i] for i in idxs)
        text = p + " " + comp
        texts.append(text)
        dirs = [tokenizer.direction_tokens.index(dir_tokens[i]) for i in idxs]
        Y = cache[p]["Y"]
        s_hat = cache[p]["s_hat"]
        R = _compute_deltaIS_reward(Y, s_hat, H, dirs)
        rewards.append(float(R))

    # Tokenize
    enc = tokenizer(texts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    tgt = torch.tensor(rewards, dtype=torch.float32)

    # Simple training loop (MSE of v_head(last_hidden) vs reward)
    opt = torch.optim.Adam([p for p in rm_model.parameters() if p.requires_grad], lr=args.lr)
    loss_fn = nn.MSELoss()
    for epoch in range(args.epochs):
        rm_model.zero_grad(set_to_none=True)
        out = rm_model.pretrained_model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True)
        last_hidden = out.hidden_states[-1]
        value = rm_model.v_head(last_hidden).squeeze(-1)  # (B, T)
        # Take last token's value as scalar
        lengths = attn.sum(dim=1) - 1
        pred = value[torch.arange(value.size(0)), lengths]
        loss = loss_fn(pred, tgt)
        loss.backward()
        opt.step()
        print({"epoch": epoch, "loss": float(loss.item())})

    # Save v_head weights
    torch.save({"v_head": rm_model.v_head.state_dict()}, args.out)
    print(f"Saved RM checkpoint to {args.out}")


if __name__ == "__main__":
    main()

