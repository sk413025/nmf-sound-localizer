#!/usr/bin/env python3
"""TRL GRPO training for K-step direction set selection with physics-based rewards.

This script keeps TRL GRPO algorithm untouched and moves all physics into the
reward function and generation-time logits processors.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids
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


def _prepare_prompts_and_cache(
    args, direction_angles: List[int]
) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
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
        Y_t: torch.Tensor = batch["Y"].squeeze(0)  # (F,N)
        Y_np = Y_t.numpy()
        prompt = " ".join(tok(Y_np))
        prompts.append(prompt)
        cache[prompt] = {"Y": Y_t.clone()}
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts, cache


def _parse_completion(text: str, K: int, direction_angles: List[int]) -> List[int]:
    # Extract <D_XXX> tokens in order
    dirs: List[int] = []
    for m in re.finditer(r"<D_(\d{3})>", text):
        angle = int(m.group(1))
        if angle in direction_angles:
            dirs.append(direction_angles.index(angle))
        if len(dirs) >= K:
            break
    return dirs


def _compute_proxyA_reward(
    Y: torch.Tensor,
    s_hat: torch.Tensor,
    H: torch.Tensor,
    selected: List[int],
) -> float:
    # Greedy standardized A[d_t]; approximate: compute A once per step using current pi
    from doa_rl.advantage import AdvantageComputer
    from nmf_localizer.core.localizer import NMFSoundLocalizer
    from nmf_localizer.config.defaults import NMFConfig

    device = torch.device("cpu")
    ncfg = NMFConfig(device=str(device))
    loc = NMFSoundLocalizer(ncfg)
    adv = AdvantageComputer(loc, W=torch.empty(1, 1), H=H, require_s_hat=True)

    D = H.shape[1]
    pi = torch.zeros(D)
    total = 0.0
    for d in selected:
        out = adv(Y, pi=pi, s_hat=s_hat)
        A = out["A"]
        A_std = (A - A.mean()) / (A.std() + 1e-8)
        total += float(A_std[d].item())
        pi[d] += 1.0
    return total


def _compute_deltaIS_reward(
    Y: torch.Tensor,
    s_hat: torch.Tensor,
    H: torch.Tensor,
    selected: List[int],
) -> float:
    eps = 1e-12
    Y = Y.clone().float()
    F, N = Y.shape
    # Enforce exact frequency grid match; no resampling allowed
    if H.shape[0] != F:
        raise ValueError(
            f"H and Y must have the same number of frequency bins (F). "
            f"Got H.F={int(H.shape[0])} vs Y.F={int(F)}. "
            f"Align STFT config (fs/n_fft/band) to match assets."
        )
    Hs = (H.float() * s_hat.view(-1, 1).float())  # (F,D)
    Y_mix = torch.full((F,), eps, dtype=torch.float32)

    def is_div(Ytrue: torch.Tensor, Ymix: torch.Tensor) -> torch.Tensor:
        Yhat = torch.clamp(Ymix.view(-1, 1).expand(F, N), min=eps)
        ratio = torch.clamp(Ytrue, min=eps) / Yhat
        return torch.sum(ratio - torch.log(ratio) - 1.0)

    prev = is_div(Y, Y_mix)
    total = 0.0
    for d in selected:
        Y_mix = torch.clamp(Y_mix + Hs[:, d], min=eps)
        cur = is_div(Y, Y_mix)
        # Relative improvement per step (scale-invariant)
        rel = (prev - cur) / torch.clamp(prev, min=eps)
        total += float(rel.item())
        prev = cur
    return total


    


def main():
    ap = argparse.ArgumentParser(description="TRL GRPO for K-step direction set selection")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--reward-mode", type=str, choices=["proxyA", "deltaIS"], default="deltaIS")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="Compute device for training/inference (auto→mps>cuda>cpu)")
    args = ap.parse_args()

    set_seed(args.seed)

    # Select device
    dev = args.device
    if dev == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    device = torch.device(dev)
    print(f"Device: {device}")

    # Discover angles and prepare prompts/cache
    direction_angles = _discover_angles(args.data_root)
    prompts, cache = _prepare_prompts_and_cache(args, direction_angles)

    # Load assets
    H_full, anglesT = load_H(args.tf_path)  # (F,D_all), (D_all,)
    H_full = H_full.float().cpu()
    anglesT = anglesT.cpu().numpy().astype(int).tolist()
    # Build column indices in H that match the discovered direction_angles
    col_idx: List[int] = []
    for a in direction_angles:
        # Require exact angle match; no nearest fallback
        if int(a) not in anglesT:
            raise RuntimeError(
                f"Angle {int(a)}° not found in TF angles {anglesT}. "
                f"Ensure dataset angles exactly match TF asset."
            )
        j = anglesT.index(int(a))
        col_idx.append(j)
    # Disallow duplicate mappings which would collapse columns
    if len(set(col_idx)) != len(col_idx):
        raise RuntimeError(
            f"Duplicate TF column mappings detected: {col_idx}. "
            f"This indicates repeated/ambiguous angles; fix assets or dataset."
        )
    H = H_full[:, col_idx].contiguous()

    # Load W as numpy for ŝ estimation
    if args.w_path.endswith('.npz'):
        W_np = np.load(args.w_path)["W"]
    else:
        W_t = load_W(args.w_path)
        W_np = W_t.cpu().numpy()

    # Build tokenizer and model
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"
    policy, _ = build_value_head_model(tokenizer)
    # Compatibility for TRL GRPO expecting certain attributes
    if not hasattr(policy, "warnings_issued"):
        policy.warnings_issued = {}

    # Precompute ŝ for each prompt (strict: no fallback)
    for p in prompts:
        Y_t = cache[p]["Y"]
        s_hat_np = USMTrainer.compute_content_s_hat(
            Y=Y_t.numpy(), W=W_np, mode="S1", n_iter=50, l1=0.0
        )
        cache[p]["s_hat"] = torch.from_numpy(s_hat_np.astype(np.float32))

    # Build dataset for GRPO: requires a 'prompt' column
    ds = Dataset.from_dict({"prompt": prompts})

    # Configure GRPO
    # Ensure generation_batch_size divisible by num_generations
    per_bs = max(1, min(args.batch_size, len(ds)))
    gen_bs = per_bs * 4  # divisible by num_generations=4 and by global batch size
    cfg = GRPOConfig(
        output_dir="trl-output-grpo",
        per_device_train_batch_size=per_bs,
        learning_rate=args.lr,
        num_generations=4,
        max_completion_length=args.K,
        temperature=1.0,
        use_cpu=(device.type == "cpu"),
        fp16=False,
        bf16=False,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        generation_batch_size=gen_bs,
    )

    # Reward function
    def reward_fn(completions, prompts=None, **_: dict):
        rews: List[float] = []
        # Normalize input forms
        def _to_text(x):
            if isinstance(x, str):
                return x
            if isinstance(x, list) and x and isinstance(x[0], dict) and "content" in x[0]:
                return x[0]["content"]
            if isinstance(x, list) and x and isinstance(x[0], str):
                return x[0]
            return str(x)

        if prompts is None:
            prompts = [""] * len(completions)
        for comp, prm in zip(completions, prompts):
            c_text = _to_text(comp)
            p_text = _to_text(prm)
            dirs = _parse_completion(c_text, args.K, direction_angles)
            # Prepare tensors
            Y = cache[p_text]["Y"]
            s_hat = cache[p_text]["s_hat"]
            if args.reward_mode == "proxyA":
                R = _compute_proxyA_reward(Y, s_hat, H, dirs)
            else:
                R = _compute_deltaIS_reward(Y, s_hat, H, dirs)
            rews.append(float(R))
        return rews

    # Create trainer
    # Move model to device before handing to trainer (accelerate will manage further)
    policy.pretrained_model.to(device)
    trainer = GRPOTrainer(
        model=policy.pretrained_model,
        reward_funcs=reward_fn,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    # Attach no-repeat logits processor for distinct directions
    direction_ids = direction_token_ids(tokenizer)
    processor = NoRepeatDirectionLogitsProcessor(direction_ids)
    # Wrap generate similarly to PPO scaffold
    def _wrap_generate(module):
        original_generate = module.generate

        def generate_with_mask(*args, **kwargs):
            existing = kwargs.get("logits_processor")
            if existing is None:
                kwargs["logits_processor"] = [processor]
            else:
                procs = list(existing)
                if processor not in procs:
                    procs.append(processor)
                kwargs["logits_processor"] = procs
            return original_generate(*args, **kwargs)

        module.generate = generate_with_mask

    _wrap_generate(trainer.model)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
