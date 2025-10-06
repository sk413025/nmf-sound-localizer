#!/usr/bin/env python3
"""Supervised warm-start (SFT) for the policy using a frozen RM as greedy teacher.

Minimal implementation that:
- Builds patch prompts from real data
- Uses the frozen RM to greedily select K unique direction tokens per sample
- Trains the policy (embeddings + optional LoRA on backbone) with teacher forcing
- Saves policy adapters + embeddings for PPO warm-start and reference snapshot

Physics remains inside the RM. No A·X solver is used here.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from transformers import set_seed

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids


def _discover_angles(root: str) -> List[int]:
    base = Path(root)
    angles = sorted({int(p.name.split("_")[1]) for p in base.glob("angle_*") if p.is_dir() and p.name.split("_")[1].isdigit()})
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _prepare_prompts(args, direction_angles: List[int]) -> List[str]:
    ds = DoADataset(args.data_root, direction_angles, fs=args.sample_rate, n_fft=args.n_fft, freq_min=args.freq_min, freq_max=args.freq_max)
    dl = create_dataloader(ds, batch_size=1, shuffle=False)
    tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
    prompts: List[str] = []
    for batch in dl:
        Y_np = batch["Y"].squeeze(0).numpy()
        prompts.append(" ".join(tok(Y_np)))
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts


def main():
    ap = argparse.ArgumentParser(description="SFT policy with RM-greedy teacher (directions-first eval)")
    # Data and RM
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--rm-adapters", type=str, required=True)
    ap.add_argument("--rm-heads", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    # Training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"])
    # Tokenizer/grid
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    # LoRA (optional on policy backbone)
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-target-modules", type=str, default="c_attn,c_proj")
    # Output
    ap.add_argument("--out", type=str, default="sft_policy_rm_greedy")
    args = ap.parse_args()

    if not HAS_PEFT:
        raise RuntimeError("peft is required. Install with: pip install peft")

    set_seed(args.seed)

    # Device
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

    # Angles/tokenizer/prompts
    direction_angles = _discover_angles(args.data_root)
    prompts = _prepare_prompts(args, direction_angles)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Build policy and (optionally) apply LoRA
    policy, _ = build_value_head_model(tokenizer)
    policy.to(device)
    # SFT does not use value head; freeze it explicitly
    for p in policy.v_head.parameters():
        p.requires_grad = False
    if args.use_lora:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        lcfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none")
        policy.pretrained_model = get_peft_model(policy.pretrained_model, lcfg)
        print(f"Applied LoRA to policy: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} targets={target_modules}")

    # Frozen RM for teacher scoring (LoRA + embeddings + v_head)
    rm_model, _ = build_value_head_model(tokenizer)
    rm_model.to(device)
    rm_model.eval()
    rm_model.pretrained_model = PeftModel.from_pretrained(rm_model.pretrained_model, args.rm_adapters)
    heads = torch.load(args.rm_heads, map_location="cpu")
    emb = rm_model.pretrained_model.get_input_embeddings()
    state_dict = heads.get("embeddings")
    if hasattr(emb, "base_layer"):
        emb.base_layer.load_state_dict(state_dict)
    else:
        emb.load_state_dict(state_dict)
    rm_model.v_head.load_state_dict(heads["v_head"])  # reward head
    for p in rm_model.parameters():
        p.requires_grad = False
    allowed = set(direction_token_ids(tokenizer))

    # Tokenize prompts once for efficiency
    enc_prompt = tokenizer(prompts, padding=True, truncation=True, max_length=tokenizer.model_max_length, return_tensors="pt")
    input_ids_prompt = enc_prompt["input_ids"].to(device)
    attn_prompt = enc_prompt["attention_mask"].to(device)

    # Helper to extract patch ids from a prompt row (non-special tokens)
    def get_patch_ids(ids_row: torch.Tensor) -> List[int]:
        toks = tokenizer.convert_ids_to_tokens(ids_row.tolist())
        return [tid for tid, tok in zip(ids_row.tolist(), toks) if tok not in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)]

    # RM scoring for a prefix of directions (directions-first)
    def rm_score_for_prefix(dir_prefix_ids: Sequence[int], patch_ids: Sequence[int]) -> float:
        bos = tokenizer.bos_token_id
        ids = [bos] + list(dir_prefix_ids) + list(patch_ids)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.ones_like(inp, device=device)
        with torch.no_grad():
            out = rm_model.pretrained_model(input_ids=inp, attention_mask=attn, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]
            vals = rm_model.v_head(last_hidden).squeeze(-1)[0]
        start = 1 + len(dir_prefix_ids)
        end = start + len(patch_ids)
        patch_vals = vals[start:end]
        return float(patch_vals.mean().item())

    # Build teacher sequences (RM-greedy)
    teacher_dir_ids: List[List[int]] = []
    for i in range(input_ids_prompt.size(0)):
        patch_ids = get_patch_ids(input_ids_prompt[i])
        chosen: List[int] = []
        for j in range(args.K):
            best_token = None
            best_score = -1e9
            for cand in allowed:
                if cand in chosen:
                    continue
                score = rm_score_for_prefix(chosen + [cand], patch_ids)
                if score > best_score:
                    best_score = score
                    best_token = cand
            if best_token is None:
                raise RuntimeError("RM-greedy teacher failed to select a direction token (empty allowed set)")
            chosen.append(best_token)
        teacher_dir_ids.append(chosen)
    print("Built teacher sequences via RM-greedy for", len(teacher_dir_ids), "samples")

    # Construct SFT sequences: prompt + teacher tokens; labels mask to only teacher positions
    texts = [None] * len(prompts)  # not used directly; keep for symmetry
    # Tokenize each constructed sequence
    seq_input_ids: List[torch.Tensor] = []
    seq_labels: List[torch.Tensor] = []
    for i in range(len(prompts)):
        # Compose ids directly to avoid re-tokenizing strings
        prompt_ids = input_ids_prompt[i]
        # Remove padding on the left
        valid = prompt_ids[attn_prompt[i] > 0]
        prefix_ids = valid.tolist()
        full_ids = prefix_ids + teacher_dir_ids[i]
        labels = [-100] * len(prefix_ids) + teacher_dir_ids[i]
        seq_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        seq_labels.append(torch.tensor(labels, dtype=torch.long))

    # Pad into a batch
    def pad_to_batch(rows: List[torch.Tensor], pad_id: int) -> torch.Tensor:
        m = max(r.numel() for r in rows)
        out = torch.full((len(rows), m), pad_id, dtype=torch.long)
        for i, r in enumerate(rows):
            out[i, -r.numel():] = r  # left pad
        return out

    input_ids = pad_to_batch(seq_input_ids, tokenizer.pad_token_id).to(device)
    labels = pad_to_batch(seq_labels, -100).to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

    # Unfreeze policy embeddings
    emb_layer = policy.pretrained_model.get_input_embeddings()
    if hasattr(emb_layer, "base_layer"):
        emb_layer.base_layer.weight.requires_grad = True
    else:
        emb_layer.weight.requires_grad = True

    # Optimizer: LoRA + embeddings only
    params = []
    if args.use_lora:
        params += [p for n, p in policy.named_parameters() if p.requires_grad and "lora" in n.lower()]
    params += [p for n, p in policy.named_parameters() if p.requires_grad and ("wte" in n.lower() or "wpe" in n.lower())]
    optimizer = torch.optim.Adam(params, lr=args.lr)
    print(f"Optimizer with {sum(p.numel() for p in params):,} trainables; lr={args.lr}")

    # Train
    policy.train()
    for ep in range(max(1, args.epochs)):
        optimizer.zero_grad(set_to_none=True)
        out = policy.pretrained_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        print({"epoch": ep, "loss": float(loss.item())})

    # Save adapters and embeddings for PPO warm start
    out_dir = f"{args.out}_policy_adapters"
    policy.pretrained_model.save_pretrained(out_dir)
    heads_path = f"{args.out}_policy_heads.pt"
    torch.save({
        "embeddings": (emb_layer.base_layer.state_dict() if hasattr(emb_layer, "base_layer") else emb_layer.state_dict()),
        "config": {
            "use_lora": args.use_lora,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
        }
    }, heads_path)
    print("Saved SFT policy:")
    print("  - adapters:", out_dir)
    print("  - embeddings:", heads_path)


if __name__ == "__main__":
    main()

