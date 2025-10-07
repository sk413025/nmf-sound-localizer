#!/usr/bin/env python3
"""Evaluate a trained Reward Model (RM) under directions-first scoring.

Checks two hard requirements on real data:
- Top-1: The ground-truth direction token must have the highest RM score
         among all allowed directions at t=0 (no prefix).
- Top-K (teacher recall): The greedy teacher sequence of length K must
         contain the ground-truth direction token.

If --require-top1/--require-topK are passed, the script exits non-zero
when the requirement is not met.

Inputs:
- Frozen RM composed of LoRA adapters (peft) + saved embeddings/v_head heads
- Real dataset root organized as angle_XXX/clip_YYY.npy

Outputs (under --out):
- summary.json: Top-1 accuracy, recall@K, counts
- per_sample.jsonl: per-sample scores, selections, and diagnostics
- subset_manifest.json: list of evaluated sample paths and angles
- run.log: concise textual summary

Note: Requires PYTHONPATH to include the project root for doa_rl/* imports.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from transformers import set_seed

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids

# Reuse scoring helpers from the SFT script to ensure identical semantics
from scripts.train_sft_policy_with_rm import (
    rm_score_for_prefix,
    get_patch_ids,
    compute_rm_greedy_teacher,
)


def _discover_angles(root: str) -> List[int]:
    base = Path(root)
    angles = sorted({
        int(p.name.split("_")[1])
        for p in base.glob("angle_*")
        if p.is_dir() and p.name.split("_")[1].isdigit()
    })
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Evaluate RM Top-1 and Top-K (directions-first)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--rm-adapters", type=str, required=True, help="Path to RM LoRA adapters dir (save_pretrained)")
    ap.add_argument("--rm-heads", type=str, required=True, help="Path to RM heads .pt (embeddings + v_head)")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"]) 
    # STFT / patch grid
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap for a quick subset evaluation")
    # Gating flags
    ap.add_argument("--require-top1", action="store_true", help="Fail if Top-1 accuracy < 1.0")
    ap.add_argument("--require-topK", action="store_true", help="Fail if recall@K < 1.0")
    # Output directory
    ap.add_argument("--out", type=str, default="results/eval_rm")
    args = ap.parse_args()

    if not HAS_PEFT:
        raise RuntimeError("peft is required. Install with: pip install peft")

    set_seed(args.seed)

    # Device selection
    dev = args.device
    if dev == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    device = torch.device(dev)

    out_dir = Path(args.out)
    _ensure_dir(out_dir)

    print(f"Device: {device}")

    # Angles, tokenizer
    direction_angles = _discover_angles(args.data_root)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Build RM and load adapters + heads
    rm_model, _ = build_value_head_model(tokenizer)
    rm_model.to(device)
    rm_model.eval()
    rm_model.pretrained_model = PeftModel.from_pretrained(rm_model.pretrained_model, args.rm_adapters)
    heads = torch.load(args.rm_heads, map_location="cpu")
    emb = rm_model.pretrained_model.get_input_embeddings()
    emb_state = heads.get("embeddings")
    if hasattr(emb, "base_layer"):
        emb.base_layer.load_state_dict(emb_state)
    else:
        emb.load_state_dict(emb_state)
    rm_model.v_head.load_state_dict(heads["v_head"])  # reward head
    for p in rm_model.parameters():
        p.requires_grad = False

    # Prepare real prompts and ground-truth angles
    ds = DoADataset(
        args.data_root,
        direction_angles,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(ds, batch_size=1, shuffle=False)
    patch_tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)

    prompts: List[str] = []
    gt_angles: List[int] = []
    src_paths: List[str] = []
    for batch in dl:
        Y_np = batch["Y"].squeeze(0).numpy()
        prompt = " ".join(patch_tok(Y_np))
        prompts.append(prompt)
        gt_angles.append(int(batch["angle_deg"]))
        p = batch.get("path", "")
        if isinstance(p, (list, tuple)):
            p = p[0] if p else ""
        src_paths.append(str(p))
        if args.max_samples and len(prompts) >= args.max_samples:
            break

    if not prompts:
        raise RuntimeError("No samples prepared from dataset; check --data-root and filters")

    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    input_ids_prompt = enc["input_ids"].to(device)

    allowed = set(direction_token_ids(tokenizer))
    if len(allowed) == 0:
        raise RuntimeError("Tokenizer has zero direction tokens; rebuild with build_patch_tokenizer()")
    if args.K < 1 or args.K > len(allowed):
        raise RuntimeError(f"Invalid K={args.K}; valid range is [1, {len(allowed)}]")

    # Per-sample evaluation
    per_sample_path = out_dir / "per_sample.jsonl"
    subset_manifest_path = out_dir / "subset_manifest.json"
    summary_path = out_dir / "summary.json"
    runlog_path = out_dir / "run.log"

    top1_hits = 0
    teacher_hits = 0

    # Compute greedy teacher once
    teacher_dir_ids: List[List[int]] = compute_rm_greedy_teacher(
        rm_model=rm_model,
        tokenizer=tokenizer,
        input_ids_prompt=input_ids_prompt,
        K=args.K,
        device=device,
    )

    with per_sample_path.open("w", encoding="utf-8") as fw:
        for i in range(input_ids_prompt.size(0)):
            row = input_ids_prompt[i]
            patch_ids = get_patch_ids(tokenizer, row)
            if len(patch_ids) == 0:
                raise RuntimeError("Prompt row has no patch tokens; cannot score patches")

            # Build candidate scores for Top-1 check (t=0, single direction)
            cand_scores: Dict[int, float] = {}
            for cand in allowed:
                score = rm_score_for_prefix(rm_model, tokenizer, [cand], patch_ids, device)
                cand_scores[cand] = float(score)

            # Determine gt token id
            gt_angle = int(gt_angles[i])
            gt_token = f"<D_{gt_angle:03d}>"
            gt_id = tokenizer.convert_tokens_to_ids(gt_token)
            if gt_id not in allowed:
                raise RuntimeError(
                    f"Ground-truth token '{gt_token}' (id={gt_id}) not in allowed direction tokens; "
                    f"ensure exact angle-token mapping"
                )

            # Top-1 argmax
            best_id = max(cand_scores, key=cand_scores.get)
            best_score = cand_scores[best_id]
            gt_score = cand_scores[gt_id]
            # Rank of GT among all candidates (0-based)
            sorted_ids = sorted(cand_scores.keys(), key=lambda x: cand_scores[x], reverse=True)
            gt_rank = int(sorted_ids.index(gt_id))
            top1_ok = (best_id == gt_id)
            if top1_ok:
                top1_hits += 1

            # Teacher recall@K
            teacher_ids_i = teacher_dir_ids[i]
            teacher_ok = (gt_id in teacher_ids_i)
            if teacher_ok:
                teacher_hits += 1

            fw.write(json.dumps({
                "idx": i,
                "path": src_paths[i],
                "angle_deg": gt_angle,
                "gt_token": gt_token,
                "gt_id": gt_id,
                "best_id": best_id,
                "best_token": tokenizer.convert_ids_to_tokens(best_id),
                "gt_rank": gt_rank,
                "best_score": best_score,
                "gt_score": gt_score,
                "margin_best_minus_gt": float(best_score - gt_score),
                "teacher_ids": teacher_ids_i,
                "teacher_tokens": tokenizer.convert_ids_to_tokens(teacher_ids_i),
                "top1_ok": top1_ok,
                "teacher_ok": teacher_ok,
            }) + "\n")

    total = input_ids_prompt.size(0)
    top1_acc = top1_hits / total
    recall_k = teacher_hits / total

    summary = {
        "device": str(device),
        "samples": total,
        "allowed_directions": len(allowed),
        "K": int(args.K),
        "top1_acc": float(top1_acc),
        "recall_at_K": float(recall_k),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    subset_manifest = {
        "samples": [
            {"idx": i, "path": src_paths[i], "angle_deg": int(gt_angles[i])}
            for i in range(total)
        ]
    }
    subset_manifest_path.write_text(json.dumps(subset_manifest, indent=2), encoding="utf-8")

    with runlog_path.open("w", encoding="utf-8") as f:
        f.write(
            f"Device: {device}\n"
            f"Samples: {total}\n"
            f"Allowed directions: {len(allowed)}\n"
            f"K: {args.K}\n"
            f"Top-1 accuracy: {top1_acc:.6f}\n"
            f"Recall@K: {recall_k:.6f}\n"
        )

    print({"top1_acc": top1_acc, "recall_at_K": recall_k})

    # Gating
    if args.require_top1 and top1_acc < 1.0:
        raise SystemExit(2)
    if args.require_topK and recall_k < 1.0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()

