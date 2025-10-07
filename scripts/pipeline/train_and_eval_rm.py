#!/usr/bin/env python3
"""Pipeline: Train Reward Model (LoRA) then evaluate directions-first Top-1 and Top-K.

This wrapper:
1) Runs scripts/train_reward_model_lora.py to produce RM adapters + heads
2) Runs scripts/eval/eval_rm_directions_first.py with gating flags

All logs are tee'd to results/<run_name>/*.log

Usage example:
  python scripts/pipeline/train_and_eval_rm.py \
    --data-root /path/to/box_dataset_root \
    --tf-path /path/to/H.pth \
    --w-path /path/to/W.pth \
    --K 3 --rm-epochs 20 --device mps \
    --run-name rm_$(date +%Y%m%d_%H%M) \
    --require-top1 --require-topK
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_and_tee(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        print("Running:", " ".join(cmd))
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            lf.write(line)
        proc.wait()
        return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Train RM (LoRA) then evaluate Top-1/Top-K with gating")
    # Shared/data
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--s-root", type=str, required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"]) 
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=0)
    # STFT / patch grid (kept consistent across train/eval)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    # Train RM hyperparameters (subset)
    ap.add_argument("--rm-epochs", type=int, default=20)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--lr-embed", type=float, default=1e-4)
    ap.add_argument("--lr-vhead", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.1)
    ap.add_argument("--lora-target-modules", type=str, default="c_attn,c_proj")
    # Gating flags
    ap.add_argument("--require-top1", action="store_true")
    ap.add_argument("--require-topK", action="store_true")
    # Output naming
    ap.add_argument("--run-name", type=str, default=None, help="Directory name under results/")
    args = ap.parse_args()

    run_name = args.run_name or "rm_run"
    results_root = Path("results") / run_name
    _ensure_dir(results_root)

    # Train RM (outputs under results/<run_name>/rm_*)
    rm_out_prefix = str(results_root / "rm_ckpt_lora")
    train_cmd = [
        sys.executable, "scripts/train_reward_model_lora.py",
        "--data-root", args.data_root,
        "--tf-path", args.tf_path,
        "--w-path", args.w_path,
        "--s-root", args.s_root,
        "--K", str(args.K),
        "--teacher", os.environ.get("RM_TEACHER", "fit"),
        "--bt-beta", os.environ.get("RM_BT_BETA", "1.0"),
        "--directions-per-sample", os.environ.get("RM_DIRS_PER_SAMPLE", "0"),
        "--pairs-per-sample", os.environ.get("RM_PAIRS_PER_SAMPLE", "64"),
        "--rm-epochs", str(args.rm_epochs),
        "--lr-lora", str(args.lr_lora),
        "--lr-embed", str(args.lr_embed),
        "--lr-vhead", str(args.lr_vhead),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--lora-target-modules", args.lora_target_modules,
        "--sample-rate", str(args.sample_rate),
        "--n-fft", str(args.n_fft),
        "--freq-min", str(args.freq_min),
        "--freq-max", str(args.freq_max),
        "--patch-fp", str(args.patch_fp),
        "--patch-np", str(args.patch_np),
        "--max-samples", str(args.max_samples),
        "--device", args.device,
        "--out", rm_out_prefix,
    ]
    rc = _run_and_tee(train_cmd, results_root / "train.log")
    if rc != 0:
        raise SystemExit(rc)

    # Evaluate RM (gating)
    rm_adapters = f"{rm_out_prefix}_adapters"
    rm_heads = f"{rm_out_prefix}_heads.pt"
    eval_out_dir = str(results_root / "eval")
    eval_cmd = [
        sys.executable, "scripts/eval/eval_rm_directions_first.py",
        "--data-root", args.data_root,
        "--rm-adapters", rm_adapters,
        "--rm-heads", rm_heads,
        "--K", str(args.K),
        "--sample-rate", str(args.sample_rate),
        "--n-fft", str(args.n_fft),
        "--freq-min", str(args.freq_min),
        "--freq-max", str(args.freq_max),
        "--patch-fp", str(args.patch_fp),
        "--patch-np", str(args.patch_np),
        "--max-samples", str(args.max_samples),
        "--device", args.device,
        "--out", eval_out_dir,
    ]
    if args.require_top1:
        eval_cmd.append("--require-top1")
    if args.require_topK:
        eval_cmd.append("--require-topK")

    rc = _run_and_tee(eval_cmd, results_root / "eval.log")
    if rc != 0:
        raise SystemExit(rc)

    print("\nAll checks passed. Results in:", results_root)


if __name__ == "__main__":
    main()
