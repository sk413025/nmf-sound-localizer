#!/usr/bin/env python3
"""Run shared-support IS-OMP (sklearn MU, IS loss) as an oracle evaluator.

Design
- Prefilter M=16 by default using teacher s_T(d) = −DIS_IS(Y || (H_d ⊙ ŝ)·1ᵀ)
- Strict guardrails: exact angle mapping; grid equality F(Y)=F(H)=F(W); ŝ valid
- Outputs per-sample numeric diagnostics under results/<out>/numeric_diagnostics.jsonl

This script adds new functionality without changing existing training flows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from doa_rl.assets import load_H, load_W
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.omp.is_omp import is_omp_select, teacher_scores_is
from doa_rl.features.nmf_utils import estimate_s_hat_torch
from nmf_localizer.utils.audio_utils import AudioProcessor


def _derive_content_path(content_root: str, y_path: str) -> Path:
    p = Path(y_path)
    if not p.name.endswith('.npy'):
        raise RuntimeError(f"Expected .npy file for Y; got: {y_path}")
    angle_dir = p.parent.name
    if not angle_dir.startswith('angle_'):
        raise RuntimeError(f"Y path does not reside in angle_* directory: {y_path}")
    s_path = Path(content_root) / angle_dir / p.name
    if not s_path.exists():
        raise RuntimeError(f"Content file not found: {s_path} (derived from {y_path})")
    return s_path


def _load_band_spectrogram_from_npy(
    npy_path: Path,
    fs: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    wav = np.load(str(npy_path))
    if wav.ndim != 1:
        raise RuntimeError(f"Expected mono waveform in {npy_path}")
    freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=n_fft, window='hann'
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


def main():
    ap = argparse.ArgumentParser(description="IS-OMP oracle (sklearn MU, IS loss)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--content-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--prefilter-M", type=int, default=16)
    ap.add_argument("--mu-warm", type=int, default=10)
    ap.add_argument("--mu-accept", type=int, default=40)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--out", type=str, default="is_omp_oracle")
    args = ap.parse_args()

    # Load assets
    H_full, anglesT = load_H(args.tf_path)
    H_full = H_full.float().cpu()
    anglesT = anglesT.cpu().numpy().astype(int).tolist()
    W_t = load_W(args.w_path).float().cpu()

    # Prepare dataset
    ds = DoADataset(
        args.data_root,
        anglesT,  # will filter below to match exact mapping
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(ds, batch_size=1, shuffle=False)

    # Results dir
    results_dir = os.path.join("results", f"{args.out}")
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = os.path.join(results_dir, "numeric_diagnostics.jsonl")
    if os.path.exists(jsonl_path):
        os.remove(jsonl_path)

    total = 0
    top1_hits = 0
    for batch in dl:
        Y_t: torch.Tensor = batch["Y"].squeeze(0)
        angle_deg = int(batch["angle_deg"])  # GT
        y_src = batch.get("path", None)
        if isinstance(y_src, (list, tuple)):
            y_src = y_src[0] if y_src else None
        if not y_src:
            raise RuntimeError("Dataset did not provide source path for Y; cannot derive content counterpart")

        # Map dataset angle set to H columns (exact match, no fallback)
        if angle_deg not in anglesT:
            raise RuntimeError(
                f"Angle {angle_deg}° not found in TF angles {anglesT}. Ensure dataset↔TF match."
            )
        # For this sample we consider all TF directions; we’ll prefilter later
        H = H_full.clone()

        # ŝ from Original/content counterpart
        content_path = _derive_content_path(args.content_root, y_src)
        Y_content = _load_band_spectrogram_from_npy(
            content_path,
            fs=args.sample_rate,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        ).float()

        # Grid guardrails
        F, N = int(Y_t.shape[0]), int(Y_t.shape[1])
        if H.shape[0] != F or W_t.shape[0] != F or Y_content.shape != Y_t.shape:
            raise RuntimeError(
                f"Grid mismatch: Y {tuple(Y_t.shape)}, H.F={int(H.shape[0])}, W.F={int(W_t.shape[0])}, Y_content {tuple(Y_content.shape)}"
            )

        s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=50, l1=0.0)
        if s_hat_t.shape[0] != F or (s_hat_t < 0).any() or not torch.isfinite(s_hat_t).all():
            raise RuntimeError("Invalid ŝ: shape or values")

        # Run IS-OMP
        Y_np = Y_t.numpy().astype(np.float64)
        H_np = H.numpy().astype(np.float64)
        W_np = W_t.numpy().astype(np.float64)
        s_hat_np = s_hat_t.numpy().astype(np.float64)

        S, X_S, logs = is_omp_select(
            Y_np, H_np, W_np, K=int(args.K), s_hat=s_hat_np,
            prefilter_M=int(args.prefilter_M), mu_iter_warm=int(args.mu_warm), mu_iter_accept=int(args.mu_accept), eps=float(args.eps)
        )

        # Teacher top-1 for evaluation
        scores = teacher_scores_is(Y_np, s_hat_np, H_np, dir_indices=None, eps=float(args.eps))
        best_teacher_idx = int(np.argmax(scores))
        # Map GT angle to TF column id
        gt_col = int(anglesT.index(angle_deg))
        top1_hits += int(best_teacher_idx == gt_col)
        total += 1

        log_row: Dict[str, object] = {
            "path": os.path.basename(y_src) or "",
            "angle_deg": int(angle_deg),
            "F": int(F),
            "N": int(N),
            "K": int(args.K),
            "prefilter_M": int(args.prefilter_M),
            "is_prev": float(logs.get("is_prev", 0.0)),
            "is_final": float(logs.get("is_final", 0.0)),
            "steps": logs.get("steps", []),
        }
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

        if args.max_samples and total >= int(args.max_samples):
            break

    summary = {
        "samples": int(total),
        "teacher_top1_acc": float(top1_hits / max(1, total)),
        "prefilter_M": int(args.prefilter_M),
        "K": int(args.K),
    }
    with open(Path(results_dir) / "summary.json", "w") as sf:
        json.dump(summary, sf, indent=2)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

