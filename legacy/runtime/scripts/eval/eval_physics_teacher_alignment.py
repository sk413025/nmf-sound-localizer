#!/usr/bin/env python3
"""Evaluate physics teacher alignment and H validity on real data (detection only).

Outputs per-sample JSONL and a run-level summary to surface when
"maximize s_d" fails to pick the true DoA and whether the current H makes
the correct angle minimize IS.

Artifacts under results/<out>/:
- teacher_alignment.jsonl: per-sample metrics
- summary.json: run-level aggregates and H similarity stats
- subset_manifest.json: sample list used

This script performs no training and is safe to run as a preflight check.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor
from doa_rl.features.nmf_utils import estimate_s_hat_torch


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


def _derive_content_path(content_root: str, y_path: str) -> Path:
    p = Path(y_path)
    if not p.name.endswith(".npy"):
        raise RuntimeError(f"Expected .npy file for Y; got: {y_path}")
    angle_dir = p.parent.name
    if not angle_dir.startswith("angle_"):
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
        wav, fs=fs, nperseg=n_fft, window="hann"
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


def _is_divergence(Y: torch.Tensor, Yhat: torch.Tensor, eps: float) -> torch.Tensor:
    Yc = torch.clamp(Y, min=eps)
    if Yhat.dim() == 1:
        Yhc = torch.clamp(Yhat.view(-1, 1).expand_as(Yc), min=eps)
    else:
        Yhc = torch.clamp(Yhat, min=eps)
    ratio = Yc / Yhc
    return torch.sum(ratio - torch.log(torch.clamp(ratio, min=eps)) - 1.0)


def _teacher_scores(
    Y: torch.Tensor,
    s_hat: torch.Tensor,
    H: torch.Tensor,
    dir_indices: Sequence[int],
    teacher: str,
    eps: float,
) -> List[float]:
    F, N = Y.shape
    Yc = torch.clamp(Y.float(), min=eps)
    sh = torch.clamp(s_hat.float(), min=eps)
    R = torch.clamp(Yc / sh.view(-1, 1), min=eps)
    Rm = torch.clamp(R.mean(dim=1), min=eps)
    scores: List[float] = []
    for d in dir_indices:
        Hd = torch.clamp(H[:, d], min=eps)
        if teacher == "fit":
            val = -_is_divergence(Yc, Hd * sh, eps=eps).item()
        elif teacher == "euc":
            diff = (Yc - (Hd * sh).view(-1, 1).expand_as(Yc))
            val = -float(torch.sum(diff * diff).item())
        elif teacher == "fit_cn":
            val = -_is_divergence(R, Hd, eps=eps).item()
        elif teacher == "fit_cn_mean":
            val = -_is_divergence(Rm.view(-1, 1), Hd, eps=eps).item()
        else:
            raise ValueError(f"Unknown teacher: {teacher}")
        scores.append(val)
    return scores


def main():
    ap = argparse.ArgumentParser(description="Evaluate physics teacher alignment and H validity (no training)")
    # Data and assets
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--content-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    # Teacher and selection
    ap.add_argument("--teacher", type=str, choices=["fit", "euc", "fit_cn", "fit_cn_mean"], default="fit_cn_mean")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--directions-per-sample", type=int, default=0)
    ap.add_argument("--max-samples", type=int, default=0)
    # STFT/grid
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    # Output and strictness
    ap.add_argument("--out", type=str, default="physics_teacher_alignment")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--strict-teacher-top1", type=float, default=0.0, help="Require teacher Top-1 ≥ threshold else fail")
    ap.add_argument("--strict-recallK", type=float, default=0.0, help="Require teacher recall@K ≥ threshold else fail")
    ap.add_argument("--strict-h-top1", type=float, default=0.0, help="Require H validity top1 (fit_cn_mean) ≥ threshold else fail")
    args = ap.parse_args()

    # Discover angles and build dataset
    direction_angles = _discover_angles(args.data_root)
    ds = DoADataset(
        args.data_root,
        direction_angles,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(ds, batch_size=1, shuffle=False)

    # Load H and select dataset-angle columns
    H_full, anglesT = load_H(args.tf_path)
    H_full = H_full.float().cpu()
    anglesT = anglesT.cpu().numpy().astype(int).tolist()
    col_idx: List[int] = []
    for a in direction_angles:
        if int(a) not in anglesT:
            raise RuntimeError(f"Angle {int(a)}° not found in TF angles {anglesT}")
        col_idx.append(anglesT.index(int(a)))
    if len(set(col_idx)) != len(col_idx):
        raise RuntimeError(f"Duplicate TF column mappings detected: {col_idx}")
    H = H_full[:, col_idx].contiguous()

    # H column similarity
    Hn = torch.clamp(H, min=1e-12)
    Hn = Hn / torch.clamp(torch.norm(Hn, dim=0, keepdim=True), min=1e-12)
    sim = (Hn.t() @ Hn).cpu().numpy()  # (D,D)
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    h_sim_stats = {
        "offdiag_max": float(off.max(initial=0.0)) if off.size else 0.0,
        "offdiag_p95": float(np.percentile(off, 95)) if off.size else 0.0,
        "offdiag_mean": float(off.mean()) if off.size else 0.0,
    }

    # Load W for ŝ
    if args.w_path.endswith(".npz"):
        W_np = np.load(args.w_path)["W"].astype(np.float32)
        W_t = torch.from_numpy(W_np)
    else:
        W_t = load_W(args.w_path).float().cpu()

    results_dir = Path("results") / args.out
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "teacher_alignment.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    # Aggregates
    hits: List[int] = []
    ranks: List[int] = []
    recalls: List[int] = []
    margins: List[float] = []
    h_valid_hits_fit: List[int] = []
    h_valid_hits_fitcn: List[int] = []
    h_valid_hits_fitcnm: List[int] = []

    subset: List[Dict[str, str]] = []

    # Iterate samples
    count = 0
    for batch in dl:
        Y = batch["Y"].squeeze(0).float()
        F, N = Y.shape
        gt_angle = int(batch.get("angle_deg", -1))
        # Resolve source path for pairing to content
        y_src = batch.get("path", None)
        if isinstance(y_src, (list, tuple)):
            y_src = y_src[0] if y_src else None
        if not y_src:
            raise RuntimeError("Dataset did not provide source path for Y; cannot derive content counterpart")
        content_path = _derive_content_path(args.content_root, y_src)
        subset.append({"y_path": str(y_src), "content_path": str(content_path)})

        # Content spectrogram and ŝ
        Y_content = _load_band_spectrogram_from_npy(
            content_path, fs=args.sample_rate, n_fft=args.n_fft, freq_min=args.freq_min, freq_max=args.freq_max
        ).float()
        if Y_content.shape != Y.shape:
            raise RuntimeError(
                f"Y_content shape {tuple(Y_content.shape)} must equal Y {tuple(Y.shape)}; align STFT grid (fs/n_fft/band)."
            )
        if H.shape[0] != F:
            raise RuntimeError(f"H.F ({int(H.shape[0])}) != Y.F ({int(F)})")
        W_F = int(W_t.shape[0]) if W_t.dim() == 2 else -1
        if W_F != F:
            raise RuntimeError(f"W.F ({W_F}) != Y.F ({int(F)})")
        s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=50, l1=0.0)
        if not torch.isfinite(s_hat_t).all() or (s_hat_t < 0).any() or int(s_hat_t.shape[0]) != F:
            raise RuntimeError("ŝ invalid: non-finite/negative/shape mismatch")

        # Candidate directions
        D = H.shape[1]
        if args.directions_per_sample and args.directions_per_sample > 0:
            sel_idx = np.random.choice(D, size=min(args.directions_per_sample, D), replace=False)
            d_indices = [int(i) for i in sel_idx]
        else:
            d_indices = list(range(D))

        # Teacher scores
        scores = _teacher_scores(Y, s_hat_t, H, d_indices, args.teacher, eps=float(args.eps))
        s_arr = np.asarray(scores, dtype=float)
        order = list(range(len(d_indices)))
        order.sort(key=lambda i: s_arr[i], reverse=True)
        best_local = d_indices[order[0]]
        # Ground truth token id corresponds to angle gt_angle at same index in direction_angles
        if gt_angle not in direction_angles:
            raise RuntimeError(f"GT angle {gt_angle} not in dataset angles {direction_angles}")
        gt_idx = direction_angles.index(gt_angle)
        # Map gt_idx to our local candidate index list if subsetted
        try:
            local_pos = d_indices.index(gt_idx)
        except ValueError:
            # GT not in candidate subset; treat as miss for Top-1/recall@K
            top1_hit = 0
            rank_true = len(d_indices)
            margin = float("-inf")
            recall_hit = 0
        else:
            top1_hit = int(best_local == gt_idx)
            rank_true = 1 + order.index(local_pos)
            margin = float(s_arr[local_pos] - max([s_arr[i] for i in range(len(order)) if i != local_pos], default=s_arr[local_pos]))
            recall_hit = int(rank_true <= int(max(1, args.K)))

        # H physics validity (independent of chosen teacher): evaluate Top-1 for multiple teacher definitions
        s_fit = _teacher_scores(Y, s_hat_t, H, d_indices, "fit", eps=float(args.eps))
        s_cna = _teacher_scores(Y, s_hat_t, H, d_indices, "fit_cn", eps=float(args.eps))
        s_cnm = _teacher_scores(Y, s_hat_t, H, d_indices, "fit_cn_mean", eps=float(args.eps))
        def top1_local(arr: List[float]) -> int:
            if gt_angle not in direction_angles:
                return 0
            try:
                local_pos2 = d_indices.index(direction_angles.index(gt_angle))
            except ValueError:
                return 0
            best = int(np.argmax(np.asarray(arr)))
            return int(best == local_pos2)

        h_fit_hit = top1_local(s_fit)
        h_cna_hit = top1_local(s_cna)
        h_cnm_hit = top1_local(s_cnm)

        # Temporal steadiness and teacher stability
        Y_np = Y.numpy()
        temporal_cv = float(np.mean(np.std(Y_np, axis=1)) / max(np.mean(Y_np), 1e-12))
        score_std = float(np.std(s_arr))
        score_cv = float(score_std / max(abs(np.mean(s_arr)), 1e-12))

        row = {
            "path": os.path.basename(str(y_src)) or "",
            "gt_angle": int(gt_angle),
            "F": int(F),
            "N": int(N),
            "teacher": args.teacher,
            "dirs_considered": int(len(d_indices)),
            "teacher_top1_hit": int(top1_hit),
            "teacher_rank_true": int(rank_true),
            "teacher_margin": float(margin),
            "teacher_recall_at_K": int(recall_hit),
            "h_valid_top1_hit_fit": int(h_fit_hit),
            "h_valid_top1_hit_fit_cn": int(h_cna_hit),
            "h_valid_top1_hit_fit_cn_mean": int(h_cnm_hit),
            "temporal_CV": temporal_cv,
            "score_std": score_std,
            "score_cv": score_cv,
        }
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(row) + "\n")

        hits.append(top1_hit)
        ranks.append(rank_true)
        recalls.append(recall_hit)
        margins.append(margin)
        h_valid_hits_fit.append(h_fit_hit)
        h_valid_hits_fitcn.append(h_cna_hit)
        h_valid_hits_fitcnm.append(h_cnm_hit)

        count += 1
        if args.max_samples and count >= args.max_samples:
            break

    # Summary
    def agg(x: List[float]) -> Dict[str, float]:
        arr = np.asarray(x, dtype=float)
        return {
            "mean": float(np.mean(arr)) if arr.size else 0.0,
            "median": float(np.median(arr)) if arr.size else 0.0,
        }

    summary = {
        "samples": int(count),
        "teacher": args.teacher,
        "K": int(args.K),
        "teacher_top1_acc": agg(hits)["mean"],
        "teacher_recall_at_K": agg(recalls)["mean"],
        "teacher_median_rank": agg(ranks)["median"],
        "teacher_margin_mean": agg(margins)["mean"],
        "H_similarity": h_sim_stats,
        "H_validity_top1": {
            "fit": agg(h_valid_hits_fit)["mean"],
            "fit_cn": agg(h_valid_hits_fitcn)["mean"],
            "fit_cn_mean": agg(h_valid_hits_fitcnm)["mean"],
        },
    }
    with open(results_dir / "summary.json", "w") as wf:
        json.dump(summary, wf, indent=2)

    print("Teacher alignment summary:")
    print(json.dumps(summary, indent=2))

    # Write subset manifest
    with open(results_dir / "subset_manifest.json", "w") as wf:
        json.dump({"count": len(subset), "samples": subset}, wf, indent=2)

    # Strictness gates
    if args.validate_only or args.strict_teacher_top1 > 0 or args.strict_recallK > 0 or args.strict_h_top1 > 0:
        fail_msgs: List[str] = []
        if args.strict_teacher_top1 > 0 and summary["teacher_top1_acc"] < args.strict_teacher_top1:
            fail_msgs.append(f"teacher_top1_acc {summary['teacher_top1_acc']:.3f} < {args.strict_teacher_top1}")
        if args.strict_recallK > 0 and summary["teacher_recall_at_K"] < args.strict_recallK:
            fail_msgs.append(f"teacher_recall_at_K {summary['teacher_recall_at_K']:.3f} < {args.strict_recallK}")
        if args.strict_h_top1 > 0 and summary["H_validity_top1"]["fit_cn_mean"] < args.strict_h_top1:
            fail_msgs.append(
                f"H_valid_top1_fit_cn_mean {summary['H_validity_top1']['fit_cn_mean']:.3f} < {args.strict_h_top1}"
            )
        if fail_msgs:
            raise RuntimeError("Validation failed: " + "; ".join(fail_msgs))


if __name__ == "__main__":
    main()
