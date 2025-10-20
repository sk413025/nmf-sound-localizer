#!/usr/bin/env python3
"""Verify ΔIS(d|∅) teacher ranks GT angle highest (manifest‑only, multi‑angle required).

This script now only accepts a manifest of absolute paths (eval subset manifest) and
verifies, per sample, that GT angle (from manifest angle_deg or angle_XXX folder) has
the highest ΔIS(d|∅). It prints progress, per-sample lines, and summary stats.

Guardrails:
- Manifest is required and must contain absolute paths to Box npy files.
- At least 2 distinct GT angles must be present; single‑angle verification is rejected.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch

from doa_rl.omp.is_omp import compute_deltais_step0
from doa_rl.assets import load_H
from doa_rl.features.nmf_utils import estimate_s_hat_torch
from nmf_localizer.utils.audio_utils import AudioProcessor


def load_band_spectrogram_from_npy(
    npy_path: Path,
    fs: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    """從 .npy 檔案載入頻段限定的 STFT spectrogram"""
    wav = np.load(str(npy_path))
    if wav.ndim != 1:
        raise RuntimeError(f"Expected mono waveform in {npy_path}")
    freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=n_fft, window="hann"
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


def derive_content_path(content_root: str, y_path: str) -> Path:
    """從觀測路徑推導對應的 content 路徑"""
    p = Path(y_path)
    angle_dir = p.parent.name
    if not angle_dir.startswith("angle_"):
        raise RuntimeError(f"Y path does not reside in angle_* directory: {y_path}")
    s_path = Path(content_root) / angle_dir / p.name
    if not s_path.exists():
        raise RuntimeError(f"Content file not found: {s_path}")
    return s_path


def parse_gt_angle_from_path(path: str) -> int:
    """從路徑解析 ground-truth 角度"""
    p = Path(path)
    angle_dir = p.parent.name
    if not angle_dir.startswith("angle_"):
        raise RuntimeError(f"Cannot parse angle from path: {path}")
    try:
        angle = int(angle_dir.split('_')[1])
        return angle
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Invalid angle directory format: {angle_dir}") from e


import sys
import time


def _require_file(p: str) -> str:
    if not p or not os.path.isabs(p) or not os.path.isfile(p):
        raise FileNotFoundError(f"Invalid or missing file: {p}")
    return p


def main():
    ap = argparse.ArgumentParser(description="Verify ΔIS(d|∅) teacher ranks GT highest (manifest‑only)")
    ap.add_argument("--manifest", type=str, required=True,
                    help="Path to eval_subset_manifest.json (absolute paths + angle_deg, if available)")
    ap.add_argument("--content-root", type=str, required=True,
                    help="Content root (Original)")
    ap.add_argument("--tf-path", type=str, required=True,
                    help="Transfer function H path")
    ap.add_argument("--w-path", type=str, required=True,
                    help="USM basis W path")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Max samples to check (0=all)")
    ap.add_argument("--s-hat-iter", type=int, default=30, help="Iterations for ŝ estimation (default 30 to speed up)")
    ap.add_argument("--mu-iter", type=int, default=5, help="MU iterations for ΔIS per-direction (default 5)")
    ap.add_argument("--progress-every", type=int, default=5, help="Print progress/ETA every N samples")
    ap.add_argument("--prefilter-M", type=int, default=0, help="(Reserved) Top-M prefilter; 0=off")
    args = ap.parse_args()

    # 載入 H 和 W
    H_full, anglesT = load_H(args.tf_path)
    H = H_full.float().cpu().numpy().astype(np.float64)
    anglesT_list = anglesT.cpu().numpy().astype(int).tolist()
    
    if args.w_path.endswith('.npz'):
        W = np.load(args.w_path)["W"].astype(np.float64)
        W_t = torch.from_numpy(W.astype(np.float32))
    else:
        from doa_rl.assets import load_W  # lazy import to avoid cycles
        W_full = load_W(args.w_path)
        W = W_full.float().cpu().numpy().astype(np.float64)
        W_t = W_full.float().cpu()

    print(f"H shape: {H.shape}, angles: {anglesT_list}")
    print(f"W shape: {W.shape}")
    print(f"Settings: s_hat_iter={args.s_hat_iter}, mu_iter={args.mu_iter}, eps={args.eps}")
    sys.stdout.flush()

    # 讀取 manifest 作為唯一來源
    mp = Path(args.manifest)
    if not mp.exists():
        raise RuntimeError(f"Manifest not found: {mp}")
    m = json.loads(mp.read_text())
    rows = m.get('files', [])
    manifest_paths: List[Dict[str, object]] = []
    for row in rows:
        pa = row.get('path_abs') or row.get('path')
        if isinstance(pa, str) and pa.endswith('.npy'):
            manifest_paths.append({
                'path_abs': _require_file(pa),
                'angle_deg': row.get('angle_deg')
            })
    total_lines = len(manifest_paths)
    if args.max_samples and args.max_samples > 0:
        total = min(total_lines, args.max_samples)
    else:
        total = total_lines
    print(f"Diagnostics samples: {total_lines} (processing {total})")
    sys.stdout.flush()

    results: List[Dict] = []
    count = 0
    start_time = time.time()

    # 遍歷 manifest 條目
    for row in manifest_paths[:total]:
        y_path = str(row['path_abs'])
        clip_name = os.path.basename(y_path)
        
        # Determine GT angle: prefer angle in manifest/diag if present; else parse from path
        gt_angle = None
        ang = row.get('angle_deg')
        if ang is not None:
            gt_angle = int(ang)
        if gt_angle is None:
            try:
                gt_angle = parse_gt_angle_from_path(y_path)
            except Exception:
                print(f"⚠️  Warning: Cannot parse ground truth angle for {y_path}")
                continue
        if gt_angle not in anglesT_list:
            print(f"⚠️  Warning: GT angle {gt_angle}° not in TF angles")
            continue
        gt_idx = anglesT_list.index(gt_angle)
            
        # 載入 Y
        Y = load_band_spectrogram_from_npy(
            Path(y_path),
            fs=args.sample_rate,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        ).float()
            
        # 載入對應的 content 並估計 ŝ
        content_path = derive_content_path(args.content_root, y_path)
        Y_content = load_band_spectrogram_from_npy(
            content_path,
            fs=args.sample_rate,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        ).float()
            
        if Y_content.shape != Y.shape:
            print(f"⚠️  Warning: Shape mismatch for {clip_name}")
            continue
        s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=int(args.s_hat_iter), l1=0.0)

        # 計算所有方向的 ΔIS
        Y_np = Y.numpy().astype(np.float64)
        s_hat_np = s_hat_t.numpy().astype(np.float64)

        ord_all, deltas_all = compute_deltais_step0(
            Y=Y_np,
            H=H,
            W=W,
            s_hat=s_hat_np,
            prefilter_M=None,  # 不使用 prefilter，計算所有方向
            mu_iter=int(args.mu_iter),
            baseline_k=2,
            eps=float(args.eps)
        )

        # 找到 GT 角度在排序中的位置
        gt_global_idx = anglesT_list.index(gt_angle)
        try:
            gt_position_in_ord = list(ord_all).index(gt_global_idx)
            gt_rank = gt_position_in_ord + 1  # 1-based rank
        except ValueError:
            print(f"⚠️  Warning: GT angle {gt_angle}° not in ord_all for {clip_name}")
            continue

        # GT 的 ΔIS 分數
        gt_deltais = deltas_all[gt_position_in_ord]

        # 最高分的方向
        best_idx = ord_all[0]
        best_angle = anglesT_list[best_idx]
        best_deltais = deltas_all[0]

        # 分數差異
        margin = gt_deltais - best_deltais  # 負數表示 GT 不是最好的

        # Top-1 命中
        top1_hit = int(best_idx == gt_global_idx)

        result = {
            "clip": clip_name,
            "gt_angle": gt_angle,
            "gt_rank": gt_rank,
            "gt_deltais": float(gt_deltais),
            "best_angle": best_angle,
            "best_deltais": float(best_deltais),
            "margin": float(margin),
            "top1_hit": top1_hit,
            "dirs_total": len(ord_all),
        }
        results.append(result)

        # 打印個別結果
        status = "✅" if top1_hit else "❌"
        print(f"{status} {clip_name:15s} | GT={gt_angle:3d}° (rank={gt_rank:2d}, ΔIS={gt_deltais:8.1f}) | "
              f"Best={best_angle:3d}° (ΔIS={best_deltais:8.1f}) | Margin={margin:8.1f}", flush=True)

        count += 1
        if args.progress_every and count % int(args.progress_every) == 0:
            elapsed = time.time() - start_time
            rate = elapsed / max(1, count)
            remaining = (total - count) * rate if total > 0 else 0.0
            print(f"Progress: {count}/{total} ({count/total*100:.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s", flush=True)
        if args.max_samples > 0 and count >= args.max_samples:
            break
    
    # 統計結果
    print("\n" + "="*80)
    print("📊 統計結果")
    print("="*80)
    
    if not results:
        print("❌ 沒有有效的樣本結果")
        return

    # 至少要有兩種不同的 GT 角度
    unique_angles = sorted({r['gt_angle'] for r in results})
    if len(unique_angles) < 2:
        print("❌ 驗證失敗：樣本只有單一角度，無法代表性驗證教師排序。請提供包含多個角度的 eval manifest。")
        print(f"角度集合: {unique_angles}")
        return
    
    top1_acc = np.mean([r["top1_hit"] for r in results])
    ranks = np.array([r["gt_rank"] for r in results])
    margins = np.array([r["margin"] for r in results])
    
    print(f"總樣本數: {len(results)}")
    print(f"Top-1 準確率: {top1_acc*100:.2f}% ({int(top1_acc*len(results))}/{len(results)})")
    print(f"\nGT 排名分布:")
    print(f"  - Median: {np.median(ranks):.1f}")
    print(f"  - Mean: {np.mean(ranks):.2f}")
    print(f"  - Min: {ranks.min()}")
    print(f"  - Max: {ranks.max()}")
    print(f"  - P25: {np.percentile(ranks, 25):.1f}")
    print(f"  - P75: {np.percentile(ranks, 75):.1f}")
    
    print(f"\n分數差異 (GT ΔIS - Best ΔIS):")
    print(f"  - Mean: {margins.mean():.2f}")
    print(f"  - Median: {np.median(margins):.2f}")
    print(f"  - Min: {margins.min():.2f}")
    print(f"  - Max: {margins.max():.2f}")
    
    # Recall@K 統計
    for K in [1, 3, 5]:
        recall_k = np.mean(ranks <= K)
        print(f"\nRecall@{K}: {recall_k*100:.2f}% ({int(recall_k*len(results))}/{len(results)})")
    
    # 分析失敗案例
    failures = [r for r in results if r["top1_hit"] == 0]
    if failures:
        print(f"\n❌ 失敗案例分析 ({len(failures)} 個):")
        for r in failures[:5]:  # 顯示前 5 個
            print(f"  - {r['clip']:15s}: GT={r['gt_angle']:3d}° (rank={r['gt_rank']:2d}) → "
                  f"Predicted={r['best_angle']:3d}° | Margin={r['margin']:.1f}")
        if len(failures) > 5:
            print(f"  ... (還有 {len(failures)-5} 個失敗案例)")
    
    # 最終結論
    print("\n" + "="*80)
    if top1_acc > 0.9:
        print("✅ 結論: ΔIS teacher 非常可靠，>90% 的樣本 GT 角度獲得最高分")
    elif top1_acc > 0.7:
        print("⚠️  結論: ΔIS teacher 基本可靠，但有 20-30% 的樣本 GT 不是最高分")
    elif top1_acc > 0.5:
        print("⚠️  結論: ΔIS teacher 可疑，接近一半的樣本 GT 不是最高分")
    else:
        print("❌ 結論: ΔIS teacher 有嚴重問題，多數樣本 GT 不是最高分")
    print("="*80)


if __name__ == "__main__":
    main()
