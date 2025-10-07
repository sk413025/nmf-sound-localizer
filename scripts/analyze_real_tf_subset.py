#!/usr/bin/env python3
"""
Analyze real transfer functions (H) on an angle subset using symlinked data.

Steps:
1) Create a temporary dataset structure with only requested angles by symlinking
   from provided original/box roots (X/Y).
2) Run DataProcessor.estimate_transfer_functions to compute H (F x D) and angles (D,).
3) Print summary metrics and save outputs for inspection.

Usage:
  python scripts/analyze_real_tf_subset.py \
    --original /path/to/white_noise_original_data_no_edge \
    --box /path/to/white_noise_box_data_no_edge \
    --out out/real_tf_subset.pth \
    --angle-start 80 --angle-end 150 --angle-step 5 \
    --n-files 3
"""

import argparse
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import torch

from nmf_localizer.config import NMFConfig
from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True, help='Path to X root (angle_XX subfolders)')
    p.add_argument('--box', required=True, help='Path to Y root (angle_XX subfolders)')
    p.add_argument('--out', default='out/real_tf_subset.pth', help='Output .pth for H and angles')
    p.add_argument('--workdir', default='.tmp/real_tf_subset', help='Working dir for symlinks')
    p.add_argument('--angle-start', type=int, default=80)
    p.add_argument('--angle-end', type=int, default=150)
    p.add_argument('--angle-step', type=int, default=5)
    p.add_argument('--n-files', type=int, default=3)
    p.add_argument('--fs', type=int, default=16000)
    # Normalization/contrast have been removed; no related flags.
    return p.parse_args()


def find_angles(root: Path):
    angles = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith('angle_'):
            try:
                ang = int(d.name.replace('angle_', ''))
                angles.append(ang)
            except ValueError:
                pass
    return sorted(angles)


def prepare_symlink_subset(x_root: Path, y_root: Path, out_root: Path, angles: list[int], n_files: int):
    x_out = out_root / 'original'
    y_out = out_root / 'box'
    if out_root.exists():
        shutil.rmtree(out_root)
    x_out.mkdir(parents=True, exist_ok=True)
    y_out.mkdir(parents=True, exist_ok=True)

    report = {}

    for a in angles:
        dx = x_root / f'angle_{a}'
        dy = y_root / f'angle_{a}'
        if not dx.exists() or not dy.exists():
            continue
        xs = sorted(dx.glob('*.npy'))
        ys = sorted(dy.glob('*.npy'))
        k = min(len(xs), len(ys), n_files)
        if k == 0:
            continue

        sx = x_out / f'angle_{a}'
        sy = y_out / f'angle_{a}'
        sx.mkdir(parents=True, exist_ok=True)
        sy.mkdir(parents=True, exist_ok=True)

        for i in range(k):
            srcx = xs[i]
            srcy = ys[i]
            dstx = sx / srcx.name
            dsty = sy / srcy.name
            try:
                os.symlink(srcx, dstx)
            except FileExistsError:
                pass
            try:
                os.symlink(srcy, dsty)
            except FileExistsError:
                pass
        report[a] = {'n_files': k}

    return x_out, y_out, report


def main():
    args = parse_args()
    x_root = Path(args.original)
    y_root = Path(args.box)
    out_path = Path(args.out)
    workdir = Path(args.workdir)

    # Angle selection
    ax = set(find_angles(x_root))
    ay = set(find_angles(y_root))
    target = list(range(args.angle_start, args.angle_end + 1, args.angle_step))
    common = sorted([a for a in target if a in ax and a in ay])
    if not common:
        print('No common angles in requested range.', file=sys.stderr)
        sys.exit(2)
    print(f'Angles to process: {common}')

    # Symlink subset
    x_sub, y_sub, rep = prepare_symlink_subset(x_root, y_root, workdir, common, args.n_files)
    print('Prepared subset symlinks:')
    for a, info in rep.items():
        print(f'  angle {a:3d}: {info["n_files"]} files')

    # Config
    cfg = NMFConfig(
        sample_rate=args.fs,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=1500.0,
        n_files_per_angle=args.n_files,
        device='cpu',
    )
    dp = DataProcessor(cfg)

    # Estimate H
    print('Estimating transfer functions...')
    H, angles, angle_folders, meta = dp.estimate_transfer_functions(x_sub, y_sub, time_pooling='geometric')

    print(f'H shape: {tuple(H.shape)}, angles: {angles.tolist()}')
    print(f'H stats: min={H.min().item():.4f} max={H.max().item():.4f} mean={H.mean().item():.4f}')

    # Separability metrics
    tfp = TransferFunctionProcessor(cfg)
    sep = tfp.analyze_separability(H)
    print('Separability metrics:')
    for k, v in sep.items():
        print(f'  {k}: {v}')

    # Angle dependence (across frequencies)
    angle_resp = torch.mean(H, dim=0)
    print(f'Angle response std (mean over freq): {torch.std(angle_resp).item():.4f}')

    # Save outputs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'H': H,
        'angles': angles,
        'config': cfg.to_dict(),
        'meta': meta,
        'subset_report': rep,
        'separability': sep,
    }, out_path)
    print(f'Saved results to: {out_path}')


if __name__ == '__main__':
    main()
