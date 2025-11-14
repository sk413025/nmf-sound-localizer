#!/usr/bin/env python3
"""
Resample speech260 waveforms from 48 kHz to 16 kHz.

This is a small utility to convert angle_*/clip_*.npy datasets (speech260)
to 16 kHz so that STFT grids align with H and DoADataset (fs=16000, n_fft=2048,
freq band 300–3000 Hz).

Usage (subset smoke):

  python scripts/conversion/resample_to_16k.py \
    --in_dir /Users/sbplab/LDV-data-processed/speech260_box_data_no_edge_sync_vad_normalized \
    --out_dir /Users/sbplab/LDV-data-processed/speech260_box_16k_smoke \
    --src_sr 48000 --dst_sr 16000 \
    --max_files_per_angle 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import resample_poly


def resample_waveform(
    x: np.ndarray,
    src_sr: int,
    dst_sr: int,
) -> np.ndarray:
    """Resample 1D waveform from src_sr to dst_sr using polyphase filter."""
    if src_sr == dst_sr:
        return x.copy()
    # Reduce ratio to simplest form
    from math import gcd

    g = gcd(src_sr, dst_sr)
    up = dst_sr // g
    down = src_sr // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Resample angle_*/clip_*.npy dataset to 16 kHz")
    ap.add_argument("--in_dir", type=str, required=True, help="Input dataset root with angle_*/ and .npy files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dataset root")
    ap.add_argument("--src_sr", type=int, default=48000, help="Source sampling rate")
    ap.add_argument("--dst_sr", type=int, default=16000, help="Target sampling rate")
    ap.add_argument(
        "--max_files_per_angle",
        type=int,
        default=0,
        help="Limit number of files per angle for smoke test (0 = all)",
    )
    args = ap.parse_args()

    in_root = Path(args.in_dir).expanduser()
    out_root = Path(args.out_dir).expanduser()

    if not in_root.exists():
        raise FileNotFoundError(f"Input directory not found: {in_root}")

    angle_dirs = [d for d in in_root.iterdir() if d.is_dir() and d.name.startswith("angle_")]
    if not angle_dirs:
        raise ValueError(f"No angle_*/ directories found under {in_root}")

    total = 0
    for angle_dir in sorted(angle_dirs, key=lambda p: int(p.name.split("_")[1])):
        npy_files = sorted(angle_dir.glob("*.npy"))
        if args.max_files_per_angle > 0:
            npy_files = npy_files[: args.max_files_per_angle]
        for f in npy_files:
            rel = f.relative_to(in_root)
            x = np.load(str(f))
            if x.ndim > 1:
                x = x[0]
            y = resample_waveform(x, args.src_sr, args.dst_sr)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), y)
            total += 1

    print(
        f"Resampled {total} files from {in_root} (src_sr={args.src_sr}) "
        f"to {out_root} (dst_sr={args.dst_sr})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

