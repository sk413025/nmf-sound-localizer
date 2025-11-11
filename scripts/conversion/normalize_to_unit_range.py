#!/usr/bin/env python3
"""
Normalize a dataset directory with angle_*/ subfolders to [0,1] per file.

This is a lightweight wrapper around the logic used in scripts/normalize_datasets.py,
parameterized to arbitrary input/output roots so we can normalize speech datasets
produced by Stage 1 (Sync-VAD).

Usage:
  python scripts/conversion/normalize_to_unit_range.py \
    --in_dir /path/to/speech_box_data_no_edge_sync_vad \
    --out_dir /path/to/speech_box_data_no_edge_sync_vad_normalized
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def normalize_single_file(input_file: Path, output_file: Path) -> None:
    data = np.load(str(input_file))
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    rng = dmax - dmin
    if rng == 0.0:
        out = np.zeros_like(data)
    else:
        out = (data - dmin) / rng
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_file), out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize dataset directory to [0,1] per file")
    ap.add_argument("--in_dir", type=str, required=True, help="Input dataset root with angle_*/ and .npy files")
    ap.add_argument("--out_dir", type=str, required=True, help="Output dataset root")
    args = ap.parse_args()

    in_root = Path(args.in_dir).expanduser()
    out_root = Path(args.out_dir).expanduser()
    if not in_root.exists():
        raise FileNotFoundError(f"Input directory not found: {in_root}")

    angle_dirs = [d for d in in_root.iterdir() if d.is_dir() and d.name.startswith("angle_")]
    if not angle_dirs:
        raise ValueError(f"No angle_*/ directories found under {in_root}")

    count = 0
    for angle_dir in sorted(angle_dirs, key=lambda p: int(p.name.split("_")[1])):
        npy_files = list(angle_dir.glob("*.npy"))
        for f in npy_files:
            rel = f.relative_to(in_root)
            normalize_single_file(f, out_root / rel)
            count += 1

    print(f"Normalized {count} files from {in_root} -> {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

