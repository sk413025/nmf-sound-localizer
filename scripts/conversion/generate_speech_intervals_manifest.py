#!/usr/bin/env python3
"""
Generate a speech interval manifest from LDV standard_file.wav.

This script reproduces the notebook's interval splitting logic using:
- librosa.effects.split with top_db=70
- padding=0 samples (to match white-noise extraction setup)
- min_duration=0.8 seconds per interval

It then filters intervals to segment1 only using LDV-data_test_log.txt time_len
and selects K representative indices deterministically (evenly spaced across
the segment1 interval list) for reproducibility.

Outputs a JSON manifest with:
- indices: list of selected interval indices (relative to the full intervals list)
- segment: "segment1"
- stats: counts and per-interval metadata (start/end samples, duration seconds)

Usage:
  python scripts/conversion/generate_speech_intervals_manifest.py \
    --dataset_path ~/LDV-data \
    --select 3 \
    --out path/to/speech_intervals.json

Notes:
- Fails fast if test log or standard_file.wav missing, or if no segment1
  intervals can be identified.
- Deterministic selection avoids ad-hoc heuristics; adjust --select if you
  want a different number of clips per angle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import librosa
import numpy as np


@dataclass
class Interval:
    start: int
    end: int

    @property
    def duration(self) -> int:
        return self.end - self.start


def parse_time_len_to_ms(time_str: str) -> int:
    """Convert HH:MM:SS.mmm string to milliseconds (accepts SS or SS.mmm)."""
    h, m, s = time_str.split(":")
    if "." in s:
        sec, ms = s.split(".")
        total_ms = (int(h) * 3600 + int(m) * 60 + int(sec)) * 1000 + int(ms.ljust(3, "0"))
    else:
        total_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000
    return total_ms


def split_intervals_with_padding(y: np.ndarray, sr: int, top_db: float, padding_sec: float) -> List[Interval]:
    raw_intervals = librosa.effects.split(y, top_db=top_db)
    pad_samples = int(padding_sec * sr)
    out: List[Interval] = []
    for start, end in raw_intervals:
        new_start = max(0, start - pad_samples)
        new_end = min(len(y), end + pad_samples)
        out.append(Interval(new_start, new_end))
    return out


def choose_evenly_spaced_indices(n: int, k: int) -> List[int]:
    if k <= 0:
        return []
    if n < k:
        # Not enough intervals; choose all available
        return list(range(n))
    # Positions at quantiles 1..k over (k+1)
    idxs = []
    for j in range(1, k + 1):
        pos = int(round(j * (n + 1) / (k + 1))) - 1
        pos = max(0, min(n - 1, pos))
        idxs.append(pos)
    # Ensure uniqueness in rare degenerate cases
    seen = set()
    unique = []
    for i in idxs:
        if i not in seen:
            unique.append(i)
            seen.add(i)
    return unique


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate speech interval manifest from standard_file.wav")
    ap.add_argument("--dataset_path", type=str, default=str(Path.home() / "LDV-data"),
                    help="Path to LDV-data root (contains standard_file.wav and LDV-data_test_log.txt)")
    ap.add_argument("--top_db", type=float, default=70.0, help="librosa split top_db threshold")
    ap.add_argument("--padding", type=float, default=0.0, help="Padding seconds around intervals")
    ap.add_argument("--min_duration", type=float, default=0.8, help="Minimum interval duration in seconds")
    ap.add_argument("--select", type=int, default=3, help="Number of intervals to select deterministically")
    ap.add_argument("--out", type=str, required=True, help="Output JSON manifest path")
    args = ap.parse_args()

    dataset_path = Path(args.dataset_path).expanduser()
    test_log_path = dataset_path / "LDV-data_test_log.txt"
    standard_file = dataset_path / "standard_file.wav"

    if not standard_file.exists():
        raise FileNotFoundError(f"Missing standard_file.wav at {standard_file}")
    if not test_log_path.exists():
        raise FileNotFoundError(f"Missing LDV-data_test_log.txt at {test_log_path}")

    info = json.loads(test_log_path.read_text(encoding="utf-8"))
    if "Audio" not in info or "segment1" not in info["Audio"]:
        raise ValueError("LDV-data_test_log.txt missing Audio.segment1 metadata")

    seg1_len_ms = parse_time_len_to_ms(info["Audio"]["segment1"]["time_len"])

    # Load standard file and split intervals
    y, sr = librosa.load(str(standard_file), sr=None)
    intervals = split_intervals_with_padding(y, sr, args.top_db, args.padding)
    # Filter by min duration
    intervals = [iv for iv in intervals if iv.duration / sr >= args.min_duration]
    if not intervals:
        raise RuntimeError("No intervals found after min_duration filtering; adjust top_db/min_duration")

    seg1_end_sample = int(seg1_len_ms / 1000.0 * sr)
    seg1_intervals = [iv for iv in intervals if iv.end <= seg1_end_sample]
    if not seg1_intervals:
        raise RuntimeError("No segment1 intervals found; check test_log Audio.segment1.time_len or splitting params")

    # Choose K evenly spaced indices within segment1
    k = int(args.select)
    chosen_local = choose_evenly_spaced_indices(len(seg1_intervals), k)
    # Map back to global indices
    first_global_idx = intervals.index(seg1_intervals[0])
    chosen_global = [first_global_idx + i for i in chosen_local]

    # Ensure pure Python ints in JSON (avoid numpy.int64)
    chosen_global = [int(x) for x in chosen_global]

    manifest: Dict[str, Any] = {
        "indices": chosen_global,
        "segment": "segment1",
        "sample_rate": int(sr),
        "params": {
            "top_db": args.top_db,
            "padding": args.padding,
            "min_duration": args.min_duration,
        },
        "counts": {
            "intervals_total": int(len(intervals)),
            "intervals_segment1": int(len(seg1_intervals)),
            "selected": int(len(chosen_global)),
        },
        "intervals_segment1": [
            {
                "start": int(iv.start),
                "end": int(iv.end),
                "duration_sec": round(iv.duration / sr, 6),
            }
            for iv in seg1_intervals
        ],
        "selected_details": [
            {
                "global_index": int(idx),
                "start": int(intervals[idx].start),
                "end": int(intervals[idx].end),
                "duration_sec": round(intervals[idx].duration / sr, 6),
            }
            for idx in chosen_global
        ],
    }

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved speech interval manifest to: {out_path}")
    print(f"Selected indices (global): {chosen_global}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
