#!/usr/bin/env python3
"""
Export all 1-D waveform .npy files under a root directory to .wav for listening.

Behavior:
- Recursively finds all .npy files.
- Loads arrays; if 1-D, writes a sibling .wav with the same stem.
- Sample rate: --sr (default 48000).
- Amplitude handling (--amplitude):
  - auto (default):
      * If data in [0,1] and mean≈0.5, recenter to [-1,1] via (x-0.5)*2.
      * Else if max abs > 1.0, normalize to [-1,1] by max abs.
      * Else (already plausible audio), write as-is.
  - raw: write as-is (clipped to [-1,1] to avoid WAV overflow).
  - normalize: scale to [-1,1] by max abs (if max abs > 0).

Usage:
  python scripts/conversion/export_npy_waveforms_to_wav.py \
    --root ~/LDV-data-processed \
    --sr 48000 \
    --amplitude auto

Logs a brief report to stdout; for reproducibility, run via a shell wrapper that tees to results/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import sys
import numpy as np


def load_writer():
    try:
        import soundfile as sf

        def write_wav(path: Path, audio: np.ndarray, sr: int):
            sf.write(str(path), audio, sr)

        return write_wav, 'soundfile'
    except Exception:
        from scipy.io import wavfile

        def write_wav(path: Path, audio: np.ndarray, sr: int):
            # scipy expects int or float; clip to [-1,1]
            wavfile.write(str(path), sr, audio.astype(np.float32))

        return write_wav, 'scipy.io.wavfile'


def prepare_audio(x: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    x = x.astype(np.float32).copy()
    note = ''
    if mode == 'raw':
        x = np.clip(x, -1.0, 1.0)
        note = 'raw_clipped'
        return x, note
    if mode == 'normalize':
        m = float(np.max(np.abs(x)))
        if m > 0:
            x = x / m
            note = 'normalized_by_maxabs'
        else:
            note = 'no_scale_zero_signal'
        return x, note

    # auto mode
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    xmean = float(np.mean(x))
    # If looks like [0,1] with center ~0.5, recenter to [-1,1]
    if xmin >= -1e-6 and xmax <= 1.0 + 1e-6 and abs(xmean - 0.5) <= 0.15:
        x = (x - 0.5) * 2.0
        note = 'auto_recentred_0_1_to_pm1'
    else:
        # If peaks > 1, normalize to [-1,1]
        m = float(np.max(np.abs(x)))
        if m > 1.0:
            x = x / m
            note = 'auto_normalized_by_maxabs'
        else:
            note = 'auto_as_is'
    x = np.clip(x, -1.0, 1.0)
    return x, note


def main() -> int:
    ap = argparse.ArgumentParser(description='Export waveform npy to wav')
    ap.add_argument('--root', type=str, required=True, help='Root directory to search for .npy files')
    ap.add_argument('--sr', type=int, default=48000, help='Output WAV sample rate')
    ap.add_argument('--amplitude', choices=['auto', 'raw', 'normalize'], default='auto', help='Amplitude handling mode')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing .wav files')
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        return 2

    write_wav, backend = load_writer()
    print(f"Using writer backend: {backend}")
    print(f"Root: {root}")
    print(f"Sample rate: {args.sr}")
    print(f"Amplitude mode: {args.amplitude}")

    npy_files = sorted(root.rglob('*.npy'))
    converted = 0
    skipped = 0
    errors = 0
    for npy in npy_files:
        try:
            arr = np.load(npy)
            if arr.ndim != 1:
                skipped += 1
                continue
            wav_path = npy.with_suffix('.wav')
            if wav_path.exists() and not args.overwrite:
                skipped += 1
                continue
            audio, note = prepare_audio(arr, args.amplitude)
            write_wav(wav_path, audio, args.sr)
            converted += 1
            if converted % 200 == 0:
                print(f"Converted {converted} files...")
        except Exception as e:
            print(f"Error converting {npy}: {e}", file=sys.stderr)
            errors += 1

    print(f"Done. Converted={converted}, skipped={skipped}, errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())

