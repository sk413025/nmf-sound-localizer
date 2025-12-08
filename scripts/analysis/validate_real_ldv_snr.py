#!/usr/bin/env python3
"""
Validate Real LDV SNR - Measure actual SNR degradation from Original to Box.

This script computes the frequency-domain SNR in the model-relevant band [300, 3000]Hz
to determine what synthetic SNR level corresponds to actual LDV Box hardware.

Expected: ~12 dB (from 4× magnitude reduction observed in commit 2a6ec4a)

Usage:
    python scripts/analysis/validate_real_ldv_snr.py \
        --original_dir ~/LDV-data-processed/white_noise_original_data_no_edge \
        --box_dir ~/LDV-data-processed/white_noise_box_data_no_edge \
        --num_clips 20
"""

import argparse
import numpy as np
from pathlib import Path
from scipy import signal as sp_signal
from typing import List, Tuple


def compute_snr_frequency_domain(
    original: np.ndarray,
    box: np.ndarray,
    fs: int = 48000,
    n_fft: int = 2048,
    freq_range: Tuple[float, float] = (300, 3000)
) -> float:
    """
    Compute SNR degradation from Original to Box in the frequency domain.

    This computes how much the Box signal power has degraded compared to Original
    in the [300, 3000]Hz band.

    SNR_dB = 10 * log10(Power_Box / Power_Original)

    Negative SNR means Box is weaker than Original (typical for LDV degradation).

    Interpretation:
    - SNR = -12 dB → Box has 1/16 the power of Original (4× magnitude reduction)
    - SNR = -20 dB → Box has 1/100 the power of Original (10× magnitude reduction)
    """
    # Compute STFT for both signals
    hop_length = n_fft // 4  # 512

    f_orig, t_orig, X_orig = sp_signal.stft(
        original, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    f_box, t_box, X_box = sp_signal.stft(
        box, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    # Select frequency band [300, 3000]Hz (model-relevant)
    freq_mask = (f_orig >= freq_range[0]) & (f_orig <= freq_range[1])

    # Compute power in the band
    X_orig_band = X_orig[freq_mask, :]
    X_box_band = X_box[freq_mask, :]

    # Power in frequency band
    orig_power = np.sum(np.abs(X_orig_band) ** 2)
    box_power = np.sum(np.abs(X_box_band) ** 2)

    if orig_power == 0:
        return -np.inf

    # SNR = Box power / Original power (typically negative)
    snr_db = 10 * np.log10(box_power / orig_power)
    return snr_db


def compute_magnitude_ratio(original: np.ndarray, box: np.ndarray) -> float:
    """
    Compute magnitude ratio between Original and Box signals.

    This is a simple time-domain metric to verify the ~4× reduction
    observed in commit 2a6ec4a.
    """
    orig_rms = np.sqrt(np.mean(original ** 2))
    box_rms = np.sqrt(np.mean(box ** 2))

    if box_rms == 0:
        return np.inf

    return orig_rms / box_rms


def validate_real_ldv_snr(
    original_dir: Path,
    box_dir: Path,
    num_clips: int = 20,
    angle: int = 0
) -> dict:
    """
    Validate real LDV SNR by comparing Original and Box recordings.

    Returns statistics on SNR and magnitude ratio across multiple clips.

    Note: Original files are in the root directory (no angle subdirs),
          Box files are organized in angle_* subdirectories.
    """
    original_dir = Path(original_dir)
    box_dir = Path(box_dir) / f"angle_{angle}"

    if not original_dir.exists():
        raise FileNotFoundError(f"Original directory not found: {original_dir}")
    if not box_dir.exists():
        raise FileNotFoundError(f"Box directory not found: {box_dir}")

    # Get Original files (they have "original_" prefix)
    original_files = sorted(original_dir.glob("original_*.npy"))[:num_clips]

    if not original_files:
        raise ValueError(f"No original_*.npy files found in {original_dir}")

    print(f"Analyzing {len(original_files)} clips:")
    print(f"  Original: {original_dir}")
    print(f"  Box:      {box_dir}")
    print("=" * 80)

    snr_values = []
    magnitude_ratios = []

    for orig_file in original_files:
        # Corresponding Box file (remove "original_" prefix)
        box_filename = orig_file.name.replace("original_", "")
        box_file = box_dir / box_filename

        if not box_file.exists():
            print(f"Warning: Box file not found: {box_file}")
            continue

        # Load signals
        original = np.load(orig_file)
        box = np.load(box_file)

        # Ensure same length (trim to shorter if needed)
        min_len = min(len(original), len(box))
        original = original[:min_len]
        box = box[:min_len]

        # Compute metrics
        snr_db = compute_snr_frequency_domain(original, box, fs=48000)
        mag_ratio = compute_magnitude_ratio(original, box)

        snr_values.append(snr_db)
        magnitude_ratios.append(mag_ratio)

        print(f"{orig_file.name:30s} | SNR: {snr_db:6.2f} dB | Mag Ratio: {mag_ratio:5.2f}×")

    print("=" * 80)

    # Compute statistics
    snr_mean = np.mean(snr_values)
    snr_std = np.std(snr_values)
    snr_median = np.median(snr_values)

    mag_mean = np.mean(magnitude_ratios)
    mag_std = np.std(magnitude_ratios)

    print(f"\nSTATISTICS (n={len(snr_values)} clips):")
    print(f"  Frequency-domain SNR [300, 3000]Hz (MODEL-RELEVANT):")
    print(f"    Mean:   {snr_mean:6.2f} ± {snr_std:5.2f} dB")
    print(f"    Median: {snr_median:6.2f} dB")
    print(f"    Range:  [{np.min(snr_values):6.2f}, {np.max(snr_values):6.2f}] dB")
    print(f"\n  Magnitude Ratio (Original/Box):")
    print(f"    Mean:   {mag_mean:5.2f} ± {mag_std:4.2f}×")
    print(f"    Expected: ~4× (from commit 2a6ec4a)")
    print(f"    Implied SNR: {20 * np.log10(mag_mean):6.2f} dB (20*log10({mag_mean:.2f}))")
    print("=" * 80)

    return {
        'snr_mean': snr_mean,
        'snr_std': snr_std,
        'snr_median': snr_median,
        'snr_min': np.min(snr_values),
        'snr_max': np.max(snr_values),
        'snr_values': snr_values,
        'magnitude_ratio_mean': mag_mean,
        'magnitude_ratio_std': mag_std,
        'magnitude_ratios': magnitude_ratios,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate real LDV SNR by comparing Original and Box recordings"
    )
    parser.add_argument(
        "--original_dir",
        type=str,
        required=True,
        help="Path to Original recordings (e.g., ~/LDV-data-processed/white_noise_original_data_no_edge)"
    )
    parser.add_argument(
        "--box_dir",
        type=str,
        required=True,
        help="Path to Box recordings (e.g., ~/LDV-data-processed/white_noise_box_data_no_edge)"
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=20,
        help="Number of clips to analyze (default: 20)"
    )
    parser.add_argument(
        "--angle",
        type=int,
        default=0,
        help="Angle to analyze (default: 0)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("REAL LDV SNR VALIDATION")
    print("=" * 80)
    print(f"Original directory: {args.original_dir}")
    print(f"Box directory: {args.box_dir}")
    print(f"Number of clips: {args.num_clips}")
    print(f"Angle: {args.angle}")
    print("=" * 80)
    print()

    results = validate_real_ldv_snr(
        Path(args.original_dir),
        Path(args.box_dir),
        num_clips=args.num_clips,
        angle=args.angle
    )

    print("\nINTERPRETATION:")
    snr_mean = results['snr_mean']
    mag_mean = results['magnitude_ratio_mean']

    print(f"Real LDV degradation (Box vs Original):")
    print(f"  Power ratio:     {snr_mean:.2f} dB (Box has {10**(snr_mean/10):.4f}× power of Original)")
    print(f"  Magnitude ratio: {mag_mean:.2f}× (Box is {mag_mean:.2f}× quieter)")
    print()

    # Expected: ~12 dB degradation (4× magnitude reduction)
    expected_snr = -12.0  # 20*log10(1/4) = -12 dB power, or 10*log10(1/16) = -12 dB power

    if -15 <= snr_mean <= -9:
        print(f"✓ Real LDV degradation ≈ {snr_mean:.1f} dB is within expected range [-15, -9] dB")
        print(f"  Expected degradation: ~-12 dB (4× magnitude reduction)")
        print(f"  Recommendation: Current SNR range is appropriate")
        print(f"  Suggested SNR sweep: inf, 30, 20, 15, 10, 5, 0 dB")
    elif snr_mean > -9:
        print(f"! Real LDV degradation ≈ {snr_mean:.1f} dB is LESS severe than expected (~-12 dB)")
        print(f"  Implication: Box signal is stronger than anticipated")
        print(f"  Recommendation: May need to adjust baseline reference")
    else:
        print(f"! Real LDV degradation ≈ {snr_mean:.1f} dB is MORE severe than expected (~-12 dB)")
        print(f"  Implication: Box signal is much weaker than anticipated ({mag_mean:.1f}× vs 4×)")
        print(f"  Recommendation: This suggests Box recordings may not be suitable for training")
        print(f"  Action: Investigate why Box signal is so weak (sensor issue? setup problem?)")

    print("\nNEXT STEPS:")
    print("1. If SNR is within expected range → Proceed to Phase 3 (full data generation)")
    print("2. If SNR is outside range → Adjust SNR levels and update experiment plan")
    print("3. Compare noise spectra (real vs synthetic) to validate spectral shaping")
    print("=" * 80)


if __name__ == "__main__":
    main()
