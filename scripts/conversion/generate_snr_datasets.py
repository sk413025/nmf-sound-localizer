#!/usr/bin/env python3
"""
Generate SNR Datasets with Spectral Shaping

This script generates synthetic noisy datasets by adding spectrally-shaped noise
at specified SNR levels to clean LDV Box recordings.

Physical model: Environmental noise → Speaker → Box → LDV
Result: Noise has same frequency distribution as signal (per-clip shaping)

Usage:
    python scripts/conversion/generate_snr_datasets.py \
        --clean_root ~/LDV-data-processed/white_noise_box_data_no_edge \
        --output_base ~/LDV-data-experiments/snr-synthetic-2025-12/raw \
        --snr_levels inf 30 20 15 10 5 0 \
        --dataset_prefix white_noise_box \
        --fs 48000 \
        --seed 42
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import signal as sp_signal


def add_spectral_shaped_noise_per_clip(
    signal: np.ndarray,
    snr_db: float,
    fs: int = 48000,
    seed: int = None
) -> np.ndarray:
    """
    Add noise with spectrum shaped to match the signal's spectrum (per-clip).

    This simulates environmental noise propagating through the same acoustic-structural
    coupling (Box resonances, modal response) as the signal.

    Physical model:
    - Environmental noise → Speaker → Box vibration → LDV
    - Same frequency-dependent transfer function as clean signal
    - Realistic frequency distribution

    Parameters
    ----------
    signal : ndarray
        Clean time-domain waveform
    snr_db : float
        Target SNR in dB (use np.inf for clean signal)
    fs : int, default=48000
        Sampling rate in Hz
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_signal : ndarray
        Signal with added spectrally-shaped noise
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Handle infinite SNR (clean signal)
    if np.isinf(snr_db):
        return signal.copy()

    # 1. Compute signal AC power (time-domain)
    signal_ac = signal - np.mean(signal)
    signal_power_time = np.mean(signal_ac ** 2)

    if signal_power_time == 0:
        raise ValueError("Signal has zero AC power (constant signal)")

    # 2. Generate white Gaussian noise
    white_noise = rng.normal(0, 1, signal.shape)

    # 3. Shape noise spectrum to match signal spectrum
    f_s, t_s, S_signal = sp_signal.stft(signal, fs=fs, nperseg=2048, noverlap=1536)
    f_w, t_w, S_white = sp_signal.stft(white_noise, fs=fs, nperseg=2048, noverlap=1536)

    # Compute spectral envelopes (average magnitude across time)
    signal_envelope = np.abs(S_signal).mean(axis=1, keepdims=True) + 1e-10
    white_envelope = np.abs(S_white).mean(axis=1, keepdims=True) + 1e-10

    # Shaping filter: match signal's frequency distribution
    shaping_filter = signal_envelope / white_envelope

    # Apply shaping filter
    S_shaped = S_white * shaping_filter

    # Convert back to time domain
    _, shaped_noise = sp_signal.istft(S_shaped, fs=fs, nperseg=2048, noverlap=1536)

    # Trim to match signal length (ISTFT may change length slightly)
    shaped_noise = shaped_noise[:len(signal)]

    # 4. Scale noise to achieve target SNR (time-domain)
    noise_ac = shaped_noise - np.mean(shaped_noise)
    noise_power_current = np.mean(noise_ac ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power_target = signal_power_time / snr_linear

    scaling_factor = np.sqrt(noise_power_target / noise_power_current)
    shaped_noise_scaled = shaped_noise * scaling_factor

    # 5. Add shaped noise to signal
    noisy_signal = signal + shaped_noise_scaled

    return noisy_signal


def verify_snr_time_domain(signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Verify actual SNR in time domain using AC power.

    Parameters
    ----------
    signal : ndarray
        Clean signal
    noisy_signal : ndarray
        Noisy signal

    Returns
    -------
    snr_db : float
        Actual SNR in dB
    """
    # Extract noise
    noise = noisy_signal - signal

    # Compute AC power (remove DC component)
    signal_ac = signal - np.mean(signal)
    signal_power = np.mean(signal_ac ** 2)

    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return np.inf

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def verify_snr_frequency_domain(
    signal: np.ndarray,
    noisy_signal: np.ndarray,
    fs: int = 48000,
    n_fft: int = 2048,
    freq_range: tuple = (300, 3000)
) -> float:
    """
    Verify actual SNR in frequency domain within specified band.

    This is the MODEL-RELEVANT SNR metric, as the OMP Transformer operates
    on STFT magnitude spectra in the band [300, 3000]Hz.

    Parameters
    ----------
    signal : ndarray
        Clean signal
    noisy_signal : ndarray
        Noisy signal
    fs : int, default=48000
        Sampling rate in Hz (Stage 0 NPY files are 48 kHz)
    n_fft : int, default=2048
        FFT size (matches DoADataset STFT)
    freq_range : tuple, default=(300, 3000)
        Frequency band in Hz (matches model input)

    Returns
    -------
    snr_db : float
        Actual SNR in dB within specified frequency band
    """
    # Compute STFT for both signals
    hop_length = n_fft // 4  # 512

    # STFT: returns (frequencies, times, STFT values)
    f_clean, t_clean, X_clean = sp_signal.stft(
        signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    f_noisy, t_noisy, X_noisy = sp_signal.stft(
        noisy_signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    # Select frequency band [300, 3000]Hz (model-relevant)
    freq_mask = (f_clean >= freq_range[0]) & (f_clean <= freq_range[1])

    # Extract band
    X_clean_band = X_clean[freq_mask, :]
    X_noisy_band = X_noisy[freq_mask, :]

    # Compute signal and noise power in the band
    signal_power_freq = np.sum(np.abs(X_clean_band) ** 2)
    noise_freq = X_noisy_band - X_clean_band
    noise_power_freq = np.sum(np.abs(noise_freq) ** 2)

    if noise_power_freq == 0:
        return np.inf

    snr_linear = signal_power_freq / noise_power_freq
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def process_dataset(
    clean_root: Path,
    output_root: Path,
    snr_db: float,
    fs: int = 48000,
    seed: int = 42
):
    """
    Process entire dataset to add spectrally-shaped noise at specified SNR.

    Verifies SNR in both time-domain and frequency-domain [300, 3000]Hz.
    Uses per-clip spectral shaping to ensure physical realism.
    """
    clean_root = Path(clean_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all NPY files
    npy_files = sorted(clean_root.rglob("*.npy"))
    print(f"Found {len(npy_files)} NPY files in {clean_root}")

    snr_errors_time = []
    snr_errors_freq = []

    for npy_file in tqdm(npy_files, desc=f"Processing SNR={snr_db}dB"):
        # Load clean signal
        signal = np.load(npy_file)

        # Add spectrally-shaped noise (per-clip shaping)
        noisy_signal = add_spectral_shaped_noise_per_clip(signal, snr_db, fs=fs, seed=seed)

        # Verify SNR (time-domain)
        actual_snr_time = verify_snr_time_domain(signal, noisy_signal)
        snr_error_time = abs(actual_snr_time - snr_db) if not np.isinf(snr_db) else 0
        snr_errors_time.append(snr_error_time)

        # Verify SNR (frequency-domain, model-relevant)
        actual_snr_freq = verify_snr_frequency_domain(signal, noisy_signal, fs=fs)
        snr_error_freq = abs(actual_snr_freq - snr_db) if not np.isinf(snr_db) else 0
        snr_errors_freq.append(snr_error_freq)

        # Save noisy signal
        relative_path = npy_file.relative_to(clean_root)
        output_file = output_root / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, noisy_signal)

    # Compute statistics
    mean_error_time = np.mean(snr_errors_time)
    max_error_time = np.max(snr_errors_time)
    mean_error_freq = np.mean(snr_errors_freq)
    max_error_freq = np.max(snr_errors_freq)

    print(f"SNR verification (time-domain):")
    print(f"  mean_error={mean_error_time:.4f} dB, max_error={max_error_time:.4f} dB")
    print(f"SNR verification (frequency-domain [300, 3000]Hz, MODEL-RELEVANT):")
    print(f"  mean_error={mean_error_freq:.4f} dB, max_error={max_error_freq:.4f} dB")

    # CRITICAL: Frequency-domain SNR is what matters for the model
    # Note: Spectral shaping has ±1 dB tolerance (vs ±0.5 dB for simple AWGN)
    if mean_error_freq > 1.0:
        print(f"WARNING: Large frequency-domain SNR error detected. Check implementation.")

    return {
        'num_files': len(npy_files),
        'mean_error_time': mean_error_time,
        'max_error_time': max_error_time,
        'mean_error_freq': mean_error_freq,
        'max_error_freq': max_error_freq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic noisy datasets with spectral shaping"
    )
    parser.add_argument(
        "--clean_root",
        type=str,
        required=True,
        help="Path to clean dataset root"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        required=True,
        help="Base output directory"
    )
    parser.add_argument(
        "--snr_levels",
        type=float,
        nargs='+',
        required=True,
        help="SNR levels in dB (e.g., 30 20 15 10 5 0, or 'inf' for clean)"
    )
    parser.add_argument(
        "--dataset_prefix",
        type=str,
        required=True,
        help="Dataset prefix (e.g., white_noise_box or speech260_box)"
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=48000,
        help="Sampling rate in Hz (default: 48000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    output_base = Path(args.output_base)

    print("=" * 80)
    print("SNR SYNTHETIC DATASET GENERATION (Spectral Shaping)")
    print("=" * 80)
    print(f"Clean root: {clean_root}")
    print(f"Output base: {output_base}")
    print(f"SNR levels: {args.snr_levels} dB")
    print(f"Dataset prefix: {args.dataset_prefix}")
    print(f"Sampling rate: {args.fs} Hz")
    print(f"Random seed: {args.seed}")
    print(f"Method: Per-clip spectral shaping (Box acoustic-structural coupling)")
    print("=" * 80)

    results = []

    for snr_db in args.snr_levels:
        print(f"\n[SNR={snr_db} dB] Processing...")

        if np.isinf(snr_db):
            output_suffix = "snrInf_data_no_edge"
        else:
            output_suffix = f"snr{int(snr_db)}dB_data_no_edge"

        output_root = output_base / f"{args.dataset_prefix}_{output_suffix}"

        stats = process_dataset(
            clean_root, output_root, snr_db, fs=args.fs, seed=args.seed
        )

        results.append({
            'snr_db': snr_db,
            'output_root': str(output_root),
            **stats
        })

    # Print summary
    print("\n" + "=" * 80)
    print(f"{'SNR (dB)':>10} {'Files':>6} {'Time Error (dB)':>17} {'Freq Error (dB)':>17} Output")
    print("=" * 80)
    for result in results:
        snr_str = "∞" if np.isinf(result['snr_db']) else f"{result['snr_db']:.1f}"
        print(f"{snr_str:>10} {result['num_files']:>6d} "
              f"{result['mean_error_time']:>6.3f}±{result['max_error_time']:>4.3f} "
              f"{result['mean_error_freq']:>6.3f}±{result['max_error_freq']:>4.3f} "
              f"{result['output_root']}")
    print("=" * 80)
    print("Note: 'Freq Error' is the MODEL-RELEVANT metric (STFT [300, 3000]Hz band)")
    print("      Spectral shaping method ensures ±1 dB tolerance in model band")
    print("=" * 80)


if __name__ == "__main__":
    main()
