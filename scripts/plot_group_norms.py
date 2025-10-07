#!/usr/bin/env python3
"""
Plot group norms over angles for a single spectrogram using a saved localizer.

Usage:
  python scripts/plot_group_norms.py --model outputs/.../models/localizer.pth \
         --audio /path/to/sample.npy --output group_norms.png
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmf_localizer.core.localizer import NMFSoundLocalizer
from nmf_localizer.config import NMFConfig


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def build_parser():
    p = argparse.ArgumentParser(
        description='Plot group norms vs angle for a single audio sample',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model', required=True, type=str, help='Path to saved localizer model (.pth)')
    p.add_argument('--audio', required=True, type=str, help='Path to waveform .npy file')
    p.add_argument('--output', required=True, type=str, help='Output image path')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    model_path = Path(args.model)
    audio_path = Path(args.audio)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    if not audio_path.exists():
        logger.error(f"Audio .npy not found: {audio_path}")
        return 1

    # Load localizer and config
    cfg = NMFConfig(device=args.device)
    localizer = NMFSoundLocalizer(cfg)
    localizer.load_model(str(model_path))

    # Load waveform and compute magnitude STFT consistent with model config
    waveform = np.load(audio_path).astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform[0]

    n_fft = localizer.config.n_fft
    hop = localizer.config.hop_length
    fs = localizer.config.sample_rate

    freqs, times, Zxx = signal.stft(
        waveform,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        window=localizer.config.window,
    )
    Y_mag_full = np.abs(Zxx)

    # Apply frequency band to match model
    mask = (freqs >= localizer.config.freq_min) & (freqs <= localizer.config.freq_max)
    Y_mag = Y_mag_full[mask, :]

    # Ensure frequency dimension matches model.A (n_freq)
    F_model = localizer.n_freq
    if Y_mag.shape[0] != F_model:
        if Y_mag.shape[0] > F_model:
            Y_mag = Y_mag[:F_model, :]
        else:
            pad = np.zeros((F_model - Y_mag.shape[0], Y_mag.shape[1]), dtype=Y_mag.dtype)
            Y_mag = np.concatenate([Y_mag, pad], axis=0)

    Y_tensor = torch.from_numpy(Y_mag).float().to(localizer.device)

    # Localize
    with torch.no_grad():
        result = localizer.localize(Y_tensor, n_sources=1)

    group_norms = result['group_norms'].cpu().numpy()
    angles = localizer.actual_angles.cpu().numpy() if torch.is_tensor(localizer.actual_angles) else np.arange(len(group_norms))
    est_deg = result['directions'].cpu().numpy()

    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(angles, group_norms, width=3.0, align='center')
    for d in np.atleast_1d(est_deg):
        plt.axvline(d, color='r', linestyle='--', label=f'Estimated {d:.1f}°')
    plt.xlabel('Angle (deg)')
    plt.ylabel('Group norm')
    plt.title('Group norms vs Angle')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    logger.info(f"Saved plot to: {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

