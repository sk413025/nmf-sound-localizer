#!/usr/bin/env python3
"""
Quick USM reconstruction check on speech260 data.

Loads the speech-trained USM W and reconstructs a few spectrograms from
speech260_original_data_no_edge_sync_vad_normalized, then saves
side-by-side time–frequency plots (original vs reconstruction) and
corresponding waveform pairs (orig / recon) as WAV files.

Outputs are written under results/usm_speech260_recon_<timestamp>/.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from nmf_localizer.utils.audio_utils import AudioProcessor


def select_examples(root: Path, angles: List[int], max_per_angle: int) -> List[Tuple[int, Path]]:
    examples: List[Tuple[int, Path]] = []
    for angle in angles:
        angle_dir = root / f"angle_{angle}"
        if not angle_dir.exists():
            continue
        npy_files = sorted(angle_dir.glob("*.npy"))
        for f in npy_files[:max_per_angle]:
            examples.append((angle, f))
    return examples


def nmf_with_fixed_W(W: torch.Tensor, V: torch.Tensor, n_iter: int = 200) -> torch.Tensor:
    """
    Simple Euclidean NMF update for H with fixed W.

    V: (F, T), W: (F, K) non-negative; returns H: (K, T).
    """
    F, K = W.shape
    Fv, T = V.shape
    assert Fv == F, f"V.shape[0]={Fv} must match W.shape[0]={F}"

    device = V.device
    W = W.to(device)

    # Initialize H with small positive values
    H = torch.rand(K, T, device=device) * 0.01
    eps = 1e-8

    WT = W.t()
    for _ in range(n_iter):
        WH = W @ H  # (F, T)
        numerator = WT @ V  # (K, T)
        denominator = WT @ WH + eps
        H = H * (numerator / denominator)

    return H


def save_wav(path: Path, audio: np.ndarray, fs: int = 16000) -> None:
    """Save mono audio as 16-bit PCM WAV after simple normalization."""
    if audio.ndim > 1:
        audio = audio.squeeze()
    x = audio.astype(np.float32)
    x = x - float(x.mean())
    maxabs = float(np.max(np.abs(x)) + 1e-8)
    if maxabs > 0:
        x = x / maxabs
    x16 = (x * 32767.0).clip(-32767, 32767).astype(np.int16)
    wavfile.write(str(path), fs, x16)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    usm_path = repo_root / "doa_speech260_config_c_corrected/models/usm.pth"
    # Use 16 kHz resampled speech260 original data to align with H/DoADataset
    data_root = Path(
        "/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / f"results/usm_speech260_recon_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load USM W
    if not usm_path.exists():
        raise FileNotFoundError(f"USM file not found: {usm_path}")
    ckpt = torch.load(usm_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "W" in ckpt:
        W = ckpt["W"]
    else:
        W = ckpt
    W = torch.as_tensor(W, dtype=torch.float32)
    F_w, K = W.shape

    # Select a few examples (angles near 0, 90, 180)
    angles = [0, 90, 180]
    examples = select_examples(data_root, angles, max_per_angle=1)
    if not examples:
        raise RuntimeError(f"No examples found under {data_root}")

    metrics = []
    audio_proc = AudioProcessor()

    for angle, path in examples:
        wav = np.load(path)
        if wav.ndim > 1:
            wav = wav[0]

        freqs, times, stft, magnitude = audio_proc.compute_stft_spectrogram(
            wav, fs=16000, nperseg=2048, window="hann"
        )
        # Band-limit to match H/DoADataset (300–3000 Hz)
        mag_tensor = torch.from_numpy(magnitude).float()
        mag_band, freqs_band = audio_proc.apply_frequency_filter(
            mag_tensor, freqs, 300.0, 3000.0
        )

        V = mag_band  # (F, T)
        F_v, T = V.shape

        if F_v != F_w:
            # Align frequency dimension to W by simple trimming when off by 1.
            if F_v > F_w:
                start = (F_v - F_w) // 2
                end = start + F_w
                V = V[start:end, :]
                F_v, T = V.shape
            if F_v != F_w:
                metrics.append(
                    {
                        "angle": angle,
                        "path": str(path),
                        "error": f"freq_dim_mismatch V={F_v}, W={F_w}",
                    }
                )
                continue

        V = V.to(torch.float32).contiguous()
        H = nmf_with_fixed_W(W, V, n_iter=200)
        V_rec = (W @ H).clamp_min(0.0)

        # Compute relative Frobenius error
        num = torch.norm(V - V_rec)
        den = torch.norm(V) + 1e-8
        rel_err = float((num / den).item())

        # Convert to numpy for plotting (log-scale for visibility) and audio reconstruction
        V_np = V.detach().cpu().numpy()
        V_rec_np = V_rec.detach().cpu().numpy()

        # Reconstruct audio using original phase for the band-limited region
        # Build a modified STFT with reconstructed magnitudes in the band
        stft_rec = stft.copy()
        # Full-band indices corresponding to 300–3000 Hz
        mask = (freqs >= 300.0) & (freqs <= 3000.0)
        band_indices = np.where(mask)[0]
        full_band_len = len(band_indices)
        out_wav_orig = None
        out_wav_rec = None
        if full_band_len > 0:
            if full_band_len >= F_w:
                start_fb = (full_band_len - F_w) // 2
                used_band_idx = band_indices[start_fb : start_fb + F_w]
            else:
                used_band_idx = band_indices
            if len(used_band_idx) == F_w:
                phase_band = np.angle(stft[used_band_idx, :])
                stft_rec[used_band_idx, :] = V_rec_np * np.exp(1j * phase_band)
                # ISTFT to get reconstructed waveform
                _, wav_rec = signal.istft(
                    stft_rec,
                    fs=16000,
                    window="hann",
                    nperseg=2048,
                    noverlap=int(2048 * 0.75),
                )
                # Align lengths for comparison
                n = min(len(wav), len(wav_rec))
                wav_orig_aligned = wav[:n]
                wav_rec_aligned = wav_rec[:n]
                base = f"angle_{angle:03d}_{path.stem}"
                out_wav_orig = out_dir / f"{base}_orig.wav"
                out_wav_rec = out_dir / f"{base}_rec.wav"
                save_wav(out_wav_orig, wav_orig_aligned, fs=16000)
                save_wav(out_wav_rec, wav_rec_aligned, fs=16000)

        def to_log(x: np.ndarray) -> np.ndarray:
            return np.log10(x + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        im0 = axes[0].imshow(
            to_log(V_np),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        axes[0].set_title("Original (log10 magnitude)")
        axes[0].set_xlabel("Time frames")
        axes[0].set_ylabel("Freq bin")

        im1 = axes[1].imshow(
            to_log(V_rec_np),
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        axes[1].set_title("Reconstruction (log10 magnitude)")
        axes[1].set_xlabel("Time frames")

        fig.suptitle(f"angle={angle}°, file={path.name}, rel_err={rel_err:.3f}")
        fig.colorbar(im0, ax=axes.ravel().tolist(), shrink=0.6)
        fig.tight_layout()

        angle_str = f"{angle:03d}"
        out_png = out_dir / f"angle_{angle_str}_{path.stem}_spec.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        metrics.append(
            {
                "angle": angle,
                "path": str(path),
                "rel_fro_error": rel_err,
                "freq_bins": F_v,
                "time_frames": T,
                "png": str(out_png),
                "wav_orig": str(out_wav_orig) if out_wav_orig is not None else None,
                "wav_rec": str(out_wav_rec) if out_wav_rec is not None else None,
            }
        )

    # Save metrics
    with (out_dir / "metrics.json").open("w") as f:
        json.dump({"examples": metrics}, f, indent=2)

    print(f"Saved reconstruction plots and metrics under: {out_dir}")


if __name__ == "__main__":
    main()
