"""
LDV dataset utilities used by training/evaluation scripts.

This module exists to provide a stable import path for:
  - `DoADataset`
  - `create_dataloader`

It is referenced by `scripts/omp-transformer-ldv.py` and related tooling.

Design goals:
- Minimal dependencies (NumPy + SciPy + PyTorch).
- Deterministic, explicit preprocessing.
- Clear data contract: returns a dict with keys `Y`, `angle_deg`, `angle_index`, `path`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy import signal as scipy_signal
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DoASample:
    path: Path
    angle_deg: float
    angle_index: int


def _angle_dir_name(angle_deg: float) -> str:
    # Dataset convention: angle_00, angle_05, ..., angle_180 (zero-padded for < 100).
    angle_int = int(round(float(angle_deg)))
    return f"angle_{angle_int:02d}"


def _load_waveform_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D waveform .npy, got shape {arr.shape} for {path}")
    # Ensure float64 for stable SciPy processing, then cast later.
    return arr.astype(np.float64, copy=False)


def _stft_mag_band(
    x: np.ndarray,
    *,
    fs: int,
    n_fft: int,
    window: str,
    freq_min: float,
    freq_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute |STFT| and return magnitude restricted to [freq_min, freq_max].

    Notes:
    - Uses scipy.signal.stft with detrend='constant' to remove DC offset.
    - Uses hop = n_fft//2 (noverlap = n_fft//2), matching common reference settings.
    """
    if n_fft <= 0:
        raise ValueError("n_fft must be positive.")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if not (freq_min < freq_max):
        raise ValueError(f"Invalid band: freq_min={freq_min} must be < freq_max={freq_max}.")

    f, _t, Zxx = scipy_signal.stft(
        x,
        fs=fs,
        window=window,
        nperseg=n_fft,
        noverlap=n_fft // 2,
        nfft=n_fft,
        detrend="constant",
        return_onesided=True,
        boundary=None,
        padded=False,
    )
    mag = np.abs(Zxx)  # (F_full, frames)
    mask = (f >= float(freq_min)) & (f <= float(freq_max))
    if not np.any(mask):
        raise ValueError(
            f"No STFT bins within [{freq_min}, {freq_max}] Hz. "
            f"Check fs={fs}, n_fft={n_fft}."
        )
    return f[mask], mag[mask]


class DoADataset(Dataset[dict[str, Any]]):
    """
    Dataset of waveform .npy files organized by angle directory.

    Expected layout:
      root/
        angle_00/clip_000.npy
        angle_00/clip_001.npy
        ...
        angle_05/...

    Returns per sample:
      {
        "Y": torch.FloatTensor of shape (F, frames)  (magnitude STFT in [freq_min,freq_max]),
        "angle_deg": float,
        "angle_index": int,
        "path": str,
      }
    """

    def __init__(
        self,
        *,
        root: str | Path,
        angles: Iterable[float],
        fs: int = 16000,
        n_fft: int = 2048,
        window: str = "hann",
        freq_min: float = 300.0,
        freq_max: float = 3000.0,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.angles_deg = [float(a) for a in angles]
        if not self.angles_deg:
            raise ValueError("angles must be non-empty.")

        self.fs = int(fs)
        self.n_fft = int(n_fft)
        self.window = str(window)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)

        samples: list[DoASample] = []
        for angle_index, angle_deg in enumerate(self.angles_deg):
            angle_dir = self.root / _angle_dir_name(angle_deg)
            if not angle_dir.is_dir():
                # Fail-fast: if the caller requested an angle list, we require it exists.
                raise FileNotFoundError(f"Missing angle directory: {angle_dir}")
            for path in sorted(angle_dir.glob("*.npy")):
                samples.append(DoASample(path=path, angle_deg=angle_deg, angle_index=angle_index))

        if not samples:
            raise ValueError(f"No .npy files found under {self.root}.")

        self._samples = samples

        # Validate expected F for default settings when possible (informative, not a hard constraint).
        # For fs=16000, n_fft=2048, band=[300,3000], expected bins = 346.
        # We do not enforce it because alternative configs may be valid.
        self.expected_F = None
        if self.fs == 16000 and self.n_fft == 2048 and self.freq_min == 300.0 and self.freq_max == 3000.0:
            self.expected_F = 346

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._samples[int(idx)]
        x = _load_waveform_npy(sample.path)
        _f, Y = _stft_mag_band(
            x,
            fs=self.fs,
            n_fft=self.n_fft,
            window=self.window,
            freq_min=self.freq_min,
            freq_max=self.freq_max,
        )
        # Convert to torch (float32) for downstream model code.
        Y_t = torch.from_numpy(Y.astype(np.float32, copy=False))  # (F, frames)
        if self.expected_F is not None and Y_t.shape[0] != self.expected_F:
            raise ValueError(
                f"Unexpected F={Y_t.shape[0]} for {sample.path} (expected {self.expected_F}). "
                f"Check fs/n_fft/band settings."
            )
        return {
            "Y": Y_t,
            "angle_deg": float(sample.angle_deg),
            "angle_index": int(sample.angle_index),
            "path": str(sample.path),
        }


def create_dataloader(dataset: Dataset[dict[str, Any]], *, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """
    Thin wrapper to keep the calling code stable.

    Default settings:
    - batch_size=1 (common for deterministic per-sample processing)
    - shuffle=False (preserve file ordering for reproducibility)
    - num_workers=0 (avoid multiprocessing overhead and ensure deterministic SciPy usage)
    """
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0)

