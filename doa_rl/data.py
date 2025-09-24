from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nmf_localizer.utils.audio_utils import AudioProcessor


class DoADataset(Dataset):
    """Dataset reading normalized Box test clips and producing Y (F,N).

    Matches docs/rl_data_spec.md (Config C): yields dict with
    - Y: (F,N) float32 magnitude spectrogram (band 300–3000 Hz)
    - angle_deg: float
    - angle_index: int (index into angles array)
    - path: str
    """

    def __init__(self, root: str, angles: List[float], fs: int = 16000,
                 n_fft: int = 2048, window: str = "hann",
                 freq_min: float = 300.0, freq_max: float = 3000.0):
        self.root = Path(root)
        self.angles = [float(a) for a in angles]
        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.freq_min = freq_min
        self.freq_max = freq_max

        self.index: List[Tuple[Path, float, int]] = []
        for angle_deg in self.angles:
            angle_dir = self.root / f"angle_{int(angle_deg)}"
            if not angle_dir.exists():
                continue
            for f in sorted(angle_dir.glob("*.npy")):
                self.index.append((f, angle_deg, int(angle_deg)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, angle_deg, _ = self.index[idx]
        wav = np.load(path)
        assert wav.ndim == 1, "Expected mono waveform"
        freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
            wav, fs=self.fs, nperseg=self.n_fft, window=self.window
        )
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        mag_band = magnitude[mask, :].astype(np.float32)
        Y = torch.from_numpy(mag_band)
        return {
            "Y": Y,
            "angle_deg": float(angle_deg),
            "angle_index": self.angles.index(float(angle_deg)),
            "path": str(path)
        }


def create_dataloader(dataset: DoADataset, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

