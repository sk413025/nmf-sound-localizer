import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from nmf_localizer.utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)

class DoAPairDataset(Dataset):
    """
    Dataset for H-Matrix Exploration.
    Loads pairs of (Microphone, LDV) spectrograms.
    
    Structure:
    - mic_root: Path to 'original' data (Multichannel).
    - ldv_root: Path to 'box' data (Single channel LDV).
    - Assumes matching directory structure: angle_X/clip_Y.npy
    """

    def __init__(self, 
                 mic_root: str, 
                 ldv_root: str, 
                 angles: Optional[List[float]] = None, 
                 fs: int = 16000,
                 n_fft: int = 2048, 
                 window: str = "hann",
                 freq_min: float = 300.0, 
                 freq_max: float = 3000.0):
        self.mic_root = Path(mic_root)
        self.ldv_root = Path(ldv_root)
        self.fs = fs
        self.n_fft = n_fft
        self.window = window
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        # Discover Pairs
        self.index: List[Tuple[Path, Path, float]] = []
        
        # If angles not provided, discover all
        search_root = self.mic_root
        
        # Find angle directories
        angle_dirs = sorted(list(search_root.glob("angle_*")))
        
        for adir in angle_dirs:
            # Parse angle
            try:
                angle_val = float(adir.name.split("_")[1])
            except:
                continue
                
            if angles is not None and angle_val not in angles:
                continue
                
            # Find clips
            clips = sorted(list(adir.glob("*.npy")))
            for mic_path in clips:
                # Construct LDV path
                # mic_path: .../angle_0/clip_0.npy
                # ldv_path: .../angle_0/clip_0.npy
                rel_path = mic_path.relative_to(self.mic_root)
                ldv_path = self.ldv_root / rel_path
                
                if ldv_path.exists():
                    self.index.append((mic_path, ldv_path, angle_val))
                else:
                    logger.warning(f"Missing LDV file for {mic_path}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mic_path, ldv_path, angle = self.index[idx]
        
        # Load Raw
        mic_wav = np.load(mic_path) # (Samples,) or (Channels, Samples)?
        ldv_wav = np.load(ldv_path) # (Samples,)
        
        # Check shapes
        # Expected: Mic (C, N) or (N, C) or (N,). 
        # Usually npy stored as (Channels, Samples) or just (Samples,).
        # We need (Channels, Samples).
        if mic_wav.ndim == 1:
            mic_wav = mic_wav[np.newaxis, :] # Treat as 1 channel if 1D
        
        # Ensure channel first for AudioProcessor? 
        # AudioProcessor.compute_stft_spectrogram expects 1D wav usually.
        # We need to process multichannel.
        
        # Process Mic (Channel-wise)
        mic_specs = []
        freqs = None
        
        for c in range(mic_wav.shape[0]):
            freqs, times, stft, mag = AudioProcessor.compute_stft_spectrogram(
                mic_wav[c], fs=self.fs, nperseg=self.n_fft, window=self.window
            )
            # stft: (F, T) complex
            mic_specs.append(stft)
            
        mic_stft = np.stack(mic_specs, axis=-1) # (F, T, C)
        
        # Process LDV
        if ldv_wav.ndim == 1:
            ldv_wav = ldv_wav.reshape(1, -1)
            
        freqs, times, ldv_stft, ldv_mag = AudioProcessor.compute_stft_spectrogram(
            ldv_wav[0], fs=self.fs, nperseg=self.n_fft, window=self.window
        )
        # ldv_stft: (F, T)
        
        # Mask Frequencies
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        
        mic_stft_band = mic_stft[mask, :, :] # (F_band, T, C)
        ldv_stft_band = ldv_stft[mask, :]    # (F_band, T)
        
        return {
            "X": torch.from_numpy(mic_stft_band), # Complex
            "Y": torch.from_numpy(ldv_stft_band), # Complex
            "angle": float(angle),
            "freqs": freqs[mask]
        }

def create_dataloader(dataset: DoAPairDataset, batch_size: int = 1, shuffle: bool = True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=None)
