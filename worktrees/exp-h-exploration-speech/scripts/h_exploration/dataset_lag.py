import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from nmf_localizer.utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)

class DoALagDataset(Dataset):
    """
    Dataset for Lag-based H-Matrix Exploration.
    Returns:
    - Context Matrix X: (M+1, F) - History of Mic frames.
    - Target y: (F) - Current LDV frame.
    
    We process the STFT as a continuous time-series and slice it into windows.
    """

    def __init__(self, 
                 mic_root: str, 
                 ldv_root: str, 
                 angle: float = 90.0,
                 max_lag: int = 16,
                 fs: int = 16000,
                 n_fft: int = 2048,
                 hop_length: Optional[int] = None, 
                 window: str = "hann",
                 freq_min: float = 300.0, 
                 freq_max: float = 3000.0):
        self.mic_root = Path(mic_root)
        self.ldv_root = Path(ldv_root)
        self.angle = angle
        self.max_lag = max_lag
        self.fs = fs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.freq_min = freq_min
        self.freq_max = freq_max
        
        # Load Clips for specific angle
        self.clips = []
        
        # Mic path: .../angle_90/clip_X.npy
        angle_str = f"angle_{int(angle)}"
        mic_dir = self.mic_root / angle_str
        
        if not mic_dir.exists():
            logger.error(f"Angle dir {mic_dir} not found")
            return
            
        for mic_path in sorted(mic_dir.glob("*.npy")):
            # Find Pair
            rel_path = mic_path.relative_to(self.mic_root)
            ldv_path = self.ldv_root / rel_path
            
            if ldv_path.exists():
                self.clips.append((mic_path, ldv_path))
                
        logger.info(f"Found {len(self.clips)} clips for angle {angle}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mic_path, ldv_path = self.clips[idx]
        
        # Load Raw
        mic_wav = np.load(mic_path)
        ldv_wav = np.load(ldv_path)
        
        if mic_wav.ndim == 1:
            mic_wav = mic_wav.reshape(1, -1) # (1, Samples)
        
        if self.hop_length is not None:
             noverlap = self.n_fft - self.hop_length
        else:
             noverlap = None

        # STFT
        _, _, stft, _ = AudioProcessor.compute_stft_spectrogram(
            mic_wav[0], fs=self.fs, nperseg=self.n_fft, noverlap=noverlap, window=self.window
        )
        mic_stft = stft.T # (T, F) Complex
        
        if ldv_wav.ndim == 1:
            _, _, stft_y, _ = AudioProcessor.compute_stft_spectrogram(
                ldv_wav, fs=self.fs, nperseg=self.n_fft, noverlap=noverlap, window=self.window
            )
        else:
            _, _, stft_y, _ = AudioProcessor.compute_stft_spectrogram(
                ldv_wav[0], fs=self.fs, nperseg=self.n_fft, noverlap=noverlap, window=self.window
            )
        ldv_stft = stft_y.T # (T, F) Complex
        
        # Truncate to same length
        min_len = min(mic_stft.shape[0], ldv_stft.shape[0])
        mic_stft = mic_stft[:min_len]
        ldv_stft = ldv_stft[:min_len]
        
        return {
            "mic_stft": torch.from_numpy(mic_stft), # Complex
            "ldv_stft": torch.from_numpy(ldv_stft),
            "clip_idx": idx
        }

def create_dataloader(dataset: DoALagDataset, batch_size: int = 1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
