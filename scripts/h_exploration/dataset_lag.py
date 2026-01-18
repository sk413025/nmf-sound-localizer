import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import scipy.io.wavfile
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
                 freq_max: float = 3000.0,
                 normalize_audio: bool = True):
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
        self.normalize_audio = normalize_audio
        
        # Load Clips for specific angle
        self.clips = []
        
        # Mic path: .../angle_90/clip_X.npy
        angle_str = f"angle_{int(angle)}"
        mic_dir = self.mic_root / angle_str
        
        if not mic_dir.exists():
            # Fallback to Flat WAV search for boy1 structure
            logger.warning(f"Angle dir {mic_dir} not found. Searching for WAV files matching pattern *{int(angle):03d}.wav")
            pattern = f"*{int(angle):03d}.wav"
            wav_files = sorted(list(self.mic_root.glob(pattern)))
            
            if not wav_files:
                logger.error(f"No WAV files found usage {pattern} in {self.mic_root}")
                return
                
            for mic_path in wav_files:
                # boy1_papercup_MIC_090.wav -> replace MIC with LDV for pair
                ldv_name = mic_path.name.replace("MIC", "LDV")
                ldv_path = self.ldv_root / ldv_name
                
                if ldv_path.exists():
                    self.clips.append((mic_path, ldv_path))
            
            logger.info(f"Found {len(self.clips)} WAV pairs for angle {angle}")
        else:
            for mic_path in sorted(mic_dir.glob("*.npy")):
                # Find Pair
                rel_path = mic_path.relative_to(self.mic_root)
                ldv_path = self.ldv_root / rel_path
                
                if ldv_path.exists():
                    self.clips.append((mic_path, ldv_path))
                
        logger.info(f"Total clips avail: {len(self.clips)}")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        mic_path, ldv_path = self.clips[idx]
        
        # Load Raw
        if mic_path.suffix == ".npy":
            mic_wav = np.load(mic_path)
            ldv_wav = np.load(ldv_path)
        else:
            # WAV
            try:
                fs_m, mic_wav = scipy.io.wavfile.read(mic_path)
                fs_l, ldv_wav = scipy.io.wavfile.read(ldv_path)
                
                # Resample if needed? Assuming 16k for now based on papercup data
                # Convert int to float
                if mic_wav.dtype == np.int16:
                    mic_wav = mic_wav.astype(np.float32) / 32768.0
                elif mic_wav.dtype == np.int32:
                     mic_wav = mic_wav.astype(np.float32) / 2147483648.0
                     
                if ldv_wav.dtype == np.int16:
                    ldv_wav = ldv_wav.astype(np.float32) / 32768.0
                elif ldv_wav.dtype == np.int32:
                     ldv_wav = ldv_wav.astype(np.float32) / 2147483648.0
                     
            except Exception as e:
                logger.error(f"Error loading WAV {mic_path}: {e}")
                # return zeros equivalent? This will likely crash downstream if we don't handle
                # But Dataset must return something. 
                # For now let it crash to debug.
                raise e

        # Normalize (Peak Norm)
        mic_max = np.max(np.abs(mic_wav))
        if mic_max > 1e-9:
            mic_wav = mic_wav / mic_max
            
        ldv_max = np.max(np.abs(ldv_wav))
        if ldv_max > 1e-9:
            ldv_wav = ldv_wav / ldv_max
        
        if mic_wav.ndim == 1:
            mic_wav = mic_wav.reshape(1, -1) # (1, Samples)

        # Normalize Audio if Requested (Max-Abs)
        if self.normalize_audio:
            max_mic = np.max(np.abs(mic_wav))
            if max_mic > 1e-9:
                mic_wav = mic_wav / max_mic
                
            max_ldv = np.max(np.abs(ldv_wav))
            if max_ldv > 1e-9:
                ldv_wav = ldv_wav / max_ldv
        
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
