"""
Default configuration and data structures for NMF localization.
"""

from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Any
import torch
from pathlib import Path


@dataclass
class NMFConfig:
    """Configuration for NMF localization experiments."""
    
    # Audio processing parameters
    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    win_length: int = 2048
    window: str = 'hann'
    
    # Frequency range settings
    freq_min: float = 500.0
    freq_max: float = 1500.0
    
    # NMF parameters
    beta: float = 0.0  # 0: IS divergence, 1: KL, 2: Euclidean
    lambda_group: float = 5.0   # Group sparsity weight (reduced for stability)
    gamma_sparse: float = 0.1   # Sparsity weight (reduced for stability)
    max_iter: int = 100
    tolerance: float = 1e-4
    epsilon: float = 1e-8
    # IS-divergence safety parameters
    transfer_epsilon: float = 1e-5      # Minimum value for transfer functions (A matrix regularization)
    reconstruction_epsilon: float = 1e-5 # Minimum value for Y_hat reconstruction (safe X updates)
    gradient_clip_max: float = 1e3      # Maximum gradient ratio for numerical stability
    
    # USM parameters
    n_atoms_per_speaker: int = 50
    usm_sparsity_weight: float = 0.1
    usm_max_iter: int = 200
    
    # Transfer function estimation (uses STFT-unified approach only)
    n_files_per_angle: int = 50
    
    
    # Data preparation
    n_speakers: int = 10
    n_files_per_speaker: int = 20
    use_90deg_only: bool = True  # For USM training consistency
    
    # Separate dataset support
    speech_data_root: Optional[str] = None  # Path to speech dataset for USM training and testing
    
    # Evaluation parameters
    tolerance_degrees: float = 10.0
    n_test_examples: int = 1000
    
    # Hardware
    device: str = 'cpu'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'NMFConfig':
        """Create config from dictionary, ignoring unknown/deprecated keys."""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in (config_dict or {}).items() if k in valid_keys}
        return cls(**filtered)


class DataPack:
    """Container for processed NMF localization data."""
    
    def __init__(self):
        # Transfer function data
        self.transfer_functions: Optional[torch.Tensor] = None  # (F, D)
        self.angles: Optional[torch.Tensor] = None  # (D,)
        self.angle_names: Optional[List[str]] = None
        self.freqs: Optional[torch.Tensor] = None
        
        # Speaker data for USM
        self.speaker_data: List[torch.Tensor] = []  # List of (F, T) tensors
        
        # Test data
        self.test_data: List[Dict] = []  # List with 'mixture' and 'directions' keys
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.config: Optional[NMFConfig] = None
        
    def validate(self) -> bool:
        """Validate data consistency."""
        try:
            if self.transfer_functions is not None:
                if self.angles is None:
                    return False
                if self.transfer_functions.shape[1] != len(self.angles):
                    return False
                    
            if self.speaker_data:
                if self.transfer_functions is not None:
                    expected_freq = self.transfer_functions.shape[0]
                    for i, speaker in enumerate(self.speaker_data):
                        if speaker.shape[0] != expected_freq:
                            return False
                            
            return True
            
        except Exception:
            return False
    
    def save(self, path: str):
        """Save data pack to file."""
        save_dict = {
            'transfer_functions': self.transfer_functions,
            'angles': self.angles,
            'angle_names': self.angle_names,
            'freqs': self.freqs,
            'speaker_data': self.speaker_data,
            'test_data': self.test_data,
            'metadata': self.metadata,
            'config': self.config.to_dict() if self.config else None
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: str) -> 'DataPack':
        """Load data pack from file."""
        data = torch.load(path, weights_only=False)
        
        pack = cls()
        pack.transfer_functions = data.get('transfer_functions')
        pack.angles = data.get('angles')
        pack.angle_names = data.get('angle_names')
        pack.freqs = data.get('freqs')
        pack.speaker_data = data.get('speaker_data', [])
        pack.test_data = data.get('test_data', [])
        pack.metadata = data.get('metadata', {})
        
        config_dict = data.get('config')
        if config_dict:
            pack.config = NMFConfig.from_dict(config_dict)
            
        return pack
    
    def __repr__(self) -> str:
        """String representation of data pack."""
        tf_shape = self.transfer_functions.shape if self.transfer_functions is not None else None
        n_angles = len(self.angles) if self.angles is not None else 0
        n_speakers = len(self.speaker_data)
        n_test = len(self.test_data)
        
        return (f"DataPack(tf_shape={tf_shape}, n_angles={n_angles}, "
                f"n_speakers={n_speakers}, n_test={n_test})")
