"""
Transfer function processing module for NMF localization.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union
import logging

from ..config.defaults import NMFConfig

logger = logging.getLogger(__name__)


class TransferFunctionProcessor:
    """Processor for directional transfer functions."""
    
    def __init__(self, config: NMFConfig):
        self.config = config
        self.epsilon = config.epsilon
    
    def load_transfer_functions(
        self, 
        path: Union[str, Path]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Load transfer functions from file.
        
        Args:
            path: Path to transfer function file (.npy or .pth)
            
        Returns:
            H: Transfer function matrix (F, D)
            angles: Angle array (D,)
            metadata: Additional information
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Transfer function file not found: {path}")
        
        logger.info(f"Loading transfer functions from: {path}")
        
        if path.suffix == '.npy':
            H = torch.from_numpy(np.load(path)).float()
            angles = torch.linspace(0, 360, H.shape[1] + 1)[:-1]  # Uniform distribution
            metadata = {'method': 'loaded_npy', 'source': str(path)}
            
        elif path.suffix == '.pth':
            data = torch.load(path, weights_only=False)
            
            if isinstance(data, dict):
                # Check for different formats
                if 'H_linear' in data:
                    H = data['H_linear']
                    metadata = {
                        'method': data.get('method', 'improved'),
                        'freqs': data.get('freqs'),
                        'source': str(path)
                    }
                elif 'H' in data:
                    H = data['H']
                    metadata = {'method': 'simple', 'source': str(path)}
                else:
                    H = data
                    metadata = {'method': 'unknown', 'source': str(path)}
                    
                # Load angles if available
                if 'angles' in data:
                    angles = data['angles']
                    if isinstance(angles, np.ndarray):
                        angles = torch.from_numpy(angles).float()
                else:
                    angles = torch.linspace(0, 360, H.shape[1] + 1)[:-1]
                    
            else:
                H = data
                angles = torch.linspace(0, 360, H.shape[1] + 1)[:-1]
                metadata = {'method': 'unknown', 'source': str(path)}
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        logger.info(f"Loaded transfer functions: H shape {H.shape}")
        logger.info(f"Angles: {angles.tolist() if len(angles) <= 20 else f'{len(angles)} angles'}")
        
        return H, angles, metadata
    
    def save_transfer_functions(
        self,
        H: torch.Tensor,
        angles: torch.Tensor,
        path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save transfer functions to file.
        
        Args:
            H: Transfer function matrix (F, D)
            angles: Angle array (D,)
            path: Output path
            metadata: Additional information to save
        """
        path = Path(path)
        
        save_dict = {
            'H': H,
            'H_linear': H,  # For compatibility
            'angles': angles
        }
        
        if metadata:
            save_dict.update(metadata)
        
        torch.save(save_dict, path)
        logger.info(f"Saved transfer functions to: {path}")
    
    def apply_frequency_limit(
        self,
        H: torch.Tensor,
        freqs: np.ndarray,
        freq_min: Optional[float] = None,
        freq_max: Optional[float] = None
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Apply frequency band limiting to transfer functions.
        
        Args:
            H: Transfer function matrix (F, D)
            freqs: Frequency array
            freq_min: Minimum frequency (Hz)
            freq_max: Maximum frequency (Hz)
            
        Returns:
            H_limited: Band-limited transfer functions
            freqs_limited: Band-limited frequency array
        """
        freq_min = freq_min or self.config.freq_min
        freq_max = freq_max or self.config.freq_max
        
        if freq_min is None or freq_max is None:
            logger.info("No frequency limits specified, returning original")
            return H, freqs
        
        # Find frequency indices
        idx_min = np.argmin(np.abs(freqs - freq_min))
        idx_max = np.argmin(np.abs(freqs - freq_max))
        
        H_limited = H[idx_min:idx_max+1, :]
        freqs_limited = freqs[idx_min:idx_max+1]
        
        logger.info(f"Applied frequency limit {freq_min}-{freq_max}Hz: "
                   f"{H.shape[0]} → {H_limited.shape[0]} freq bins")
        logger.info(f"Actual frequency range: {freqs_limited[0]:.1f} - {freqs_limited[-1]:.1f} Hz")
        
        return H_limited, freqs_limited
    
    def normalize_transfer_functions(
        self,
        H: torch.Tensor,
        method: str = 'none',
        reference_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Normalize transfer functions.
        
        Args:
            H: Input transfer functions (F, D)
            method: Normalization method ('none', 'global', 'per_freq', 'reference')
            reference_idx: Reference direction index (for 'reference' method)
            
        Returns:
            Normalized transfer functions
        """
        if method == 'none':
            return H
        
        elif method == 'global':
            H_max = H.max()
            H_norm = H / (H_max + self.epsilon)
            logger.info(f"Applied global normalization: max = {H_max:.6f}")
            return H_norm
        
        elif method == 'per_freq':
            H_norm = H.clone()
            for f in range(H.shape[0]):
                max_val = H[f, :].max()
                if max_val > self.epsilon:
                    H_norm[f, :] = H[f, :] / max_val
            logger.info("Applied per-frequency normalization")
            return H_norm
        
        elif method == 'reference':
            if reference_idx is None:
                raise ValueError("Reference index required for reference normalization")
                
            reference_spectrum = H[:, reference_idx:reference_idx+1]
            H_relative = H / (reference_spectrum + self.epsilon)
            
            logger.info(f"Applied reference normalization using direction {reference_idx}")
            return H_relative
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # Contrast enhancement API removed.
    
    def analyze_separability(self, H: torch.Tensor) -> Dict[str, float]:
        """
        Analyze directional separability of transfer functions.
        
        Args:
            H: Transfer function matrix (F, D)
            
        Returns:
            Separability metrics
        """
        # Move to CPU for analysis
        H_cpu = H.cpu()
        
        # Compute correlation matrix between directions
        H_norm = H_cpu / (torch.norm(H_cpu, dim=0, keepdim=True) + self.epsilon)
        corr_matrix = H_norm.T @ H_norm
        
        # Exclude diagonal (self-correlation = 1)
        mask = torch.eye(corr_matrix.shape[0], dtype=bool)
        off_diagonal = corr_matrix[~mask]
        
        mean_correlation = off_diagonal.mean().item()
        max_correlation = off_diagonal.max().item()
        min_correlation = off_diagonal.min().item()
        
        # Condition number
        try:
            _, s, _ = torch.svd(H_cpu)
            condition_number = (s.max() / (s.min() + self.epsilon)).item()
        except:
            condition_number = float('inf')
        
        # Range per frequency
        freq_ranges = torch.max(H_cpu, dim=1)[0] - torch.min(H_cpu, dim=1)[0]
        mean_range = freq_ranges.mean().item()
        
        separability = {
            'mean_correlation': mean_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'condition_number': condition_number,
            'mean_freq_range': mean_range,
            'n_directions': H.shape[1],
            'n_freq': H.shape[0]
        }
        
        logger.info(f"Transfer function separability:")
        logger.info(f"  Mean correlation: {mean_correlation:.3f}")
        logger.info(f"  Condition number: {condition_number:.3f}")
        logger.info(f"  Mean freq range: {mean_range:.3f}")
        
        return separability
    
    def get_direction_index(self, angles: torch.Tensor, target_angle: float) -> int:
        """
        Find the index of the direction closest to the target angle.
        
        Args:
            angles: Angle array (D,)
            target_angle: Target angle in degrees
            
        Returns:
            Index of closest direction
        """
        # Handle angle wraparound
        diff = torch.abs(angles - target_angle)
        wrapped_diff = torch.min(diff, 360 - diff)
        
        return torch.argmin(wrapped_diff).item()
    
    def process_transfer_functions(
        self,
        H: torch.Tensor,
        angles: torch.Tensor,
        freqs: Optional[np.ndarray] = None,
        reference_angle: float = 90.0
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete processing pipeline for transfer functions.
        
        Args:
            H: Raw transfer functions (F, D)
            angles: Angle array (D,)
            freqs: Frequency array (optional)
            reference_angle: Reference angle for normalization
            
        Returns:
            Processed transfer functions
            Processing information
        """
        processing_info = {
            'original_shape': H.shape,
            'steps_applied': []
        }
        
        H_processed = H.clone()
        
        # 1. Apply frequency limiting if frequencies are provided
        if freqs is not None:
            H_processed, freqs_limited = self.apply_frequency_limit(H_processed, freqs)
            processing_info['frequency_limited'] = {
                'original_range': f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz",
                'limited_range': f"{freqs_limited[0]:.1f}-{freqs_limited[-1]:.1f} Hz",
                'n_freq_bins': H_processed.shape[0]
            }
            processing_info['steps_applied'].append('frequency_limiting')
        
        # 2. No normalization or contrast enhancement: preserve absolute scale end-to-end
        processing_info['normalization'] = 'none'
        
        # 4. Analyze final separability
        separability = self.analyze_separability(H_processed)
        processing_info['separability'] = separability
        
        processing_info['final_shape'] = H_processed.shape
        
        logger.info(f"Transfer function processing complete:")
        logger.info(f"  Shape: {H.shape} → {H_processed.shape}")
        logger.info(f"  Steps: {', '.join(processing_info['steps_applied'])}")
        
        return H_processed, processing_info
