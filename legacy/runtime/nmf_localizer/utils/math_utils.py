"""
Mathematical utilities for NMF localization.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MathUtils:
    """Mathematical utility functions."""
    
    @staticmethod
    def compute_angular_difference(angle1: float, angle2: float) -> float:
        """
        Compute angular difference with wraparound.
        
        Args:
            angle1: First angle in degrees
            angle2: Second angle in degrees
            
        Returns:
            Minimum angular difference (0-180 degrees)
        """
        diff = abs(angle1 - angle2)
        return min(diff, 360 - diff)
    
    @staticmethod
    def normalize_angles(angles: torch.Tensor) -> torch.Tensor:
        """
        Normalize angles to [0, 360) range.
        
        Args:
            angles: Input angles in degrees
            
        Returns:
            Normalized angles
        """
        return angles % 360
    
    @staticmethod
    def compute_correlation_matrix(
        data: torch.Tensor, 
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute correlation matrix between columns.
        
        Args:
            data: Input data matrix (features x samples)
            normalize: Whether to normalize columns first
            
        Returns:
            Correlation matrix (samples x samples)
        """
        if normalize:
            # Normalize columns to unit length
            data_norm = data / (torch.norm(data, dim=0, keepdim=True) + 1e-8)
        else:
            data_norm = data
        
        # Compute correlation
        corr_matrix = data_norm.T @ data_norm
        
        return corr_matrix
    
    @staticmethod
    def compute_condition_number(matrix: torch.Tensor) -> float:
        """
        Compute condition number of a matrix.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Condition number
        """
        try:
            U, s, V = torch.svd(matrix)
            condition_number = (s.max() / (s.min() + 1e-10)).item()
        except:
            condition_number = float('inf')
        
        return condition_number
    
    @staticmethod
    def safe_log(x: torch.Tensor, epsilon: float = 1e-10) -> torch.Tensor:
        """
        Compute logarithm with numerical stability.
        
        Args:
            x: Input tensor
            epsilon: Small value to add for stability
            
        Returns:
            Logarithm of x
        """
        return torch.log(torch.clamp(x, min=epsilon))
    
    @staticmethod 
    def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Apply soft thresholding (shrinkage) operator.
        
        Args:
            x: Input tensor
            threshold: Threshold value
            
        Returns:
            Soft-thresholded tensor
        """
        return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)
    
    @staticmethod
    def project_simplex(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Project tensor onto probability simplex.
        
        Args:
            x: Input tensor
            dim: Dimension along which to project
            
        Returns:
            Projected tensor (sums to 1 along dim)
        """
        # Sort in descending order
        x_sorted, _ = torch.sort(x, dim=dim, descending=True)
        
        # Compute cumulative sums
        cumsum = torch.cumsum(x_sorted, dim=dim)
        
        # Find threshold
        n = x.shape[dim]
        k = torch.arange(1, n + 1, device=x.device, dtype=x.dtype).view(
            [1 if i != dim else n for i in range(x.ndim)]
        )
        
        threshold = (cumsum - 1) / k
        
        # Find largest k such that x_sorted[k] > threshold[k]
        mask = x_sorted > threshold
        rho = mask.sum(dim=dim, keepdim=True) - 1
        
        # Select threshold based on rho
        tau = threshold.gather(dim, rho.long())
        
        # Project
        return torch.clamp(x - tau, min=0)
    
    @staticmethod
    def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
        """
        Compute signal-to-noise ratio in dB.
        
        Args:
            signal: Signal tensor
            noise: Noise tensor
            
        Returns:
            SNR in dB
        """
        signal_power = torch.mean(signal ** 2)
        noise_power = torch.mean(noise ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * torch.log10(signal_power / noise_power)
        return snr.item()
    
    @staticmethod
    def apply_window_function(
        data: torch.Tensor, 
        window_type: str = 'hann',
        dim: int = -1
    ) -> torch.Tensor:
        """
        Apply window function along specified dimension.
        
        Args:
            data: Input data
            window_type: Type of window ('hann', 'hamming', 'blackman')
            dim: Dimension along which to apply window
            
        Returns:
            Windowed data
        """
        length = data.shape[dim]
        
        if window_type == 'hann':
            window = torch.hann_window(length, device=data.device, dtype=data.dtype)
        elif window_type == 'hamming':
            window = torch.hamming_window(length, device=data.device, dtype=data.dtype)
        elif window_type == 'blackman':
            window = torch.blackman_window(length, device=data.device, dtype=data.dtype)
        else:
            raise ValueError(f"Unknown window type: {window_type}")
        
        # Reshape window for broadcasting
        window_shape = [1] * data.ndim
        window_shape[dim] = length
        window = window.view(window_shape)
        
        return data * window