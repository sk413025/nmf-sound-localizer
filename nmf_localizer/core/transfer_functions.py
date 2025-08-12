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
    
    def enhance_contrast(
        self,
        H: torch.Tensor,
        reference_idx: Optional[int] = None,
        enhancement_factor: float = 2.0
    ) -> torch.Tensor:
        """
        Apply per-frequency contrast enhancement.
        
        Args:
            H: Input transfer functions (F, D)
            reference_idx: Reference direction index
            enhancement_factor: Enhancement factor
            
        Returns:
            Enhanced transfer functions
        """
        if not self.config.apply_contrast_enhancement:
            logger.info("Contrast enhancement disabled in config")
            return H

        H_enhanced = H.clone()
        
        for f_idx in range(H_enhanced.shape[0]):
            freq_values = H_enhanced[f_idx, :]
            min_val = freq_values.min()
            max_val = freq_values.max()
            range_val = max_val - min_val
            
            if range_val > 1e-6:  # Only enhance if there's meaningful variation
                if reference_idx is not None:
                    # Enhanced around reference
                    ref_val = H_enhanced[f_idx, reference_idx]
                    centered = freq_values - ref_val
                    enhanced_centered = centered * enhancement_factor
                    H_enhanced[f_idx, :] = enhanced_centered + ref_val
                    # Ensure non-negative
                    H_enhanced[f_idx, :] = torch.clamp(H_enhanced[f_idx, :], min=0.01)
                else:
                    # Global enhancement
                    normalized = (freq_values - min_val) / range_val
                    enhanced = normalized * enhancement_factor
                    H_enhanced[f_idx, :] = enhanced * range_val + min_val
        
        # Log enhancement effect
        original_range = torch.mean(torch.max(H, dim=1)[0] - torch.min(H, dim=1)[0])
        enhanced_range = torch.mean(torch.max(H_enhanced, dim=1)[0] - torch.min(H_enhanced, dim=1)[0])
        
        logger.info(f"Contrast enhancement: mean range {original_range:.4f} → {enhanced_range:.4f} "
                   f"(factor: {enhanced_range/original_range:.2f})")

        return H_enhanced

    def smooth_transfer_functions(
        self,
        H: torch.Tensor,
        kernel_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Smooth transfer functions along frequency axis using moving average.

        Args:
            H: Input transfer functions (F, D)
            kernel_size: Moving average kernel size

        Returns:
            Smoothed transfer functions
        """
        ks = kernel_size or self.config.smoothing_kernel_size
        ks = max(1, int(ks))
        # Ensure odd kernel size for same-length output
        if ks % 2 == 0:
            ks += 1
        if ks <= 1:
            return H

        pad = ks // 2
        # Use 1D conv via unfold for each direction
        H_pad = torch.nn.functional.pad(H, (0, 0, pad, pad), mode='reflect')
        kernel = torch.ones((ks,), dtype=H.dtype, device=H.device) / ks
        # Apply per direction (column)
        out = torch.empty_like(H)
        for d in range(H.shape[1]):
            col = H_pad[:, d]
            conv = torch.nn.functional.conv1d(
                col.view(1, 1, -1), kernel.view(1, 1, -1), padding=0
            ).view(-1)
            out[:, d] = conv
        return out

    def prune_low_variance_bins(
        self,
        H: torch.Tensor,
        threshold: Optional[float] = None,
        min_bins: Optional[int] = None,
        freqs: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
        """
        Remove frequency bins with very low directional variation.

        Args:
            H: Transfer function matrix (F, D)
            threshold: Keep bins with range >= threshold * median(range)
            min_bins: Minimum number of bins to keep
            freqs: Optional frequency array to prune synchronously

        Returns:
            H_pruned, freqs_pruned, stats
        """
        thr = self.config.low_variance_threshold if threshold is None else threshold
        min_keep = self.config.min_freq_bins if min_bins is None else min_bins

        ranges = (H.max(dim=1)[0] - H.min(dim=1)[0]).detach().cpu()
        median_range = torch.median(ranges)
        keep_mask = ranges >= (thr * (median_range + self.epsilon))

        # Ensure at least min_keep bins by keeping top-k ranges if needed
        if keep_mask.sum().item() < min_keep:
            topk = min(min_keep, len(ranges))
            topk_idx = torch.topk(ranges, k=topk).indices
            mask_np = torch.zeros_like(keep_mask, dtype=torch.bool)
            mask_np[topk_idx] = True
            keep_mask = mask_np

        H_pruned = H[keep_mask, :]
        if freqs is not None:
            freqs_pruned = freqs[keep_mask.numpy()]
        else:
            freqs_pruned = np.arange(H_pruned.shape[0])

        stats = {
            'original_bins': int(H.shape[0]),
            'kept_bins': int(H_pruned.shape[0]),
            'median_range': float(median_range.item()),
            'threshold': float(thr),
        }
        return H_pruned, freqs_pruned, stats

    def compute_frequency_weights(
        self,
        H: torch.Tensor,
        method: Optional[str] = None,
        clip_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute per-frequency weights to emphasize informative bins.

        Args:
            H: Transfer function matrix (F, D)
            method: 'range' or 'variance'
            clip_factor: clip weights to mean ± factor*std

        Returns:
            weights (F,)
        """
        m = (method or self.config.freq_weight_method).lower()
        cf = self.config.freq_weight_clip if clip_factor is None else clip_factor

        Hc = H.detach().cpu()
        if m == 'variance':
            w = torch.var(Hc, dim=1)
        else:  # default 'range'
            w = Hc.max(dim=1)[0] - Hc.min(dim=1)[0]

        # Normalize to mean 1
        w = w / (w.mean() + self.epsilon)
        # Clip extreme weights
        mean = w.mean()
        std = w.std(unbiased=False) + self.epsilon
        lo = mean - cf * std
        hi = mean + cf * std
        w = torch.clamp(w, min=lo, max=hi)

        return w.to(H.device)
    
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

        # Baseline separability before processing
        try:
            base_sep = self.analyze_separability(H_processed)
            processing_info['baseline_separability'] = base_sep
        except Exception:
            pass
        
        # 1. Apply frequency limiting if frequencies are provided
        if freqs is not None:
            H_processed, freqs_limited = self.apply_frequency_limit(H_processed, freqs)
            processing_info['frequency_limited'] = {
                'original_range': f"{freqs[0]:.1f}-{freqs[-1]:.1f} Hz",
                'limited_range': f"{freqs_limited[0]:.1f}-{freqs_limited[-1]:.1f} Hz",
                'n_freq_bins': H_processed.shape[0]
            }
            processing_info['steps_applied'].append('frequency_limiting')
        
        # 2. Apply reference normalization if reference angle is available
        try:
            ref_idx = self.get_direction_index(angles, reference_angle)
            H_processed = self.normalize_transfer_functions(
                H_processed, method='reference', reference_idx=ref_idx
            )
            processing_info['reference_normalization'] = {
                'reference_angle': reference_angle,
                'reference_index': ref_idx
            }
            processing_info['steps_applied'].append('reference_normalization')
        except Exception as e:
            logger.warning(f"Reference normalization failed: {e}")
            # Fall back to per-frequency normalization
            H_processed = self.normalize_transfer_functions(H_processed, method='per_freq')
            processing_info['steps_applied'].append('per_freq_normalization')

        # 3. Optional smoothing to reduce narrowband spikes
        if self.config.enable_frequency_smoothing:
            H_processed = self.smooth_transfer_functions(
                H_processed, kernel_size=self.config.smoothing_kernel_size
            )
            processing_info['steps_applied'].append('frequency_smoothing')

        # 4. Optional pruning of low-variance bins
        if self.config.enable_low_variance_pruning:
            H_processed, pruned_freqs, prune_stats = self.prune_low_variance_bins(
                H_processed, threshold=self.config.low_variance_threshold,
                min_bins=self.config.min_freq_bins,
                freqs=freqs if freqs is not None else None
            )
            processing_info['low_variance_pruning'] = prune_stats
            if freqs is not None:
                freqs = pruned_freqs
            processing_info['steps_applied'].append('low_variance_pruning')

        # 5. Apply contrast enhancement (after smoothing/pruning)
        if self.config.apply_contrast_enhancement:
            ref_idx = processing_info.get('reference_normalization', {}).get('reference_index')
            H_processed = self.enhance_contrast(
                H_processed,
                reference_idx=ref_idx,
                enhancement_factor=self.config.contrast_enhancement_factor
            )
            processing_info['steps_applied'].append('contrast_enhancement')

        # 6. Optional frequency weighting
        if self.config.enable_auto_frequency_weights:
            weights = self.compute_frequency_weights(H_processed)
            processing_info['frequency_weights'] = weights
            processing_info['steps_applied'].append('frequency_weighting')

        # 7. Analyze final separability
        separability = self.analyze_separability(H_processed)
        processing_info['separability'] = separability

        processing_info['final_shape'] = H_processed.shape
        
        logger.info(f"Transfer function processing complete:")
        logger.info(f"  Shape: {H.shape} → {H_processed.shape}")
        logger.info(f"  Steps: {', '.join(processing_info['steps_applied'])}")
        
        return H_processed, processing_info
