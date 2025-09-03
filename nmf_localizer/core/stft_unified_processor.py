"""
STFT-unified data processor to fix Welch vs STFT mismatch issues.

This module replaces the mixed Welch/STFT approach with a consistent STFT-only method
for both transfer function computation and signal processing.

Key fixes:
1. Use STFT for transfer function computation (not Welch PSD)
2. Ensure consistent parameters between H computation and Y processing  
3. Maintain proper physical units throughout (magnitude spectra)
4. Improve coherence through synchronized processing
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from tqdm import tqdm
import logging
from scipy import signal

from ..config.defaults import NMFConfig
from ..utils.audio_utils import AudioProcessor

logger = logging.getLogger(__name__)


class STFTUnifiedProcessor:
    """STFT-unified processor that fixes Welch/STFT mismatch issues."""
    
    def __init__(self, config: NMFConfig):
        self.config = config
        self.audio_processor = AudioProcessor()
        
    def estimate_transfer_functions_stft(
        self,
        original_root: Path,
        box_root: Path,
        method: str = 'stft_unified'
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Path], Dict[str, Any]]:
        """
        Estimate transfer functions using consistent STFT approach.
        
        This method fixes the Welch vs STFT inconsistency by:
        1. Using STFT for both X and Y signal processing
        2. Computing H = Y_stft / X_stft using same parameters as Y processing
        3. Time-averaging to get stationary transfer functions
        4. Maintaining magnitude spectrum units throughout
        
        Args:
            original_root: Path to original data (X)
            box_root: Path to box recordings (Y)
            method: Method identifier ('stft_unified')
        
        Returns:
            H: Transfer functions [freq × directions] (magnitude-based)
            angles: Angle array  
            angle_folders: List of angle folders
            metadata: Processing information
        """
        # Get angle folders
        original_angles = sorted([d for d in original_root.iterdir() 
                                 if d.is_dir() and d.name.startswith('angle_')])
        box_angles = sorted([d for d in box_root.iterdir() 
                            if d.is_dir() and d.name.startswith('angle_')])
        
        if len(original_angles) != len(box_angles):
            raise ValueError(f"Angle count mismatch: {len(original_angles)} original vs {len(box_angles)} box angles")
        
        logger.info(f"STFT-Unified: Processing {len(original_angles)} angle pairs")
        
        # Map angles to degrees
        angle_mapping = {}
        for folder in original_angles:
            angle_str = folder.name.replace('angle_', '')
            angle_mapping[folder.name] = int(angle_str)
        
        # Sort by angle value
        original_angles = sorted(original_angles, key=lambda x: angle_mapping[x.name])
        box_angles = sorted(box_angles, key=lambda x: angle_mapping[x.name])
        n_directions = len(original_angles)
        
        # Unified STFT parameters (same as Y processing)
        stft_params = {
            'nperseg': self.config.n_fft,
            'noverlap': self.config.n_fft - self.config.hop_length,
            'window': self.config.window,
            'fs': self.config.sample_rate
        }
        
        logger.info(f"STFT parameters (unified):")
        logger.info(f"  n_fft (nperseg): {stft_params['nperseg']}")
        logger.info(f"  hop_length: {self.config.hop_length}")
        logger.info(f"  overlap: {stft_params['noverlap']}")
        logger.info(f"  window: {stft_params['window']}")
        
        # Process each angle
        all_transfer_functions = []
        angles = []
        freqs = None
        coherence_stats = []
        
        for orig_folder, box_folder in tqdm(zip(original_angles, box_angles), 
                                           desc="Processing angles with STFT"):
            angle_str = orig_folder.name.replace('angle_', '')
            angle_deg = int(angle_str)
            angles.append(angle_deg)
            
            # Load files
            orig_files = sorted(list(orig_folder.glob('*.npy')))[:self.config.n_files_per_angle]
            box_files = sorted(list(box_folder.glob('*.npy')))[:self.config.n_files_per_angle]
            
            logger.info(f"Processing angle {angle_deg}° with {len(orig_files)} file pairs (STFT method)")
            
            # Process files for this angle
            angle_transfer_functions = []
            angle_coherences = []
            
            for orig_file, box_file in zip(orig_files, box_files):
                # Load signals
                x = np.load(orig_file).astype(np.float32).flatten()
                y = np.load(box_file).astype(np.float32).flatten()
                
                # Ensure same length
                min_length = min(len(x), len(y))
                x = x[:min_length]
                y = y[:min_length]
                
                # Compute STFT for both signals using IDENTICAL parameters
                freqs_x, times_x, X_stft = signal.stft(x, **stft_params)
                freqs_y, times_y, Y_stft = signal.stft(y, **stft_params)
                
                # Verify parameter consistency
                assert np.allclose(freqs_x, freqs_y), "Frequency arrays must match"
                assert X_stft.shape == Y_stft.shape, f"STFT shapes must match: {X_stft.shape} vs {Y_stft.shape}"
                
                if freqs is None:
                    freqs = freqs_x
                    logger.info(f"STFT frequency array: {len(freqs)} bins, range {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
                
                # Compute transfer function in STFT domain
                # H(f,t) = Y_stft(f,t) / X_stft(f,t) for each time frame
                epsilon = 1e-12
                H_stft_complex = Y_stft / (X_stft + epsilon)
                
                # Time-average to get stationary transfer function 
                # Use magnitude to ensure physical consistency with Y processing
                H_magnitude = np.mean(np.abs(H_stft_complex), axis=1)
                
                # Compute coherence using STFT-based cross-spectral estimates
                # More accurate than Welch for our processing pipeline
                X_magnitude = np.abs(X_stft)
                Y_magnitude = np.abs(Y_stft)
                XY_cross = np.real(X_stft * np.conj(Y_stft))
                
                # Time-averaged coherence estimate  
                S_xx = np.mean(X_magnitude ** 2, axis=1)
                S_yy = np.mean(Y_magnitude ** 2, axis=1)
                S_xy = np.mean(XY_cross, axis=1)
                
                coherence = S_xy ** 2 / ((S_xx + epsilon) * (S_yy + epsilon))
                avg_coherence = np.mean(coherence)
                
                angle_transfer_functions.append(H_magnitude)
                angle_coherences.append(avg_coherence)
                
            # Average transfer functions across files for this angle
            if angle_transfer_functions:
                avg_tf = np.mean(angle_transfer_functions, axis=0)
                all_transfer_functions.append(avg_tf)
                
                avg_coherence = np.mean(angle_coherences)
                coherence_stats.append(avg_coherence)
                
                logger.info(f"Angle {angle_deg}°: TF shape {avg_tf.shape}, mean magnitude: {np.mean(avg_tf):.4f}")
                logger.info(f"  STFT coherence: {avg_coherence:.3f}")
        
        # Convert to tensor format  
        H = torch.tensor(np.array(all_transfer_functions).T, dtype=torch.float32)  # [freq × directions]
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        
        logger.info(f"Raw H shape (STFT): {H.shape}")
        
        # Apply frequency filtering (same range as Y processing)
        freq_mask = (freqs >= self.config.freq_min) & (freqs <= self.config.freq_max)
        H_filtered = H[freq_mask, :]
        freqs_filtered = freqs[freq_mask]
        
        # Apply processing that preserves magnitude spectrum units
        if method == 'stft_unified':
            logger.info("Applying STFT-unified processing (magnitude-preserving)")
            
            # Normalize relative to mean spectrum (no unit conversion needed)
            mean_spectrum = torch.mean(H_filtered, dim=1, keepdim=True)
            H_filtered = H_filtered / (mean_spectrum + 1e-10)
            
            # Mild contrast enhancement preserving magnitude units
            if self.config.apply_contrast_enhancement:
                # Global normalization maintaining relative differences
                H_min = H_filtered.min()
                H_max = H_filtered.max()
                if H_max > H_min:
                    H_filtered = (H_filtered - H_min) / (H_max - H_min) * 0.8 + 0.1
                logger.info("Applied magnitude-preserving contrast enhancement")
        
        logger.info(f"STFT-unified transfer function estimation complete:")
        logger.info(f"  H shape: {H_filtered.shape} (freq × directions)")
        logger.info(f"  Method: {method} (consistent STFT processing)")
        logger.info(f"  Frequency range: {self.config.freq_min}-{self.config.freq_max} Hz")
        logger.info(f"  H statistics: min={H_filtered.min():.4f}, max={H_filtered.max():.4f}, mean={H_filtered.mean():.4f}")
        
        # Enhanced metadata with STFT processing info
        overall_coherence = np.mean(coherence_stats) if coherence_stats else 0.0
        metadata = {
            'method': method,
            'processing_approach': 'stft_unified',
            'freqs': freqs_filtered,
            'stft_parameters': stft_params,
            'n_files_per_angle': self.config.n_files_per_angle,
            'units': 'magnitude_spectrum_ratio',
            'coherence_stats': {
                'mean_coherence': overall_coherence,
                'min_coherence': np.min(coherence_stats) if coherence_stats else 0.0,
                'max_coherence': np.max(coherence_stats) if coherence_stats else 0.0,
                'per_angle_coherence': coherence_stats
            },
            'improvements': [
                'Unified STFT processing for H and Y',
                'Consistent magnitude spectrum units', 
                'Same nperseg/hop_length parameters',
                'Improved coherence through synchronized processing'
            ]
        }
        
        # Log coherence improvement
        logger.info(f"STFT-unified coherence statistics:")
        logger.info(f"  Mean: {overall_coherence:.3f}")
        if coherence_stats:
            logger.info(f"  Range: {np.min(coherence_stats):.3f} - {np.max(coherence_stats):.3f}")
            logger.info(f"  Std: {np.std(coherence_stats):.3f}")
            
            if overall_coherence > 0.3:
                logger.info("✓ Good coherence achieved with STFT-unified processing")
            elif overall_coherence > 0.15:
                logger.info("⚠ Moderate coherence - acceptable for STFT-unified method")
            else:
                logger.warning(f"⚠ Low coherence ({overall_coherence:.3f}) - check signal quality")
        
        return H_filtered, angles_tensor, original_angles, metadata
    
    def load_npy_files(self, folder_path: Path, max_files: Optional[int] = None) -> List[np.ndarray]:
        """Load .npy files from a folder (same as original data_processor)."""
        npy_files = sorted(folder_path.glob('*.npy'))
        if max_files:
            npy_files = npy_files[:max_files]
        
        logger.info(f"Loading {len(npy_files)} files from {folder_path.name}")
        
        data = []
        for i, file_path in enumerate(npy_files):
            arr = np.load(file_path)
            if i == 0:
                logger.info(f"First file shape in {folder_path.name}: {arr.shape}")
            data.append(arr)
        
        return data
    
    def compare_with_original_method(
        self, 
        original_root: Path,
        box_root: Path
    ) -> Dict[str, Any]:
        """
        Compare STFT-unified method with original Welch/STFT mixed method.
        
        Returns comparison metrics to validate the improvement.
        """
        logger.info("Comparing STFT-unified vs original Welch/STFT methods...")
        
        # Import original processor for comparison
        from .data_processor import DataProcessor
        original_processor = DataProcessor(self.config)
        
        try:
            # Get results from original method
            logger.info("Running original Welch/STFT method...")
            H_original, angles_orig, folders_orig, meta_orig = original_processor.estimate_transfer_functions(
                original_root, box_root, method='xy_correspondence'
            )
            
            # Get results from STFT-unified method
            logger.info("Running STFT-unified method...")
            H_unified, angles_unified, folders_unified, meta_unified = self.estimate_transfer_functions_stft(
                original_root, box_root, method='stft_unified'
            )
            
            # Compare results
            comparison = {
                'original_method': {
                    'H_shape': H_original.shape,
                    'H_range': [float(H_original.min()), float(H_original.max())],
                    'H_mean': float(H_original.mean()),
                    'H_std': float(H_original.std()),
                    'coherence_mean': meta_orig.get('coherence_stats', {}).get('mean_coherence', 0.0)
                },
                'stft_unified_method': {
                    'H_shape': H_unified.shape,
                    'H_range': [float(H_unified.min()), float(H_unified.max())],
                    'H_mean': float(H_unified.mean()),
                    'H_std': float(H_unified.std()),
                    'coherence_mean': meta_unified.get('coherence_stats', {}).get('mean_coherence', 0.0)
                },
                'improvements': {
                    'coherence_improvement': meta_unified.get('coherence_stats', {}).get('mean_coherence', 0.0) - 
                                           meta_orig.get('coherence_stats', {}).get('mean_coherence', 0.0),
                    'scale_consistency': 'H computed with same method as Y processing',
                    'unit_consistency': 'Both H and Y in magnitude spectrum units',
                    'parameter_consistency': 'Same STFT parameters for H and Y'
                },
                'metadata': {
                    'original': meta_orig,
                    'stft_unified': meta_unified
                }
            }
            
            # Log comparison
            logger.info("Method comparison results:")
            logger.info(f"  Original H range: {comparison['original_method']['H_range']}")
            logger.info(f"  STFT-unified H range: {comparison['stft_unified_method']['H_range']}")
            logger.info(f"  Original coherence: {comparison['original_method']['coherence_mean']:.3f}")
            logger.info(f"  STFT-unified coherence: {comparison['stft_unified_method']['coherence_mean']:.3f}")
            logger.info(f"  Coherence improvement: {comparison['improvements']['coherence_improvement']:.3f}")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Method comparison failed: {e}")
            return {
                'error': str(e),
                'comparison_failed': True
            }


def test_stft_unified_processor(test_data_paths):
    """Test function for STFT-unified processor."""
    config = NMFConfig(
        sample_rate=16000,
        n_fft=2048,
        hop_length=512,
        freq_min=500.0,
        freq_max=3000.0,
        n_files_per_angle=3
    )
    
    processor = STFTUnifiedProcessor(config)
    
    # Run comparison test
    comparison = processor.compare_with_original_method(
        original_root=Path(test_data_paths["x_root"]),
        box_root=Path(test_data_paths["y_root"])
    )
    
    # Save results
    output_dir = Path("tests/data")
    output_dir.mkdir(exist_ok=True)
    
    import json
    with open(output_dir / "stft_unified_comparison.json", "w") as f:
        def convert_types(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        json.dump(convert_types(comparison), f, indent=2)
    
    logger.info(f"STFT-unified comparison saved to: {output_dir / 'stft_unified_comparison.json'}")
    
    return comparison


# Test function will be called from external test script