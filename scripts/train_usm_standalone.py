#!/usr/bin/env python3
"""
Standalone USM (Universal Source Model) Training Script

Trains USM dictionary W for NMF-based acoustic localization.
This script creates the W matrix needed for the Y ≈ diag(H) @ W @ X factorization.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from scipy import signal

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmf_localizer.config.defaults import NMFConfig
from nmf_localizer.core.usm_trainer import USMTrainer


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_training_data(
    data_root: Path,
    config: NMFConfig,
    logger: logging.Logger
) -> List[torch.Tensor]:
    """
    Load training data from angle directories.
    
    Args:
        data_root: Root directory with angle_*/ subdirectories
        config: NMF configuration
        logger: Logger instance
        
    Returns:
        List of tensors (magnitude spectrograms) for USM training
    """
    logger.info(f"Loading training data from: {data_root}")
    
    # Get all angle directories
    angle_dirs = sorted(data_root.glob("angle_*"))
    logger.info(f"Found {len(angle_dirs)} angle directories")
    
    if not angle_dirs:
        raise ValueError(f"No angle directories found in {data_root}")
    
    # Load data from each angle directory
    all_spectrograms = []
    total_files = 0
    
    for angle_dir in angle_dirs:
        angle_name = angle_dir.name
        npy_files = sorted(angle_dir.glob("*.npy"))
        
        if not npy_files:
            logger.warning(f"{angle_name}: No NPY files found, skipping")
            continue
        
        logger.debug(f"{angle_name}: Loading {len(npy_files)} files")
        
        for npy_file in npy_files:
            # Load audio data
            audio_data = np.load(npy_file).astype(np.float32)
            
            # Convert to spectrogram using STFT
            # This matches the parameters used in H matrix estimation
            
            f, t, Zxx = signal.stft(
                audio_data,
                fs=config.sample_rate,
                window=config.window,
                nperseg=config.n_fft,
                noverlap=config.n_fft - config.hop_length,
                return_onesided=True
            )
            
            # Get magnitude spectrogram
            magnitude_spec = np.abs(Zxx).astype(np.float32)
            
            # Apply frequency filtering
            freq_mask = (f >= config.freq_min) & (f <= config.freq_max)
            magnitude_spec = magnitude_spec[freq_mask, :]
            
            # Convert to tensor
            spec_tensor = torch.from_numpy(magnitude_spec)
            
            all_spectrograms.append(spec_tensor)
            total_files += 1
    
    logger.info(f"Loaded {total_files} spectrograms from {len(angle_dirs)} angles")
    if all_spectrograms:
        logger.info(f"Spectrogram shape example: {all_spectrograms[0].shape}")
    
    return all_spectrograms


def train_usm_model(
    data_root: Path,
    output_path: Path,
    config: NMFConfig,
    logger: logging.Logger
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Train USM model and save to file.
    
    Args:
        data_root: Root directory with training data
        output_path: Path to save trained USM model
        config: NMF configuration
        logger: Logger instance
        
    Returns:
        Tuple of (W matrix, training metadata)
    """
    logger.info("="*80)
    logger.info("USM TRAINING")
    logger.info("="*80)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Beta divergence: {config.beta}")
    logger.info(f"Max iterations: {config.usm_max_iter}")
    logger.info(f"Atoms per speaker: {config.n_atoms_per_speaker}")
    logger.info(f"Frequency range: {config.freq_min}-{config.freq_max} Hz")
    
    # Load training data
    start_time = time.time()
    spectrograms = load_training_data(data_root, config, logger)
    loading_time = time.time() - start_time
    logger.info(f"Data loading complete in {loading_time:.1f}s")
    
    # Initialize USM trainer
    usm_trainer = USMTrainer(config)
    
    # Move data to device
    logger.info(f"Moving {len(spectrograms)} spectrograms to {config.device}...")
    spectrograms_device = [spec.to(config.device) for spec in spectrograms]
    
    # Train USM
    logger.info("Starting USM training...")
    train_start = time.time()
    W, train_info = usm_trainer.train_usm(spectrograms_device)
    training_time = time.time() - train_start
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"Training time: {training_time:.1f}s")
    logger.info(f"W shape: {W.shape}")
    logger.info(f"W statistics: min={W.min():.6f}, max={W.max():.6f}, mean={W.mean():.6f}")
    
    # Add metadata
    metadata = {
        'training_time': training_time,
        'loading_time': loading_time,
        'total_files': len(spectrograms),
        'data_root': str(data_root),
        'config': {
            'beta': config.beta,
            'usm_max_iter': config.usm_max_iter,
            'n_atoms_per_speaker': config.n_atoms_per_speaker,
            'freq_min': config.freq_min,
            'freq_max': config.freq_max,
            'sample_rate': config.sample_rate,
            'n_fft': config.n_fft,
            'hop_length': config.hop_length,
        },
        **train_info
    }
    
    # Save model
    logger.info(f"Saving USM to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'W': W,
        'metadata': metadata,
        'config': config.__dict__,
    }, output_path)
    
    logger.info(f"✓ USM saved successfully ({output_path.stat().st_size / 1024:.1f} KB)")
    
    return W, metadata


def main() -> int:
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Train Universal Source Model (USM) for NMF-based localization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument('data_root', type=str, help='Root directory with angle_*/ subdirectories')
    parser.add_argument('--output', '-o', required=True, type=str, help='Output path for trained USM (.pth)')
    
    # NMF parameters
    parser.add_argument('--beta', type=float, default=0.0, help='Beta divergence (0: IS, 1: KL, 2: L2)')
    parser.add_argument('--max-iter', type=int, default=200, help='Max iterations for USM training')
    parser.add_argument('--n-atoms', type=int, default=50, help='Number of atoms per speaker for USM')
    
    # Audio/STFT parameters
    parser.add_argument('--sample-rate', type=int, default=16000, help='Audio sample rate (Hz)')
    parser.add_argument('--n-fft', type=int, default=2048, help='FFT size')
    parser.add_argument('--hop-length', type=int, default=512, help='Hop length for STFT')
    parser.add_argument('--freq-min', type=float, default=300.0, help='Minimum frequency (Hz)')
    parser.add_argument('--freq-max', type=float, default=3000.0, help='Maximum frequency (Hz)')
    
    # Device and logging
    parser.add_argument('--device', type=str, default='cpu', help="Device: 'cpu', 'cuda', or 'mps'")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate paths
    data_root = Path(args.data_root)
    output_path = Path(args.output)
    
    if not data_root.exists():
        logger.error(f"Data root not found: {data_root}")
        return 1
    
    # Create configuration
    config = NMFConfig(
        beta=args.beta,
        usm_max_iter=args.max_iter,
        n_atoms_per_speaker=args.n_atoms,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        device=args.device,
    )
    
    try:
        # Train USM
        W, metadata = train_usm_model(data_root, output_path, config, logger)
        
        logger.info("\n" + "="*80)
        logger.info("SUCCESS")
        logger.info("="*80)
        logger.info(f"USM dictionary: {W.shape}")
        logger.info(f"Output file: {output_path}")
        logger.info(f"Training time: {metadata['training_time']:.1f}s")
        logger.info(f"Files processed: {metadata['total_files']}")
        
        return 0
        
    except Exception as e:
        logger.exception(f"USM training failed: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
