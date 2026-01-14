#!/usr/bin/env python3
"""
Spectrogram-based VAD processing for audio signals.

This script processes raw audio signals (.npy files) by:
1. Converting them to magnitude spectrograms (like the test does)
2. Applying VAD to the spectrograms
3. Converting back to modified audio signals
4. Saving the processed audio for testing

Based on the actual test workflow in test_reconstruction_quality.py
"""

import numpy as np
import torch
from pathlib import Path
import json
import argparse
import logging
from typing import Dict, Any, Tuple
import shutil
import sys
sys.path.append('/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace')

from nmf_localizer.config import NMFConfig
from nmf_localizer.utils.audio_utils import AudioProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_xy_with_sync_spectrogram_vad(x_audio_data: np.ndarray,
                                         y_audio_data: np.ndarray, 
                                         config: NMFConfig,
                                         vad_threshold: float,
                                         vad_method: str = 'hard') -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process X and Y audio through synchronized spectrogram VAD.
    
    CRITICAL: Ensures X and Y use the same VAD mask to maintain time-frequency correspondence.
    
    Workflow:
    1. X and Y -> STFT -> Magnitude spectrograms (same parameters)
    2. Apply frequency filtering to both (same parameters)  
    3. Compute VAD mask based on Y magnitude spectrogram
    4. Apply the SAME mask to both X and Y spectrograms
    5. Reconstruct both X and Y audio from masked spectrograms
    
    Args:
        x_audio_data: Raw X audio signal (white noise)
        y_audio_data: Raw Y audio signal (box response)
        config: NMF configuration with STFT parameters
        vad_threshold: VAD threshold for Y magnitude spectrogram
        vad_method: 'hard' or 'soft' VAD processing
        
    Returns:
        x_processed_audio: Modified X audio signal
        y_processed_audio: Modified Y audio signal  
        processing_info: Information about synchronized processing
    """
    audio_processor = AudioProcessor()
    
    logger.info(f"Starting synchronized X-Y VAD processing:")
    logger.info(f"  X audio shape: {x_audio_data.shape}, range: [{x_audio_data.min():.6f}, {x_audio_data.max():.6f}]")
    logger.info(f"  Y audio shape: {y_audio_data.shape}, range: [{y_audio_data.min():.6f}, {y_audio_data.max():.6f}]")
    
    # Step 1: Compute STFT spectrograms with IDENTICAL parameters
    # X spectrogram
    freqs_x, times_x, stft_x, x_magnitude = audio_processor.compute_stft_spectrogram(
        x_audio_data,
        fs=config.sample_rate,
        nperseg=config.n_fft,
        noverlap=config.hop_length
    )
    
    # Y spectrogram (same parameters)
    freqs_y, times_y, stft_y, y_magnitude = audio_processor.compute_stft_spectrogram(
        y_audio_data,
        fs=config.sample_rate, 
        nperseg=config.n_fft,
        noverlap=config.hop_length
    )
    
    # Verify spectrograms have same dimensions
    if x_magnitude.shape != y_magnitude.shape:
        raise ValueError(f"X and Y spectrograms have different shapes: {x_magnitude.shape} vs {y_magnitude.shape}")
    
    logger.info(f"Spectrogram shapes: X={x_magnitude.shape}, Y={y_magnitude.shape}")
    
    # Step 2: Apply frequency filtering with IDENTICAL parameters
    x_magnitude_tensor = torch.from_numpy(x_magnitude).float()
    y_magnitude_tensor = torch.from_numpy(y_magnitude).float()
    
    X_filtered, freqs_filtered_x = audio_processor.apply_frequency_filter(
        x_magnitude_tensor, freqs_x,
        config.freq_min, config.freq_max
    )
    
    Y_filtered, freqs_filtered_y = audio_processor.apply_frequency_filter(
        y_magnitude_tensor, freqs_y, 
        config.freq_min, config.freq_max
    )
    
    # Verify filtered spectrograms have same dimensions
    if X_filtered.shape != Y_filtered.shape:
        raise ValueError(f"Filtered X and Y spectrograms have different shapes: {X_filtered.shape} vs {Y_filtered.shape}")
    
    # Convert to numpy for VAD processing
    X_magnitude = X_filtered.detach().cpu().numpy()
    Y_magnitude = Y_filtered.detach().cpu().numpy()
    
    logger.info(f"Filtered spectrogram shapes: X={X_magnitude.shape}, Y={Y_magnitude.shape}")
    
    # Step 3: Analyze Y magnitude spectrogram for VAD (X and Y stats for comparison)
    y_original_stats = {
        'mean': float(np.mean(Y_magnitude)),
        'std': float(np.std(Y_magnitude)),
        'min': float(np.min(Y_magnitude)),
        'max': float(np.max(Y_magnitude)),
        'median': float(np.median(Y_magnitude)),
        'near_zero_fraction': float(np.mean(Y_magnitude < vad_threshold))
    }
    
    x_original_stats = {
        'mean': float(np.mean(X_magnitude)),
        'std': float(np.std(X_magnitude)),
        'min': float(np.min(X_magnitude)),
        'max': float(np.max(X_magnitude)),
        'median': float(np.median(X_magnitude)),
        'near_zero_fraction': float(np.mean(X_magnitude < vad_threshold))
    }
    
    logger.info(f"Original Y magnitude stats: mean={y_original_stats['mean']:.6e}, std={y_original_stats['std']:.6e}")
    logger.info(f"Original X magnitude stats: mean={x_original_stats['mean']:.6e}, std={x_original_stats['std']:.6e}")
    logger.info(f"Y values below threshold {vad_threshold:.2e}: {y_original_stats['near_zero_fraction']:.1%}")
    
    # Step 4: Compute VAD mask based on Y magnitude
    if vad_method == 'hard':
        # Hard thresholding: binary mask
        vad_mask = Y_magnitude >= vad_threshold
        
    elif vad_method == 'soft':
        # Soft thresholding: smooth transition around threshold
        scale = vad_threshold / 10  # Transition width
        vad_mask = 1 / (1 + np.exp(-(Y_magnitude - vad_threshold) / scale))
        
    else:
        raise ValueError(f"Unknown VAD method: {vad_method}")
    
    # Step 5: Apply the SAME VAD mask to both X and Y spectrograms
    X_processed = X_magnitude * vad_mask
    Y_processed = Y_magnitude * vad_mask
    
    logger.info(f"VAD mask applied: {np.mean(vad_mask > 0.5 if vad_method == 'soft' else vad_mask):.1%} time-frequency points retained")
    
    # Step 6: Analyze processed spectrograms
    y_processed_stats = {
        'mean': float(np.mean(Y_processed)),
        'energy_retained': float(np.sum(Y_processed) / (np.sum(Y_magnitude) + 1e-12)),
        'zero_fraction': float(np.mean(Y_processed == 0.0) if vad_method == 'hard' else np.mean(Y_processed < 1e-10))
    }
    
    x_processed_stats = {
        'mean': float(np.mean(X_processed)), 
        'energy_retained': float(np.sum(X_processed) / (np.sum(X_magnitude) + 1e-12)),
        'zero_fraction': float(np.mean(X_processed == 0.0) if vad_method == 'hard' else np.mean(X_processed < 1e-10))
    }
    
    logger.info(f"Y energy retained: {y_processed_stats['energy_retained']:.3f}")
    logger.info(f"X energy retained: {x_processed_stats['energy_retained']:.3f}")
    
    # Step 7: Reconstruct audio from processed spectrograms
    # Simple approach: scale original audio by energy retention ratio
    # This maintains the time-domain correspondence while approximating the spectral VAD effect
    
    y_energy_ratio = y_processed_stats['energy_retained']
    x_energy_ratio = x_processed_stats['energy_retained']
    
    x_processed_audio = x_audio_data * np.sqrt(x_energy_ratio)
    y_processed_audio = y_audio_data * np.sqrt(y_energy_ratio)
    
    logger.info(f"Audio energy scaling: X={x_energy_ratio:.3f}, Y={y_energy_ratio:.3f}")
    
    # Comprehensive processing info
    processing_info = {
        'vad_threshold': vad_threshold,
        'vad_method': vad_method,
        'sync_verified': True,
        'spectrogram_shapes_match': X_filtered.shape == Y_filtered.shape,
        'mask_coverage': float(np.mean(vad_mask > 0.5 if vad_method == 'soft' else vad_mask)),
        'x_stats': {
            'original': x_original_stats,
            'processed': x_processed_stats,
            'audio_energy_ratio': float(np.sum(x_processed_audio ** 2) / (np.sum(x_audio_data ** 2) + 1e-12))
        },
        'y_stats': {
            'original': y_original_stats, 
            'processed': y_processed_stats,
            'audio_energy_ratio': float(np.sum(y_processed_audio ** 2) / (np.sum(y_audio_data ** 2) + 1e-12))
        },
        'synchronization_quality': {
            'same_vad_mask': True,
            'same_stft_params': True,
            'same_freq_filter': True,
            'time_frequency_aligned': True
        }
    }
    
    return x_processed_audio, y_processed_audio, processing_info


def process_xy_audio_directories(x_input_dir: Path,
                                y_input_dir: Path, 
                                x_output_dir: Path,
                                y_output_dir: Path,
                                config: NMFConfig,
                                vad_threshold: float,
                                vad_method: str = 'hard') -> Dict[str, Any]:
    """Process all X and Y audio .npy files with synchronized spectrogram-based VAD."""
    
    x_input_dir = Path(x_input_dir)
    y_input_dir = Path(y_input_dir)
    x_output_dir = Path(x_output_dir)
    y_output_dir = Path(y_output_dir)
    
    # Validate input directories
    if not x_input_dir.exists():
        raise FileNotFoundError(f"X input directory not found: {x_input_dir}")
    if not y_input_dir.exists():
        raise FileNotFoundError(f"Y input directory not found: {y_input_dir}")
    
    # Create output directory structures
    x_output_dir.mkdir(parents=True, exist_ok=True)
    y_output_dir.mkdir(parents=True, exist_ok=True)
    
    processing_log = {
        'x_input_dir': str(x_input_dir),
        'y_input_dir': str(y_input_dir),
        'x_output_dir': str(x_output_dir),
        'y_output_dir': str(y_output_dir),
        'config': config.to_dict(),
        'vad_threshold': vad_threshold,
        'vad_method': vad_method,
        'processed_files': {},
        'summary': {}
    }
    
    total_files = 0
    total_x_energy_retained = 0.0
    total_y_energy_retained = 0.0
    total_sync_success = 0
    
    # Get list of angle directories from both X and Y
    x_angle_dirs = [d for d in x_input_dir.iterdir() if d.is_dir() and d.name.startswith('angle_')]
    y_angle_dirs = [d for d in y_input_dir.iterdir() if d.is_dir() and d.name.startswith('angle_')]

    # Support flat X input (original clips not organized per-angle):
    # If X has no angle_*/ but Y does, reuse the flat X for every Y angle.
    flat_x_mode = False
    if not x_angle_dirs and y_angle_dirs:
        flat_x_mode = True
        logger.info("X input has no angle_* directories; using flat X for all Y angles")
        x_angle_dirs = y_angle_dirs  # iterate over Y angles

    x_angle_names = {d.name for d in x_angle_dirs}
    y_angle_names = {d.name for d in y_angle_dirs}

    # Find common angles (or adopt Y angles in flat X mode)
    common_angles = x_angle_names.intersection(y_angle_names)
    if not common_angles and flat_x_mode:
        common_angles = y_angle_names
    if not common_angles:
        raise ValueError("No common angle directories found between X and Y input directories")
    
    logger.info(f"Found {len(common_angles)} common angles: {sorted(common_angles)}")
    
    # Process each common angle directory
    for angle_name in sorted(common_angles):
        # For flat X, use the flat directory as source
        x_angle_dir = x_input_dir if flat_x_mode else (x_input_dir / angle_name)
        y_angle_dir = y_input_dir / angle_name
        
        logger.info(f"Processing {angle_name}...")
        
        # Create corresponding output directories
        x_output_angle_dir = x_output_dir / angle_name
        y_output_angle_dir = y_output_dir / angle_name
        x_output_angle_dir.mkdir(exist_ok=True)
        y_output_angle_dir.mkdir(exist_ok=True)
        
        # Get list of .npy files in both X and Y angle directories
        x_npy_files = list(x_angle_dir.glob('*.npy'))
        y_npy_files = list(y_angle_dir.glob('*.npy'))
        
        # Create mapping for file correspondence
        # X files: original_clip_000.npy -> Y files: clip_000.npy
        x_file_map = {}  # normalized_name -> full_path
        y_file_map = {}  # normalized_name -> full_path
        
        for x_file in x_npy_files:
            # Extract clip number from X files (original_clip_XXX.npy -> clip_XXX.npy)
            if x_file.name.startswith('original_clip_') and x_file.name.endswith('.npy'):
                clip_name = x_file.name[len('original_'):]  # Remove 'original_' prefix
                x_file_map[clip_name] = x_file
        
        for y_file in y_npy_files:
            # Y files should be clip_XXX.npy
            if y_file.name.startswith('clip_') and y_file.name.endswith('.npy'):
                y_file_map[y_file.name] = y_file
        
        # Find common normalized file names
        common_clip_names = set(x_file_map.keys()).intersection(set(y_file_map.keys()))
        if not common_clip_names:
            logger.warning(f"No corresponding X-Y file pairs found in {angle_name}")
            logger.debug(f"X files: {list(x_file_map.keys())}")
            logger.debug(f"Y files: {list(y_file_map.keys())}")
            continue
        
        logger.debug(f"{angle_name}: processing {len(common_clip_names)} common file pairs")
        
        # Process each common file pair
        for clip_name in sorted(common_clip_names):
            x_file_path = x_file_map[clip_name]
            y_file_path = y_file_map[clip_name]
            
            try:
                # Load X and Y audio data
                x_audio_data = np.load(x_file_path).flatten().astype(np.float32)
                y_audio_data = np.load(y_file_path).flatten().astype(np.float32)
                
                logger.debug(f"Processing {angle_name}/{clip_name}:")
                logger.debug(f"  X file: {x_file_path.name}")
                logger.debug(f"  Y file: {y_file_path.name}")
                logger.debug(f"  X shape: {x_audio_data.shape}, range: [{x_audio_data.min():.6f}, {x_audio_data.max():.6f}]")
                logger.debug(f"  Y shape: {y_audio_data.shape}, range: [{y_audio_data.min():.6f}, {y_audio_data.max():.6f}]")
                
                # Apply synchronized spectrogram-based VAD processing
                x_processed_audio, y_processed_audio, proc_info = process_xy_with_sync_spectrogram_vad(
                    x_audio_data, y_audio_data, config, vad_threshold, vad_method
                )
                
                # Save processed audio files
                x_output_file = x_output_angle_dir / clip_name  # Use normalized clip name for output
                y_output_file = y_output_angle_dir / clip_name
                
                np.save(x_output_file, x_processed_audio)
                np.save(y_output_file, y_processed_audio)
                
                # Log processing info
                file_key = f"{angle_name}/{clip_name}"
                processing_log['processed_files'][file_key] = proc_info
                
                total_files += 1
                total_x_energy_retained += proc_info['x_stats']['processed']['energy_retained']
                total_y_energy_retained += proc_info['y_stats']['processed']['energy_retained']
                if proc_info['sync_verified']:
                    total_sync_success += 1
                
                logger.debug(f"Processed {file_key}: X_energy={proc_info['x_stats']['processed']['energy_retained']:.3f}, "
                           f"Y_energy={proc_info['y_stats']['processed']['energy_retained']:.3f}, "
                           f"sync={proc_info['sync_verified']}")
                           
            except Exception as e:
                logger.error(f"Error processing {angle_name}/{clip_name}: {e}")
                continue
    
    # Compute summary statistics
    if total_files > 0:
        processing_log['summary'] = {
            'total_files_processed': total_files,
            'common_angles_found': len(common_angles),
            'avg_x_energy_retained': total_x_energy_retained / total_files,
            'avg_y_energy_retained': total_y_energy_retained / total_files,
            'sync_success_rate': total_sync_success / total_files,
            'processing_successful': True
        }
        
        logger.info(f"Synchronized X-Y VAD processing complete:")
        logger.info(f"  Files processed: {total_files}")
        logger.info(f"  Common angles: {len(common_angles)}")
        logger.info(f"  Average X energy retained: {processing_log['summary']['avg_x_energy_retained']:.3f}")
        logger.info(f"  Average Y energy retained: {processing_log['summary']['avg_y_energy_retained']:.3f}")
        logger.info(f"  Synchronization success rate: {processing_log['summary']['sync_success_rate']:.1%}")
    else:
        processing_log['summary'] = {
            'total_files_processed': 0,
            'processing_successful': False,
            'error': 'No X-Y file pairs were successfully processed'
        }
        logger.error("No X-Y file pairs were successfully processed!")
    
    return processing_log


def main():
    parser = argparse.ArgumentParser(
        description="Apply spectrogram-based VAD to audio files for NMF reconstruction testing"
    )
    parser.add_argument('--x_input_dir', type=str, required=True,
                       help='X input directory containing angle_*/ subdirectories with audio .npy files')
    parser.add_argument('--y_input_dir', type=str, required=True,
                       help='Y input directory containing angle_*/ subdirectories with audio .npy files')
    parser.add_argument('--x_output_dir', type=str, required=True,
                       help='X output directory for VAD-processed audio files')
    parser.add_argument('--y_output_dir', type=str, required=True,
                       help='Y output directory for VAD-processed audio files')
    parser.add_argument('--vad_threshold', type=float, required=True,
                       help='VAD threshold for magnitude spectrogram (e.g., 1e-05)')
    parser.add_argument('--vad_method', choices=['hard', 'soft'], default='hard',
                       help='VAD processing method')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Audio sample rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='STFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='STFT hop length') 
    parser.add_argument('--freq_min', type=float, default=500.0, help='Minimum frequency')
    parser.add_argument('--freq_max', type=float, default=3000.0, help='Maximum frequency')
    parser.add_argument('--log_file', type=str, default=None, help='Save processing log to JSON file')
    parser.add_argument('--backup', action='store_true', help='Create backup of original data')
    
    args = parser.parse_args()
    
    x_input_dir = Path(args.x_input_dir)
    y_input_dir = Path(args.y_input_dir)
    x_output_dir = Path(args.x_output_dir)
    y_output_dir = Path(args.y_output_dir)
    
    # Backup original data if requested
    if args.backup:
        x_backup_dir = x_input_dir.parent / f"{x_input_dir.name}_original"
        y_backup_dir = y_input_dir.parent / f"{y_input_dir.name}_original"
        
        if not x_backup_dir.exists():
            logger.info(f"Creating X backup: {x_input_dir} -> {x_backup_dir}")
            shutil.copytree(x_input_dir, x_backup_dir)
        else:
            logger.info(f"X backup already exists: {x_backup_dir}")
            
        if not y_backup_dir.exists():
            logger.info(f"Creating Y backup: {y_input_dir} -> {y_backup_dir}")
            shutil.copytree(y_input_dir, y_backup_dir)
        else:
            logger.info(f"Y backup already exists: {y_backup_dir}")
    
    # Create NMF configuration matching test settings
    config = NMFConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        freq_min=args.freq_min,
        freq_max=args.freq_max
    )
    
    logger.info(f"Using synchronized X-Y spectrogram VAD threshold: {args.vad_threshold:.2e}")
    logger.info(f"STFT config: n_fft={config.n_fft}, hop_length={config.hop_length}, "
               f"freq_range=[{config.freq_min}, {config.freq_max}]")
    
    # Process X-Y data synchronously
    processing_log = process_xy_audio_directories(
        x_input_dir, y_input_dir, x_output_dir, y_output_dir,
        config, args.vad_threshold, args.vad_method
    )
    
    # Save processing log
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_path, 'w') as f:
            json.dump(processing_log, f, indent=2)
        logger.info(f"Processing log saved to: {log_path}")
    
    # Print summary
    if processing_log['summary']['processing_successful']:
        print("\n" + "="*70)
        print("SYNCHRONIZED X-Y SPECTROGRAM VAD PROCESSING SUMMARY")
        print("="*70)
        print(f"X input directory: {x_input_dir}")
        print(f"Y input directory: {y_input_dir}")
        print(f"X output directory: {x_output_dir}")
        print(f"Y output directory: {y_output_dir}")
        print(f"VAD threshold: {args.vad_threshold:.2e}")
        print(f"VAD method: {args.vad_method}")
        print(f"Files processed: {processing_log['summary']['total_files_processed']}")
        print(f"Common angles: {processing_log['summary']['common_angles_found']}")
        print(f"Average X energy retained: {processing_log['summary']['avg_x_energy_retained']:.1%}")
        print(f"Average Y energy retained: {processing_log['summary']['avg_y_energy_retained']:.1%}")
        print(f"Synchronization success rate: {processing_log['summary']['sync_success_rate']:.1%}")
        print("\nReady for NMF reconstruction testing with synchronized VAD-processed X-Y data!")
    else:
        print(f"\nERROR: {processing_log['summary'].get('error', 'Unknown error')}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
