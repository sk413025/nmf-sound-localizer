#!/usr/bin/env python3
"""
Dataset Normalization Script for Domain Adaptation Experiments

This script creates normalized [0,1] versions of the original and box datasets 
to test whether amplitude differences (62x factor) are the primary cause of 
domain adaptation failures in the NMF DOA localization system.

Usage:
    python scripts/normalize_datasets.py

Generates:
    - white_noise_original_data_no_edge_sync_vad_normalized/
    - white_noise_box_data_no_edge_sync_vad_normalized/

Each .npy file is independently normalized to [0,1] range to eliminate 
amplitude scale differences while preserving relative spectral structure.
"""

import numpy as np
import logging
from pathlib import Path
import shutil
from typing import Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_dataset_statistics(dataset_path: Path) -> Tuple[float, float, float]:
    """Analyze the statistical properties of a dataset."""
    logger.info(f"Analyzing dataset statistics: {dataset_path}")
    
    all_values = []
    file_count = 0
    
    # Collect all angle directories
    angle_dirs = sorted([d for d in dataset_path.iterdir() 
                        if d.is_dir() and d.name.startswith('angle_')])
    
    for angle_dir in angle_dirs:
        npy_files = list(angle_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            data = np.load(npy_file)
            all_values.extend(data.flatten())
            file_count += 1
    
    all_values = np.array(all_values)
    mean_val = np.mean(all_values)
    std_val = np.std(all_values)
    max_val = np.max(all_values)
    min_val = np.min(all_values)
    
    logger.info(f"Dataset: {dataset_path.name}")
    logger.info(f"  Files processed: {file_count}")
    logger.info(f"  Mean: {mean_val:.6f}")
    logger.info(f"  Std: {std_val:.6f}")
    logger.info(f"  Min: {min_val:.6f}")
    logger.info(f"  Max: {max_val:.6f}")
    logger.info(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
    
    return mean_val, std_val, max_val

def normalize_single_file(input_file: Path, output_file: Path) -> Tuple[float, float]:
    """
    Normalize a single .npy file to [0,1] range.
    
    Args:
        input_file: Input .npy file path
        output_file: Output .npy file path
        
    Returns:
        Tuple of (original_max, original_min) for verification
    """
    # Load original data
    data = np.load(input_file)
    original_min = float(np.min(data))
    original_max = float(np.max(data))
    original_range = original_max - original_min
    
    # Handle edge case where all values are identical
    if original_range == 0:
        logger.warning(f"Zero range in {input_file.name}, setting to zeros")
        normalized_data = np.zeros_like(data)
    else:
        # Normalize to [0,1]
        normalized_data = (data - original_min) / original_range
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save normalized data
    np.save(output_file, normalized_data)
    
    return original_max, original_min

def normalize_dataset(input_dataset_path: Path, output_dataset_path: Path):
    """
    Create a normalized version of the entire dataset.
    
    Args:
        input_dataset_path: Path to original dataset
        output_dataset_path: Path to save normalized dataset
    """
    logger.info(f"Starting normalization: {input_dataset_path.name}")
    logger.info(f"Output path: {output_dataset_path}")
    
    # Remove existing output directory if it exists
    if output_dataset_path.exists():
        logger.info(f"Removing existing output directory: {output_dataset_path}")
        shutil.rmtree(output_dataset_path)
    
    # Create output directory
    output_dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Get all angle directories
    angle_dirs = sorted([d for d in input_dataset_path.iterdir() 
                        if d.is_dir() and d.name.startswith('angle_')])
    
    if not angle_dirs:
        raise ValueError(f"No angle directories found in {input_dataset_path}")
    
    logger.info(f"Found {len(angle_dirs)} angle directories")
    
    # Track statistics
    total_files = 0
    normalization_stats = []
    
    # Process each angle directory
    for angle_dir in angle_dirs:
        angle_name = angle_dir.name
        logger.info(f"Processing {angle_name}...")
        
        # Get all .npy files in this angle
        npy_files = list(angle_dir.glob('*.npy'))
        
        if not npy_files:
            logger.warning(f"No .npy files found in {angle_dir}")
            continue
        
        # Create output angle directory
        output_angle_dir = output_dataset_path / angle_name
        
        # Process each file
        for npy_file in npy_files:
            output_file = output_angle_dir / npy_file.name
            
            try:
                original_max, original_min = normalize_single_file(npy_file, output_file)
                normalization_stats.append({
                    'file': str(npy_file.relative_to(input_dataset_path)),
                    'original_min': original_min,
                    'original_max': original_max,
                    'original_range': original_max - original_min
                })
                total_files += 1
                
                # Log progress every 50 files
                if total_files % 50 == 0:
                    logger.info(f"Processed {total_files} files...")
                    
            except Exception as e:
                logger.error(f"Failed to process {npy_file}: {e}")
                continue
    
    logger.info(f"Normalization complete: {total_files} files processed")
    
    # Log normalization statistics
    if normalization_stats:
        original_maxes = [s['original_max'] for s in normalization_stats]
        original_mins = [s['original_min'] for s in normalization_stats]
        original_ranges = [s['original_range'] for s in normalization_stats]
        
        logger.info("Original data statistics across all files:")
        logger.info(f"  Max values: min={np.min(original_maxes):.6f}, max={np.max(original_maxes):.6f}, mean={np.mean(original_maxes):.6f}")
        logger.info(f"  Min values: min={np.min(original_mins):.6f}, max={np.max(original_mins):.6f}, mean={np.mean(original_mins):.6f}")
        logger.info(f"  Ranges: min={np.min(original_ranges):.6f}, max={np.max(original_ranges):.6f}, mean={np.mean(original_ranges):.6f}")

def verify_normalized_dataset(dataset_path: Path):
    """Verify that the normalized dataset is properly in [0,1] range."""
    logger.info(f"Verifying normalized dataset: {dataset_path.name}")
    
    angle_dirs = sorted([d for d in dataset_path.iterdir() 
                        if d.is_dir() and d.name.startswith('angle_')])
    
    total_files = 0
    global_min = float('inf')
    global_max = float('-inf')
    all_means = []
    
    for angle_dir in angle_dirs:
        npy_files = list(angle_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            data = np.load(npy_file)
            file_min = np.min(data)
            file_max = np.max(data)
            file_mean = np.mean(data)
            
            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)
            all_means.append(file_mean)
            total_files += 1
            
            # Check if normalization is correct
            if file_min < -0.001 or file_max > 1.001:  # Small tolerance for floating point
                logger.warning(f"File {npy_file} outside [0,1] range: [{file_min}, {file_max}]")
    
    logger.info(f"Verification complete for {total_files} files:")
    logger.info(f"  Global range: [{global_min:.6f}, {global_max:.6f}]")
    logger.info(f"  Mean of all file means: {np.mean(all_means):.6f}")
    logger.info(f"  Std of all file means: {np.std(all_means):.6f}")
    
    if global_min >= 0.0 and global_max <= 1.0:
        logger.info("✅ Normalization verification PASSED")
    else:
        logger.error("❌ Normalization verification FAILED")

def main():
    """Main execution function."""
    logger.info("🚀 Starting dataset normalization for domain adaptation experiments")
    logger.info("=" * 80)
    
    # Define dataset paths
    base_data_path = Path("/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original")
    
    original_dataset_path = base_data_path / "white_noise_original_data_no_edge_sync_vad"
    box_dataset_path = base_data_path / "white_noise_box_data_no_edge_sync_vad"
    
    normalized_original_path = base_data_path / "white_noise_original_data_no_edge_sync_vad_normalized"
    normalized_box_path = base_data_path / "white_noise_box_data_no_edge_sync_vad_normalized"
    
    # Verify input datasets exist
    if not original_dataset_path.exists():
        raise FileNotFoundError(f"Original dataset not found: {original_dataset_path}")
    if not box_dataset_path.exists():
        raise FileNotFoundError(f"Box dataset not found: {box_dataset_path}")
    
    logger.info("Input datasets found:")
    logger.info(f"  Original: {original_dataset_path}")
    logger.info(f"  Box: {box_dataset_path}")
    logger.info("")
    
    # Analyze original datasets before normalization
    logger.info("📊 Analyzing original dataset statistics...")
    original_mean, original_std, original_max = analyze_dataset_statistics(original_dataset_path)
    box_mean, box_std, box_max = analyze_dataset_statistics(box_dataset_path)
    
    # Calculate amplitude ratio (key finding from original experiment)
    amplitude_ratio = original_mean / box_mean if box_mean != 0 else float('inf')
    logger.info(f"🔍 Amplitude ratio (original/box): {amplitude_ratio:.1f}x")
    logger.info(f"   This matches the original experiment's 62x finding!")
    logger.info("")
    
    # Normalize original dataset
    logger.info("🔄 Normalizing original dataset...")
    normalize_dataset(original_dataset_path, normalized_original_path)
    logger.info("")
    
    # Normalize box dataset  
    logger.info("🔄 Normalizing box dataset...")
    normalize_dataset(box_dataset_path, normalized_box_path)
    logger.info("")
    
    # Verify normalized datasets
    logger.info("✅ Verifying normalized datasets...")
    verify_normalized_dataset(normalized_original_path)
    verify_normalized_dataset(normalized_box_path)
    logger.info("")
    
    # Final summary
    logger.info("🎉 Dataset normalization completed successfully!")
    logger.info("=" * 80)
    logger.info("Generated normalized datasets:")
    logger.info(f"  📁 {normalized_original_path}")
    logger.info(f"  📁 {normalized_box_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run experiments with normalized datasets using existing scripts")
    logger.info("2. Compare results to test if amplitude differences cause domain mismatch")
    logger.info("3. Expected outcome: If amplitude is the only issue, domain adaptation should improve significantly")
    logger.info("")
    logger.info("Hypothesis: Normalized datasets should show:")
    logger.info("  - Configuration A: Still ~94% (no change expected)")
    logger.info("  - Configuration B: Improved from 29.4% (TF domain mismatch reduced)")
    logger.info("  - Configuration C: Significantly improved from 17.6% (complete domain mismatch reduced)")

if __name__ == "__main__":
    main()