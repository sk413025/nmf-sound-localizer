#!/usr/bin/env python3
"""
Basic NMF localization experiment example.

This script demonstrates how to run a simple NMF localization experiment
using the modular pipeline.
"""

import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from nmf_localizer import NMFLocalizationPipeline, NMFConfig, STFTUnifiedProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Run basic NMF localization experiment."""
    
    # Configuration
    config = NMFConfig(
        # Audio processing
        sample_rate=16000,
        freq_min=500.0,
        freq_max=1500.0,
        
        # NMF parameters
        beta=0.0,              # Itakura-Saito divergence
        lambda_group=20.0,     # Group sparsity weight
        gamma_sparse=1.0,      # L1 sparsity weight
        max_iter=100,
        
        # USM parameters
        n_atoms_per_speaker=50,
        
        # Evaluation
        tolerance_degrees=10.0,
        n_test_examples=100,   # Reduced for quick testing
        
        # Hardware - adjust as needed
        device='cpu'  # Use 'cuda' or 'mps' if available
    )
    
    # Create pipeline
    pipeline = NMFLocalizationPipeline(config)
    
    # Note: Pipeline now uses STFT-unified transfer function estimation by default
    # This fixes the previous Welch vs STFT scale/units mismatch issue
    # Alternative: directly use STFTUnifiedProcessor for custom processing
    # processor = STFTUnifiedProcessor(config)
    # H, angles, folders, metadata = processor.estimate_transfer_functions_stft(...)
    
    # Run experiment
    print("Starting basic NMF localization experiment...")
    print(f"Configuration: beta={config.beta}, lambda_group={config.lambda_group}")
    print("Using STFT-unified processing (fixes Welch/STFT mismatch)")
    
    try:
        # Use test data paths from conftest.py for demonstration
        test_data_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge"
        
        # Check if test data exists, otherwise fall back to 'root'
        from pathlib import Path as PathLib
        if PathLib(test_data_root).exists():
            print(f"Using test data from: {test_data_root}")
            data_root = test_data_root
        else:
            print("Test data not found, using 'root' directory")
            data_root = "root"
        
        results = pipeline.run_full_experiment(
            data_root=data_root,
            output_dir="outputs/basic_experiment",
            n_sources=1,
            test_multiple_betas=False,
            save_models=True
        )
        
        # Extract key results
        accuracy = results['stages']['evaluation']['results']['accuracy']
        mean_error = results['stages']['evaluation']['results']['mean_error']
        processing_time = results['stages']['evaluation']['results']['mean_processing_time']
        total_duration = results['total_duration']
        
        # Print results
        print("\n" + "="*50)
        print("EXPERIMENT RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.1f}%")
        print(f"Mean Error: {mean_error:.1f}°")
        print(f"Processing Time: {processing_time*1000:.1f}ms per sample")
        print(f"Total Duration: {total_duration:.1f}s")
        print(f"Results saved to: outputs/basic_experiment/")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the 'root' directory exists with angle data.")
        print("Expected structure:")
        print("root/")
        print("├─ angle_00/")
        print("│   ├─ clip_000.npy")
        print("│   └─ ...")
        print("├─ angle_18/")
        print("└─ ...")
        return 1
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())