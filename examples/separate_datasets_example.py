#!/usr/bin/env python3
"""
Example usage of separate datasets feature for NMF sound localization.

This example demonstrates how to:
1. Estimate transfer functions from white noise data
2. Use speech data for localization testing
3. Achieve proper train/test separation
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmf_localizer.config.defaults import NMFConfig
from nmf_localizer.pipeline.full_pipeline import NMFLocalizationPipeline


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Run separate datasets example."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("SEPARATE DATASETS EXAMPLE")
    logger.info("=" * 60)
    
    # Example paths (modify these to match your actual data paths)
    noise_data_path = "/path/to/noise_dataset"  # Contains angle_00/, angle_05/, etc. with noise
    speech_data_path = "/path/to/speech_dataset"  # Contains angle_00/, angle_05/, etc. with speech
    output_dir = "./results/separate_datasets_experiment"
    
    logger.info("This example demonstrates two workflows:")
    logger.info("")
    logger.info("WORKFLOW 1: Traditional single dataset (baseline)")
    logger.info("- Uses same dataset for TF estimation and localization testing")
    logger.info("- May suffer from data leakage")
    logger.info("")
    logger.info("WORKFLOW 2: Separate datasets (recommended)")
    logger.info("- Uses noise data for TF estimation")
    logger.info("- Uses speech data for localization testing")
    logger.info("- Proper train/test separation")
    logger.info("")
    
    # Create configuration optimized for 5-degree intervals
    config = NMFConfig(
        # Transfer function estimation
        tf_method='improved',
        n_files_per_angle=50,
        freq_min=500.0,
        freq_max=1500.0,
        apply_contrast_enhancement=True,
        
        # USM parameters
        n_speakers=10,
        n_files_per_speaker=20,
        use_90deg_only=True,
        
        # Evaluation
        tolerance_degrees=5.0,  # Tighter tolerance for 5-degree intervals
        n_test_examples=500,
        
        # Hardware
        device='mps'  # Use Apple Silicon GPU if available
    )
    
    # Initialize pipeline
    pipeline = NMFLocalizationPipeline(config)
    
    # === WORKFLOW 1: Traditional single dataset ===
    logger.info("=" * 40)
    logger.info("WORKFLOW 1: Single dataset (baseline)")
    logger.info("=" * 40)
    
    try:
        if Path(speech_data_path).exists():
            logger.info(f"Running single dataset experiment with: {speech_data_path}")
            
            single_results = pipeline.run_full_experiment(
                data_root=speech_data_path,
                output_dir=f"{output_dir}/single_dataset",
                n_sources=1,
                save_models=True
            )
            
            logger.info(f"Single dataset accuracy: {single_results['stages']['evaluation']['results']['accuracy']:.1f}%")
        else:
            logger.warning(f"Speech data path not found: {speech_data_path}")
            logger.info("Skipping single dataset workflow...")
    
    except Exception as e:
        logger.error(f"Single dataset workflow failed: {e}")
    
    # === WORKFLOW 2: Separate datasets ===
    logger.info("=" * 40)
    logger.info("WORKFLOW 2: Separate datasets")
    logger.info("=" * 40)
    
    try:
        if Path(noise_data_path).exists() and Path(speech_data_path).exists():
            logger.info("Step 1: Estimating transfer functions from noise data...")
            
            # Step 1: Pre-compute transfer functions from noise
            tf_output_path = f"{output_dir}/transfer_functions/tf_from_noise.pth"
            Path(tf_output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use the standalone script approach
            logger.info(f"Run this command to estimate transfer functions:")
            logger.info(f"python scripts/estimate_transfer_functions.py {noise_data_path} --output {tf_output_path}")
            
            # For now, we'll simulate by running the pipeline with speech data
            # In practice, you would first run the TF estimation script
            logger.info("Step 2: Running localization experiment with speech data...")
            
            separate_results = pipeline.run_full_experiment(
                data_root=speech_data_path,  # Will be ignored when tf_path is provided
                output_dir=f"{output_dir}/separate_datasets",
                tf_path=None,  # Set this to tf_output_path after running TF estimation
                speech_data_root=speech_data_path,
                n_sources=1,
                save_models=True
            )
            
            logger.info(f"Separate datasets accuracy: {separate_results['stages']['evaluation']['results']['accuracy']:.1f}%")
            
        else:
            logger.warning("Noise and/or speech data paths not found")
            logger.info("Please update the paths in this example script")
            
    except Exception as e:
        logger.error(f"Separate datasets workflow failed: {e}")
    
    # === Summary ===
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("To use the separate datasets feature:")
    logger.info("")
    logger.info("1. Estimate transfer functions from noise data:")
    logger.info("   python scripts/estimate_transfer_functions.py /path/to/noise_data --output tf_noise.pth")
    logger.info("")
    logger.info("2. Run localization with speech data:")
    logger.info("   pipeline.run_full_experiment(")
    logger.info("       data_root='dummy',  # Not used when tf_path provided")
    logger.info("       tf_path='tf_noise.pth',")
    logger.info("       speech_data_root='/path/to/speech_data',")
    logger.info("       output_dir='results'")
    logger.info("   )")
    logger.info("")
    logger.info("This ensures proper separation between training (noise) and testing (speech) data.")
    

if __name__ == "__main__":
    main()