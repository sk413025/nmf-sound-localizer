#!/usr/bin/env python3
"""
Standalone script to estimate transfer functions from noise dataset.

This script processes white noise data to compute directional transfer functions
that can be later used for speech localization experiments, enabling proper
separation between training and testing datasets.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nmf_localizer.config.defaults import NMFConfig
from nmf_localizer.core.data_processor import DataProcessor
from nmf_localizer.core.transfer_functions import TransferFunctionProcessor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description='Estimate transfer functions from noise dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - estimate from noise dataset (geometric time pooling)
  python scripts/estimate_transfer_functions.py /path/to/noise_data --output tf_noise.pth \
    --freq-min 500 --freq-max 1500 --files-per-angle 100 --time-pooling geometric
        """
    )
    
    parser.add_argument(
        'noise_data_root', 
        type=str,
        help='Root directory containing noise data with angle_*/ subdirectories'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for saved transfer functions (.pth file)'
    )
    
    parser.add_argument(
        '--time-pooling',
        type=str,
        choices=['linear', 'geometric'],
        default='geometric',
        help='Time pooling mode for TF estimation (default: geometric)'
    )
    
    parser.add_argument(
        '--files-per-angle',
        type=int,
        default=50,
        help='Number of files to use per angle (default: 50)'
    )
    
    parser.add_argument(
        '--freq-min',
        type=float,
        default=500.0,
        help='Minimum frequency (Hz) for analysis (default: 500.0)'
    )
    
    parser.add_argument(
        '--freq-max',
        type=float,
        default=1500.0,
        help='Maximum frequency (Hz) for analysis (default: 1500.0)'
    )
    
    # Normalization and contrast enhancement have been removed from the project.
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    noise_data_root = Path(args.noise_data_root)
    if not noise_data_root.exists():
        logger.error(f"Noise data directory does not exist: {noise_data_root}")
        sys.exit(1)
        
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("TRANSFER FUNCTION ESTIMATION FROM NOISE DATA")
    logger.info("=" * 60)
    logger.info(f"Input directory: {noise_data_root}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Frequency range: {args.freq_min}-{args.freq_max} Hz")
    logger.info(f"Files per angle: {args.files_per_angle}")
    logger.info(f"Time pooling: {args.time_pooling}")
    
    try:
        # Create configuration
        config = NMFConfig(
            n_files_per_angle=args.files_per_angle,
            freq_min=args.freq_min,
            freq_max=args.freq_max
        )
        
        # Initialize processors
        data_processor = DataProcessor(config)
        tf_processor = TransferFunctionProcessor(config)
        
        logger.info("Starting transfer function estimation...")
        
        # Estimate transfer functions
        H, angles, angle_folders, metadata = data_processor.estimate_transfer_functions(
            noise_data_root,
            time_pooling=args.time_pooling
        )
        
        logger.info(f"Raw transfer functions estimated: {H.shape}")
        logger.info(f"Angles: {angles.tolist()}")
        
        # Apply processing pipeline
        if metadata.get('freqs') is not None:
            logger.info("Applying transfer function processing pipeline...")
            H_processed, processing_info = tf_processor.process_transfer_functions(
                H, 
                angles, 
                freqs=metadata['freqs']
            )
            
            # Update metadata
            metadata.update(processing_info)
        else:
            H_processed = H
            logger.info("No frequency information available, using raw transfer functions")
        
        # Save transfer functions
        logger.info(f"Saving transfer functions to: {output_path}")
        tf_processor.save_transfer_functions(
            H_processed,
            angles,
            output_path,
            metadata={
                **metadata,
                'noise_data_root': str(noise_data_root),
                'estimation_params': {
                    'time_pooling': args.time_pooling,
                    'files_per_angle': args.files_per_angle,
                    'freq_min': args.freq_min,
                    'freq_max': args.freq_max
                }
            }
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("TRANSFER FUNCTION ESTIMATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final shape: {H_processed.shape}")
        logger.info(f"Number of directions: {len(angles)}")
        logger.info(f"Angle range: {angles.min():.1f}° - {angles.max():.1f}°")
        if 'separability' in metadata:
            sep = metadata['separability']
            logger.info(f"Mean correlation: {sep['mean_correlation']:.3f}")
            logger.info(f"Condition number: {sep['condition_number']:.2f}")
        
        logger.info(f"Transfer functions saved to: {output_path}")
        logger.info("\nYou can now use these transfer functions in experiments:")
        logger.info(f"  pipeline.run_full_experiment(tf_path='{output_path}', speech_data_root='...', ...)")
        
    except Exception as e:
        logger.error(f"Transfer function estimation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
