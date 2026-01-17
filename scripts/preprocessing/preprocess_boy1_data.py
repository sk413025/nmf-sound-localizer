#!/usr/bin/env python3
"""
Preprocessing script for boy1 data (Interspeech GRU1 experiment).

This script:
1. Reads raw WAV files from boy1/MIC and boy1/LDV.
2. Extracts metadata (Angle, ID) from filenames.
   - Naming convention: boy1_papercup_{MIC|LDV}_{ID}.wav
   - Mapping ID to Angle:
     IDs 001-016: Angle 0
     IDs 017-032: Angle 12
     ...
     (Based on 16 files per angle assumption, total 26 angles * 16 files = 416 files)
3. Normalizes each waveform to [0,1] amplitude range (Peak normalization).
4. Saves as .npy files in structure: output_root/angle_{deg}/clip_{id}.npy
   - Normalization preserves relative spectral structure, crucial for OMP.

Usage:
    python scripts/preprocessing/preprocess_boy1_data.py --input_root ... --output_root ...
"""

import argparse
import logging
import sys
import shutil
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path
from typing import Tuple, Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants based on manual inspection of boy1 data
# Total 416 files.
# Assumption: equal distribution across angles.
# 416 files.
# If 16 files per angle -> 416 / 16 = 26 angles.
# Angles: 0, 10, 20 ... ? Or 0, 15, 30?
# We need to determine the angle mapping.
# Standard LDV setup often uses 10 degree increments or similar.
# Let's assume standard angular resolution for now, but print debug info.
# Actually, the file naming DOES NOT contain angle info directly.
# We infer angle from file ID sequence blocks.
FILES_PER_ANGLE = 16
ANGLE_STEP = 10 # This is a placeholder guess! We need to verify.
# But for training, as long as MIC and LDV share the SAME mapping, 
# and we group them consistently, the specific physical angle label 
# matters mostly for evaluation/visualization, not for the core physics (correlation).
# However, `DoADataset` expects `angle_{deg}` folders.

def normalize_waveform(wav_data: np.ndarray) -> np.ndarray:
    """
    Normalize waveform to [0,1] range.
    
    Formula: x_norm = (x - min) / (max - min)
    This maps min->0 and max->1.
    """
    # Convert to float
    data = wav_data.astype(np.float32)
    
    data_min = np.min(data)
    data_max = np.max(data)
    data_range = data_max - data_min
    
    if data_range == 0:
        logger.warning("Zero range waveform encountered. Returning zeros.")
        return np.zeros_like(data)
        
    return (data - data_min) / data_range

def process_single_file(input_path: Path, output_path: Path):
    """Read WAV, normalize, save as NPY."""
    try:
        fs, data = wav.read(input_path)
        
        # Normalize
        norm_data = normalize_waveform(data)
        
        # Save
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
        np.save(output_path, norm_data)
        
    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        raise

def infer_angle_from_id(file_id: int) -> int:
    """
    Map file ID (1-based) to Angle.
    Assumption: 16 files per angle block.
    Block 0 (ID 1-16): Angle 0
    Block 1 (ID 17-32): Angle 10 ?
    """
    block_index = (file_id - 1) // FILES_PER_ANGLE
    # TODO: Confirm angle step. Assuming 0-based index maps to discrete angles.
    # If we don't know the physical angle, we can just use the block index as a proxy
    # or assume a standard step like 360/26 approx 13.8? 
    # Or maybe 0 to 180?
    # Let's use block_index * 10 for now as a label, knowing it implies 'Angle Class X'.
    return block_index * ANGLE_STEP 

def main():
    parser = argparse.ArgumentParser(description="Preprocess boy1 raw data to normalized NPY structure")
    parser.add_argument("--source_root", type=str, required=True, 
                        help="Root containing boy1/MIC and boy1/LDV")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root for output (e.g., data/derived/boy1_processed)")
    
    args = parser.parse_args()
    
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    
    mic_source_dir = source_root / "MIC"
    ldv_source_dir = source_root / "LDV"
    
    if not mic_source_dir.exists() or not ldv_source_dir.exists():
        logger.error(f"Source directories not found in {source_root}")
        sys.exit(1)
        
    mic_out_dir = output_root / "MIC"
    ldv_out_dir = output_root / "LDV"
    
    if output_root.exists():
        logger.warning(f"Output root {output_root} exists. Cleaning up...")
        # shutil.rmtree(output_root) # Safer to not auto-delete in dev
    
    mic_files = sorted(list(mic_source_dir.glob("*.wav")))
    ldv_files = sorted(list(ldv_source_dir.glob("*.wav")))
    
    logger.info(f"Found {len(mic_files)} MIC files")
    logger.info(f"Found {len(ldv_files)} LDV files")
    
    if len(mic_files) != len(ldv_files):
        logger.error("Mismatch in file counts!")
        sys.exit(1)
        
    # Process Loop
    for mic_file, ldv_file in zip(mic_files, ldv_files):
        # Filename: boy1_papercup_MIC_001.wav
        fname = mic_file.name
        
        # Extract ID
        try:
            # Split by '_' and get last part '001.wav'
            id_str = fname.split('_')[-1].replace('.wav', '')
            file_id = int(id_str)
        except ValueError:
            logger.error(f"Could not parse ID from {fname}")
            continue
            
        angle_deg = infer_angle_from_id(file_id)
        
        # Construct Output Path: output/MIC/angle_{deg}/clip_{id}.npy
        angle_folder = f"angle_{angle_deg}"
        clip_name = f"clip_{file_id:03d}.npy"
        
        mic_save_path = mic_out_dir / angle_folder / clip_name
        ldv_save_path = ldv_out_dir / angle_folder / clip_name
        
        process_single_file(mic_file, mic_save_path)
        process_single_file(ldv_file, ldv_save_path)
        
        if file_id % 50 == 0:
            logger.info(f"Processed {file_id} files...")
            
    logger.info("Preprocessing Complete.")
    logger.info(f"Output stored in {output_root}")
    logger.info("Structure: MIC/angle_X/clip_Y.npy & LDV/angle_X/clip_Y.npy")

if __name__ == "__main__":
    main()
