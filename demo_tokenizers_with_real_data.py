#!/usr/bin/env python3
"""
Demonstration script for extended tokenizers with real data.

This script shows how to use NMFAtomTokenizer and DirectionProjectionTokenizer
with actual audio data and pre-trained matrices (W and H).

Usage:
    python demo_tokenizers_with_real_data.py
"""

import sys
import torch
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from doa_rl.features.tokenizers_extended import (
    NMFAtomTokenizer,
    DirectionProjectionTokenizer,
)
from doa_rl.features.tokenizers import PatchTokenizer


def load_W_matrix(w_path: str) -> np.ndarray:
    """Load NMF dictionary W from usm.pth file."""
    logger.info(f"Loading W matrix from {w_path}")
    data = torch.load(w_path, map_location="cpu")
    
    if isinstance(data, dict):
        # Check for common keys
        if "W" in data:
            W = data["W"]
        elif "model_state_dict" in data:
            W = data["model_state_dict"]["W"]
        else:
            # Try to find W in the dictionary
            W = next((v for k, v in data.items() if "W" in k.upper()), None)
            if W is None:
                raise ValueError(f"Could not find W matrix in {list(data.keys())}")
    else:
        W = data
    
    if isinstance(W, torch.Tensor):
        W = W.numpy()
    
    logger.info(f"Loaded W matrix with shape {W.shape}")
    return W


def load_H_matrix(h_path: str) -> tuple[np.ndarray, list[int]]:
    """Load transfer function H from h_matrix.pth file."""
    logger.info(f"Loading H matrix from {h_path}")
    data = torch.load(h_path, map_location="cpu")
    
    if isinstance(data, dict):
        # Try to extract H and angles
        if "H" in data:
            H = data["H"]
        elif "transfer_matrix" in data:
            H = data["transfer_matrix"]
        else:
            # Assume the main tensor is H
            H = next((v for k, v in data.items() if isinstance(v, torch.Tensor)), None)
            if H is None:
                raise ValueError(f"Could not find H matrix in {list(data.keys())}")
        
        # Try to get angles
        if "angles" in data:
            angles = data["angles"]
            if isinstance(angles, torch.Tensor):
                angles = angles.tolist()
        else:
            # Default to 0-180 degrees in 5-degree steps
            angles = list(range(0, 181, 5))
    else:
        H = data
        angles = list(range(0, 181, 5))
    
    if isinstance(H, torch.Tensor):
        H = H.numpy()
    
    # Ensure H is (F, D) format
    if H.shape[0] < H.shape[1]:
        H = H.T
        logger.info(f"Transposed H from {(H.shape[1], H.shape[0])} to {H.shape}")
    
    logger.info(f"Loaded H matrix with shape {H.shape}, {len(angles)} angles")
    return H, angles


def load_audio_sample(data_root: str, angle: int = 90) -> np.ndarray:
    """Load a sample audio file from the dataset."""
    angle_dir = Path(data_root) / f"angle_{angle:03d}"
    
    if not angle_dir.exists():
        raise FileNotFoundError(f"Directory not found: {angle_dir}")
    
    # Find first .npy file
    npy_files = list(angle_dir.glob("clip_*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {angle_dir}")
    
    audio_path = npy_files[0]
    logger.info(f"Loading audio from {audio_path}")
    
    audio = np.load(audio_path)
    logger.info(f"Loaded audio with shape {audio.shape}")
    
    return audio


def compute_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute STFT spectrogram from audio."""
    from scipy import signal
    
    # STFT parameters (matching the codebase)
    nperseg = 512
    noverlap = 256
    
    f, t, Zxx = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap)
    Y = np.abs(Zxx)
    
    logger.info(f"Computed spectrogram with shape {Y.shape}")
    return Y


def main():
    """Main demonstration."""
    print("\n" + "="*80)
    print("EXTENDED TOKENIZERS DEMONSTRATION WITH REAL DATA")
    print("="*80 + "\n")
    
    # Paths (adjust these to your actual paths)
    w_path = "doa_normalized_config_c_corrected/models/usm.pth"
    h_path = "h_matrix_normalized_original_to_box.pth"
    data_root = "doa_normalized_config_c_corrected"
    
    # Check if files exist
    if not Path(w_path).exists():
        logger.warning(f"W matrix file not found: {w_path}")
        logger.info("Using synthetic W matrix for demonstration")
        W = np.random.rand(116, 64)
    else:
        try:
            W = load_W_matrix(w_path)
        except Exception as e:
            logger.warning(f"Error loading W matrix: {e}")
            logger.info("Using synthetic W matrix")
            W = np.random.rand(116, 64)
    
    if not Path(h_path).exists():
        logger.warning(f"H matrix file not found: {h_path}")
        logger.info("Using synthetic H matrix for demonstration")
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
    else:
        try:
            H, angles = load_H_matrix(h_path)
        except Exception as e:
            logger.warning(f"Error loading H matrix: {e}")
            logger.info("Using synthetic H matrix")
            H = np.random.rand(116, 37)
            angles = list(range(0, 181, 5))
    
    # Determine expected frequency dimension from W matrix
    expected_F = W.shape[0]
    logger.info(f"Expected frequency dimension from W: {expected_F}")
    
    # Load audio sample
    if not Path(data_root).exists():
        logger.warning(f"Data directory not found: {data_root}")
        logger.info("Using synthetic audio for demonstration")
        Y = np.random.rand(expected_F, 189)
    else:
        try:
            audio = load_audio_sample(data_root, angle=90)
            Y = compute_spectrogram(audio)
            # Adjust Y to match W frequency dimension if needed
            if Y.shape[0] != expected_F:
                logger.warning(f"Spectrogram F dimension {Y.shape[0]} doesn't match W {expected_F}")
                logger.info(f"Creating synthetic spectrogram with F={expected_F}")
                Y = np.random.rand(expected_F, 189)
        except Exception as e:
            logger.warning(f"Error loading audio: {e}")
            logger.info("Using synthetic spectrogram")
            Y = np.random.rand(expected_F, 189)
    
    # Ensure H matches W frequency dimension
    if H.shape[0] != expected_F:
        logger.warning(f"H frequency dimension {H.shape[0]} doesn't match W {expected_F}")
        logger.info(f"Creating synthetic H matrix with F={expected_F}")
        H = np.random.rand(expected_F, len(angles))
    
    print("\n" + "-"*80)
    print("1. CREATING TOKENIZERS")
    print("-"*80)
    
    # Create tokenizers
    patch_tok = PatchTokenizer(Fp=16, Np=10, n_levels=16)
    atom_tok = NMFAtomTokenizer(W, top_k=8, n_levels=16)
    dir_tok = DirectionProjectionTokenizer(H, angles, top_m=5, n_levels=16)
    
    print(f"✓ PatchTokenizer: Fp=16, Np=10, n_levels=16")
    print(f"✓ NMFAtomTokenizer: K={W.shape[1]}, top_k=8, n_levels=16")
    print(f"✓ DirectionProjectionTokenizer: D={H.shape[1]}, top_m=5, n_levels=16")
    
    print("\n" + "-"*80)
    print("2. GENERATING TOKENS")
    print("-"*80)
    
    # Generate tokens
    patch_tokens = patch_tok(Y)
    atom_tokens = atom_tok(Y)
    dir_tokens = dir_tok(Y)
    
    print(f"\nPatch tokens ({len(patch_tokens)} total):")
    print(f"  {' '.join(patch_tokens[:5])} ...")
    
    print(f"\nAtom tokens ({len(atom_tokens)} total):")
    print(f"  {' '.join(atom_tokens)}")
    
    print(f"\nDirection tokens ({len(dir_tokens)} total):")
    print(f"  {' '.join(dir_tokens)}")
    
    print("\n" + "-"*80)
    print("3. BUILDING MULTI-MODAL PROMPT")
    print("-"*80)
    
    # Build multi-modal prompt (physics-first ordering)
    all_tokens = dir_tokens + atom_tokens + patch_tokens
    prompt = " ".join(all_tokens)
    
    print(f"\nMulti-modal prompt ({len(all_tokens)} tokens):")
    print(f"  Structure: [Direction×{len(dir_tokens)}] + [Atom×{len(atom_tokens)}] + [Patch×{len(patch_tokens)}]")
    print(f"\nFirst 100 characters:")
    print(f"  {prompt[:100]}...")
    print(f"\nFull prompt length: {len(prompt)} characters")
    
    print("\n" + "-"*80)
    print("4. PROMPT COMPOSITION ANALYSIS")
    print("-"*80)
    
    # Analyze token distribution
    total_tokens = len(all_tokens)
    dir_ratio = len(dir_tokens) / total_tokens * 100
    atom_ratio = len(atom_tokens) / total_tokens * 100
    patch_ratio = len(patch_tokens) / total_tokens * 100
    
    print(f"\nToken distribution:")
    print(f"  Direction:  {len(dir_tokens):3d} tokens ({dir_ratio:5.1f}%) - Physical prior")
    print(f"  Atom:       {len(atom_tokens):3d} tokens ({atom_ratio:5.1f}%) - Spectral structure")
    print(f"  Patch:      {len(patch_tokens):3d} tokens ({patch_ratio:5.1f}%) - Fine details")
    print(f"  Total:      {total_tokens:3d} tokens")
    
    print("\n" + "-"*80)
    print("5. TOKEN EXAMPLES")
    print("-"*80)
    
    # Show example of each type
    if dir_tokens:
        example_dir = dir_tokens[0]
        angle_str = example_dir.split("_")[1].split(":")[0]
        level_str = example_dir.split(":")[1][:-1]
        print(f"\nDirection token: {example_dir}")
        print(f"  → Angle: {angle_str}° (direction ID)")
        print(f"  → Level: {level_str} (correlation strength: 0-15)")
    
    if atom_tokens:
        example_atom = atom_tokens[0]
        atom_id = example_atom.split("_")[1].split(":")[0]
        level_str = example_atom.split(":")[1][:-1]
        print(f"\nAtom token: {example_atom}")
        print(f"  → Atom ID: {atom_id} (NMF component)")
        print(f"  → Level: {level_str} (activation strength: 0-15)")
    
    if patch_tokens:
        example_patch = patch_tokens[0]
        parts = example_patch[3:-1].split("_")  # Remove <P_ and >
        print(f"\nPatch token: {example_patch}")
        print(f"  → Freq patch: {parts[0]} (time-frequency bin)")
        print(f"  → Time patch: {parts[1]}")
        print(f"  → Level: {parts[2]} (log-magnitude: 0-15)")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Use these tokenizers in DoAICLDataset (Day 3-4)")
    print("  2. Update HF tokenizer vocabulary to include new tokens (Day 7)")
    print("  3. Integrate with training scripts (Day 8-9)")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
