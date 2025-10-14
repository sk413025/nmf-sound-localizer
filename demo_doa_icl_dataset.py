#!/usr/bin/env python3
"""
Demo script for DoAICLDataset with real W/H matrices and data.

This script demonstrates:
1. Loading real W matrix (NMF dictionary) and H matrix (transfer functions)
2. Creating tokenizers with real matrices
3. Building multi-modal prompts with actual audio data
4. ICL prompt generation with context examples
5. Token distribution analysis
6. Comparison of different ICL sampling strategies

Run: python demo_doa_icl_dataset.py
"""

import sys
from pathlib import Path
import numpy as np
import torch

from doa_rl.data import DoAICLDataset
from doa_rl.features import (
    PatchTokenizer,
    NMFAtomTokenizer,
    DirectionProjectionTokenizer,
    MultiModalPromptBuilder,
    PromptConfig
)


def load_matrices():
    """Load W and H matrices from disk."""
    print("="*70)
    print("Loading Matrices")
    print("="*70)
    
    # Load W matrix (NMF dictionary)
    w_path = "doa_normalized_config_c_corrected/models/usm.pth"
    w_data = torch.load(w_path)
    W = w_data["W"]  # (F, K)
    print(f"✅ Loaded W matrix from {w_path}")
    print(f"   Shape: {W.shape} (F={W.shape[0]}, K={W.shape[1]} atoms)")
    
    # Load H matrix (transfer functions)
    h_path = "h_matrix_normalized_original_to_box.pth"
    h_data = torch.load(h_path, weights_only=False)
    H = h_data["H"]  # (F, D)
    angles = h_data["angles"]
    print(f"✅ Loaded H matrix from {h_path}")
    print(f"   Shape: {H.shape} (F={H.shape[0]}, D={H.shape[1]} directions)")
    print(f"   Angles: {angles}")
    
    # Handle dimension mismatch
    if W.shape[0] != H.shape[0]:
        print(f"\n⚠️  Dimension mismatch: W has {W.shape[0]} freqs, H has {H.shape[0]} freqs")
        # Pad or truncate to match
        target_F = W.shape[0]
        if H.shape[0] < target_F:
            pad_size = target_F - H.shape[0]
            H_padded = torch.cat([H, torch.zeros(pad_size, H.shape[1])], dim=0)
            print(f"   Padded H to {H_padded.shape}")
            H = H_padded
        else:
            H = H[:target_F, :]
            print(f"   Truncated H to {H.shape}")
    
    return W, H, angles


def create_tokenizers_and_builder(W, H, angles):
    """Create tokenizers and prompt builder."""
    print("\n" + "="*70)
    print("Creating Tokenizers and Prompt Builder")
    print("="*70)
    
    # Create individual tokenizers
    patch_tok = PatchTokenizer()
    print("✅ Created PatchTokenizer")
    
    atom_tok = NMFAtomTokenizer(
        W=W.numpy(),
        top_k=8,
        n_levels=16
    )
    print(f"✅ Created NMFAtomTokenizer (top_k=8, {W.shape[1]} total atoms)")
    
    dir_tok = DirectionProjectionTokenizer(
        H=H.numpy(),
        angles=angles,
        top_m=5,
        n_levels=16
    )
    print(f"✅ Created DirectionProjectionTokenizer (top_m=5, {len(angles)} total angles)")
    
    # Create prompt builder
    config = PromptConfig(
        use_patches=True,
        use_atoms=True,
        use_directions=True,
        ordering="physics_first",
        max_tokens=None  # No limit for demo
    )
    
    builder = MultiModalPromptBuilder(
        patch_tokenizer=patch_tok,
        atom_tokenizer=atom_tok,
        direction_tokenizer=dir_tok,
        config=config
    )
    print(f"✅ Created MultiModalPromptBuilder (ordering={config.ordering})")
    
    return builder, config


def demo_basic_prompts(dataset):
    """Demo 1: Basic multi-modal prompt generation."""
    print("\n" + "="*70)
    print("Demo 1: Basic Multi-Modal Prompt Generation")
    print("="*70)
    
    # Get a few samples
    for i in range(3):
        batch = dataset[i]
        prompt = batch["prompt"]
        angle = batch["angle_deg"]
        
        print(f"\nSample {i+1}:")
        print(f"  Angle: {angle}°")
        print(f"  Spectrogram shape: {batch['Y'].shape}")
        print(f"  Prompt length: {len(prompt.split())} tokens")
        print(f"  Prompt (first 150 chars): {prompt[:150]}...")
        
        # Analyze token types
        tokens = prompt.split()
        dir_tokens = [t for t in tokens if t.startswith("<R_")]
        atom_tokens = [t for t in tokens if t.startswith("<AT_")]
        patch_tokens = [t for t in tokens if t.startswith("<P_")]
        
        print(f"  Token breakdown:")
        print(f"    Direction: {len(dir_tokens)} ({len(dir_tokens)/len(tokens)*100:.1f}%)")
        print(f"    Atom: {len(atom_tokens)} ({len(atom_tokens)/len(tokens)*100:.1f}%)")
        print(f"    Patch: {len(patch_tokens)} ({len(patch_tokens)/len(tokens)*100:.1f}%)")


def demo_icl_prompts(icl_dataset):
    """Demo 2: ICL prompts with context examples."""
    print("\n" + "="*70)
    print("Demo 2: ICL Prompts with Context Examples")
    print("="*70)
    
    batch = icl_dataset[0]
    prompt = batch["prompt"]
    context_indices = batch["context_indices"]
    
    print(f"\nQuery angle: {batch['angle_deg']}°")
    print(f"Context indices: {context_indices}")
    
    # Get context angles
    context_angles = [icl_dataset.index[i][1] for i in context_indices]
    print(f"Context angles: {context_angles}")
    
    print(f"\nICL prompt length: {len(prompt.split())} tokens")
    
    # Count direction tokens (should be n_shots)
    dir_tokens = [t for t in prompt.split() if t.startswith("<D_")]
    print(f"Direction tokens (<D_>): {len(dir_tokens)} (one per context example)")
    print(f"Direction tokens: {dir_tokens}")
    
    # Show prompt structure
    print(f"\nPrompt structure (first 200 chars):")
    print(f"  {prompt[:200]}...")


def demo_sampling_strategies(data_root, builder):
    """Demo 3: Comparison of ICL sampling strategies."""
    print("\n" + "="*70)
    print("Demo 3: ICL Context Sampling Strategies")
    print("="*70)
    
    angles = [float(a) for a in range(0, 181, 10)]  # 0, 10, 20, ..., 180
    
    strategies = ["random", "nearest", "diverse"]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.upper()} ---")
        
        dataset = DoAICLDataset(
            root=data_root,
            angles=angles,
            prompt_builder=builder,
            icl_mode=True,
            n_shots=3,
            context_strategy=strategy,
            exclude_query_angle=True
        )
        
        # Test on sample at 90 degrees
        query_indices = [i for i, (path, angle, _) in enumerate(dataset.index) if angle == 90.0]
        if not query_indices:
            print("  No sample at 90° found, using first sample")
            query_idx = 0
        else:
            query_idx = query_indices[0]
        
        batch = dataset[query_idx]
        query_angle = batch["angle_deg"]
        context_angles = [dataset.index[i][1] for i in batch["context_indices"]]
        
        print(f"  Query angle: {query_angle}°")
        print(f"  Context angles: {context_angles}")
        
        # Calculate angle distances
        distances = [abs(query_angle - ca) for ca in context_angles]
        print(f"  Angle distances: {distances}")
        print(f"  Avg distance: {np.mean(distances):.1f}°")
        
        # For diverse, show min pairwise distance
        if strategy == "diverse":
            min_dist = float('inf')
            for i in range(len(context_angles)):
                for j in range(i+1, len(context_angles)):
                    dist = abs(context_angles[i] - context_angles[j])
                    min_dist = min(min_dist, dist)
            print(f"  Min pairwise distance: {min_dist:.1f}°")


def demo_token_budgets(dataset, builder):
    """Demo 4: Token budget management."""
    print("\n" + "="*70)
    print("Demo 4: Token Budget Management")
    print("="*70)
    
    # Test different budget settings
    budgets = [None, 200, 100, 50]
    
    # Get a sample spectrogram
    batch = dataset[0]
    Y = batch["Y"].numpy()
    
    for budget in budgets:
        config = PromptConfig(
            use_patches=True,
            use_atoms=True,
            use_directions=True,
            ordering="physics_first",
            max_tokens=budget
        )
        
        test_builder = MultiModalPromptBuilder(
            patch_tokenizer=builder.patch_tok,
            atom_tokenizer=builder.atom_tok,
            direction_tokenizer=builder.dir_tok,
            config=config
        )
        
        prompt = test_builder.build_prompt(Y)
        tokens = prompt.split()
        
        dir_tokens = [t for t in tokens if t.startswith("<R_")]
        atom_tokens = [t for t in tokens if t.startswith("<AT_")]
        patch_tokens = [t for t in tokens if t.startswith("<P_")]
        
        budget_str = f"{budget}" if budget else "unlimited"
        print(f"\nBudget: {budget_str}")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Direction: {len(dir_tokens)} ({len(dir_tokens)/len(tokens)*100:.1f}%)")
        print(f"  Atom: {len(atom_tokens)} ({len(atom_tokens)/len(tokens)*100:.1f}%)")
        print(f"  Patch: {len(patch_tokens)} ({len(patch_tokens)/len(tokens)*100:.1f}%)")
        
        if budget and len(tokens) > 0:
            baseline_tokens = len(dataset[0]["prompt"].split())
            compression = baseline_tokens / len(tokens)
            print(f"  Compression: {compression:.1f}x")


def demo_token_statistics(dataset, n_samples=10):
    """Demo 5: Token distribution statistics."""
    print("\n" + "="*70)
    print(f"Demo 5: Token Distribution Statistics ({n_samples} samples)")
    print("="*70)
    
    dir_counts = []
    atom_counts = []
    patch_counts = []
    total_counts = []
    
    for i in range(min(n_samples, len(dataset))):
        batch = dataset[i]
        tokens = batch["prompt"].split()
        
        dir_tokens = len([t for t in tokens if t.startswith("<R_")])
        atom_tokens = len([t for t in tokens if t.startswith("<AT_")])
        patch_tokens = len([t for t in tokens if t.startswith("<P_")])
        
        dir_counts.append(dir_tokens)
        atom_counts.append(atom_tokens)
        patch_counts.append(patch_tokens)
        total_counts.append(len(tokens))
    
    print(f"\nToken statistics across {n_samples} samples:")
    print(f"  Direction tokens:")
    print(f"    Mean: {np.mean(dir_counts):.1f} ± {np.std(dir_counts):.1f}")
    print(f"    Range: [{min(dir_counts)}, {max(dir_counts)}]")
    
    print(f"  Atom tokens:")
    print(f"    Mean: {np.mean(atom_counts):.1f} ± {np.std(atom_counts):.1f}")
    print(f"    Range: [{min(atom_counts)}, {max(atom_counts)}]")
    
    print(f"  Patch tokens:")
    print(f"    Mean: {np.mean(patch_counts):.1f} ± {np.std(patch_counts):.1f}")
    print(f"    Range: [{min(patch_counts)}, {max(patch_counts)}]")
    
    print(f"  Total tokens:")
    print(f"    Mean: {np.mean(total_counts):.1f} ± {np.std(total_counts):.1f}")
    print(f"    Range: [{min(total_counts)}, {max(total_counts)}]")


def main():
    """Main demo execution."""
    print("\n" + "="*70)
    print("DoAICLDataset Demo with Real Data")
    print("="*70)
    
    # Check for data directory
    data_dirs = [
        "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized",
        "doa_normalized_config_c_corrected",  # Fallback (if it has angle_* dirs)
    ]
    
    data_root = None
    for dir_path in data_dirs:
        if Path(dir_path).exists():
            # Check if it has angle_* subdirs
            angle_dirs = list(Path(dir_path).glob("angle_*"))
            if angle_dirs:
                data_root = dir_path
                break
    
    if not data_root:
        print("\n❌ Error: Could not find data directory with angle_* subdirectories")
        print("   Searched:")
        for dir_path in data_dirs:
            print(f"     - {dir_path}")
        print("\n   Please update data_dirs in demo script or ensure data is available")
        return 1
    
    print(f"\n✅ Using data directory: {data_root}")
    
    # Extract angles from directory names
    angle_dirs = sorted(Path(data_root).glob("angle_*"))
    angles = []
    for angle_dir in angle_dirs:
        try:
            angle = float(angle_dir.name.replace("angle_", ""))
            angles.append(angle)
        except ValueError:
            continue
    
    print(f"   Found {len(angles)} angles: {angles}")
    
    # Load matrices
    W, H, h_angles = load_matrices()
    
    # Create tokenizers and builder
    builder, config = create_tokenizers_and_builder(W, H, h_angles)
    
    # Create basic dataset
    basic_dataset = DoAICLDataset(
        root=data_root,
        angles=angles,
        prompt_builder=builder,
        icl_mode=False
    )
    print(f"\n✅ Created basic dataset with {len(basic_dataset)} samples")
    
    # Create ICL dataset
    icl_dataset = DoAICLDataset(
        root=data_root,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="random"
    )
    print(f"✅ Created ICL dataset with {len(icl_dataset)} samples")
    
    # Run demos
    demo_basic_prompts(basic_dataset)
    demo_icl_prompts(icl_dataset)
    demo_sampling_strategies(data_root, builder)
    demo_token_budgets(basic_dataset, builder)
    demo_token_statistics(basic_dataset, n_samples=10)
    
    # Summary
    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\n✅ All demos executed successfully")
    print("\nKey Takeaways:")
    print("  1. Multi-modal prompts combine direction, atom, and patch tokens")
    print("  2. ICL mode adds few-shot context examples with <D_> direction labels")
    print("  3. Different sampling strategies (random, nearest, diverse) available")
    print("  4. Token budgets control prompt length and information density")
    print("  5. DoAICLDataset seamlessly integrates with existing training pipeline")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
