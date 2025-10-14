#!/usr/bin/env python3
"""
Validation script for DoAICLDataset (no pytest required).

Validates:
1. Basic prompt generation (single and multi-modal)
2. ICL mode with context examples
3. Context sampling strategies (random, nearest, diverse)
4. Edge cases and error handling
5. Integration with base DoADataset

Run: python validate_doa_icl_dataset.py
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
from typing import List

from doa_rl.data import DoADataset, DoAICLDataset
from doa_rl.features import PromptConfig


# ========== Mock Components ==========

class MockPatchTokenizer:
    def __call__(self, Y: np.ndarray) -> List[str]:
        return [f"<P_{i}>" for i in range(5)]


class MockAtomTokenizer:
    def __call__(self, Y: np.ndarray) -> List[str]:
        return [f"<AT_{i}>" for i in range(3)]


class MockDirectionTokenizer:
    def __call__(self, Y: np.ndarray, top_m: int = 3) -> List[str]:
        return [f"<R_{i*30:03d}>" for i in range(top_m)]


class MockPromptBuilder:
    """Mock MultiModalPromptBuilder for testing."""
    
    def __init__(self, patch_tok=None, atom_tok=None, dir_tok=None, config=None):
        self.patch_tok = patch_tok or MockPatchTokenizer()
        self.atom_tok = atom_tok or MockAtomTokenizer()
        self.dir_tok = dir_tok or MockDirectionTokenizer()
        self.config = config or PromptConfig()
    
    def build_prompt(self, Y: np.ndarray) -> str:
        tokens = []
        if self.config.use_directions:
            tokens.extend(self.dir_tok(Y))
        if self.config.use_atoms:
            tokens.extend(self.atom_tok(Y))
        if self.config.use_patches:
            tokens.extend(self.patch_tok(Y))
        return " ".join(tokens)


def create_temp_dataset():
    """Create temporary dataset for testing."""
    tmpdir = tempfile.mkdtemp()
    
    angles = [0, 30, 60, 90, 120, 150, 180]
    for angle in angles:
        angle_dir = Path(tmpdir) / f"angle_{angle}"
        angle_dir.mkdir()
        
        for clip_idx in range(3):
            mock_wav = np.random.randn(16000).astype(np.float32)
            clip_path = angle_dir / f"clip_{clip_idx:03d}.npy"
            np.save(clip_path, mock_wav)
    
    return tmpdir


# ========== Test Functions ==========

def test_basic_initialization(temp_dir):
    """Test 1: Basic DoAICLDataset initialization."""
    print("\n" + "="*60)
    print("Test 1: Basic Initialization")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    config = PromptConfig(use_patches=True, use_atoms=True, use_directions=True)
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=False
    )
    
    assert len(dataset) == 21, f"Expected 21 samples, got {len(dataset)}"
    assert dataset.prompt_builder is builder
    assert dataset.icl_mode is False
    
    print("✅ Dataset initialized correctly")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Angles: {angles}")
    return True


def test_basic_prompt_generation(temp_dir):
    """Test 2: Basic prompt generation."""
    print("\n" + "="*60)
    print("Test 2: Basic Prompt Generation")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0]
    config = PromptConfig(use_patches=True, use_atoms=True, use_directions=True)
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=False
    )
    
    batch = dataset[0]
    
    # Check batch structure
    assert "Y" in batch
    assert "angle_deg" in batch
    assert "prompt" in batch
    assert "context_indices" not in batch  # Not in basic mode
    
    prompt = batch["prompt"]
    assert "<R_" in prompt, "Missing direction tokens"
    assert "<AT_" in prompt, "Missing atom tokens"
    assert "<P_" in prompt, "Missing patch tokens"
    
    print("✅ Prompt generated correctly")
    print(f"   Prompt: {prompt}")
    print(f"   Angle: {batch['angle_deg']}°")
    return True


def test_icl_mode_prompt(temp_dir):
    """Test 3: ICL mode prompt generation."""
    print("\n" + "="*60)
    print("Test 3: ICL Mode Prompt Generation")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0]
    config = PromptConfig(use_patches=True, use_atoms=True, use_directions=True)
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=3
    )
    
    batch = dataset[0]
    
    assert "context_indices" in batch
    assert len(batch["context_indices"]) == 3
    
    prompt = batch["prompt"]
    direction_tokens = [t for t in prompt.split() if t.startswith("<D_")]
    assert len(direction_tokens) >= 3, "Should have direction tokens for context"
    
    print("✅ ICL prompt generated correctly")
    print(f"   Context indices: {batch['context_indices']}")
    print(f"   Number of <D_> tokens: {len(direction_tokens)}")
    print(f"   Prompt length: {len(prompt.split())} tokens")
    return True


def test_random_sampling(temp_dir):
    """Test 4: Random context sampling."""
    print("\n" + "="*60)
    print("Test 4: Random Context Sampling")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="random"
    )
    
    batch = dataset[0]
    
    assert len(batch["context_indices"]) == 3
    assert 0 not in batch["context_indices"]  # Query excluded
    
    print("✅ Random sampling works")
    print(f"   Context indices: {batch['context_indices']}")
    return True


def test_nearest_sampling(temp_dir):
    """Test 5: Nearest angle sampling."""
    print("\n" + "="*60)
    print("Test 5: Nearest Angle Sampling")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="nearest",
        exclude_query_angle=True
    )
    
    # Find query at 90 degrees
    query_indices = [i for i, (path, angle, _) in enumerate(dataset.index) if angle == 90.0]
    query_idx = query_indices[0]
    
    batch = dataset[query_idx]
    context_angles = [dataset.index[i][1] for i in batch["context_indices"]]
    
    assert 90.0 not in context_angles  # Query angle excluded
    
    print("✅ Nearest sampling works")
    print(f"   Query angle: 90°")
    print(f"   Context angles: {context_angles}")
    print(f"   Context indices: {batch['context_indices']}")
    return True


def test_diverse_sampling(temp_dir):
    """Test 6: Diverse angle sampling."""
    print("\n" + "="*60)
    print("Test 6: Diverse Angle Sampling")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="diverse"
    )
    
    batch = dataset[0]
    context_angles = [dataset.index[i][1] for i in batch["context_indices"]]
    
    # Calculate minimum pairwise distance
    min_dist = float('inf')
    for i in range(len(context_angles)):
        for j in range(i+1, len(context_angles)):
            dist = abs(context_angles[i] - context_angles[j])
            min_dist = min(min_dist, dist)
    
    assert min_dist >= 20.0, f"Expected min distance ≥20°, got {min_dist}°"
    
    print("✅ Diverse sampling works")
    print(f"   Context angles: {context_angles}")
    print(f"   Min pairwise distance: {min_dist}°")
    return True


def test_edge_case_insufficient_samples(temp_dir):
    """Test 7: Edge case - insufficient context samples."""
    print("\n" + "="*60)
    print("Test 7: Edge Case - Insufficient Context Samples")
    print("="*60)
    
    angles = [0.0]  # Only one angle
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=5,  # Request more than available
        exclude_query_angle=False
    )
    
    batch = dataset[0]
    
    # Should return fewer than requested
    assert len(batch["context_indices"]) < 5
    assert len(batch["context_indices"]) <= len(dataset) - 1
    
    print("✅ Handles insufficient samples gracefully")
    print(f"   Requested shots: 5")
    print(f"   Actual shots: {len(batch['context_indices'])}")
    return True


def test_edge_case_invalid_strategy(temp_dir):
    """Test 8: Edge case - invalid strategy."""
    print("\n" + "="*60)
    print("Test 8: Edge Case - Invalid Strategy")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=2,
        context_strategy="invalid_strategy"
    )
    
    try:
        batch = dataset[0]
        print("❌ Should have raised ValueError")
        return False
    except ValueError as e:
        if "Unknown context_strategy" in str(e):
            print("✅ Correctly raises ValueError for invalid strategy")
            print(f"   Error message: {e}")
            return True
        else:
            print(f"❌ Wrong error: {e}")
            return False


def test_compatibility_with_base_dataset(temp_dir):
    """Test 9: Compatibility with DoADataset."""
    print("\n" + "="*60)
    print("Test 9: Compatibility with Base DoADataset")
    print("="*60)
    
    angles = [0.0, 30.0, 60.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    base_dataset = DoADataset(temp_dir, angles)
    icl_dataset = DoAICLDataset(
        temp_dir,
        angles,
        prompt_builder=builder,
        icl_mode=False
    )
    
    assert len(base_dataset) == len(icl_dataset)
    
    base_batch = base_dataset[0]
    icl_batch = icl_dataset[0]
    
    assert base_batch["angle_deg"] == icl_batch["angle_deg"]
    assert torch.allclose(base_batch["Y"], icl_batch["Y"])
    
    print("✅ Compatible with base DoADataset")
    print(f"   Both have {len(base_dataset)} samples")
    print(f"   Matching angles and spectrograms")
    return True


def test_dataloader_integration(temp_dir):
    """Test 10: DataLoader integration."""
    print("\n" + "="*60)
    print("Test 10: DataLoader Integration")
    print("="*60)
    
    from torch.utils.data import DataLoader
    
    angles = [0.0, 30.0, 60.0]
    config = PromptConfig()
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_dir,
        angles=angles,
        prompt_builder=builder,
        icl_mode=True,
        n_shots=2
    )
    
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    for batch in loader:
        assert "Y" in batch
        assert "prompt" in batch
        assert "context_indices" in batch
        print("✅ DataLoader integration works")
        print(f"   Batch size: {batch['Y'].shape[0]}")
        print(f"   Prompts: {len(batch['prompt'])}")
        return True
    
    return False


# ========== Main Execution ==========

def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("DoAICLDataset Validation Script")
    print("="*60)
    
    # Create temporary dataset
    print("\nCreating temporary test dataset...")
    temp_dir = create_temp_dataset()
    
    try:
        tests = [
            test_basic_initialization,
            test_basic_prompt_generation,
            test_icl_mode_prompt,
            test_random_sampling,
            test_nearest_sampling,
            test_diverse_sampling,
            test_edge_case_insufficient_samples,
            test_edge_case_invalid_strategy,
            test_compatibility_with_base_dataset,
            test_dataloader_integration,
        ]
        
        results = []
        for test_func in tests:
            try:
                result = test_func(temp_dir)
                results.append(result)
            except Exception as e:
                print(f"\n❌ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                results.append(False)
        
        # Summary
        print("\n" + "="*60)
        print("Validation Summary")
        print("="*60)
        
        passed = sum(results)
        total = len(results)
        
        print(f"\nTests passed: {passed}/{total}")
        
        if passed == total:
            print("\n🎉 All tests passed! DoAICLDataset is working correctly.")
            return 0
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please review the output above.")
            return 1
    
    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    sys.exit(main())
