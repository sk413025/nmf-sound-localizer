"""
Comprehensive test suite for DoAICLDataset.

Tests cover:
1. Basic prompt generation (single-modal and multi-modal)
2. ICL batch creation with different sampling strategies
3. Context sampling (random, nearest, diverse)
4. Edge cases (insufficient data, invalid strategies)
5. Integration with existing DoADataset functionality
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from typing import List
import tempfile
import shutil

from doa_rl.data import DoADataset, DoAICLDataset
from doa_rl.features import PromptConfig


# Mock tokenizers for testing
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
        """Build simple mock prompt."""
        tokens = []
        if self.config.use_directions:
            tokens.extend(self.dir_tok(Y))
        if self.config.use_atoms:
            tokens.extend(self.atom_tok(Y))
        if self.config.use_patches:
            tokens.extend(self.patch_tok(Y))
        return " ".join(tokens)


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with mock audio files."""
    tmpdir = tempfile.mkdtemp()
    
    # Create angle directories with mock .npy files
    angles = [0, 30, 60, 90, 120, 150, 180]
    for angle in angles:
        angle_dir = Path(tmpdir) / f"angle_{angle}"
        angle_dir.mkdir()
        
        # Create 3 clips per angle
        for clip_idx in range(3):
            # Mock audio: 16000 Hz * 1 second = 16000 samples
            mock_wav = np.random.randn(16000).astype(np.float32)
            clip_path = angle_dir / f"clip_{clip_idx:03d}.npy"
            np.save(clip_path, mock_wav)
    
    yield tmpdir
    
    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def base_dataset(temp_data_dir):
    """Create base DoADataset for testing."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    return DoADataset(temp_data_dir, angles, fs=16000, n_fft=2048)


@pytest.fixture
def mock_prompt_builder():
    """Create mock prompt builder."""
    config = PromptConfig(
        use_patches=True,
        use_atoms=True,
        use_directions=True,
        ordering="physics_first"
    )
    return MockPromptBuilder(config=config)


# ========== Basic Functionality Tests ==========

def test_doa_icl_dataset_initialization(temp_data_dir, mock_prompt_builder):
    """Test DoAICLDataset initializes correctly."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=False
    )
    
    assert len(dataset) == 21  # 7 angles × 3 clips
    assert dataset.prompt_builder is mock_prompt_builder
    assert dataset.icl_mode is False


def test_basic_prompt_generation(temp_data_dir, mock_prompt_builder):
    """Test basic prompt generation (non-ICL mode)."""
    angles = [0.0, 30.0, 60.0, 90.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=False
    )
    
    batch = dataset[0]
    
    # Check batch structure
    assert "Y" in batch
    assert "angle_deg" in batch
    assert "angle_index" in batch
    assert "path" in batch
    assert "prompt" in batch
    
    # Check prompt format (direction, atom, patch tokens)
    prompt = batch["prompt"]
    assert "<R_" in prompt
    assert "<AT_" in prompt
    assert "<P_" in prompt
    
    # No context indices in basic mode
    assert "context_indices" not in batch


def test_multi_modal_prompt_content(temp_data_dir):
    """Test multi-modal prompt contains all token types."""
    config = PromptConfig(
        use_patches=True,
        use_atoms=True,
        use_directions=True
    )
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=[0.0, 90.0],
        prompt_builder=builder,
        icl_mode=False
    )
    
    batch = dataset[0]
    prompt = batch["prompt"]
    
    # Count token types
    tokens = prompt.split()
    direction_tokens = [t for t in tokens if t.startswith("<R_")]
    atom_tokens = [t for t in tokens if t.startswith("<AT_")]
    patch_tokens = [t for t in tokens if t.startswith("<P_")]
    
    assert len(direction_tokens) > 0
    assert len(atom_tokens) > 0
    assert len(patch_tokens) > 0


def test_single_modal_prompt_patches_only(temp_data_dir):
    """Test prompt with only patch tokens."""
    config = PromptConfig(
        use_patches=True,
        use_atoms=False,
        use_directions=False
    )
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=[0.0, 90.0],
        prompt_builder=builder,
        icl_mode=False
    )
    
    batch = dataset[0]
    prompt = batch["prompt"]
    
    # Should only have patch tokens
    tokens = prompt.split()
    assert all(t.startswith("<P_") for t in tokens)


# ========== ICL Mode Tests ==========

def test_icl_mode_prompt_structure(temp_data_dir, mock_prompt_builder):
    """Test ICL prompt has correct structure with context examples."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3
    )
    
    batch = dataset[0]
    
    # Check ICL-specific fields
    assert "prompt" in batch
    assert "context_indices" in batch
    assert len(batch["context_indices"]) == 3
    
    # Check prompt contains direction tokens for context examples
    prompt = batch["prompt"]
    direction_tokens = [t for t in prompt.split() if t.startswith("<D_")]
    assert len(direction_tokens) >= 3  # At least one per context example


def test_icl_context_indices_valid(temp_data_dir, mock_prompt_builder):
    """Test ICL context indices are valid and exclude query."""
    angles = [0.0, 30.0, 60.0, 90.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3
    )
    
    query_idx = 5
    batch = dataset[query_idx]
    context_indices = batch["context_indices"]
    
    # Context indices should not include query
    assert query_idx not in context_indices
    
    # All indices should be valid
    assert all(0 <= idx < len(dataset) for idx in context_indices)


# ========== Context Sampling Strategy Tests ==========

def test_random_sampling_strategy(temp_data_dir, mock_prompt_builder):
    """Test random context sampling."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="random"
    )
    
    batch = dataset[0]
    
    # Should have n_shots context examples
    assert len(batch["context_indices"]) == 3
    
    # Run multiple times to check randomness
    indices_sets = []
    for _ in range(5):
        batch = dataset[0]
        indices_sets.append(tuple(sorted(batch["context_indices"])))
    
    # Should have some variation (not all identical)
    unique_sets = set(indices_sets)
    # Note: Small chance all 5 could be identical, but unlikely
    assert len(unique_sets) > 1 or len(indices_sets) == 1  # Accept if deterministic for small dataset


def test_nearest_sampling_strategy(temp_data_dir, mock_prompt_builder):
    """Test nearest angle context sampling."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="nearest",
        exclude_query_angle=True
    )
    
    # Query at 90 degrees
    query_indices = [i for i, (path, angle, _) in enumerate(dataset.index) if angle == 90.0]
    query_idx = query_indices[0]
    
    batch = dataset[query_idx]
    context_indices = batch["context_indices"]
    
    # Get angles of context examples
    context_angles = [dataset.index[i][1] for i in context_indices]
    
    # Nearest to 90 should be 60, 120 (exclude 90 itself)
    # With 3 shots: should include 60, 120, and either 30 or 150
    assert 60.0 in context_angles or 120.0 in context_angles
    assert 90.0 not in context_angles  # Excluded


def test_diverse_sampling_strategy(temp_data_dir, mock_prompt_builder):
    """Test diverse angle context sampling."""
    angles = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3,
        context_strategy="diverse"
    )
    
    batch = dataset[0]
    context_indices = batch["context_indices"]
    
    # Get angles of context examples
    context_angles = [dataset.index[i][1] for i in context_indices]
    
    # Check pairwise angle distances
    min_pairwise_dist = float('inf')
    for i in range(len(context_angles)):
        for j in range(i+1, len(context_angles)):
            dist = abs(context_angles[i] - context_angles[j])
            min_pairwise_dist = min(min_pairwise_dist, dist)
    
    # Diverse sampling should have larger minimum distance than random
    # (at least 20 degrees apart for 7 angles)
    assert min_pairwise_dist >= 20.0


# ========== Edge Cases Tests ==========

def test_insufficient_context_samples(temp_data_dir, mock_prompt_builder):
    """Test behavior when not enough context samples available."""
    # Create dataset with only 2 samples
    angles = [0.0]  # Only one angle, 3 clips
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=5,  # Request more than available
        exclude_query_angle=False  # Don't exclude to have more options
    )
    
    batch = dataset[0]
    
    # Should return available samples (less than n_shots)
    assert len(batch["context_indices"]) <= 5
    assert len(batch["context_indices"]) <= len(dataset) - 1  # Exclude query


def test_exclude_query_angle_option(temp_data_dir, mock_prompt_builder):
    """Test exclude_query_angle parameter."""
    angles = [0.0, 30.0, 60.0, 90.0]
    
    # Test with exclusion
    dataset_exclude = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=2,
        exclude_query_angle=True
    )
    
    query_idx = 0
    query_angle = dataset_exclude.index[query_idx][1]
    batch = dataset_exclude[query_idx]
    context_angles = [dataset_exclude.index[i][1] for i in batch["context_indices"]]
    
    assert query_angle not in context_angles
    
    # Test without exclusion
    dataset_include = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=2,
        exclude_query_angle=False
    )
    
    batch = dataset_include[query_idx]
    # Context may include same angle (different clip)


def test_invalid_context_strategy(temp_data_dir, mock_prompt_builder):
    """Test invalid context strategy raises error."""
    angles = [0.0, 30.0, 60.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=2,
        context_strategy="invalid_strategy"
    )
    
    with pytest.raises(ValueError, match="Unknown context_strategy"):
        batch = dataset[0]


# ========== Integration Tests ==========

def test_compatibility_with_base_dataset(temp_data_dir, mock_prompt_builder):
    """Test DoAICLDataset maintains compatibility with DoADataset."""
    angles = [0.0, 30.0, 60.0]
    
    base_dataset = DoADataset(temp_data_dir, angles)
    icl_dataset = DoAICLDataset(
        temp_data_dir,
        angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=False
    )
    
    # Same length
    assert len(base_dataset) == len(icl_dataset)
    
    # Same basic fields
    base_batch = base_dataset[0]
    icl_batch = icl_dataset[0]
    
    assert base_batch["angle_deg"] == icl_batch["angle_deg"]
    assert base_batch["angle_index"] == icl_batch["angle_index"]
    assert torch.allclose(base_batch["Y"], icl_batch["Y"])


def test_dataloader_integration(temp_data_dir, mock_prompt_builder):
    """Test DoAICLDataset works with DataLoader."""
    from torch.utils.data import DataLoader
    
    angles = [0.0, 30.0, 60.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=2
    )
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test iteration
    for batch in loader:
        assert "Y" in batch
        assert "prompt" in batch
        assert "context_indices" in batch
        break  # Just test first batch


def test_prompt_consistency_across_calls(temp_data_dir):
    """Test that prompt generation is deterministic for same input."""
    config = PromptConfig(use_patches=True, use_atoms=True, use_directions=True)
    builder = MockPromptBuilder(config=config)
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=[0.0, 30.0, 60.0],
        prompt_builder=builder,
        icl_mode=False
    )
    
    # Get same sample multiple times
    prompts = [dataset[0]["prompt"] for _ in range(5)]
    
    # All prompts should be identical (deterministic)
    assert all(p == prompts[0] for p in prompts)


# ========== Performance Tests ==========

def test_prompt_generation_performance(temp_data_dir, mock_prompt_builder):
    """Test prompt generation is reasonably fast."""
    import time
    
    angles = [0.0, 30.0, 60.0, 90.0, 120.0]
    
    dataset = DoAICLDataset(
        root=temp_data_dir,
        angles=angles,
        prompt_builder=mock_prompt_builder,
        icl_mode=True,
        n_shots=3
    )
    
    # Time 10 samples
    start = time.time()
    for i in range(10):
        batch = dataset[i]
    elapsed = time.time() - start
    
    # Should be under 1 second for 10 samples (very generous)
    assert elapsed < 1.0
    
    # Average time per sample
    avg_time = elapsed / 10
    print(f"\nAverage prompt generation time: {avg_time*1000:.2f}ms")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
