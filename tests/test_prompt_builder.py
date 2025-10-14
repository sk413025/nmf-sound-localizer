"""
Tests for multi-modal prompt builder.

This test suite validates the MultiModalPromptBuilder and PromptConfig classes
with comprehensive coverage of:
- Basic prompt building with different token combinations
- Token ordering strategies
- Token limits and budgets
- ICL prompt building
- Context sampling strategies
"""

import pytest
import numpy as np
from unittest.mock import Mock

from doa_rl.features.prompt_builder import (
    MultiModalPromptBuilder,
    PromptConfig,
)


# Mock tokenizers for testing
class MockPatchTokenizer:
    def __call__(self, Y):
        # Generate simple patch tokens
        return [f"<P_{i}_{j}_5>" for i in range(2) for j in range(3)]


class MockAtomTokenizer:
    def __call__(self, Y):
        # Generate atom tokens
        return [f"<AT_{i}:10>" for i in [5, 12, 23, 41]]


class MockDirectionTokenizer:
    def __call__(self, Y, top_m=5):
        # Generate direction tokens
        angles = [90, 85, 95, 80, 100]
        return [f"<R_{angle:03d}:14>" for angle in angles[:top_m]]


@pytest.fixture
def tokenizers():
    """Create mock tokenizers."""
    return {
        "patch": MockPatchTokenizer(),
        "atom": MockAtomTokenizer(),
        "direction": MockDirectionTokenizer(),
    }


@pytest.fixture
def sample_Y():
    """Create a sample spectrogram."""
    return np.random.rand(116, 189)


class TestPromptConfig:
    """Test PromptConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PromptConfig()
        assert config.use_patches is True
        assert config.use_atoms is True
        assert config.use_directions is True
        assert config.ordering == "physics_first"
        assert config.max_tokens is None
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PromptConfig(
            use_patches=False,
            ordering="structure_first",
            max_tokens=100,
        )
        assert config.use_patches is False
        assert config.ordering == "structure_first"
        assert config.max_tokens == 100


class TestMultiModalPromptBuilder:
    """Test MultiModalPromptBuilder class."""
    
    def test_initialization_all_tokenizers(self, tokenizers):
        """Test initialization with all tokenizers."""
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        assert builder.patch_tok is not None
        assert builder.atom_tok is not None
        assert builder.dir_tok is not None
        assert builder.config.use_patches is True
        assert builder.config.use_atoms is True
        assert builder.config.use_directions is True
    
    def test_initialization_patch_only(self, tokenizers):
        """Test initialization with only patch tokenizer."""
        config = PromptConfig(use_atoms=False, use_directions=False)
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            config=config,
        )
        assert builder.patch_tok is not None
        assert builder.atom_tok is None
        assert builder.dir_tok is None
    
    def test_auto_disable_missing_tokenizers(self, tokenizers):
        """Test automatic disabling of features when tokenizers are missing."""
        config = PromptConfig(use_atoms=True, use_directions=True)
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            atom_tokenizer=None,
            direction_tokenizer=None,
            config=config,
        )
        # Should auto-disable
        assert builder.config.use_atoms is False
        assert builder.config.use_directions is False
    
    def test_build_prompt_all_tokens(self, tokenizers, sample_Y):
        """Test building prompt with all token types."""
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        prompt = builder.build_prompt(sample_Y)
        
        # Check that prompt contains all token types
        assert "<R_090:14>" in prompt  # Direction token
        assert "<AT_5:10>" in prompt   # Atom token
        assert "<P_0_0_5>" in prompt   # Patch token
    
    def test_build_prompt_patch_only(self, tokenizers, sample_Y):
        """Test building prompt with only patch tokens."""
        config = PromptConfig(use_atoms=False, use_directions=False)
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        
        # Should only contain patch tokens
        assert "<P_" in prompt
        assert "<AT_" not in prompt
        assert "<R_" not in prompt
    
    def test_ordering_physics_first(self, tokenizers, sample_Y):
        """Test physics_first ordering strategy."""
        config = PromptConfig(ordering="physics_first")
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Direction tokens should come first
        assert tokens[0].startswith("<R_")
        # Find first atom token
        first_atom_idx = next(i for i, t in enumerate(tokens) if t.startswith("<AT_"))
        # Find first patch token
        first_patch_idx = next(i for i, t in enumerate(tokens) if t.startswith("<P_"))
        
        # Verify ordering
        assert first_atom_idx > 0  # After at least one direction token
        assert first_patch_idx > first_atom_idx
    
    def test_ordering_structure_first(self, tokenizers, sample_Y):
        """Test structure_first ordering strategy."""
        config = PromptConfig(ordering="structure_first")
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Atom tokens should come first
        assert tokens[0].startswith("<AT_")
    
    def test_ordering_patch_first(self, tokenizers, sample_Y):
        """Test patch_first ordering strategy."""
        config = PromptConfig(ordering="patch_first")
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Patch tokens should come first
        assert tokens[0].startswith("<P_")
    
    def test_ordering_interleaved(self, tokenizers, sample_Y):
        """Test interleaved ordering strategy."""
        config = PromptConfig(ordering="interleaved")
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Check that different token types are mixed
        # First few tokens should contain mix of types
        first_10_types = [t.split("_")[0] for t in tokens[:10]]
        assert "<R" in first_10_types
        assert "<AT" in first_10_types
        assert "<P" in first_10_types
    
    def test_max_tokens_limit(self, tokenizers, sample_Y):
        """Test global max_tokens limit."""
        config = PromptConfig(max_tokens=10)
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        assert len(tokens) <= 10
    
    def test_max_patch_tokens_limit(self, tokenizers, sample_Y):
        """Test max_patch_tokens limit."""
        config = PromptConfig(
            max_patch_tokens=3,
            use_atoms=False,
            use_directions=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        assert len(tokens) <= 3
        assert all(t.startswith("<P_") for t in tokens)
    
    def test_max_atom_tokens_limit(self, tokenizers, sample_Y):
        """Test max_atom_tokens limit."""
        config = PromptConfig(
            max_atom_tokens=2,
            use_patches=False,
            use_directions=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        assert len(tokens) <= 2
        assert all(t.startswith("<AT_") for t in tokens)
    
    def test_max_direction_tokens_limit(self, tokenizers, sample_Y):
        """Test max_direction_tokens limit."""
        config = PromptConfig(
            max_direction_tokens=3,
            use_patches=False,
            use_atoms=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            None,
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        assert len(tokens) <= 3
        assert all(t.startswith("<R_") for t in tokens)
    
    def test_direction_top_m_override(self, tokenizers, sample_Y):
        """Test direction_top_m config override."""
        config = PromptConfig(
            direction_top_m=2,
            use_patches=False,
            use_atoms=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            None,
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Should only have 2 direction tokens
        assert len(tokens) == 2
    
    def test_atom_top_k_override(self, tokenizers, sample_Y):
        """Test atom_top_k config override."""
        config = PromptConfig(
            atom_top_k=2,
            use_patches=False,
            use_directions=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        tokens = prompt.split()
        
        # Should only have 2 atom tokens
        assert len(tokens) == 2
    
    def test_build_icl_prompt_basic(self, tokenizers, sample_Y):
        """Test basic ICL prompt building."""
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        # Create context examples
        context_Y1 = np.random.rand(116, 189)
        context_Y2 = np.random.rand(116, 189)
        context_examples = [
            (context_Y1, "<D_045>"),
            (context_Y2, "<D_090>"),
        ]
        
        prompt = builder.build_icl_prompt(sample_Y, context_examples)
        
        # Should contain context targets
        assert "<D_045>" in prompt
        assert "<D_090>" in prompt
        # Should contain multiple sets of tokens (context + query)
        assert prompt.count("<R_090:14>") >= 1  # At least in query
    
    def test_build_icl_prompt_with_target(self, tokenizers, sample_Y):
        """Test ICL prompt with target token."""
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        context_examples = [
            (np.random.rand(116, 189), "<D_030>"),
        ]
        
        prompt = builder.build_icl_prompt(
            sample_Y,
            context_examples,
            target_token="<D_120>",
        )
        
        # Should end with target token
        assert prompt.endswith("<D_120>")
    
    def test_sample_context_random(self, tokenizers):
        """Test random context sampling strategy."""
        # Create mock dataset
        mock_dataset = [
            {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
            for angle in range(0, 180, 15)
        ]
        
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        context = builder.sample_context_examples(
            mock_dataset,
            query_idx=5,
            n_shots=3,
            strategy="random",
        )
        
        assert len(context) == 3
        # Each context is (Y, target_token)
        for Y, target in context:
            assert Y.shape == (116, 189)
            assert target.startswith("<D_")
    
    def test_sample_context_nearest_angle(self, tokenizers):
        """Test nearest_angle context sampling strategy."""
        # Create mock dataset
        mock_dataset = [
            {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
            for angle in range(0, 180, 5)
        ]
        
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        # Query at 90 degrees (index 18)
        context = builder.sample_context_examples(
            mock_dataset,
            query_idx=18,
            n_shots=3,
            strategy="nearest_angle",
        )
        
        assert len(context) == 3
        # Extract angles from targets
        angles = [int(target.split("_")[1].split(">")[0]) for _, target in context]
        # Should be close to 90
        for angle in angles:
            assert abs(angle - 90) <= 20  # Reasonable proximity
    
    def test_sample_context_diverse_angle(self, tokenizers):
        """Test diverse_angle context sampling strategy."""
        # Create mock dataset
        mock_dataset = [
            {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
            for angle in range(0, 180, 10)
        ]
        
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        context = builder.sample_context_examples(
            mock_dataset,
            query_idx=9,
            n_shots=3,
            strategy="diverse_angle",
        )
        
        assert len(context) == 3
        # Extract angles
        angles = [int(target.split("_")[1].split(">")[0]) for _, target in context]
        # Check diversity: all angles should be different
        assert len(set(angles)) == 3
    
    def test_sample_context_exclude_same_angle(self, tokenizers):
        """Test excluding same angle from context."""
        # Create mock dataset with duplicate angles
        mock_dataset = [
            {"Y": np.random.rand(116, 189), "angle_deg": 90.0}
            for _ in range(5)
        ] + [
            {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
            for angle in [30, 60, 120, 150]
        ]
        
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        context = builder.sample_context_examples(
            mock_dataset,
            query_idx=0,  # 90 degrees
            n_shots=3,
            strategy="random",
            exclude_same_angle=True,
        )
        
        # Should not include any 90 degree samples
        angles = [int(target.split("_")[1].split(">")[0]) for _, target in context]
        assert 90 not in angles
    
    def test_empty_prompt_all_disabled(self, tokenizers, sample_Y):
        """Test that disabling all token types produces empty prompt."""
        config = PromptConfig(
            use_patches=False,
            use_atoms=False,
            use_directions=False,
        )
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        prompt = builder.build_prompt(sample_Y)
        
        assert prompt == ""
    
    def test_invalid_ordering_raises_error(self, tokenizers, sample_Y):
        """Test that invalid ordering strategy raises error."""
        config = PromptConfig(ordering="invalid_ordering")
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
            config=config,
        )
        
        with pytest.raises(ValueError, match="Unknown ordering strategy"):
            builder.build_prompt(sample_Y)
    
    def test_invalid_sampling_strategy_raises_error(self, tokenizers):
        """Test that invalid sampling strategy raises error."""
        mock_dataset = [
            {"Y": np.random.rand(116, 189), "angle_deg": 90.0}
            for _ in range(5)
        ]
        
        builder = MultiModalPromptBuilder(
            tokenizers["patch"],
            tokenizers["atom"],
            tokenizers["direction"],
        )
        
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            builder.sample_context_examples(
                mock_dataset,
                query_idx=0,
                n_shots=2,
                strategy="invalid_strategy",
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
