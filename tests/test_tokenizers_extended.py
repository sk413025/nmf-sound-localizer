"""
Unit tests for extended tokenizers (NMFAtomTokenizer, DirectionProjectionTokenizer).
"""

import pytest
import numpy as np
from doa_rl.features.tokenizers_extended import (
    NMFAtomTokenizer,
    DirectionProjectionTokenizer,
)


class TestNMFAtomTokenizer:
    """Test suite for NMFAtomTokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization with valid parameters."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W, top_k=8, n_levels=16)
        
        assert tokenizer.K == 64
        assert tokenizer.top_k == 8
        assert tokenizer.n_levels == 16
    
    def test_initialization_invalid_W_shape(self):
        """Test that 1D W raises ValueError."""
        W = np.random.rand(116)
        with pytest.raises(ValueError, match="W must be 2D"):
            NMFAtomTokenizer(W)
    
    def test_top_k_exceeds_K(self):
        """Test that top_k is clamped to K when top_k > K."""
        W = np.random.rand(116, 10)
        tokenizer = NMFAtomTokenizer(W, top_k=20)
        assert tokenizer.top_k == 10  # Should be clamped to K
    
    def test_call_valid_input(self):
        """Test token generation with valid spectrogram."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W, top_k=5, n_levels=16)
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        # Check number of tokens
        assert len(tokens) == 5
        
        # Check token format
        for token in tokens:
            assert token.startswith("<AT_")
            assert token.endswith(">")
            assert ":" in token
    
    def test_token_format(self):
        """Test that tokens follow <AT_atom_id:level> format."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W, top_k=3, n_levels=16)
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        for token in tokens:
            # Extract atom_id and level
            token_body = token[4:-1]  # Remove <AT_ and >
            parts = token_body.split(":")
            
            assert len(parts) == 2
            atom_id = int(parts[0])
            level = int(parts[1])
            
            assert 0 <= atom_id < 64
            assert 0 <= level < 16
    
    def test_call_invalid_Y_shape(self):
        """Test that 1D Y raises ValueError."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W)
        Y = np.random.rand(116)
        
        with pytest.raises(ValueError, match="Y must be 2D"):
            tokenizer(Y)
    
    def test_call_mismatched_frequency_dimension(self):
        """Test that mismatched F dimension raises ValueError."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W)
        Y = np.random.rand(100, 189)  # Wrong F dimension
        
        with pytest.raises(ValueError, match="doesn't match"):
            tokenizer(Y)
    
    def test_quantization_levels(self):
        """Test that quantization produces valid levels."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W, top_k=8, n_levels=10)
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        for token in tokens:
            level = int(token.split(":")[1][:-1])
            assert 0 <= level < 10
    
    def test_deterministic_with_same_input(self):
        """Test that same input produces same tokens."""
        W = np.random.rand(116, 64)
        tokenizer = NMFAtomTokenizer(W, top_k=5, n_levels=16, n_iter=100)
        Y = np.random.rand(116, 189)
        
        tokens1 = tokenizer(Y)
        tokens2 = tokenizer(Y)
        
        assert tokens1 == tokens2
    
    def test_top_atoms_ordering(self):
        """Test that tokens are ordered by activation strength."""
        W = np.eye(64, 116).T  # Simple identity-like W
        tokenizer = NMFAtomTokenizer(W, top_k=5, n_levels=16)
        
        # Create Y with known strong components
        Y = np.zeros((116, 10))
        Y[0, :] = 10.0  # Atom 0 strongest
        Y[1, :] = 5.0   # Atom 1 second
        Y[2, :] = 2.0   # Atom 2 third
        
        tokens = tokenizer(Y)
        
        # First token should be atom 0
        assert tokens[0].startswith("<AT_0:")


class TestDirectionProjectionTokenizer:
    """Test suite for DirectionProjectionTokenizer."""
    
    def test_initialization(self):
        """Test tokenizer initialization with valid parameters."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles, top_m=5, n_levels=16)
        
        assert tokenizer.D == 37
        assert tokenizer.top_m == 5
        assert tokenizer.n_levels == 16
        assert tokenizer.metric == "correlation"
    
    def test_initialization_invalid_H_shape(self):
        """Test that 1D H raises ValueError."""
        H = np.random.rand(116)
        angles = list(range(37))
        with pytest.raises(ValueError, match="H must be 2D"):
            DirectionProjectionTokenizer(H, angles)
    
    def test_initialization_mismatched_angles(self):
        """Test that angle count mismatch raises ValueError."""
        H = np.random.rand(116, 37)
        angles = list(range(30))  # Wrong number of angles
        with pytest.raises(ValueError, match="doesn't match H columns"):
            DirectionProjectionTokenizer(H, angles)
    
    def test_top_m_exceeds_D(self):
        """Test that top_m is clamped to D when top_m > D."""
        H = np.random.rand(116, 10)
        angles = list(range(10))
        tokenizer = DirectionProjectionTokenizer(H, angles, top_m=20)
        assert tokenizer.top_m == 10  # Should be clamped to D
    
    def test_call_valid_input_correlation(self):
        """Test token generation with correlation metric."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(
            H, angles, top_m=3, n_levels=16, metric="correlation"
        )
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        assert len(tokens) == 3
        for token in tokens:
            assert token.startswith("<R_")
            assert token.endswith(">")
            assert ":" in token
    
    def test_call_valid_input_is_divergence(self):
        """Test token generation with IS divergence metric."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(
            H, angles, top_m=3, n_levels=16, metric="is_divergence"
        )
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        assert len(tokens) == 3
    
    def test_token_format(self):
        """Test that tokens follow <R_angle:level> format."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles, top_m=5, n_levels=16)
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        for token in tokens:
            # Extract angle and level
            token_body = token[3:-1]  # Remove <R_ and >
            parts = token_body.split(":")
            
            assert len(parts) == 2
            angle = int(parts[0])
            level = int(parts[1])
            
            assert angle in angles
            assert 0 <= level < 16
    
    def test_call_invalid_Y_shape(self):
        """Test that 1D Y raises ValueError."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles)
        Y = np.random.rand(116)
        
        with pytest.raises(ValueError, match="Y must be 2D"):
            tokenizer(Y)
    
    def test_call_mismatched_frequency_dimension(self):
        """Test that mismatched F dimension raises ValueError."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles)
        Y = np.random.rand(100, 189)  # Wrong F dimension
        
        with pytest.raises(ValueError, match="doesn't match"):
            tokenizer(Y)
    
    def test_override_top_m_in_call(self):
        """Test that top_m can be overridden in __call__."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles, top_m=5)
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y, top_m=3)
        assert len(tokens) == 3
    
    def test_quantization_levels_correlation(self):
        """Test quantization for correlation metric."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(
            H, angles, top_m=5, n_levels=10, metric="correlation"
        )
        Y = np.random.rand(116, 189)
        
        tokens = tokenizer(Y)
        
        for token in tokens:
            level = int(token.split(":")[1][:-1])
            assert 0 <= level < 10
    
    def test_unknown_metric(self):
        """Test that unknown metric raises ValueError."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(
            H, angles, metric="unknown"  # type: ignore
        )
        Y = np.random.rand(116, 189)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            tokenizer(Y)
    
    def test_deterministic_with_same_input(self):
        """Test that same input produces same tokens."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(H, angles, top_m=5)
        Y = np.random.rand(116, 189)
        
        tokens1 = tokenizer(Y)
        tokens2 = tokenizer(Y)
        
        assert tokens1 == tokens2
    
    def test_correlation_range(self):
        """Test that correlation scores are properly normalized."""
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        tokenizer = DirectionProjectionTokenizer(
            H, angles, top_m=5, n_levels=16, metric="correlation"
        )
        
        # Create Y similar to one H column
        Y = np.tile(H[:, 0:1], (1, 10))  # Should have high correlation with angle 0
        tokens = tokenizer(Y)
        
        # First token should be angle 0 with high level
        assert tokens[0].startswith("<R_000:")
        level = int(tokens[0].split(":")[1][:-1])
        assert level >= 10  # Should be high correlation


class TestIntegration:
    """Integration tests using both tokenizers together."""
    
    def test_combined_tokenization(self):
        """Test using both tokenizers on the same spectrogram."""
        # Create mock matrices
        W = np.random.rand(116, 64)
        H = np.random.rand(116, 37)
        angles = list(range(0, 181, 5))
        
        atom_tok = NMFAtomTokenizer(W, top_k=5)
        dir_tok = DirectionProjectionTokenizer(H, angles, top_m=3)
        
        Y = np.random.rand(116, 189)
        
        atom_tokens = atom_tok(Y)
        dir_tokens = dir_tok(Y)
        
        # Verify both produce tokens
        assert len(atom_tokens) == 5
        assert len(dir_tokens) == 3
        
        # Verify token formats are different
        assert atom_tokens[0].startswith("<AT_")
        assert dir_tokens[0].startswith("<R_")
    
    def test_realistic_scenario(self):
        """Test with realistic dimensions from actual data."""
        # Realistic dimensions from the codebase
        F = 116  # Frequency bins
        N = 189  # Time frames
        K = 64   # NMF atoms
        D = 37   # Directions (0° to 180° in 5° steps)
        
        W = np.random.rand(F, K)
        H = np.random.rand(F, D)
        angles = list(range(0, 181, 5))
        
        atom_tok = NMFAtomTokenizer(W, top_k=8, n_levels=16)
        dir_tok = DirectionProjectionTokenizer(H, angles, top_m=5, n_levels=16)
        
        # Simulate 10 spectrograms
        for _ in range(10):
            Y = np.random.rand(F, N)
            
            atom_tokens = atom_tok(Y)
            dir_tokens = dir_tok(Y)
            
            assert len(atom_tokens) == 8
            assert len(dir_tokens) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
