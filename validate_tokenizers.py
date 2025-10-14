#!/usr/bin/env python3
"""
Quick validation script for extended tokenizers.
Runs basic sanity checks without requiring pytest.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from doa_rl.features.tokenizers_extended import (
    NMFAtomTokenizer,
    DirectionProjectionTokenizer,
)


def test_nmf_atom_tokenizer():
    """Test NMFAtomTokenizer basic functionality."""
    print("=" * 60)
    print("Testing NMFAtomTokenizer...")
    print("=" * 60)
    
    # Create mock W matrix
    F, K = 116, 64
    W = np.random.rand(F, K)
    
    # Initialize tokenizer
    tokenizer = NMFAtomTokenizer(W, top_k=8, n_levels=16)
    print(f"✓ Initialized tokenizer with W shape {W.shape}")
    print(f"  - K={tokenizer.K}, top_k={tokenizer.top_k}, n_levels={tokenizer.n_levels}")
    
    # Create mock spectrogram
    Y = np.random.rand(F, 189)
    
    # Generate tokens
    tokens = tokenizer(Y)
    print(f"\n✓ Generated {len(tokens)} tokens from Y shape {Y.shape}")
    print(f"  Sample tokens: {tokens[:3]}")
    
    # Verify token format
    for i, token in enumerate(tokens):
        assert token.startswith("<AT_"), f"Token {i} doesn't start with <AT_: {token}"
        assert token.endswith(">"), f"Token {i} doesn't end with >: {token}"
        assert ":" in token, f"Token {i} doesn't contain colon: {token}"
        
        # Parse token
        token_body = token[4:-1]  # Remove <AT_ and >
        parts = token_body.split(":")
        assert len(parts) == 2, f"Token {i} has wrong format: {token}"
        
        atom_id = int(parts[0])
        level = int(parts[1])
        assert 0 <= atom_id < K, f"Token {i} has invalid atom_id: {atom_id}"
        assert 0 <= level < 16, f"Token {i} has invalid level: {level}"
    
    print("✓ All tokens have valid format")
    print("\n✅ NMFAtomTokenizer tests PASSED\n")
    return True


def test_direction_projection_tokenizer():
    """Test DirectionProjectionTokenizer basic functionality."""
    print("=" * 60)
    print("Testing DirectionProjectionTokenizer...")
    print("=" * 60)
    
    # Create mock H matrix
    F, D = 116, 37
    H = np.random.rand(F, D)
    angles = list(range(0, 181, 5))
    
    # Test correlation metric
    print("\n[1] Testing with correlation metric...")
    tokenizer = DirectionProjectionTokenizer(
        H, angles, top_m=5, n_levels=16, metric="correlation"
    )
    print(f"✓ Initialized tokenizer with H shape {H.shape}")
    print(f"  - D={tokenizer.D}, top_m={tokenizer.top_m}, metric={tokenizer.metric}")
    
    # Create mock spectrogram
    Y = np.random.rand(F, 189)
    
    # Generate tokens
    tokens = tokenizer(Y)
    print(f"\n✓ Generated {len(tokens)} tokens from Y shape {Y.shape}")
    print(f"  Sample tokens: {tokens[:3]}")
    
    # Verify token format
    for i, token in enumerate(tokens):
        assert token.startswith("<R_"), f"Token {i} doesn't start with <R_: {token}"
        assert token.endswith(">"), f"Token {i} doesn't end with >: {token}"
        assert ":" in token, f"Token {i} doesn't contain colon: {token}"
        
        # Parse token
        token_body = token[3:-1]  # Remove <R_ and >
        parts = token_body.split(":")
        assert len(parts) == 2, f"Token {i} has wrong format: {token}"
        
        angle = int(parts[0])
        level = int(parts[1])
        assert angle in angles, f"Token {i} has invalid angle: {angle}"
        assert 0 <= level < 16, f"Token {i} has invalid level: {level}"
    
    print("✓ All tokens have valid format (correlation)")
    
    # Test IS divergence metric
    print("\n[2] Testing with IS divergence metric...")
    tokenizer_is = DirectionProjectionTokenizer(
        H, angles, top_m=3, n_levels=16, metric="is_divergence"
    )
    tokens_is = tokenizer_is(Y)
    print(f"✓ Generated {len(tokens_is)} tokens with IS divergence")
    print(f"  Sample tokens: {tokens_is}")
    
    # Test override top_m
    print("\n[3] Testing top_m override...")
    tokens_override = tokenizer(Y, top_m=3)
    assert len(tokens_override) == 3, f"Expected 3 tokens, got {len(tokens_override)}"
    print(f"✓ Override works: {len(tokens_override)} tokens")
    
    print("\n✅ DirectionProjectionTokenizer tests PASSED\n")
    return True


def test_integration():
    """Test using both tokenizers together."""
    print("=" * 60)
    print("Testing Integration (Both Tokenizers)...")
    print("=" * 60)
    
    # Create mock matrices
    F, K, D = 116, 64, 37
    W = np.random.rand(F, K)
    H = np.random.rand(F, D)
    angles = list(range(0, 181, 5))
    
    # Initialize both tokenizers
    atom_tok = NMFAtomTokenizer(W, top_k=5, n_levels=16)
    dir_tok = DirectionProjectionTokenizer(H, angles, top_m=3, n_levels=16)
    
    print(f"✓ Initialized both tokenizers")
    
    # Create mock spectrogram
    Y = np.random.rand(F, 189)
    
    # Generate tokens from both
    atom_tokens = atom_tok(Y)
    dir_tokens = dir_tok(Y)
    
    print(f"\n✓ Generated tokens:")
    print(f"  - Atom tokens: {len(atom_tokens)} -> {atom_tokens[:2]}")
    print(f"  - Direction tokens: {len(dir_tokens)} -> {dir_tokens}")
    
    # Combine into a multi-modal prompt
    all_tokens = dir_tokens + atom_tokens
    prompt = " ".join(all_tokens)
    
    print(f"\n✓ Combined multi-modal prompt ({len(all_tokens)} tokens):")
    print(f"  {prompt[:100]}...")
    
    print("\n✅ Integration test PASSED\n")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("EXTENDED TOKENIZERS VALIDATION")
    print("=" * 60 + "\n")
    
    tests = [
        test_nmf_atom_tokenizer,
        test_direction_projection_tokenizer,
        test_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"\n❌ Test {test.__name__} FAILED:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
