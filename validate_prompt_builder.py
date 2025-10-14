#!/usr/bin/env python3
"""
Validation script for MultiModalPromptBuilder (Day 3-4 implementation).

This script validates the prompt builder without requiring pytest,
using only standard library and numpy.

Tests:
1. Basic prompt building with all token types
2. Token ordering strategies
3. Token limits and budgets
4. ICL prompt construction
5. Context sampling strategies
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from doa_rl.features.prompt_builder import (
    MultiModalPromptBuilder,
    PromptConfig,
)


# Mock tokenizers for testing
class MockPatchTokenizer:
    def __call__(self, Y):
        return [f"<P_{i}_{j}_5>" for i in range(2) for j in range(3)]


class MockAtomTokenizer:
    def __call__(self, Y):
        return [f"<AT_{i}:10>" for i in [5, 12, 23, 41]]


class MockDirectionTokenizer:
    def __call__(self, Y, top_m=5):
        angles = [90, 85, 95, 80, 100]
        return [f"<R_{angle:03d}:14>" for angle in angles[:top_m]]


def test_basic_prompt_building():
    """Test 1: Basic prompt building with all token types."""
    print("\n" + "="*70)
    print("Test 1: Basic Prompt Building")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok)
    Y = np.random.rand(116, 189)
    
    prompt = builder.build_prompt(Y)
    
    print(f"Generated prompt ({len(prompt.split())} tokens):")
    print(prompt[:200] + "...")
    
    # Validate
    assert "<R_090:14>" in prompt, "Missing direction token"
    assert "<AT_5:10>" in prompt, "Missing atom token"
    assert "<P_0_0_5>" in prompt, "Missing patch token"
    
    print("✅ PASS: All token types present")
    return True


def test_ordering_strategies():
    """Test 2: Token ordering strategies."""
    print("\n" + "="*70)
    print("Test 2: Token Ordering Strategies")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    Y = np.random.rand(116, 189)
    
    strategies = ["physics_first", "structure_first", "patch_first", "interleaved"]
    
    for strategy in strategies:
        config = PromptConfig(ordering=strategy)
        builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
        prompt = builder.build_prompt(Y)
        tokens = prompt.split()
        
        print(f"\n{strategy}:")
        print(f"  First 5 tokens: {' '.join(tokens[:5])}")
        
        # Validate ordering
        if strategy == "physics_first":
            assert tokens[0].startswith("<R_"), f"Expected direction first, got {tokens[0]}"
        elif strategy == "structure_first":
            assert tokens[0].startswith("<AT_"), f"Expected atom first, got {tokens[0]}"
        elif strategy == "patch_first":
            assert tokens[0].startswith("<P_"), f"Expected patch first, got {tokens[0]}"
        elif strategy == "interleaved":
            # Check mix in first 10 tokens
            first_10_types = set(t.split("_")[0] for t in tokens[:10])
            assert len(first_10_types) >= 2, "Expected mixed token types"
        
        print(f"  ✅ Ordering correct")
    
    print("\n✅ PASS: All ordering strategies work correctly")
    return True


def test_token_limits():
    """Test 3: Token limits and budgets."""
    print("\n" + "="*70)
    print("Test 3: Token Limits and Budgets")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    Y = np.random.rand(116, 189)
    
    # Test max_tokens
    config = PromptConfig(max_tokens=10)
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
    prompt = builder.build_prompt(Y)
    tokens = prompt.split()
    print(f"\nmax_tokens=10: {len(tokens)} tokens generated")
    assert len(tokens) <= 10, f"Expected ≤10 tokens, got {len(tokens)}"
    print("✅ max_tokens limit respected")
    
    # Test max_patch_tokens
    config = PromptConfig(
        max_patch_tokens=3,
        use_atoms=False,
        use_directions=False,
    )
    builder = MultiModalPromptBuilder(patch_tok, config=config)
    prompt = builder.build_prompt(Y)
    tokens = prompt.split()
    print(f"\nmax_patch_tokens=3: {len(tokens)} tokens generated")
    assert len(tokens) <= 3, f"Expected ≤3 tokens, got {len(tokens)}"
    assert all(t.startswith("<P_") for t in tokens), "Expected only patch tokens"
    print("✅ max_patch_tokens limit respected")
    
    # Test atom_top_k
    config = PromptConfig(
        atom_top_k=2,
        use_patches=False,
        use_directions=False,
    )
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, config=config)
    prompt = builder.build_prompt(Y)
    tokens = prompt.split()
    print(f"\natom_top_k=2: {len(tokens)} tokens generated")
    assert len(tokens) == 2, f"Expected 2 tokens, got {len(tokens)}"
    assert all(t.startswith("<AT_") for t in tokens), "Expected only atom tokens"
    print("✅ atom_top_k limit respected")
    
    # Test direction_top_m
    config = PromptConfig(
        direction_top_m=3,
        use_patches=False,
        use_atoms=False,
    )
    builder = MultiModalPromptBuilder(patch_tok, None, dir_tok, config)
    prompt = builder.build_prompt(Y)
    tokens = prompt.split()
    print(f"\ndirection_top_m=3: {len(tokens)} tokens generated")
    assert len(tokens) == 3, f"Expected 3 tokens, got {len(tokens)}"
    assert all(t.startswith("<R_") for t in tokens), "Expected only direction tokens"
    print("✅ direction_top_m limit respected")
    
    print("\n✅ PASS: All token limits work correctly")
    return True


def test_icl_prompt_building():
    """Test 4: ICL prompt construction."""
    print("\n" + "="*70)
    print("Test 4: ICL Prompt Building")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok)
    
    # Create context examples
    context_Y1 = np.random.rand(116, 189)
    context_Y2 = np.random.rand(116, 189)
    query_Y = np.random.rand(116, 189)
    
    context_examples = [
        (context_Y1, "<D_045>"),
        (context_Y2, "<D_090>"),
    ]
    
    # Build ICL prompt without target
    prompt = builder.build_icl_prompt(query_Y, context_examples)
    print(f"\nICL prompt (no target): {len(prompt.split())} tokens")
    print(prompt[:200] + "...")
    
    assert "<D_045>" in prompt, "Missing context target 1"
    assert "<D_090>" in prompt, "Missing context target 2"
    print("✅ Context examples included")
    
    # Build ICL prompt with target
    prompt_with_target = builder.build_icl_prompt(
        query_Y,
        context_examples,
        target_token="<D_120>",
    )
    print(f"\nICL prompt (with target): {len(prompt_with_target.split())} tokens")
    
    assert prompt_with_target.endswith("<D_120>"), "Target not at end"
    print("✅ Target token appended correctly")
    
    print("\n✅ PASS: ICL prompt building works correctly")
    return True


def test_context_sampling():
    """Test 5: Context sampling strategies."""
    print("\n" + "="*70)
    print("Test 5: Context Sampling Strategies")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok)
    
    # Create mock dataset
    mock_dataset = [
        {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
        for angle in range(0, 180, 10)
    ]
    print(f"\nMock dataset: {len(mock_dataset)} samples (0° to 170° in 10° steps)")
    
    # Test random sampling
    context = builder.sample_context_examples(
        mock_dataset,
        query_idx=9,  # 90 degrees
        n_shots=3,
        strategy="random",
    )
    print(f"\nRandom sampling: {len(context)} examples")
    angles = [int(t.split("_")[1].split(">")[0]) for _, t in context]
    print(f"  Sampled angles: {angles}")
    assert len(context) == 3, f"Expected 3 examples, got {len(context)}"
    print("✅ Random sampling works")
    
    # Test nearest_angle sampling
    context = builder.sample_context_examples(
        mock_dataset,
        query_idx=9,  # 90 degrees
        n_shots=3,
        strategy="nearest_angle",
    )
    angles = [int(t.split("_")[1].split(">")[0]) for _, t in context]
    print(f"\nNearest angle sampling (query=90°): {angles}")
    # Should be close to 90
    for angle in angles:
        assert abs(angle - 90) <= 30, f"Angle {angle} too far from 90"
    print("✅ Nearest angle sampling works")
    
    # Test diverse_angle sampling
    context = builder.sample_context_examples(
        mock_dataset,
        query_idx=9,
        n_shots=3,
        strategy="diverse_angle",
    )
    angles = [int(t.split("_")[1].split(">")[0]) for _, t in context]
    print(f"\nDiverse angle sampling: {angles}")
    # All should be different
    assert len(set(angles)) == 3, "Expected all different angles"
    print("✅ Diverse angle sampling works")
    
    # Test exclude_same_angle
    duplicate_dataset = [
        {"Y": np.random.rand(116, 189), "angle_deg": 90.0}
        for _ in range(5)
    ] + [
        {"Y": np.random.rand(116, 189), "angle_deg": float(angle)}
        for angle in [30, 60, 120, 150]
    ]
    
    context = builder.sample_context_examples(
        duplicate_dataset,
        query_idx=0,  # 90 degrees
        n_shots=3,
        strategy="random",
        exclude_same_angle=True,
    )
    angles = [int(t.split("_")[1].split(">")[0]) for _, t in context]
    print(f"\nExclude same angle (query=90°): {angles}")
    assert 90 not in angles, "90° should be excluded"
    print("✅ Exclude same angle works")
    
    print("\n✅ PASS: All context sampling strategies work correctly")
    return True


def test_edge_cases():
    """Test 6: Edge cases and error handling."""
    print("\n" + "="*70)
    print("Test 6: Edge Cases and Error Handling")
    print("="*70)
    
    patch_tok = MockPatchTokenizer()
    atom_tok = MockAtomTokenizer()
    dir_tok = MockDirectionTokenizer()
    Y = np.random.rand(116, 189)
    
    # Test all disabled
    config = PromptConfig(
        use_patches=False,
        use_atoms=False,
        use_directions=False,
    )
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
    prompt = builder.build_prompt(Y)
    print(f"\nAll tokens disabled: '{prompt}'")
    assert prompt == "", "Expected empty prompt"
    print("✅ Empty prompt when all disabled")
    
    # Test auto-disable when tokenizers missing
    config = PromptConfig(use_atoms=True, use_directions=True)
    builder = MultiModalPromptBuilder(patch_tok, None, None, config)
    print(f"\nAuto-disable: atoms={builder.config.use_atoms}, directions={builder.config.use_directions}")
    assert builder.config.use_atoms is False, "Should auto-disable atoms"
    assert builder.config.use_directions is False, "Should auto-disable directions"
    print("✅ Auto-disable missing tokenizers works")
    
    # Test invalid ordering
    config = PromptConfig(ordering="invalid_ordering")
    builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok, config)
    try:
        builder.build_prompt(Y)
        assert False, "Should raise ValueError for invalid ordering"
    except ValueError as e:
        print(f"\n✅ Invalid ordering raises error: {e}")
    
    # Test invalid sampling strategy
    mock_dataset = [{"Y": np.random.rand(116, 189), "angle_deg": 90.0}]
    try:
        builder.sample_context_examples(
            mock_dataset,
            query_idx=0,
            n_shots=1,
            strategy="invalid_strategy",
        )
        assert False, "Should raise ValueError for invalid strategy"
    except ValueError as e:
        print(f"✅ Invalid sampling strategy raises error: {e}")
    
    print("\n✅ PASS: Edge cases handled correctly")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("MultiModalPromptBuilder Validation")
    print("Day 3-4 Implementation")
    print("="*70)
    
    tests = [
        test_basic_prompt_building,
        test_ordering_strategies,
        test_token_limits,
        test_icl_prompt_building,
        test_context_sampling,
        test_edge_cases,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(("PASS", test.__name__))
        except Exception as e:
            print(f"\n❌ FAIL: {test.__name__}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append(("FAIL", test.__name__))
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for status, name in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {name}: {status}")
    
    passed = sum(1 for s, _ in results if s == "PASS")
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! MultiModalPromptBuilder is ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
