#!/usr/bin/env python3
"""
Validation script for extended HF tokenizer vocabulary (Day 7).

This script validates:
1. Backward compatibility: Basic tokenizer still works
2. Extended vocab: NMF Atom and Direction Projection tokens are properly added
3. Token encoding/decoding: Multi-modal prompts can be tokenized correctly
4. Vocab size: Expected counts for each token type
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from doa_rl.hf.tokenizer import build_patch_tokenizer


def test_basic_tokenizer():
    """Test 1: Basic tokenizer (without extended vocab) still works."""
    print("=" * 70)
    print("Test 1: Basic Tokenizer (Backward Compatibility)")
    print("=" * 70)
    
    direction_angles = [80, 85, 90, 95, 100]
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=False,
    )
    
    # Check special tokens
    assert tokenizer.pad_token == "<PAD>"
    assert tokenizer.bos_token == "<BOS>"
    assert tokenizer.eos_token == "<EOS>"
    assert tokenizer.unk_token == "<UNK>"
    print("✅ Special tokens: <PAD>, <BOS>, <EOS>, <UNK>")
    
    # Check vocab size (4 special + 7*18*16 patch + 5 direction = 4 + 2016 + 5 = 2025)
    expected_size = 4 + (7 * 18 * 16) + len(direction_angles)
    actual_size = tokenizer.vocab_size
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
    print(f"✅ Vocab size: {actual_size} (4 special + 2016 patch + {len(direction_angles)} direction)")
    
    # Test encoding patch tokens
    prompt = "<BOS> <P_0_0_5> <P_0_1_8> <D_090>"
    encoded = tokenizer.encode(prompt)
    decoded = tokenizer.decode(encoded)
    print(f"✅ Encode/decode patch prompt: '{prompt}' -> {len(encoded)} tokens")
    
    # Verify direction tokens exist
    for angle in direction_angles:
        token = f"<D_{angle:03d}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, f"Direction token {token} not found"
    print(f"✅ All {len(direction_angles)} direction tokens present")
    
    print("\n✅ Test 1 PASSED: Basic tokenizer works correctly\n")
    return True


def test_extended_tokenizer():
    """Test 2: Extended tokenizer includes NMF Atom and Direction Projection tokens."""
    print("=" * 70)
    print("Test 2: Extended Tokenizer (Multi-Modal Vocabulary)")
    print("=" * 70)
    
    direction_angles = [80, 85, 90, 95, 100]
    n_atoms = 64
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=n_atoms,
    )
    
    # Check vocab size
    # 4 special + 2016 patch + 1024 atom (64*16) + 592 direction_proj (37*16) + 5 direction
    expected_size = 4 + (7 * 18 * 16) + (n_atoms * 16) + (37 * 16) + len(direction_angles)
    actual_size = tokenizer.vocab_size
    assert actual_size == expected_size, f"Expected {expected_size}, got {actual_size}"
    print(f"✅ Vocab size: {actual_size}")
    print(f"   - 4 special tokens")
    print(f"   - 2016 patch tokens (7 × 18 × 16)")
    print(f"   - {n_atoms * 16} NMF atom tokens ({n_atoms} atoms × 16 levels)")
    print(f"   - 592 direction projection tokens (37 angles × 16 levels)")
    print(f"   - {len(direction_angles)} direction tokens")
    
    # Test NMF Atom tokens
    atom_tokens = [
        "<AT_0:0>", "<AT_0:15>",  # First atom, min/max level
        "<AT_5:12>", "<AT_23:8>",  # Random atoms
        "<AT_63:7>",  # Last atom (63 = 64-1)
    ]
    for token in atom_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, f"Atom token {token} not found"
        decoded = tokenizer.convert_ids_to_tokens(token_id)
        assert decoded == token, f"Decode mismatch: {token} -> {decoded}"
    print(f"✅ NMF Atom tokens validated: {atom_tokens}")
    
    # Test Direction Projection tokens
    dir_proj_tokens = [
        "<R_000:0>", "<R_000:15>",  # 0°, min/max level
        "<R_090:14>", "<R_085:12>",  # Common angles
        "<R_180:3>",  # 180°
    ]
    for token in dir_proj_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        assert token_id != tokenizer.unk_token_id, f"Direction projection token {token} not found"
        decoded = tokenizer.convert_ids_to_tokens(token_id)
        assert decoded == token, f"Decode mismatch: {token} -> {decoded}"
    print(f"✅ Direction Projection tokens validated: {dir_proj_tokens}")
    
    # Test multi-modal prompt encoding
    multi_modal_prompt = (
        "<BOS> <R_090:14> <R_085:12> <AT_5:12> <AT_23:8> "
        "<P_0_0_5> <P_0_1_8> <D_090>"
    )
    encoded = tokenizer.encode(multi_modal_prompt)
    decoded = tokenizer.decode(encoded)
    print(f"✅ Multi-modal prompt encoded: {len(encoded)} tokens")
    print(f"   Input:  '{multi_modal_prompt}'")
    print(f"   Output: '{decoded}'")
    
    print("\n✅ Test 2 PASSED: Extended tokenizer includes all multi-modal tokens\n")
    return True


def test_token_type_coverage():
    """Test 3: Verify all token types are present and distinct."""
    print("=" * 70)
    print("Test 3: Token Type Coverage")
    print("=" * 70)
    
    direction_angles = list(range(80, 101))  # 21 angles
    n_atoms = 50
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=n_atoms,
    )
    
    vocab = tokenizer.get_vocab()
    
    # Count token types
    special_count = sum(1 for tok in vocab if tok in ["<PAD>", "<BOS>", "<EOS>", "<UNK>"])
    patch_count = sum(1 for tok in vocab if tok.startswith("<P_"))
    atom_count = sum(1 for tok in vocab if tok.startswith("<AT_"))
    dir_proj_count = sum(1 for tok in vocab if tok.startswith("<R_"))
    dir_count = sum(1 for tok in vocab if tok.startswith("<D_"))
    
    print(f"Token type counts:")
    print(f"  Special tokens:              {special_count:4d} (expected 4)")
    print(f"  Patch tokens:                {patch_count:4d} (expected 2016 = 7×18×16)")
    print(f"  NMF Atom tokens:             {atom_count:4d} (expected {n_atoms * 16} = {n_atoms}×16)")
    print(f"  Direction Projection tokens: {dir_proj_count:4d} (expected 592 = 37×16)")
    print(f"  Direction tokens:            {dir_count:4d} (expected {len(direction_angles)})")
    
    assert special_count == 4
    assert patch_count == 7 * 18 * 16
    assert atom_count == n_atoms * 16
    assert dir_proj_count == 37 * 16
    assert dir_count == len(direction_angles)
    
    print(f"\n✅ Total vocab size: {tokenizer.vocab_size}")
    print(f"   Sum check: {special_count} + {patch_count} + {atom_count} + {dir_proj_count} + {dir_count} = {special_count + patch_count + atom_count + dir_proj_count + dir_count}")
    assert tokenizer.vocab_size == special_count + patch_count + atom_count + dir_proj_count + dir_count
    
    print("\n✅ Test 3 PASSED: All token types accounted for\n")
    return True


def test_real_world_prompts():
    """Test 4: Encode/decode realistic multi-modal prompts."""
    print("=" * 70)
    print("Test 4: Real-World Multi-Modal Prompts")
    print("=" * 70)
    
    direction_angles = list(range(80, 101))
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    
    # Test case 1: Physics-first ordering (direction -> atom -> patch)
    prompt1 = (
        "<BOS> "
        "<R_090:15> <R_085:14> <R_095:13> <R_080:12> <R_100:11> "  # Direction projections
        "<AT_5:14> <AT_12:13> <AT_23:12> <AT_8:11> <AT_41:10> <AT_15:9> <AT_27:8> <AT_33:7> "  # Atoms
        "<P_0_0_5> <P_0_1_8> <P_1_0_12> <P_1_1_6> <P_2_0_9> "  # Patches
        "<D_090>"  # Target
    )
    enc1 = tokenizer.encode(prompt1)
    dec1 = tokenizer.decode(enc1)
    print(f"✅ Physics-first prompt: {len(enc1)} tokens")
    print(f"   Input:  {prompt1[:80]}...")
    print(f"   Output: {dec1[:80]}...")
    
    # Test case 2: ICL few-shot (context examples + query)
    prompt2 = (
        "<BOS> "
        # Example 1
        "<R_080:14> <AT_3:12> <P_0_0_7> <D_080> "
        # Example 2
        "<R_090:15> <AT_5:13> <P_0_1_9> <D_090> "
        # Example 3
        "<R_100:13> <AT_8:11> <P_1_0_8> <D_100> "
        # Query (no direction)
        "<R_095:14> <AT_6:12> <P_0_2_10>"
    )
    enc2 = tokenizer.encode(prompt2)
    dec2 = tokenizer.decode(enc2)
    print(f"✅ ICL few-shot prompt: {len(enc2)} tokens")
    print(f"   Input:  {prompt2[:80]}...")
    print(f"   Output: {dec2[:80]}...")
    
    # Test case 3: Minimal prompt (only essential tokens)
    prompt3 = "<BOS> <R_090:15> <AT_5:14> <P_0_0_5> <D_090>"
    enc3 = tokenizer.encode(prompt3)
    dec3 = tokenizer.decode(enc3)
    print(f"✅ Minimal prompt: {len(enc3)} tokens")
    print(f"   Input:  '{prompt3}'")
    print(f"   Output: '{dec3}'")
    assert prompt3 == dec3, "Minimal prompt should decode exactly"
    
    # Test case 4: Unknown token handling
    prompt4 = "<BOS> <R_090:15> <UNKNOWN_TOKEN> <AT_5:14> <D_090>"
    enc4 = tokenizer.encode(prompt4)
    dec4 = tokenizer.decode(enc4)
    print(f"✅ Unknown token handling: {len(enc4)} tokens")
    print(f"   Input:  '{prompt4}'")
    print(f"   Output: '{dec4}'")
    assert "<UNK>" in dec4, "Unknown tokens should map to <UNK>"
    
    print("\n✅ Test 4 PASSED: Real-world prompts encode/decode correctly\n")
    return True


def test_edge_cases():
    """Test 5: Edge cases and boundary conditions."""
    print("=" * 70)
    print("Test 5: Edge Cases and Boundary Conditions")
    print("=" * 70)
    
    # Test with minimal atoms
    tokenizer_min = build_patch_tokenizer(
        [90],
        enable_extended_vocab=True,
        n_atoms=1,
    )
    assert "<AT_0:0>" in tokenizer_min.get_vocab()
    assert "<AT_1:0>" not in tokenizer_min.get_vocab()  # Only 1 atom
    print("✅ Minimal atoms (n_atoms=1): Correct vocab")
    
    # Test with maximum atoms
    tokenizer_max = build_patch_tokenizer(
        [90],
        enable_extended_vocab=True,
        n_atoms=128,
    )
    assert "<AT_127:15>" in tokenizer_max.get_vocab()
    print("✅ Maximum atoms (n_atoms=128): Correct vocab")
    
    # Test boundary angles for direction projection
    tokenizer_angles = build_patch_tokenizer(
        [90],
        enable_extended_vocab=True,
        n_atoms=64,
    )
    assert "<R_000:0>" in tokenizer_angles.get_vocab()  # Min angle
    assert "<R_180:15>" in tokenizer_angles.get_vocab()  # Max angle
    assert "<R_090:8>" in tokenizer_angles.get_vocab()  # Middle angle
    assert "<R_001:0>" not in tokenizer_angles.get_vocab()  # Not 5-degree increment
    print("✅ Direction projection angle range: 0°-180° in 5° increments")
    
    # Test maximum level (15)
    assert "<AT_0:15>" in tokenizer_angles.get_vocab()
    assert "<R_090:15>" in tokenizer_angles.get_vocab()
    assert "<P_0_0_15>" in tokenizer_angles.get_vocab()
    print("✅ Maximum level (15): All token types support it")
    
    # Test empty prompt
    enc_empty = tokenizer_angles.encode("")
    dec_empty = tokenizer_angles.decode(enc_empty)
    print(f"✅ Empty prompt: '{dec_empty}' ({len(enc_empty)} tokens)")
    
    # Test BOS/EOS handling
    enc_bos = tokenizer_angles.encode("<BOS> <P_0_0_5>", add_special_tokens=False)
    enc_no_bos = tokenizer_angles.encode("<P_0_0_5>", add_special_tokens=False)
    print(f"✅ BOS handling: with BOS={len(enc_bos)} tokens, without={len(enc_no_bos)} tokens")
    
    print("\n✅ Test 5 PASSED: Edge cases handled correctly\n")
    return True


def main():
    """Run all validation tests."""
    print("\n" + "=" * 70)
    print("Hugging Face Tokenizer Extended Vocabulary Validation (Day 7)")
    print("=" * 70 + "\n")
    
    tests = [
        ("Basic Tokenizer", test_basic_tokenizer),
        ("Extended Tokenizer", test_extended_tokenizer),
        ("Token Type Coverage", test_token_type_coverage),
        ("Real-World Prompts", test_real_world_prompts),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error:")
            print(f"   {type(e).__name__}: {e}\n")
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    for name, status in results:
        status_symbol = "✅" if status == "PASS" else "❌"
        print(f"{status_symbol} {name:30s} {status}")
    
    passed = sum(1 for _, status in results if status == "PASS")
    total = len(results)
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests PASSED! Day 7 implementation is complete.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
