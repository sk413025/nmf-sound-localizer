#!/usr/bin/env python3
"""
Demo script for extended HF tokenizer with real multi-modal prompts (Day 7).

This demonstrates:
1. Basic tokenizer vs Extended tokenizer vocabulary size comparison
2. Tokenizing multi-modal prompts with Direction Projection + NMF Atom + Patch tokens
3. ICL few-shot prompt construction
4. Token budget analysis and compression strategies
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from doa_rl.hf.tokenizer import build_patch_tokenizer


def demo_vocab_comparison():
    """Demo 1: Compare basic vs extended vocabulary sizes."""
    print("=" * 80)
    print("Demo 1: Vocabulary Size Comparison")
    print("=" * 80)
    
    direction_angles = list(range(80, 101))  # 21 angles (80°-100° in 1° increments)
    
    # Basic tokenizer (patch + direction only)
    tokenizer_basic = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=False,
    )
    
    # Extended tokenizer (multi-modal)
    tokenizer_extended = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    
    print(f"\n📊 Basic Tokenizer (Patch + Direction only):")
    print(f"   Vocab size: {tokenizer_basic.vocab_size}")
    print(f"   - Special tokens:     4")
    print(f"   - Patch tokens:       {7 * 18 * 16:4d}  (7 freq × 18 time × 16 levels)")
    print(f"   - Direction tokens:   {len(direction_angles):4d}  (<D_080> to <D_100>)")
    
    print(f"\n📊 Extended Tokenizer (Multi-Modal):")
    print(f"   Vocab size: {tokenizer_extended.vocab_size}")
    print(f"   - Special tokens:              4")
    print(f"   - Patch tokens:                {7 * 18 * 16:4d}  (7 freq × 18 time × 16 levels)")
    print(f"   - NMF Atom tokens:             {64 * 16:4d}  (64 atoms × 16 levels)")
    print(f"   - Direction Projection tokens: {37 * 16:4d}  (37 angles [0°-180°/5°] × 16 levels)")
    print(f"   - Direction tokens:            {len(direction_angles):4d}  (<D_080> to <D_100>)")
    
    expansion = tokenizer_extended.vocab_size - tokenizer_basic.vocab_size
    expansion_pct = (expansion / tokenizer_basic.vocab_size) * 100
    print(f"\n🔄 Expansion: +{expansion} tokens ({expansion_pct:.1f}% increase)")
    print(f"   Multi-modal tokens: {64 * 16} (atoms) + {37 * 16} (directions) = {64 * 16 + 37 * 16}")
    
    print("\n✅ Demo 1 Complete\n")
    return tokenizer_basic, tokenizer_extended


def demo_multimodal_prompt():
    """Demo 2: Tokenize a multi-modal prompt with all token types."""
    print("=" * 80)
    print("Demo 2: Multi-Modal Prompt Tokenization")
    print("=" * 80)
    
    direction_angles = [80, 85, 90, 95, 100]
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    
    # Construct a multi-modal prompt with all token types
    # Order: BOS -> Direction Projections -> NMF Atoms -> Patches -> Direction Target
    prompt = (
        "<BOS> "
        # Direction Projection tokens (physics prior)
        "<R_090:15> <R_085:14> <R_095:13> <R_080:12> <R_100:11> "
        # NMF Atom tokens (spectral structure)
        "<AT_5:14> <AT_12:13> <AT_23:12> <AT_8:11> <AT_41:10> <AT_15:9> <AT_27:8> <AT_33:7> "
        # Patch tokens (fine-grained details)
        "<P_0_0_5> <P_0_1_8> <P_1_0_12> <P_1_1_6> <P_2_0_9> <P_2_1_11> <P_3_0_7> <P_3_1_10> "
        # Direction target
        "<D_090>"
    )
    
    print(f"\n📝 Multi-Modal Prompt:")
    print(f"   {prompt}\n")
    
    # Tokenize
    encoded = tokenizer.encode(prompt, add_special_tokens=False)
    decoded = tokenizer.decode(encoded)
    
    print(f"🔢 Tokenization Results:")
    print(f"   Input length:  {len(prompt)} characters")
    print(f"   Token count:   {len(encoded)} tokens")
    print(f"   Token IDs:     {encoded[:10]}... (showing first 10)")
    print(f"   Decoded:       {decoded[:80]}... (showing first 80 chars)")
    
    # Token type breakdown
    tokens = prompt.split()
    bos_count = sum(1 for t in tokens if t == "<BOS>")
    dir_proj_count = sum(1 for t in tokens if t.startswith("<R_"))
    atom_count = sum(1 for t in tokens if t.startswith("<AT_"))
    patch_count = sum(1 for t in tokens if t.startswith("<P_"))
    dir_count = sum(1 for t in tokens if t.startswith("<D_"))
    
    print(f"\n📊 Token Type Distribution:")
    print(f"   BOS:                   {bos_count:2d}")
    print(f"   Direction Projections: {dir_proj_count:2d}  ({dir_proj_count / len(tokens) * 100:5.1f}%)")
    print(f"   NMF Atoms:             {atom_count:2d}  ({atom_count / len(tokens) * 100:5.1f}%)")
    print(f"   Patches:               {patch_count:2d}  ({patch_count / len(tokens) * 100:5.1f}%)")
    print(f"   Direction Target:      {dir_count:2d}  ({dir_count / len(tokens) * 100:5.1f}%)")
    print(f"   Total:                 {len(tokens):2d}")
    
    print("\n✅ Demo 2 Complete\n")
    return tokenizer


def demo_icl_fewshot():
    """Demo 3: ICL few-shot prompt construction."""
    print("=" * 80)
    print("Demo 3: In-Context Learning Few-Shot Prompts")
    print("=" * 80)
    
    direction_angles = list(range(80, 101))
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    
    # Example context (3-shot)
    examples = [
        {
            "angle": 80,
            "dir_proj": "<R_080:15> <R_075:13> <R_085:12>",
            "atoms": "<AT_3:14> <AT_12:11> <AT_5:9>",
            "patches": "<P_0_0_7> <P_0_1_5> <P_1_0_8>",
        },
        {
            "angle": 90,
            "dir_proj": "<R_090:15> <R_085:14> <R_095:13>",
            "atoms": "<AT_5:15> <AT_8:12> <AT_15:10>",
            "patches": "<P_0_0_9> <P_0_1_11> <P_1_0_6>",
        },
        {
            "angle": 100,
            "dir_proj": "<R_100:15> <R_095:14> <R_105:12>",
            "atoms": "<AT_8:14> <AT_23:11> <AT_12:9>",
            "patches": "<P_0_0_8> <P_0_1_10> <P_1_0_7>",
        },
    ]
    
    # Query (predict 95°)
    query = {
        "dir_proj": "<R_095:15> <R_090:14> <R_100:13>",
        "atoms": "<AT_6:14> <AT_10:12> <AT_18:10>",
        "patches": "<P_0_0_10> <P_0_1_12> <P_1_0_9>",
    }
    
    # Construct few-shot prompt
    prompt_parts = ["<BOS>"]
    
    # Add examples (with labels)
    for ex in examples:
        context = f"{ex['dir_proj']} {ex['atoms']} {ex['patches']}"
        label = f"<D_{ex['angle']:03d}>"
        prompt_parts.append(f"{context} {label}")
    
    # Add query (no label)
    query_context = f"{query['dir_proj']} {query['atoms']} {query['patches']}"
    prompt_parts.append(query_context)
    
    icl_prompt = " ".join(prompt_parts)
    
    print(f"\n📚 Few-Shot ICL Prompt (3-shot):\n")
    print(f"   Context Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"      {i}. Angle={ex['angle']:3d}° → {ex['dir_proj'][:30]}... <D_{ex['angle']:03d}>")
    print(f"\n   Query:")
    print(f"      ?. Angle=??? → {query['dir_proj'][:30]}... [predict]")
    
    # Tokenize
    encoded = tokenizer.encode(icl_prompt, add_special_tokens=False)
    print(f"\n🔢 Tokenization:")
    print(f"   Prompt length: {len(icl_prompt)} characters")
    print(f"   Token count:   {len(encoded)} tokens")
    print(f"   Avg tokens/example: {len(encoded) / (len(examples) + 1):.1f}")
    
    # Show first 100 chars of prompt
    print(f"\n📝 Prompt Preview:")
    print(f"   {icl_prompt[:120]}...")
    print(f"   ...")
    print(f"   ...{icl_prompt[-80:]}")
    
    print("\n✅ Demo 3 Complete\n")
    return icl_prompt


def demo_token_budget_strategies():
    """Demo 4: Token budget analysis and compression strategies."""
    print("=" * 80)
    print("Demo 4: Token Budget Analysis")
    print("=" * 80)
    
    direction_angles = list(range(80, 101))
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    
    # Full multi-modal prompt (no budget constraints)
    full_prompt = (
        "<BOS> "
        "<R_090:15> <R_085:14> <R_095:13> <R_080:12> <R_100:11> "  # 5 dir proj
        "<AT_5:14> <AT_12:13> <AT_23:12> <AT_8:11> <AT_41:10> <AT_15:9> <AT_27:8> <AT_33:7> "  # 8 atoms
        "<P_0_0_5> <P_0_1_8> <P_1_0_12> <P_1_1_6> <P_2_0_9> <P_2_1_11> <P_3_0_7> <P_3_1_10> "
        "<P_4_0_6> <P_4_1_9> <P_5_0_8> <P_5_1_10> <P_6_0_7> <P_6_1_11> "  # 14 patches
        "<D_090>"
    )
    
    # Compressed prompts with different budgets
    prompts = {
        "Full (No Budget)": full_prompt,
        "High Budget (50 tokens)": (
            "<BOS> "
            "<R_090:15> <R_085:14> <R_095:13> "  # Top 3 dir proj
            "<AT_5:14> <AT_12:13> <AT_23:12> <AT_8:11> <AT_41:10> "  # Top 5 atoms
            "<P_0_0_5> <P_0_1_8> <P_1_0_12> <P_1_1_6> <P_2_0_9> "  # Top 5 patches
            "<D_090>"
        ),
        "Medium Budget (20 tokens)": (
            "<BOS> "
            "<R_090:15> <R_085:14> "  # Top 2 dir proj
            "<AT_5:14> <AT_12:13> <AT_23:12> "  # Top 3 atoms
            "<P_0_0_5> <P_0_1_8> <P_1_0_12> "  # Top 3 patches
            "<D_090>"
        ),
        "Low Budget (10 tokens)": (
            "<BOS> "
            "<R_090:15> "  # Top 1 dir proj
            "<AT_5:14> <AT_12:13> "  # Top 2 atoms
            "<P_0_0_5> <P_0_1_8> "  # Top 2 patches
            "<D_090>"
        ),
        "Minimal (Direction Only)": (
            "<BOS> "
            "<R_090:15> <R_085:14> <R_095:13> <R_080:12> <R_100:11> "
            "<D_090>"
        ),
    }
    
    print(f"\n📊 Token Budget Comparison:\n")
    print(f"{'Strategy':<30s} {'Tokens':>8s} {'Dir Proj':>10s} {'Atoms':>8s} {'Patches':>8s} {'Compression':>12s}")
    print("-" * 80)
    
    baseline_tokens = len(tokenizer.encode(full_prompt, add_special_tokens=False))
    
    for name, prompt in prompts.items():
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        token_list = prompt.split()
        
        dir_proj_count = sum(1 for t in token_list if t.startswith("<R_"))
        atom_count = sum(1 for t in token_list if t.startswith("<AT_"))
        patch_count = sum(1 for t in token_list if t.startswith("<P_"))
        
        compression = f"{baseline_tokens / len(tokens):.1f}x" if len(tokens) > 0 else "N/A"
        
        print(f"{name:<30s} {len(tokens):>8d} {dir_proj_count:>10d} {atom_count:>8d} {patch_count:>8d} {compression:>12s}")
    
    print(f"\n💡 Recommendations:")
    print(f"   - Full prompt: Use when training data is abundant")
    print(f"   - High budget: Balanced performance and efficiency")
    print(f"   - Medium budget: Good for real-time inference")
    print(f"   - Low budget: Minimal overhead, physics-guided")
    print(f"   - Direction only: Baseline for ablation studies")
    
    print("\n✅ Demo 4 Complete\n")


def demo_real_world_workflow():
    """Demo 5: Real-world workflow simulation."""
    print("=" * 80)
    print("Demo 5: Real-World Training Workflow Simulation")
    print("=" * 80)
    
    direction_angles = list(range(80, 101))
    
    print(f"\n📋 Workflow Steps:\n")
    
    # Step 1: Create basic tokenizer for baseline
    print(f"1️⃣  Create baseline tokenizer (patch-only)...")
    tokenizer_baseline = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=False,
    )
    print(f"    ✅ Vocab size: {tokenizer_baseline.vocab_size}")
    
    # Step 2: Create extended tokenizer for multi-modal training
    print(f"\n2️⃣  Create extended tokenizer (multi-modal)...")
    tokenizer_extended = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=64,
    )
    print(f"    ✅ Vocab size: {tokenizer_extended.vocab_size}")
    
    # Step 3: Generate sample prompts
    print(f"\n3️⃣  Generate sample prompts for batch processing...")
    
    baseline_prompts = [
        "<BOS> <P_0_0_5> <P_0_1_8> <P_1_0_12> <D_090>",
        "<BOS> <P_0_0_7> <P_0_1_6> <P_1_0_9> <D_085>",
        "<BOS> <P_0_0_9> <P_0_1_11> <P_1_0_8> <D_095>",
    ]
    
    multimodal_prompts = [
        "<BOS> <R_090:15> <AT_5:14> <P_0_0_5> <P_0_1_8> <D_090>",
        "<BOS> <R_085:14> <AT_3:13> <P_0_0_7> <P_0_1_6> <D_085>",
        "<BOS> <R_095:15> <AT_6:14> <P_0_0_9> <P_0_1_11> <D_095>",
    ]
    
    print(f"    ✅ Generated {len(baseline_prompts)} baseline prompts")
    print(f"    ✅ Generated {len(multimodal_prompts)} multi-modal prompts")
    
    # Step 4: Tokenize batches
    print(f"\n4️⃣  Tokenize batches...")
    
    baseline_encoded = [
        tokenizer_baseline.encode(p, add_special_tokens=False) 
        for p in baseline_prompts
    ]
    multimodal_encoded = [
        tokenizer_extended.encode(p, add_special_tokens=False) 
        for p in multimodal_prompts
    ]
    
    baseline_avg = np.mean([len(enc) for enc in baseline_encoded])
    multimodal_avg = np.mean([len(enc) for enc in multimodal_encoded])
    
    print(f"    Baseline:    Avg {baseline_avg:.1f} tokens/sample")
    print(f"    Multi-modal: Avg {multimodal_avg:.1f} tokens/sample")
    print(f"    Overhead:    +{multimodal_avg - baseline_avg:.1f} tokens/sample ({(multimodal_avg - baseline_avg) / baseline_avg * 100:.1f}%)")
    
    # Step 5: Compare model input shapes
    print(f"\n5️⃣  Model input shape comparison...")
    
    max_baseline_len = max(len(enc) for enc in baseline_encoded)
    max_multimodal_len = max(len(enc) for enc in multimodal_encoded)
    
    print(f"    Baseline:    Max sequence length = {max_baseline_len}")
    print(f"    Multi-modal: Max sequence length = {max_multimodal_len}")
    print(f"    Memory overhead: ~{(max_multimodal_len - max_baseline_len) / max_baseline_len * 100:.1f}% increase")
    
    # Step 6: Training implications
    print(f"\n6️⃣  Training implications...")
    print(f"    🔹 Batch size: May need to reduce by ~{(multimodal_avg - baseline_avg) / baseline_avg * 100:.0f}% to fit GPU memory")
    print(f"    🔹 Training time: +{(multimodal_avg - baseline_avg) / baseline_avg * 100:.0f}% due to longer sequences")
    print(f"    🔹 Benefits: Richer context, better few-shot learning, physics-guided predictions")
    
    print("\n✅ Demo 5 Complete\n")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 80)
    print("Extended HF Tokenizer Demo - Multi-Modal ICL Support (Day 7)")
    print("=" * 80 + "\n")
    
    demos = [
        ("Vocabulary Comparison", demo_vocab_comparison),
        ("Multi-Modal Prompt", demo_multimodal_prompt),
        ("ICL Few-Shot", demo_icl_fewshot),
        ("Token Budget Strategies", demo_token_budget_strategies),
        ("Real-World Workflow", demo_real_world_workflow),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            demo_func()
        except Exception as e:
            print(f"\n❌ {name} failed with error:")
            print(f"   {type(e).__name__}: {e}\n")
            return 1
    
    # Summary
    print("=" * 80)
    print("Demo Summary")
    print("=" * 80)
    print(f"✅ All {len(demos)} demonstrations completed successfully!")
    print(f"\n🎯 Key Takeaways:")
    print(f"   1. Extended vocab adds ~1,600 tokens (atoms + direction projections)")
    print(f"   2. Multi-modal prompts increase sequence length by ~20-50%")
    print(f"   3. Token budget strategies allow flexible compression (1x to 10x)")
    print(f"   4. ICL few-shot prompts enable context-aware predictions")
    print(f"   5. Backward compatibility maintained with enable_extended_vocab flag")
    
    print(f"\n🚀 Next Steps (Day 8-9):")
    print(f"   - Modify training scripts to support --use-multi-modal flag")
    print(f"   - Load W (NMF) and H (transfer function) matrices")
    print(f"   - Replace DoADataset with DoAICLDataset")
    print(f"   - Add token budget controls (--max-tokens)")
    
    print(f"\n📚 Implementation Files:")
    print(f"   - doa_rl/hf/tokenizer.py (updated)")
    print(f"   - doa_rl/features/tokenizers_extended.py (NMF/Direction tokenizers)")
    print(f"   - doa_rl/features/prompt_builder.py (prompt orchestration)")
    print(f"   - doa_rl/data.py (DoAICLDataset)")
    
    print("\n" + "=" * 80 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
