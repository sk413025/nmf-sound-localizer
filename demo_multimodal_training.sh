#!/bin/bash
# Demo: Multi-Modal Training Workflow
# Demonstrates how to use the newly integrated multi-modal ICL system

set -e  # Exit on error

echo "=========================================================================="
echo " Multi-Modal ICL Training Workflow Demo"
echo "=========================================================================="
echo ""
echo "This script demonstrates the complete training pipeline with multi-modal"
echo "tokenization (Direction + Atom + Patch tokens)."
echo ""
echo "Prerequisites:"
echo "  - Data directory with audio files"
echo "  - W matrix: doa_normalized_config_c_corrected/models/usm.pth"
echo "  - H matrix: h_matrix_normalized_original_to_box.pth"
echo ""
echo "=========================================================================="

# Configuration
DATA_ROOT="${DATA_ROOT:-doa_normalized_config_c_corrected}"
S_ROOT="${S_ROOT:-doa_normalized_config_c_corrected}"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
H_PATH="h_matrix_normalized_original_to_box.pth"
RESULTS_DIR="results/multimodal_demo"

# Check prerequisites
if [ ! -f "$W_PATH" ]; then
    echo "❌ Error: W matrix not found at $W_PATH"
    exit 1
fi

if [ ! -f "$H_PATH" ]; then
    echo "❌ Error: H matrix not found at $H_PATH"
    exit 1
fi

echo "✓ Prerequisites check passed"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "=========================================================================="
echo " Step 1: Train Reward Model (Multi-Modal)"
echo "=========================================================================="
echo ""

python scripts/train_reward_model_lora.py \
    --data-root "$DATA_ROOT" \
    --tf-path "$H_PATH" \
    --w-path "$W_PATH" \
    --s-root "$S_ROOT" \
    --use-multi-modal \
    --token-ordering physics_first \
    --max-tokens 200 \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --rm-epochs 2 \
    --batch-size 4 \
    --max-samples 10 \
    --lora-r 8 \
    --lora-alpha 16 \
    --out "$RESULTS_DIR/rm_multimodal"

echo ""
echo "✓ Step 1 completed: RM trained with multi-modal prompts"
echo "  - Adapters: $RESULTS_DIR/rm_multimodal_adapters/"
echo "  - Heads: $RESULTS_DIR/rm_multimodal_heads.pt"
echo ""

echo "=========================================================================="
echo " Step 2: Train SFT Policy (Multi-Modal)"
echo "=========================================================================="
echo ""

python scripts/train_sft_policy_with_rm.py \
    --data-root "$DATA_ROOT" \
    --rm-adapters "$RESULTS_DIR/rm_multimodal_adapters" \
    --rm-heads "$RESULTS_DIR/rm_multimodal_heads.pt" \
    --use-multi-modal \
    --w-path "$W_PATH" \
    --tf-path "$H_PATH" \
    --token-ordering physics_first \
    --max-tokens 200 \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --epochs 2 \
    --batch-size 4 \
    --max-samples 10 \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --out "$RESULTS_DIR/sft_multimodal"

echo ""
echo "✓ Step 2 completed: SFT policy trained with multi-modal prompts"
echo "  - Adapters: $RESULTS_DIR/sft_multimodal_policy_adapters/"
echo "  - Heads: $RESULTS_DIR/sft_multimodal_policy_heads.pt"
echo ""

echo "=========================================================================="
echo " Step 3: Train PPO Policy (Multi-Modal, optional)"
echo "=========================================================================="
echo ""

python scripts/train_trl_ppo_with_rm.py \
    --data-root "$DATA_ROOT" \
    --rm-adapters "$RESULTS_DIR/rm_multimodal_adapters" \
    --rm-heads "$RESULTS_DIR/rm_multimodal_heads.pt" \
    --sft-policy-adapters "$RESULTS_DIR/sft_multimodal_policy_adapters" \
    --sft-policy-heads "$RESULTS_DIR/sft_multimodal_policy_heads.pt" \
    --use-multi-modal \
    --w-path "$W_PATH" \
    --tf-path "$H_PATH" \
    --token-ordering physics_first \
    --max-tokens 200 \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --epochs 1 \
    --batch-size 2 \
    --ppo-epochs 1 \
    --max-samples 10

echo ""
echo "✓ Step 3 completed: PPO training with multi-modal prompts"
echo ""

echo "=========================================================================="
echo " ✅ Multi-Modal Training Workflow Completed!"
echo "=========================================================================="
echo ""
echo "Summary:"
echo "  - RM trained with physics_first token ordering"
echo "  - SFT policy learned from RM-guided teacher"
echo "  - PPO policy optimized with RM rewards"
echo ""
echo "Key Multi-Modal Features Used:"
echo "  ✓ Direction Projection tokens (top-5, from H matrix)"
echo "  ✓ NMF Atom tokens (top-8, from W matrix)"
echo "  ✓ Patch tokens (fine details)"
echo "  ✓ Physics-first ordering (Direction → Atom → Patch)"
echo "  ✓ Token budget control (200 max tokens)"
echo ""
echo "Next Steps:"
echo "  1. Compare with baseline (patch-only) training"
echo "  2. Try different token orderings (structure_first, etc.)"
echo "  3. Enable ICL mode with --icl-mode --n-shots 3"
echo "  4. Evaluate policy accuracy and attention weights"
echo ""
echo "=========================================================================="
