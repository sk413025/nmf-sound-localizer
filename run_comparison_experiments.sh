#!/bin/bash
# Comparison Experiments: Baseline vs Multi-Modal
# Tests different configurations to validate multi-modal improvements

set -e

echo "=========================================================================="
echo " Comparison Experiments: Baseline vs Multi-Modal"
echo "=========================================================================="
echo ""

# Configuration
DATA_ROOT="${DATA_ROOT:-doa_normalized_config_c_corrected}"
S_ROOT="${S_ROOT:-doa_normalized_config_c_corrected}"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
H_PATH="h_matrix_normalized_original_to_box.pth"
RESULTS_BASE="results/comparison"

# Experiment parameters
EPOCHS=10
SAMPLES=100
BATCH_SIZE=8

mkdir -p "$RESULTS_BASE"

echo "Experiment Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Max Samples: $SAMPLES"
echo "  - Batch Size: $BATCH_SIZE"
echo ""

# ============================================================================
# Experiment A: Baseline (Patch-only)
# ============================================================================
echo "=========================================================================="
echo " Experiment A: Baseline (Patch-only tokens)"
echo "=========================================================================="
echo ""

python scripts/train_reward_model_lora.py \
    --data-root "$DATA_ROOT" \
    --tf-path "$H_PATH" \
    --w-path "$W_PATH" \
    --s-root "$S_ROOT" \
    --K 3 \
    --rm-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$SAMPLES" \
    --out "$RESULTS_BASE/baseline_rm"

echo ""
echo "✓ Experiment A completed"
echo ""

# ============================================================================
# Experiment B: Multi-Modal (Physics-first)
# ============================================================================
echo "=========================================================================="
echo " Experiment B: Multi-Modal (physics_first ordering)"
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
    --rm-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$SAMPLES" \
    --out "$RESULTS_BASE/multimodal_physics_rm"

echo ""
echo "✓ Experiment B completed"
echo ""

# ============================================================================
# Experiment C: Multi-Modal (Structure-first)
# ============================================================================
echo "=========================================================================="
echo " Experiment C: Multi-Modal (structure_first ordering)"
echo "=========================================================================="
echo ""

python scripts/train_reward_model_lora.py \
    --data-root "$DATA_ROOT" \
    --tf-path "$H_PATH" \
    --w-path "$W_PATH" \
    --s-root "$S_ROOT" \
    --use-multi-modal \
    --token-ordering structure_first \
    --max-tokens 200 \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --rm-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$SAMPLES" \
    --out "$RESULTS_BASE/multimodal_structure_rm"

echo ""
echo "✓ Experiment C completed"
echo ""

# ============================================================================
# Experiment D: Multi-Modal with ICL (3-shot, nearest)
# ============================================================================
echo "=========================================================================="
echo " Experiment D: Multi-Modal + ICL (3-shot, nearest sampling)"
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
    --icl-mode \
    --n-shots 3 \
    --context-strategy nearest \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --rm-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$SAMPLES" \
    --out "$RESULTS_BASE/multimodal_icl_3shot_rm"

echo ""
echo "✓ Experiment D completed"
echo ""

# ============================================================================
# Experiment E: Token Budget Ablation (compressed)
# ============================================================================
echo "=========================================================================="
echo " Experiment E: Multi-Modal with Token Budget (50 tokens)"
echo "=========================================================================="
echo ""

python scripts/train_reward_model_lora.py \
    --data-root "$DATA_ROOT" \
    --tf-path "$H_PATH" \
    --w-path "$W_PATH" \
    --s-root "$S_ROOT" \
    --use-multi-modal \
    --token-ordering physics_first \
    --max-tokens 50 \
    --top-k-atoms 8 \
    --top-m-directions 5 \
    --n-atoms 50 \
    --K 3 \
    --rm-epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-samples "$SAMPLES" \
    --out "$RESULTS_BASE/multimodal_budget50_rm"

echo ""
echo "✓ Experiment E completed"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================================================="
echo " ✅ All Comparison Experiments Completed!"
echo "=========================================================================="
echo ""
echo "Experiments run:"
echo "  A. Baseline (Patch-only)"
echo "  B. Multi-Modal (physics_first)"
echo "  C. Multi-Modal (structure_first)"
echo "  D. Multi-Modal + ICL (3-shot, nearest)"
echo "  E. Multi-Modal + Token Budget (50 tokens)"
echo ""
echo "Results saved to: $RESULTS_BASE/"
echo ""
echo "Next Steps:"
echo "  1. Evaluate each model's performance:"
echo "     - Top-1 accuracy"
echo "     - Top-K recall"
echo "     - RM score correlation"
echo ""
echo "  2. Analyze attention weights:"
echo "     - Do Direction tokens get higher attention?"
echo "     - Does token ordering affect attention patterns?"
echo ""
echo "  3. Compare training metrics:"
echo "     - Loss convergence speed"
echo "     - Final loss values"
echo ""
echo "  4. Visualization:"
echo "     - Plot accuracy curves"
echo "     - Heatmap of attention weights"
echo "     - Token distribution analysis"
echo ""
echo "=========================================================================="
