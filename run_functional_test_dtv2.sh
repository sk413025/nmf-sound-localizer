#!/bin/bash
# Functional test for DTMinPointerV2: 10 epochs on full 111 samples
# Purpose: Validate convergence, compare to baseline, verify no overfitting
# Expected: Expert acc >50%, atom acc >40%, test loss within 20% of train loss

set -e  # Exit on error

# Setup environment
export PYTHONPATH=/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer:$PYTHONPATH

# Output directory
OUT_DIR="results/dt_v2_functional"
mkdir -p "$OUT_DIR"

echo "================================================================================"
echo "DTMinPointerV2 Functional Test"
echo "================================================================================"
echo "Training: 10 epochs on full 111 samples"
echo "Expected: Expert acc >50%, atom acc >40%"
echo "Output: $OUT_DIR"
echo ""

# Run training
python -u scripts/dt_pointer_ldv_v2.py \
    --traj_dir results/dt_traj_g_full \
    --out_dir "$OUT_DIR" \
    --epochs 10 \
    --batch_size 8 \
    --lr 3e-3 \
    --device cpu \
    --test_split 0.15 \
    --split_seed 42 \
    2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "================================================================================"
echo "Functional Test Complete"
echo "================================================================================"
echo "Results: $OUT_DIR/run.log"
echo "Checkpoints: $OUT_DIR/ckpt_best.pth, $OUT_DIR/ckpt_final.pth"
echo ""
echo "Verification steps:"
echo "1. Check test loss < 2.0 after 10 epochs"
echo "2. Check expert acc > 0.5 on test set"
echo "3. Check atom acc > 0.4 on test set"
echo "4. Check no NaN/Inf in logs"
echo "5. Compare to DTMinPointer baseline (if available)"
