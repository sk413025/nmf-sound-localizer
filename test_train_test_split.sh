#!/bin/bash
# Test script for train/test split functionality
# This runs a short training to verify the split works correctly

cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer

echo "Testing Decision Transformer with Train/Test Split"
echo "=================================================="
echo ""
echo "Running 10 epochs to observe train vs test loss dynamics..."
echo ""

python -u scripts/dt_pointer_ldv.py \
  --traj_dir results/dt_traj_qk_kmeans \
  --out_dir results/dt_min_train_test_split_test \
  --epochs 10 \
  --batch_size 4 \
  --lr 3e-3 \
  --d_model 128 \
  --nhead 2 \
  --nlayers 1 \
  --device cpu \
  --distill_weight 0.5 \
  --distill_T 1.0 \
  --warmup_epochs 2 \
  --test_split 0.15 \
  --split_seed 42 \
  2>&1 | tee results/dt_min_train_test_split_test/run.log

echo ""
echo "=================================================="
echo "Test complete! Check results/dt_min_train_test_split_test/run.log"
echo ""
echo "Key things to look for:"
echo "  1. Train/Test split summary at the start"
echo "  2. Train loss vs Test loss comparison each epoch"
echo "  3. If test loss stops decreasing -> potential overfitting signal"
echo "  4. Best test loss epoch marker (✓)"
