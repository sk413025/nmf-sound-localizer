#!/bin/bash
# Phase 2: Hard Gumbel-Softmax + Atom Masking Training
# Background: Phase 1 (hard Gumbel only) achieved 97% accuracy but only 21.7% atom diversity
# Motivation: Model needs to learn to avoid selecting the same atom across steps
# Purpose: Train with hard=True + atom masking to force selection of different atoms
# Expected: Similar accuracy (~97%) with MUCH better atom diversity (>80%)

set -e

export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/mdp-decision-transformer

OUT_DIR="results/qk_phase2_hard_gumbel_masking_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"

echo "Phase 2: QK Training with Hard Gumbel-Softmax + Atom Masking"
echo "Output directory: $OUT_DIR"
echo ""
echo "Key changes from Phase 1:"
echo "  - Added atom masking during training"
echo "  - Step 0: Normal selection"
echo "  - Step 1: Masks out atoms selected in step 0"
echo "  - Forces model to learn diverse atom selection"
echo ""

PYTHONUNBUFFERED=1 python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk \
  --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 \
  --batch_size 16 \
  --d_model 128 \
  --nhead 2 \
  --nlayers 1 \
  --steps 2 \
  --no_type_bias \
  --score_center_atoms \
  --score_center_expert \
  --score_norm std \
  --expert_agg l2 \
  --atom_reduce_mode kmeans \
  --n_atoms 8 \
  --lr 0.001 \
  --use_hard_gumbel \
  --out_dir "$OUT_DIR" \
  2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "Training complete!"
echo "Results saved to: $OUT_DIR"
echo ""
echo "Next: Test atom diversity with this model"
