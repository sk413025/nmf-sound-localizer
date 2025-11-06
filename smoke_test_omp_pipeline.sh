#!/bin/bash
# Quick Smoke Test: OMP Teacher Pipeline
# Purpose: Verify the OMP training pipeline works end-to-end without running full 480 epochs
# Runtime: ~2-3 minutes (vs several hours for full training)

set -e

echo "================================================================================"
echo "OMP Teacher Pipeline - Smoke Test (5 epochs)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Test configuration
TRAJ_DIR="results/dt_traj_omp_smoke_test"
OUT_DIR="results/dt_min_omp_smoke_test"
TEACHER="omp"
K=6
N_ATOMS=8
EPOCHS=5  # Short run for smoke test
BATCH_SIZE=4
LR=3e-3
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
TEST_SPLIT=0.2
SPLIT_SEED=42

# Data paths
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHON_CMD="/Users/sbplab/jnrle/.venv/bin/python"

echo "Test Configuration:"
echo "  Epochs: $EPOCHS (smoke test)"
echo "  Teacher: $TEACHER"
echo "  Expected accuracy: ~65% (limited by OMP's 64.9% trajectory quality)"
echo ""

# Phase 1: Generate trajectories
echo "Phase 1: Generating OMP trajectories..."
mkdir -p "$TRAJ_DIR"

$PYTHON_CMD -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher "$TEACHER" \
  --K "$K" \
  --n_atoms "$N_ATOMS" \
  --atom_reduce_mode kcenter \
  --fs 16000 \
  --n_fft 2048 \
  --freq_min 300 \
  --freq_max 3000 \
  --seed 42 \
  --h_path "$H_PATH" \
  --w_path "$W_PATH" \
  --dataset_root "$DATASET_ROOT" \
  --out_dir "$TRAJ_DIR" \
  --device cpu \
  2>&1 | tee "$TRAJ_DIR/generation.log"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "✗ Trajectory generation failed"
    exit 1
fi

echo "✓ Trajectories generated"
echo ""

# Phase 2: Quick verification
echo "Phase 2: Verifying trajectory quality..."
$PYTHON_CMD verify_omp_teacher_trajectories.py --traj_dir "$TRAJ_DIR" \
  2>&1 | tee "$TRAJ_DIR/verification.log"

echo ""

# Phase 3: Short training run
echo "Phase 3: Training for $EPOCHS epochs (smoke test)..."
mkdir -p "$OUT_DIR"

$PYTHON_CMD -u scripts/dt_pointer_ldv.py \
  --traj_dir "$TRAJ_DIR" \
  --out_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --d_model "$D_MODEL" \
  --nhead "$NHEAD" \
  --nlayers "$NLAYERS" \
  --device "$DEVICE" \
  --test_split "$TEST_SPLIT" \
  --split_seed "$SPLIT_SEED" \
  2>&1 | tee "$OUT_DIR/training.log"

TRAIN_EXIT=$?

echo ""
echo "================================================================================"
echo "Smoke Test Completed!"
echo "End time: $(date)"
echo "================================================================================"
echo ""

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Pipeline works end-to-end"
    echo ""
    echo "Results:"
    echo "  Trajectories: $TRAJ_DIR"
    echo "  Training outputs: $OUT_DIR"
    echo ""
    echo "Expected behavior:"
    echo "  - Trajectory accuracy: ~64.9% (OMP limitation)"
    echo "  - Initial DT accuracy: ~50-60% after 5 epochs"
    echo "  - Full training (480 epochs) expected: ~65-70%"
    echo ""
    echo "To run full 480-epoch training:"
    echo "  ./run_480epochs_omp.sh"
    echo ""
    echo "To compare with G-teacher:"
    echo "  # G-teacher achieves ~98% (see commit 63b8190)"
    echo "  # OMP achieves ~65-70% (limited by trajectory quality)"
else
    echo "✗ Smoke test failed with exit code $TRAIN_EXIT"
    echo "Check logs:"
    echo "  $TRAJ_DIR/generation.log"
    echo "  $OUT_DIR/training.log"
fi

exit $TRAIN_EXIT
