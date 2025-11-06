#!/bin/bash
# Decision Transformer 480 Epochs Training Script - OMP Teacher Variant
# Date: 2025-01-06
# Purpose: Extended training using traditional OMP teacher trajectories
# Note: Expected lower performance (~65-70% accuracy) vs G-teacher (98%+)
#       due to OMP's 64.9% trajectory accuracy (see commit a654557)

set -e  # Exit on error

echo "================================================================================"
echo "Decision Transformer - 480 Epochs Training (OMP Teacher)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# ============================================================================
# Phase 1: Generate OMP Teacher Trajectories
# ============================================================================

echo "Phase 1: Generating OMP teacher trajectories..."
echo ""

# OMP trajectory configuration
TRAJ_DIR="results/dt_traj_omp_480epochs_$(date +%Y%m%d_%H%M%S)"
TEACHER="omp"
K=6
N_ATOMS=8
ATOM_REDUCE_MODE="kcenter"
FS=16000
N_FFT=2048
FREQ_MIN=300
FREQ_MAX=3000
SEED=42

# Data paths (same as G-teacher for fair comparison)
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"

echo "Trajectory Configuration:"
echo "  Teacher: $TEACHER (traditional OMP, pure greedy selection)"
echo "  Output: $TRAJ_DIR"
echo "  K: $K"
echo "  Atoms: $N_ATOMS (kcenter reduction)"
echo "  STFT: fs=$FS, n_fft=$N_FFT, band=[$FREQ_MIN, $FREQ_MAX] Hz"
echo "  Seed: $SEED"
echo ""

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use venv Python
PYTHON_CMD="/Users/sbplab/jnrle/.venv/bin/python"

# Create output directory for logs
mkdir -p "$TRAJ_DIR"

# Generate trajectories
echo "Generating trajectories (this will take 1-2 minutes)..."
$PYTHON_CMD -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher "$TEACHER" \
  --K "$K" \
  --n_atoms "$N_ATOMS" \
  --atom_reduce_mode "$ATOM_REDUCE_MODE" \
  --fs "$FS" \
  --n_fft "$N_FFT" \
  --freq_min "$FREQ_MIN" \
  --freq_max "$FREQ_MAX" \
  --seed "$SEED" \
  --h_path "$H_PATH" \
  --w_path "$W_PATH" \
  --dataset_root "$DATASET_ROOT" \
  --out_dir "$TRAJ_DIR" \
  --device cpu \
  2>&1 | tee "$TRAJ_DIR/generation.log"

TRAJ_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAJ_EXIT_CODE -ne 0 ]; then
    echo "✗ Trajectory generation failed with exit code $TRAJ_EXIT_CODE"
    echo "Check log: $TRAJ_DIR/generation.log"
    exit $TRAJ_EXIT_CODE
fi

echo "✓ Trajectories generated successfully"
echo ""

# ============================================================================
# Phase 2: Verify OMP Trajectory Quality (Optional but Recommended)
# ============================================================================

echo "Phase 2: Verifying trajectory quality..."
echo ""

$PYTHON_CMD verify_omp_teacher_trajectories.py --traj_dir "$TRAJ_DIR" \
  2>&1 | tee "$TRAJ_DIR/verification.log"

VERIFY_EXIT_CODE=${PIPESTATUS[0]}

if [ $VERIFY_EXIT_CODE -ne 0 ]; then
    echo "⚠ Verification failed but continuing with training"
else
    echo "✓ Verification completed"
fi

echo ""

# ============================================================================
# Phase 3: Train Decision Transformer on OMP Trajectories
# ============================================================================

echo "Phase 3: Training Decision Transformer..."
echo ""

# DT training configuration
OUT_DIR="results/dt_min_480epochs_omp_$(date +%Y%m%d_%H%M%S)"
EPOCHS=480
BATCH_SIZE=4
LR=3e-3
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
DISTILL_WEIGHT=0.5
DISTILL_T=1.0
WARMUP_EPOCHS=2
TEST_SPLIT=0.2
SPLIT_SEED=42

echo "Training Configuration:"
echo "  Trajectory source: $TRAJ_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Test split: $TEST_SPLIT (seed: $SPLIT_SEED)"
echo "  Device: $DEVICE"
echo ""
echo "⚠ EXPECTED PERFORMANCE WARNING:"
echo "  Traditional OMP trajectories have 64.9% first-step accuracy"
echo "  (vs G-teacher's 100% accuracy, see commit a654557)"
echo "  Expected DT test accuracy ceiling: ~65-70% (vs 98%+ with G-teacher)"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Run training with output logging
echo "Starting training (this will take several hours)..."
echo "Logs will be saved to: $OUT_DIR/training.log"
echo ""

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
  --distill_weight "$DISTILL_WEIGHT" \
  --distill_T "$DISTILL_T" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --test_split "$TEST_SPLIT" \
  --split_seed "$SPLIT_SEED" \
  2>&1 | tee "$OUT_DIR/training.log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
echo "Training completed!"
echo "End time: $(date)"
echo "================================================================================"
echo ""

# ============================================================================
# Summary and Analysis
# ============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training finished successfully"
    echo ""
    echo "Generated files:"
    echo "Trajectories ($TRAJ_DIR):"
    ls -lh "$TRAJ_DIR" | head -10
    echo ""
    echo "Training outputs ($OUT_DIR):"
    ls -lh "$OUT_DIR"
    echo ""
    echo "To analyze results, run:"
    echo "  python3 scripts/analyze_training.py --out_dir $OUT_DIR"
    echo ""
    echo "To compare with G-teacher results:"
    echo "  # G-teacher baseline (from previous runs, expected ~98% accuracy)"
    echo "  # OMP teacher (this run, expected ~65-70% accuracy)"
    echo "  # Performance gap matches trajectory quality (64.9% vs 100%)"
    echo ""
    echo "Key findings from OMP experiment (commit a654557):"
    echo "  - Traditional OMP: 64.9% first-step accuracy"
    echo "  - G-teacher: 100% first-step accuracy"
    echo "  - Root cause: OMP ignores expert (direction) structure"
    echo "  - Recommendation: Use G-teacher for production systems"
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo "Check logs:"
    echo "  Trajectory generation: $TRAJ_DIR/generation.log"
    echo "  Training: $OUT_DIR/training.log"
fi

exit $TRAIN_EXIT_CODE
