#!/bin/bash
# Complete 480-Epoch Training Pipeline
# Date: 2025-11-06
# Purpose: Full training from OMP trajectory generation to DT training with physics reconstruction

set -e

echo "================================================================================"
echo "Complete Training Pipeline: OMP Trajectories → Decision Transformer (480 epochs)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# ============================================================================
# Configuration
# ============================================================================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Paths
TRAJ_DIR="results/dt_traj_omp_full_${TIMESTAMP}"
OUT_DIR="results/dt_full_training_${TIMESTAMP}"

# OMP Trajectory settings
TEACHER="omp"
K=6
N_ATOMS=8
ATOM_REDUCE_MODE="kcenter"
FS=16000
N_FFT=2048
FREQ_MIN=300
FREQ_MAX=3000
SEED=42

# Data paths
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"

# Training settings
EPOCHS=480
BATCH_SIZE=4
LR=3e-3
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
TEST_SPLIT=0.2
SPLIT_SEED=42

# Physics reconstruction settings (optional)
USE_PHYSICS=false
PHYSICS_WEIGHT=0.1
PHYSICS_WARMUP_EPOCHS=100
RESIDUAL_WEIGHT=1.0
DIRECTION_WEIGHT=1.0  # Increased from 0.5
COHERENCE_WEIGHT=0.1

# ============================================================================
# Environment Setup
# ============================================================================

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
PYTHON_CMD="/Users/sbplab/jnrle/.venv/bin/python"

echo "Configuration Summary:"
echo "  OMP Teacher: K=$K, M=$N_ATOMS, method=$ATOM_REDUCE_MODE"
echo "  Training: epochs=$EPOCHS, batch_size=$BATCH_SIZE, lr=$LR"
echo "  Physics reconstruction: $USE_PHYSICS"
if [ "$USE_PHYSICS" = true ]; then
    echo "    - Physics weight: $PHYSICS_WEIGHT"
    echo "    - Warmup epochs: $PHYSICS_WARMUP_EPOCHS"
fi
echo "  Output directory: $OUT_DIR"
echo ""

# ============================================================================
# Phase 1: Generate OMP Trajectories
# ============================================================================

echo "================================================================================"
echo "Phase 1: Generating OMP Trajectories"
echo "================================================================================"
echo ""

mkdir -p "$TRAJ_DIR"

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
    exit $TRAJ_EXIT_CODE
fi

echo ""
echo "✓ OMP trajectories generated successfully"
echo "  Location: $TRAJ_DIR/trajectories.jsonl"
echo ""

# Extract trajectory quality metrics
if [ -f "$TRAJ_DIR/generation.log" ]; then
    echo "Trajectory Quality Metrics:"
    grep -A 5 "First-step accuracy" "$TRAJ_DIR/generation.log" || echo "  (Metrics not found in log)"
    echo ""
fi

# ============================================================================
# Phase 2: Train Decision Transformer
# ============================================================================

echo "================================================================================"
echo "Phase 2: Training Decision Transformer (480 epochs)"
echo "================================================================================"
echo ""

mkdir -p "$OUT_DIR"

# Build physics flags
if [ "$USE_PHYSICS" = true ]; then
    PHYSICS_FLAGS="--use_physics_reconstruction \
  --physics_weight $PHYSICS_WEIGHT \
  --physics_warmup_epochs $PHYSICS_WARMUP_EPOCHS \
  --residual_weight $RESIDUAL_WEIGHT \
  --direction_weight $DIRECTION_WEIGHT \
  --coherence_weight $COHERENCE_WEIGHT"
else
    PHYSICS_FLAGS=""
fi

echo "Starting training..."
echo "  This will take several hours (estimated: 2-4 hours)"
echo "  Progress will be logged to: $OUT_DIR/training.log"
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
  --test_split "$TEST_SPLIT" \
  --split_seed "$SPLIT_SEED" \
  $PHYSICS_FLAGS \
  2>&1 | tee "$OUT_DIR/training.log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
echo "Training Complete"
echo "================================================================================"
echo "End time: $(date)"
echo ""

# ============================================================================
# Phase 3: Extract and Display Results
# ============================================================================

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training finished successfully"
    echo ""
    
    echo "================================================================================"
    echo "RESULTS SUMMARY"
    echo "================================================================================"
    echo ""
    
    echo "1. OMP Trajectory Quality:"
    echo "   Location: $TRAJ_DIR/generation.log"
    if grep -q "First-step accuracy" "$TRAJ_DIR/generation.log" 2>/dev/null; then
        grep "First-step accuracy" "$TRAJ_DIR/generation.log" | tail -1
    else
        echo "   (Check generation.log for details)"
    fi
    echo ""
    
    echo "2. Decision Transformer Performance:"
    echo "   Location: $OUT_DIR/training.log"
    echo ""
    echo "   Final Epoch Results:"
    if [ -f "$OUT_DIR/training.log" ]; then
        # Extract final epoch metrics
        grep "Epoch $EPOCHS/$EPOCHS:" "$OUT_DIR/training.log" -A 5 | head -6
        echo ""
        
        # Extract best test loss
        echo "   Best Performance:"
        grep "Best test loss" "$OUT_DIR/training.log" | tail -1
    else
        echo "   (Check training.log for details)"
    fi
    echo ""
    
    echo "3. Generated Files:"
    echo "   Trajectories:"
    ls -lh "$TRAJ_DIR" | grep -E "trajectories|manifest|generation"
    echo ""
    echo "   Training outputs:"
    ls -lh "$OUT_DIR" | grep -E "log|ckpt|json|npz"
    echo ""
    
    echo "================================================================================"
    echo "Next Steps:"
    echo "================================================================================"
    echo ""
    echo "1. Review detailed results:"
    echo "   - OMP metrics: cat $TRAJ_DIR/generation.log"
    echo "   - Training log: cat $OUT_DIR/training.log"
    echo ""
    echo "2. Analyze training dynamics:"
    echo "   - Plot learning curves from training.log"
    echo "   - Compare with baseline (OMP accuracy vs DT accuracy)"
    echo ""
    echo "3. Evaluate model:"
    echo "   - Load checkpoint: $OUT_DIR/best_model.ckpt"
    echo "   - Run inference on test set"
    echo ""
    
    if [ "$USE_PHYSICS" = true ]; then
        echo "4. Physics reconstruction analysis:"
        echo "   - Extract physics metrics: grep 'Physics metrics' $OUT_DIR/training.log"
        echo "   - Verify embeddings are interpretable"
        echo ""
    fi
    
else
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    echo ""
    echo "Check logs:"
    echo "  - Trajectory generation: $TRAJ_DIR/generation.log"
    echo "  - Training: $OUT_DIR/training.log"
    echo ""
fi

exit $TRAIN_EXIT_CODE
