#!/bin/bash
# Physics-Aware OMP Teacher Training Script
# Date: 2025-11-06
# Purpose: Train DTMin with physics reconstruction head to learn interpretable embeddings

set -e

echo "================================================================================"
echo "DTMin Physics-Aware Training (OMP Teacher + Physics Reconstruction)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Configuration
TRAJ_DIR="results/dt_traj_omp_480epochs_20251106_122948"  # Re-use existing OMP trajectories
OUT_DIR="results/dt_physics_test_$(date +%Y%m%d_%H%M%S)"
EPOCHS=100  # Shorter test run
BATCH_SIZE=4
LR=3e-3
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
TEST_SPLIT=0.2
SPLIT_SEED=42

# Physics reconstruction parameters (NEW)
USE_PHYSICS=true
PHYSICS_WEIGHT=0.1
PHYSICS_WARMUP_EPOCHS=30  # First 30 epochs emphasize physics
RESIDUAL_WEIGHT=1.0
DIRECTION_WEIGHT=0.5
COHERENCE_WEIGHT=0.1

echo "Configuration:"
echo "  Trajectory source: $TRAJ_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Epochs: $EPOCHS"
echo "  Test split: $TEST_SPLIT"
echo "  Device: $DEVICE"
echo ""
echo "Physics Reconstruction Settings:"
echo "  Enabled: $USE_PHYSICS"
echo "  Physics weight: $PHYSICS_WEIGHT"
echo "  Warmup epochs: $PHYSICS_WARMUP_EPOCHS (70% physics, 30% action)"
echo "  After warmup: ${PHYSICS_WEIGHT} physics, 1.0 action"
echo "  Component weights:"
echo "    - Residual reconstruction: $RESIDUAL_WEIGHT"
echo "    - Direction classification: $DIRECTION_WEIGHT"
echo "    - Spectral coherence: $COHERENCE_WEIGHT"
echo ""

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use venv Python
PYTHON_CMD="/Users/sbplab/jnrle/.venv/bin/python"

# Create output directory
mkdir -p "$OUT_DIR"

# Run training with physics reconstruction
echo "Starting physics-aware training..."
echo "Logs will be saved to: $OUT_DIR/training.log"
echo ""

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

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
echo "Training completed!"
echo "End time: $(date)"
echo "================================================================================"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training finished successfully"
    echo ""
    echo "Generated files:"
    ls -lh "$OUT_DIR"
    echo ""
    echo "Key metrics to check:"
    echo "  1. Physics reconstruction losses (residual, direction, coherence)"
    echo "  2. Action prediction accuracy (expert, atom)"
    echo "  3. Comparison: Physics-aware vs baseline (OMP)"
    echo ""
    echo "Expected observations:"
    echo "  - Early epochs: High physics loss, lower action accuracy"
    echo "  - After warmup: Balanced physics + action"
    echo "  - Token embeddings should capture:"
    echo "    * Residual predictions (next-step r_{t+1})"
    echo "    * Direction identification (expert classification)"
    echo "    * Spectral coherence (γ² ≈ 0.7-0.9)"
    echo ""
    echo "To analyze results:"
    echo "  grep 'Physics metrics' $OUT_DIR/training.log | tail -20"
    echo "  python scripts/analyze_training.py --out_dir $OUT_DIR"
else
    echo "✗ Training failed with exit code $EXIT_CODE"
    echo "Check log file: $OUT_DIR/training.log"
fi

exit $EXIT_CODE
