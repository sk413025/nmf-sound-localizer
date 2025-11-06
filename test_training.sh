#!/bin/bash
# Decision Transformer Test Script
# Purpose: Test if training pipeline works (1 epoch only)

set -e  # Exit on error

echo "================================================================================"
echo "Decision Transformer - Test Run (1 epoch)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Configuration
TRAJ_DIR="results/dt_traj_qk_kmeans"
OUT_DIR="results/dt_test_$(date +%Y%m%d_%H%M%S)"
EPOCHS=1
BATCH_SIZE=4
LR=3e-3
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
DISTILL_WEIGHT=0.5
DISTILL_T=1.0
WARMUP_EPOCHS=0
TEST_SPLIT=0.2
SPLIT_SEED=42

echo "Configuration:"
echo "  Trajectory directory: $TRAJ_DIR"
echo "  Output directory: $OUT_DIR"
echo "  Epochs: $EPOCHS (test run)"
echo "  Test split: $TEST_SPLIT (seed: $SPLIT_SEED)"
echo "  Device: $DEVICE"
echo ""

# Verify trajectory files exist
if [ ! -f "$TRAJ_DIR/manifest.json" ]; then
    echo "✗ ERROR: Trajectory manifest not found at $TRAJ_DIR/manifest.json"
    exit 1
fi

echo "✓ Trajectory files verified"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Use venv Python
PYTHON_CMD="/Users/sbplab/jnrle/.venv/bin/python"

# Run training with output logging
echo "Starting test training..."
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

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "================================================================================"
echo "Test run completed!"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Test run finished successfully"
    echo ""
    echo "Generated files:"
    ls -lh "$OUT_DIR"
    echo ""
    echo "To run full 480-epoch training, use:"
    echo "  ./run_480epochs.sh"
else
    echo "✗ Test run failed with exit code $EXIT_CODE"
    echo "Check log file: $OUT_DIR/training.log"
fi

exit $EXIT_CODE
