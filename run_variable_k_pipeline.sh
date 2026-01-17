#!/bin/bash
set -e

# Define Project Root
PROJECT_ROOT="exp-interspeech-GRU1"

# Activate Env
source /Users/jnrle/Documents/LDVReorientation/worktrees/.venv/bin/activate
# Fix PYTHONPATH to include the project root so 'scripts' module is found
export PYTHONPATH=$(pwd)/$PROJECT_ROOT:$(pwd):$PYTHONPATH

# Absolute or Relative paths relative to execution root (worktrees)
DATA_ROOT="$PROJECT_ROOT/data/derived/boy1_processed"
WORK_DIR="$PROJECT_ROOT/results/interspeech_gru1"

mkdir -p $WORK_DIR
DATE_TAG=$(date +%Y%m%d_%H%M%S)

echo "=================================================="
echo "Starting Variable K Pipeline"
echo "Work Dir: $WORK_DIR"
echo "Date: $DATE_TAG"
echo "=================================================="

# 1. Generate Variable K Trajectories from boy1
# Input: data/derived/boy1_processed
# Output: results/interspeech_gru1/trajectories_varK.pt
# Max Items: 100 (for quick iteration)
# Variants: 5 per clip

echo "[Step 1] Generating Trajectories (Variable K)..."
python3 $PROJECT_ROOT/scripts/h_exploration/generate_lag_omp.py \
    --mic_root $DATA_ROOT/MIC \
    --ldv_root $DATA_ROOT/LDV \
    --out_dir $WORK_DIR \
    --max_items 100 \
    --variants_per_clip 5 \
    --angle 0 

# Note: generate_lag_omp.py takes --angle. We preprocessed into angle_0, angle_10 etc.
# If we used '0' in preprocessing for block 0, we use 0 here.

mv $WORK_DIR/lag_trajectories.pt $WORK_DIR/trajectories_varK_boy1.pt

echo "[Step 1] Done."

# 2. Train MDP Agent
# Input: trajectories_varK_boy1.pt
# Config: Epochs 50, Batch 128

echo "[Step 2] Training MDP Agent..."
python3 $PROJECT_ROOT/scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path $WORK_DIR/trajectories_varK_boy1.pt \
    --out_dir $WORK_DIR/model_varK \
    --epochs 50 \
    --batch_size 128 \
    --freq_range 5,1024

echo "=================================================="
echo "Pipeline Completed Successfully."
echo "Model Artifacts: $WORK_DIR/model_varK"
echo "=================================================="
