#!/bin/bash
# Pipeline for M=32 (Lags -32..32, Total 65), K=32
# Full 200 Epoch training

# Setup Environment
export PYTHONPATH=$PWD

# Default to the venv found in workspace
PYTHON_EXE="/Users/jnrle/Documents/LDVReorientation/worktrees/.venv/bin/python"

# Fallback to standard python if venv not found
if [ ! -f "$PYTHON_EXE" ]; then
    PYTHON_EXE="python"
fi

# Try to activate conda if configured (for user convenience if they run manually)
if [ -f ~/.zshrc ]; then
    source ~/.zshrc
    if typeset -f conda > /dev/null; then
        conda activate trl-training
        PYTHON_EXE="python" # Prefer active conda python
    fi
fi

echo "Using Python: $PYTHON_EXE"

EXP_NAME="exp_m32_k32_full"
DATA_ROOT="/Users/sbplab/LDV-data-processed"
MIC_ROOT="$DATA_ROOT/speech260_original_16k_no_edge_sync_vad_normalized"
LDV_ROOT="$DATA_ROOT/speech260_box_16k_no_edge_sync_vad_normalized"

OUT_DIR="results/$EXP_NAME"
mkdir -p $OUT_DIR

# Log start
echo "Starting Pipeline $EXP_NAME at $(date)" | tee "$OUT_DIR/pipeline.log"

# 1. Generation
echo "=== 1. Generating Trajectories (M=65, K=32) ===" | tee -a "$OUT_DIR/pipeline.log"
# We increase max_items to cover most/all of the dataset (approx 700 clips usually)
$PYTHON_EXE scripts/h_exploration/generate_lag_omp.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$OUT_DIR/data" \
    --max_lag 32 \
    --max_k 32 \
    --tw 16 \
    --variants_per_clip 5 \
    --max_items 1000 \
    --angle 90.0 >> "$OUT_DIR/pipeline.log" 2>&1

if [ ! -f "$OUT_DIR/data/lag_trajectories.pt" ]; then
    echo "Error: Trajectory generation failed. Check $OUT_DIR/pipeline.log" | tee -a "$OUT_DIR/pipeline.log"
    exit 1
fi

# 2. Training
echo "=== 2. Training DTmin Agent (200 Epochs) ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path "$OUT_DIR/data/lag_trajectories.pt" \
    --out_dir "$OUT_DIR/model" \
    --epochs 200 \
    --batch_size 256 \
    --lr 5e-4 >> "$OUT_DIR/pipeline.log" 2>&1

if [ ! -f "$OUT_DIR/model/dt_freq_aware_best.pth" ]; then
    echo "Error: Training failed (no checkpoint)." | tee -a "$OUT_DIR/pipeline.log"
    exit 1
fi

# 3. Evaluation
echo "=== 3. Evaluation ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE scripts/eval_energy_capture_generic.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --ckpt_path "$OUT_DIR/model/dt_freq_aware_best.pth" \
    --out_dir "$OUT_DIR/eval" \
    --max_lag 32 \
    --max_k 32 \
    --tw 16 \
    --visualize >> "$OUT_DIR/pipeline.log" 2>&1

echo "=== Pipeline Completed Successfully at $(date) ===" | tee -a "$OUT_DIR/pipeline.log"
