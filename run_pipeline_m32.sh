#!/bin/bash
# Pipeline for Tw=32, max_lag=50 (Lags -50..50, Total 101), K=16
# Full 50 Epoch training

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

EXP_NAME="exp_interspeech_gru2_tw32_lag50_k16_ep50"
DATA_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1"
MIC_ROOT="$DATA_ROOT/MIC"
LDV_ROOT="$DATA_ROOT/LDV"

OUT_DIR="results/$EXP_NAME"
mkdir -p $OUT_DIR

# Log start
echo "Starting Pipeline $EXP_NAME at $(date)" | tee "$OUT_DIR/pipeline.log"

# 1. Generation
echo "=== 1. Generating Trajectories (Tw=32, Lag=50, K=16) ===" | tee -a "$OUT_DIR/pipeline.log"
# We increase max_items to cover most/all of the dataset (approx 700 clips usually)
$PYTHON_EXE -u scripts/h_exploration/generate_lag_omp.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$OUT_DIR/data" \
    --hop_length 160 \
    --max_lag 50 \
    --max_k 16 \
    --tw 32 \
    --gain 100.0 \
    --variants_per_clip 5 \
    --max_items 1000 \
    --all_angles >> "$OUT_DIR/pipeline.log" 2>&1

if [ ! -f "$OUT_DIR/data/lag_trajectories.pt" ]; then
    echo "Error: Trajectory generation failed. Check $OUT_DIR/pipeline.log" | tee -a "$OUT_DIR/pipeline.log"
    exit 1
fi

# 1b. OMP vs Random gap (Tw sensitivity)
echo "=== 1b. OMP vs Random (gap diagnostic) ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE -u scripts/h_exploration/eval_lag_capture_omp_vs_random.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --out_dir "$OUT_DIR/omp_vs_random" \
    --hop_length 160 \
    --max_items 10 \
    --max_lag 50 \
    --max_k 16 \
    --tw 32 \
    --gain 100.0 \
    --random_trials 50 \
    --seed 0 \
    --all_angles >> "$OUT_DIR/pipeline.log" 2>&1

# 2. Training
echo "=== 2. Training DTmin Agent (50 Epochs) ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path "$OUT_DIR/data/lag_trajectories.pt" \
    --out_dir "$OUT_DIR/model" \
    --epochs 50 \
    --batch_size 256 \
    --lr 5e-4 \
    --rtg_dim 2 \
    --rtg_mode target1 \
    --target_return 1.0 >> "$OUT_DIR/pipeline.log" 2>&1

if [ ! -f "$OUT_DIR/model/dt_freq_aware_best.pth" ]; then
    echo "Error: Training failed (no checkpoint)." | tee -a "$OUT_DIR/pipeline.log"
    exit 1
fi

# 3. Evaluation
echo "=== 3. Evaluation (DT vs OMP) ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE -u scripts/eval_energy_capture_generic.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --ckpt_path "$OUT_DIR/model/dt_freq_aware_best.pth" \
    --out_dir "$OUT_DIR/eval" \
    --hop_length 160 \
    --max_lag 50 \
    --max_k 16 \
    --tw 32 \
    --gain 100.0 \
    --rtg_dim 2 \
    --num_clips 10 \
    --all_angles \
    --visualize >> "$OUT_DIR/pipeline.log" 2>&1

# 3b. RTG1 override grid (sanity that RTG2D matters)
echo "=== 3b. RTG1 override grid ===" | tee -a "$OUT_DIR/pipeline.log"
$PYTHON_EXE -u scripts/h_exploration/run_rtg_override_grid_eval.py \
    --mic_root "$MIC_ROOT" \
    --ldv_root "$LDV_ROOT" \
    --ckpt_path "$OUT_DIR/model/dt_freq_aware_best.pth" \
    --out_dir "$OUT_DIR/rtg_grid" \
    --num_clips 3 \
    --hop_length 160 \
    --max_lag 50 \
    --max_k 16 \
    --tw 32 \
    --gain 100.0 \
    --rtg1_values 0.0,0.25,0.5,0.75,1.0 \
    --all_angles >> "$OUT_DIR/pipeline.log" 2>&1

echo "=== Pipeline Completed Successfully at $(date) ===" | tee -a "$OUT_DIR/pipeline.log"
