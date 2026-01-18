#!/bin/bash
set -euo pipefail

# Smoke: Tw=32, max_lag=50, K=16, RTG2D
# Generates tiny trajectories, trains briefly, and runs fast evals.

export PYTHONPATH=$PWD

PYTHON_EXE="/Users/jnrle/Documents/LDVReorientation/worktrees/.venv/bin/python"
if [ ! -f "$PYTHON_EXE" ]; then
  PYTHON_EXE="python"
fi

if [ -f ~/.zshrc ]; then
  source ~/.zshrc
  if typeset -f conda > /dev/null; then
    conda activate trl-training || true
    PYTHON_EXE="python"
  fi
fi

echo "Using Python: $PYTHON_EXE"

DATA_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1"
MIC_ROOT="$DATA_ROOT/MIC"
LDV_ROOT="$DATA_ROOT/LDV"

TS=$(date +%Y%m%d_%H%M%S)
EXP_NAME="smoke_gru2_tw32_lag50_k16_${TS}"
OUT_DIR="results/${EXP_NAME}"
mkdir -p "$OUT_DIR"

LOG="$OUT_DIR/run.log"
echo "Starting SMOKE $EXP_NAME at $(date)" | tee "$LOG"

echo "=== 1) Generate trajectories ===" | tee -a "$LOG"
$PYTHON_EXE -u scripts/h_exploration/generate_lag_omp.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --out_dir "$OUT_DIR/data" \
  --hop_length 160 \
  --max_lag 50 \
  --max_k 16 \
  --tw 32 \
  --gain 100.0 \
  --variants_per_clip 1 \
  --max_items 3 \
  --all_angles 2>&1 | tee -a "$LOG"

test -f "$OUT_DIR/data/lag_trajectories.pt"

echo "=== 2) Train (2 epochs) ===" | tee -a "$LOG"
$PYTHON_EXE -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path "$OUT_DIR/data/lag_trajectories.pt" \
  --out_dir "$OUT_DIR/model" \
  --epochs 2 \
  --batch_size 128 \
  --lr 5e-4 \
  --rtg_dim 2 \
  --rtg_mode target1 \
  --target_return 1.0 2>&1 | tee -a "$LOG"

test -f "$OUT_DIR/model/dt_freq_aware_best.pth"

echo "=== 3) OMP vs Random gap check ===" | tee -a "$LOG"
$PYTHON_EXE -u scripts/h_exploration/eval_lag_capture_omp_vs_random.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --out_dir "$OUT_DIR/omp_vs_random" \
  --hop_length 160 \
  --max_items 2 \
  --max_lag 50 \
  --max_k 16 \
  --tw 32 \
  --gain 100.0 \
  --random_trials 10 \
  --seed 0 \
  --all_angles 2>&1 | tee -a "$LOG"

test -f "$OUT_DIR/omp_vs_random/omp_vs_random_summary.json"

echo "=== 4) DT vs OMP capture eval (fast) ===" | tee -a "$LOG"
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
  --num_clips 2 \
  --all_angles 2>&1 | tee -a "$LOG"

echo "=== 5) RTG1 override grid (sanity) ===" | tee -a "$LOG"
$PYTHON_EXE -u scripts/h_exploration/run_rtg_override_grid_eval.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --ckpt_path "$OUT_DIR/model/dt_freq_aware_best.pth" \
  --out_dir "$OUT_DIR/rtg_grid" \
  --num_clips 1 \
  --hop_length 160 \
  --max_lag 50 \
  --max_k 16 \
  --tw 32 \
  --gain 100.0 \
  --rtg1_values 0.0,0.5,1.0 \
  --all_angles 2>&1 | tee -a "$LOG"

test -f "$OUT_DIR/rtg_grid/rtg1_override_grid.json"

echo "SMOKE completed OK at $(date)" | tee -a "$LOG"
