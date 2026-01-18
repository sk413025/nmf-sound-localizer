#!/usr/bin/env bash
set -euo pipefail

# Run step 2+3 only, reusing generated trajectories from the full pipeline.
# This is useful if the pipeline was interrupted after trajectory generation.

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

# Default to the venv found in workspace
PYTHON_EXE="/Users/jnrle/Documents/LDVReorientation/worktrees/.venv/bin/python"

# Fallback to standard python if venv not found
if [ ! -f "$PYTHON_EXE" ]; then
  PYTHON_EXE="python"
fi

# Try to activate conda if configured (for user convenience if they run manually)
if [ -f ~/.zshrc ]; then
  # shellcheck disable=SC1090
  source ~/.zshrc
  if typeset -f conda > /dev/null; then
    conda activate trl-training
    PYTHON_EXE="python" # Prefer active conda python
  fi
fi

echo "Using Python: $PYTHON_EXE"

EXP_NAME="exp_m32_k16_full_gain100_ep50"
OUT_DIR="results/${EXP_NAME}"

DATA_ROOT="/Users/sbplab/LDV-data-processed"
MIC_ROOT="$DATA_ROOT/speech260_original_16k_no_edge_sync_vad_normalized"
LDV_ROOT="$DATA_ROOT/speech260_box_16k_no_edge_sync_vad_normalized"

DATA_PATH="$OUT_DIR/data/lag_trajectories.pt"
CKPT_PATH="$OUT_DIR/model/dt_freq_aware_best.pth"

if [[ ! -f "$DATA_PATH" ]]; then
  echo "Missing $DATA_PATH. Run run_pipeline_m32.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR/model" "$OUT_DIR/eval"

{
  echo "=== Resume Pipeline (train+eval only) ${EXP_NAME} @ $(date) ==="
  echo "Data: $DATA_PATH"
} | tee -a "$OUT_DIR/pipeline.log"

$PYTHON_EXE scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path "$DATA_PATH" \
  --out_dir "$OUT_DIR/model" \
  --epochs 50 \
  --batch_size 256 \
  --lr 5e-4 2>&1 | tee -a "$OUT_DIR/pipeline.log"

if [[ ! -f "$CKPT_PATH" ]]; then
  echo "Missing checkpoint $CKPT_PATH after training." | tee -a "$OUT_DIR/pipeline.log"
  exit 1
fi

$PYTHON_EXE scripts/eval_energy_capture_generic.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT_PATH" \
  --out_dir "$OUT_DIR/eval" \
  --max_lag 32 \
  --max_k 16 \
  --tw 16 \
  --gain 100.0 \
  --visualize 2>&1 | tee -a "$OUT_DIR/pipeline.log"

echo "=== Done ${EXP_NAME} @ $(date) ===" | tee -a "$OUT_DIR/pipeline.log"
