#!/bin/bash
set -e

# Activate Environment
source /Users/jnrle/Documents/LDVReorientation/worktrees/.venv/bin/activate
export PYTHONPATH=$(pwd)
cd /Users/jnrle/Documents/LDVReorientation/worktrees/exp-freq-aware-policy

echo "========================================"
echo "Starting Full Frequency-Aware Experiment"
echo "Date: $(date)"
echo "========================================"

# 1. Dataset Generation
echo "[Step 1] Generating Full Dataset (Speech 260 Clips)..."
# We set max_items=300 to ensure we process all ~260 clips
python3 scripts/h_exploration/generate_lag_omp.py \
    --out_dir results/full_data \
    --max_items 300 \
    --angle 90.0

echo "[Step 1] Data Generation Complete."

# 2. Model Training
echo "[Step 2] Training Frequency-Aware DT Policy..."
# Train on bins 5-1024 (Full spectrum except DC/Very Low Freq)
# Epochs increased to 30 for full convergence
python3 scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path results/full_data/lag_trajectories.pt \
    --out_dir results/dt_freq_aware_full \
    --epochs 30 \
    --batch_size 256 \
    --freq_range 5,1024

echo "========================================"
echo "Experiment Finished Successfully!"
echo "Date: $(date)"
echo "Artifacts are in results/dt_freq_aware_full"
echo "========================================"
