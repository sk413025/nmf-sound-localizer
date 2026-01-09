#!/bin/bash
set -e

# Config
DATA_DIR="converted_trajectories/speech_rtg_exp"
RESULTS_DIR="results/rtg_experiment"
K=4
ITEMS=200 
EPOCHS=15

# 1. Generate (Now with 'reductions' saved)
echo "1. Generating Trajectories (with Stepwise Rewards)..."
python3 scripts/h_exploration/generate_lag_omp.py \
    --out_dir "$DATA_DIR" \
    --max_items $ITEMS

# 2. Train RTG Model
echo "2. Training RTG Sequence Model..."
python3 scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path "$DATA_DIR/lag_trajectories.pt" \
    --out_dir "$RESULTS_DIR" \
    --epochs $EPOCHS \
    --batch_size 256

# 3. Evaluate with High Target
echo "3. Evaluating Model (Target RTG = 1.0)..."
python3 scripts/h_exploration/test_lag_model_rtg.py \
    --model_path "$RESULTS_DIR/dt_lag_rtg_best.pth" \
    --max_items 50 \
    --target_rtg 1.0

# 3b. Evaluate with Medium Target (Check controllability)
echo "3b. Evaluating Model (Target RTG = 0.5)..."
python3 scripts/h_exploration/test_lag_model_rtg.py \
    --model_path "$RESULTS_DIR/dt_lag_rtg_best.pth" \
    --max_items 50 \
    --target_rtg 0.5
