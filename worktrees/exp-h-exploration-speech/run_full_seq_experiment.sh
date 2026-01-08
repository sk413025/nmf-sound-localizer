#!/bin/bash
set -e

# Config
DATA_DIR="converted_trajectories/speech_seq_exp"
RESULTS_DIR="results/seq_experiment"
K=4
ITEMS=200 # Enough for training
EPOCHS=10

echo "1. Generating Trajectories with K=$K on $ITEMS clips..."
python scripts/h_exploration/generate_lag_omp.py \
    --out_dir "$DATA_DIR" \
    --max_items $ITEMS \
    # Output will be in $DATA_DIR/lag_trajectories.pt

echo "2. Training Sequence Model (RNN/DT)..."
python scripts/h_exploration/train_dt_lag_seq.py \
    --data_path "$DATA_DIR/lag_trajectories.pt" \
    --out_dir "$RESULTS_DIR" \
    --epochs $EPOCHS \
    --batch_size 256

echo "3. Evaluating Model on Test Loop (Real Physics)..."
python scripts/h_exploration/test_lag_model.py \
    --model_path "$RESULTS_DIR/dt_lag_seq_best.pth" \
    --max_items 50 \
    # Creates logs

echo "Experiment Complete."
