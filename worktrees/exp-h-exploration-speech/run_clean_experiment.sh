#!/bin/bash
set -e

# Experiment: Clean Physics + RTG Scale Fix
# Date: 2026-01-12
# Purpose: Validate if fixing OMP update logic (Unit Atom) and DTmin Input Scaling improves performance.

# 1. Generate Data (Slow Step)
# This uses the MODIFIED generate_lag_omp.py which enforces Normalized Atom Projection.
echo "=== Step 1: Generating OMP Trajectories (Clean Physics) ==="
export PYTHONPATH=$PYTHONPATH:$(pwd)
python3 scripts/h_exploration/generate_lag_omp.py \
    --out_dir data/omp_traj_clean_physics \
    --angle 90.0 \
    --max_items 500  # Increased from defaults to get robust train set

# 2. Train DTmin (Fast Step)
# This uses the MODIFIED train_dt_lag_seq_rtg.py which has LayerNorm on inputs.
echo "=== Step 2: Training DTmin (Seq + RTG + InputNorm) ==="
python3 scripts/h_exploration/train_dt_lag_seq_rtg.py \
    --data_path data/omp_traj_clean_physics/lag_trajectories.pt \
    --out_dir results/dtmin_clean_physics \
    --epochs 20 \
    --batch_size 128 \
    --lr 5e-4 

# 3. Evaluate (Single Clip Inspection)
# Just reuse the training loop's validation output for now, or running inspect_flow manually.
echo "Done. Experiment Completed."
echo "Check results/dtmin_clean_physics/training.log or similar."
