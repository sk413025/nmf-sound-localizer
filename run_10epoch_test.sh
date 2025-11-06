#!/bin/bash
# Quick 10-epoch validation test for unpacking bug fixes
# Tests all model() call sites: training eval, final eval, controllability

set -e  # Exit on error

# Environment setup
export PYTHONPATH="$(pwd):${PYTHONPATH}"

# Configuration
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=3e-3
DEVICE="cpu"  # MPS doesn't support linalg_lstsq, use CPU
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="results/dt_test_10epoch_${TIMESTAMP}"

echo "=== 10-Epoch Validation Test ==="
echo "Output: ${OUTPUT_DIR}"
echo "Testing unpacking fixes at lines 728, 778, 865"
echo ""

# Step 1: Generate OMP trajectories (use same data as full training)
echo "[Step 1/2] Generating OMP trajectories..."
TRAJ_DIR="${OUTPUT_DIR}/trajectories"
mkdir -p ${TRAJ_DIR}

python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher omp \
    --K 6 \
    --n_atoms 8 \
    --atom_reduce_mode kcenter \
    --fs 16000 \
    --n_fft 2048 \
    --freq_min 300 \
    --freq_max 3000 \
    --seed 42 \
    --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
    --w_path doa_normalized_config_c_corrected/models/usm.pth \
    --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
    --out_dir ${TRAJ_DIR} \
    --device ${DEVICE}

# Step 2: Train Decision Transformer (10 epochs)
echo ""
echo "[Step 2/2] Training Decision Transformer (10 epochs)..."
python scripts/dt_pointer_ldv.py \
    --traj_dir ${TRAJ_DIR} \
    --out_dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --device ${DEVICE} \
    --d_model 128 \
    --nhead 2 \
    --nlayers 1 \
    --test_split 0.2 \
    --split_seed 42

echo ""
echo "=== Test Complete ==="
echo "Check log: ${OUTPUT_DIR}/training.log"
echo "If no crashes, all unpacking fixes validated!"
