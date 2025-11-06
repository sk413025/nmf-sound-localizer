#!/bin/bash
# Run traditional OMP teacher and verify accuracy
# Compares OMP vs G-teacher performance

set -e

# Environment setup
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH

# Configuration (matching g-teacher verification)
TEACHER="omp"
K=6
N_ATOMS=8
ATOM_REDUCE_MODE="kcenter"
FS=16000
N_FFT=2048
FREQ_MIN=300.0
FREQ_MAX=3000.0
SEED=42

# Paths (matching g-teacher)
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"

# Output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="results/dt_traj_omp_verification_${TIMESTAMP}"

# Create output directory
mkdir -p ${OUT_DIR}

echo "========================================================================"
echo "Traditional OMP Teacher Trajectory Generation and Verification"
echo "========================================================================"
echo "Teacher: ${TEACHER}"
echo "K: ${K}"
echo "N_atoms: ${N_ATOMS}"
echo "Atom reduce mode: ${ATOM_REDUCE_MODE}"
echo "Seed: ${SEED}"
echo "Output: ${OUT_DIR}"
echo "========================================================================"
echo ""

# Step 1: Generate trajectories with traditional OMP
echo "Step 1: Generating OMP trajectories..."
python -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher ${TEACHER} \
  --K ${K} \
  --n_atoms ${N_ATOMS} \
  --atom_reduce_mode ${ATOM_REDUCE_MODE} \
  --fs ${FS} \
  --n_fft ${N_FFT} \
  --freq_min ${FREQ_MIN} \
  --freq_max ${FREQ_MAX} \
  --seed ${SEED} \
  --h_path ${H_PATH} \
  --w_path ${W_PATH} \
  --dataset_root ${DATASET_ROOT} \
  --out_dir ${OUT_DIR} \
  --device cpu \
  2>&1 | tee ${OUT_DIR}/generation.log

echo ""
echo "Step 2: Verifying OMP trajectories..."
python -u verify_omp_teacher_trajectories.py \
  --traj_dir ${OUT_DIR} \
  --h_path ${H_PATH} \
  --w_path ${W_PATH} \
  2>&1 | tee ${OUT_DIR}/verification.log

echo ""
echo "========================================================================"
echo "Verification Complete!"
echo "========================================================================"
echo "Results directory: ${OUT_DIR}"
echo "Generation log: ${OUT_DIR}/generation.log"
echo "Verification log: ${OUT_DIR}/verification.log"
echo ""
echo "Key files:"
echo "  - trajectories.jsonl: Generated trajectories"
echo "  - numeric_diagnostics.jsonl: Per-sample diagnostics"
echo "  - manifest.json: Configuration and metadata"
echo "  - verification.log: Accuracy comparison OMP vs G-teacher"
echo "========================================================================"
