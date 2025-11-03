#!/bin/bash
# Reproduction script for dt_traj_g_full (g-teacher OMP trajectories)
# Generated: 2025-11-03
# Purpose: Generate offline DT trajectories using classic OMP (|g| energies)

set -e  # Exit on error

echo "================================================================================"
echo "Reproducing dt_traj_g_full: Offline DT Trajectories (g-teacher)"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

# Configuration (matches original run)
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
OUT_DIR="results/dt_traj_g_full"
TEACHER="g"
K=6
N_ATOMS=8
ATOM_REDUCE_MODE="kcenter"
FS=16000
N_FFT=2048
FREQ_MIN=300.0
FREQ_MAX=3000.0
RTG_TARGET_RESID=0.02
RTG_TARGET_ACC=0.95
SEED=42
DEVICE="cpu"

echo "Configuration:"
echo "  Teacher policy: $TEACHER (classic OMP with |g| energies)"
echo "  H matrix: $H_PATH"
echo "  W matrix: $W_PATH"
echo "  Dataset: $DATASET_ROOT"
echo "  Output: $OUT_DIR"
echo "  K (trajectory steps): $K"
echo "  M (atoms per expert): $N_ATOMS (reduced via $ATOM_REDUCE_MODE)"
echo "  STFT: fs=$FS, n_fft=$N_FFT, band=[$FREQ_MIN, $FREQ_MAX] Hz"
echo "  RTG targets: resid=$RTG_TARGET_RESID, acc=$RTG_TARGET_ACC"
echo "  Random seed: $SEED"
echo "  Device: $DEVICE"
echo ""

# Verify prerequisites
if [ ! -f "$H_PATH" ]; then
    echo "✗ ERROR: H matrix not found at $H_PATH"
    exit 1
fi

if [ ! -f "$W_PATH" ]; then
    echo "✗ ERROR: W matrix not found at $W_PATH"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "✗ ERROR: Dataset root not found at $DATASET_ROOT"
    exit 1
fi

echo "✓ Prerequisites verified"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run trajectory generation
echo "Generating trajectories..."
echo ""

python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher "$TEACHER" \
    --K "$K" \
    --n_atoms "$N_ATOMS" \
    --atom_reduce_mode "$ATOM_REDUCE_MODE" \
    --h_path "$H_PATH" \
    --w_path "$W_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --fs "$FS" \
    --n_fft "$N_FFT" \
    --freq_min "$FREQ_MIN" \
    --freq_max "$FREQ_MAX" \
    --rtg_target_resid "$RTG_TARGET_RESID" \
    --rtg_target_acc "$RTG_TARGET_ACC" \
    2>&1 | tee "$OUT_DIR/run.log"

echo ""
echo "================================================================================"
echo "Trajectory generation complete"
echo "Output directory: $OUT_DIR"
echo "End time: $(date)"
echo "================================================================================"

# Verification
echo ""
echo "Verifying outputs..."
python -c "
import json
import os

out_dir = '$OUT_DIR'
required_files = ['trajectories.jsonl', 'numeric_diagnostics.jsonl', 'manifest.json', 'code_state.json', 'run.log']

print('Checking required files:')
for f in required_files:
    path = os.path.join(out_dir, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  ✓ {f} ({size} bytes)')
    else:
        print(f'  ✗ {f} MISSING')
        exit(1)

# Verify trajectory count
with open(os.path.join(out_dir, 'trajectories.jsonl')) as f:
    n_trajs = sum(1 for _ in f)
print(f'\nTrajectory count: {n_trajs} (expected: 111 = 37 angles × 3 clips)')
assert n_trajs == 111, f'Expected 111 trajectories, got {n_trajs}'

# Verify manifest
with open(os.path.join(out_dir, 'manifest.json')) as f:
    manifest = json.load(f)
print(f'Manifest verification:')
print(f'  Teacher: {manifest[\"teacher\"]}')
print(f'  Dataset fingerprint: {manifest[\"fingerprint_md5\"]}')
print(f'  Samples: {len(manifest[\"samples\"])}')
print(f'  E={manifest[\"E\"]}, M={manifest[\"M\"]}, K={manifest[\"K\"]}, F={manifest[\"F\"]}')

print('\n✅ All verification checks passed!')
"

echo ""
echo "✅ Reproduction successful!"
