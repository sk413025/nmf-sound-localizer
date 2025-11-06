#!/bin/bash
################################################################################
# Reproduction Script for Commit 8d5f10b
# "Results: DT‑min on K‑means QK trajectories — extended to 80 epochs (atomic)"
################################################################################
#
# This script reproduces the experiment from commit 8d5f10bc0b008a6b924422d3c8d63ee59cdbf59e
# Training continues from 40→80 epochs in 8×5-epoch chunks
#
# Expected results (from commit message):
# - Loss: 4.81 → 3.59 (final chunk)
# - Step imitation: expert≈0.628, atom≈0.791
# - Teacher-forced step match: ≈0.577
# - Angle accuracy: DT-min t=0≈0.405; DT-min t=K-1≈0.351
#
################################################################################

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Reproducing Commit 8d5f10b: DT-min 40→80 epochs training${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Configuration
TRAJ_DIR="results/dt_traj_qk_kmeans"
OUT_DIR="results/dt_min_qk_kmeans_distill_reproduction"
SCRIPT="scripts/dt_pointer_ldv.py"
LOG_FILE="${OUT_DIR}/run.log"

# Training hyperparameters (from commit message)
EPOCHS_PER_CHUNK=5
NUM_CHUNKS=8
BATCH_SIZE=8
D_MODEL=128
NHEAD=2
NLAYERS=1
DISTILL_WEIGHT=0.7
DISTILL_T=1.0
WARMUP_EPOCHS=3
DEVICE="cpu"

# Validate required files exist
echo -e "\n${YELLOW}[1/4] Validating environment...${NC}"

if [ ! -f "${SCRIPT}" ]; then
    echo -e "ERROR: Training script not found: ${SCRIPT}"
    exit 1
fi
echo "✓ Training script found: ${SCRIPT}"

if [ ! -d "${TRAJ_DIR}" ]; then
    echo -e "ERROR: Trajectory directory not found: ${TRAJ_DIR}"
    exit 1
fi
echo "✓ Trajectory directory found: ${TRAJ_DIR}"

# Create output directory
mkdir -p "${OUT_DIR}"
echo "✓ Output directory ready: ${OUT_DIR}"

# Check Python dependencies
echo -e "\n${YELLOW}[2/4] Checking Python dependencies...${NC}"
python3 -c "import torch; import numpy; print('✓ PyTorch version:', torch.__version__)" || {
    echo "ERROR: PyTorch not found. Please install: pip install torch"
    exit 1
}

# Note about checkpoint
echo -e "\n${YELLOW}[3/4] Checkpoint status...${NC}"
if [ -f "results/dt_min_qk_kmeans_distill/ckpt_latest.pth" ]; then
    CKPT_SIZE=$(wc -c < "results/dt_min_qk_kmeans_distill/ckpt_latest.pth")
    if [ ${CKPT_SIZE} -lt 1000 ]; then
        echo -e "${YELLOW}⚠ WARNING: Checkpoint appears to be LFS pointer (${CKPT_SIZE} bytes)${NC}"
        echo "  The script will start training from scratch if checkpoint cannot be loaded."
        echo "  To get the actual checkpoint, run: git lfs pull"
        echo ""
        echo "  Alternatively, if you want to start from 40 epochs as in the original experiment:"
        echo "  1. Checkout commit 0cfe0c1 (40 epochs state)"
        echo "  2. Copy the checkpoint: cp results/dt_min_qk_kmeans_distill/ckpt_latest.pth ${OUT_DIR}/"
        echo "  3. Run this script"
    else
        echo "✓ Checkpoint file exists (${CKPT_SIZE} bytes)"
        # Copy checkpoint if reproducing from 40 epochs
        if [ ! -f "${OUT_DIR}/ckpt_latest.pth" ]; then
            echo "  Copying checkpoint to reproduction directory..."
            cp "results/dt_min_qk_kmeans_distill/ckpt_latest.pth" "${OUT_DIR}/"
        fi
    fi
else
    echo "⚠ No checkpoint found. Training will start from scratch."
fi

# Run training
echo -e "\n${YELLOW}[4/4] Starting training (8 chunks × 5 epochs each)...${NC}"
echo "  This will train from epoch 40 → 80 (assuming checkpoint at epoch 40)"
echo "  Training parameters:"
echo "    - Batch size: ${BATCH_SIZE}"
echo "    - Model: d_model=${D_MODEL}, nhead=${NHEAD}, nlayers=${NLAYERS}"
echo "    - Distillation: weight=${DISTILL_WEIGHT}, temperature=${DISTILL_T}"
echo "    - Warmup epochs: ${WARMUP_EPOCHS}"
echo "    - Device: ${DEVICE}"
echo ""
echo "  Log file: ${LOG_FILE}"
echo ""

# Initialize log file
echo "Reproduction of commit 8d5f10b started at $(date)" > "${LOG_FILE}"
echo "Command parameters: epochs=${EPOCHS_PER_CHUNK}, chunks=${NUM_CHUNKS}" >> "${LOG_FILE}"
echo "================================================================================" >> "${LOG_FILE}"

# Training loop (8 chunks of 5 epochs each)
for i in $(seq 1 ${NUM_CHUNKS}); do
    echo -e "${GREEN}>>> Chunk ${i}/${NUM_CHUNKS} (epochs $(((i-1)*EPOCHS_PER_CHUNK+1))-$((i*EPOCHS_PER_CHUNK)) of continuation)${NC}"
    
    PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) python3 -u "${SCRIPT}" \
        --traj_dir "${TRAJ_DIR}" \
        --out_dir "${OUT_DIR}" \
        --epochs ${EPOCHS_PER_CHUNK} \
        --batch_size ${BATCH_SIZE} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --nlayers ${NLAYERS} \
        --distill_weight ${DISTILL_WEIGHT} \
        --distill_T ${DISTILL_T} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --device ${DEVICE} \
        2>&1 | tee -a "${LOG_FILE}" || {
            echo -e "${YELLOW}Training interrupted at chunk ${i}${NC}"
            break
        }
    
    echo -e "${GREEN}✓ Chunk ${i} completed${NC}\n"
done

echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Training completed!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Results saved to: ${OUT_DIR}"
echo "  - run.log: Training log with all metrics"
echo "  - ckpt_latest.pth: Final checkpoint (80 epochs)"
echo ""
echo "To compare with original results:"
echo "  1. Check final metrics in: ${LOG_FILE}"
echo "  2. Compare with commit 8d5f10b results:"
echo "     - Loss: should go from ~4.81 → ~3.59"
echo "     - Step imitation: expert≈0.628, atom≈0.791"
echo "     - Angle accuracy t=0: ≈0.405"
echo ""
echo "To view results:"
echo "  tail -50 ${LOG_FILE}"
echo ""
