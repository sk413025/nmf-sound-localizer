#!/bin/bash
################################################################################
# Setup Script: Copy LFS Files from Original Worktree
################################################################################
# This script copies all necessary LFS files from the original worktree
# to enable reproduction of experiments without relying on LFS server
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

ORIG_WORKTREE="/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/mdp-decision-transformer"
NEW_WORKTREE="/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Copying LFS Files from Original Worktree${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"

# Check if original worktree exists
if [ ! -d "${ORIG_WORKTREE}" ]; then
    echo -e "${YELLOW}ERROR: Original worktree not found at${NC}"
    echo "  ${ORIG_WORKTREE}"
    exit 1
fi

cd "${NEW_WORKTREE}"

echo -e "\n${YELLOW}[1/4] Copying trajectory data...${NC}"
rsync -av --progress "${ORIG_WORKTREE}/results/dt_traj_qk_kmeans/" results/dt_traj_qk_kmeans/
echo -e "${GREEN}✓ Trajectory data copied${NC}"

echo -e "\n${YELLOW}[2/4] Copying model files...${NC}"
rsync -av --progress "${ORIG_WORKTREE}/doa_normalized_config_c_corrected/" doa_normalized_config_c_corrected/
echo -e "${GREEN}✓ Model files copied${NC}"

echo -e "\n${YELLOW}[3/5] Copying teacher model...${NC}"
# Copy teacher model checkpoint
if [ -d "${ORIG_WORKTREE}/results/exp_H_qk_encoder_on_atom_d128_20251026_233228" ]; then
    rsync -av --progress "${ORIG_WORKTREE}/results/exp_H_qk_encoder_on_atom_d128_20251026_233228/" \
          "${NEW_WORKTREE}/results/exp_H_qk_encoder_on_atom_d128_20251026_233228/"
    echo -e "${GREEN}✓ Teacher model copied${NC}"
else
    echo -e "${YELLOW}⚠ Teacher model directory not found${NC}"
fi

echo -e "\n${YELLOW}[4/5] Copying 40-epoch checkpoint...${NC}"
# Checkout 0cfe0c1 in original worktree to get the 40-epoch checkpoint
cd "${ORIG_WORKTREE}"
git stash > /dev/null 2>&1 || true
git checkout 0cfe0c1 > /dev/null 2>&1

if [ -f "results/dt_min_qk_kmeans_distill/ckpt_latest.pth" ]; then
    CKPT_SIZE=$(wc -c < "results/dt_min_qk_kmeans_distill/ckpt_latest.pth")
    if [ ${CKPT_SIZE} -gt 1000000 ]; then
        cp "results/dt_min_qk_kmeans_distill/ckpt_latest.pth" \
           "${NEW_WORKTREE}/results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth"
        echo -e "${GREEN}✓ Checkpoint copied ($(ls -lh ${NEW_WORKTREE}/results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth | awk '{print $5}'))${NC}"
    else
        echo -e "${YELLOW}⚠ Checkpoint appears to be LFS pointer${NC}"
    fi
fi

# Restore original worktree
git checkout feature/mdp-decision-transformer > /dev/null 2>&1
git stash pop > /dev/null 2>&1 || true

echo -e "\n${YELLOW}[5/5] Verifying files...${NC}"
cd "${NEW_WORKTREE}"

# Check manifest.json
if python3 -c "import json; json.load(open('results/dt_traj_qk_kmeans/manifest.json'))" 2>/dev/null; then
    echo -e "${GREEN}✓ manifest.json is valid JSON${NC}"
else
    echo -e "${YELLOW}✗ manifest.json is not valid JSON${NC}"
    exit 1
fi

# Check usm.pth
if [ -f "doa_normalized_config_c_corrected/models/usm.pth" ]; then
    USM_SIZE=$(wc -c < "doa_normalized_config_c_corrected/models/usm.pth")
    if [ ${USM_SIZE} -gt 10000 ]; then
        echo -e "${GREEN}✓ usm.pth exists ($(ls -lh doa_normalized_config_c_corrected/models/usm.pth | awk '{print $5}'))${NC}"
    else
        echo -e "${YELLOW}⚠ usm.pth may be LFS pointer${NC}"
    fi
fi

# Check checkpoint
if [ -f "results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth" ]; then
    CKPT_SIZE=$(wc -c < "results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth")
    if [ ${CKPT_SIZE} -gt 1000000 ]; then
        echo -e "${GREEN}✓ checkpoint exists ($(ls -lh results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth | awk '{print $5}'))${NC}"
    else
        echo -e "${YELLOW}⚠ checkpoint may be LFS pointer${NC}"
    fi
fi

# Check teacher model
if [ -f "results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth" ]; then
    TEACHER_SIZE=$(wc -c < "results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth")
    if [ ${TEACHER_SIZE} -gt 1000000 ]; then
        echo -e "${GREEN}✓ teacher model exists ($(ls -lh results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth | awk '{print $5}'))${NC}"
    else
        echo -e "${YELLOW}⚠ teacher model may be LFS pointer${NC}"
    fi
fi

echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Setup completed!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "You can now run the reproduction script:"
echo "  ./reproduce_8d5f10b.sh"
echo ""
