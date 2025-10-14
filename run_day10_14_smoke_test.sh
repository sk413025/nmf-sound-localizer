#!/bin/bash
# Day 10-14 Experimental Validation - Smoke Test
# Purpose: Quick validation that multi-modal ICL system works end-to-end
# Usage: bash run_day10_14_smoke_test.sh

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================================================="
echo " Day 10-14: Experimental Validation - Smoke Test"
echo " Testing: Baseline vs Multi-Modal Comparison"
echo "==========================================================================${NC}"
echo ""

# Navigate to worktree
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f

# Set PYTHONPATH to include current directory
export PYTHONPATH="/Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f:$PYTHONPATH"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install conda first.${NC}"
    exit 1
fi

# Activate environment (adjust if needed)
echo -e "${YELLOW}Using current Python environment...${NC}"

# Verify Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# Verify doa_rl is installed
python -c "import doa_rl" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Installing doa_rl package...${NC}"
    pip install -e . --quiet
fi

echo -e "${GREEN}✓ Environment ready${NC}"
echo ""

# Create results directory
mkdir -p results/day10_14_smoke

# ============================================================================
# Step 1: Create Synthetic Test Data (if needed)
# ============================================================================
echo -e "${BLUE}=========================================================================="
echo " Step 1/3: Prepare Test Data"
echo "==========================================================================${NC}"

python << 'PYTHON_SCRIPT'
import os
from pathlib import Path
import numpy as np
import torch

# Create synthetic data directory structure
data_root = Path("doa_normalized_config_c_corrected")

# Check if we need to create synthetic data
angles = [80, 85, 90, 95, 100]
needs_data = False

for angle in angles:
    angle_dir = data_root / f"angle_{angle:03d}"
    if not angle_dir.exists() or len(list(angle_dir.glob("*.npy"))) < 5:
        needs_data = True
        break

if needs_data:
    print("Creating synthetic audio data for smoke test...")
    np.random.seed(42)
    
    for angle in angles:
        angle_dir = data_root / f"angle_{angle:03d}"
        angle_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 5 synthetic audio clips per angle
        for i in range(5):
            # Synthetic audio: 145920 samples (matching expected format)
            audio = np.random.randn(145920).astype(np.float32) * 0.1
            clip_path = angle_dir / f"clip_{i:03d}.npy"
            np.save(clip_path, audio)
        
        print(f"  Created {angle_dir} with 5 clips")
    
    print("✓ Synthetic data created")
else:
    print("✓ Test data already exists")

# Verify W and H matrices exist
w_path = data_root / "models" / "usm.pth"
h_path = Path("h_matrix_normalized_original_to_box.pth")

if not w_path.exists():
    print("\nCreating synthetic W matrix...")
    w_path.parent.mkdir(parents=True, exist_ok=True)
    W = torch.randn(116, 50)  # (F, K)
    torch.save({"W": W}, w_path)
    print(f"  Created {w_path}")

if not h_path.exists():
    print("\nCreating synthetic H matrix...")
    H = torch.randn(116, 17)  # (F, D)
    angles_h = list(range(80, 101, 5))  # 80-100 degrees
    torch.save({"H": H, "angles": angles_h}, h_path)
    print(f"  Created {h_path}")

print("\n✓ All required data files ready")
PYTHON_SCRIPT

echo -e "${GREEN}✓ Step 1 Complete${NC}"
echo ""

# ============================================================================
# Step 2: Run Baseline Experiment (Patch-only)
# ============================================================================
echo -e "${BLUE}=========================================================================="
echo " Step 2/3: Baseline Experiment (Patch-only tokens)"
echo "==========================================================================${NC}"

python scripts/train_reward_model_lora.py \
    --data-root doa_normalized_config_c_corrected \
    --tf-path h_matrix_normalized_original_to_box.pth \
    --w-path doa_normalized_config_c_corrected/models/usm.pth \
    --s-root doa_normalized_config_c_corrected \
    --K 2 \
    --teacher fit \
    --rm-epochs 2 \
    --batch-size 2 \
    --max-samples 5 \
    --lora-r 4 \
    --lora-alpha 8 \
    --device cpu \
    --out results/day10_14_smoke/baseline_rm

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Baseline experiment completed${NC}"
else
    echo -e "${RED}✗ Baseline experiment failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 3: Run Multi-Modal Experiment
# ============================================================================
echo -e "${BLUE}=========================================================================="
echo " Step 3/3: Multi-Modal Experiment (Direction + Atom + Patch tokens)"
echo "==========================================================================${NC}"

python scripts/train_reward_model_lora.py \
    --data-root doa_normalized_config_c_corrected \
    --tf-path h_matrix_normalized_original_to_box.pth \
    --w-path doa_normalized_config_c_corrected/models/usm.pth \
    --s-root doa_normalized_config_c_corrected \
    --use-multi-modal \
    --token-ordering physics_first \
    --max-tokens 150 \
    --top-k-atoms 5 \
    --top-m-directions 3 \
    --n-atoms 50 \
    --K 2 \
    --teacher fit \
    --rm-epochs 2 \
    --batch-size 2 \
    --max-samples 5 \
    --lora-r 4 \
    --lora-alpha 8 \
    --device cpu \
    --out results/day10_14_smoke/multimodal_rm

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Multi-modal experiment completed${NC}"
else
    echo -e "${RED}✗ Multi-modal experiment failed${NC}"
    exit 1
fi
echo ""

# ============================================================================
# Step 4: Summary
# ============================================================================
echo -e "${GREEN}=========================================================================="
echo " Smoke Test Summary"
echo "==========================================================================${NC}"
echo ""
echo "✓ All experiments completed successfully!"
echo ""
echo "Results saved to:"
echo "  - Baseline:    results/day10_14_smoke/baseline_rm_*"
echo "  - Multi-modal: results/day10_14_smoke/multimodal_rm_*"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Compare the training logs to validate multi-modal improvements"
echo "  2. Run full experiments with more epochs and samples"
echo "  3. Use scripts/evaluate_comparison.py for detailed analysis"
echo ""
echo -e "${GREEN}Day 10-14 smoke test completed successfully! 🎉${NC}"
