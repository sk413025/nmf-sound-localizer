# Experiment Reproduction Guide: Commit 8d5f10b

## Overview
This document explains how to reproduce the experiment from commit `8d5f10bc0b008a6b924422d3c8d63ee59cdbf59e`:
**"Results: DT‑min on K‑means QK trajectories — extended to 80 epochs (atomic)"**

## Experiment Background

### What This Experiment Does
- Continues training a Decision Transformer (DT-min) model on K-means QK trajectories
- Extends training from **40 epochs → 80 epochs**
- Uses teacher distillation with K-means atomic actions
- Training is done in **8 chunks of 5 epochs each** for incremental monitoring

### Key Configuration
- **Trajectory Data**: `results/dt_traj_qk_kmeans/`
- **Output Directory**: `results/dt_min_qk_kmeans_distill/` (original) or `results/dt_min_qk_kmeans_distill_reproduction/` (reproduction)
- **Training Script**: `scripts/dt_pointer_ldv.py`

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `--epochs` | 5 | Epochs per chunk (run 8 times) |
| `--batch_size` | 8 | Batch size for training |
| `--d_model` | 128 | Transformer embedding dimension |
| `--nhead` | 2 | Number of attention heads |
| `--nlayers` | 1 | Number of transformer layers |
| `--distill_weight` | 0.7 | Weight for distillation loss |
| `--distill_T` | 1.0 | Temperature for distillation |
| `--warmup_epochs` | 3 | Warmup epochs for learning rate |
| `--device` | cpu | Training device |

## Expected Results

From the commit message, the final chunk (epochs 76-80) should show:

### Loss
- Starting: ~4.81
- Final: ~3.59

### Imitation Accuracy
- **Expert-level**: ~0.628
- **Atom-level**: ~0.791

### Teacher-Forced Step Match
- ~0.577

### Angle Accuracy
- **DT-min at t=0**: ~0.405
- **DT-min at t=K-1**: ~0.351
- **Teacher QK at t=0**: ~0.946
- **Ground truth g(y)**: 1.000

## How to Reproduce

### Option 1: Quick Start (Automated Script)

Run the provided reproduction script:

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer
./reproduce_8d5f10b.sh
```

This script will:
1. Validate the environment and required files
2. Check for the checkpoint from 40 epochs
3. Run 8 training chunks (5 epochs each)
4. Save results to `results/dt_min_qk_kmeans_distill_reproduction/`

### Option 2: Manual Execution

If you prefer to run commands manually:

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer

# Create output directory
mkdir -p results/dt_min_qk_kmeans_distill_reproduction

# Run training (8 chunks of 5 epochs each)
for i in {1..8}; do \
  PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) python3 -u scripts/dt_pointer_ldv.py \
    --traj_dir results/dt_traj_qk_kmeans \
    --out_dir results/dt_min_qk_kmeans_distill_reproduction \
    --epochs 5 --batch_size 8 --d_model 128 --nhead 2 --nlayers 1 \
    --distill_weight 0.7 --distill_T 1.0 --warmup_epochs 3 --device cpu \
    2>&1 | tee -a results/dt_min_qk_kmeans_distill_reproduction/run.log || break; \
done
```

### Option 3: Start from 40-Epoch Checkpoint (Exact Reproduction)

To exactly reproduce starting from the 40-epoch state:

```bash
# 1. Checkout the 40-epoch state
git checkout 0cfe0c1

# 2. If LFS is available, pull the checkpoint
git lfs pull

# 3. Copy the checkpoint
mkdir -p results/dt_min_qk_kmeans_distill_reproduction
cp results/dt_min_qk_kmeans_distill/ckpt_latest.pth \
   results/dt_min_qk_kmeans_distill_reproduction/

# 4. Return to the current branch
git checkout ldv-mdp-decision-transformer

# 5. Run the reproduction script
./reproduce_8d5f10b.sh
```

## Understanding the Output

### Training Log (`run.log`)

Each 5-epoch chunk produces output like:

```
================================================================================
DT-Min Trainer (hierarchical pointer)
================================================================================
Traj dir: results/dt_traj_qk_kmeans
Out dir: results/dt_min_qk_kmeans_distill_reproduction
D: F=346, E=37, M=8, P=296
[Resume] Loaded checkpoint from results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth
Epoch 1/5: loss=X.XXXX
Epoch 2/5: loss=X.XXXX
...
Final train step-acc: expert=X.XXX, atom=X.XXX
Teacher-forced step match: X.XXX
Angle acc — DT-min t=0: X.XXX; DT-min t=K-1: X.XXX; teacher qk t=0: X.XXX; g(y): X.XXX
```

### Key Metrics to Monitor

1. **Loss**: Should decrease from ~7.4 (epoch 1) to ~3.6 (epoch 80)
2. **Expert step-acc**: Accuracy at expert-level action selection
3. **Atom step-acc**: Accuracy at atomic action selection within expert
4. **Teacher-forced step match**: How well the model matches teacher actions
5. **Angle accuracy**:
   - `DT-min t=0`: Model's angle accuracy at first timestep
   - `DT-min t=K-1`: Model's angle accuracy at last timestep
   - `teacher qk t=0`: Teacher's angle accuracy (baseline)

### Checkpoint File

- **Location**: `results/dt_min_qk_kmeans_distill_reproduction/ckpt_latest.pth`
- **Contents**: Model state, optimizer state, training history
- **Size**: ~8MB (if not LFS pointer)

## Troubleshooting

### Issue: "Checkpoint not found" or checkpoint is too small

**Cause**: LFS files not downloaded

**Solution**:
```bash
git lfs install
git lfs pull
```

Or start training from scratch (will not match original results exactly).

### Issue: "CUDA out of memory"

**Cause**: GPU memory insufficient

**Solution**: The experiment uses `--device cpu`, which should work on any machine. If you want to use GPU, ensure sufficient memory or reduce `--batch_size`.

### Issue: Dependencies missing

**Required packages**:
```bash
pip install torch numpy
```

Check project requirements:
```bash
cat requirements.txt  # if exists
```

### Issue: Training diverges or produces different results

**Possible causes**:
1. Different random seed (not set in original experiment)
2. Different PyTorch version
3. Starting from different checkpoint state
4. Different trajectory data

**To verify trajectory data**:
```bash
cat results/dt_traj_qk_kmeans/manifest.json
```

## Comparing with Original Results

After training completes, compare your results with the original:

```bash
# View final metrics from reproduction
tail -30 results/dt_min_qk_kmeans_distill_reproduction/run.log

# View original results (if available)
git show 8d5f10b:results/dt_min_qk_kmeans_distill/run.log | tail -30
```

### Expected Progression (Approximate)

| Epoch Range | Loss (start → end) | Expert Acc | Atom Acc | Angle t=0 |
|-------------|-------------------|------------|----------|-----------|
| 1-5         | 7.4 → 5.6         | ~0.32      | ~0.51    | ~0.14     |
| 36-40       | 6.9 → 5.2         | ~0.39      | ~0.52    | ~0.19     |
| 76-80       | 4.8 → 3.6         | ~0.63      | ~0.79    | ~0.41     |

## Next Steps After Reproduction

From the commit message, suggested next experiments:

1. **Extend to 120 epochs**: Continue training with same hyperparameters
2. **Use sklearn K-means**: Switch to sklearn implementation for comparison
3. **Controllability test (B)**: Test RTG augmentation effects

## File Structure

```
results/
├── dt_traj_qk_kmeans/           # Input: Trajectory data
│   ├── trajectories.jsonl
│   ├── manifest.json
│   └── ...
├── dt_min_qk_kmeans_distill/    # Original experiment outputs
│   ├── ckpt_latest.pth
│   └── run.log
└── dt_min_qk_kmeans_distill_reproduction/  # Your reproduction
    ├── ckpt_latest.pth          # Final checkpoint (80 epochs)
    ├── run.log                  # Complete training log
    └── controllability.jsonl    # (if generated)
```

## References

- **Commit**: `8d5f10bc0b008a6b924422d3c8d63ee59cdbf59e`
- **Parent Commit** (40 epochs): `0cfe0c1`
- **Next Commit** (120 epochs): `7f37266`
- **Training Script**: `scripts/dt_pointer_ldv.py`
- **Trajectory Builder**: See earlier commits for trajectory generation

---

**Last Updated**: October 28, 2025  
**Worktree**: `/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer`  
**Branch**: `ldv-mdp-decision-transformer`
