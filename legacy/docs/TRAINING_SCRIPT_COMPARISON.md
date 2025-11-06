# Training Script Comparison: G-Teacher vs OMP Teacher

## Overview

Two variants of the 480-epoch Decision Transformer training pipeline are now available:

1. **`run_480epochs.sh`** - Uses pre-generated trajectories (originally QK-kmeans, can be adapted for G-teacher)
2. **`run_480epochs_omp.sh`** - Generates OMP trajectories and trains DT in one workflow

## Key Differences

### run_480epochs.sh (Original)
- **Trajectory Source**: Pre-generated from `results/dt_traj_qk_kmeans`
- **Teacher Type**: Depends on pre-generated trajectories (typically QK or G-teacher)
- **Workflow**: Single-phase (training only)
- **Expected Performance**: 98%+ test accuracy with G-teacher trajectories

### run_480epochs_omp.sh (New Variant)
- **Trajectory Source**: Generated on-the-fly using traditional OMP
- **Teacher Type**: Traditional OMP (pure greedy selection, no hierarchical structure)
- **Workflow**: Three-phase
  1. Generate OMP trajectories
  2. Verify trajectory quality (optional)
  3. Train Decision Transformer
- **Expected Performance**: ~65-70% test accuracy (matches OMP's 64.9% trajectory accuracy)

## Performance Comparison (Based on Commit a654557)

| Teacher | First-Step Accuracy | Expected DT Accuracy | Training Time |
|---------|---------------------|----------------------|---------------|
| **G-Teacher** | 100% | 98%+ | ~Several hours |
| **Traditional OMP** | 64.9% | ~65-70% | ~Several hours + trajectory generation |

## When to Use Each Script

### Use `run_480epochs.sh` for:
- ✅ Production training with maximum accuracy
- ✅ When you already have high-quality trajectories
- ✅ Faster iteration (no trajectory generation overhead)
- ✅ Reproducible experiments with fixed trajectories

### Use `run_480epochs_omp.sh` for:
- 🔬 Research/experimental comparison of teacher methods
- 🔬 Baseline to demonstrate value of hierarchical selection
- 🔬 Understanding how trajectory quality affects DT performance
- 🔬 Educational purposes (showing why structure matters)

## Usage Examples

### Running G-Teacher Pipeline (Recommended for Production)

```bash
# Step 1: Generate G-teacher trajectories
python doa_rl/trajectories/offline_dt_dataset.py \
  --teacher g --K 6 --n_atoms 8 --atom_reduce_mode kcenter \
  --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --seed 42 \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --out_dir results/dt_traj_g_480epochs \
  --device cpu

# Step 2: Modify run_480epochs.sh to use G-teacher trajectories
# Edit line 16: TRAJ_DIR="results/dt_traj_g_480epochs"

# Step 3: Run training
./run_480epochs.sh
```

### Running OMP Pipeline (For Research/Comparison)

```bash
# All-in-one script (generates trajectories + trains DT)
./run_480epochs_omp.sh

# The script will:
# - Generate OMP trajectories in results/dt_traj_omp_480epochs_<timestamp>
# - Verify trajectory quality (64.9% expected)
# - Train DT for 480 epochs
# - Save results to results/dt_min_480epochs_omp_<timestamp>
```

## Expected Outputs

### With G-Teacher Trajectories
```
Epoch 480/480:
  Train Loss: 0.025
  Test Loss: 0.031
  Test Accuracy: 98.2% ✓
  Test First-Step Accuracy: 98.9% ✓
```

### With OMP Trajectories
```
Epoch 480/480:
  Train Loss: 0.089
  Test Loss: 0.095
  Test Accuracy: 67.3% (limited by 64.9% trajectory quality)
  Test First-Step Accuracy: 69.1%
```

## Why OMP Underperforms

From first-principles analysis (commit a654557):

**Mathematical Insight:**
- **OMP optimizes**: `j_omp = argmax_j |d_j^T @ r|` (selects atom with highest correlation)
- **Problem**: Can select high-correlation atom from WRONG expert due to strong spectral component W
- **G-teacher optimizes**: `e = argmax_e sum_m |g_{e,m}|` (aggregates evidence across atoms per expert)
- **Advantage**: Averages M samples of each direction, robust to atom variability

**Physical Interpretation:**
- Dictionary structure: `D = H ⊙ W` where H encodes spatial direction, W encodes spectrum
- OMP ignores that experts = directions (spatial physics)
- G-teacher exploits hierarchical structure: commit to direction first, refine with atom second
- Result: 13/37 angles fail completely with OMP (0% accuracy) vs 37/37 succeed with G-teacher (100%)

## Recommendations

1. **For Production**: Use G-teacher (`run_480epochs.sh` with G-teacher trajectories)
   - Highest accuracy (98%+)
   - Proven reliable across all angles
   - Aligns algorithm with physics (hierarchical structure matches D = H ⊙ W)

2. **For Research**: Use OMP as baseline to demonstrate improvement
   - Shows value of domain knowledge (expert structure)
   - Quantifies cost of ignoring physics (35% accuracy drop)
   - Educational example of structure-aware vs structure-blind algorithms

3. **For Ablation Studies**: Compare side-by-side
   ```bash
   # Terminal 1: G-teacher training
   ./run_480epochs.sh  # (after generating G-teacher trajectories)
   
   # Terminal 2: OMP training
   ./run_480epochs_omp.sh
   
   # Compare final accuracies to confirm ~35% gap
   ```

## File Locations

- **Scripts**:
  - `run_480epochs.sh` - Original training script
  - `run_480epochs_omp.sh` - New OMP variant
  
- **Core Implementation**:
  - `doa_rl/trajectories/offline_dt_dataset.py` - Contains both `hierarchical_pick_g()` and `traditional_omp_pick()`
  
- **Verification Tools**:
  - `verify_omp_teacher_trajectories.py` - Compare OMP vs G-teacher accuracy
  - `analyze_omp_by_angle.py` - Per-angle breakdown
  
- **Documentation**:
  - `OMP_TEACHER_VERIFICATION_RESULTS.md` - Detailed experimental analysis
  - Commit `a654557` - Results commit with full findings

## References

- Commit `a654557`: Traditional OMP vs G-Teacher comparison
- Commit `63b8190`: G-teacher achieves 98%+ accuracy (proven baseline)
- `G_TEACHER_VERIFICATION_SUCCESS_20251106.md`: G-teacher 100% first-step accuracy verification

## Conclusion

While traditional OMP provides a useful baseline and educational example, **G-teacher is the recommended approach for production systems** due to its:
- Superior accuracy (100% vs 64.9% trajectory quality)
- Physical alignment (hierarchical selection matches D = H ⊙ W structure)
- Robustness (no angle-level failures)

The `run_480epochs_omp.sh` script exists primarily for research purposes and to demonstrate the importance of structure-aware algorithm design.
