# 480-Epoch Training Quick Guide

## Prerequisites

✅ **Required before running:**
1. Physics reconstruction implemented (commit dcfd7c0)
2. Data pipeline verified (see DATA_PIPELINE_EXPLANATION.md)
3. Conda environment: `trl-training`
4. Data files available:
   - H matrix: `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`
   - W matrix: `doa_normalized_config_c_corrected/models/usm.pth`
   - Dataset: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`

## Quick Start (3 Commands)

```bash
# 1. Activate environment
conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH

# 2. Run complete training pipeline
./run_full_training.sh

# 3. Monitor progress (in another terminal)
./monitor_training.sh results/dt_full_training_*/training.log
```

## What Happens

### Phase 1: OMP Trajectory Generation (~5-10 minutes)
- Generates baseline trajectories using traditional OMP
- Expected: ~65% first-step expert accuracy
- Output: `results/dt_traj_omp_full_*/trajectories.jsonl`

### Phase 2: Decision Transformer Training (~2-4 hours)
- Trains DT for 480 epochs on OMP trajectories
- Expected: Should match or exceed OMP baseline (94.6% in previous runs)
- Output: `results/dt_full_training_*/training.log`

### Phase 3: Results Summary
- Automatically extracts and displays key metrics
- Compares OMP baseline vs DT performance
- Guides next steps

## Expected Outputs

```
results/
├── dt_traj_omp_full_YYYYMMDD_HHMMSS/
│   ├── trajectories.jsonl          # Training data
│   ├── manifest.json                # Metadata
│   └── generation.log               # OMP metrics
└── dt_full_training_YYYYMMDD_HHMMSS/
    ├── training.log                 # Full training log
    ├── best_model.ckpt              # Best checkpoint
    ├── final_model.ckpt             # Final checkpoint
    └── final_metrics.npz            # Metrics archive
```

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch training progress
tail -f results/dt_full_training_*/training.log

# Check specific metrics
grep "Epoch" results/dt_full_training_*/training.log | tail -20
grep "Best test loss" results/dt_full_training_*/training.log
```

### Key Milestones
- **Epoch 10**: Early training stability check
- **Epoch 100**: Physics warmup complete (if enabled)
- **Epoch 240**: Midpoint - should show clear convergence
- **Epoch 480**: Final evaluation

## Expected Performance

Based on OMP baseline (commit a654557):
- **OMP first-step accuracy**: 64.9% (expert), 94.6% (atom)
- **DT target**: ≥94% expert accuracy (480 epochs)
- **Physics test (100 epochs)**: 38.3% expert (needs full 480 for fair comparison)

## Troubleshooting

### Common Issues

**1. Data not found**
```bash
# Verify data exists
ls -lh /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
ls -lh doa_normalized_config_c_corrected/models/usm.pth
ls -d /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized
```

**2. Environment not activated**
```bash
# Check conda env
conda env list | grep trl-training

# Activate if needed
conda activate trl-training
```

**3. PYTHONPATH not set**
```bash
# Verify PYTHONPATH
echo $PYTHONPATH | grep mdp-decision-transformer

# Set if needed
export PYTHONPATH=$(pwd):$PYTHONPATH
```

**4. Training crashes**
- Check memory: Training uses ~2-4GB RAM
- Check logs: Look for error messages in training.log
- Verify data: Ensure all .npy files are valid

## Customization

### Enable Physics Reconstruction
Edit `run_full_training.sh`:
```bash
# Line 44-49
USE_PHYSICS=true              # Enable physics tasks
PHYSICS_WEIGHT=0.1            # Overall physics weight
PHYSICS_WARMUP_EPOCHS=100     # Warmup period
RESIDUAL_WEIGHT=1.0           # Residual prediction
DIRECTION_WEIGHT=1.0          # Direction classification
COHERENCE_WEIGHT=0.1          # Spectral coherence
```

### Adjust Training Hyperparameters
```bash
# Line 52-58
EPOCHS=480          # Total epochs
BATCH_SIZE=4        # Batch size
LR=3e-3            # Learning rate
D_MODEL=128        # Model dimension
NHEAD=2            # Attention heads
NLAYERS=1          # Transformer layers
```

### Change OMP Settings
```bash
# Line 24-31
K=6                      # Selection budget
N_ATOMS=8                # Atom set size
ATOM_REDUCE_MODE="kcenter"  # Atom selection method
FS=16000                 # Sampling rate
N_FFT=2048              # FFT size
FREQ_MIN=300            # Low freq cutoff
FREQ_MAX=3000           # High freq cutoff
```

## After Training

### 1. Document Results
```bash
# Copy template
cp FULL_TRAINING_RESULTS_TEMPLATE.md results/FULL_TRAINING_RESULTS_$(date +%Y%m%d).md

# Fill in with actual metrics from logs
```

### 2. Commit Results
Follow AGENTS.md guidelines:
- Include all artifacts (logs, checkpoints, config)
- Document OMP baseline metrics
- Document DT final metrics
- Analyze comparison (BECAUSE/DUE TO/THEREFORE)
- Extract principles for future work

### 3. Next Experiments
Based on results:
- If DT >> OMP: Try harder tasks, increase model capacity
- If DT ≈ OMP: Investigate architectural improvements
- If DT < OMP: Debug training dynamics, check data quality

## References

- **Data pipeline**: DATA_PIPELINE_EXPLANATION.md
- **Physics details**: PHYSICS_RECONSTRUCTION_SUMMARY.md
- **OMP baseline**: OMP_TEACHER_VERIFICATION_RESULTS.md
- **Results template**: FULL_TRAINING_RESULTS_TEMPLATE.md
- **Project guidelines**: AGENTS.md

## Estimated Timeline

| Phase | Duration | Can Monitor? |
|-------|----------|--------------|
| OMP trajectory generation | 5-10 min | Yes (generation.log) |
| DT training (480 epochs) | 2-4 hours | Yes (training.log) |
| Results extraction | <1 min | Automatic |
| Documentation | 30-60 min | Manual |
| **Total** | **3-5 hours** | - |

**Recommendation:** Start training before a break or overnight run.
