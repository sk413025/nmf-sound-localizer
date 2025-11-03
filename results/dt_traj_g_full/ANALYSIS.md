# dt_traj_g_full: Analysis Report

**Date**: 2025-11-03  
**Commit**: 0fca4de7814ac070bf3d15aa4fb1c1f613082f3d  
**Purpose**: Generate offline DT trajectories using classic OMP (g-teacher)

---

## Executive Summary

Successfully generated **111 OMP trajectories** (37 angles × 3 clips) using the classic greedy selection policy based on **|g| = |D^T r| energies**. These trajectories serve as the baseline teacher for Decision Transformer training and provide ground truth for comparing different OMP variants.

---

## Configuration

### Teacher Policy: g (Classic OMP)
- **Selection criterion**: `argmax |D^T r|` (inner product energies)
- **Hierarchical**: Two-level (expert → atom)
  - Expert selection: `e* = argmax Σ_m |g_{e,m}|` (L1 aggregation)
  - Atom selection: `m* = argmax |g_{e*,m}|`
- **No learning**: Purely geometric/physical-based selection

### Dictionary Configuration
- **H matrix**: (F=346, E=37) - Transfer functions for 37 angles
- **W matrix**: (F=346, M=8) - 8 atoms selected from 50 via k-center clustering
- **D = H ⊙ W**: (F=346, P=296) - 296 total atoms (37 × 8)
- **Mutual coherence**: μ_max=0.9977, μ_mean=0.3034

### STFT Configuration
- **Sampling rate**: fs=16,000 Hz
- **FFT size**: n_fft=2048
- **Frequency band**: [300, 3000] Hz → F=346 bins
- **Time averaging**: Mean over all frames (converts to static spectrum)

### Trajectory Parameters
- **K (steps)**: 6 greedy selections per trajectory
- **RTG targets**: 
  - Residual: ||r||² ≤ 0.02
  - Accuracy: p_true ≥ 0.95
- **Softmax temperature**: T=1.0 (for angle probability estimation)

---

## Results

### Overall Statistics (111 samples)

#### Final Residual ||r_K||²
- **Mean**: 0.1913
- **Std**: 0.0344
- **Range**: [0.1274, 0.2746]
- **Interpretation**: After K=6 steps, ~81% energy explained on average

#### Final Angle Confidence p_true
- **Mean**: 0.0431
- **Std**: 0.0053
- **Range**: [0.0359, 0.0615]
- **Interpretation**: Classic OMP has **low discrimination** (~4.3% vs random 2.7%)
  - This is expected: greedy g-based selection doesn't optimize for angle classification
  - Confirms need for learned routing (qk-teacher shows ~95% at t=0)

### Per-Step Progress

#### Δ||r||² (Residual Reduction per Step)
- **Mean of means**: 0.1348
- **Mean of medians**: 0.0711
- **Distribution**: Heavy-tailed (first steps contribute more)
  - Step 0: ~40% reduction
  - Step 1: ~19% reduction
  - Step 5: ~1.3% reduction

#### RTG Distribution (Return-to-Go across all steps)
- **RTG_resid**: mean=0.2985, range=[0.1074, 0.6470]
- **RTG_acc**: mean=0.9083, range=[0.8838, 0.9202]
- **Interpretation**: Large gap to targets → high RTG values throughout

### Action Space Coverage
- **Experts used**: 37/37 (100% coverage)
- **Atoms used**: 8/8 (100% coverage)
- **No dead zones**: All actions reachable from dataset distribution

---

## Sample Trajectory Breakdown

**Angle**: 0.0° (index=0)  
**Path**: `angle_0/clip_000.npy`

| Step | Expert | Atom | Δ&#124;&#124;r&#124;&#124;² | p_true | RTG_resid | RTG_acc |
|------|--------|------|------------|--------|-----------|---------|
| 0    | 0      | 6    | 0.4048     | 0.0371 | 0.5752    | 0.9129  |
| 1    | 0      | 2    | 0.1909     | 0.0409 | 0.3843    | 0.9091  |
| 2    | 22     | 3    | 0.0982     | 0.0389 | 0.2861    | 0.9111  |
| 3    | 0      | 7    | 0.0643     | 0.0416 | 0.2219    | 0.9084  |
| 4    | 19     | 4    | 0.0342     | 0.0418 | 0.1877    | 0.9082  |
| 5    | 25     | 1    | 0.0127     | 0.0410 | 0.1750    | 0.9090  |

**Observations**:
1. **Expert stickiness**: Expert 0 (true angle) selected 3/6 times
2. **Atom diversity**: All different atoms (no duplicates within expert)
3. **Monotonic residual decrease**: Each step reduces ||r||²
4. **Low p_true throughout**: Hovers around 4% (near uniform 2.7%)
5. **Off-angle selections**: Steps 2, 4, 5 choose experts far from true angle (22, 19, 25)

---

## Physical Analysis

### First Principles: Why Low p_true?

**Energy vs. Angle Discrimination**:
- g-based selection maximizes `|D^T r|` → **reconstruction fidelity**
- Does NOT optimize for `max Σ_m |g_{e,m}|` where `e = true_angle`
- Atoms from different experts can have high correlations (μ_max=0.9977)
- **Causal chain**: High μ → energy leaks to off-angle atoms → greedy selects highest |g| regardless of angle → low p_true

**Mathematical Relationship**:
```
p_true = softmax(Σ_m |D_e^T ŷ|)[e_true]
```
Where ŷ = D_S x_S (LS reconstruction from selected atoms)

If S contains many off-angle atoms with high |g|:
- ŷ becomes a "mixture" representation
- Energy spreads across multiple experts
- p_true remains low (~4%)

**Physical Constraint**: Classic OMP is **agnostic to angle labels** - it only sees geometry (inner products). Without supervised signal, it cannot learn to cluster selections by angle.

---

## Cross-Experiment Analysis

### Pattern Recognition (vs. qk-teacher trajectories from commit fe4ef60)

| Metric              | g-teacher (this) | qk-teacher (fe4ef60) |
|---------------------|------------------|----------------------|
| Final ||r||²        | 0.1913           | ~0.15 (estimated)    |
| Final p_true        | 0.0431           | ~0.95 (documented)   |
| t=0 accuracy        | N/A              | 94.6%                |
| Expert coverage     | 37/37            | 37/37                |
| Atom coverage       | 8/8              | 8/8                  |

**Success Factors** (qk vs g):
- qk-teacher achieves **22x higher p_true** BECAUSE:
  1. Supervised training with angle labels
  2. QK scores optimized for angle discrimination
  3. Expert-level normalization reduces μ effects
  4. Learned to prioritize discriminative atoms over high-energy atoms

**Failure Mode** (g-teacher):
- Low p_true DUE TO:
  1. No supervision signal (purely geometric)
  2. High dictionary coherence μ_max=0.9977
  3. Greedy optimization of wrong objective (reconstruction vs. classification)

### Method Effectiveness

**When g-teacher works**:
- Reconstruction tasks (minimize ||r||²)
- Low-coherence dictionaries (μ_max < 0.3)
- Single-source localization (no angle discrimination needed)

**When g-teacher fails**:
- Multi-class angle discrimination
- High-coherence dictionaries
- Requires semantic understanding of atom groupings

---

## Extracted Principles

### Design Principles
**THEREFORE**:
1. Use g-teacher as **baseline** for DT training comparisons
2. Expect DT trained on g-trajectories to learn reconstruction but NOT angle discrimination
3. Reserve qk-teacher for angle-discriminative tasks
4. Consider hybrid: g for first K/2 steps (rough reconstruction), qk for refinement

### Hypothesis Formation
**GIVEN** the 22x p_true gap:
- Predict DT trained on g-trajectories will achieve high reconstruction fidelity but low test accuracy
- Predict DT trained on qk-trajectories will achieve both high reconstruction AND high accuracy
- Predict DT can learn to improve upon teacher IF given RTG targets beyond teacher's capability

### Resource Allocation
**BASED ON** the energy-vs-accuracy tradeoff:
- Invest in qk-teacher training for production systems (accuracy matters)
- Use g-teacher for ablation studies and computational budget constraints
- Prioritize dictionary coherence reduction (μ_max↓) over algorithm sophistication for g-teacher

### Risk Mitigation
- Always report BOTH ||r||² and p_true metrics (reconstruction ≠ discrimination)
- Verify teacher policy evaluation BEFORE generating full trajectories
- Include action space coverage checks (avoid dead zones)

---

## Reproducibility

### Dataset Lineage
- **Root**: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`
- **Fingerprint (MD5)**: `713c0635878a04b32f4ee30208904d11`
- **File count**: 111 .npy files
- **Preprocessing**: Sync + VAD + normalization (from commit c96860b)

### Reproduction Commands
```bash
# Full reproduction
./REPRODUCE_dt_traj_g_full.sh

# Manual reproduction
export PYTHONPATH=/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer:$PYTHONPATH
python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher g \
    --K 6 \
    --n_atoms 8 \
    --atom_reduce_mode kcenter \
    --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
    --w_path doa_normalized_config_c_corrected/models/usm.pth \
    --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
    --out_dir results/dt_traj_g_full \
    --device cpu \
    --seed 42 \
    --fs 16000 \
    --n_fft 2048 \
    --freq_min 300.0 \
    --freq_max 3000.0 \
    --rtg_target_resid 0.02 \
    --rtg_target_acc 0.95 \
    2>&1 | tee results/dt_traj_g_full/run.log
```

### Verification
```bash
# Check outputs
ls -lah results/dt_traj_g_full/
# Expected: trajectories.jsonl (111 lines), numeric_diagnostics.jsonl (111 lines),
#           manifest.json, code_state.json, run.log

# Verify trajectory count
wc -l results/dt_traj_g_full/*.jsonl
# Expected: 111 trajectories, 111 diagnostics

# Verify data integrity
python -c "
import json
with open('results/dt_traj_g_full/manifest.json') as f:
    m = json.load(f)
assert m['fingerprint_md5'] == '713c0635878a04b32f4ee30208904d11'
assert len(m['samples']) == 111
assert m['teacher'] == 'g'
print('✅ Verification passed')
"
```

---

## Next Steps

### Immediate (DT Training)
1. **Train DT on g-trajectories** (this dataset):
   ```bash
   python scripts/dt_pointer_ldv.py \
       --traj_dir results/dt_traj_g_full \
       --out_dir results/dt_min_g_full \
       --epochs 480 \
       --batch_size 16 \
       --lr 3e-3 \
       --device cpu
   ```
   - Expected: High reconstruction, low test accuracy
   - Serves as baseline for DT capability assessment

2. **Generate qk-teacher trajectories** (for comparison):
   ```bash
   python doa_rl/trajectories/offline_dt_dataset.py \
       --teacher qk \
       --qk_ckpt results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth \
       --out_dir results/dt_traj_qk_full \
       [same other args as g-teacher]
   ```
   - Expected: High reconstruction AND high p_true

3. **Train DT on qk-trajectories**:
   - Same DT architecture as step 1
   - Compare final test accuracy g-DT vs qk-DT
   - Assess if DT can match teacher performance

### Future Work
1. **Hybrid teacher**: g for first 3 steps, qk for last 3 steps
2. **RTG ablation**: Train DT with different RTG targets (aggressive vs conservative)
3. **Dictionary coherence study**: Retrain USM with μ_max < 0.5, regenerate trajectories
4. **Multi-step lookahead**: Extend OMP to 2-step beam search, compare vs greedy

---

## Artifacts

All results stored under `results/dt_traj_g_full/`:
- **trajectories.jsonl**: 111 lines, 171 KB (step-by-step actions + diagnostics)
- **numeric_diagnostics.jsonl**: 111 lines, 68 KB (per-sample summary stats)
- **manifest.json**: 21 KB (configuration + dataset metadata)
- **code_state.json**: 1.3 KB (git hash + file SHA256s + args)
- **run.log**: 2.3 KB (stdout/stderr from execution)

**Total size**: ~264 KB (lightweight, suitable for version control)

---

## Appendix: Key Equations

### g-based Selection (Classic OMP)
```
g = D^T r                          # (P,) correlation energies
g_em = reshape(g, [E, M])          # (E, M) expert-atom grid
energy_e = Σ_m |g_{e,m}|           # (E,) L1 aggregation
e* = argmax energy_e
m* = argmax |g_{e*,m}|
```

### Residual Update (Orthogonal Projection)
```
S_{t+1} = S_t ∪ {j}               # Add selected atom index
D_S = D[:, S_{t+1}]                # (F, |S|) selected dictionary
x_S = (D_S^T D_S)^{-1} D_S^T y     # LS solution
ŷ = D_S x_S                        # Reconstruction
r_{t+1} = y - ŷ                    # Updated residual
```

### Angle Probability (Softmax over Expert Energies)
```
ĝ = D^T ŷ                          # (P,) energies from reconstruction
ĝ_em = reshape(ĝ, [E, M])          # (E, M) grid
energy_e = Σ_m |ĝ_{e,m}|           # (E,) per-expert energies
logits_e = energy_e / T            # Temperature-scaled
p = softmax(logits_e)              # (E,) angle probabilities
p_true = p[e_true]                 # Confidence for true angle
```

### RTG (Return-to-Go)
```
RTG_resid = max(0, ||r_t||² - target_resid)
RTG_acc = max(0, target_acc - p_true_t)
```

---

**End of Report**
