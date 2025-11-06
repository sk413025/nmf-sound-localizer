# Traditional OMP vs G-Teacher Comparison
## Date: 2025-11-06

## 📋 Experiment Objective

Compare traditional OMP (Orthogonal Matching Pursuit) with hierarchical G-teacher for acoustic source localization trajectory generation.

**Research Question**: Does the hierarchical expert-first structure of G-teacher provide meaningful advantages over traditional greedy OMP?

---

## ✅ Key Results

| Metric | Traditional OMP | G-Teacher | Difference |
|--------|----------------|-----------|------------|
| **First-step accuracy** | 72/111 = **64.9%** | 111/111 = **100.0%** | **-35.1%** ❌ |
| **GT expert in sequence** | 108/111 = 97.3% | 111/111 = 100.0% | -2.7% |
| **Angles with 100% accuracy** | 24/37 = 64.9% | 37/37 = 100.0% | -35.1% |
| **First-step agreement (OMP vs G)** | 72/111 = 64.9% | - | - |

**Conclusion**: Traditional OMP has **significantly lower accuracy** than G-teacher (64.9% vs 100%). The hierarchical structure provides substantial benefit.

---

## 🔍 Detailed Algorithm Comparison

### Traditional OMP
```python
def traditional_omp_pick(D, r, E, M):
    """
    Pure greedy: select atom with max |D^T @ r|
    No structure awareness
    """
    g = D.T @ r  # (P,)
    j = argmax(|g|)  # Select from all P atoms
    e = j // M  # Derive expert post-hoc
    m = j % M
    return e, m, j
```

**Characteristics**:
- ✓ Optimal single-atom correlation
- ✗ Ignores expert (direction) structure
- ✗ May select high-correlation atom from wrong expert

### G-Teacher (Hierarchical)
```python
def hierarchical_pick_g(D, r, E, M):
    """
    Two-stage: expert first, then atom
    Leverages dictionary structure
    """
    g = D.T @ r  # (P,)
    g_em = g.view(E, M)  # Reshape to (E, M)
    
    # Stage 1: Select expert
    energy_e = sum_m |g_{e,m}|  # Total energy per expert
    e = argmax(energy_e)
    
    # Stage 2: Select atom within expert
    m = argmax(|g_{e,·}|)
    j = e * M + m
    return e, m, j
```

**Characteristics**:
- ✓ First commits to direction (expert)
- ✓ Then refines with atom selection
- ✓ Leverages D = H ⊙ W structure
- ✓ More robust to atom correlations

---

## 📊 Angle-by-Angle Analysis

### Perfect Angles for OMP (24/37)
Angles where OMP achieves 100% accuracy:
- 0°, 10°, 15°, 25°, 30°, 35°, 40°, 45°, 50°, 60°
- 85°, 90°, 110°, 115°, 135°, 140°, 145°, 150°
- 155°, 160°, 165°, 170°, 175°, 180°

### Failed Angles for OMP (13/37)
Angles where OMP achieves **0%** accuracy:
- 5°, 20°, 55°, 65°, 70°, 75°, 80°
- 95°, 100°, 105°, 120°, 125°, 130°

**Pattern**: Failed angles are scattered across the angular space (no obvious clustering), suggesting the issue is atom-level rather than spatial.

---

## 🧮 First-Principles Analysis

### Why Does OMP Fail?

**Physical explanation**:

1. **Dictionary Structure**: D = H ⊙ W
   - Each expert e has M atoms: {D[:, e*M], ..., D[:, e*M+M-1]}
   - Atoms within expert share direction H[:, e]
   - But differ in W component (spectral characteristics)

2. **OMP Selection**:
   ```
   j_omp = argmax_j |d_j^T @ r|
   ```
   - Selects atom with highest individual correlation
   - May pick high-correlation atom from **wrong expert**
   - E.g., if d_{e',m'} has high correlation due to W component
     but e' ≠ e_true, direction is wrong

3. **G-Teacher Selection**:
   ```
   e_g = argmax_e sum_m |d_{e,m}^T @ r|
   ```
   - Aggregates evidence across all atoms for each expert
   - More robust to individual atom variations
   - Commits to correct direction first

**Mathematical insight**:
```
For OMP:  max_j |h_{j//M}^T r · w_{j%M}^T r|
For G:    max_e sum_m |h_e^T r · w_m^T r|
```

OMP can be fooled by one strong W component, while G-teacher averages over M samples of the direction.

---

## 💡 Why G-Teacher Achieves 100%

**Success factors**:

1. **Energy Aggregation**: 
   - `energy_e = sum_m |g_{e,m}|` provides robust direction estimate
   - Reduces impact of atom-level noise/variability

2. **Two-Stage Decision**:
   - First stage: Find correct direction (expert)
   - Second stage: Refine with best atom within that direction
   - Separation of concerns improves robustness

3. **Physical Alignment**:
   - Expert = spatial direction (physical constraint)
   - Atom = spectral characteristics (optional refinement)
   - Hierarchy matches physics: direction dominates

---

## 📈 Implications for Decision Transformer Training

### Using OMP Trajectories (Not Recommended)

**Risks**:
- ❌ Only 64.9% first-step accuracy → noisy supervision
- ❌ 13/37 angles completely fail → biased training distribution
- ❌ DT learns from suboptimal demonstrations
- ❌ May learn wrong patterns (atom correlations vs direction)

**Expected DT performance**:
- Lower test accuracy ceiling (~65-70% vs ~98%)
- Slower convergence
- Potential angle-specific failure modes

### Using G-Teacher Trajectories (Recommended)

**Benefits**:
- ✅ 100% first-step accuracy → clean supervision
- ✅ All angles succeed → unbiased training
- ✅ DT learns correct strategy (direction-first)
- ✅ Proven success: 98%+ test accuracy in commit 63b8190

---

## 🔬 Experiment Configuration

### Matched Parameters (Both Teachers)
```bash
K=6                         # Trajectory length
N_ATOMS=8                   # Atoms per expert
ATOM_REDUCE_MODE="kcenter"  # Atom reduction method
FS=16000                    # Sampling rate
N_FFT=2048                  # FFT size
FREQ_MIN=300.0              # Min frequency (Hz)
FREQ_MAX=3000.0             # Max frequency (Hz)
SEED=42                     # Random seed
```

### Dictionary Configuration
- **H matrix**: (346, 37) - 37 expert directions
- **W matrix**: (346, 50) → kcenter reduction → (346, 8)
- **Dictionary D**: (346, 296) - 37×8 = 296 atoms
- **Mutual coherence**: μ_max=0.9977, μ_mean=0.3034

### Dataset
- **Source**: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`
- **Total samples**: 111 (37 angles × 3 clips)
- **Angle range**: 0° - 180° (every 5°)
- **Data fingerprint**: `713c0635878a04b32f4ee30208904d11`

---

## 🚀 Recommendations

### 1. Use G-Teacher for Production (Confirmed)

**Evidence**:
- 35.1% accuracy advantage
- 100% vs 64.9% first-step accuracy
- Proven DT training success

**Command**:
```bash
python doa_rl/trajectories/offline_dt_dataset.py \
  --teacher g \
  --K 6 \
  --n_atoms 8 \
  --out_dir results/dt_traj_g_production
```

### 2. Traditional OMP Not Recommended

Traditional OMP is **unsuitable** for this task due to:
- Systematic failures on 13/37 angles
- 35% lower accuracy than hierarchical approach
- Physical mismatch (ignores direction structure)

### 3. Future Research Directions

**If exploring alternatives to G-teacher**:

a) **Weighted OMP**: Add expert-level regularization
   ```python
   # Penalize cross-expert selections
   score_j = |g_j| * expert_confidence[j // M]
   ```

b) **Block OMP**: Force selection within single expert per iteration
   ```python
   # Standard block-OMP with expert blocks
   ```

c) **Learned Routing** (QK-teacher):
   - Previous result: 94.6% accuracy (commit abde66a)
   - Lower than G-teacher (100%)
   - But potentially useful for complex scenarios

---

## 📁 Generated Files

**Output directory**: `results/dt_traj_omp_verification_20251106_120030/`

```
- trajectories.jsonl          # 111 OMP trajectories
- numeric_diagnostics.jsonl   # Per-sample statistics
- manifest.json               # Configuration metadata
- code_state.json             # Git state and file hashes
- generation.log              # Trajectory generation log
- verification.log            # Accuracy verification results
```

---

## ✅ Reproduction Instructions

### Environment Setup
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Generate OMP Trajectories
```bash
./run_omp_teacher_verification.sh
```

Or manually:
```bash
python doa_rl/trajectories/offline_dt_dataset.py \
  --teacher omp \
  --K 6 \
  --n_atoms 8 \
  --atom_reduce_mode kcenter \
  --fs 16000 \
  --n_fft 2048 \
  --freq_min 300.0 \
  --freq_max 3000.0 \
  --seed 42 \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --out_dir results/dt_traj_omp_test \
  --device cpu
```

### Verify Accuracy
```bash
python verify_omp_teacher_trajectories.py \
  --traj_dir results/dt_traj_omp_test
```

### Angle-Level Analysis
```bash
python analyze_omp_by_angle.py
```

---

## 📚 Related Work

### Previous Experiments
- **G-teacher baseline**: commit `abde66acacebf59ed71ea2fb6a2f8343ab287722`
- **G-teacher verification**: `G_TEACHER_VERIFICATION_SUCCESS_20251106.md` (100% accuracy)
- **QK-teacher**: commit `abde66a` (94.6% accuracy)
- **DT training success**: commit `63b8190` (480 epochs, 98%+ test accuracy)

### References
- Dictionary structure: D = H ⊙ W (direction × spectrum)
- Hierarchical OMP: Expert-level aggregation before atom selection
- Traditional OMP: Greedy correlation-based selection

---

## 🎯 Conclusion

**Traditional OMP is significantly inferior to G-teacher for acoustic source localization:**

1. **Accuracy**: 64.9% vs 100% first-step accuracy (-35.1%)
2. **Coverage**: Fails completely on 13/37 angles
3. **Physics**: Ignores expert (direction) structure
4. **Training**: Would produce lower-quality DT supervision

**Recommendation**: **Continue using G-teacher** for trajectory generation. The hierarchical structure is essential for this structured prediction task.

**The 35% accuracy gap demonstrates that exploiting domain structure (experts as directions) is crucial—pure correlation-based methods are insufficient.**

---

## 📝 Code Changes

### New Files
1. `verify_omp_teacher_trajectories.py` - OMP vs G-teacher verification script
2. `run_omp_teacher_verification.sh` - Automated verification workflow
3. `analyze_omp_by_angle.py` - Angle-level accuracy breakdown

### Modified Files
1. `doa_rl/trajectories/offline_dt_dataset.py`:
   - Added `traditional_omp_pick()` function
   - Added `--teacher omp` option
   - Supports all three teachers: g, qk, omp

---

**Generated**: 2025-11-06  
**Experiment Duration**: 0.62s (trajectory generation) + 2.1s (verification)  
**Verification Script**: `verify_omp_teacher_trajectories.py`  
**Output Directory**: `results/dt_traj_omp_verification_20251106_120030/`
