# 480-Epoch Decision Transformer Training Verification Run
## Date: 2025-11-06

## 📋 Original Experiment Reference

**Original Commit**: `63b8190` (2025-10-30)
- Commit message: "[results] 480-epoch training: 98.2% test accuracy, 73% loss improvement over 120-epoch baseline"
- Most recent successful run: `results/dt_min_480epochs_20251031_191857/`

### Original Key Results
- **Best test loss**: 0.6410 (epoch 381) vs 2.36 (120-epoch baseline)
- **Test accuracy**: expert 98.2%, atom 98.2%
- **Teacher alignment**: 98.9% step match
- **Training time**: ~4-5 hours on CPU
- **No overfitting**: Train/test loss gap stable at 0.39-0.58

---

## 🔍 Current Verification Run Details

### Run Information
- **Start time**: 2025-11-06 11:36:37 CST
- **Output directory**: `results/dt_min_480epochs_20251106_113637/`
- **Git commit**: `29b3fad` (code-only state)
- **Status**: ✅ Training started successfully

### Configuration Parameters
```bash
TRAJ_DIR="results/dt_traj_qk_kmeans"
EPOCHS=480
BATCH_SIZE=4
LR=3e-3 (0.003)
D_MODEL=128
NHEAD=2
NLAYERS=1
DEVICE="cpu"
DISTILL_WEIGHT=0.5
DISTILL_T=1.0
WARMUP_EPOCHS=2
TEST_SPLIT=0.2
SPLIT_SEED=42
```

---

## 📊 Data Sources and Statistics

### Dataset Information
**Source**: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`

#### Dataset Characteristics
- **Total samples**: 111 .npy files
- **Data fingerprint (MD5)**: `713c0635878a04b32f4ee30208904d11`
- **Angle coverage**: 37 angles (0° to 180° in 5° steps)
- **Angles**: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

#### Train/Test Split
- **Train samples**: 74 (66.7%)
- **Test samples**: 37 (33.3%)
- **Train steps per epoch**: 444
- **Test steps per epoch**: 222
- **Split seed**: 42 (deterministic)

#### Data Preprocessing
- **Sampling rate (fs)**: 16000 Hz
- **STFT parameters**: 
  - n_fft: 2048
  - freq_min: 300.0 Hz
  - freq_max: 3000.0 Hz
  - Frequency bins (F): 346
- **Processing**: Synchronized VAD + Normalization

---

## 🧮 Matrix and Model Sources

### 1. H Matrix (Transfer Function)
**Path**: `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`
- **Shape**: (346, 37)
  - 346 frequency bins
  - 37 expert directions (angles)
- **Type**: Box configuration LDV data
- **File size**: ~61 KB
- **Status**: ✅ Verified exists

### 2. W Matrix (Source Separation)
**Path**: `doa_normalized_config_c_corrected/models/usm.pth`
- **Shape**: (346, 50)
  - 346 frequency bins
  - 50 atoms (original dictionary)
- **File size**: ~70 KB
- **Training**: USM (Unsupervised Source Model)
- **Status**: ✅ Verified exists

### 3. Dictionary D (H ⊙ W)
**Computed**: D = H ⊙ W (element-wise product after atom reduction)
- **Shape**: (346, 296)
  - F=346 frequency bins
  - P=296 total dictionary elements (after K-means reduction)
- **Mutual coherence**: 
  - μ_max = 0.9941
  - μ_mean = 0.4640

### 4. Atom Reduction (K-means)
**Method**: K-means clustering applied to 50 original atoms
- **Resulting atoms (M)**: 8 clusters
- **Cluster sizes**: [1, 2, 6, 1, 35, 1, 1, 3]
- **Total combinations (P)**: 8 atoms × 37 experts = 296 elements

---

## 🤖 MDP/Teacher Model Sources

### QK Teacher Model
**Path**: `results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth`
- **Model type**: QK (Query-Key) Transformer Encoder
- **Architecture**:
  - d_model: 128
  - nhead: 2
  - nlayers: 1
- **File size**: ~3.7 MB
- **Training date**: 2025-10-26
- **Status**: ✅ Verified exists

### Teacher Configuration
```python
routing_mode = "qk"
score_norm_mode = "std"
score_center_expert = True
score_center_atoms = True
no_type_bias = True
expert_agg = "l2"
```

### Teacher Performance (t=0 on dataset)
- **Accuracy**: 94.6%
- **Expert distribution**: All 37 angles covered (0-3 samples per angle initially)

---

## 🎯 Model Architecture (DT-Min)

### Decision Transformer Parameters
- **Model dimension (d_model)**: 128
- **Attention heads (nhead)**: 2
- **Transformer layers (nlayers)**: 1
- **Sequence length (K)**: 6 (6-step decision sequences)
- **Architecture**: Hierarchical pointer network
  - Level 1: Expert (angle) selection
  - Level 2: Atom selection per expert

### Training Objectives
1. **Supervised loss**: Cross-entropy on expert/atom predictions
2. **Distillation loss**: KL divergence from teacher model
   - Weight: 0.5
   - Temperature: 1.0
3. **Return-to-go (RTG) targets**:
   - resid: 0.02 (residual improvement goal)
   - acc: 0.95 (accuracy goal)

---

## 📈 Expected Performance Metrics

Based on commit `63b8190` results, we expect:

### Training Dynamics (by epoch range)
1. **Epochs 1-100**: Rapid convergence
   - Test loss: 5.6 → ~2.5
   - Expert/atom acc: ~10% → 40-60%

2. **Epochs 100-381**: Steady improvement
   - Test loss: 2.5 → 0.64
   - Expert acc: 60% → 98%
   - Atom acc: → 98%

3. **Epochs 381-480**: Performance plateau
   - Test loss: 0.64-0.83 (minor fluctuations)
   - Accuracy: 96-98% (stable)

### Final Expected Metrics (epoch 480)
- **Best test loss**: ~0.64 (around epoch 381)
- **Expert accuracy**: 98.2%
- **Atom accuracy**: 98.2%
- **Teacher-forced step match**: 98.9%
- **Angle accuracy (t=0)**: 94.6%
- **Angle accuracy (t=K-1)**: ~53% (known degradation)
- **Train/test gap**: 0.4-0.5 (excellent generalization)

---

## 🔬 Trajectory Data (Offline RL)

### Trajectory Source
**Directory**: `results/dt_traj_qk_kmeans/`
- **Total trajectories**: 111 (one per data sample)
- **Trajectory file**: `trajectories.jsonl` (~164 KB)
- **Manifest**: `manifest.json` (~22 KB)
- **Diagnostics**: `numeric_diagnostics.jsonl` (~67 KB)

### Trajectory Generation Method
- **Teacher**: QK model
- **Selection strategy**: Greedy (teacher-forced)
- **Sequence length**: K=6 steps
- **RTG computation**: Based on residual reduction and accuracy

### Trajectory Structure (per sample)
Each trajectory contains K=6 decision steps:
```json
{
  "expert_indices": [e0, e1, e2, e3, e4, e5],  // 6 angle selections
  "atom_indices": [a0, a1, a2, a3, a4, a5],    // 6 atom selections
  "rtg": [r0, r1, r2, r3, r4, r5],            // return-to-go values
  "angle_deg": <true_angle>,
  "trajectory_id": <id>
}
```

---

## ⚙️ Execution Environment

### Python Environment
- **Python version**: 3.13.3
- **Virtual environment**: `/Users/sbplab/jnrle/.venv/bin/python`
- **PYTHONPATH**: Set to project root

### Hardware
- **Device**: CPU (no GPU acceleration)
- **Platform**: macOS (Apple Silicon/MPS available but using CPU)
- **Expected runtime**: ~4-5 hours for 480 epochs

### Key Dependencies
- PyTorch (for model training)
- NumPy (for data handling)
- JSON (for trajectory/manifest storage)

---

## 🔍 Verification Checklist

### Pre-run Verification ✅
- [x] Dataset exists and accessible (111 files)
- [x] Dataset fingerprint matches: `713c0635878a04b32f4ee30208904d11`
- [x] H matrix exists: `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`
- [x] W matrix exists: `doa_normalized_config_c_corrected/models/usm.pth`
- [x] QK teacher model exists: `results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth`
- [x] Trajectory data exists: `results/dt_traj_qk_kmeans/trajectories.jsonl`
- [x] Python environment configured
- [x] Training started successfully

### During Training (to monitor)
- [ ] Epoch 1 baseline matches (~5.6 test loss)
- [ ] Rapid convergence in first 100 epochs
- [ ] No crashes or NaN losses
- [ ] Train/test split correct (74/37 samples)
- [ ] Checkpoint saving works

### Post-run Verification (after completion)
- [ ] Best checkpoint saved (around epoch 381)
- [ ] Final test accuracy ≥95%
- [ ] Training curves generated
- [ ] No overfitting (train/test gap <1.0)
- [ ] Metrics match original run (±2% tolerance)

---

## 📝 Files Generated During Training

Expected output files in `results/dt_min_480epochs_20251106_113637/`:
1. **training.log** - Full training stdout/stderr
2. **ckpt_best.pth** - Best model checkpoint (~7.9 MB)
3. **ckpt_latest.pth** - Latest model checkpoint (~7.9 MB)
4. **metrics.npz** - NumPy archive of all metrics (~44 KB)
5. **controllability.jsonl** - RTG controllability analysis (~16 KB)
6. **code_state.json** - Git commit and file hashes (~924 B)
7. **numeric_diagnostics.jsonl** - Numerical diagnostics (~94 B)

---

## 🎓 Summary of Key Quantities

| Quantity | Value | Description |
|----------|-------|-------------|
| **Dataset** |
| Samples | 111 | Total data files |
| Angles | 37 | From 0° to 180° (5° steps) |
| Train/Test | 74/37 | 66.7% / 33.3% split |
| **Matrices** |
| H shape | (346, 37) | Transfer function (freq × angles) |
| W shape | (346, 50) | Source atoms (freq × atoms) |
| D shape | (346, 296) | Dictionary after reduction |
| Atoms (M) | 8 | K-means clusters |
| Experts (E) | 37 | Angle directions |
| Dictionary (P) | 296 | M × E combinations |
| **STFT** |
| Freq bins (F) | 346 | From 300-3000 Hz |
| fs | 16000 Hz | Sampling rate |
| n_fft | 2048 | FFT size |
| **Training** |
| Epochs | 480 | Total training epochs |
| Batch size | 4 | Samples per batch |
| Learning rate | 0.003 | Adam optimizer |
| d_model | 128 | Transformer dimension |
| Sequence (K) | 6 | Decision steps |
| **Expected Results** |
| Test acc | 98.2% | Expert & atom accuracy |
| Best loss | ~0.64 | At epoch ~381 |
| Teacher match | 98.9% | Step-wise alignment |

---

## 🚀 Next Steps

After training completes (~4-5 hours):
1. ✅ Verify final metrics match expectations
2. ✅ Compare training curves to original run
3. ✅ Check best checkpoint epoch number
4. ✅ Validate no overfitting occurred
5. ✅ Document any deviations from expected behavior
6. ⚠️ If metrics deviate >5%, investigate causes

---

**Document generated**: 2025-11-06 11:40:00 CST  
**Training status**: In progress  
**Estimated completion**: 2025-11-06 ~16:30 CST
