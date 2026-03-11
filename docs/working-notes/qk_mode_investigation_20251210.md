# Investigation: QK Mode White Noise Accuracy Analysis (2025-12-10)

## Background

User recalled an experiment where QK mode achieved 100% accuracy on white noise data (0°-180°, 5° intervals). Investigation was conducted to:
1. Locate the original commit with 100% accuracy
2. Determine if it was QK mode or g-mode
3. Understand the evolution of the codebase and why changes were made

## Key Findings

### 1. The 100% Accuracy Experiment Was g-mode, Not QK Mode

| Commit | Routing Mode | Dataset | Accuracy | Notes |
|--------|--------------|---------|----------|-------|
| `1f6b68c` | **g-mode** | White noise (37 angles) | **100.0%** | All angles correct |
| `fe4ef60` | **QK mode** | White noise (37 angles) | **97.3%** | Angle 95° = 0% |

### 2. QK Mode Best Result: 97.3% (Commit fe4ef60)

Per-angle accuracy from `fe4ef60`:
- Angles 0°-90°: 100% (all correct)
- Angle 95°: **0%** (all 3 samples wrong)
- Angles 100°-180°: 100% (all correct)

### 3. Code Evolution Analysis

From `1f6b68c` to `5008b18`, key changes were made to fix QK routing:

| Commit | Date | Change | Purpose |
|--------|------|--------|---------|
| `1f6b68c` | 2025-10-26 | Original g-mode 100% | Baseline physics-based routing |
| `70ba576` | 2025-10-26 | QK vs Hybrid comparison | Diagnose QK stagnation |
| `7e188c6` | 2025-10-26 | **Score normalization** | Fix QK common-mode logits |
| `34c4555` | 2025-10-27 | Hard Gumbel + Atom Masking | Enable discrete selection |
| `06bf65d` | 2025-11-14 | Train/Val split | Speech260 experiments |

### 4. Critical Code Difference

In commit `7e188c6`, score normalization was added **after** routing mode selection:

```python
# This applies to ALL routing modes (qk, g, hybrid) if enabled!
if self.score_norm_mode == 'std':
    scores_expert = (scores_expert - se_mean) / (se_std + 1e-8)
if self.score_center_expert:
    scores_expert = scores_expert - scores_expert.mean()
```

**Impact**: If using g-mode with `--score_norm std` or `--score_center_*` flags enabled, the physics-based scores will be modified, potentially affecting results.

## Reproduction Results

### Experiment 1: Reproduce fe4ef60 QK Mode (checkout to fe4ef60)

**Command**:
```bash
git checkout fe4ef60
source ~/.zshrc && conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH

python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk \
  --hybrid_alpha 0.0 \
  --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/reproduce_qk_d128_20251210_181912
```

**Result**: 97.3% accuracy (identical to original)

### Experiment 2: Current Version with Hard Gumbel

**Command**:
```bash
# On commit 5008b18
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk \
  --hybrid_alpha 0.0 \
  --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --use_hard_gumbel \
  --out_dir results/qk_hard_gumbel_white_noise_20251210_183424
```

**Result**: 94.6% accuracy (two problem angles: 95°, 175°)

## Data Lineage

### External Dependencies (NOT in git)

| File | Path | MD5 | Size |
|------|------|-----|------|
| H matrix | `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth` | `3d573394192b581db363b6f1bc039cad` | 62,573 bytes |
| Dataset | `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/` | `bdc5299806f13cf87e482f02ee4d6d07` | 111 files |

### Tracked Dependencies (in git-lfs)

| File | Git LFS SHA | MD5 |
|------|-------------|-----|
| USM (W) | `doa_normalized_config_c_corrected/models/usm.pth` | `95cf7db9a505189945cca98662027125` |
| Original Results | `results/exp_H_qk_encoder_on_atom_d128_20251026_233228/` | (multiple files) |

### Dataset Configuration

- **STFT**: fs=16000, n_fft=2048, freq_band=[300, 3000]Hz
- **Frequency bins**: F=346
- **Angles**: E=37 (0° to 180°, 5° intervals)
- **Samples per angle**: 3
- **Total samples**: 111

## K-means Difference

The K-means implementation changed between versions:

| Version | Implementation | Cluster Sizes |
|---------|---------------|---------------|
| `fe4ef60` | sklearn KMeans | `[1, 3, 3, 1, 1, 1, 1, 39]` |
| `5008b18` | NumPy custom | `[1, 2, 6, 1, 35, 1, 1, 3]` |

This may contribute to slight accuracy differences.

## Conclusions

1. **No QK mode experiment achieved 100% accuracy on white noise** - the 100% result was g-mode
2. **QK mode best accuracy**: 97.3% (commit `fe4ef60`)
3. **Hard Gumbel reduces accuracy**: 94.6% vs 97.3%
4. **Angle 95° is consistently problematic** for QK mode
5. **Code changes were intentional** - score normalization was added to fix QK's common-mode logit problem

## Reproduction Steps

### Step 1: Environment Setup
```bash
source ~/.zshrc
conda activate trl-training
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Step 2: Verify Data Dependencies
```bash
# Check H matrix
ls -la /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
md5 /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
# Expected: 3d573394192b581db363b6f1bc039cad

# Check dataset
find /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized -name "*.npy" | wc -l
# Expected: 111

# Check USM
ls -la doa_normalized_config_c_corrected/models/usm.pth
```

### Step 3: Reproduce fe4ef60 Result (97.3%)
```bash
git checkout fe4ef60
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/reproduce_fe4ef60_$(date +%Y%m%d_%H%M%S)
```

### Step 4: Return to Current Branch
```bash
git checkout experiment/snr-synthetic-datasets
git stash pop  # if needed
```
