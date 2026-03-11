# Results: QK Mode SNR Robustness Study (2025-12-10)

## Background

Following the Phase 6 SNR experiment (commit `5008b18`) which validated g-mode's noise degradation behavior,
this study investigates how QK mode (Transformer attention-based routing) performs across the same SNR sweep.

**Prior findings**:
- Commit `fe4ef60`: QK mode achieved 97.3% accuracy on original clean white noise dataset
- Commit `1f6b68c`: g-mode achieved 100% accuracy on the same dataset
- Commit `5008b18`: g-mode SNR sweep showed expected degradation pattern (100% → 16.2%)

**Research question**: Does QK mode exhibit similar or worse SNR degradation compared to g-mode?

## Motivation

QK mode uses learned attention weights to route signals, while g-mode uses physics-based correlation ($g = D^T @ r$).
Understanding the noise robustness difference between these two routing strategies is critical for:
1. Deciding which routing mode to deploy in real-world noisy environments
2. Understanding the fundamental limitations of learned vs. physics-based approaches
3. Informing future model architecture decisions

## Purpose

Systematically evaluate QK mode across 7 SNR levels (Inf, 30dB, 20dB, 15dB, 10dB, 5dB, 0dB) using:
- Identical model architecture (d_model=128, nhead=2, nlayers=1)
- Identical training protocol (10 epochs, batch_size=16, lr=3e-3)
- Identical preprocessing (STFT, VAD, normalization)
- Same synthetic SNR datasets used in Phase 6

## Expected Results (Prior Hypothesis)

Based on QK mode's 97.3% on clean data vs g-mode's 100%:
- Expected QK mode would perform ~2-3% worse at each SNR level
- Expected similar degradation threshold around 10dB

## Actual Results

| SNR Level | QK Mode | g-mode (Phase 6) | Δ Accuracy | Status |
|-----------|---------|------------------|------------|--------|
| **Inf**   | 94.6%   | 100.0%           | **-5.4%**  | ⚠️ |
| **30 dB** | 91.9%   | 100.0%           | **-8.1%**  | ⚠️ |
| **20 dB** | 91.9%   | 100.0%           | **-8.1%**  | ⚠️ |
| **15 dB** | 86.5%   | 100.0%           | **-13.5%** | ❌ |
| **10 dB** | 73.0%   | 91.9%            | **-18.9%** | ❌ |
| **5 dB**  | 27.0%   | 37.8%            | **-10.8%** | ❌ |
| **0 dB**  | 2.7%    | 16.2%            | **-13.5%** | ❌ |

### Problem Angles Per SNR Level

| SNR | Problem Angles (0% accuracy) | Count |
|-----|------------------------------|-------|
| Inf | 95°, 175° | 2 |
| 30dB | 20°, 95°, 175° | 3 |
| 20dB | 25°, 95°, 175° | 3 |
| 15dB | 10°, 25°, 40°, 80°, 95° | 5 |
| 10dB | 5°, 10°, 25°, 35°, 40°, 105°, 145°, 160°, 165°, 170° | 10 |
| 5dB | 27 angles at 0% | 27 |
| 0dB | 36 angles at 0% (only 50° correct) | 36 |

## Comparison to Expectation

| Aspect | Expected | Actual | Analysis |
|--------|----------|--------|----------|
| Clean data (Inf) | ~97% | 94.6% | **Worse** - likely due to train/val split |
| Gap vs g-mode | ~2-3% | **5-19%** | **Much worse** than expected |
| Degradation threshold | ~10dB | **15dB** | Earlier degradation |
| Low SNR (0dB) | Random (~2.7%) | 2.7% | As expected - model fails |

## Physical/Mathematical Analysis

### First Principles

1. **g-mode routing**: Uses physics-based correlation $g = D^T @ r$ where $D$ is the dictionary and $r$ is the residual.
   - Directly computes correlation between signal and known spatial patterns
   - Noise affects both signal and pattern equally, but correlation is preserved at high SNR

2. **QK mode routing**: Uses learned attention weights $\text{softmax}(Q @ K^T / \sqrt{d})$
   - Relies on learned representations that may overfit to clean signal characteristics
   - No explicit physics constraint - purely data-driven

### Why QK Mode Fails Earlier

Mathematical explanation:
- **g-mode**: $g_i = \langle d_i, r \rangle = \langle d_i, s + n \rangle = \langle d_i, s \rangle + \langle d_i, n \rangle$
  - At high SNR: $\langle d_i, s \rangle >> \langle d_i, n \rangle$ → correct routing
  - Noise term $\langle d_i, n \rangle$ averages to 0 for uncorrelated noise

- **QK mode**: $\text{score}_i = f_{learned}(x)$
  - Learned function $f$ was optimized on clean data
  - Additive noise shifts input distribution outside training manifold
  - Attention weights become unreliable when input statistics change

### Information Theory Perspective

- g-mode: Exploits known structure (H matrix) as prior information
- QK mode: Must learn structure from limited data (74 training samples)
- Under noise: Prior knowledge (g-mode) is more robust than learned knowledge (QK mode)

## Cross-Experiment Analysis

### Pattern Recognition
1. **QK mode consistently underperforms g-mode** across all SNR levels by 5-19%
2. **Degradation starts earlier**: QK at 15dB vs g-mode at 10dB
3. **Angle 95°** is problematic for QK mode even at clean data (consistent with previous investigation)

### Success Factors (for g-mode)
- Physics-based routing provides inherent noise robustness
- Direct correlation computation doesn't require learning
- Prior knowledge of spatial patterns (H matrix) acts as regularization

### Failure Modes (for QK mode)
- Learned attention overfits to clean signal statistics
- No explicit noise robustness mechanism
- Limited training data (74 samples) insufficient for robust representations

## Extracted Principles

### Design Principles
1. **Use g-mode for noisy environments** - 5-19% better accuracy
2. **QK mode only suitable for clean data** - acceptable at SNR ≥ 20dB
3. **Physics-based priors beat pure learning** when prior knowledge is available

### Hypothesis Formation for Future Experiments
1. Noise-augmented training may improve QK mode robustness
2. Hybrid mode (combining QK and g) might balance learning and physics
3. Larger d_model or more layers may help QK mode generalize better

## Data Lineage

### External Dependencies (NOT in git)

| File | Path | MD5 | Size |
|------|------|-----|------|
| H matrix | `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth` | `3d573394192b581db363b6f1bc039cad` | 62,573 bytes |

### Tracked Dependencies (in git-lfs)

| File | Path | MD5 |
|------|------|-----|
| USM (W) | `doa_normalized_config_c_corrected/models/usm.pth` | `95cf7db9a505189945cca98662027125` |

### SNR Datasets

| SNR Level | Path | Fingerprint | Files |
|-----------|------|-------------|-------|
| Inf | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snrInf_sync_vad_normalized` | `1050c90ee73a30ab92b11698ead11a9d` | 111 |
| 30dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr30dB_sync_vad_normalized` | `ce065de36b02a96dcfaeab8027ca789b` | 111 |
| 20dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr20dB_sync_vad_normalized` | `f709f953c12759ebd672669fc71789dc` | 111 |
| 15dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr15dB_sync_vad_normalized` | `40919a5bafac7fe54bf21c7320129543` | 111 |
| 10dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr10dB_sync_vad_normalized` | `f49c15a03c5d638e3754d250beb3f89a` | 111 |
| 5dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr5dB_sync_vad_normalized` | `e714b6fa71690dbfcb249b439c236a1d` | 111 |
| 0dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr0dB_sync_vad_normalized` | `34b49de468af049ff6f3d80296a02712` | 111 |

### Dataset Configuration
- **STFT**: fs=16000 (resampled from 48kHz), n_fft=2048, freq_band=[300, 3000]Hz
- **Frequency bins**: F=346
- **Angles**: E=37 (0° to 180°, 5° intervals)
- **Samples per angle**: 3
- **Total samples**: 111 per SNR level
- **Train/Val split**: 74 train / 37 val (clip_id % 5)

## Reproduction Instructions

### Step 1: Environment Setup
```bash
source ~/.zshrc
conda activate wavtokenizer
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Step 2: Verify Dependencies
```bash
# Verify H matrix
md5 /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
# Expected: 3d573394192b581db363b6f1bc039cad

# Verify USM
md5 doa_normalized_config_c_corrected/models/usm.pth
# Expected: 95cf7db9a505189945cca98662027125

# Verify datasets exist
ls /Users/sbplab/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr*_sync_vad_normalized
```

### Step 3: Run QK Mode SNR Sweep
```bash
# SNR Inf
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snrInf_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snrInf_white_noise_YYYYMMDD

# SNR 30dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr30dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr30dB_white_noise_YYYYMMDD

# SNR 20dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr20dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr20dB_white_noise_YYYYMMDD

# SNR 15dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr15dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr15dB_white_noise_YYYYMMDD

# SNR 10dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr10dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr10dB_white_noise_YYYYMMDD

# SNR 5dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr5dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr5dB_white_noise_YYYYMMDD

# SNR 0dB
python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr0dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr0dB_white_noise_YYYYMMDD
```

### Step 4: Verify Results
Check each results directory for:
- `results.png`: Visualization of per-angle accuracy
- `diagnostics.jsonl`: Training metrics per epoch
- `model_best.pth`: Best model checkpoint

Expected accuracy values:
- Inf: ~94.6%
- 30dB: ~91.9%
- 20dB: ~91.9%
- 15dB: ~86.5%
- 10dB: ~73.0%
- 5dB: ~27.0%
- 0dB: ~2.7%

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| routing_mode | qk | Transformer attention-based |
| d_model | 128 | Embedding dimension |
| nhead | 2 | Attention heads |
| nlayers | 1 | Transformer layers |
| n_atoms | 8 | K-means reduced atoms |
| steps | 6 | OMP iterations |
| top_e | 2 | Top experts per step |
| top_l | 2 | Top atoms per expert |
| score_norm | std | Standardize routing scores |
| score_center_atoms | True | Center atom scores |
| score_center_expert | True | Center expert scores |
| no_type_bias | True | No learnable type bias |
| epochs | 10 | Training epochs |
| batch_size | 16 | Batch size |
| lr | 3e-3 | Learning rate |

## Conclusions

1. **QK mode is significantly less robust to noise than g-mode** (5-19% worse across all SNR levels)

2. **Degradation threshold**:
   - g-mode: stable until 10dB
   - QK mode: starts degrading at 15dB

3. **Recommendation**: Use **g-mode for any real-world deployment** where noise is expected

4. **Root cause**: Learned attention weights overfit to clean signal statistics, while physics-based correlation is inherently more robust

## Next Experiments

1. **Noise-augmented training**: Train QK mode with added noise to improve robustness
2. **Hybrid mode exploration**: Combine QK and g-mode for balanced approach
3. **Architecture scaling**: Test if larger models (d_model=256, nlayers=2) help QK mode
4. **Speech data**: Repeat experiment with speech SNR datasets
