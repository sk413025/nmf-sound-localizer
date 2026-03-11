# Results: QK Mode 30 Epochs SNR Sweep (2025-12-10)

## Background

This experiment follows up on the QK Mode 20 Epochs SNR Sweep (commit `cfdc4d9`), which showed significant improvements over 10 epochs. The key observation was that all experiments had `best_epoch = 20`, suggesting models may not have fully converged.

### Previous Results (20 epochs, commit cfdc4d9)

| SNR Level | 10 epochs | 20 epochs | Δ (10→20) | vs g-mode |
|-----------|-----------|-----------|-----------|-----------|
| Inf       | 94.6%     | 100.0%    | +5.4%     | Equal     |
| 30 dB     | 91.9%     | 100.0%    | +8.1%     | Equal     |
| 20 dB     | 91.9%     | 97.3%     | +5.4%     | -2.7%     |
| 15 dB     | 86.5%     | 100.0%    | +13.5%    | Equal     |
| 10 dB     | 73.0%     | 94.6%     | +21.6%    | +2.7%     |
| 5 dB      | 27.0%     | 73.0%     | +46.0%    | +35.2%    |
| 0 dB      | 2.7%      | 51.4%     | +48.7%    | +35.2%    |

**Key question**: Can 30 epochs further improve performance, especially for:
- SNR 20dB (97.3%) - Can it reach 100%?
- SNR 10dB (94.6%) - Can it reach 100%?
- SNR 5dB (73.0%) and 0dB (51.4%) - How much improvement potential remains?

## Motivation

1. **Convergence indicator**: All 20-epoch experiments had `best_epoch = 20`, suggesting continued learning
2. **Large improvement magnitude**: Low SNR showed +46~49% improvement from 10→20 epochs
3. **Determine optimal training duration**: Establish epoch recommendations for different SNR conditions

## Purpose

Test whether 30 epochs training further improves QK mode performance:
1. Can SNR 20dB and 10dB reach 100%?
2. How much can SNR 5dB and 0dB improve?
3. Identify convergence points for each SNR level

## Hypothesis

- **Optimistic**:
  - SNR 20dB: 97.3% → 100%
  - SNR 10dB: 94.6% → 100%
  - SNR 5dB: 73% → 80%+
  - SNR 0dB: 51.4% → 60%+
- **Pessimistic**: Overfitting causes validation accuracy to decrease
- **Neutral**: High SNR maintains 100%, low SNR shows minor improvement (+2-5%)

## Results

### Summary Table

| SNR Level | 10 epochs | 20 epochs | 30 epochs | Δ (20→30) | best_epoch | Converged? |
|-----------|-----------|-----------|-----------|-----------|------------|------------|
| Inf       | 94.6%     | 100.0%    | **100.0%** | 0%        | 20         | ✓ at 20    |
| 30 dB     | 91.9%     | 100.0%    | **100.0%** | 0%        | 20         | ✓ at 20    |
| 20 dB     | 91.9%     | 97.3%     | **100.0%** | +2.7%     | 30         | ✓ at 30    |
| 15 dB     | 86.5%     | 100.0%    | **100.0%** | 0%        | 20         | ✓ at 20    |
| 10 dB     | 73.0%     | 94.6%     | **100.0%** | +5.4%     | 30         | ✓ at 30    |
| 5 dB      | 27.0%     | 73.0%     | **86.5%**  | +13.5%    | 30         | Still learning |
| 0 dB      | 2.7%      | 51.4%     | **62.2%**  | +10.8%    | 30         | Still learning |

### Per-Angle Analysis (Failed Angles)

| SNR Level | Failed Angles | Total Failed |
|-----------|---------------|--------------|
| Inf       | None          | 0/37         |
| 30 dB     | None          | 0/37         |
| 20 dB     | None          | 0/37         |
| 15 dB     | None          | 0/37         |
| 10 dB     | None          | 0/37         |
| 5 dB      | 75°, 85°, 95°, 165°, 175° | 5/37 |
| 0 dB      | 25°, 35°, 75°, 85°, 95°, 100°, 105°, 120°, 135°, 145°, 150°, 160°, 165°, 170° | 14/37 |

**Key observation**: Angles 75°, 85°, 95° are consistently problematic at low SNR. Angle 95° has been problematic since the original QK mode investigation (commit `1a4f05a`).

### Comparison to Expectation

| Hypothesis | Prediction | Actual | Status |
|------------|------------|--------|--------|
| SNR 20dB → 100% | ✓ | 100.0% | **Achieved** |
| SNR 10dB → 100% | ✓ | 100.0% | **Achieved** |
| SNR 5dB → 80%+ | ✓ | 86.5% | **Exceeded** |
| SNR 0dB → 60%+ | ✓ | 62.2% | **Achieved** |
| No overfitting | ✓ | No degradation | **Confirmed** |

**Result**: Optimistic hypothesis was correct across all SNR levels.

## Key Findings

### 1. Convergence Points Identified

| SNR Range | Convergence Point | Recommendation |
|-----------|-------------------|----------------|
| ≥15 dB    | 20 epochs         | 20 epochs sufficient |
| 20 dB, 10 dB | 30 epochs      | 30 epochs optimal |
| ≤5 dB     | >30 epochs        | Consider 40+ epochs |

### 2. Complete Results: 10 → 20 → 30 Epochs Evolution

| SNR | 10 epochs | 20 epochs | 30 epochs | Total Δ (10→30) |
|-----|-----------|-----------|-----------|-----------------|
| Inf | 94.6% | 100.0% | 100.0% | +5.4% |
| 30dB | 91.9% | 100.0% | 100.0% | +8.1% |
| 20dB | 91.9% | 97.3% | 100.0% | +8.1% |
| 15dB | 86.5% | 100.0% | 100.0% | +13.5% |
| 10dB | 73.0% | 94.6% | 100.0% | +27.0% |
| 5dB | 27.0% | 73.0% | 86.5% | +59.5% |
| 0dB | 2.7% | 51.4% | 62.2% | +59.5% |

### 3. QK Mode vs g-mode Final Comparison (30 epochs)

| SNR | QK (30 epochs) | g-mode | Δ | Winner |
|-----|----------------|--------|---|--------|
| Inf | 100.0% | 100.0% | 0% | Tie |
| 30dB | 100.0% | 100.0% | 0% | Tie |
| 20dB | 100.0% | 100.0% | 0% | Tie |
| 15dB | 100.0% | 100.0% | 0% | Tie |
| 10dB | **100.0%** | 91.9% | **+8.1%** | **QK** |
| 5dB | **86.5%** | 37.8% | **+48.7%** | **QK** |
| 0dB | **62.2%** | 16.2% | **+46.0%** | **QK** |

**Conclusion**: QK mode with sufficient training (30 epochs) achieves 100% accuracy at all SNR ≥10dB and dramatically outperforms g-mode at low SNR.

## Physical/Mathematical Analysis

### First Principles Explanation

1. **g-mode**: Physics-based routing using `g = D^T @ r` (correlation with dictionary atoms)
   - Fixed, non-trainable computation
   - Directly correlates signal with known patterns
   - Performance bounded by signal quality

2. **QK mode**: Learned attention-based routing using `scores = softmax(QK^T / sqrt(d))`
   - Trainable projection matrices learn noise-robust features
   - Can adapt to noise-corrupted inputs
   - Performance improves with training duration

### Why QK Outperforms at Low SNR

The learned attention mechanism provides three key advantages:

1. **Adaptive frequency weighting**: QK attention learns to downweight frequency bins susceptible to noise
2. **Feature extraction**: Encoder learns noise-robust representations before routing
3. **Non-linear separation**: Transformer learns non-linear mappings that better separate signal from noise

### Mathematical Relationship

```
At high SNR (Signal >> Noise):
  - Physics-based correlation is optimal
  - Both methods achieve 100%

At low SNR (Signal ≈ Noise):
  - Physics-based correlation degrades rapidly
  - Learned features provide robust advantage
  - QK mode outperforms by 46-49%
```

### Convergence Dynamics

| SNR Range | Epochs to 100% | Learning Difficulty |
|-----------|----------------|---------------------|
| ≥15 dB    | ~20 epochs     | Easy (high signal)  |
| 10-20 dB  | ~30 epochs     | Medium              |
| <10 dB    | >30 epochs     | Hard (low signal)   |

## Cross-Experiment Analysis

### Pattern Recognition (10 → 20 → 30 epochs)

1. **Improvement rate**: Higher at low SNR
   - High SNR: +0% to +2.7% per 10 epochs
   - Low SNR: +10.8% to +13.5% per 10 epochs

2. **Convergence order**: High SNR converges first
   - SNR ≥15dB: Converged at 20 epochs
   - SNR 10-20dB: Converged at 30 epochs
   - SNR ≤5dB: Still learning at 30 epochs

3. **Problematic angles persist**: 75°, 85°, 95° consistently fail at low SNR

### Success Factors

1. **Sufficient training epochs**: Critical for QK mode (20-30+ epochs)
2. **Score normalization**: `--score_center_atoms --score_center_expert --score_norm std`
3. **Soft Gumbel selection**: NOT hard Gumbel
4. **d_model=128**: Sufficient capacity for learning

### Failure Modes

1. **Underfitting**: 10 epochs is insufficient for QK mode
2. **Hard Gumbel**: Reduces accuracy vs soft Gumbel
3. **Certain angles**: 75°, 85°, 95° are inherently difficult

## Extracted Principles

### Design Principles

1. **Epoch selection by SNR**:
   - High SNR (≥15dB): 20 epochs sufficient
   - Medium SNR (10-20dB): 30 epochs recommended
   - Low SNR (<10dB): 40+ epochs may help

2. **QK mode is preferred for low SNR applications**
3. **Always verify convergence** by checking if `best_epoch < total_epochs`

### Hypothesis Formation for Future Experiments

1. **40 epochs experiment**: Predict +5-10% improvement for SNR 5dB/0dB
2. **Learning rate tuning**: May accelerate convergence at low SNR
3. **Data augmentation**: May improve generalization at low SNR

## Data Lineage

### External Dependencies (NOT in git)

| File | Path | MD5 |
|------|------|-----|
| H matrix | `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth` | `3d573394192b581db363b6f1bc039cad` |
| USM (W) | `doa_normalized_config_c_corrected/models/usm.pth` | `95cf7db9a505189945cca98662027125` |

### SNR Datasets

| SNR Level | Path | Files |
|-----------|------|-------|
| Inf | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snrInf_sync_vad_normalized` | 111 |
| 30dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr30dB_sync_vad_normalized` | 111 |
| 20dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr20dB_sync_vad_normalized` | 111 |
| 15dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr15dB_sync_vad_normalized` | 111 |
| 10dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr10dB_sync_vad_normalized` | 111 |
| 5dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr5dB_sync_vad_normalized` | 111 |
| 0dB | `~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr0dB_sync_vad_normalized` | 111 |

### Dataset Configuration

- **STFT**: fs=16000, n_fft=2048, freq_band=[300, 3000]Hz
- **Frequency bins**: F=346
- **Angles**: E=37 (0° to 180°, 5° intervals)
- **Samples per angle**: 3
- **Total samples**: 111 per SNR level
- **Train/Val split**: 74/37 (clip_id % 5)

## Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| routing_mode | qk | Transformer attention-based routing |
| d_model | 128 | Embedding dimension |
| nhead | 2 | Multi-head attention |
| nlayers | 1 | Single transformer layer |
| epochs | **30** | Changed from 20 |
| batch_size | 16 | |
| lr | 3e-3 | Default learning rate |
| score_norm | std | Standardization of scores |
| score_center_atoms | True | Center atom scores |
| score_center_expert | True | Center expert scores |
| no_type_bias | True | No learnable type bias |
| use_hard_gumbel | False | Soft Gumbel selection |

## Result Artifacts

| SNR | Directory | best_epoch | Accuracy |
|-----|-----------|------------|----------|
| Inf | `results/qk_snrInf_30epochs_20251210/` | 20 | 100.0% |
| 30dB | `results/qk_snr30dB_30epochs_20251210/` | 20 | 100.0% |
| 20dB | `results/qk_snr20dB_30epochs_20251210/` | 30 | 100.0% |
| 15dB | `results/qk_snr15dB_30epochs_20251210/` | 20 | 100.0% |
| 10dB | `results/qk_snr10dB_30epochs_20251210/` | 30 | 100.0% |
| 5dB | `results/qk_snr5dB_30epochs_20251210/` | 30 | 86.5% |
| 0dB | `results/qk_snr0dB_30epochs_20251210/` | 30 | 62.2% |

Each directory contains:
- `model_best.pth` - Best model checkpoint
- `metrics.npz` - Training metrics and per-angle accuracy
- `diagnostics.jsonl` - Training diagnostics
- `results.png` - Visualization
- `preprocessing.pth` - Preprocessing state
- `code_state.json` - Git state at experiment time

## Reproduction Steps

### Step 1: Environment Setup
```bash
source ~/.zshrc
conda activate wavtokenizer
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Step 2: Verify Dependencies
```bash
# Check H matrix
md5 /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
# Expected: 3d573394192b581db363b6f1bc039cad

# Check USM
md5 doa_normalized_config_c_corrected/models/usm.pth
# Expected: 95cf7db9a505189945cca98662027125

# Check datasets (each should have 111 files)
for snr in Inf 30dB 20dB 15dB 10dB 5dB 0dB; do
  echo "SNR $snr: $(find ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr${snr}_sync_vad_normalized -name '*.npy' | wc -l) files"
done
```

### Step 3: Run Experiments

```bash
# SNR Inf (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snrInf_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snrInf_30epochs_20251210

# SNR 30dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr30dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr30dB_30epochs_20251210

# SNR 20dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr20dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr20dB_30epochs_20251210

# SNR 15dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr15dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr15dB_30epochs_20251210

# SNR 10dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr10dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr10dB_30epochs_20251210

# SNR 5dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr5dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr5dB_30epochs_20251210

# SNR 0dB (30 epochs)
python -u scripts/omp-transformer-ldv.py --routing_mode qk --device cpu \
  --dataset_root ~/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr0dB_sync_vad_normalized \
  --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 \
  --no_type_bias --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/qk_snr0dB_30epochs_20251210
```

### Step 4: Verify Results
```bash
python3 << 'EOF'
import numpy as np

snr_levels = ['Inf', '30dB', '20dB', '15dB', '10dB', '5dB', '0dB']
for snr in snr_levels:
    m = np.load(f'results/qk_snr{snr}_30epochs_20251210/metrics.npz', allow_pickle=True)
    print(f"SNR {snr}: {m['best_accuracy']*100:.1f}% (best_epoch={m['best_epoch']})")
EOF
```

## Next Experiments

1. **40 epochs for low SNR**: Test if SNR 5dB/0dB continue to improve
2. **Learning rate tuning**: May accelerate convergence at low SNR
3. **Angle 95° investigation**: Deep dive into why this angle is problematic
4. **Speech data test**: Apply same configuration to speech datasets

## Related Commits

- `cfdc4d9` - QK Mode 20 Epochs SNR Sweep (immediate predecessor)
- `e37f512` - QK Mode 10 Epochs SNR Sweep (original study)
- `1a4f05a` - QK Mode Investigation (identified 97.3% maximum)
- `5008b18` - Phase 6 g-mode SNR Experiment (comparison baseline)
