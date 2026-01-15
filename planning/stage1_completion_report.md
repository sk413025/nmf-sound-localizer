# Stage 1 Completion Report

> **Date**: 2026-01-15
> **Status**: ✅ Complete
> **Key Commit**: 2067cec (Freq-Aware Policy Full Scale Experiment)

---

## Executive Summary

Stage 1 of the two-stage learning framework has been successfully implemented and validated. The Frequency-Aware DTMin model achieves **97.11% Energy Reduction**, approaching the OMP Oracle upper bound of 97.88% (gap: 0.77%).

---

## Stage 1 Design Requirements

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Single fixed angle θ₀ | Trained on 90° only | ✅ |
| Learn Mic→LDV transformation | 97.11% Energy Reduction | ✅ |
| Angle-agnostic features | No angle labels in training | ✅ |
| Frequency-aware | Freq Embedding added | ✅ |

---

## Training Configuration

### Dataset
| Parameter | Value |
|-----------|-------|
| Source | Speech-260 (real speech clips) |
| Angle | 90.0° (single fixed) |
| Frequency Range | 5-300 Hz bins |
| Total Clips | 260 |
| Training Sequences | 159,595 |

### Model Architecture (SeqDT_FreqAware)
```
Input: (correlation_state, RTG, freq_idx)
  │
  ├─ State Embedding: Linear(M_lags=16, d_model=128)
  ├─ RTG Embedding: Linear(1, d_model=128)
  └─ Freq Embedding: Embedding(max_freq=1025, d_model=128)
  │
  ↓ (element-wise sum + LayerNorm)
  │
  GRU(input=128, hidden=256, layers=2)
  │
  ↓
  Policy Head: Linear(256, M_lags=16)
  │
Output: Lag selection logits (16-way)
```

### Training Parameters
| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch Size | 256 |
| Optimizer | Adam |
| Final Val Loss | 1.5525 |
| Final Accuracy | 54.05% |

---

## Results

### Energy Reduction Comparison

| Method | Energy Reduction | Notes |
|--------|------------------|-------|
| OMP Oracle | 97.88% | Upper bound (per-sample optimal) |
| **Freq-Aware DTMin** | **97.11%** | Single model, all frequencies |
| Global (no freq) | 50.8% | Frequency-agnostic baseline |
| Single-bin Oracle | 73.75% | Per-frequency-bin optimal |

### Key Achievement
- **+46.31%** improvement over frequency-agnostic baseline
- **0.77%** gap to Oracle upper bound
- Single model handles all frequency bins (5-300 Hz)

---

## Verification: Angle-Agnostic Implementation

### Evidence of No Angle Information Leakage

1. **Data Loading** (`generate_lag_omp.py`):
   ```python
   angle_str = f"angle_{int(angle)}"  # Only loads "angle_90"
   ```

2. **Training Script** (`run_full_exp.sh`):
   ```bash
   --angle 90.0  # Single angle hardcoded
   ```

3. **Model Input**:
   - `freq_idx`: Frequency bin index (0-1024)
   - `correlation_state`: OMP correlation vector
   - `RTG`: Return-to-go (reward signal)
   - **No angle index or label**

4. **Dataset Class** (`DoALagDataset`):
   - Loads from single angle directory
   - No angle metadata passed to model

---

## Physical Interpretation

### What Stage 1 Learned

The Frequency-Aware DTMin learned to select optimal time lags for Mic→LDV reconstruction **per frequency bin**:

```
For each frequency f:
  Input: Mic spectrogram correlation state
  Output: Optimal lag sequence [ℓ₁, ℓ₂, ℓ₃, ℓ₄]

Physical meaning:
  - Different frequencies have different optimal lags
  - Phase relationship: Δφ = 2πf·Δτ
  - Frequency embedding resolves this ambiguity
```

### Why This Works for Stage 2

The learned features are:
- **Angle-agnostic**: Trained at single angle (90°)
- **Frequency-aware**: Each frequency bin handled appropriately
- **Transferable**: Can be combined with angle-specific H(f,θ)

---

## Files and Artifacts

### Code
| File | Purpose |
|------|---------|
| `scripts/h_exploration/train_dt_lag_seq_rtg.py` | Training script |
| `scripts/h_exploration/generate_lag_omp.py` | OMP teacher generation |
| `scripts/h_exploration/validate_global_dt.py` | Physical validation |
| `run_full_exp.sh` | Full experiment runner |

### Results
| Path | Content |
|------|---------|
| `results/dt_freq_aware_full/` | Model checkpoints |
| `results/full_data/lag_trajectories.pt` | Training data |
| `results/full_data/full_exp.log` | Training logs |

### Model Checkpoint
- Location: `results/dt_freq_aware_full/best_model.pt`
- Can be loaded for Stage 2 fine-tuning

---

## Next Steps: Stage 2

### Objective
Use Stage 1 features + H(f, θ₁...θ₃₇) for direction estimation (37-way classification)

### Plan
1. Load pre-trained Stage 1 encoder
2. Add Direction Head: `f(features, H_matrices) → direction_logits`
3. Ablation experiments:
   - A: Frozen encoder (validate feature quality)
   - B: Fine-tune all (best accuracy)
   - C: End-to-end baseline (prove pretraining value)

### Expected Deliverables
- `scripts/stage2_direction/train_direction.py`
- `scripts/stage2_direction/eval_direction.py`
- Direction accuracy > 2.7% (random baseline = 1/37)

---

## Conclusion

Stage 1 successfully implemented the core concept of learning **angle-agnostic, frequency-aware transformation features**. The 97.11% Energy Reduction validates that the model has learned effective Mic→LDV transformation without relying on angle information.

This sets the foundation for Stage 2, where these features will be combined with angle-specific H matrices for direction estimation.

**Stage 1: ✅ Complete**
