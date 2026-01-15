# Experiment Report: Full-Spectrum Frequency-Aware Policy
**Date:** January 16, 2026  
**Commit:** Full-Spectrum Training & Evaluation

## 1. Overview
This experiment extends the `Frequency-Aware Decision Transformer (DTmin)` from a limited low-frequency band (5-300) to the **Full Spectrum (Bins 5-1024)**. The goal is to verify if a single policy model can learn to align phases across all meaningful frequencies, including challenging high-frequency components.

## 2. Data Processing Pipeline
The data preparation ensures that the model learns from "Physical Oracle" (OMP) trajectories.

### A. Input Data pairs
- **Mic (Input)**: Multichannel STFT, processed into slices.
- **LDV (Target)**: Single-channel Laser Doppler Vibrometer STFT (Ground Truth).
- **Format**: `complex64` STFT with `n_fft=2048`, `hop=default`.

### B. OMP (Teacher) Generation
We use **Orthogonal Matching Pursuit (OMP)** to generate ground truth "Lag Selection" trajectories.
- **Algorithm**: Per-Bin Independent OMP.
- **Dictionary**: Time-shifted versions of the input Mic STFT (Lags: -16 to +16).
- **Selection**: At each step $k$, OMP selects the Lag that maximizes correlation with the current residual.
- **Reduction Metric**: **Energy Reduction** (L2 Squared).
  $$ \text{Reduction} = \frac{||y||_2^2 - ||r||_2^2}{||y||_2^2} $$
- **Output**: A dataset of sequences `(Correlation_State, Action, Reward/Reduction)` for 1025 frequency bins.

## 3. Model Architecture (DTmin)
The student model is a **Sequence-to-Action Decision Transformer**.

### Architecture
- **Input**:
  - `Observation`: Correlation profile (Magnitude of correlation with all 16 lags).
  - `RTG (Return-to-Go)`: Desired remaining energy reduction (Target=1.0).
  - `Frequency Hint`: **Learned Embedding** for Frequency ID (0-1024).
- **Core**: GRU / Transformer Encoder.
- **Output**: Logits for Lag Selection (Actions).

### Frequency Awareness
The key innovation is the `freq_embed`. The model shares all weights across the spectrum but receives a "Hint" about which frequency it is processing. This allows it to adapt its phase-alignment strategy for Low vs High frequencies.

## 4. Training Results
- **Scope**: Frequency Bins 5 to 1024 (Full Spectrum).
- **Dataset**: ~550k sequences generated from 260 clips.
- **Epochs**: 30.

### Training Trajectory
![Training Curve](results/dt_freq_aware_full/training_curve.png)

- **Loss**: Converged from 2.22 to ~1.72.
- **Accuracy**: Stabilized around **36-39%** (Top-1 Lag Prediction).
- **Observation**: Accuracy is lower than the single-band experiment (~74%) because high-frequency bins are inherently noisy and harder to predict exactly.

## 5. Evaluation Results
We evaluated the trained model on unseen clips using the **Energy Reduction** metric, compared against the OMP Oracle.

### Performance Comparison
![Reduction Comparison](results/dt_freq_aware_full/reduction_comparison.png)

| Method | Mean Energy Reduction | Notes |
| :--- | :--- | :--- |
| **OMP (Oracle)** | **78.99%** | Theoretical upper bound per-bin. |
| **DTmin (Model)** | **65.46%** | Real-world performance. |

### Recovery Analysis
- **Recovery Ratio**: The model recovers **~83%** of the Oracle's performance.
- **Low-Freq Performance**: In previous tests (Bins 5-300), OMP achieved >97% reduction.
- **High-Freq Impact**: The inclusion of high frequencies (300-1024) lowered the average OMP ceiling to 79%, indicating these bands are physically harder to align.

## 6. Conclusion
1.  **Feasibility**: It is possible to train a single Frequency-Aware Policy for the full spectrum.
2.  **Effectiveness**: Achieving **65% Energy Reduction** across the full band is a strong result, suitable for initialization or coarse alignment.
3.  **Strategy**: For highest fidelity, a **Hybrid Approach** (DTmin for Low/Mid + OMP/Identity for High) or split-band models might yield better results due to the disparate physics of low vs high frequency signals.
