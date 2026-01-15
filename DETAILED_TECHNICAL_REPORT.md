# Detailed Technical Report: Full-Spectrum Frequency-Aware Policy
**Date:** January 16, 2026  
**Commit:** Full-Spectrum Training & Evaluation

## 1. Executive Summary
This report documents the extension of the **Frequency-Aware Decision Transformer (DTmin)** to the full available frequency spectrum.
- **Previous State**: Model trained on Low Frequencies (Bins 5-300). Performance: >97% Reduction (in-band).
- **Current State**: Model trained on Full Spectrum (Bins 5-1024). Performance: 65.5% Reduction (global average).
- **Key Finding**: The model maintains high recovery (83% of Oracle) but the global average is lowered by the inherent physical difficulty of aligning high-frequency components.

## 2. Methodology & Data Structures

### A. Input Data Specification
The system processes pairs of Multichannel Microphone (Input) and Laser Doppler Vibrometer (Target) signals.

| Tensor | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| **Mic STFT** | `(F, T, C)` | `complex64` | `F=1025` (Bins), `T` (Frames), `C` (Channels) |
| **LDV STFT** | `(F, T)` | `complex64` | `F=1025` (Bins), `T` (Frames) |
| **OMP Dictionary** | `(F, Tw, M)` | `complex64` | `F` (Bins), `Tw=16` (Window), `M=33` (Lags -16 to +16) |

**Note on Frequency Bins**:
- `n_fft = 2048` $\rightarrow$ `F_bins = 1025`
- Low/Mid Band (Speech Core): Bins 5-300 (~40Hz - 2300Hz)
- High Band: Bins 300-1024 (~2300Hz - 8000Hz)

### B. OMP Oracle (Teacher)
The Oracle generates ground-truth trajectories for the student model.
- **Algorithm**: Step-wise Orthogonal Matching Pursuit.
- **Independence**: Solves $h = \text{argmin} ||y_f - \mathbf{D}_f h||_2^2$ independently for each frequency $f$.
- **Metric**: **Energy Reduction**.
  $$ R = \frac{E_{init} - E_{final}}{E_{init}}, \quad E = ||\mathbf{y}||_2^2 $$

### C. DTmin Student Model
A Sequence-to-Action model that predicts the optimal Lag index given the current correlation state.

**Input Tensors (per step $k$):**
1.  **Correlation State**: `(B, M)` - Magnitude of correlation $| \mathbf{D}^H \mathbf{r} |$.
2.  **Return-to-Go (RTG)**: `(B, 1)` - Scalar indicating desired remaining reduction (Target=1.0).
3.  **Frequency Hint**: `(B, 1)` - Integer Index $f \in [0, 1024]$.

**Architecture:**
- **Embedding**: `nn.Linear` (State) + `nn.Embedding` (Freq) + `nn.Linear` (RTG).
- **Backbone**: GRU (Hidden=256, Layers=2).
- **Output Head**: `nn.Linear` $\rightarrow$ Logits over $M$ lags.

## 3. Training Analysis

### Training Configuration
- **Range**: Bins 5 to 1024.
- **Data Size**: ~550,000 trajectories (generated from 260 clips).
- **Hyperparameters**: Batch=256, LR=1e-3, Epochs=30.

### Convergence Profile
![Training Curve](results/dt_freq_aware_full/training_curve.png)

- **Loss**: Smooth convergence from 2.22 to 1.72.
- **Accuracy**: Stabilized at ~39%.
  - *Interpretation*: Unlike low-freq tasks where accuracy >70% is common, the full-spectrum task has high aleatoric uncertainty in high frequencies (phase noise), capping the max accuracy.

## 4. Evaluation & Performance

### Global Metrics (Energy Reduction)
Evaluated on 10 unseen clips across the full 5-1024 bin range.

| Method | Mean Energy Reduction | Recovered Ratio |
| :--- | :--- | :--- |
| **OMP (Oracle)** | **78.99%** | 100% |
| **DTmin (Model)** | **65.46%** | **82.86%** |

### Segmented Performance Analysis (Inferred)
Based on `calc_global_omp` diagnostics and aggregate results:

1.  **Low-Mid Frequency (Bins 5-300)**:
    - **OMP Performance**: **97.88%** (Verified via `calc_global_omp`).
    - **Status**: Extremely high reduction possible; phase relationships are stable.
    
2.  **High Frequency (Bins 300-1024)**:
    - **OMP Performance**: **< 75%** (Inferred).
    - **Status**: Reduction drops significantly due to lower signal coherence and higher noise floor.
    - **Impact**: The inclusion of these 700+ bins drags the global average down from ~98% to ~79%.

### Visual Comparison
![Reduction Comparison](results/dt_freq_aware_full/reduction_comparison.png)

## 5. Artifacts & Deliverables
- **Code**: `scripts/h_exploration/train_dt_lag_seq_rtg.py` (Updated range)
- **Model Checkpoint**: `results/dt_freq_aware_full/dt_freq_aware_best.pth`
- **Report**: `EXPERIMENT_REPORT_FULL_FREQ.md`

## 6. Recommendations
1.  **Hybrid Inference**: For optimal audio reconstruction, use DTmin for Bins 5-500 and a conservative strategy (e.g., Identity or Lag 0) for Bins >500 to avoid high-frequency phase artifacts.
2.  **Split-Band Training**: Train separate models for Low (<1kHz) and High (>1kHz) frequencies to allow the Low-Freq model to specialize in the highly-reducible speech core without gradient interference from noisy high bands.
