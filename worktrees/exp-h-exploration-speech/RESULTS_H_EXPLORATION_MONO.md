# Results: H-Matrix Exploration (Mono Speech)

> **Date**: 2026-01-08
> **Experiment**: `exp-h-exploration-speech`
> **Status**: ✅ Completed (Trivial Result)

## 1. Executive Summary

This experiment aimed to use **OMP (Orthogonal Matching Pursuit)** to discover the optimal H-matrix (beamforming weights) for mapping complex Speech data (Microphone) to LDV data, and then distill this into a **DTmin (Decision Transformer)** student.

**Result**: The experiment revealed that the `speech260_original_16k` dataset is **Mono (Single Channel)**.
-   **OMP Selection**: Because there is only 1 channel, OMP has no "selection" to make. It simply selects Index 0.
-   **DTmin Learning**: The student learned to predict Action 0 with 100% accuracy and ~0.0 loss.
-   **Reconstruction**: The OMP/Least Squares fit on this single channel provides a "best fit" scalar transfer function per frequency, but does not leverage multi-channel beamforming.

## 2. Methodology & Setup

### 2.1 Dataset
-   **Source**: `/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized` (Mic)
-   **Target**: `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized` (LDV)
-   **Sample Rate**: 16kHz
-   **FFT**: 2048 point STFT, Hann window.
-   **Band**: 300Hz - 3000Hz.
-   **Processing**:
    -   Computed STFT for Mic ($X$) and LDV ($Y$).
    -   Input $X$ shape found to be `[F, T, 1]` (Mono).

### 2.2 Algorithms
-   **Teacher (OMP)**: configured to select up to $K$ channels.
    -   Since $C=1$, $K=1$.
    -   Solved $h = (X^H X)^{-1} X^H Y$.
-   **Student (DTmin)**:
    -   Input: OMP Observation (correlation magnitudes), RTG (Final MSE).
    -   Output: Channel Index.

### 2.3 Execution
-   **Trajectory Generation**: `scripts/h_exploration/generate_omp_trajectories.py`
    -   Processed 9,620 clips.
    -   Generated 336,700 frequency-bin trajectories.
-   **Training**: `scripts/h_exploration/train_dtmin_h.py`
    -   5 Epochs.
    -   Batch Size 32.

## 3. Findings

### 3.1 Quantitative Results
-   **Action Distribution**: 100% of actions are Index 0.
-   **MSE**: Average Mean Squared Error of $5.78 \times 10^{-7}$.
-   **Training Loss**: Converged to `0.0` at Epoch 3.

![Action Distribution](results/exp_h_full/plots/action_distribution.png)

### 3.2 Physical Interpretation
The "Original" speech microphone utilized in `speech260_original` appears to be a single reference microphone, rather than a microphone array. 
Therefore, the problem $X_{mic} \cdot H = Y_{ldv}$ reduces to finding a scalar transfer function $H(f)$ such that $H(f) \approx Y(f)/X(f)$. 
There is no "exploration" or "sparsity" involved, as there are no competing channels to select from.

## 4. Next Steps

To achieve the original goal of **Beamforming Discovery / H-Matrix Exploration**:
1.  **Locate Multichannel Data**: We need a dataset where $X$ has shape `[F, T, C]` with $C > 1$ (e.g., Eigenmike or Tetrahedral array).
2.  **Synthetic Array**: We could synthesize a microphone array by using `speech260` sources and simulating room impulse responses (RIRs) for an array geometry.
3.  **Frequency Bin Selection**: Alternatively, we could frame the problem as selecting *which frequency bins* are most informative for reconstruction (sparse spectral sampling), but this differs from spatial beamforming.

This experiment serves as a **baseline valid pipeline** implementation. The code works; the data was just degenerate for the specific task.
