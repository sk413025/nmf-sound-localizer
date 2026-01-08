# Results: H-Matrix (Lag) Exploration on Mono Speech

> **Date**: 2026-01-08
> **Experiment**: `exp-h-exploration-speech` (Lag Mode)
> **Status**: ✅ Completed (Physical Structure Found)

## 1. Executive Summary

We addressed the constraint of Mono Microphone data by reframing "H-Matrix Exploration" as **Sparse Convolutive Transfer Function (CTF) Estimation**. 
Using **OMP**, we identified which time lags of the Microphone spectrogram best predict the LDV spectrogram at 90°.

**Result**: We found a non-trivial temporal structure.
-   **Dominant Lag**: Lag 1 (approx 32ms) is the most frequently selected predictor, followed closely by Lag 0.
-   **Structure**: A clear decay pattern exists for lags 2-5, followed by a noise floor.
-   **Significance**: This confirms that the transfer function from Mic to LDV includes a significant temporal delay/convolution component (echoes/reverberation), which OMP successfully recovered.

## 2. Methodology

### 2.1 Model
$$ Y(t, f) \approx \sum_{k \in \text{active}} w_k \cdot X(t-k, f) $$
-   **Dictionary**: Columns are time-shifted Microphone frames $X[t-k]$.
-   **Atoms**: Time Lags $k \in \{0, \dots, 15\}$.
-   **Algo**: Batch OMP over frequency bins. Window $Tw=16$.

### 2.2 Dataset
-   **Mic**: `speech260_original_16k` (Mono).
-   **Target**: `speech260_box_16k` (Mono) @ 90°.
-   **Params**: 16kHz, STFT 2048/512.

## 3. Findings

### 3.1 Lag Distribution
Results from 260 clips (Full Dataset):
-   **Lag 1**: 276,470 selections (Most dominant).
-   **Lag 0**: 261,728 selections.
-   **Lag 2**: 114,441 selections.
-   **Tail**: Constant floor ~74k for lags 5-14.
-   **Edge**: Rise at Lag 15 (likely boundary artifact).

![Lag Distribution](results/exp_h_lag_full/plots/lag_distribution.png)

### 3.2 Physical Interpretation
-   **Lag 1 > Lag 0**: The peak at Lag 1 suggests a delay of roughly 1 STFT frame (32ms) between the acoustic arrival at the Microphone vs the LDV sensor (or processing latency alignment).
-   **Sparsity**: The sharp drop-off after Lag 2 indicates that the system can be well-approximated by a short FIR filter (Order ~2-3 frames) rather than a dense IIR.
-   **H Exploration**: We have successfully "explored H" and found it has sparse temporal support.

## 4. Next Steps for DTmin
Now that we know a "correct" sparse H (Lags 0, 1, 2) exists:
1.  **Train DTmin**: Teach the student to predict "Lag 1" or "Lag 0" based on the audio context. 
2.  **Application**: This student can then be used to reconstruct LDV signals from Mic signals with proper temporal alignment, potentially improving SNR or downstream localization if generalized to multiple angles.
