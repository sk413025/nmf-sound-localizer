# Plan: H-Matrix (Impulse Response) Exploration on Mono Speech

## 1. Objective
Since the input Microphone data is **Mono**, we cannot perform Spatial Beamforming (Channel Selection).
Instead, we interpret "H Matrix Exploration" as **Acoustic Impulse Response (AIR) Estimation** using a **Sparse Filter** model.
We aim to discover the sparse temporal structure (echoes/delays) that maps the Microphone signal to the LDV signal for a specific angle (e.g., 90°).

## 2. Methodology: Convolutive Transfer Function (CTF)
We operate in the STFT domain.
For each frequency bin $f$, we model the LDV signal $Y(t, f)$ as a sparse linear combination of past Microphone frames $X(t, f)$:
$$ Y(t, f) \approx \sum_{\tau=0}^{K} w_\tau \cdot X(t-\tau, f) $$

### 2.1 Teacher: OMP for Lag Selection
- **Dictionary**: Matrix where columns are time-lagged versions of the Microphone spectrogram.
    - $D = [X_t, X_{t-1}, X_{t-2}, \dots, X_{t-M}]$
- **Target**: Current LDV frame $Y_t$.
- **Optimization**: OMP selects the most significant time lags (atoms) and their weights.
- **Result**: A sparse trajectory of Lag selections (e.g., "Lag 0 is dominant", "Lag 3 is an echo").

### 2.2 Student: DTmin Distillation
- **Input**: History of Microphone Spectrograms ($X_{t:t-M}$).
- **Output**: Predict the significant Time Lags (Actions).
- **Goal**: Learn the temporal mapping structure directly from data.

## 3. Dataset Settings
- **Source**: `speech260_original_16k` (Mic), `speech260_box_16k` (LDV).
- **Angle**: Fixed at **90°**.
- **Preprocessing**: 
    - 16kHz Sampling Rate.
    - STFT: n_fft=2048, hop=512 (Standard).
    - Dictionary Depth: $M=16$ frames (~0.5s context).

## 4. Implementation Changes
- **Dataset**: Update to return "Context Windows" of $X$ rather than single frames.
- **OMP**: Change Dictionary from "Channels" to "Time Lags".
- **Visualization**: Plot Lag Distribution (Temporal structure of H).

This approach validates the "OMP + DTmin" distillation pipeline on complex speech data by discovering physically meaningful temporal features (Impulse Response) despite the Mono constraint.
