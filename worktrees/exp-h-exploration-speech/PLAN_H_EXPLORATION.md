# Plan: H Matrix Exploration with OMP and DTmin on Speech

## 1. Objective
Explore the estimation of the H matrix (Beamforming/Transfer Function weights) that maps Microphone data to LDV data using OMP (Orthogonal Matching Pursuit) as a Teacher and DTmin (Decision Transformer) as a Student.
This experiment shifts from "Sound Source Localization" (finding angles) to "Signal Reconstruction" (finding weights to match LDV).

## 2. Dataset
- **Type**: Complex Speech Data.
- **Source**: `speech260` dataset (16kHz).
- **Pairs**: 
    - Input: Microphone Data (`signal_original`).
    - Target: LDV Data (`signal_box`).
- **Structure**: Single angle mapping. $Mic(\theta) \times H \approx LDV(\theta)$.

## 3. Methodology

### 3.1. Data Preparation
- Load aligned chunks of `speech260_original_16k` and `speech260_box_16k`.
- Compute STFT.
- Dimensions:
    - Mic: $X \in [F, T, C]$ (Frequency, Time, Channels).
    - LDV: $Y \in [F, T, 1]$.
    - We operate per frequency bin $f$ (or jointly?). Usually per-bin or slight bandwidth.

### 3.2. Teacher: OMP for H Estimation
For each frequency $f$ (and valid time frames):
1.  **Problem**: Find sparse weights $h \in \mathbb{C}^C$ minimizing $||y - X h||_2$.
2.  **Algorithm**: Orthogonal Matching Pursuit.
    -   Iteratively select microphone channel $c$ that best matches the residual.
    -   Update active set.
    -   Solve Least Squares for active set.
    -   Update residual.
3.  **Output**: Trajectories of $(State, Action, Reward)$.
    -   State: Current Residual (or Mic & LDV context).
    -   Action: Selected Channel Index.
    -   Reward: Reduction in reconstruction error (negative MSE).

### 3.3. Student: DTmin Distillation
1.  **Architecture**: Causal Transformer (DT).
2.  **Input**: Sequence of (Residual, RTG, Action).
3.  **Output**: Next Action (Channel Selection).
4.  **Goal**: Learn to proactively select the best channels to reconstruct LDV, potentially faster or with less info than full OMP?
    -   Or simply distill the OMP logic.

## 4. Implementation Steps
1.  **Dataset Class (`DoAPairDataset`)**: Load Mic & LDV.
2.  **OMP Script**: Generate trajectories from Speech data.
3.  **Training Script**: Train DTmin on these trajectories.
4.  **Evaluation**: Compare OMP vs DTmin reconstruction error on test set.

## 5. Experiment Config
- **Dataset**: `speech260` (Train/Val split).
- **Audio**: 16kHz, STFT (n_fft=2048 or similar).
- **OMP Steps (K)**: Max channels (e.g. 4 or 6).
- **Model**: TinyDT (as in previous exps).

