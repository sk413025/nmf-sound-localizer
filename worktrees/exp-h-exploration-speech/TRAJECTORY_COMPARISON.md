# Trajectory Comparison: DTmin (Current) vs. Soft-OMP Native (Reference)

This document provides a systematic comparison of the input trajectory structures used in the current H-Exploration experiment (`DTmin`) versus the reference `soft-omp-native-repro` experiment.

## 1. High-Level Concept

| Feature | Current DTmin (H-Exploration) | Soft-OMP Native (Reference) |
| :--- | :--- | :--- |
| **Domain** | **Time-Lag** (Temporal Impulse Response) | **Angle-Space** (Spatial Beamforming) |
| **Goal** | Find best *Time Delay* ($0 \dots 16$) | Find best *Angle/Atom* ($0 \dots 37 \times 8$) |
| **State** | Raw **Correlation** Vector | **Residual Embedding** Sequence (`h_seq`) |
| **Input Shape** | `(B, 16)` (Correlation values) | `(B, K, d_model)` (Embedding vectors) |
| **Context** | **None** (Stepwise Independent) | **Sequential** (Transformer Context) |

---

## 2. Trajectory Structure Detailed Comparison

### A. Current DTmin (`h_exploration`)
The current model treats every step as an independent classification problem (Markov Assumption). It does not see the history of previous choices.

*   **Data Source**: `lag_trajectories.pt`
*   **Sample Shape (Single Step)**:
    *   **Input (`corrs`)**: `Float32[16]`
        *   Represents: $\langle r_{residual}, \text{Lag}_k \rangle$ for $k \in [0, 15]$.
        *   Physics: "How much energy does Lag $k$ explain *right now*?"
    *   **Target (`actions`)**: `Int64` (Scalar)
        *   Represents: The index $k$ selected by OMP.

*   **Missing Information**:
    *   **Step Index**: Model doesn't know if this is Step 1 or Step 3.
    *   **Frequency Bin**: Model doesn't know if this is 50Hz or 3000Hz.
    *   **History**: Model doesn't know what was picked previously.

### B. Reference Soft-OMP (`train_angle_range_dtmin.py`)
The reference implementation uses a full **Decision Transformer** architecture that consumes a sequence of states.

*   **Data Source**: Dictionary of Tensors (`collate` function)
*   **Sample Shape (Sequence of Length K)**:
    *   **Input (`h_seq`)**: `Float32[K, d_model]`
        *   Represents: A learned embedding of the residual state at each step.
        *   Note: It is *not* just raw correlations; it often includes projected features.
    *   **Input (`rtg_seq`)** (Optional): `Float32[K, 2]`
        *   Represents: Return-to-Go (Desired Accuracy, Desired Energy).
    *   **Input (`prob_seq`)**: `Float32[K, E*M]`
        *   Represents: The full probability distribution (soft correlations) over all atoms.
    *   **Target (`expert_gt`, `atom_gt`)**:
        *   Hierarchical targets (Which Angle?, Which Atom?).

---

## 3. Why the Performance Gap?

The comparison reveals why current H-Exploration performance (36%) is low compared to the Oracle (74%).

| Dimension | Current DTmin | Reference Soft-OMP | Impact |
| :--- | :--- | :--- | :--- |
| **Information** | **Correlation Only** | **Embedding + RTG + Position** | Current model is "blind" to context. |
| **Memory** | **Memoryless** (Markov) | **Sequential** (Attention) | Current model cannot learn strategies like "If Step 1 was Lag 0, Step 2 should check Lag 15". |
| **Feature Richness** | **Scalar Correlations** | **High-Dim Embeddings** | Scalar values (e.g., 0.8 vs 0.79) are hard for MLP to distinguish noise from signal without context. |

## 4. Proposed Fix (trajectory_alignment)

To match the reference performance, we should upgrade the H-Exploration trajectory to include:

1.  **State Embedding**: Concatenate `Correlations` with `Step_Encoding` and `Frequency_Encoding`.
    *   $x = [Corr_{0\dots15}, PosEnc(k), FreqEnc(f)]$
2.  **Sequence Modeling**: Feed the full sequence of 3 steps into a Transformer, not just individual steps into an MLP.
3.  **Output**: Predict the full sequence of actions.

This aligns the H-Exploration "Lag" problem with the proven "Angle" architecture.
