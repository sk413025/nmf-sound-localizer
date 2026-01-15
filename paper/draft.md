---
title: "Frequency-Aware Decision Transformer for Acoustic Transfer Function Learning"
abstract: |
  We propose a Frequency-Aware Decision Transformer for learning acoustic transfer functions between microphone and laser Doppler vibrometer (LDV) signals. The key challenge is phase ambiguity: the same phase difference corresponds to different time lags at different frequencies ($\Delta\phi = 2\pi f \Delta\tau$). While Orthogonal Matching Pursuit (OMP) achieves 97.88% energy reduction through iterative search, it cannot perform one-shot inference. Our method adds frequency embeddings to condition the policy on frequency context, enabling the model to learn frequency-specific phase-lag strategies. Experiments on 260 speech clips show that the Frequency-Aware DT achieves 97.11% energy reduction, matching OMP's physical limit while enabling single forward-pass inference.
---

# 1. Introduction

Estimating the acoustic transfer function from microphone to laser Doppler vibrometer (LDV) signals is fundamental for remote speech sensing applications. The transfer function can be modeled as a sparse FIR filter:

$$y_t = \sum_{k=0}^{K} \alpha_k x_{t-\tau_k}$$

where $y$ is the LDV signal, $x$ is the microphone signal, and $\tau_k$ are the selected time lags.

**Challenge: Phase Ambiguity.** The relationship between phase difference and time lag is frequency-dependent:

$$\Delta\phi = 2\pi f \Delta\tau$$

This creates aliasing: the same correlation pattern can correspond to different optimal lags at different frequencies. A frequency-blind policy must learn a "compromise" strategy that fails to capture frequency-specific dynamics.

**Existing Limitations.** Orthogonal Matching Pursuit (OMP) achieves near-optimal energy reduction (97.88%) but requires iterative greedy search. Previous attempts at learning-based approaches:
- Stepwise MLP: 36.4% (no context)
- Sequential GRU: 44.8% (history but frequency-blind)
- Global Decision Transformer: 50.8% (still frequency-blind)

**Our Contribution.** We propose a Frequency-Aware Decision Transformer that:
1. Adds frequency embedding to condition the policy on frequency context
2. Achieves 97.11% energy reduction (within 0.77% of OMP physical limit)
3. Enables single forward-pass inference, replacing iterative search

# 2. Method

## 2.1 Problem Formulation

Given microphone signal $x$ and LDV signal $y$, we seek a sparse set of time lags $\{\tau_0, \tau_1, ..., \tau_{K-1}\}$ that maximizes energy capture:

$$\text{Reduction} = 1 - \frac{\|y - \hat{y}\|^2}{\|y\|^2}$$

where $\hat{y} = \sum_k \alpha_k x_{t-\tau_k}$ and coefficients $\alpha_k$ are computed via projection.

## 2.2 Frequency-Aware Architecture

The model receives three inputs per step:
- **Correlation vector**: $\mathbf{c} \in \mathbb{R}^{16}$ (correlations with 16 candidate lags)
- **Return-to-Go (RTG)**: Remaining energy reducible
- **Frequency index**: $f \in \{0, 1, ..., 1024\}$ (frequency bin)

The embedding combines these inputs:

$$\mathbf{e} = \text{LayerNorm}(\text{Linear}(\mathbf{c})) + \text{FreqEmbed}(f) + \text{RTGEmbed}(\text{rtg})$$

Key architectural choice: `nn.Embedding(1025, d_model)` for frequency conditioning.

The Transformer processes the sequence of $(s_t, a_t, \text{rtg}_t)$ tokens and outputs action logits over 16 possible lags.

![Architecture](figures/architecture.png)

## 2.3 Training

We use behavior cloning from OMP oracle trajectories:
- Generate trajectories using OMP on training data
- Each trajectory: $(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_{K-1}, a_{K-1}, r_{K-1})$
- Train with cross-entropy loss on action predictions

# 3. Experiments

## 3.1 Dataset

- **Speech260**: 260 speech clips, 16kHz sampling rate
- **Frequency range**: Bins 5-300 (~150 Hz to 9 kHz)
- **Episode length**: K=3 steps (select 3 lags per frequency bin)
- **Train/Val split**: 80%/20%

## 3.2 Baselines

| Method | Description | Energy Reduction |
|--------|-------------|------------------|
| Naive (Lag 0,1,2) | Fixed lag selection | 35.4% |
| Stepwise MLP | Markov assumption | 36.4% |
| Sequential GRU | History context | 44.8% |
| GRU + RTG | Goal conditioning | 46.0% |
| Global DT | Transformer, freq-blind | 50.8% |
| **OMP Oracle** | Iterative search | **97.88%** |

## 3.3 Implementation Details

- Model: Transformer with 4 layers, 4 heads, d_model=128
- Optimizer: AdamW, lr=5e-4
- Training: 20 epochs, batch size 256
- Hardware: Apple M2 Ultra (MPS)

# 4. Results

## 4.1 Main Results

**Frequency-Aware DT achieves 97.11% energy reduction**, within 0.77% of the OMP physical limit.

| Method | Energy Reduction | Gap to OMP |
|--------|------------------|------------|
| Global DT (freq-blind) | 50.83% | -47.05% |
| Single-Bin (50-60) | 73.75% | -24.13% |
| **Freq-Aware DT** | **97.11%** | **-0.77%** |
| OMP Oracle | 97.88% | -- |

![Results Comparison](figures/results_comparison.png)

## 4.2 Ablation Study

| Variant | Accuracy | Energy Reduction |
|---------|----------|------------------|
| Full model (Freq-Aware) | -- | 97.11% |
| w/o Freq Embedding | 50.83% | 50.83% |
| Single-Bin only (50-60) | 73.75% | 73.75% |

The frequency embedding is essential: removing it causes a 46% drop in performance.

## 4.3 Analysis

**Frequency Embedding Similarity.** Visualization shows the model learns meaningful frequency representations:
- Similar frequencies have similar embeddings
- Distinct clusters form for different frequency ranges

![Frequency Embedding Similarity](figures/freq_embedding_sim.png)

**Phase Strategy Adaptation.** The model learns different lag-selection strategies for different frequencies:
- Low frequencies (Bin 50): Prefers larger lags
- High frequencies (Bin 200): Prefers smaller lags

This matches the physical expectation from $\Delta\phi = 2\pi f \Delta\tau$.

# 5. Conclusion

We presented Frequency-Aware Decision Transformer for acoustic transfer function learning. By conditioning on frequency context, our method resolves phase ambiguity and achieves 97.11% energy reduction, matching OMP's physical limit while enabling single-pass inference.

**Limitations.** Current work focuses on single-source scenarios with stationary sources.

**Future Work.** Extensions to multi-source separation and moving source tracking.

# References

<!-- TODO: Add references -->
