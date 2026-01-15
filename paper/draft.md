---
title: "Two-Stage Learning for Cross-Sensor Speech Source Localization"
abstract: |
  We propose a two-stage learning framework for speech source localization using microphone and laser Doppler vibrometer (LDV) signals. Stage 1 learns angle-agnostic transformation features through frequency-aware behavior cloning from Orthogonal Matching Pursuit (OMP), achieving 97.11% energy reduction. Stage 2 combines these features with pre-computed transfer functions H(f,θ) across 37 angles for direction estimation. The key insight is separating content-dependent transformation learning from angle-dependent localization. Experiments on 260 speech clips demonstrate that our method [TBD: achieves XX% direction accuracy], outperforming [TBD: SRP-PHAT baseline] while enabling single forward-pass inference.
---

# 1. Introduction

Speech source localization is fundamental for applications including smart speakers, hearing aids, and surveillance systems. Traditional methods like SRP-PHAT rely on time-difference-of-arrival (TDOA) estimation between microphone arrays. However, cross-sensor localization---using heterogeneous sensors such as microphones and laser Doppler vibrometers (LDV)---presents unique challenges due to fundamentally different sensing modalities.

**Challenge: Frequency-Dependent Phase Ambiguity.** The acoustic transfer function between microphone and LDV signals exhibits frequency-dependent behavior. The relationship between phase difference and time lag follows:

$$\Delta\phi = 2\pi f \Delta\tau$$

This creates aliasing: the same correlation pattern corresponds to different optimal lags at different frequencies. A frequency-blind model must learn a "compromise" strategy that fails to capture frequency-specific dynamics.

**Challenge: Entangled Content and Direction.** Speech signals contain both content information (what is said) and directional information (where the speaker is). Directly learning direction estimation conflates these two aspects, making generalization difficult.

**Our Solution: Two-Stage Separation.** We propose separating content-dependent transformation learning from angle-dependent localization:

- **Stage 1**: Learn Mic→LDV transformation features at a single fixed angle, producing angle-agnostic representations that capture speech content dynamics.
- **Stage 2**: Combine Stage 1 features with pre-computed transfer functions H(f,θ) from white noise measurements for 37-way direction classification.

**Contributions:**

1. A two-stage framework that decouples content features from directional estimation
2. Frequency-aware behavior cloning achieving 97.11% energy reduction (Stage 1)
3. [TBD: Direction estimation achieving XX% accuracy (Stage 2)]

# 2. Related Work

**Acoustic Source Localization.** SRP-PHAT [@dibiase2001srp] remains the standard baseline for microphone array localization. Learning-based approaches have emerged [@chakrabarty2017broadband], but typically require large labeled datasets with direction annotations.

**Cross-Sensor Speech Processing.** LDV-based speech sensing has gained attention for remote and contactless applications [@li2020ldv]. The acoustic transfer function between microphone and LDV signals depends on both frequency and source direction.

**Decision Transformers.** Chen et al. [@chen2021decision] showed that sequence modeling can solve reinforcement learning problems. We adapt this approach for sparse signal reconstruction, treating lag selection as a sequential decision problem.

**Sparse Reconstruction.** OMP [@pati1993orthogonal] provides optimal sparse approximations through iterative greedy search. Our work distills OMP's behavior into a learned policy that enables single-pass inference.

# 3. Method

## 3.1 Problem Setup

Consider a microphone signal $x$ and LDV signal $y$ recorded simultaneously. The acoustic transfer function H(f,θ) characterizes how sound transforms between these sensors at frequency $f$ and source direction $θ$.

**Transfer Function Pre-computation.** We measure H(f,θ) using white noise excitation at 37 discrete angles (0°-180°, 5° step). Since white noise is content-independent, H captures purely the angle-dependent acoustic properties of the Mic-to-LDV system.

**Task Formulation.** Given a speech signal, we aim to:
1. **Stage 1**: Learn a transformation $\hat{y} = T(x)$ that reconstructs LDV from microphone, measured by energy reduction
2. **Stage 2**: Classify source direction among 37 angles using learned features and H(f,θ)

## 3.2 Stage 1: Transformation Feature Learning

We model the Mic→LDV transformation as sparse FIR filtering:

$$y_t = \sum_{k=0}^{K} \alpha_k x_{t-\tau_k}$$

where $\tau_k$ are selected time lags and $\alpha_k$ are computed via projection.

**OMP Teacher.** Orthogonal Matching Pursuit iteratively selects lags that maximize residual energy reduction. For each frequency bin, OMP produces optimal lag sequences achieving 97.88% energy reduction.

**Frequency-Aware Architecture.** The model receives per-step inputs:
- **Correlation state**: $\mathbf{c} \in \mathbb{R}^{16}$ (correlations with 16 candidate lags)
- **Return-to-Go (RTG)**: Remaining achievable energy reduction
- **Frequency index**: $f \in \{0, 1, ..., 1024\}$ (frequency bin)

The embedding combines these:

$$\mathbf{e} = \text{LayerNorm}(\text{Linear}(\mathbf{c})) + \text{FreqEmbed}(f) + \text{RTGEmbed}(\text{rtg})$$

A GRU processes the sequence and outputs action logits over 16 possible lags.

**Training.** We use behavior cloning from OMP trajectories with cross-entropy loss on action predictions. Training uses only a single fixed angle (90°), ensuring the learned features are angle-agnostic.

## 3.3 Stage 2: Direction Estimation

[TBD: This section will be completed after Stage 2 experiments]

Stage 2 introduces the pre-computed transfer functions H(f,θ) for direction estimation.

**Architecture:**
```
Mic Speech ──→ [Stage 1 Encoder] ──→ Features z
                 (frozen/fine-tune)        │
                                           ↓
                              ┌────────────────────────┐
                              │   Direction Head       │
                              │   f(z, H(θ₁...θ₃₇))   │
                              └────────────────────────┘
                                           │
                                           ↓
                                Direction logits (37-way)
```

**Direction Head.** [TBD: Architecture details pending implementation]

**Training.** [TBD: Training procedure pending implementation]

# 4. Experiments

## 4.1 Dataset

- **Speech-260**: 260 speech clips at 16kHz sampling rate
- **Frequency range**: Bins 5-300 (~150 Hz to 9 kHz)
- **Angles**: 37 directions (0°-180°, 5° step)
- **H measurement**: White noise recordings per angle
- **Train/Val split**: 80%/20%

## 4.2 Baselines

| Method | Type | Description |
|--------|------|-------------|
| Random | Lower bound | Uniform 37-way selection |
| [TBD: SRP-PHAT] | Traditional | Time-difference-of-arrival |
| Freq-blind DT | Ablation | No frequency embedding |
| OMP Oracle | Upper bound | Iterative optimal search |

## 4.3 Stage 1 Results: Transformation Learning

**Frequency-Aware DTMin achieves 97.11% energy reduction**, within 0.77% of the OMP physical limit.

| Method | Energy Reduction | Gap to OMP |
|--------|------------------|------------|
| Freq-blind DT | 50.83% | -47.05% |
| Single-Bin (50-60) | 73.75% | -24.13% |
| **Freq-Aware DTMin** | **97.11%** | **-0.77%** |
| OMP Oracle | 97.88% | -- |

The frequency embedding is essential: removing it causes a 46% drop in performance.

**Frequency Embedding Analysis.** Visualization shows the model learns meaningful frequency representations with similar frequencies having similar embeddings and distinct clusters forming for different frequency ranges.

## 4.4 Stage 2 Results: Direction Estimation

[TBD: Results pending Stage 2 experiments]

| Method | Top-1 Accuracy | Accuracy@10° |
|--------|----------------|--------------|
| Random Baseline | 2.7% | 13.5% |
| [TBD: SRP-PHAT] | [TBD]% | [TBD]% |
| [TBD: End-to-End] | [TBD]% | [TBD]% |
| [TBD: Frozen Stage 1] | [TBD]% | [TBD]% |
| [TBD: Fine-tuned] | [TBD]% | [TBD]% |

## 4.5 Ablation Study

[TBD: Complete ablation pending Stage 2]

| Variant | Energy Reduction | Direction Acc |
|---------|------------------|---------------|
| Full model | 97.11% | [TBD]% |
| w/o Freq Embedding | 50.83% | [TBD]% |
| w/o Stage 1 pretrain | N/A | [TBD]% |

# 5. Discussion

**Why Two Stages?** Separating content-dependent transformation learning (Stage 1) from angle-dependent localization (Stage 2) has several benefits:
1. Stage 1 features are reusable across different downstream tasks
2. H(f,θ) provides explicit angle information without requiring labeled speech data
3. The framework is interpretable: Stage 1 captures "what" while H captures "where"

**Frequency Awareness.** The phase-frequency relationship $\Delta\phi = 2\pi f \Delta\tau$ makes frequency conditioning essential. Without it, the model cannot disambiguate optimal lags across frequencies.

[TBD: Additional discussion based on Stage 2 results]

# 6. Conclusion

We presented a two-stage learning framework for cross-sensor speech source localization. Stage 1 learns angle-agnostic transformation features through frequency-aware behavior cloning, achieving 97.11% energy reduction. Stage 2 combines these features with pre-computed transfer functions for direction estimation [TBD: achieving XX% accuracy].

**Key Insight:** Separating content-dependent features from angle-dependent localization enables effective cross-sensor speech processing without requiring large labeled direction datasets.

**Limitations.** Current work focuses on single-source scenarios with stationary sources.

**Future Work.** Extensions to multi-source separation and moving source tracking.

# References
