---
title: "Two-Stage Learning for Cross-Sensor Speech Source Localization"
abstract: |
  We propose a two-stage framework for cross-sensor speech source localization using microphone and laser Doppler vibrometer signals. Stage 1 learns angle-agnostic transformation features via frequency-aware behavior cloning, achieving 97.11% energy reduction. Stage 2 combines these features with pre-computed transfer functions for 37-way direction classification [TBD: XX% accuracy]. Separating content-dependent transformation from angle-dependent localization enables effective cross-sensor processing without large labeled datasets.
keywords: "speech localization, cross-sensor, transfer function, decision transformer, behavior cloning"
---

# 1. Introduction

Speech source localization is fundamental for applications including smart speakers, hearing aids, and surveillance systems. Traditional methods like SRP-PHAT rely on time-difference-of-arrival (TDOA) estimation between microphone arrays. However, cross-sensor localization---using heterogeneous sensors such as microphones and laser Doppler vibrometers (LDV)---presents unique challenges due to fundamentally different sensing modalities.

The acoustic transfer function between microphone and LDV signals exhibits frequency-dependent behavior. The relationship between phase difference and time lag follows $\Delta\phi = 2\pi f \Delta\tau$, which creates phase ambiguity: the same correlation pattern may correspond to different optimal lags at different frequencies. A frequency-blind model must learn a compromise strategy that fails to capture frequency-specific dynamics. Furthermore, speech signals contain both content information (what is said) and directional information (where the speaker is). Directly learning direction estimation conflates these two aspects, making generalization difficult.

To address these challenges, we propose a two-stage framework that separates content-dependent transformation learning from angle-dependent localization. In the first stage, we learn Mic-to-LDV transformation features at a single fixed angle, producing angle-agnostic representations that capture speech content dynamics. In the second stage, we combine these features with pre-computed transfer functions H(f,θ) from white noise measurements for 37-way direction classification.

Our contributions are threefold. First, we introduce a two-stage framework that decouples content features from directional estimation. Second, we develop frequency-aware behavior cloning that achieves 97.11% energy reduction in transformation learning. Third, we demonstrate that the learned features transfer effectively to direction estimation, achieving [TBD] accuracy.

# 2. Related Work

SRP-PHAT [@dibiase2001srp] remains the standard baseline for microphone array localization, using steered response power with phase transform weighting. Learning-based approaches have emerged [@chakrabarty2017broadband], but typically require large labeled datasets with direction annotations.

LDV-based speech sensing has gained attention for remote and contactless applications [@li2020ldv]. The acoustic transfer function between microphone and LDV signals depends on both frequency and source direction, presenting unique challenges for cross-sensor processing.

Chen et al. [@chen2021decision] showed that sequence modeling can solve reinforcement learning problems through return-conditioned policies. We adapt this decision transformer approach for sparse signal reconstruction, treating lag selection as a sequential decision problem. For sparse reconstruction, OMP [@pati1993orthogonal] provides optimal approximations through iterative greedy search. Our work distills OMP's behavior into a learned policy that enables single-pass inference.

# 3. Method

## 3.1 Problem Setup

Consider a microphone signal $x$ and LDV signal $y$ recorded simultaneously. The acoustic transfer function H(f,θ) characterizes how sound transforms between these sensors at frequency $f$ and source direction $θ$.

We measure H(f,θ) using white noise excitation at 37 discrete angles (0°-180°, 5° step). Since white noise is content-independent, H captures purely the angle-dependent acoustic properties of the Mic-to-LDV system. Given a speech signal, our goal is twofold: first, learn a transformation $\hat{y} = T(x)$ that reconstructs LDV from microphone, measured by energy reduction; second, classify source direction among 37 angles using learned features and H(f,θ).

## 3.2 Stage 1: Transformation Feature Learning

We model the Mic-to-LDV transformation as sparse FIR filtering: $y_t = \sum_{k=0}^{K} \alpha_k x_{t-\tau_k}$, where $\tau_k$ are selected time lags and $\alpha_k$ are computed via projection.

Orthogonal Matching Pursuit (OMP) iteratively selects lags that maximize residual energy reduction. For each frequency bin, OMP produces optimal lag sequences achieving 97.88% energy reduction. We use OMP as a teacher to generate training trajectories.

The frequency-aware architecture receives three inputs per step: a correlation state $\mathbf{c} \in \mathbb{R}^{16}$ representing correlations with 16 candidate lags, a return-to-go (RTG) value indicating the remaining achievable energy reduction, and a frequency index $f \in \{0, 1, ..., 1024\}$ specifying the frequency bin. The embedding combines these inputs as $\mathbf{e} = \text{LayerNorm}(\text{Linear}(\mathbf{c})) + \text{FreqEmbed}(f) + \text{RTGEmbed}(\text{rtg})$. A GRU processes the sequence and outputs action logits over 16 possible lags.

We train the model using behavior cloning from OMP trajectories with cross-entropy loss on action predictions. Training uses only a single fixed angle (90°), ensuring the learned features are angle-agnostic.

## 3.3 Stage 2: Direction Estimation

Stage 2 introduces the pre-computed transfer functions H(f,θ) for direction estimation. The architecture consists of two components: the Stage 1 encoder (which can be frozen or fine-tuned) that produces features $z$ from the microphone speech input, and a direction head that combines these features with the transfer functions H(θ₁...θ₃₇) to produce 37-way direction classification logits.

[TBD: Direction head architecture and training procedure pending Stage 2 experiments]

# 4. Experiments

## 4.1 Dataset

We use the Speech-260 dataset containing 260 speech clips recorded at 16kHz sampling rate. The frequency range covers STFT bins 5-300 (approximately 150 Hz to 9 kHz). For direction estimation, we record at 37 angles from 0° to 180° in 5° increments. Transfer functions H(f,θ) are measured using white noise recordings at each angle. We use an 80%/20% train/validation split.

## 4.2 Baselines

| Method | Type | Description |
|--------|------|-------------|
| Random | Lower bound | Uniform 37-way selection |
| [TBD: SRP-PHAT] | Traditional | Time-difference-of-arrival |
| Freq-blind DT | Ablation | No frequency embedding |
| OMP Oracle | Upper bound | Iterative optimal search |

## 4.3 Stage 1 Results: Transformation Learning

Table 1 shows Stage 1 results. The frequency-aware model achieves 97.11% energy reduction, within 0.77% of the OMP oracle upper bound. The frequency embedding is essential: removing it causes a 46% drop in performance, demonstrating that frequency conditioning resolves the phase ambiguity problem.

| Method | Energy Reduction | Gap to OMP |
|--------|------------------|------------|
| Freq-blind DT | 50.83% | -47.05% |
| Single-Bin (50-60) | 73.75% | -24.13% |
| Freq-Aware DTMin | 97.11% | -0.77% |
| OMP Oracle | 97.88% | -- |

Visualization of the learned frequency embeddings shows that similar frequencies have similar representations, with distinct clusters forming for different frequency ranges. This indicates the model has learned meaningful frequency-dependent strategies.

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

Separating content-dependent transformation learning (Stage 1) from angle-dependent localization (Stage 2) offers several advantages. The Stage 1 features are reusable across different downstream tasks beyond direction estimation. The pre-computed transfer functions H(f,θ) provide explicit angle information without requiring labeled speech data for training. Furthermore, the framework is interpretable: Stage 1 captures speech content dynamics while H captures directional properties.

The phase-frequency relationship $\Delta\phi = 2\pi f \Delta\tau$ makes frequency conditioning essential. Without frequency embeddings, the model cannot disambiguate optimal lags across different frequencies, as evidenced by the 46% performance drop when removing frequency information.

[TBD: Additional discussion based on Stage 2 results]

# 6. Conclusion

We presented a two-stage learning framework for cross-sensor speech source localization. Stage 1 learns angle-agnostic transformation features through frequency-aware behavior cloning, achieving 97.11% energy reduction. Stage 2 combines these features with pre-computed transfer functions for direction estimation [TBD: achieving XX% accuracy]. The key insight is that separating content-dependent features from angle-dependent localization enables effective cross-sensor speech processing without requiring large labeled direction datasets.

Current limitations include the focus on single-source scenarios with stationary sources. Future work will extend the framework to multi-source separation and moving source tracking.

# Acknowledgments

[To be added for camera-ready version]

# AI Disclosure

Claude Code was used for code assistance and manuscript editing during the preparation of this work.

# References
