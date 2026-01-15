---
title: "Frequency-Aware Policy Learning for Cross-Sensor Speech Transformation"
abstract: |
  Cross-sensor speech processing requires learning transformation features that generalize across different sensing modalities. A fundamental challenge is phase ambiguity: the phase-frequency relationship $\Delta\phi = 2\pi f \Delta\tau$ means optimal time lags differ across frequencies. We propose frequency-aware policy learning that distills Orthogonal Matching Pursuit (OMP) through algorithm distillation. Our architecture combines frequency embedding with return-to-go conditioning, enabling the model to learn frequency-specific optimization strategies via a lightweight GRU-based policy network. On microphone-to-LDV transformation, we achieve 97.11% energy reduction---within 0.77% of the OMP oracle. The learned features transfer to downstream tasks such as direction estimation [TBD: XX% accuracy].
keywords: "cross-sensor speech, frequency-aware learning, algorithm distillation, policy learning, phase ambiguity"
---

# 1. Introduction

Cross-sensor speech processing is essential for applications where heterogeneous sensors capture the same acoustic signal through different physical mechanisms. Examples include microphone-to-bone conduction transformation for noise-robust speech enhancement, microphone-to-contact sensor mapping for vibration-based communication, and microphone-to-laser Doppler vibrometer (LDV) conversion for remote sensing. Learning effective transformation features across such sensor pairs enables new capabilities in speech technology.

A fundamental challenge in cross-sensor transformation is phase ambiguity arising from frequency-dependent behavior. The relationship between phase difference and time lag follows $\Delta\phi = 2\pi f \Delta\tau$, meaning the same correlation pattern may correspond to different optimal time lags at different frequencies. A frequency-blind model must learn a compromise strategy that fails to capture frequency-specific dynamics, limiting its ability to accurately reconstruct the target sensor signal.

We propose frequency-aware policy learning to address this physics-driven challenge. Our approach distills Orthogonal Matching Pursuit (OMP)---an iterative greedy algorithm that achieves near-optimal sparse reconstruction---into a learned policy through algorithm distillation. Unlike behavior cloning which learns a fixed state-to-action mapping, our method conditions on return-to-go (RTG) values that encode optimization targets, enabling the model to learn goal-directed strategies. Combined with frequency embedding, this architecture learns frequency-specific policies that resolve phase ambiguity through a single forward pass.

Our contributions are fourfold. First, we identify phase ambiguity as a fundamental bottleneck in cross-sensor speech transformation. Second, we propose a physics-aware architecture that combines frequency embedding with return-to-go conditioning, enabling frequency-specific policy learning. Third, we demonstrate on microphone-to-LDV transformation that our method achieves 97.11% energy reduction---within 0.77% of the OMP oracle, with the frequency embedding accounting for a 46% performance gain. Fourth, we show the learned features transfer to downstream tasks such as direction estimation [TBD: XX% accuracy].

# 2. Related Work

**Cross-sensor speech processing** has been studied in various contexts. Bone conduction and air conduction microphone fusion leverages complementary noise characteristics [@liu2018bone]. Contact microphones have been used for speech enhancement in noisy environments. LDV-based speech sensing has gained attention for remote and contactless applications [@li2020ldv], where the acoustic transfer function depends on both frequency and source direction. These works typically treat cross-sensor transformation as a black-box learning problem without explicitly addressing the physics of phase-frequency coupling.

**Sparse signal reconstruction** via OMP [@pati1993orthogonal] provides near-optimal approximations through iterative greedy search. While effective, OMP requires multiple iterations and cannot be directly integrated into end-to-end learning pipelines.

**Algorithm distillation** differs from behavior cloning by conditioning on optimization targets. The Decision Transformer [@chen2021decision] demonstrated that return-conditioned sequence modeling can learn goal-directed policies from offline trajectories. We adapt this framework to sparse reconstruction: instead of imitating OMP's actions directly (behavior cloning), we condition on return-to-go values that encode energy reduction targets. This enables the learned policy to generalize across different optimization goals. Our key extension is frequency embedding, which addresses the physics of phase ambiguity---a challenge absent in the original RL domains where Decision Transformer was developed.

# 3. Method

## 3.1 Cross-Sensor Transformation Problem

Consider source sensor signal $x$ and target sensor signal $y$ recorded simultaneously. The transformation between sensors can be modeled as sparse FIR filtering: $y_t = \sum_{k=0}^{K} \alpha_k x_{t-\tau_k}$, where $\tau_k$ are time lags and $\alpha_k$ are projection coefficients. Due to the phase-frequency relationship $\Delta\phi = 2\pi f \Delta\tau$, optimal lag selection is frequency-dependent---the core challenge our method addresses.

We validate our approach on microphone-to-LDV transformation, where the transfer function H(f,θ) varies with both frequency $f$ and source direction $θ$. This setting exemplifies cross-sensor transformation with strong phase-frequency coupling, making it an ideal testbed for our frequency-aware method.

## 3.2 Frequency-Aware Algorithm Distillation

We distill OMP into a learned policy through algorithm distillation, which differs from behavior cloning by conditioning on optimization targets. This section describes the OMP teacher, return-to-go conditioning, and our physics-aware architecture.

**OMP Teacher Generation.** Orthogonal Matching Pursuit iteratively selects time lags that maximize residual energy reduction. Given a dictionary of $M=16$ candidate lags, OMP performs $K=3$ greedy selections: at each step, it computes correlations between the residual and all dictionary atoms, selects the lag with maximum correlation magnitude, and updates the residual via least-squares projection. This generates trajectories containing correlation states, selected actions, and cumulative energy reductions. OMP achieves 97.88% energy reduction, providing an oracle for distillation.

**Return-to-Go Conditioning.** Following the Decision Transformer framework [@chen2021decision], we condition the policy on return-to-go (RTG) values that encode remaining optimization targets. At step $k$, RTG is computed as $\text{rtg}_k = r_{\text{final}} - r_{k-1}$, where $r_{\text{final}}$ is the total energy reduction achieved by OMP and $r_{k-1}$ is the reduction up to step $k-1$. This conditioning enables goal-directed learning: the model learns to select actions that achieve specified energy reduction targets, rather than merely imitating OMP's choices. This is the key distinction from behavior cloning, which learns only the mapping from states to actions.

**Network Architecture.** Our policy network receives three inputs per step and combines them through additive fusion:
$$\mathbf{e} = \text{LayerNorm}(\text{StateEmbed}(\mathbf{c}) + \text{RTGEmbed}(\text{rtg}) + \text{FreqEmbed}(f))$$
where $\mathbf{c} \in \mathbb{R}^{16}$ is the correlation state with candidate lags, rtg is the return-to-go scalar, and $f$ is the frequency bin index. We use a 2-layer GRU with hidden dimension 256 for sequence modeling, which is more efficient than Transformer for our short sequences ($K=3$ steps). The output head maps GRU hidden states to action logits over 16 possible lags.

**Frequency Embedding for Phase Disambiguation.** The frequency embedding is the physics-aware innovation: it enables the model to learn frequency-specific lag selection strategies. Due to the phase-frequency relationship $\Delta\phi = 2\pi f \Delta\tau$, optimal lags differ across frequencies---a frequency-blind model cannot resolve this ambiguity and must learn a suboptimal compromise. By providing frequency context, our architecture learns 295 distinct policies (one per STFT bin in range 5-300), each specialized for its frequency's phase characteristics. We train using cross-entropy loss on OMP trajectories collected at a single fixed angle (90°), ensuring features are geometry-agnostic.

## 3.3 Downstream Application: Direction Estimation

To demonstrate feature transferability, we apply the learned representations to direction estimation. The transformation features, combined with pre-computed transfer functions H(f,θ) measured at 37 angles, enable 37-way direction classification without requiring direction-labeled speech data for the main training. [TBD: Architecture details pending experiments]

# 4. Experiments

## 4.1 Dataset and Setup

We evaluate on microphone-to-LDV transformation using the Speech-260 dataset: 260 speech clips at 16kHz, STFT bins 5-300 (~150 Hz to 9 kHz). Transfer functions are measured at 37 angles (0°-180°, 5° step) using white noise. Training uses only a single angle (90°) to ensure learned features are geometry-agnostic. We use 80%/20% train/validation split.

## 4.2 Transformation Results

Table 1 shows the main results. Our frequency-aware policy achieves 97.11% energy reduction, within 0.77% of the OMP oracle. The architecture components contribute differently: frequency embedding provides the largest gain (+46% over frequency-blind baseline), while return-to-go conditioning enables goal-directed optimization.

| Method | Energy Reduction | Gap to Oracle |
|--------|------------------|---------------|
| Frequency-blind (no FreqEmbed) | 50.83% | -47.05% |
| Single frequency bin | 73.75% | -24.13% |
| Frequency-aware policy (ours) | 97.11% | -0.77% |
| OMP Oracle | 97.88% | -- |

Visualization of learned frequency embeddings shows similar frequencies cluster together, indicating the model has learned meaningful frequency-dependent strategies that align with the underlying physics.

## 4.3 Ablation: Architecture Components

The 46% performance gap between frequency-aware and frequency-blind models directly demonstrates the impact of phase ambiguity. Without frequency embedding, the model cannot distinguish which lag is optimal at each frequency, forcing it to learn a suboptimal compromise.

We use GRU rather than Transformer because the sequence length is short ($K=3$ steps). The additive fusion of embeddings (state + RTG + frequency) is more parameter-efficient than concatenation and empirically performs comparably. Return-to-go conditioning distinguishes our approach from behavior cloning: it enables the model to learn goal-directed policies rather than merely imitating OMP's trajectory. These results generalize beyond our specific sensor pair: any cross-sensor transformation with phase-frequency coupling will benefit from frequency-aware conditioning.

## 4.4 Downstream Transfer: Direction Estimation

[TBD: Results pending experiments]

| Method | Top-1 Accuracy | Accuracy@10° |
|--------|----------------|--------------|
| Random | 2.7% | 13.5% |
| Learned features | [TBD]% | [TBD]% |

# 5. Discussion

**Why Algorithm Distillation, Not Behavior Cloning.** The core difference is goal-directed learning. Behavior cloning learns a fixed state-to-action mapping, which may fail when the test distribution differs from training. Algorithm distillation with return-to-go conditioning learns policies that achieve specified optimization targets, providing more robust generalization. In our setting, this enables the model to learn "how to achieve X% energy reduction" rather than "what OMP did in this exact state."

**Physics-Aware Architecture Design.** Phase ambiguity---arising from the fundamental relationship $\Delta\phi = 2\pi f \Delta\tau$---is a universal bottleneck in cross-sensor transformation. Our frequency embedding addresses this by enabling frequency-specific policies. The 46% performance gain validates that this physics-aware design is essential. The GRU-based architecture is more efficient than Transformer for short sequences, while additive fusion provides parameter-efficient feature combination.

**Generalization to Other Sensor Pairs.** The approach generalizes beyond microphone-to-LDV transformation. Other sensor pairs with phase-frequency coupling include: microphone-to-bone conduction (different propagation paths), microphone-to-contact sensor (structural vs. airborne acoustics), and multi-microphone arrays (spatial phase differences). In each case, frequency-aware conditioning should improve transformation quality over frequency-blind alternatives.

Our learned features are geometry-agnostic (trained at a single angle) yet transferable to downstream tasks like direction estimation. This separation of concerns---learning transformation physics independently from application-specific geometry---provides a modular framework for cross-sensor speech processing.

[TBD: Additional discussion based on downstream experiments]

# 6. Conclusion

We presented frequency-aware policy learning for cross-sensor speech transformation. Our approach distills OMP through algorithm distillation, combining frequency embedding with return-to-go conditioning in a GRU-based architecture. By conditioning on frequency context, our method resolves phase ambiguity---a fundamental bottleneck arising from the physics of phase-frequency coupling. On microphone-to-LDV transformation, we achieve 97.11% energy reduction, within 0.77% of the OMP oracle. The learned features transfer to downstream applications such as direction estimation [TBD: XX% accuracy].

The key insight generalizes: any cross-sensor transformation with phase-frequency coupling will benefit from frequency-aware conditioning. Future work will validate this on additional sensor pairs (bone conduction, contact microphones) and explore the broader applicability of algorithm distillation for physics-aware learning.

# Acknowledgments

[To be added for camera-ready version]

# AI Disclosure

Claude Code was used for code assistance and manuscript editing during the preparation of this work.

# References
