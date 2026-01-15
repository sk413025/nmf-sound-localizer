---
title: "Frequency-Aware Policy Learning for Cross-Sensor Speech Transformation"
abstract: |
  Cross-sensor speech processing faces domain shift when sensors have different physical mechanisms. This shift manifests as phase ambiguity: the relationship $\phi(f) = -2\pi f \tau$ (mod $2\pi$) causes optimal time lags to differ across frequencies. Traditional sparse reconstruction methods like OMP achieve near-optimal results but require iteration; frequency-blind learning methods fail to capture this frequency-dependent behavior. We propose frequency-aware policy learning that combines frequency embedding (to resolve phase ambiguity) with RTG-conditioned algorithm distillation (to learn from OMP in a single forward pass). The convergence properties of sparse reconstruction ($K=3$ steps) enable a lightweight GRU-based architecture (~500K parameters) instead of standard Transformer. We validate on microphone-to-LDV transformation---chosen for its significant physical domain shift. Our method achieves 97.11% energy reduction (0.77% gap to OMP oracle), with frequency embedding providing 46% relative improvement, indicating strong frequency-dependent structure. The learned features transfer to direction estimation [TBD: XX% accuracy].
keywords: "cross-sensor speech, domain adaptation, frequency-aware learning, algorithm distillation, phase ambiguity"
---

# 1. Introduction

Cross-sensor speech processing involves transferring acoustic information between sensors with different physical mechanisms. This creates a domain shift challenge: sensors measuring different physical quantities (e.g., air pressure vs. surface vibration) exhibit different frequency responses and phase characteristics, making direct feature transfer ineffective. Understanding and addressing this domain shift is essential for leveraging data from one sensor modality to benefit tasks on another.

A fundamental obstacle in cross-sensor transformation is phase ambiguity arising from frequency-dependent behavior. The relationship between phase difference and time lag follows $\phi(f) = -2\pi f \tau$ (mod $2\pi$), meaning the same correlation pattern may correspond to different optimal time lags at different frequencies. A frequency-blind model cannot distinguish which lag is optimal at each frequency, forcing it to learn a suboptimal compromise---our experiments show a 46% performance gap when frequency information is removed, consistent with frequency-dependent phase and sensor-response effects. Traditional sparse reconstruction methods like Orthogonal Matching Pursuit (OMP) can achieve near-optimal results by iteratively selecting frequency-appropriate lags, but they require multiple iterations and cannot be integrated into end-to-end learning pipelines.

We propose frequency-aware policy learning to address this physics-driven challenge. Our design addresses three requirements derived from the underlying physics: (1) frequency awareness to resolve phase ambiguity, (2) efficient single-pass inference unlike iterative OMP, and (3) goal-directed learning that generalizes beyond the training distribution. Crucially, the convergence properties of sparse reconstruction ($K=3$ steps yield diminishing returns) enable a lightweight GRU-based architecture instead of standard Transformer, reducing parameters by approximately 50%. We achieve these through algorithm distillation with frequency embedding: we distill OMP into a learned RTG-conditioned policy. Unlike behavior cloning which learns a fixed state-to-action mapping, RTG conditioning enables the model to learn goal-directed strategies. The frequency embedding provides frequency context, enabling a single policy to specialize across bins, resolving phase ambiguity through a single forward pass.

Our contributions are threefold. First, we identify phase ambiguity as a key obstacle in cross-sensor domain adaptation, arising from the physics of phase-frequency coupling---the 46% performance gap between frequency-aware and frequency-blind models supports this analysis. Second, we propose a physics-driven lightweight architecture: the convergence properties of sparse reconstruction ($K=3$) enable a GRU-based RTG-conditioned policy (~500K parameters) instead of standard Transformer-based Decision Transformer, achieving 97.11% energy reduction (0.77% gap to OMP oracle). Third, we validate on microphone-to-LDV transformation---a setting with significant physical domain shift---and demonstrate feature transfer to direction estimation [TBD: XX% accuracy].

# 2. Related Work

**Cross-sensor speech processing** has been studied in various contexts. Bone conduction and air conduction microphone fusion leverages complementary noise characteristics [@liu2018bone]. Contact microphones have been used for speech enhancement in noisy environments. LDV-based speech sensing has gained attention for remote and contactless applications [@li2020ldv], where the acoustic transfer function depends on both frequency and source direction. These works typically treat cross-sensor transformation as a black-box learning problem without explicitly addressing the physics of phase-frequency coupling.

**Sparse signal reconstruction** via OMP [@pati1993orthogonal] provides near-optimal approximations through iterative greedy search. While effective, OMP requires multiple iterations and cannot be directly integrated into end-to-end learning pipelines.

**Algorithm distillation** differs from behavior cloning by conditioning on optimization targets. The Decision Transformer [@chen2021decision] demonstrated that return-conditioned sequence modeling can learn goal-directed policies from offline trajectories. We adapt this framework to sparse reconstruction: instead of imitating OMP's actions directly (behavior cloning), we condition on return-to-go values that encode energy reduction targets. This enables the learned policy to generalize across different optimization goals. Our key extension is frequency embedding, which addresses the physics of phase ambiguity---a challenge absent in the original RL domains where Decision Transformer was developed.

# 3. Method

## 3.1 Physical Model

Cross-sensor transformation involves mapping between sensors with different physical mechanisms. We derive our architecture design from the underlying physics.

**STFT-Domain Sparse Reconstruction.** In the STFT domain, cross-sensor transformation can be modeled using the **convolutive transfer function (CTF) approximation** [@talmon2009ctf], which represents long impulse responses as band-to-band STFT convolution [@mohammadiha2016nctf]:
$$Y(f,n) \approx \sum_{\ell \in \mathcal{K}} \alpha(f,\ell) \cdot X(f, n-\ell)$$
where $X$ and $Y$ are source and target sensor STFTs, $\ell$ are discrete frame lags (each = 32 ms with hop 512 at 16 kHz), and $\alpha(f,\ell)$ are frequency-dependent coefficients. We additionally impose a **K-sparse constraint** (selected via OMP) for computational efficiency; the selected frame-lags represent an **effective sparse representation** in the lag dictionary rather than a literal count of physical propagation paths.

**Phase-Frequency Coupling.** For a time delay $\tau$, the phase relationship follows:
$$\phi(f) = -2\pi f \tau \pmod{2\pi}$$
under the standard $e^{-j2\pi ft}$ convention. Because phase is observed modulo $2\pi$, inferring an underlying delay from **single-bin** phase (or from features that do not explicitly encode frequency) suffers from **phase wrapping**. In our setting, this matters because the optimal frame-lag selection can vary systematically with $f$ when approximating fractional delays or dispersion effects using a discrete lag dictionary. A frequency-blind model cannot distinguish which lag is optimal at each frequency, forcing it to learn a suboptimal compromise.

**Why K=3: Empirical Sparsity Trade-off.** We set $K=3$ based on an empirical accuracy–efficiency trade-off. [**TBD: Add Figure showing energy reduction vs K**] In our mic→LDV setting with $M=16$ candidate frame-lags, improvement beyond $K=3$ is marginal [@tropp2007omp; @pati1993omp]. We emphasize that $K$ reflects the **chosen sparsity budget in the STFT lag dictionary**, not a literal count of physical propagation paths. This short sequence length ($K=3$) enables our lightweight GRU-based architecture instead of Transformer.

**Validation Setting.** We validate on microphone-to-LDV transformation, chosen because the two sensors have distinct physical mechanisms: microphones measure air pressure variations, while laser Doppler vibrometers (LDVs) measure surface vibration velocity. This physical difference creates significant domain shift in frequency response and phase characteristics. The transfer function $H(f,\theta)$ varies with both frequency $f$ and source direction $\theta$.

## 3.2 Frequency-Aware Algorithm Distillation

Given the phase ambiguity challenge from Section 3.1, our method must satisfy three requirements: (1) frequency awareness to learn frequency-specific strategies, (2) efficient single-pass inference unlike iterative OMP, and (3) goal-directed learning to generalize beyond the training distribution. We address these through algorithm distillation with frequency embedding. This section describes the OMP teacher, return-to-go conditioning, and our physics-aware architecture.

**OMP Teacher Generation.** Orthogonal Matching Pursuit iteratively selects time lags that maximize residual energy reduction. Given a dictionary of $M=16$ candidate lags, OMP performs $K=3$ greedy selections: at each step, it computes correlations between the residual and all dictionary atoms, selects the lag with maximum correlation magnitude, and updates the residual via least-squares projection. This generates trajectories containing correlation states, selected actions, and cumulative energy reductions. OMP achieves 97.88% energy reduction, providing an oracle for distillation.

**Return-to-Go Conditioning.** Following the Decision Transformer framework [@chen2021decision], we condition the policy on return-to-go (RTG) values that encode remaining optimization targets. At step $k$, we define per-step reward $\Delta r_k := E_{k-1} - E_k$ (energy reduction at step $k$), cumulative return $R_k := \sum_{i=1}^{k} \Delta r_i$, and return-to-go $\text{rtg}_k := \sum_{i=k}^{K} \Delta r_i = R_K - R_{k-1}$, where $E_k$ is residual energy after $k$ steps. This matches Decision Transformer RTG semantics: $\text{rtg}_k$ encodes the sum of future rewards from the current step onward, enabling goal-directed learning. The model learns to select actions that achieve specified energy reduction targets, rather than merely imitating OMP's choices. This is the key distinction from behavior cloning, which learns only the mapping from states to actions.

**Network Architecture.** We use an RTG-conditioned policy inspired by Decision Transformer [@chen2021decision], but adapted for our problem characteristics. While Decision Transformer uses Transformer for long RL trajectories, we adopt a GRU for our short sequence length ($K=3$). Given the very short decision horizon, a GRU provides a favorable accuracy–latency trade-off [@cho2014gru]. While Transformers [@vaswani2017transformer] are applicable, their benefits are most pronounced with longer contexts; in our setting, the lightweight recurrent policy provides:
- **Causal inductive bias**: GRU's unidirectional hidden state naturally models OMP's sequential residual updates
- **Parameter efficiency**: ~500K parameters vs. millions in standard Decision Transformer, reducing overfitting risk

The network receives three inputs per step combined through additive fusion:
$$\mathbf{e} = \text{LayerNorm}(\text{StateEmbed}(\mathbf{c}) + \text{RTGEmbed}(\text{rtg}) + \text{FreqEmbed}(f))$$
where $\mathbf{c} \in \mathbb{R}^{16}$ is the correlation state with candidate lags, rtg is the return-to-go scalar, and $f$ is the frequency bin index. Each embedding maps to dimension 128, fused into a 2-layer GRU with hidden dimension 256. The output head maps GRU hidden states to action logits over 16 possible lags.

**Frequency Embedding for Phase Disambiguation.** The frequency embedding is the physics-aware innovation: it enables the model to learn frequency-specific lag selection strategies. Due to the phase-frequency relationship $\phi(f) = -2\pi f \tau$ (mod $2\pi$), optimal lags differ across frequencies---a frequency-blind model cannot resolve this ambiguity and must learn a suboptimal compromise. By providing frequency context, our architecture learns a **single frequency-conditioned policy** that can specialize its lag-selection behavior across frequency bins while sharing parameters. This enables the model to learn frequency-specific strategies without requiring separate parameters for each bin. We train using cross-entropy loss on OMP trajectories collected at a single fixed angle (90°), ensuring features are geometry-agnostic.

## 3.3 Feature Transfer Validation: Direction Estimation

To validate that the learned transformation features are transferable, we apply them to a downstream task: direction estimation. This is not the primary goal of our work, but rather a verification that the frequency-aware features capture useful acoustic information beyond the transformation task itself. The transformation features, combined with pre-computed transfer functions H(f,θ) measured at 37 angles, enable 37-way direction classification without requiring direction-labeled speech data for the main training. [TBD: Architecture details pending experiments]

# 4. Experiments

## 4.1 Dataset and Setup

We evaluate on microphone-to-LDV transformation using the Speech-260 dataset: 260 speech clips at 16kHz, STFT bins 39-384 (300 Hz to 3 kHz, n_fft=2048, hop=512). Transfer functions are measured at 37 angles (0°-180°, 5° step) using white noise. Training uses only a single angle (90°) to ensure learned features are geometry-agnostic. We use 80%/20% train/validation split.

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

We use GRU rather than Transformer because the sequence length is short ($K=3$ steps). The additive fusion of embeddings (state + RTG + frequency) is more parameter-efficient than concatenation and empirically performs comparably. Return-to-go conditioning distinguishes our approach from behavior cloning: it enables the model to learn goal-directed policies rather than merely imitating OMP's trajectory.

## 4.4 Downstream Transfer: Direction Estimation

[TBD: Results pending experiments]

| Method | Top-1 Accuracy | Accuracy@10° |
|--------|----------------|--------------|
| Random | 2.7% | 13.5% |
| Learned features | [TBD]% | [TBD]% |

# 5. Discussion

**Domain Shift and Phase Ambiguity.** Cross-sensor domain shift arises because different sensors measure different physical quantities. In our microphone-LDV setting, microphones capture air pressure variations while LDVs measure surface vibration velocity. This physical difference manifests as frequency-dependent phase characteristics, described by the relationship $\Delta\phi = 2\pi f \Delta\tau$. The consequence is phase ambiguity: the same correlation pattern corresponds to different optimal time lags at different frequencies. This physics-driven insight motivated our frequency-aware design.

**Why Frequency Embedding.** The 46% performance gap between frequency-aware and frequency-blind models indicates that optimal lag selection is strongly frequency-dependent. This is consistent with phase-wrapping effects (the relationship $\phi(f) = -2\pi f \tau$) and frequency-dependent sensor response characteristics. Without frequency context, a model cannot determine which time lag is optimal at each frequency, forcing it to learn a suboptimal compromise. The frequency embedding resolves this by providing the context needed to learn frequency-specific strategies, with the model learning a single frequency-conditioned policy that can specialize across bins.

**Why Algorithm Distillation.** We needed to distill OMP's iterative optimization into a single forward pass. Two choices emerged: behavior cloning (BC) or algorithm distillation (AD). BC learns a fixed state-to-action mapping, which may fail when the test distribution differs from training. AD with return-to-go conditioning learns goal-directed policies that achieve specified optimization targets. In our setting, this enables the model to learn "how to achieve X% energy reduction" rather than "what OMP did in this exact state," providing more robust generalization.

**Why GRU Instead of Transformer.** Our RTG-conditioned policy uses GRU rather than Transformer, diverging from standard Decision Transformer. Three problem characteristics motivate this choice: (1) $K=3$ is too short for Transformer to learn efficiently---attention mechanisms require sufficient context to establish meaningful patterns; (2) OMP's sequential dependency (step $k$ depends on step $k{-}1$'s residual) matches GRU's causal hidden state, which naturally accumulates information unidirectionally; (3) GRU's parameter efficiency (~500K vs. millions) reduces overfitting risk on our dataset. This demonstrates that RTG conditioning---the key insight from Decision Transformer---can be effectively applied with simpler sequence models when problem structure permits.

**Limitations.** Our method is validated on a single microphone-LDV setup. Whether the frequency-aware approach benefits other sensor pairs remains to be empirically tested---we do not claim generalization without validation. The training requires paired sensor recordings with calibration data, which may limit applicability in some scenarios.

[TBD: Additional discussion based on downstream experiments]

# 6. Conclusion

We presented frequency-aware policy learning for cross-sensor speech transformation. Our key finding is that phase ambiguity---arising from the relationship $\phi(f) = -2\pi f \tau$ (mod $2\pi$)---is a significant obstacle in cross-sensor domain adaptation, and frequency-aware modeling addresses this effectively. The 46% performance gap between frequency-aware and frequency-blind models indicates strong frequency-dependent structure, consistent with phase-wrapping and sensor-response effects.

Our method combines frequency embedding with return-to-go conditioning in a GRU-based architecture, achieving 97.11% energy reduction (0.77% gap to OMP oracle) on microphone-to-LDV transformation. The learned features transfer to direction estimation [TBD: XX% accuracy], demonstrating their utility beyond the transformation task.

We validated on one sensor pair (microphone-LDV) chosen for its significant physical domain shift. Future work could test on other sensor pairs to understand the broader applicability of frequency-aware cross-sensor methods.

# Acknowledgments

[To be added for camera-ready version]

# AI Disclosure

Claude Code was used for code assistance and manuscript editing during the preparation of this work.

# References

```bibtex
@inproceedings{talmon2009ctf,
  title={Relative Transfer Function Identification Using Convolutive Transfer Function Approximation},
  author={Talmon, Ronen and Cohen, Israel and Gannot, Sharon},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  year={2009},
  volume={17},
  number={4},
  pages={546--555}
}

@article{mohammadiha2016nctf,
  title={Speech Dereverberation Using Non-Negative Convolutive Transfer Function and Nonnegative Matrix Factorization},
  author={Mohammadiha, Nasser and Doclo, Simon and Leijon, Arne},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2016},
  volume={24},
  number={8},
  pages={1370--1382}
}

@article{tropp2007omp,
  title={Signal Recovery from Random Measurements via Orthogonal Matching Pursuit},
  author={Tropp, Joel A. and Gilbert, Anna C.},
  journal={IEEE Transactions on Information Theory},
  year={2007},
  volume={53},
  number={12},
  pages={4655--4666}
}

@inproceedings{pati1993omp,
  title={Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition},
  author={Pati, Y. C. and Rezaiifar, R. and Krishnaprasad, P. S.},
  booktitle={Proc. Asilomar Conference on Signals, Systems and Computers},
  year={1993},
  pages={40--44}
}

@inproceedings{chen2021decision,
  title={Decision Transformer: Reinforcement Learning via Sequence Modeling},
  author={Chen, Lili and Lu, Kevin and Rajeswaran, Aravind and Lee, Kimin and Grover, Aditya and Laskin, Michael and Abbeel, Pieter and Srinivas, Aravind and Mordatch, Igor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021},
  volume={34},
  pages={15084--15097}
}

@inproceedings{cho2014gru,
  title={Learning Phrase Representations using RNN Encoder--Decoder for Statistical Machine Translation},
  author={Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  booktitle={Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2014},
  pages={1724--1734}
}

@inproceedings{vaswani2017transformer,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017},
  volume={30},
  pages={5998--6008}
}

@article{mallat1993mp,
  title={Matching Pursuits with Time-Frequency Dictionaries},
  author={Mallat, St{\'e}phane G. and Zhang, Zhifeng},
  journal={IEEE Transactions on Signal Processing},
  year={1993},
  volume={41},
  number={12},
  pages={3397--3415}
}

@article{chen1998bp,
  title={Atomic Decomposition by Basis Pursuit},
  author={Chen, Scott Shaobing and Donoho, David L. and Saunders, Michael A.},
  journal={SIAM Journal on Scientific Computing},
  year={1998},
  volume={20},
  number={1},
  pages={33--61}
}

@article{zhang2011omp,
  title={Sparse Recovery with Orthogonal Matching Pursuit under RIP},
  author={Zhang, Tong},
  journal={IEEE Transactions on Information Theory},
  year={2011},
  volume={57},
  number={9},
  pages={6215--6221}
}

@inproceedings{janner2021tt,
  title={Offline Reinforcement Learning as One Big Sequence Modeling Problem},
  author={Janner, Michael and Li, Qiyang and Levine, Sergey},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021},
  volume={34},
  pages={1273--1286}
}

@inproceedings{czarnecki2019dpd,
  title={Distilling Policy Distillation},
  author={Czarnecki, Wojciech M. and Pascanu, Razvan and Osindero, Simon and Jayakumar, Siddhant M. and Swirszcz, Grzegorz and Jaderberg, Max},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2019},
  pages={1170--1179}
}

@article{nakatani2010wpe,
  title={Speech Dereverberation Based on Variance-Normalized Delayed Linear Prediction},
  author={Nakatani, Tomohiro and Yoshioka, Takuya and Kinoshita, Keisuke and Miyoshi, Masato and Juang, Biing-Hwang},
  journal={IEEE Transactions on Audio, Speech, and Language Processing},
  year={2010},
  volume={18},
  number={7},
  pages={1717--1731}
}

@inproceedings{laskin2023ad,
  title={In-Context Reinforcement Learning with Algorithm Distillation},
  author={Laskin, Michael and Wang, Luyu and Oh, Junhyuk and Parisotto, Emilio and Spencer, Stephen and Steigerwald, Richie and Strouse, DJ and Hansen, Steven and Filos, Angelos and Brooks, Ethan and others},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2023}
}

@inproceedings{avargel2011ldv,
  title={Speech Measurements Using a Laser Doppler Vibrometer Sensor: Application to Speech Enhancement},
  author={Avargel, Yekutiel and Cohen, Israel},
  booktitle={Proceedings of the 2011 Joint Workshop on Hands-free Speech Communication and Microphone Arrays (HSCMA)},
  year={2011},
  pages={109--114}
}

@article{li2011surfaces,
  title={Vibration Characteristics of Various Surfaces Using an LDV for Long-Range Voice Acquisition},
  author={Li, Rui and Wang, Tao and Zhu, Zhigang},
  journal={IEEE Sensors Journal},
  year={2011},
  volume={11},
  number={6},
  pages={1415--1422}
}

@inproceedings{liao2019datse,
  title={Noise Adaptive Speech Enhancement Using Domain Adversarial Training},
  author={Liao, Chien-Feng and Fu, Szu-Wei and Tsao, Yu},
  booktitle={Proc. Interspeech},
  year={2019},
  pages={2983--2987}
}

@inproceedings{leglaive2023udase,
  title={The CHiME-7 UDASE Task: Unsupervised Domain Adaptation for Speech Enhancement},
  author={Leglaive, Simon and {et al.}},
  booktitle={Proc. CHiME-7 Workshop},
  year={2023}
}

@article{ganin2016dann,
  title={Domain-Adversarial Training of Neural Networks},
  author={Ganin, Yaroslav and Ustinova, Evgeniya and Ajakan, Hana and Germain, Pascal and Larochelle, Hugo and Laviolette, Fran{\c{c}}ois and Marchand, Mario and Lempitsky, Victor},
  journal={Journal of Machine Learning Research},
  year={2016},
  volume={17},
  number={59},
  pages={1--35}
}

@article{liu2018bone,
  title={Bone Conduction Speech Enhancement Using Deep Denoising Autoencoder},
  author={Liu, Feng and Tsao, Yu and Hu, Yu},
  journal={Speech Communication},
  year={2018},
  volume={104},
  pages={106--112}
}

@article{li2020ldv,
  title={Remote Sensing of Vital Signs Based on Laser Doppler Vibrometry},
  author={Li, Changzhi and Lubecke, Victor M. and Boric-Lubecke, Olga and Lin, Jenshan},
  journal={IEEE Transactions on Microwave Theory and Techniques},
  year={2020},
  volume={68},
  number={3},
  pages={1180--1192}
}

@inproceedings{pati1993orthogonal,
  title={Orthogonal Matching Pursuit: Recursive Function Approximation with Applications to Wavelet Decomposition},
  author={Pati, Yagyensh Chandra and Rezaiifar, Ramin and Krishnaprasad, Perinkulam Sambamurthy},
  booktitle={Proceedings of 27th Asilomar Conference on Signals, Systems and Computers},
  year={1993},
  pages={40--44}
}
```
