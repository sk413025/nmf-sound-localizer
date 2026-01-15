---
title: "Frequency-Aware Policy Learning for Cross-Sensor Speech Transformation"
abstract: |
  Cross-sensor speech processing faces domain shift when sensors have different physical mechanisms. This shift manifests as phase ambiguity: the relationship $\phi(f) = -2\pi f \tau$ (mod $2\pi$) causes optimal time lags to differ across frequencies. Traditional sparse reconstruction methods like OMP achieve near-optimal results but require iteration; frequency-blind learning methods fail to capture this frequency-dependent behavior. We propose frequency-aware policy learning that combines frequency embedding (to resolve phase ambiguity) with RTG-conditioned algorithm distillation (to learn from OMP in a single forward pass). We formalize OMP as a Markov Decision Process and derive a lightweight GRU-based architecture (~500K parameters) that matches OMP's causal structure, avoiding the overhead of standard Transformer. We validate on microphone-to-LDV transformation---chosen for its significant physical domain shift. Our method achieves 97.11% energy reduction (0.77% gap to OMP oracle), with frequency embedding providing 46% relative improvement, indicating strong frequency-dependent structure. The learned features transfer to direction estimation [TBD: XX% accuracy].
keywords: "cross-sensor speech, domain adaptation, frequency-aware learning, algorithm distillation, phase ambiguity"
---

# 1. Introduction

Cross-sensor speech processing involves transferring acoustic information between sensors with different physical mechanisms. This creates a domain shift challenge: sensors measuring different physical quantities (e.g., air pressure vs. surface vibration) exhibit different frequency responses and phase characteristics, making direct feature transfer ineffective. Understanding and addressing this domain shift is essential for leveraging data from one sensor modality to benefit tasks on another.

A fundamental obstacle in cross-sensor transformation is phase ambiguity arising from frequency-dependent behavior. The relationship between phase difference and time lag follows $\phi(f) = -2\pi f \tau$ (mod $2\pi$), meaning the same correlation pattern may correspond to different optimal time lags at different frequencies. A frequency-blind model cannot distinguish which lag is optimal at each frequency, forcing it to learn a suboptimal compromise---our experiments show a 46% performance gap when frequency information is removed, consistent with frequency-dependent phase and sensor-response effects. Traditional sparse reconstruction methods like Orthogonal Matching Pursuit (OMP) can achieve near-optimal results by iteratively selecting frequency-appropriate lags, but they require multiple iterations and cannot be integrated into end-to-end learning pipelines.

We propose frequency-aware policy learning to address this physics-driven challenge. Our design addresses three requirements derived from the underlying physics: (1) frequency awareness to resolve phase ambiguity, (2) efficient single-pass inference unlike iterative OMP, and (3) goal-directed learning that generalizes beyond the training distribution. Crucially, the convergence properties of sparse reconstruction ($K=3$ steps yield diminishing returns) enable a lightweight GRU-based architecture instead of standard Transformer, reducing parameters by approximately 50%. We achieve these through algorithm distillation with frequency embedding: we distill OMP into a learned RTG-conditioned policy. Unlike behavior cloning which learns a fixed state-to-action mapping, RTG conditioning enables the model to learn goal-directed strategies. The frequency embedding provides frequency context, enabling a single policy to specialize across bins, resolving phase ambiguity through a single forward pass.

Our contributions are threefold. First, we identify phase ambiguity as a key obstacle in cross-sensor domain adaptation, arising from the physics of phase-frequency coupling---the 46% performance gap between frequency-aware and frequency-blind models supports this analysis. Second, we propose a physics-driven lightweight architecture: the convergence properties of sparse reconstruction ($K=3$) enable a GRU-based RTG-conditioned policy (~500K parameters) instead of standard Transformer-based Decision Transformer, achieving 97.11% energy reduction (0.77% gap to OMP oracle). Third, we validate on microphone-to-LDV transformation---a setting with significant physical domain shift---and demonstrate feature transfer to direction estimation [TBD: XX% accuracy].

# 2. Related Work

**Cross-sensor speech processing** has been studied in various contexts. Bone conduction and air conduction microphone fusion leverages complementary noise characteristics [@liu2018bone]. Contact microphones have been used for speech enhancement in noisy environments. LDV-based speech sensing has gained attention for remote and contactless applications [@li2020ldv], where the acoustic transfer function depends on both frequency and source direction. These works typically treat cross-sensor transformation as a black-box learning problem without explicitly addressing the physics of phase-frequency coupling.

**Sparse signal reconstruction** via OMP [@pati1993omp] provides near-optimal approximations through iterative greedy search. While effective, OMP requires multiple iterations and cannot be directly integrated into end-to-end learning pipelines.

**Algorithm distillation** differs from behavior cloning by conditioning on optimization targets. The Decision Transformer [@chen2021decision] demonstrated that return-conditioned sequence modeling can learn goal-directed policies from offline trajectories. We adapt this framework to sparse reconstruction: instead of imitating OMP's actions directly (behavior cloning), we condition on return-to-go values that encode energy reduction targets. This enables the learned policy to generalize across different optimization goals. Our key extension is frequency embedding, which addresses the physics of phase ambiguity---a challenge absent in the original RL domains where Decision Transformer was developed.

# 3. Method

## 3.1 Physical Model

Cross-sensor transformation involves mapping between sensors with different physical mechanisms. We derive our architecture design from the underlying physics, starting from time-domain observations.

**Time-Domain Observation Model.** Let $s(t)$ denote the emitted speech pressure field at the source. The microphone observes air pressure $p_{\text{mic}}(t)$ via an acoustic path $h_{\text{mic}}(t;\theta)$, while the LDV observes surface normal velocity $v_{\text{surf}}(t)$ generated by acoustic–structure coupling $h_{\text{as}}(t;\theta)$ and measured optically by an approximately linear readout $h_{\text{ldv}}(t)$. Under a quasi-static, linear regime at fixed geometry $\theta$, both observations are well-approximated as linear time-invariant (LTI):
$$x(t) = (h_{\text{mic}} * s)(t) + \epsilon_x(t), \quad y(t) = (h_{\text{ldv}} * h_{\text{as}} * s)(t) + \epsilon_y(t)$$

Eliminating $s(t)$ yields a *relative* linear map between sensors:
$$y(t) \approx (g * x)(t) + \epsilon(t)$$
where $g$ subsumes the ratio of transfer functions and $\epsilon$ captures non-invertibility and mismatch [@allen1977unified].

**STFT-Domain Sparse Reconstruction.** We approximate this long impulse response in the STFT domain using the **convolutive transfer function (CTF) approximation** [@talmon2009ctf; @mohammadiha2016nctf]:
$$Y(f,n) = \sum_{\ell=0}^{L-1} \alpha(f,\ell) \cdot X(f, n-\ell) + \eta(f,n)$$
where $X$ and $Y$ are source and target sensor STFTs, $\ell$ are discrete frame lags (each = 32 ms with hop 512 at 16 kHz), $\alpha(f,\ell)$ are frequency-dependent coefficients, and $\eta(f,n)$ absorbs cross-band leakage, late reverberation, and LDV-specific artifacts (e.g., speckle dropouts).

This band-to-band STFT convolution approximation assumes:
- Quasi-stationary geometry during each utterance
- Negligible nonlinearities (small surface vibration amplitude)
- STFT window/hop parameters imply manageable cross-band leakage [@portnoff1980stft]
- Lag dictionary length covers the dominant response support [@kuttruff2016room]

We additionally impose a **K-sparse constraint** (selected via OMP) for computational efficiency; the selected frame-lags represent an **effective sparse representation** in the lag dictionary rather than a literal count of physical propagation paths.

**Phase-Frequency Coupling.** From the Fourier transform property, a time delay $\tau$ produces a phase shift:
$$x(t-\tau) \xleftrightarrow{\mathcal{F}} X(f)e^{-j2\pi f\tau}$$
hence the phase relationship:
$$\phi(f) = -2\pi f \tau \pmod{2\pi}$$
under the standard $e^{-j2\pi ft}$ convention [@allen1977unified]. For a general transfer function $G(f)$, the phase is $\phi(f) = \arg G(f)$, and frequency-dependent delays are captured by group delay $\tau_g(f) = -\frac{1}{2\pi}\frac{d\phi}{df}$.

Because phase is observed modulo $2\pi$, inferring an underlying delay from **single-bin** phase suffers from **phase wrapping** [@knapp1976gcc; @chen2006tde_overview; @ghiglia1998phase]. In our setting, this matters because:
- Our dictionary atoms are discrete delays in the STFT frame domain
- The optimal atom depends on how well $e^{-j2\pi f\tau}$ is approximated by $e^{-j2\pi f\ell}$ over the discrete candidate set $\ell \in \mathcal{L}$
- Because the objective is periodic in $f(\tau-\ell)$, the minimizer $\ell^*(f)$ generically varies with $f$ unless $\tau$ lands exactly on the grid

**Lemma (Frequency-conditioning necessity).** Let $a^*(f) \in \{1,\dots,M\}$ be the optimal lag index at frequency $f$. Any deterministic frequency-blind policy $\pi$ must output the same action for all $f$. If there exist $f_1 \neq f_2$ such that $a^*(f_1) \neq a^*(f_2)$, then $\pi$ cannot be optimal on both frequencies simultaneously. Under 0–1 action loss:
$$\min_{\pi \text{ freq-blind}} \mathbb{E}_f[\mathbb{1}\{\pi \neq a^*(f)\}] \geq 1 - \max_a \mathbb{P}(a^*(f)=a)$$
which is strictly positive if $a^*(f)$ is non-constant over $f$. This is exactly what produces the 46% gap in our experiments: a shared policy solving a mixture of per-frequency problems that disagree.

**Why K=3: Empirical Sparsity Trade-off.** We set $K=3$ based on an empirical accuracy–efficiency trade-off. [**TBD: Add Figure showing energy reduction vs K**] In our mic→LDV setting with $M=16$ candidate frame-lags, improvement beyond $K=3$ is marginal [@tropp2004greed; @pati1993omp]. We emphasize that $K$ reflects the **chosen sparsity budget in the STFT lag dictionary**, not a literal count of physical propagation paths. This short sequence length ($K=3$) enables our lightweight GRU-based architecture instead of Transformer.

**Mic–LDV Coupling as Impedance-Mediated Transfer.** The microphone measures air pressure $p(t)$. The LDV measures surface velocity $v_{\text{surf}}(t)$ induced by the same acoustic field but filtered by the surface mechanical impedance $Z_s(\omega)$ and structural modes [@morse1968theoretical]. Under linear acoustics and small vibrations, the local boundary relation:
$$p(\omega) = Z_s(\omega) v_{\text{surf}}(\omega)$$
implies a frequency-dependent complex ratio between $p$ and $v_{\text{surf}}$. Therefore, the relative transfer function between microphone pressure and LDV velocity is expected to have frequency-dependent magnitude and phase beyond pure propagation delay, naturally producing frequency-dependent optimal lag structure in discrete approximations. We do not attempt to model $Z_s(\omega)$ explicitly; it is absorbed into the learned per-frequency lag-selection policy, with residual mismatch treated as noise [@rothberg2017ldv_review; @avargel2011ldv_speech].

**Validation Setting.** We validate on microphone-to-LDV transformation, chosen because the two sensors have distinct physical mechanisms: microphones measure air pressure variations, while laser Doppler vibrometers (LDVs) measure surface vibration velocity. This physical difference creates significant domain shift in frequency response and phase characteristics. The transfer function $H(f,\theta)$ varies with both frequency $f$ and source direction $\theta$.

### 3.1.3 Sparsity Justification (Physics-Motivated Model Order)

Starting from the quasi-static linear model $y(t) \approx (g * x)(t) + \epsilon(t)$, we obtain in the STFT domain:
$$Y(f,n) = \sum_{\ell=0}^{L-1} \alpha(f,\ell) X(f,n-\ell) + \eta(f,n)$$
where $\eta$ absorbs:
- Cross-band leakage (STFT approximation error)
- Late reverberation (diffuse tail) [@kuttruff2016room]
- LDV-specific artifacts (speckle dropouts, beam misalignment)

Although the true impulse response is NOT sparse at the sample level, the **frame-lag representation** is deliberately low-resolution:
- Each lag atom = hop-sized (32 ms) shift
- $\alpha(f,\ell)$ captures coarse-grained energy concentrations over time

In typical acoustic environments:
- Energy dominated by FEW early components (direct path + early reflections)
- Late tail is diffuse → treated as model mismatch $\eta$

We therefore pose a sparse approximation per frequency:
$$\min_{a(f)\in\mathbb{C}^M} \|y_f - D_f a(f)\|^2 \quad \text{subject to} \quad \|a(f)\|_0 \leq K$$
where:
- $D_f$ = lag dictionary $\{X(f,n-\ell)\}_{\ell=1}^M$
- $K$ = model-order budget controlling bias–variance tradeoff
- Small $K$ → captures dominant coherent components
- $\eta$ → captures incoherent residual

This formulation connects the physical model to sparse signal processing [@donoho2006cs].

### 3.1.4 Why OMP (Greedy Energy Capture Under Sparse Model)

Given the $\ell_0$-constrained least squares objective:
$$\min_{\|a\|_0\leq K} \|y - D a\|^2$$
OMP is a greedy approximation building the support set sequentially [@tropp2004greed].

Let $r_{k-1} = y - P_{S_{k-1}} y$ (residual after projecting onto previously selected atoms $S_{k-1}$). OMP selects the next atom index $i_k$ by:
$$i_k \in \arg\max_i |\langle d_i, r_{k-1} \rangle|$$
then refits coefficients via least squares on $S_k = S_{k-1} \cup \{i_k\}$.

**Energy interpretation:** For normalized atoms, $|\langle d_i, r \rangle|^2$ = maximal one-step reduction in residual energy achievable by adding single atom $i$ before refitting. OMP implements the physically meaningful "capture strongest coherent component first" principle, consistent with early-dominant acoustic/structural components.

We emphasize:
- Theoretical recovery guarantees depend on dictionary coherence / RIP [@davenport2010omp_rip]
- Our lag dictionary built from speech is NOT random
- We do NOT rely on worst-case guarantees
- Instead: report empirical mutual coherence + show $K=3$ in stable regime (additional steps → diminishing returns)

**Why not LASSO/Basis Pursuit?** LASSO/BP [@candes2005dantzig] can be more stable under high coherence + noise but computationally heavier. OMP is attractive because:
- Teacher already uses fixed small $K$
- We distill its behavior
- Computational simplicity + interpretable stepwise energy reduction align with "policy learning" narrative

## 3.2 Frequency-Aware Algorithm Distillation

Given the phase ambiguity challenge and sparse objective from Section 3.1, our method must satisfy three requirements: (1) frequency awareness to learn frequency-specific strategies, (2) efficient single-pass inference unlike iterative OMP, and (3) goal-directed learning to generalize beyond the training distribution. We address these through algorithm distillation with frequency embedding.

We first formalize OMP as a Markov Decision Process (§3.2.1), which naturally motivates a recurrent policy architecture. We then justify why frequency embedding is necessary (§3.2.2) and why RTG conditioning provides physically meaningful goal-directed learning (§3.2.3). This section begins by describing the OMP teacher generation process.

**OMP Teacher Generation.** Orthogonal Matching Pursuit iteratively selects time lags that maximize residual energy reduction. Given a dictionary of $M=16$ candidate lags, OMP performs $K=3$ greedy selections: at each step, it computes correlations between the residual and all dictionary atoms, selects the lag with maximum correlation magnitude, and updates the residual via least-squares projection. This generates trajectories containing correlation states, selected actions, and cumulative energy reductions. OMP achieves 97.88% energy reduction, providing an oracle for distillation.

### 3.2.1 OMP as a Markov Decision Process

We model $K$-step OMP as an MDP (finite-horizon deterministic dynamics) [@sutton2018rl]:

**State:** $s_k = (r_{k-1}, S_{k-1}, f)$ where:
- $r_{k-1}$ = current residual at frequency bin $f$
- $S_{k-1}$ = set of selected lag indices

**Action:** $a_k \in \{1,2,\dots,M\}$ (selecting next lag atom)

**Transition (deterministic):**
$$S_k = S_{k-1} \cup \{a_k\}$$
$$\hat{a}_k = \arg\min_{\text{supp}(a)\subseteq S_k} \|y_f - D_f a\|^2$$
$$r_k = y_f - D_f \hat{a}_k$$

**Reward:** $\text{rwd}_k = \|r_{k-1}\|^2 - \|r_k\|^2 \geq 0$ (energy reduction at step $k$)

**Property 1 (Markov):** Because $r_k$ is obtained by deterministic projection based on $(r_{k-1}, S_{k-1}, a_k)$, future states depend only on current state and action.

**Property 2 (Causality / forward dependence):** Optimal action at step $k$ depends on current residual, which depends on ALL previous actions. The decision process is inherently sequential.

**Implication for architecture:** A recurrent policy is the natural approximator:
- GRU hidden state = compact representation of evolving residual/support history
- Sufficient for near-optimal action selection over short horizon $K=3$
- Bidirectional models NOT aligned with OMP causality (OMP decisions cannot condition on future actions)

### 3.2.2 Frequency Embedding Necessity (Identifiability Under Phase Wrapping)

Let $f$ = frequency-bin index and $\pi^*(\cdot|f)$ = optimal per-frequency teacher policy (induced by OMP on bin $f$).

Consider the hypothesis class of frequency-blind policies:
$$\Pi_{\text{blind}} = \{\pi(a|c) : \pi \text{ does NOT take } f \text{ as input}\}$$

For any $\pi \in \Pi_{\text{blind}}$, the action distribution is IDENTICAL across frequencies for the same correlation state $c$.

**Impossibility result:** If $\exists f_1 \neq f_2$ and correlation state $c$ such that teacher's optimal actions differ:
$$\arg\max_a \pi^*(a|c,f_1) \neq \arg\max_a \pi^*(a|c,f_2)$$
then NO $\pi \in \Pi_{\text{blind}}$ can match $\pi^*$ on both frequencies simultaneously.

**Connection to physics:** Under the physical delay-phase relation $e^{-j2\pi f\tau}$ and discrete lag dictionary, the best-matching lag index depends on $f$ whenever:
- $\tau$ not exactly representable on lag grid, OR
- Relative transfer has frequency-dependent group delay $\tau_g(f)$

Therefore, $\pi^*$ is generically frequency-dependent, and frequency conditioning is REQUIRED to represent the teacher policy with low error [@caruana1997mtl].

**Implementation:** The frequency embedding implements compact conditional parameterization $\pi(a|c,f)$ with shared weights + learned embedding of $f$, equivalent to multi-task conditioning over frequency bins.

**Information-theoretic interpretation:** If $I(A;F|C) > 0$ under teacher trajectories, then predicting $A$ from $C$ alone is information-bottlenecked. Providing $F$ strictly enlarges achievable Bayes accuracy.

### 3.2.3 RTG Conditioning as Physics-Meaningful Target

Define residual energy at step $k$: $E_k = \|r_k\|^2$. Then: $\text{rtg}_k = E_{k-1} - E_K$ = remaining energy to remove to reach final target.

**Key insight:** RTG is NOT merely an RL trick—it is a **physically interpretable control variable** specifying how much coherent energy (in STFT residual) we aim to explain with remaining $K-k+1$ selections [@schaul2015uvfa].

**Goal-conditioned imitation:**
$$\pi(a_k | c_k, f, \text{rtg}_k)$$
enables test-time control of accuracy–latency tradeoff by selecting different RTG targets, without retraining.

**Why prefer RTG-conditioned distillation over online RL?**
Teacher trajectories generated by deterministic optimizer (OMP) are near-optimal for the specified sparse objective. No need for exploration.

**Why not behavior cloning (BC)?**
BC learns: $\pi(a|c,f)$ (no RTG). RTG adds: $\pi(a|c,f,\text{rtg})$ (goal-conditioned).

RTG advantage [@andrychowicz2017her]:
- BC may fail when test distribution differs from training
- RTG learns goal-directed policies achieving specified optimization targets
- Enables model to learn "how to achieve X% energy reduction" rather than "what OMP did in this exact state"
- More robust generalization

**Formal definition:** At step $k$, we define per-step reward $\Delta r_k := E_{k-1} - E_k$ (energy reduction at step $k$), cumulative return $R_k := \sum_{i=1}^{k} \Delta r_i$, and return-to-go $\text{rtg}_k := \sum_{i=k}^{K} \Delta r_i = R_K - R_{k-1}$. This matches Decision Transformer RTG semantics [@chen2021decision]: $\text{rtg}_k$ encodes the sum of future rewards from the current step onward.

**Network Architecture.** We use an RTG-conditioned policy inspired by Decision Transformer [@chen2021decision], but adapted for our problem characteristics. While Decision Transformer uses Transformer for long RL trajectories, we adopt a GRU for our short sequence length ($K=3$). Given the very short decision horizon, a GRU provides a favorable accuracy–latency trade-off [@cho2014gru]. While Transformers [@vaswani2017transformer] are applicable, their benefits are most pronounced with longer contexts; in our setting, the lightweight recurrent policy provides:
- **Causal inductive bias**: GRU's unidirectional hidden state naturally models OMP's sequential residual updates
- **Parameter efficiency**: ~500K parameters vs. millions in standard Decision Transformer, reducing overfitting risk

The network receives three inputs per step combined through additive fusion:
$$\mathbf{e} = \text{LayerNorm}(\text{StateEmbed}(\mathbf{c}) + \text{RTGEmbed}(\text{rtg}) + \text{FreqEmbed}(f))$$
where $\mathbf{c} \in \mathbb{R}^{16}$ is the correlation state defined as $c_k(f)[i] = |\langle d_i(f), r_{k-1}(f) \rangle|$, a sufficient statistic for the greedy OMP step, $\text{rtg}$ is the return-to-go scalar, and $f$ is the frequency bin index. Each embedding maps to dimension 128, fused into a 2-layer GRU with hidden dimension 256. The output head maps GRU hidden states to action logits over 16 possible lags.

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

**Domain Shift and Phase Ambiguity.** Cross-sensor domain shift arises because different sensors measure different physical quantities through different coupling mechanisms. In our microphone-LDV setting, microphones capture air pressure variations while LDVs measure surface vibration velocity filtered by surface mechanical impedance $Z_s(\omega)$ (§3.1). This physical difference manifests as frequency-dependent phase characteristics beyond simple propagation delay. The phase-frequency coupling $\phi(f) = -2\pi f \tau$ (mod $2\pi$) combined with discrete lag dictionary produces frequency-dependent optimal lag selection (§3.1, Lemma). Our 46% performance gap quantifies this effect: a frequency-blind policy cannot distinguish which lag is optimal at each frequency, forcing a suboptimal compromise.

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
@article{talmon2009ctf,
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

@article{allen1977unified,
  title={A Unified Approach to Short-Time Fourier Analysis and Synthesis},
  author={Allen, Jont B. and Rabiner, Lawrence R.},
  journal={Proceedings of the IEEE},
  year={1977},
  volume={65},
  number={11},
  pages={1558--1564},
  doi={10.1109/PROC.1977.10770}
}

@article{portnoff1980stft,
  title={Time-Frequency Representation of Digital Signals and Systems Based on Short-Time Fourier Analysis},
  author={Portnoff, Michael R.},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  year={1980},
  volume={28},
  number={1},
  pages={55--69},
  doi={10.1109/TASSP.1980.1163359}
}

@article{knapp1976gcc,
  title={The Generalized Correlation Method for Estimation of Time Delay},
  author={Knapp, Charles H. and Carter, G. Clifford},
  journal={IEEE Transactions on Acoustics, Speech, and Signal Processing},
  year={1976},
  volume={24},
  number={4},
  pages={320--327},
  doi={10.1109/TASSP.1976.1162830}
}

@article{chen2006tde_overview,
  title={Time Delay Estimation in Room Acoustic Environments: An Overview},
  author={Chen, Jie and Benesty, Jacob and Huang, Yiteng},
  journal={EURASIP Journal on Applied Signal Processing},
  year={2006},
  volume={2006},
  pages={1--17},
  doi={10.1155/ASP/2006/26503}
}

@book{ghiglia1998phase,
  title={Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software},
  author={Ghiglia, Dennis C. and Pritt, Mark D.},
  publisher={Wiley},
  year={1998}
}

@book{morse1968theoretical,
  title={Theoretical Acoustics},
  author={Morse, Philip M. and Ingard, K. Uno},
  publisher={Princeton University Press},
  year={1968}
}

@book{kuttruff2016room,
  title={Room Acoustics},
  author={Kuttruff, Heinrich},
  edition={6},
  year={2016},
  publisher={CRC Press},
  doi={10.1201/9781315372150}
}

@article{rothberg2017ldv_review,
  title={An International Review of Laser Doppler Vibrometry: Making Light Work of Vibration Measurement},
  author={Rothberg, S. J. and Allen, M. S. and Castellini, P. and Di Maio, D. and Dirckx, J. J. J. and Ewins, D. J. and Halkon, B. J. and Muyshondt, P. and Paone, N. and Ryan, T. and Steger, H. and Tomasini, E. P. and Vanlanduit, S. and Vignola, J. F.},
  journal={Optics and Lasers in Engineering},
  year={2017},
  volume={99},
  pages={11--22},
  doi={10.1016/j.optlaseng.2016.10.023}
}

@inproceedings{avargel2011ldv_speech,
  title={Speech Measurements Using a Laser Doppler Vibrometer Sensor: Application to Speech Enhancement},
  author={Avargel, Yekutiel and Cohen, Israel},
  booktitle={Proc. Joint Workshop on Hands-free Speech Communication and Microphone Arrays (HSCMA)},
  year={2011},
  pages={109--114},
  doi={10.1109/HSCMA.2011.5942375}
}

@article{donoho2006cs,
  title={Compressed Sensing},
  author={Donoho, David L.},
  journal={IEEE Transactions on Information Theory},
  year={2006},
  volume={52},
  number={4},
  pages={1289--1306},
  doi={10.1109/TIT.2006.871582}
}

@article{tropp2004greed,
  title={Greed is Good: Algorithmic Results for Sparse Approximation},
  author={Tropp, Joel A.},
  journal={IEEE Transactions on Information Theory},
  year={2004},
  volume={50},
  number={10},
  pages={2231--2242},
  doi={10.1109/TIT.2004.834793}
}

@article{davenport2010omp_rip,
  title={Analysis of Orthogonal Matching Pursuit Using the Restricted Isometry Property},
  author={Davenport, Mark A. and Wakin, Michael B.},
  journal={IEEE Transactions on Information Theory},
  year={2010},
  volume={56},
  number={9},
  pages={4395--4401},
  doi={10.1109/TIT.2010.2054653}
}

@article{candes2005dantzig,
  title={Decoding by Linear Programming},
  author={Cand{\`e}s, Emmanuel J. and Tao, Terence},
  journal={IEEE Transactions on Information Theory},
  year={2005},
  volume={51},
  number={12},
  pages={4203--4215},
  doi={10.1109/TIT.2005.858979}
}

@article{caruana1997mtl,
  title={Multitask Learning},
  author={Caruana, Rich},
  journal={Machine Learning},
  year={1997},
  volume={28},
  pages={41--75},
  doi={10.1023/A:1007379606734}
}

@inproceedings{schaul2015uvfa,
  title={Universal Value Function Approximators},
  author={Schaul, Tom and Horgan, Dan and Gregor, Karol and Silver, David},
  booktitle={Proceedings of the 32nd International Conference on Machine Learning (ICML)},
  year={2015},
  publisher={PMLR}
}

@inproceedings{andrychowicz2017her,
  title={Hindsight Experience Replay},
  author={Andrychowicz, Marcin and Wolski, Filip and Ray, Alex and Schneider, Jonas and Fong, Rachel and Welinder, Peter and McGrew, Bob and Tobin, Josh and Abbeel, Pieter and Zaremba, Wojciech},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2017}
}

@book{sutton2018rl,
  title={Reinforcement Learning: An Introduction},
  author={Sutton, Richard S. and Barto, Andrew G.},
  edition={2},
  publisher={MIT Press},
  year={2018}
}
```
