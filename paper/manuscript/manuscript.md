# Working title (replace)

## Abstract
Direction-of-arrival (DOA) estimation is typically framed as an array-processing problem, requiring spatially separated sensors to measure phase and time-difference cues. Yet, in many practical settings, sensor arrays are constrained by size, placement, or harsh environments. Here we present a physics-first formulation of **single-point DOA sensing** in which a target structure acts as a *physical encoder*: incident sound couples into structure-borne vibrations through a direction-dependent superposition of dispersive modes, producing a characteristic single-point spectral signature measurable by a single non-contact laser Doppler vibrometer (LDV) spot. We translate this physical process into a mathematical physics model and show how discretization yields a transfer matrix whose singular-value structure reveals a limited effective number of dominant channels. This motivates a structured sparse inverse problem over a physical dictionary, naturally admitting greedy sparse pursuit (orthogonal matching pursuit, OMP) as a principled baseline. Building on this foundation, we derive a **physics-guided deep unrolling** network that replaces heuristic atom selection with learnable attention-based routing while retaining residual-consistency constraints. In speech conditions, the proposed model achieves top-1 accuracy of `[X]% / [Y]% / [Z]%` at SNR = `10 / 5 / 0 dB` (mean ± s.d. over `n=[X]` trials) and maintains cross-material RMSE below `[X]°` across five everyday targets spanning a broad spectrum of physical complexity. Critically, the results support a discovery-first conclusion: **everyday objects physically encode DOA at a single point through a dispersive mechanism that persists across diverse targets**, suggesting a path toward compact DOA sensing without array microphones.

## Introduction
Direction-of-arrival (DOA) estimation is classically solved by array processing, where spatially separated sensors capture phase and time-difference cues that can be inverted via beamforming or subspace methods [@krim1996array]. However, arrays are often impractical when the sensing footprint is constrained or when sensors must be embedded on surfaces exposed to harsh environments. Recent studies suggest that a single structural vibration measurement can nonetheless carry directional information because incident sound excites structure-borne waves whose response depends on incidence direction [@dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor].

In this work, we focus on a discovery question relevant to real-world sensing: **do everyday objects intrinsically encode acoustic direction in their material-dependent dynamics, such that a single non-contact vibration measurement can decode DOA?** We use a laser Doppler vibrometer (LDV) because it enables **non-contact** readout of the vibration field and therefore avoids altering the object’s mechanical boundary conditions and mass loading. This distinction is not cosmetic: contact sensors such as piezoelectric patches or accelerometers can perturb local stiffness and damping and can smear or shift dispersive signatures, potentially distorting the very physical encoding we seek to characterize.

A central open question is whether this apparent single-point “fingerprint” reflects a reproducible physical mechanism or merely target-specific idiosyncrasies. Here we treat the target structure as a *physical encoder* that transforms incident direction into a direction-dependent superposition of dispersive modes. This view suggests two falsifiable predictions: (i) the angle-to-spectrum mapping should be governed by a small number of dominant physical channels, and (ii) inference should remain stable under moderate noise and should transfer across structurally distinct targets if the underlying mechanism is universal.

We operationalize these predictions by translating structural dynamics into a linear angle–frequency response model, interrogating its effective degrees of freedom via singular value decomposition (SVD), and posing DOA inference as a structured sparse inverse problem. We use orthogonal matching pursuit (OMP) as an analytical probe and derive a physics-guided deep unrolling network that replaces heuristic atom selection with attention-based routing while retaining residual-consistency constraints. We then evaluate robustness under additive noise and architectural ablations, analyze learned routing statistics as mechanistic evidence, and test cross-material generality across targets spanning a broad spectrum of physical complexity (Figs. 1–6).

## Results

### Physical encoding by a complex structure (Fig. 1)
We probe the physical-encoder hypothesis using the experimental configuration shown in Fig. 1a, where an incident acoustic field excites a target structure and a laser Doppler vibrometer (LDV) measures single-point vibration. Although the resulting time-domain waveform can appear irregular due to multimodal dynamics and multiple scattering, the key observation (Fig. 1b) is that different incidence directions produce reproducible, direction-dependent **spectral fingerprints**. Quantitatively, fingerprint stability is reflected by high within-angle similarity `[X]±[Y]` and substantially lower between-angle similarity `[X]±[Y]` on the same feature representation (Methods; Evaluation and statistics). Starting from this observation, we hypothesize that the fingerprints arise from a dispersive, modal superposition mechanism that concentrates DOA information into a small set of dominant physical channels. We test this hypothesis by (i) deriving a minimal physical model and exposing dominant channels via SVD (Fig. 2), (ii) using sparse pursuit (OMP) and a physics-guided unrolled solver as inference probes (Fig. 3), (iii) stress-testing robustness and isolating causal components by ablation (Fig. 4), (iv) verifying mechanistic alignment by analyzing learned routing statistics (Fig. 5), and (v) falsifying target-specific explanations via cross-material evaluation (Fig. 6).

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.

### Mathematical physics, discretization, and the SVD view (Fig. 2)
This section makes the “direction-dependent spectral fingerprint” concept precise by converting the physical process into mathematical physics, discretizing it into a linear inverse model, and then using the SVD to expose the effective degrees of freedom that motivate sparse inference. Importantly, we treat the model and its linear algebra as **tools for discovery**: they allow us to ask *where* DOA information can live in a single-point measurement and *how many* dominant physical channels are effectively accessible.

Notation and dimensions are summarized in Methods (Notation and dimensions).

#### Physics model → single-point response
Under small-amplitude dynamics, a thin plate is well described by a linear operator equation (Kirchhoff–Love theory) [@timoshenko1959plates]. In the frequency domain, one representative form is

$$
\left(D_p\nabla^4 - \rho t\,\omega^2 + i\omega c_d\right) W(x,y,\omega)
\;=\;
P(x,y,\theta,\omega),
$$

where \(W(x,y,\omega)\) is the complex displacement field, \(P(x,y,\theta,\omega)\) is the effective forcing induced by an incident field from direction \(\theta\), \(D_p\) is the bending stiffness, \(\rho t\) is the areal mass density (density \(\rho\) times thickness \(t\)), and \(c_d\) is an effective damping term. A single-point sensor measures at \((x_L,y_L)\), and linearity implies a Green’s function representation

$$
W(x_L,y_L,\omega)
\;=\;
\iint_{\Omega}
G\!\left((x_L,y_L),(x',y'),\omega\right)\,
P(x',y',\theta,\omega)\,\mathrm{d}A'.
$$

**Interpretation.** These equations formalize the core physical idea behind Fig. 1b: DOA enters through the forcing pattern \(P(\cdot;\theta)\) and couples into dispersive structural dynamics via a linear response kernel \(G\). Even when the time-domain waveform appears irregular, the frequency-domain response can carry stable direction-dependent structure because it is governed by the same underlying linear operator during the measurement window.

We define the complex single-point response \(Y(\omega;\theta)=W(x_L,y_L,\omega)\).

Under linear dynamics, the Green’s function response can be interpreted as a superposition of a limited number of dispersive components whose weights depend on incident direction, motivating an approximate separable form

$$
Y(\omega;\theta) \;\approx\; \sum_{m=1}^{M} s_m(\omega)\,\alpha_m(\theta),
$$

where \(s_m(\omega)\) captures a frequency-dependent spectral signature and \(\alpha_m(\theta)\) captures angle-dependent coupling. Discretizing and stacking angles yields an approximately low-rank spectral–angular factorization of the empirical response matrix,

$$
H \;\approx\; \sum_{m=1}^{M} u_m v_m^\top,
$$

where \(u_m\in\mathbb{R}^F\) is a frequency signature and \(v_m\in\mathbb{R}^E\) is an angle signature, motivating the SVD interpretation in Fig. 2b.

On a discrete frequency grid \(\{\omega_f\}_{f=1}^F\), we form a real feature vector

$$
y[f] \;=\; \phi\!\left(\left|Y(\omega_f;\theta)\right|\right),
\qquad f=1,\dots,F,
$$

where \(\phi(\cdot)\) is a fixed transform (e.g., magnitude, power, or log-power) defined in Methods (Signal representation).

#### Discretization → angle response matrix
For each candidate direction \(\theta_e\), we form an angle-conditioned atom \(h_e\in\mathbb{R}^F\) and stack these atoms into the **angle response matrix**

$$
H \;=\; [h_1,\dots,h_E] \in \mathbb{R}^{F\times E}.
$$

Given an observation \(y\), we model it as a sparse combination of candidate angles:

$$
y \approx Hx + n,
\qquad \lVert x\rVert_0 \le K,
$$

where \(x\in\mathbb{R}^E\) is sparse over angles and \(n\) captures noise and mismatch. For a single source, \(x\) is ideally one-hot (\(K=1\)); multi-source settings correspond to \(K>1\).

**Interpretation (empirical encoding map).** The matrix \(H\) is the discrete, empirical representation of the physical encoder: it maps an angle-indexed sparse code \(x\) to a single-point spectral feature \(y\). The discovery question now becomes linear-algebraic and falsifiable: *does \(H\) concentrate its energy in a small number of dominant channels, or is the mapping effectively high-dimensional and fragile?*

#### SVD → dominant physical subspace
We use the SVD to expose effective degrees of freedom. Let

$$
H = U\Sigma V^\top,
$$

where \(\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_{\min(F,E)})\) with \(\sigma_1\ge\dots\ge 0\). A rapidly decaying singular spectrum indicates that only a limited number of dominant channels contribute strongly to the sensor response, closely related to eigenchannel analyses of transmission matrices in complex media [@davy2015eigenchannels].

Equivalently, the SVD can be written as an outer-product expansion,

$$
H = \sum_{m=1}^{\min(F,E)} \sigma_m\, u_m v_m^\top,
$$

where \(u_m\) and \(v_m\) are columns of \(U\) and \(V\), respectively. In our setting, this directly links the linear algebra to the physics interpretation above: \(u_m\) acts as a frequency-selective spectral signature, \(v_m\) acts as an angle-selective coupling/directivity pattern, and \(\sigma_m\) quantifies the strength of the corresponding effective channel (Fig. 2b).

**Interpretation (what SVD reveals and predicts).** In this work, the SVD is not merely a mathematical convenience: it provides a direct diagnostic of the physical encoder’s effective degrees of freedom. If the singular spectrum decays rapidly (as in Fig. 2a), then (i) DOA-relevant variability should live largely in a low-dimensional subspace, and (ii) inference should be feasible and relatively stable when restricted to that dominant subspace. This motivates the projected formulation below and sets up robustness tests (Fig. 4) and generalization tests (Fig. 6).

In our measurements, the leading `r=[X]` components capture `[Y]%` of the singular-value energy and define the dominant spectral channels used throughout the subsequent sparse inference and unrolled models.

To make the SVD link operational for inference, we project the inverse model into the rank-\(r\) left-singular subspace. Let \(U_r\in\mathbb{R}^{F\times r}\) contain the first \(r\) columns of \(U\) (chosen by an energy criterion described in Methods (Dictionary construction)). Define

$$
z = U_r^\top y \in \mathbb{R}^r,
\qquad
A = U_r^\top H \in \mathbb{R}^{r\times E},
$$

so the forward model becomes

$$
z \approx Ax + \tilde n.
$$

This projected formulation preserves the angle-indexed structure while focusing inference on the dominant spectral channels, enabling sparse pursuit and neural unrolling in the same channel coordinates (Fig. 3).

Figure 2 summarizes the mechanism: dominant channels (Fig. 2a), a spectral–directional interpretation (Fig. 2b), and a structured physical dictionary over angles (Fig. 2c). In the simplest instantiation of the derivation above, the dictionary is the angle response matrix \(H\) (one atom per candidate angle; denoted \(D\) in Fig. 2c); richer structured dictionaries can be built by expanding the atom index (e.g., mode–angle), while keeping the same sparse-inference backbone.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes (sparsity/low-rank structure).
b, Modal decomposition into frequency-selective spectra \(u_r(f)\) and direction-selective polar patterns \(v_r(\theta)\), forming virtual directional sensing channels.
c, Structured physical dictionary \(D\) assembled by combining spectral and directional components to produce distinct mode–angle atoms with characteristic dispersion signatures.

### From OMP to a physics-guided neural unrolling (Fig. 3)
We now turn the discovery question into an inference problem: given \(z\) and the projected dictionary \(A\), which small set of candidate angles best explains the observation? In this framing, sparse solvers act as *inference probes*—they attempt to invert the physical encoding and, through their successes and failures, reveal whether the hypothesized low-dimensional structure is sufficient under noise and mismatch.

We solve the projected sparse inverse problem \(z \approx Ax\) derived above:

$$
\min_{x}\; \lVert z - Ax\rVert_2^2
\quad \text{s.t.}\quad \lVert x\rVert_0 \le K,
$$

where \(A\in\mathbb{R}^{r\times E}\) is the SVD-projected dictionary, \(z\in\mathbb{R}^r\) is the projected observation, and \(x\in\mathbb{R}^E\) is a sparse coefficient vector over candidate angles. In the single-source case (\(K=1\)), a natural estimate is \(\hat{\theta}=\theta_{\arg\max_e x[e]}\).

In this basis, \(x\) selects a small set of angle-indexed atoms whose coupling patterns explain the observed dominant spectral-channel content. Unless otherwise stated, we use a pursuit budget of \(K=[X]\) stages/atoms for both analytical OMP and the unrolled network.

#### OMP baseline in the SVD-projected space
Orthogonal matching pursuit (OMP) is a canonical greedy solver for the \(\ell_0\)-constrained least-squares problem [@tropp2007omp]. Written in the projected space, OMP iterates:

1. Initialize residual \(r_0=z\), support \(S_0=\varnothing\), and \(x_0=0\).
2. For \(t=0,\dots,K-1\):
   - Correlate residual with atoms: \(g_t = A^\top r_t \in \mathbb{R}^E\).
   - Select an index \(i_t = \arg\max_e |g_t[e]|\) and update \(S_{t+1}=S_t\cup\{i_t\}\).
   - Refit coefficients by least squares on the selected support:
     $$
     x_{S_{t+1}} = \arg\min_{u}\; \lVert z - A_{S_{t+1}}u\rVert_2^2.
     $$
   - Update the residual \(r_{t+1} = z - A x_{t+1}\).

This formulation eliminates symbol discontinuities: the same projected dictionary \(A=U_r^\top H\) and projected observation \(z=U_r^\top y\) appear in both the SVD analysis (Fig. 2) and the sparse pursuit solver (Fig. 3).

#### Neural (unrolled) OMP with attention-based routing
OMP’s argmax selection is a fixed heuristic and can be brittle under noise and model mismatch in complex media. We therefore derive a neural solver by **unrolling** \(K\) pursuit stages into a network and replacing the discrete selection rule with learnable, attention-based routing, while retaining physics-consistent residual updates [@monga2021unrolling].

The unrolled network retains the same angle-indexed atoms and residual-consistent updates, but replaces OMP’s hard argmax selection with a learnable attention-based routing rule over atoms.

At stage \(t\), we start from residual \(r_t\in\mathbb{R}^r\) and correlations \(g_t=A^\top r_t\in\mathbb{R}^E\). We parameterize a routing distribution over atoms via dot-product attention [@vaswani2017attention]. Let a query be computed from the current state,

$$
q_t = W_q r_t \in \mathbb{R}^d,
$$

and let each atom \(a_e\in\mathbb{R}^r\) (column \(e\) of \(A\)) be embedded as a key \(k_e = W_k a_e \in \mathbb{R}^d\). The routing scores and weights are

$$
s_t[e] = \frac{\langle q_t, k_e\rangle}{\sqrt{d}},
\qquad
w_t = \mathrm{softmax}(s_t)\in\mathbb{R}^E.
$$

These weights gate a sparse update in coefficient space. One simple, physics-consistent choice matching the unrolled residual update in Fig. 3 is

$$
\Delta x_t = \eta_t\,(w_t \odot g_t),
\qquad
r_{t+1} = r_t - A\,\Delta x_t,
$$

where \(\eta_t\) is a step size (learned or fixed) and \(\odot\) denotes element-wise product. After \(K\) stages, we accumulate \(x=\sum_{t=0}^{K-1}\Delta x_t\) and map the resulting coefficient mass to a DOA estimate.

**Interpretation (attention as learned physical routing).** This construction makes the OMP→attention link explicit: when \(w_t\) concentrates to a one-hot vector at the maximally correlated atom, the update reduces to greedy selection; unrolling yields a differentiable, data-driven analogue of sparse pursuit. Because \(A=U_r^\top H\), the routing operates in the dominant spectral basis that approximates the encoder’s spectral–angular components \((u_m,v_m)\). Crucially, because atoms are indexed by candidate angles, the routing weights \(w_t\) become a mechanistic observable: they can be analyzed to test whether the learned solver aligns with the physical manifold (Fig. 5), and whether that alignment persists under target changes (Fig. 6).

![](../figures/fig03_unrolled-attention-omp.jpg)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating for sparse DOA inference.**
At stage \(t\), the residual \(r_t\) is correlated with the (projected) physics dictionary \(A\) to form a physical match \(g=A^\top r_t\); a transformer encoder outputs attention weights that gate sparse updates \(\Delta x\), followed by residual update \(r_{t+1}=r_t-A\Delta x\). Unrolling across stages accumulates a sparse vector \(x_T\), which is mapped to the final DOA estimate \(\hat{\theta}\).

### Robustness under additive noise and architectural ablations (Fig. 4)
We next treat robustness as a hypothesis test. If DOA information is concentrated in dominant physical channels and inference respects residual-consistent structure, then performance should degrade gracefully under additive noise; if not, performance should collapse and ablations should show no consistent mechanism. We therefore evaluate under additive white noise and isolate the impact of architectural components. Figure 4a reports top-1 validation accuracy at SNR levels of 10 dB, 5 dB, and 0 dB, comparing the full physics-aware model against a no-transformer variant and a fixed heuristic baseline; points denote independent trials and horizontal bars denote means across \(n=[X]\) trials (two-sided t-test, ***\(P<0.001\)). Figure 4b further ablates components, contrasting physics-aware sparse routing with dense routing and fixed heuristics.

Across SNR = `10 / 5 / 0 dB`, the full model achieves `[X] / [Y] / [Z]%` accuracy, compared to the no-transformer variant (`[X] / [Y] / [Z]%`) and a fixed heuristic baseline (`[X] / [Y] / [Z]%`). Replacing sparse, physics-aware routing with dense routing reduces accuracy by `[X]` points at `0 dB`, indicating that both channelization (Fig. 2) and structured sparse selection (Fig. 3) contribute to noise robustness.

Full dataset, split protocol, and statistical reporting are described in Methods (Experimental setup; Evaluation and statistics).

![](../figures/fig04_noise-robustness-ablation.jpg)

**Fig. 4 | Performance under additive noise and architectural ablations.**
a, Validation accuracy under additive white noise (SNR = 10, 5 and 0 dB) comparing the full physics-aware model, a no-transformer variant, and a fixed heuristic baseline; points denote independent trials and horizontal bars indicate means (two-sided t-test, ***P < 0.001).
b, Ablation of core components comparing the full model with no-transformer, dense routing, and fixed heuristic baselines.

### Interpreting learned routing: global structure, micro-mechanism, and macro-robustness (Fig. 5)
Nature Communications-style claims require mechanistic support beyond accuracy. Because our inference is explicitly defined over a physics-structured dictionary, the most meaningful internal signal to analyze is the model’s **routing/gating distribution over dictionary atoms** (rather than generic token self-attention). Figure 5 provides three complementary pieces of evidence. First, the global attention pattern exhibits a near-diagonal structure over the physical manifold (Fig. 5a), consistent with locality in the structured dictionary index. Second, a micro-level case study contrasts analytical OMP with physics-aware routing (Fig. 5b): OMP can select spurious atoms, whereas physics-aware routing yields a sparse selection aligned with the ground-truth DOA and a sharper angular estimate. Third, macro-level statistics aggregated across all angles show that physics-aware selection probability concentrates along the true DOA diagonal (Fig. 5c), indicating globally consistent physical alignment and reduced off-manifold errors.

The physical manifold index and off-diagonal mass metrics are defined in Methods (Evaluation and statistics).

Quantitatively, the learned routing reduces off-diagonal selection mass from `[X]` (analytical OMP) to `[Y]` (physics-aware AI) and increases diagonal concentration/sharpness by a factor of `[X]` across all angles, consistent with the interpretation that the network learns a physically aligned selection policy rather than exploiting spurious correlations.

![](../figures/fig05_routing-mechanism-analysis.png)

**Fig. 5 | Deciphering model behaviour across scales: attention structure, micro-mechanism and macro-robustness.**
a, Global self-attention map exhibiting a physics-consistent near-diagonal correlation structure across the physical manifold.
b, Micro-level case study (\(\theta_{\mathrm{true}}=60^\circ\)) comparing analytical OMP and physics-aware selection against ground truth, and the resulting angular estimate.
c, Selection-probability statistics across all angles showing off-diagonal errors for traditional OMP and a sharp diagonal alignment for physics-aware AI, indicating globally consistent physical selection.

### Cross-material generality and robust performance under complexity (Fig. 6)
We now test the central scientific claim: **is the single-point DOA encoding target-specific, or does it reflect a universal physical mechanism?** A physics-first hypothesis predicts that dispersion-driven signatures should persist across targets, while purely target-specific explanations would not transfer. Figure 6 evaluates this hypothesis across targets spanning a broad spectrum of material and geometric complexity: an acrylic plate, a paper cup, a wooden board, a cardboard box, and a laptop shell (Fig. 6a). Despite differing damping and internal structure, representative heatmaps exhibit shared dispersion-signature structure (Fig. 6b). Quantitatively, physics-aware inference maintains low DOA estimation error across materials, while analytical OMP degrades substantially as complexity increases (Fig. 6c). Together, these results support the interpretation that the encoding arises from a common dispersive mechanism rather than idiosyncratic features of a single target.

Per-material sample sizes and the RMSE computation protocol are reported in Methods (Evaluation and statistics).

Across acrylic / paper / wood / cardboard / laptop targets, the physics-aware model achieves RMSE of `[X]/[Y]/[Z]/[X]/[Y]°` (n=`[X]` per material), whereas analytical OMP increases to `[X]/[Y]/[Z]/[X]/[Y]°`, consistent with failure under increasing physical complexity and stable performance when routing remains physics-aligned.

![](../figures/fig06_cross-material-universality.jpg)

**Fig. 6 | Universal physical encoding across diverse materials and robust cross-material performance.**
a, Targets spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box and a laptop shell).
b, Representative dictionary/response heatmaps for each material, highlighting shared dispersion-signature structure despite differing physical properties.
c, DOA estimation error (RMSE) across materials comparing analytical OMP and physics-aware AI, showing degradation of OMP under increasing complexity and stable low error for the physics-aware model.

## Discussion
Our results provide mechanistic evidence that a single vibration measurement on a target structure can encode DOA through reproducible direction-dependent spectral fingerprints (Fig. 1). This reframes DOA estimation from “array geometry” to “physics of structural dispersion”: the structure itself can serve as an encoding medium whose response varies systematically with incident direction.

Interrogating the discretized encoder via the SVD (Fig. 2) exposes a limited set of dominant channels that capture most response variability. This provides a concrete, physically interpretable pathway from mechanism to inference: project to the dominant subspace (\(z=U_r^\top y\)) and solve a sparse inverse problem over projected atoms (\(z\approx Ax\)), using greedy pursuit (OMP) and its unrolled, learnable analogue as complementary probes (Fig. 3). The robustness and ablation results (Fig. 4) are consistent with the prediction that exploiting dominant-channel structure and enforcing residual consistency improves stability under additive noise and model mismatch.

The cross-material study (Fig. 6) supports the central discovery claim: the encoding mechanism persists across targets spanning diverse material properties and structural complexity. The stability of physics-aware inference, together with the degradation of heuristic sparse selection under complexity, is consistent with a universal dispersive encoding mechanism that can be probed through structured sparse solvers and validated through controlled stress tests.

Beyond performance, the unrolled attention routing provides a window into mechanism (Fig. 5): routing weights \(w_t\) form an interpretable, angle-indexed statistic that allows us to test whether the learned solver aligns with the physical manifold rather than exploiting spurious cues. This “model-as-probe” view is central to the discovery-first framing: mathematical physics and learning are used to formulate falsifiable hypotheses and to validate (or refute) them under noise, ablation, and target shifts.

### Limitations and falsification
The framework assumes a discrete set of candidate angles and a time window over which the structure can be treated as approximately linear and time-invariant; large-amplitude nonlinearities or changing boundary conditions would require explicit modeling or recalibration. While the SVD reveals effective degrees of freedom, it does not by itself establish uniqueness or information-theoretic limits of DOA identifiability for arbitrary targets. Finally, universality is supported by a representative set of targets, but could be falsified by targets whose responses do not exhibit a dominant low-dimensional subspace on the measurement grid, or by systematic failure under standardized complexity metrics and larger-scale benchmarking.

## Methods
### Experimental setup
Experiments were conducted in `[X]` (e.g., a quiet room / anechoic chamber) with a single loudspeaker source positioned at a fixed radius of `[X] m` from the target. The source direction was discretized to an angle grid \(\{\theta_e\}_{e=1}^E\) with step size of `[X]°` (total \(E=[X]\) directions), referenced to a fixed object coordinate frame (Methods; Evaluation and statistics).

We measured out-of-plane structural vibration using a non-contact laser Doppler vibrometer (LDV; `[manufacturer/model]`), targeting a single fixed point \((x_L,y_L)\) on each object. Non-contact LDV readout avoids sensor loading that can perturb the encoder itself: attaching a contact sensor can change local mass and damping and modify boundary conditions, potentially shifting or smearing the dispersive signatures used for encoding. To mitigate speckle and surface reflectivity issues, we used `[reflective tape / surface preparation]` and verified stable LDV signal quality before each run.

For each angle, we played `[stimulus type: white noise / speech / chirp]` for `[X] s` at a calibrated level of `[X] dB SPL` (measured at `[X]`), and recorded the LDV velocity signal at sampling rate \(f_s=\)[X] Hz with `[X]`-bit resolution. Each condition was repeated for `[X]` trials per angle (randomizing playback seeds where applicable). Targets spanned five everyday objects/material configurations (acrylic plate, paper cup, wooden board, cardboard box, laptop shell), mounted with `[mounting condition: free / clamped / foam support]` to ensure repeatable boundary conditions across trials.

### Signal representation
Let \(v(t)\) denote the LDV-measured velocity at \((x_L,y_L)\). We compute a short-time Fourier transform (STFT) with a `[Hann]` window of length `[X]` samples (`[X] ms`), hop `[X]` samples, and FFT size `[X]`. For each trial and angle, we extract a single-point spectrum \(Y(\omega_f;\theta)\) by aggregating STFT magnitudes over time frames (e.g., mean power across frames). The real-valued feature vector \(y\in\mathbb{R}^{F}\) is then defined as

$$
y[f] = \phi\!\left(\left|Y(\omega_f;\theta)\right|\right),
\qquad f=1,\dots,F,
$$

with \(\phi(\cdot)\) chosen as `[magnitude / power / log-power]` and normalized by `[per-frequency z-score / global scaling / none]`. Unless otherwise stated, we restrict the feature band to `[X]–[Y] Hz` and set \(F=[X]\) frequency bins after masking.

### Notation and dimensions
We define a fixed set of candidate angles \(\{\theta_e\}_{e=1}^E\) and a fixed frequency grid \(\{\omega_f\}_{f=1}^F\). The underlying response \(Y(\omega;\theta)\) is complex-valued, but we build real-valued magnitude/power features for inference.

| Symbol | Meaning | Shape |
|---|---|---|
| \(W(x,y,\omega)\) | Complex displacement field (frequency domain) | — |
| \(Y(\omega;\theta)\) | Complex single-point response at the LDV location | — |
| \(y\) | Single-point feature vector (magnitude/power over \(F\) bins) | \(\mathbb{R}^{F}\) |
| \(h_e\) | Angle-conditioned feature atom for \(\theta_e\) | \(\mathbb{R}^{F}\) |
| \(H=[h_1,\dots,h_E]\) | Angle response matrix (physics dictionary over angles) | \(\mathbb{R}^{F\times E}\) |
| \(x\) | Sparse coefficient vector over candidate angles | \(\mathbb{R}^{E}\) |
| \(K\) | Sparsity budget / number of pursuit stages | — |
| \(U_r\) | Top-\(r\) left singular vectors of \(H\) | \(\mathbb{R}^{F\times r}\) |
| \(z=U_r^\top y\) | Projected observation | \(\mathbb{R}^{r}\) |
| \(A=U_r^\top H\) | Projected dictionary | \(\mathbb{R}^{r\times E}\) |

### Dictionary construction
We construct an angle-indexed response matrix \(H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E}\), where each column \(h_e\) is the feature vector extracted from the single-point response at candidate angle \(\theta_e\). This matrix serves as the core physics dictionary over angles (denoted \(D\) in Fig. 2c). For inference, we optionally project to the dominant SVD subspace: compute \(H=U\Sigma V^\top\), choose rank \(r\), and form \(A=U_r^\top H\) and \(z=U_r^\top y\).

For each \(\theta_e\), we compute \(h_e\) as the `[mean/median]` feature across `[X]` trials/clips at that angle, with the same preprocessing used at inference time. Rank \(r\) is selected by retaining the smallest \(r\) that captures a fixed fraction of the singular-value energy, set to `[X]%` unless otherwise stated:

$$
\frac{\sum_{i=1}^r\sigma_i^2}{\sum_{i}\sigma_i^2} \ge [X]\%.
$$

We use one atom per angle (one column per \(\theta_e\)). Extensions that expand the atom index (e.g., mode–angle dictionaries) are left for future work.

### Sparse inference baseline (OMP)
OMP greedily approximates the sparse inverse problem by iteratively selecting atoms with large correlation to the residual and refitting coefficients over the selected support [@tropp2007omp]. In our formulation, OMP is applied to the projected model \(z\approx Ax\) (or directly to \(y\approx Hx\) if no projection is used), with correlations \(g_t = A^\top r_t\) (or \(H^\top r_t\)).

Unless otherwise stated, we run OMP on the projected pair \((A,z)\). At each stage, we select the atom index maximizing absolute correlation, \(i_t=\arg\max_e |g_t[e]|\), refit coefficients by least squares on the selected support using `[QR / pseudoinverse]`, and stop after \(K=[X]\) selections. The final DOA estimate is \(\hat{\theta}=\theta_{\arg\max_e x[e]}\) for the single-source setting.

### Physics-guided unrolled network
We unroll \(K\) pursuit stages into a feed-forward network, keeping the residual-consistency update and the fixed dictionary (\(A\) in the projected space), while learning a data-driven routing rule over atoms [@monga2021unrolling]. A transformer module parameterizes attention-like routing conditioned on the current residual state, producing gating weights \(w_t\) over atoms that modulate updates \(\Delta x_t\) (e.g., \(\Delta x_t=\eta_t (w_t\odot g_t)\)) [@vaswani2017attention].

We use \(K=[X]\) unrolled stages, embedding dimension \(d=[X]\), `[X]` transformer layers, and `[X]` attention heads. Training minimizes `[cross-entropy over angles / MSE over DOA / hybrid loss]` with optimizer `[Adam/AdamW]`, learning rate `[X]`, batch size `[X]`, and `[X]` epochs. All runs report random seeds (`[X]`) and use early stopping on validation accuracy with patience `[X]`.

### Evaluation and statistics
We report top-1 accuracy for classification-style DOA estimation and angular RMSE (degrees) where appropriate. Unless stated otherwise, results aggregate \(n\) independent trials with different random seeds; point estimates and summary statistics are reported as in Figs. 4–6.

**Splits and trials.** We use a `[train/validation/test]` split of `[X]/[Y]/[Z]%` at the clip level, stratified by angle. For cross-material evaluation, we train on `[X]` materials and test on `[X]` held-out materials (or train/test within each material as specified per experiment). Each reported point aggregates `n=[X]` independent trials (`[seed / re-sampling / repeated recordings]`).

**Additive noise.** For robustness experiments, we add zero-mean white noise in the time domain to achieve target SNR in dB, defined as

$$
\mathrm{SNR} = 10\log_{10}\frac{\mathbb{E}[v(t)^2]}{\mathbb{E}[\epsilon(t)^2]}.
$$

We report results at SNR = `10 / 5 / 0 dB`.

**Metrics.** Top-1 accuracy is computed on the discrete angle grid. Angular RMSE is computed in degrees with wrap-around handling:

$$
\Delta(\theta,\hat\theta)=\mathrm{wrap}_{[-180^\circ,180^\circ)}(\hat\theta-\theta),
\qquad
\mathrm{RMSE} = \sqrt{\mathbb{E}[\Delta(\theta,\hat\theta)^2]}.
$$

**Fingerprint similarity (Fig. 1).** Within-angle and between-angle similarity are computed as `[cosine similarity / Pearson correlation]` between normalized feature vectors \(y\) (or projected vectors \(z\)), aggregated across trials.

**Routing interpretability (Fig. 5).** We define a physical manifold index by ordering atoms by angle (and, if applicable, by mode–angle index). Off-diagonal mass is computed as the fraction of routing probability outside a `[±X]`-index band around the ground-truth diagonal in the selection-probability map. Diagonal sharpness is summarized by `[peak-to-mean ratio / entropy]` of \(w_t\) aggregated across stages and samples.

**Statistics.** Statistical significance is assessed using two-sided t-tests across independent trials; multiple-comparison corrections are applied where appropriate (`[method]`).

## Data availability
All data required to reproduce the figures will be made available at `[repository / DOI / link]` upon publication. Access to raw recordings is `[open / restricted]` due to `[X]`.

## Code availability
Code to reproduce the experiments and figure generation will be made available at `[GitHub link]` upon publication, including trained model checkpoints and scripts to rebuild the physical dictionaries.

## Acknowledgements
This work was supported by `[X]`.

## Author contributions
`[X]` conceived the study. `[X]` performed experiments. `[X]` developed models and analysis. All authors wrote and revised the manuscript.

## Competing interests
The authors declare no competing interests.

# References

::: {#refs}
:::
