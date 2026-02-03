# Working title (replace)

## Abstract
Direction-of-arrival (DOA) estimation is typically framed as an array-processing problem, requiring spatially separated sensors to measure phase and time-difference cues. Yet, in many practical settings, sensor arrays are constrained by size, placement, or harsh environments. Here we present a physics-first formulation of **single-point DOA sensing** in which a target structure acts as a *physical encoder*: incident sound couples into structure-borne vibrations through a direction-dependent superposition of dispersive modes, producing a characteristic single-point spectral signature. We translate this physical process into a mathematical physics model and show how discretization yields a transfer matrix whose singular-value structure reveals a limited effective number of dominant channels. This motivates a structured sparse inverse problem over a physical dictionary, naturally admitting greedy sparse pursuit (orthogonal matching pursuit, OMP) as a principled baseline. Building on this foundation, we derive a **physics-guided deep unrolling** network that replaces heuristic atom selection with learnable attention-based routing while retaining residual-consistency constraints. Across additive noise and architectural ablations, the resulting physics-aware model improves robustness, exhibits physically consistent selection behaviour, and generalizes across targets spanning a broad spectrum of material complexity.

## Fig. 1 — Physical encoding by a complex structure
Conventional DOA estimation relies on the spatial sampling of an acoustic wavefront by a microphone array, followed by parametric or subspace processing to infer direction [@krim1996array]. Recent work has shown that a **single structural vibration sensor** placed on an elastic panel can encode DOA information through structure-borne dynamics, enabling DOA estimation without a conventional microphone array at the sensing front-end [@dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor]. In this work, we adopt and generalize this perspective: a target structure is treated as a *physical encoder* that transforms incident direction into a direction-dependent vibration spectrum measurable at a single point.

Figure 1a shows the experimental configuration, where an incident acoustic field excites a target structure and a laser Doppler vibrometer (LDV) measures single-point vibration. Although the resulting time-domain waveform can appear irregular due to multimodal dynamics and multiple scattering, the key observation (Fig. 1b) is that different incidence directions produce reproducible, direction-dependent **spectral fingerprints**. The remainder of the manuscript makes this intuition precise by translating the physical process into mathematical physics, then into a sparse inverse problem, and finally into a neural (unrolled) inference architecture.

![](paper/figures/Figure-1.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.

## Fig. 2 — Mathematical physics, discretization, and the SVD view
This section makes the “direction-dependent spectral fingerprint” concept precise by converting the physical process into mathematical physics, discretizing it into a linear inverse model, and then using the SVD to expose the effective degrees of freedom that motivate sparse inference.

**Notation and dimensions.** We define a fixed set of candidate angles \(\{\theta_e\}_{e=1}^E\) and a fixed frequency grid \(\{\omega_f\}_{f=1}^F\). The underlying response \(Y(\omega;\theta)\) is complex-valued, but we build real-valued magnitude/power features for inference.

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

### Physics model → single-point response
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

We define the complex single-point response \(Y(\omega;\theta)=W(x_L,y_L,\omega)\). On a discrete frequency grid \(\{\omega_f\}_{f=1}^F\), we form a real feature vector

$$
y[f] \;=\; \phi\!\left(\left|Y(\omega_f;\theta)\right|\right),
\qquad f=1,\dots,F,
$$

where \(\phi(\cdot)\) is a fixed transform (e.g., magnitude, power, or log-power). **TBD:** specify the exact feature transform and normalization used in experiments.

### Discretization → angle response matrix
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

### SVD → dominant physical subspace
We use the SVD to expose effective degrees of freedom. Let

$$
H = U\Sigma V^\top,
$$

where \(\Sigma=\mathrm{diag}(\sigma_1,\dots,\sigma_{\min(F,E)})\) with \(\sigma_1\ge\dots\ge 0\). A rapidly decaying singular spectrum indicates that only a limited number of dominant channels contribute strongly to the sensor response, closely related to eigenchannel analyses of transmission matrices in complex media [@davy2015eigenchannels].

To make the SVD link operational for inference, we project the inverse model into the rank-\(r\) left-singular subspace. Let \(U_r\in\mathbb{R}^{F\times r}\) contain the first \(r\) columns of \(U\) (chosen by an energy criterion; **TBD:** specify the threshold). Define

$$
z = U_r^\top y \in \mathbb{R}^r,
\qquad
A = U_r^\top H \in \mathbb{R}^{r\times E},
$$

so the forward model becomes

$$
z \approx Ax + \tilde n.
$$

This projected formulation preserves the angle-indexed structure while focusing inference on the dominant physical subspace. It is the mathematical bridge from the SVD interpretation (Fig. 2a–b) to sparse pursuit and neural unrolling (Fig. 3).

Figure 2 summarizes the mechanism: dominant channels (Fig. 2a), a spectral–directional interpretation (Fig. 2b), and a structured physical dictionary over angles (Fig. 2c). In the simplest instantiation of the derivation above, the dictionary is the angle response matrix \(H\) (one atom per candidate angle; denoted \(D\) in Fig. 2c); richer structured dictionaries can be built by expanding the atom index (e.g., mode–angle), while keeping the same sparse-inference backbone.

![](paper/figures/Figure-2.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes (sparsity/low-rank structure).
b, Modal decomposition into frequency-selective spectra \(u_r(f)\) and direction-selective polar patterns \(v_r(\theta)\), forming virtual directional sensing channels.
c, Structured physical dictionary \(D\) assembled by combining spectral and directional components to produce distinct mode–angle atoms with characteristic dispersion signatures.

## Fig. 3 — From OMP to a physics-guided neural unrolling
We now solve the projected sparse inverse problem \(z \approx Ax\) derived above:

$$
\min_{x}\; \lVert z - Ax\rVert_2^2
\quad \text{s.t.}\quad \lVert x\rVert_0 \le K,
$$

where \(A\in\mathbb{R}^{r\times E}\) is the SVD-projected dictionary, \(z\in\mathbb{R}^r\) is the projected observation, and \(x\in\mathbb{R}^E\) is a sparse coefficient vector over candidate angles. In the single-source case (\(K=1\)), a natural estimate is \(\hat{\theta}=\theta_{\arg\max_e x[e]}\).

### OMP baseline in the SVD-projected space
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

### Neural (unrolled) OMP with attention-based routing
OMP’s argmax selection is a fixed heuristic and can be brittle under noise and model mismatch in complex media. We therefore derive a neural solver by **unrolling** \(K\) pursuit stages into a network and replacing the discrete selection rule with learnable, attention-based routing, while retaining physics-consistent residual updates [@monga2021unrolling].

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

This construction makes the OMP→attention link explicit: when \(w_t\) concentrates to a one-hot vector at the maximally correlated atom, the update reduces to greedy selection; unrolling yields a differentiable, data-driven analogue of sparse pursuit whose routing statistics are interpretable over the angle index (Fig. 5).

![](paper/figures/Figure-3.jpg)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating for sparse DOA inference.**
At stage \(t\), the residual \(r_t\) is correlated with the (projected) physics dictionary \(A\) to form a physical match \(g=A^\top r_t\); a transformer encoder outputs attention weights that gate sparse updates \(\Delta x\), followed by residual update \(r_{t+1}=r_t-A\Delta x\). Unrolling across stages accumulates a sparse vector \(x_T\), which is mapped to the final DOA estimate \(\hat{\theta}\).

## Fig. 4 — Robustness under additive noise and architectural ablations
We evaluate the resulting physics-aware model under additive white noise and isolate the impact of architectural components. Figure 4a reports top-1 validation accuracy at SNR levels of 10 dB, 5 dB, and 0 dB, comparing the full physics-aware model against a no-transformer variant and a fixed heuristic baseline; points denote independent trials and horizontal bars denote means across \(n=5\) trials (two-sided t-test, ***\(P<0.001\)). Figure 4b further ablates components, contrasting physics-aware sparse routing with dense routing and fixed heuristics.

TBD: Provide the exact dataset description (angle grid, split protocol, number of clips per angle) and the numeric table underlying Fig. 4 (mean ± s.d., n, and test specification).

![](paper/figures/Figure-4.jpg)

**Fig. 4 | Performance under additive noise and architectural ablations.**
a, Validation accuracy under additive white noise (SNR = 10, 5 and 0 dB) comparing the full physics-aware model, a no-transformer variant, and a fixed heuristic baseline; points denote independent trials and horizontal bars indicate means (two-sided t-test, ***P < 0.001).
b, Ablation of core components comparing the full model with no-transformer, dense routing, and fixed heuristic baselines.

## Fig. 5 — Interpreting learned routing: global structure, micro-mechanism, and macro-robustness
Because our inference is explicitly defined over a physics-structured dictionary, the most meaningful internal signal to analyze is the model’s **routing/gating distribution over dictionary atoms** (rather than generic token self-attention). Figure 5 provides three complementary views. First, the global attention pattern exhibits a near-diagonal structure over the physical manifold (Fig. 5a), consistent with locality in the structured dictionary index. Second, a micro-level case study contrasts analytical OMP with physics-aware routing (Fig. 5b): OMP can select spurious atoms, whereas physics-aware routing yields a sparse selection aligned with the ground-truth DOA and a sharper angular estimate. Third, macro-level statistics aggregated across all angles show that physics-aware selection probability concentrates along the true DOA diagonal (Fig. 5c), indicating globally consistent physical alignment and reduced off-manifold errors.

TBD: Define the “physical manifold index” used for Fig. 5a (index ordering, normalization, and aggregation of routing scores), and report a quantitative summary of off-diagonal mass for Fig. 5c.

![](paper/figures/Figure-5.png)

**Fig. 5 | Deciphering model behaviour across scales: attention structure, micro-mechanism and macro-robustness.**
a, Global self-attention map exhibiting a physics-consistent near-diagonal correlation structure across the physical manifold.
b, Micro-level case study (\(\theta_{\mathrm{true}}=60^\circ\)) comparing analytical OMP and physics-aware selection against ground truth, and the resulting angular estimate.
c, Selection-probability statistics across all angles showing off-diagonal errors for traditional OMP and a sharp diagonal alignment for physics-aware AI, indicating globally consistent physical selection.

## Fig. 6 — Cross-material generality and robust performance under complexity
A central hypothesis of the physics-first formulation is that the encoding mechanism is not tied to a single target, but is a generic consequence of dispersive structural dynamics. Figure 6 evaluates this hypothesis across targets spanning a broad spectrum of material and geometric complexity: an acrylic plate, a paper cup, a wooden board, a cardboard box, and a laptop shell (Fig. 6a). Despite differing damping and internal structure, representative heatmaps exhibit shared dispersion-signature structure (Fig. 6b). Quantitatively, physics-aware inference maintains low DOA estimation error across materials, while analytical OMP degrades substantially as complexity increases (Fig. 6c).

TBD: Provide the per-material sample sizes, the exact RMSE computation protocol (degrees, wrap-around handling), and the numeric table underlying Fig. 6c.

![](paper/figures/Figure-6.jpg)

**Fig. 6 | Universal physical encoding across diverse materials and robust cross-material performance.**
a, Targets spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box and a laptop shell).
b, Representative dictionary/response heatmaps for each material, highlighting shared dispersion-signature structure despite differing physical properties.
c, DOA estimation error (RMSE) across materials comparing analytical OMP and physics-aware AI, showing degradation of OMP under increasing complexity and stable low error for the physics-aware model.

## Discussion
This manuscript develops a derivation chain from physics to inference: (i) incident direction modulates coupling into structure-borne dispersive dynamics, producing direction-dependent single-point spectral signatures (Fig. 1); (ii) a mathematical physics model and discretization yield a transfer matrix whose singular structure indicates limited effective degrees of freedom (Fig. 2); (iii) this motivates a structured sparse inverse formulation solvable by OMP, and a neural (unrolled) extension that learns routing while preserving residual consistency (Fig. 3); and (iv) the resulting physics-aware model is robust under noise, interpretable through dictionary-index routing behaviour, and generalizes across materials (Fig. 4–6).

Several limitations remain. First, the formulation currently assumes a discrete set of candidate angles and a time window over which the structure can be treated as approximately linear and time-invariant; deviations (e.g., large-amplitude nonlinearities, changing boundary conditions) would require explicit modeling or recalibration. Second, while SVD provides a useful lens on effective degrees of freedom, it does not by itself establish uniqueness or information-theoretic limits of DOA identifiability for arbitrary targets; establishing such limits would require additional targeted analyses and experiments. Finally, cross-material generality is demonstrated over a representative set of targets, but broader generalization would benefit from standardized complexity metrics and larger-scale benchmarking.

## Methods
### Experimental setup
TBD: Describe the acoustic source, target geometry, LDV measurement point(s), mounting/boundary conditions, sampling rate, and calibration procedure.

### Signal representation
TBD: Define the time–frequency representation used for the single-point response \(Y(\omega;\theta)\) and the real-valued feature vector \(y\) (e.g., STFT parameters, window/hop, frequency band selection, and whether \(y\) uses magnitude, power, or log-power with normalization).

### Dictionary construction
We construct an angle-indexed response matrix \(H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E}\), where each column \(h_e\) is the feature vector extracted from the single-point response at candidate angle \(\theta_e\). This matrix serves as the core physics dictionary over angles (denoted \(D\) in Fig. 2c). For inference, we optionally project to the dominant SVD subspace: compute \(H=U\Sigma V^\top\), choose rank \(r\), and form \(A=U_r^\top H\) and \(z=U_r^\top y\).

TBD: Provide the exact construction recipe for \(H\) used in the experiments (angle grid, normalization, and any extensions beyond one atom per angle).

### Sparse inference baseline (OMP)
OMP greedily approximates the sparse inverse problem by iteratively selecting atoms with large correlation to the residual and refitting coefficients over the selected support [@tropp2007omp]. In our formulation, OMP is applied to the projected model \(z\approx Ax\) (or directly to \(y\approx Hx\) if no projection is used), with correlations \(g_t = A^\top r_t\) (or \(H^\top r_t\)).

TBD: Specify whether OMP runs on \((A,z)\) or \((H,y)\), the selection score (e.g., absolute correlation), the least-squares refit method, and any stopping criteria beyond fixed \(K\).

### Physics-guided unrolled network
We unroll \(K\) pursuit stages into a feed-forward network, keeping the residual-consistency update and the fixed dictionary (\(A\) in the projected space), while learning a data-driven routing rule over atoms [@monga2021unrolling]. A transformer module parameterizes attention-like routing conditioned on the current residual state, producing gating weights \(w_t\) over atoms that modulate updates \(\Delta x_t\) (e.g., \(\Delta x_t=\eta_t (w_t\odot g_t)\)) [@vaswani2017attention].

TBD: Specify the number of stages, the routing parameterization (query/key definitions and any hard top-\(K\) mechanism), the loss terms, and the training protocol (optimizer, learning rate, epochs, seeds).

### Evaluation and statistics
TBD: Define accuracy and RMSE metrics, trial definition, and statistical testing (including multiple-comparison handling if applicable).

## Data and code availability
TBD: Provide access conditions/links for code and datasets, or a statement describing restrictions if applicable.

## Acknowledgements
TBD.

## Author contributions
TBD.

## Competing interests
The authors declare no competing interests.

# References

::: {#refs}
:::
