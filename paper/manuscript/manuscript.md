# Non-contact acoustic sensing via the natural physical encoding of everyday objects

## Abstract
Conventional acoustic sensing relies on spatial sampling using microphone arrays, a paradigm that fundamentally limits miniaturization and deployment in harsh or constrained environments. Here we present a physics-first formulation of **single-point Direction-of-Arrival (DOA) sensing** in which the target structure itself acts as a *physical encoder*. We demonstrate that incident sound couples into structure-borne vibrations through a direction-dependent superposition of dispersive modes, producing a characteristic single-point spectral signature measurable by a non-contact laser Doppler vibrometer (LDV). By translating this physical process into a mathematical model, we reveal that the singular-value structure of the response matrix admits a limited effective number of dominant channels. This motivates a structured sparse inverse problem over a physical dictionary, solvable via a physics-guided deep unrolling network that replaces heuristic atom selection with learnable attention-based routing while strictly enforcing residual consistency. In speech conditions, our model achieves high top-1 accuracy (e.g., `[X]%` on a `[X]`-angle grid at SNR=`[X]dB`) and maintains cross-material robustness across five everyday targets spanning a broad spectrum of physical complexity. This establishes a framework for compact, array-free DOA sensing that leverages the natural physical encoding properties of everyday objects.

## Introduction
Direction-of-arrival (DOA) estimation is classically solved by array processing, where spatially separated sensors capture phase and time-difference cues that can be inverted via beamforming or subspace methods [@krim1996array]. Representative approaches include adaptive beamforming and high-resolution estimators [@capon1969fkw] and subspace methods such as MUSIC and ESPRIT [@schmidt1986music; @roy1989esprit], with comprehensive treatments in standard array and microphone-array texts [@vantrees2002optimum; @johnson1993array; @brandstein2001microphone_arrays; @benesty2008microphone_array]. However, the physical necessity of an array aperture imposes severe constraints on size and placement, limiting integration in compact devices or on surfaces exposed to harsh environments.

In this work, we propose a paradigm shift inspired by computational wave physics: **can everyday objects intrinsically encode acoustic direction in their material-dependent dynamics, such that a single non-contact vibration measurement is sufficient to decode DOA?** This concept parallels developments in time-reversed acoustics [@fink1997time; @draeger1997one], single-pixel imaging [@duarte2008single], structural health monitoring [@ing2008lamb], and transmission-matrix / wavefront-shaping views of complex media [@popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media], where complexity is treated as a computational resource. If valid, the sensing footprint reduces from an array aperture to a single optical spot, enabling compact integration where microphone arrays are infeasible.

We employ a laser Doppler vibrometer (LDV) not merely as a convenient readout, but as a methodological necessity to preserve the integrity of the physical encoder. LDV provides non-contact vibration measurement with high bandwidth and sensitivity [@rothberg2017ldv; @castellini2006ldv]. In contrast, contact sensors (e.g., piezoelectric patches or accelerometers) introduce local mass and stiffness perturbations that modify measured frequency-response functions and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Recent studies suggest that a single structural vibration measurement can carry directional cues because acoustic incidence excites structure-borne waves whose response depends on incidence direction [@dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor]. In parallel, sparse reconstruction offers an alternative DOA viewpoint by discretizing an angular manifold and recovering a sparse angular spectrum from linear measurements [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling]. Convex relaxations such as basis pursuit and \(\ell_1\)-regularized regression provide classical baselines [@chen2001basis_pursuit; @tibshirani1996lasso], while greedy pursuit algorithms (matching pursuit and OMP) offer efficient solvers that retain interpretability [@mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free alternatives also exist in the form of acoustic vector sensors, which infer DOA from collocated pressure–particle-velocity measurements but require specialized instrumentation [@nehorai1994vector_sensor].

A central open question is whether this apparent single-point “fingerprint” reflects a reproducible physical mechanism or merely target-specific idiosyncrasies. Here we treat the target structure as a *physical encoder* that transforms incident direction into a direction-dependent superposition of dispersive modes. This view suggests two falsifiable predictions: (i) the angle-to-spectrum mapping should be governed by a small number of dominant physical channels, and (ii) inference should remain stable under moderate noise and should transfer across structurally distinct targets if the underlying mechanism is universal.

We operationalize these predictions by translating structural dynamics into a linear angle–frequency response model, interrogating its effective degrees of freedom via singular value decomposition (SVD), and posing DOA inference as a structured sparse inverse problem. We use orthogonal matching pursuit (OMP) [@tropp2007omp] as an analytical probe and derive a physics-guided deep unrolling network [@monga2021unrolling] that replaces heuristic atom selection with attention-based routing while retaining residual-consistency constraints. We then evaluate robustness under additive noise and architectural ablations, analyze learned routing statistics as mechanistic evidence, and test cross-material generality across targets spanning a broad spectrum of physical complexity (Figs. 1–6).

## Results

### Everyday objects as reproducible physical encoders (Fig. 1)
We probe the physical-encoder hypothesis using the experimental configuration shown in Fig. 1a. An incident acoustic field excites a target structure, and we measure the resulting single-point vibration. To validate that the encoding is intrinsic to the object and not distorted by measurement loading, we compared non-contact LDV with a contact-sensor control (mimicking a piezoelectric mass). We found that contact loading significantly degrades fingerprint separability (reducing within-angle similarity from `[X]` to `[X]`) and DOA accuracy (dropping from `[X]%` to `[Y]%`). Consequently, all subsequent analyses use non-contact LDV to guarantee that the observed dynamics are purely structural.

Using this fidelity-first setup, we observed that while time-domain waveforms appear chaotic, the frequency-domain response reveals highly reproducible, direction-dependent **spectral fingerprints** (Fig. 1b). Quantitatively, these fingerprints exhibit high within-angle stability (`[X]±[Y]`) and distinct between-angle separability (`[X]±[Y]`). This robustness suggests that the fingerprints arise not from random scattering, but from a dispersive, modal superposition mechanism that concentrates DOA information. We strictly test this hypothesis by deriving a physical model (Fig. 2) and exposing its dominant channels.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.



### Discretizing the physical encoder via SVD (Fig. 2)
This section formalizes the “spectral fingerprint” concept. We treat the model and its linear algebra as **tools for discovery**: they allow us to ask *where* DOA information resides and *how many* dominant physical channels are effectively accessible.

#### Physics model → single-point response
Under small-amplitude dynamics, a thin plate is described by the Kirchhoff–Love operator [@timoshenko1959plates]. In the frequency domain:

$$ 
\left(D_p\nabla^4 - \rho t\,\omega^2 + i\omega c_d\right) W(x,y,\omega)
\;=\;
P(x,y,\theta,\omega),
$$ 

where \(W\) is displacement, \(P\) is the direction-dependent forcing, and \(D_p, \rho t, c_d\) are material parameters. Linearity implies a Green’s function representation:

$$ 
 W(x_L,y_L,\omega)
\;=\;
\iint_{\Omega}
 G\!\left((x_L,y_L),(x',y'),\omega\right)\,\ P(x',y',\theta,\omega)\,\mathrm{d}A'.
$$ 

**Interpretation.** DOA enters through the forcing pattern \(P(\cdot;\theta)\) and couples into dispersive structural dynamics via \(G\). Even when the time-domain waveform appears chaotic, the frequency response is governed by this stable linear operator. Because the Green's function \(G\) for a finite structure effectively acts as a compact operator with smooth kernels, the resulting response does not fill the entire Hilbert space. Instead, it concentrates energy into a limited number of modes. We define the complex single-point response \(Y(\omega;\theta)=W(x_L,y_L,\omega)\) and model it as a superposition of dispersive components:

$$ 
 Y(\omega;\theta) \;\approx\; \sum_{m=1}^{M} s_m(\omega)\,\alpha_m(\theta),
$$ 

Discretizing yields the empirical response matrix \(H \approx \sum_{m=1}^{M} u_m v_m^\top\), motivating the SVD interpretation.

#### SVD → dominant physical subspace
We use the SVD of the angle response matrix \(H = U\Sigma V^\top\) to expose effective degrees of freedom. A rapidly decaying singular spectrum (Fig. 2a) indicates that only a limited number of dominant channels contribute strongly, analogous to eigenchannels in complex media [@davy2015eigenchannels].

The leading `r=[X]` components capture `[Y]%` of the singular-value energy. We project the inverse model into this rank-\(r\) subspace: \(z = U_r^\top y\) and \(A = U_r^\top H\), reducing the problem to \(z \approx Ax\). This projection preserves the angle-indexed structure while focusing inference on the physically significant spectral channels.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes.
b, Modal decomposition into frequency-selective spectra \(u_r(f)\) and direction-selective polar patterns \(v_r(\theta)\).
c, Structured physical dictionary \(D\) assembled by combining spectral and directional components.

### Physics-guided sparse inference (Fig. 3)
We frame DOA estimation as a sparse inverse problem: given \(z\) and dictionary \(A\), find the sparse angle vector \(x\).

$$ 
\min_{x}\; \lVert z - Ax\rVert_2^2
\quad \text{s.t.}\quad \lVert x\rVert_0 \le K.
$$ 

This sparse-reconstruction viewpoint parallels classical sparse DOA methods that discretize a propagation manifold and solve for a sparse angular spectrum [@malioutov2005sparse_doa].

We compare two solvers:
1.  **OMP Baseline:** A greedy heuristic that iteratively selects atoms maximizing correlation with the residual [@mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. However, OMP makes "hard" binary decisions at each step; if noise causes an incorrect atom selection early on, the error propagates irreversibly—a known fragility in complex scattering media.
2.  **Physics-Guided Unrolled Network:** We unroll \(K\) pursuit stages into a deep network [@gregor2010lista; @hershey2014deep_unfolding; @monga2021unrolling]. Critically, we replace the brittle hard selection of OMP with a learnable **attention-based routing** mechanism (Fig. 3). Unlike a standard "black-box" CNN, this architecture is structurally constrained to follow the iterative physics of signal decomposition. The network computes a query from the residual \(r_t\) and keys from atoms \(a_e\), producing soft routing weights \(w_t\) that gate the coefficient update \(\Delta x_t\) [@vaswani2017attention; @bahdanau2015attention; @luong2015attention].

$$ 
 s_t[e] = \frac{\langle q_t, k_e\rangle}{\sqrt{d}}, \quad w_t = \mathrm{softmax}(s_t), \quad r_{t+1} = r_t - A\,\Delta x_t.
$$ 

**Physical Interpretation.** The attention weights \(w_t\) represent the model's probabilistic belief over the physical angle manifold. By maintaining a "soft" distribution of candidate directions rather than committing to a single angle immediately, the network effectively manages the uncertainty inherent in dispersive, noisy measurements, resolving ambiguities through residual consistency across layers.

![](../figures/fig03_unrolled-attention-omp.jpg)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating.**
At stage \(t\), the residual \(r_t\) is correlated with the physical dictionary \(A\). A transformer encoder generates attention weights that gate sparse updates \(\Delta x\), enforcing residual consistency \(r_{t+1}=r_t-A\Delta x\).

### Robustness and Ablation (Fig. 4)
We evaluate performance under additive white noise (SNR = 10, 5, 0 dB). The full physics-aware model achieves top-1 accuracy of `[X] / [Y] / [Z]%`, significantly outperforming a no-transformer variant (`[X]%` at 0 dB) and a fixed heuristic baseline (`[X]%` at 0 dB). Replacing sparse routing with dense routing causes an accuracy drop of `[X]` points at 0 dB, confirming that exploiting the structured sparsity of the physical dictionary is crucial for noise robustness.

![](../figures/fig04_noise-robustness-ablation.jpg)

**Fig. 4 | Performance under additive noise and architectural ablations.**
a, Validation accuracy at SNR = 10, 5, and 0 dB.
b, Component ablation confirming the value of physics-aware sparse routing.

### Deciphering the learned mechanism (Fig. 5)
Because our inference operates on a physical dictionary, we can inspect the learned routing weights \(w_t\) to verify mechanistic alignment.
1.  **Global Structure:** The self-attention map (Fig. 5a) exhibits a near-diagonal structure, indicating the model learns to route information based on proximity in the physical angle manifold.
2.  **Micro-Mechanism:** In a case study (Fig. 5b, \(\theta=60^\circ\)), analytical OMP selects spurious off-axis atoms due to noise, whereas the physics-aware network suppresses these errors, producing a sharp peak at the true DOA.
3.  **Macro-Statistics:** Aggregated across all angles (Fig. 5c), the physics-aware routing concentrates probability mass `[X]` times more densely on the true diagonal than OMP.

![](../figures/fig05_routing-mechanism-analysis.png)

**Fig. 5 | Deciphering model behaviour: attention structure and physical alignment.**
a, Global attention map showing physics-consistent diagonal structure.
b, Comparison of OMP vs. physics-aware selection for a single trial.
c, Aggregated selection probability showing superior diagonal concentration for the physics-aware model.

### Universality across complex materials (Fig. 6)
To test the universality of the encoding mechanism, we evaluated performance across five targets with distinct material properties (damping, stiffness) and geometries: acrylic, paper cup, wood, cardboard, and a laptop shell (Fig. 6a). Despite these differences, all targets exhibit direction-dependent dispersion signatures (Fig. 6b). The physics-aware model maintains low RMSE (`[X]–[Y]°`) across all materials, whereas analytical OMP degrades significantly (`> [Z]°`) on complex targets like cardboard and wood.

**Mechanism of Transfer.** While specific material parameters (stiffness \(D_p\), density \(\rho\)) shift the resonance frequencies and mode shapes, the fundamental **linear superposition principle** governs all targets. The physics-aware solver does not merely memorize spectral templates; it learns to invert this universal superposition logic, allowing it to adapt to the distinct "physical dictionaries" of different everyday objects.

![](../figures/fig06_cross-material-universality.jpg)

**Fig. 6 | Universal physical encoding across diverse materials.**
a, The five target objects.
b, Representative dictionary heatmaps showing shared dispersive structure.
c, Cross-material RMSE comparison.

## Discussion
Our results provide mechanistic evidence that single-point vibration measurements can robustly encode DOA through reproducible spectral fingerprints. By projecting the physical response into its dominant SVD subspace and solving the resulting sparse inverse problem with a physics-guided network, we achieve high accuracy and robustness without the need for sensor arrays.

The cross-material stability (Fig. 6) supports the claim that this encoding is a general property of finite elastic structures, not an artifact of a specific target. This paradigm leverages the natural complexity of the environment—treating the object as a computational resource—to simplify the sensing hardware.

**Limitations.** The current framework assumes a linear, time-invariant response during the measurement window. Performance may degrade for highly damped materials (e.g., foams, textiles) where modal overlap is extreme, rendering the linear separation of direction-dependent modes ill-conditioned. In such regimes, the data-driven corrections of the unrolled network become even more critical than in high-Q materials (like acrylic) to disentangle the smeared spectral signatures. Future work will also explore identifying the information-theoretic limits of this encoding and extending the model to broadband, multi-source scenarios.

## Methods
### Experimental setup
Experiments were conducted in a controlled acoustic environment with a single loudspeaker source at a radius of `[X] m`. The source angle was varied on a `[X]`-point grid (`[X]°` step). We measured out-of-plane velocity using a `[Manufacturer]` LDV at a sampling rate of `[X] kHz`.

**Object Selection.** We selected five objects (acrylic, paper, wood, cardboard, laptop) to span a range of **Q-factors** (damping) and structural complexity, testing the limits of the encoding mechanism.

### Signal Processing
We computed short-time Fourier transform (STFT) features (`[Window]` window, `[X]` length) [@allen1977stft], aggregated them into mean-power spectral vectors \(y \in \mathbb{R}^F\), and normalized them. The dictionary \(H\) was constructed from trial-averaged responses at each angle, consistent with standard vibroacoustic treatments of structure-borne sound and sound–structure interaction [@cremer2005structure_borne_sound; @fahy2007sound_structural_vibration; @rose2014guided_waves].

### Neural Network Training
The unrolled network (`K=[X]` layers) was trained to minimize `[Loss Function]` using `[Optimizer]`. We used a `[X]/[Y]/[Z]` train/val/test split, stratified by angle.

### Statistics
We report mean ± s.d. over `n=[X]` independent trials. Significance was assessed via two-sided t-tests.

## Data availability
Data available at `[Repository Link]`.

## Code availability
Code available at `[GitHub Link]`.

## Acknowledgements
Supported by `[Grant Info]`.

## References
::: {#refs} 
:::
