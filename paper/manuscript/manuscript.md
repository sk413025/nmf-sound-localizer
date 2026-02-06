# Non-contact acoustic sensing via the natural physical encoding of everyday objects

## Abstract
Conventional acoustic sensing relies on spatial sampling using microphone arrays, a paradigm that fundamentally limits miniaturization and deployment in harsh or constrained environments. Here we present a physics-first formulation of **single-point Direction-of-Arrival (DOA) sensing** in which the target structure itself acts as a *physical encoder*. We demonstrate that incident sound couples into structure-borne vibrations through a direction-dependent superposition of dispersive modes, producing a characteristic single-point spectral signature measurable by a non-contact laser Doppler vibrometer (LDV). By translating this physical process into a mathematical model, we reveal that the singular-value structure of the response matrix admits a limited effective number of dominant channels. This motivates a structured sparse inverse problem over a physical dictionary, solvable via a physics-guided deep unrolling network that replaces heuristic atom selection with learnable attention-based routing while strictly enforcing residual consistency. In speech conditions, our model achieves {TBD_SPEECH_TOP1_ACC_PCT}% top-1 accuracy on a {TBD_N_ANGLES}-angle grid ({TBD_ANGLE_RANGE_DEG} at {TBD_ANGLE_STEP_DEG}° resolution) and remains robust under additive noise down to {TBD_SNR_MIN_DB} dB SNR. Across five everyday targets, we reproduce direction-dependent fingerprints and maintain low DOA error under per-object calibration and retraining.

## Introduction
Extracting spatial information from wave fields is a recurring challenge spanning acoustics, optics, and seismology. In real environments, structural dispersion, multiple scattering, and boundary reverberation entangle propagation paths and produce seemingly chaotic time-domain signals [@rotter2017complex_media; @kuttruff2025room_acoustics]. Classical DOA estimation therefore relies on **spatial sampling** with sensor arrays and analytical inversion—beamforming and high-resolution estimators [@capon1969fkw], or subspace methods such as MUSIC and ESPRIT [@schmidt1986music; @roy1989esprit]—with standard treatments in array and microphone-array texts [@krim1996array; @vantrees2002optimum; @johnson1993array; @brandstein2001microphone_arrays; @benesty2008microphone_array]. However, the need for an aperture imposes severe constraints on size and placement, limiting integration in compact devices or harsh environments [@grumiaux2022_ssl_survey_deep_learning].

In this work, we propose a paradigm shift inspired by computational wave physics: **can everyday objects intrinsically encode acoustic direction in their material-dependent dynamics, such that a single non-contact vibration measurement is sufficient to decode DOA?** This concept parallels developments in time-reversed acoustics [@fink1997time; @draeger1997one], single-pixel imaging [@duarte2008single], structural health monitoring [@ing2008lamb], and transmission-matrix / wavefront-shaping views of complex media [@popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media], where complexity is treated as a computational resource. If valid, the sensing footprint reduces from an array aperture to a single optical spot, enabling compact integration where microphone arrays are infeasible.

Here we show that (i) everyday objects act as reproducible **direction-dependent physical encoders** for sound; (ii) the resulting angle–frequency response is governed by a small number of dominant channels revealed by SVD, motivating a **structured sparse inverse problem** on the physical angle manifold; and (iii) a physics-guided unrolled solver with learnable routing can decode DOA from a single LDV spot with high accuracy and robustness in speech conditions.

We employ a laser Doppler vibrometer (LDV) not merely as a convenient readout, but as a methodological necessity to preserve the integrity of the physical encoder. LDV provides non-contact vibration measurement with high bandwidth and sensitivity [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration]. In contrast, contact sensors (e.g., piezoelectric patches or accelerometers) introduce local mass and stiffness perturbations that modify measured frequency-response functions and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Recent studies suggest that a single structural vibration measurement can carry directional cues because acoustic incidence excites structure-borne waves whose response depends on incidence direction [@dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. In parallel, sparse reconstruction offers an alternative DOA viewpoint by discretizing an angular manifold and recovering a sparse angular spectrum from linear measurements [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling]. Convex relaxations such as basis pursuit and \(\ell_1\)-regularized regression provide classical baselines [@chen2001basis_pursuit; @tibshirani1996lasso], while greedy pursuit algorithms (matching pursuit and OMP) offer efficient solvers that retain interpretability [@mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free alternatives also exist in the form of acoustic vector sensors, which infer DOA from collocated pressure–particle-velocity measurements but require specialized instrumentation [@nehorai1994vector_sensor].

A central open question is whether this apparent single-point “fingerprint” reflects a reproducible physical mechanism or merely target-specific idiosyncrasies. Here we treat the target structure as a *physical encoder* that transforms incident direction into a direction-dependent superposition of dispersive modes. This view suggests two falsifiable predictions: (i) the angle-to-spectrum mapping should be governed by a small number of dominant physical channels, and (ii) inference should remain stable under moderate noise and should transfer across structurally distinct targets if the underlying mechanism is universal.

We operationalize these predictions by translating structural dynamics into a linear angle–frequency response model, interrogating its effective degrees of freedom via singular value decomposition (SVD), and posing DOA inference as a structured sparse inverse problem. We use orthogonal matching pursuit (OMP) [@tropp2007omp] as an analytical probe and derive a physics-guided deep unrolling network [@monga2021unrolling; @shlezinger2023model_based_deep_learning; @karniadakis2021physics_informed_ml] that replaces heuristic atom selection with attention-based routing while retaining residual-consistency constraints. We then evaluate robustness under additive noise and architectural ablations, analyze learned routing statistics as mechanistic evidence, and test cross-material generality across targets spanning a broad spectrum of physical complexity (Figs. 1–6).

**Notation and road map.** We denote the out-of-plane displacement field by \(W(x,y,\omega)\) and the LDV-measured velocity by \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\). For an incident direction \(\theta\), we define the complex single-point response \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). Because the observed DOA fingerprints are defined by magnitude statistics rather than phase, we summarize each clip by a time-averaged velocity power spectrum \(S(\omega_k;\theta)\) (estimated as a band-limited time-averaged power spectrum; Methods) and construct a log-power feature vector \(y=\phi(S)\) and its standardized version \(\tilde y\) (Methods). White-noise calibration yields an empirical per-angle prototype dictionary \(H=[h_1,\dots,h_E]\) (Methods), which we compress by SVD \(H=U\Sigma V^\top\) and use the rank-\(r\) projected variables \(z=U_r^\top \tilde y\) and \(A=U_r^\top H\). DOA inference is posed as sparse recovery over angles with coefficient vector \(x\) and pursuit depth \(K\). In Results, Fig. 1 establishes reproducible direction-dependent fingerprints; Fig. 2 reveals a low-dimensional physical subspace and defines the dictionary; Fig. 3 introduces a physics-guided unrolled solver; Fig. 4–5 provide robustness and mechanistic routing evidence; and Fig. 6 tests universality across materials under per-object calibration and retraining.

## Results

### Everyday objects act as naturally randomized physical encoders (Fig. 1)
Single-point vibroacoustic spectra of everyday objects exhibit reproducible, direction-dependent fingerprints (Fig. 1b): structural complexity—often dismissed as disorder—acts as a robust mechanism for spatial encoding. This spectral order persists even when the corresponding time-domain vibration appears irregular due to dispersion, multiple scattering, and room reverberation (Fig. 1a) [@rotter2017complex_media; @kuttruff2025room_acoustics]. A non-contact LDV provides the single-point readout while preserving the target’s native boundary conditions (Methods: Experimental setup).

These fingerprints exhibit high specificity. On a {TBD_N_ANGLES}-angle white-noise repeatability set ({TBD_WN_EVAL_CLIPS_PER_ANGLE} clips per angle), standardized log-power fingerprints are highly repeatable within angle (Pearson \(\rho_\mathrm{within}\) = {TBD_SIM_WITHIN_MEAN} ± {TBD_SIM_WITHIN_SD}) and approximately uncorrelated across angles (\(\rho_\mathrm{between}\) = {TBD_SIM_BETWEEN_MEAN} ± {TBD_SIM_BETWEEN_SD}; Methods: Fingerprint similarity analysis). This contrast confirms that the fingerprints are not random noise artifacts but deterministic, angle-specific modal superpositions. A contact-loading control (Methods; {TBD_CONTACT_LOADING_PANEL_REF}) yields {TBD_CONTACT_LOADING_RESULT_SUMMARY}, consistent with local mass–stiffness loading perturbing boundary conditions and shifting the measured frequency-response structure [@ewins2000modal; @bi2013transducer_mass_loading]. We therefore use non-contact LDV to preserve the fidelity of the natural physical encoder.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point “spectral fingerprints”.

### Spectral fingerprints arise from a low-dimensional physical manifold (Fig. 2)
The measured angle–frequency response exhibits a pronounced low-dimensional structure: the singular spectrum of the response matrix decays rapidly, indicating that only a small number of dominant spectral–spatial channels account for most direction-dependent variability (Fig. 2a–c). For a representative response matrix \(H\in\mathbb{R}^{F\times E}\), the leading \(r\) = {TBD_SVD_R} components capture {TBD_SVD_ENERGY_PCT}% of the singular-value energy (Fig. 2a), motivating a compressed inference space that remains physically interpretable while reducing sensitivity to noise and mismatch. We use a minimal physics model and linear algebra as **tools for discovery** to formalize this observation and define the structured dictionary used for inference.

#### A single-point response maps direction to spectrum
Under small-amplitude dynamics, plate-like targets can be approximated as linear systems that map a direction-dependent acoustic forcing into a single-point velocity response. Representative operator and Green’s-function formulations are provided in Methods (Derivation details and assumptions). In such models, direction enters through the forcing term \(P(\cdot,\theta,\omega)\), and the object converts it into a frequency-dependent response measured at the LDV spot—consistent with the empirical direction-dependent fingerprints in Fig. 1b.

The LDV measures out-of-plane velocity \(v(t)=\partial W(x_L,y_L,t)/\partial t\); in the frequency domain \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\). Throughout, we define the complex single-point response \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). Because our fingerprints are defined by magnitude statistics, we use the time-averaged velocity power spectrum \(S(\omega;\theta)\) (Methods) rather than the phase of \(Y\). This mapping can be viewed as a truncated modal expansion and, empirically, as a low-rank decomposition of the measured angle–frequency response (Fig. 2), motivating an approximate separable form (modal superposition)

$$ 
 Y(\omega;\theta) \;\approx\; \sum_{m=1}^{M} s_m(\omega)\,\alpha_m(\theta),
$$ 

where \(s_m(\omega)\) captures a dispersive spectral signature and \(\alpha_m(\theta)\) captures angle-dependent coupling [@ewins2000modal; @meirovitch2001fundamentals].
Even after compressing phase into magnitude statistics, the fingerprints retain a low effective dimensionality: only a small number of dominant channels are needed to explain most angle-dependent variation (Fig. 2a–c).

#### An angle-indexed dictionary formalizes fingerprints
To analyze these fingerprints on a discrete frequency grid \(\{\omega_f\}_{f=1}^F\), we represent each trial by a real feature vector

$$
y[f] \;=\; \phi\!\left(S(\omega_f;\theta)\right),
\qquad f=1,\dots,F,
$$

where \(\phi(\cdot)\) is a fixed transform (e.g., log-power) defined in Methods (Signal processing pipeline). This step isolates the stable spectral signature that differentiates directions (Fig. 1b) and defines the feature space used to construct the dictionary (Fig. 2c). Each atom \(h_e\) is an empirical prototype fingerprint for direction \(\theta_e\) estimated from calibration recordings (Methods: Dictionary calibration and data splits). For concision, we write \(y\) for the standardized feature vector \(\tilde y\) in the inverse model. We stack these atoms into the angle response matrix

$$
H \;=\; [h_1,\dots,h_E] \in \mathbb{R}^{F\times E}.
$$

Stacking prototypes across the angle grid yields the structured response matrix visualized in Fig. 2c.

Given an observation \(y\), we model it as a sparse combination of candidate angles:

$$
y \approx Hx + n,
\qquad \lVert x\rVert_0 \le K,
$$

where \(x\in\mathbb{R}^E\) is sparse over angles and \(n\) captures noise and mismatch.
This sparse-prototype model provides an interpretable link between fingerprint separability (Fig. 1) and pursuit-based inference (Fig. 3).

In our single-source setting, the ideal coefficient vector is 1-sparse, but we retain a \(K\)-stage pursuit formulation as an optimization budget that provides residual correction under noise and feature mismatch. Because \(\phi\) is generally nonlinear, the linear sparse model is interpreted as a prototype approximation in the resulting feature space; for a single source, it reduces to selecting the best-matching column of \(H\).

#### A few dominant channels explain most variability
The Singular Value Decomposition (SVD) of \(H = U\Sigma V^\top\) exposes effective degrees of freedom [@golub2013matrix_computations]. By the Eckart–Young theorem, truncating to the leading \(r\) components gives the best rank-\(r\) approximation (in Frobenius norm), providing a principled notion of a dominant physical subspace [@eckart1936approximation]. Equivalently,

$$
H = \sum_{m=1}^{\min(F,E)} \sigma_m\, u_m v_m^\top.
$$

A rapidly decaying singular spectrum (Fig. 2a) indicates that only a limited number of dominant channels contribute strongly, analogous to eigenchannels in complex media [@davy2015eigenchannels].

The leading rank-\(r\) components (with \(r\) = {TBD_SVD_R}) capture {TBD_SVD_ENERGY_PCT}% of the singular-value energy. Projecting the inverse model into this rank-\(r\) subspace via \(z = U_r^\top y\) and \(A = U_r^\top H\) reduces the problem to \(z \approx Ax\). Premultiplying by \(U_r^\top\) yields \(z = Ax + n'\), where \(n' = U_r^\top n\) captures projected noise and mismatch. This projection preserves the angle-indexed structure while focusing inference on the physically significant spectral channels.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes.
b, Modal decomposition into frequency-selective spectra \(u_r(f)\) and direction-selective polar patterns \(v_r(\theta)\).
c, The structured angle dictionary \(H\) and its rank-\(r\) SVD compression used for inference.

### Physics-guided routing resolves dispersive ambiguity (Fig. 3)
Dispersive fingerprints are informative but intrinsically ambiguous: multiple candidate angles can produce partially overlapping spectral responses, and greedy selection can pick off-axis atoms when noise corrupts correlations. We resolve this ambiguity by coupling a structured sparse inverse formulation to a learnable routing mechanism that remains anchored to the physical dictionary and enforces residual-consistent updates (Fig. 3). Evidence for ambiguity resolution is reflected in sustained accuracy under additive noise (Fig. 4) and the suppression of off-axis selections in the case study (Fig. 5b). Formally, we frame DOA estimation as sparse recovery on the physical angle manifold: the coefficient vector \(x\) indexes candidate directions, and the projected dictionary \(A\) encodes their SVD-compressed spectral signatures.

$$ 
\min_{x}\; \lVert z - Ax\rVert_2^2
\quad \text{s.t.}\quad \lVert x\rVert_0 \le K.
$$ 

Here \(K\) is the pursuit depth (selection budget) used by both analytical OMP and the unrolled solver in Fig. 3, providing residual correction under noise and feature mismatch.

This sparse-reconstruction viewpoint parallels classical sparse DOA methods that discretize a propagation manifold and solve for a sparse angular spectrum [@malioutov2005sparse_doa; @baraniuk2007compressive_sensing].

#### Greedy selection can drift off-axis under overlap (OMP baseline)
Orthogonal matching pursuit (OMP) is a canonical greedy solver for the \(\ell_0\)-constrained least-squares problem [@tropp2007omp; @tropp2004greed]. We use OMP as an interpretable baseline that probes the informativeness of the SVD-compressed physical dictionary: it greedily selects the atom most correlated with the current residual and refits coefficients on the selected support. In complex media, noise and partial spectral overlap can cause OMP to select off-axis angles (Fig. 5b), motivating a learned router that regularizes selection while preserving residual consistency. Algorithmic details are provided in Methods (Inference algorithms).

#### Learned routing stabilizes pursuit while enforcing residual consistency
OMP’s argmax selection is a fixed heuristic and can be brittle under noise and model mismatch in complex media. We therefore derive a neural solver by **unrolling** \(K\) pursuit stages into a network and replacing the discrete selection rule with learnable attention-based routing, while retaining physics-consistent residual updates [@gregor2010lista; @hershey2014deep_unfolding; @monga2021unrolling].

At stage \(t\), we start from residual \(r_t\in\mathbb{R}^r\) and correlations \(g_t=A^\top r_t\in\mathbb{R}^E\), where \(g_t\) provides a physically grounded match between the residual and each candidate angle. Routing weights \(w_t\in\mathbb{R}^E\) (a distribution over atoms) are produced by an attention router that conditions on the residual state and the full dictionary (Methods: Derivation details and assumptions). These weights gate a sparse update in coefficient space:

$$
\Delta x_t = \eta_t\,(w_t \odot g_t),
\qquad
r_{t+1} = r_t - A\,\Delta x_t,
$$

where \(\eta_t\) is a step size (learned or fixed) and \(\odot\) denotes element-wise product. After \(K\) stages, we accumulate \(x=\sum_{t=0}^{K-1}\Delta x_t\) and estimate DOA on the discrete grid from the coefficient mass (e.g., \(\hat\theta=\theta_{\arg\max_e |x[e]|}\); Methods).

**Physical Interpretation.** Attention makes the OMP→unrolling link explicit: when \(w_t\) concentrates to a one-hot vector at the maximally correlated atom, the update reduces to greedy selection; residual consistency keeps the solver anchored to the SVD-compressed physical dictionary.
As depicted in Fig. 3, the update alternates between physical matching (correlation with \(A\)) and learned manifold-aware routing (weights \(w_t\)), providing a mechanism for suppressing spurious off-axis correlations (Fig. 5b).

![](../figures/fig03_unrolled-attention-omp.jpg)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating.**
At stage \(t\), the residual \(r_t\) is correlated with the physical dictionary \(A\). A transformer encoder generates routing weights that gate sparse updates \(\Delta x_t\), enforcing residual consistency \(r_{t+1}=r_t-A\Delta x_t\).

### Experimental validation: end-to-end DOA decoding in speech remains robust under noise (Fig. 4)
On speech recordings (Methods), the physics-aware model decodes DOA under non-stationary content with high accuracy. On a dataset comprising {TBD_SPEECH_N_CLIPS_TOTAL} LDV clips ({TBD_SPEECH_CLIPS_PER_ANGLE} per angle) on a {TBD_N_ANGLES}-angle grid ({TBD_ANGLE_RANGE_DEG} at {TBD_ANGLE_STEP_DEG}° resolution), it achieves {TBD_SPEECH_TOP1_ACC_PCT}% top-1 validation accuracy (MAE {TBD_SPEECH_MAE_DEG}°, \(p_{95}\) = {TBD_SPEECH_P95_DEG}°).

Across additive white-noise conditions (mean over {TBD_N_INDEP_RUNS} independent runs), performance remains high at SNR = 10 and 5 dB and degrades gracefully at 0 dB (Fig. 4a; top-1 accuracy = {TBD_ACC_SNR10_PCT}% / {TBD_ACC_SNR5_PCT}% / {TBD_ACC_SNR0_PCT}% at 10/5/0 dB). Architectural ablations confirm that learned sparse routing is essential (Fig. 4b; ablation definitions in Methods): removing the transformer reduces accuracy to {TBD_ACC_ABL_NO_TRANSFORMER_PCT}%, while replacing learned routing with a fixed heuristic or dense routing collapses performance to {TBD_ACC_ABL_FIXED_HEURISTIC_PCT}% and {TBD_ACC_ABL_DENSE_ROUTING_PCT}% (chance-level for {TBD_N_ANGLES}-way classification).

![](../figures/fig04_noise-robustness-ablation.jpg)

**Fig. 4 | Physics-guided sparse routing remains robust under noise.**
a, Validation accuracy at SNR = 10, 5, and 0 dB.
b, Component ablation confirming the value of physics-aware sparse routing.

### Learned routing aligns with the physical angle manifold (Fig. 5)
Learned routing weights \(w_t\) reveal a physics-consistent mechanism (Fig. 5a–c). The global self-attention map exhibits a near-diagonal structure, indicating that the model routes information according to proximity on the physical angle manifold (Fig. 5a). In a representative case study, analytical OMP selects a spurious off-axis angle due to content-induced spectral ambiguity, whereas the physics-aware network suppresses this error and concentrates probability mass at the true DOA (Fig. 5b). Aggregated across all angles, learned routing concentrates probability mass {TBD_DIAGONAL_CONC_FACTOR} times more densely on the true diagonal than OMP (Fig. 5c).

![](../figures/fig05_routing-mechanism-analysis.png)

**Fig. 5 | Learned routing aligns with the physical angle manifold.**
a, Global attention map showing physics-consistent diagonal structure.
b, Comparison of OMP vs. physics-aware selection for a single trial.
c, Aggregated selection probability showing superior diagonal concentration for the physics-aware model.

### Physical encoding is universal across everyday materials (Fig. 6)
Across five targets with distinct material properties (damping, stiffness) and geometries—acrylic, paper cup, wood, cardboard, and a laptop shell (Fig. 6a)—we consistently observe direction-dependent dispersion signatures (Fig. 6b). Under a per-object calibrate-and-retrain protocol (Methods: Dictionary calibration and data splits), the physics-aware model maintains low DOA error (Fig. 6c; RMSE {TBD_RMSE_RANGE_DEG}°), whereas analytical OMP degrades substantially on the most complex targets (> {TBD_RMSE_OMP_COMPLEX_DEG}°).

**Mechanism of Universality.** While specific material parameters (stiffness \(D_p\), density \(\rho\)) shift resonance frequencies and mode shapes, the fundamental **linear superposition principle** governs all targets. Calibration yields an object-specific dictionary \(H\) that captures each object’s dispersive fingerprints, and retraining learns an inverse mapping that exploits the shared sparse-superposition logic. This supports the interpretation that the solver learns the physical logic encoded by \(H\) rather than memorizing individual trials.

![](../figures/fig06_cross-material-universality.jpg)

**Fig. 6 | Universal physical encoding across diverse materials.**
a, The five target objects.
b, Representative dictionary heatmaps showing shared dispersive structure.
c, Cross-material RMSE comparison.

## Discussion
These results establish a complementary sensing paradigm in which passive everyday structures function as naturally randomized **physical encoders** for sound. A single, non-contact vibration readout contains enough information to decode DOA because the object converts incidence direction into a reproducible spectral fingerprint (Fig. 1), shifting the sensing footprint from an array aperture to a single optical spot.

**Mechanism (physics).** Directional information arises because an incident acoustic field excites structure-borne waves whose coupling depends on incidence direction and boundary conditions. In the modal view, direction modulates the participation weights \(\alpha_m(\theta)\), consistent with the interpretive separable form \(Y(\omega;\theta)\approx \sum_m s_m(\omega)\alpha_m(\theta)\) introduced in Results (Fig. 2b). This step assumes approximately linear time-invariant dynamics over each analysis window and is used here to explain the observed fingerprints rather than to identify individual modes. White-noise calibration operationalizes this mapping by estimating per-angle prototypes \(h_e\), which serve as empirical “atoms” that encode how the object’s dispersive dynamics respond to each direction (Fig. 1b).

**Mechanism (low effective rank).** The rapid decay of the singular spectrum (Fig. 2a) indicates that the measured angle–frequency response is governed by a small number of dominant channels, consistent with eigenchannel interpretations of complex media. This low effective rank explains why projecting into the dominant subspace improves robustness: it retains the physically informative variation while suppressing directions that contribute weakly or inconsistently. As damping increases, modal overlap grows and directional identifiability can degrade because distinct directions become less separable in the compressed channel space [@ewins2000modal; @inman2013engineering_vibration].

**Mechanism (routing on an angle manifold).** Greedy pursuit can fail when content-induced spectral overlap causes spurious off-axis correlations (Fig. 5b). The physics-guided unrolled solver addresses this by learning a soft routing distribution \(w_t\) that regularizes selection using context across the entire angle manifold, while preserving residual-consistent updates anchored to the physical dictionary (Fig. 3). The emergent near-diagonal attention structure (Fig. 5a,c) provides mechanistic evidence that the solver exploits manifold smoothness—nearby angles share similar dispersive structure—thereby stabilizing inference under noise and nonstationary speech (Fig. 4).

**Scientific frontiers and outlook.** Our current pipeline requires per-object calibration to construct an object-specific dictionary \(H\) and retraining to learn the inverse mapping (Fig. 6), highlighting a frontier question: which invariances of the angle–frequency manifold are shared across objects, and which are inherently object-specific? A second frontier is to quantify information-theoretic limits of single-point decoding as damping and modal overlap increase (Fig. 2a), and to determine how mounting conditions and LDV spot placement trade off robustness versus sensitivity. Extending the framework to multi-source and moving-source regimes will require modeling how fingerprints superpose and evolve across time windows, and identifying when the residual-correction depth \(K\) captures meaningful additional structure versus overfitting.

## Methods
### Software and hardware
All processing and learning were implemented in Python. Neural models were implemented in PyTorch and trained on Apple Silicon using Metal Performance Shaders (MPS) in the `trl-training` conda environment. Unless otherwise stated, random seeds were fixed to {TBD_RANDOM_SEED} (NumPy and PyTorch).

### Experimental setup
Experiments were conducted in a controlled acoustic environment with a single loudspeaker source at a radius of {TBD_RADIUS_M} m. We define \(\theta=0^\circ\) as {TBD_ANGLE_ZERO_REFERENCE} and increase \(\theta\) in {TBD_ANGLE_SIGN_CONVENTION} direction. The source angle was varied on a {TBD_N_ANGLES}-point grid covering {TBD_ANGLE_RANGE_DEG} with step {TBD_ANGLE_STEP_DEG}°. We measured out-of-plane surface velocity using a {TBD_LDV_MODEL} laser Doppler vibrometer (LDV). All downstream processing uses waveforms resampled to {TBD_FS_HZ} Hz. The LDV spot location was {TBD_LDV_SPOT_LOCATION} and the object mounting/boundary condition was {TBD_OBJECT_MOUNTING_CONDITION}.

We used two excitation regimes: (i) white-noise playback for dictionary calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end DOA decoding and robustness experiments. Unless otherwise stated, each clip duration is {TBD_TRIAL_DURATION_S} s.

**Object Selection.** We selected five objects (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell) to span a range of **Q-factors** (damping) and structural complexity, testing the limits of the encoding mechanism.

### Controls and additional protocols
**Contact-loading control.** To test whether directional fingerprints depend on preserving the target’s native boundary conditions, we repeated white-noise measurements while mechanically loading the target at the sensing site with a small contact transducer (mimicking a bonded piezoelectric sensor). Specifically, we attached a {TBD_CONTACT_SENSOR_TYPE} of mass {TBD_CONTACT_SENSOR_MASS_G} g at {TBD_CONTACT_SENSOR_LOCATION} using {TBD_CONTACT_SENSOR_ATTACHMENT}, and repeated the repeatability and decoding analyses under identical playback and geometry. We report the resulting changes in within/between fingerprint similarity and DOA decoding as {TBD_CONTACT_LOADING_RESULT_SUMMARY} ({TBD_CONTACT_LOADING_PANEL_REF}). This control is intended to isolate mass–stiffness loading effects on the measured response [@ewins2000modal; @bi2013transducer_mass_loading].

### Dictionary calibration and data splits
The angle dictionary \(H\) is constructed exclusively from white-noise calibration recordings. For each target object, we acquire a white-noise dataset organized by angle and partition it into (i) a **calibration subset** used solely to estimate \(H\) and (ii) a disjoint **evaluation subset** used to quantify fingerprint repeatability/separability (Fig. 1) and verify dictionary validity. The calibration subset contains {TBD_WN_CALIB_CLIPS_PER_ANGLE} clips per angle selected by {TBD_WN_SPLIT_RULE}; the remaining {TBD_WN_EVAL_CLIPS_PER_ANGLE} clips per angle are reserved for evaluation. Calibration clips are never used for training or evaluation of the DOA decoder.

For speech decoding (Fig. 4–5), we train the decoder on a speech dataset split into train/validation/test sets stratified by angle using {TBD_SPEECH_SPLIT_RULE} (counts: {TBD_SPLIT_TRAIN} / {TBD_SPLIT_VAL} / {TBD_SPLIT_TEST} clips). All speech results use the fixed \(H\) estimated from white-noise calibration only.

For cross-material experiments (Fig. 6), we repeat the full pipeline per target: acquire object-specific white-noise calibration data, estimate that object’s \(H\), and retrain a separate model for that object. Universality is therefore assessed at the level of a shared physical mechanism and inference recipe, not by reusing weights across objects.

### Signal processing pipeline
**Definition ladder (waveform → dictionary).** For a clip at direction \(\theta\), the LDV provides an out-of-plane velocity waveform \(v[n]\) (m/s). We compute an STFT \(V[k,t]\), form a time-averaged power spectrum estimate \(\widehat S(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2\), and restrict it to a frequency band to obtain a band-limited fingerprint. Features are defined as \(y=\phi(\widehat S)\), standardized to \(\tilde y\), and averaged across calibration trials to form per-angle prototypes \(h_e\) and the dictionary \(H=[h_1,\dots,h_E]\).

To extract stable spectral features from the measured velocity time series \(v[n]\) (sampled at rate \(f_s\); see Parameters), we use a fixed short-time Fourier transform (STFT) pipeline [@allen1977stft]. With a Hann window \(w[m]\) of length \(N_\mathrm{win}\) samples and hop size \(N_\mathrm{hop}\) samples (overlap; see Parameters), we compute
$$
V[k,t] = \sum_{m=0}^{N_\mathrm{win}-1} v[tN_\mathrm{hop}+m]\, w[m]\, e^{-i2\pi km/N_\mathrm{FFT}},
$$
where \(N_\mathrm{FFT}\) is the FFT size and \(k\) indexes frequency bins. We form a time-averaged power spectrum
$$
P[k] = \frac{1}{T}\sum_{t=1}^{T} \left|V[k,t]\right|^2,
$$
apply a band mask \(f_k\in[f_\min,f_\max]\) Hz (Parameters) to obtain \(P_\mathrm{band}\in\mathbb{R}^{F}\). We interpret \(P_\mathrm{band}[k]\) as a finite-sample estimator of the band-limited power fingerprint \(S(\omega_k;\theta)\) used in Results. The feature vector is
$$
\widehat S(\omega_k;\theta) := P_\mathrm{band}[k],
\qquad
y[k] = \phi\!\left(\widehat S(\omega_k;\theta)\right) = \log_{10}\!\left(\widehat S(\omega_k;\theta) + \epsilon\right).
$$
This definition makes explicit the link between the physical complex response \(Y(\omega;\theta)\) and the observable fingerprint: we use a time-averaged magnitude statistic \(\widehat S\) (no phase) to obtain stable, repeatable fingerprints. In Results, we write \(S\) for the underlying power-spectrum fingerprint, with \(\widehat S\) denoting its finite-sample estimator.

We estimate per-frequency standardization statistics \((\mu_k,\sigma_k)\) from the white-noise calibration subset and define the standardized feature
$$
\tilde y[k] = \frac{y[k]-\mu_k}{\sigma_k+\epsilon_z}.
$$
The same \((\mu_k,\sigma_k)\) is applied to the white-noise evaluation subset and all speech clips so that observations and dictionary atoms live in the same feature space.

**Angle dictionary construction.** For each candidate direction \(\theta_e\), we compute the per-trial feature vectors \(\tilde y_{e,j}\in\mathbb{R}^F\) and form the angle-conditioned atom
$$
h_e = \frac{1}{R_e^{\mathrm{cal}}}\sum_{j=1}^{R_e^{\mathrm{cal}}}\tilde y^{\mathrm{cal}}_{e,j}, \qquad H=[h_1,\dots,h_E]\in\mathbb{R}^{F\times E},
$$
where \(\tilde y^{\mathrm{cal}}_{e,j}\) denotes features extracted from the white-noise calibration subset (Methods: Dictionary calibration and data splits), \(R_e^{\mathrm{cal}}\) is the number of calibration trials at angle \(\theta_e\), and \(E\) = {TBD_N_ANGLES}. This construction yields a structured, angle-indexed dictionary consistent with standard vibroacoustic treatments of structure-borne sound and sound–structure interaction [@cremer2005structure_borne_sound; @fahy2007sound_structural_vibration; @rose2014guided_waves].

**SVD compression.** We compute \(H=U\Sigma V^\top\) and define the rank-\(r\) singular-value energy fraction as
$$
\mathrm{Energy}(r) = \frac{\sum_{m=1}^{r}\sigma_m^2}{\sum_{m=1}^{\min(F,E)}\sigma_m^2}.
$$
For analysis (Fig. 2), we report the rank \(r\) = {TBD_SVD_R} subspace that captures {TBD_SVD_ENERGY_PCT}% singular-value energy. The rank used for inference and learning is {TBD_SVD_INFERENCE_POLICY}. For inference, we project the feature vector and dictionary into the rank-\(r\) subspace via \(z=U_r^\top \tilde y\) and \(A=U_r^\top H\).

### Inference algorithms
Given the projected feature \(z\in\mathbb{R}^r\) and projected dictionary \(A\in\mathbb{R}^{r\times E}\), we solve the sparse inverse problem \(z\approx Ax\) under a \(\lVert x\rVert_0\le K\) constraint using OMP as an analytical baseline and an unrolled, physics-guided neural solver (Results, Fig. 3).

**OMP baseline.** In the projected space, OMP is defined by the recursion. Let \(S_0=\varnothing\), \(x^{(0)}=0\), and \(r_0=z\). For \(t=0,\dots,K-1\), define
$$
\begin{aligned}
i_t &= \arg\max_{e\in\{1,\dots,E\}} \left|a_e^\top r_t\right|,\\
S_{t+1} &= S_t \cup \{i_t\},\\
x^{(t+1)}_{S_{t+1}} &= \arg\min_{u}\;\lVert z - A_{S_{t+1}}u\rVert_2^2,\qquad x^{(t+1)}_{S_{t+1}^c}=0,\\
r_{t+1} &= z - A x^{(t+1)},
\end{aligned}
$$
where \(a_e\) is the \(e\)-th column of \(A\) and \(A_{S}\) denotes the subdictionary restricted to indices \(S\). After \(K\) stages, we set \(x=x^{(K)}\) and predict \(\hat\theta=\theta_{\arg\max_e |x[e]|}\) on the discrete angle grid.

**Physics-guided unrolling.** We unroll \(K\) pursuit stages into a network, replace the discrete selection rule with learnable attention-based routing, and retain residual-consistent updates (Results, Fig. 3). This yields a differentiable solver whose routing weights can be inspected as mechanistic evidence (Results, Fig. 5).

### Derivation details and assumptions
**Assumptions.** The derivations above and below rely on three modeling choices. First, over each analysis window the target is approximated as linear and time-invariant under small-amplitude dynamics, so that direction-dependent responses superpose in the frequency domain. Second, after the nonlinear feature transform \(\phi\) (log-power), we interpret \(Hx\) as a sparse *prototype approximation* in the resulting standardized feature space, not as a literal physical power-additivity law. Third, SVD projection preserves the angle-indexed structure while mapping the noise/mismatch term to \(n' = U_r^\top n\).

**Representative physical operator and Green’s function (Fig. 2).** Plate-like targets (e.g., the acrylic plate) can be described by a linear operator equation (Kirchhoff–Love theory) [@timoshenko1959plates]; more generally, linear plate/shell operators yield the same direction-to-spectrum mapping used in Results. In the frequency domain, one representative form is
$$ 
 \left(D_p\nabla^4 - \rho t\,\omega^2 + i\omega c_d\right) W(x,y,\omega)
 \;=\;
 P(x,y,\theta,\omega),
$$ 
where \(W(x,y,\omega)\) is the complex out-of-plane displacement field, \(P(x,y,\theta,\omega)\) is the effective forcing induced by an incident field from direction \(\theta\), \(D_p\) is bending stiffness, \(\rho t\) is areal mass density, and \(c_d\) is an effective damping term. A single-point LDV measurement at \((x_L,y_L)\) admits a Green’s function representation [@kress2014linear_integral_equations]
$$
W(x_L,y_L,\omega)
\;=\;
\iint_{\Omega}
G\!\left((x_L,y_L),(x',y'),\omega\right)\,
P(x',y',\theta,\omega)\,\mathrm{d}A'.
$$
The LDV measures out-of-plane velocity \(v(t)=\partial W(x_L,y_L,t)/\partial t\); in the frequency domain \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), so the complex single-point response used in Results is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\).

**Attention routing equations (Fig. 3).** In the rank-\(r\) projected space, at stage \(t\) we start from residual \(r_t\in\mathbb{R}^r\) and compute correlations \(g_t=A^\top r_t\in\mathbb{R}^E\). We parameterize routing weights over atoms via dot-product attention [@vaswani2017attention; @bahdanau2015attention; @luong2015attention]. A query is computed from the current state,
$$
q_t = W_q r_t \in \mathbb{R}^d,
$$
and each atom \(a_e\in\mathbb{R}^r\) (column \(e\) of \(A\)) is embedded as a key \(k_e = W_k a_e \in \mathbb{R}^d\). The routing scores and weights are
$$
 s_t[e] = \frac{\langle q_t, k_e\rangle}{\sqrt{d}},
 \qquad
 w_t = \mathrm{softmax}(s_t)\in\mathbb{R}^E.
$$
These weights gate a sparse update in coefficient space,
$$
\Delta x_t = \eta_t\,(w_t \odot g_t),
\qquad
r_{t+1} = r_t - A\,\Delta x_t,
$$
which can be interpreted as a masked gradient step on the least-squares objective: \(g_t=A^\top r_t\) provides a descent direction, while \(w_t\) provides a learned soft selection mask. After \(K\) stages, we accumulate \(x=\sum_{t=0}^{K-1}\Delta x_t\) and predict \(\hat\theta\) on the discrete grid (e.g., \(\hat\theta=\theta_{\arg\max_e |x[e]|}\)).

### Neural Network Implementation and Training
The physics-guided unrolled network was implemented in PyTorch [@paszke2019pytorch]. The architecture consists of \(K\) unrolled pursuit stages with a transformer router (embedding dimension \(d_\mathrm{model}\), \(h\) attention heads, and {TBD_N_LAYERS} encoder layers; Results, Fig. 3). We use \(K\) = {TBD_K_STAGES}, \(d_\mathrm{model}\) = {TBD_D_MODEL}, and \(h\) = {TBD_N_HEADS}. At stage \(t\), the current residual \(r_t\) is correlated with the dictionary to form a physical match \(g_t=A^\top r_t\). The router uses scaled dot-product attention to produce routing weights \(w_t\), and the stage update uses residual-consistent subtraction \(r_{t+1}=r_t-A\Delta x_t\). Expert-level logits are obtained by {TBD_LOGIT_CONSTRUCTION}, and the predicted direction is \(\hat\theta=\arg\max_e p[e]\).

*   **Loss:** {TBD_TRAIN_LOSS_DESCRIPTION}.
*   **Optimization:** {TBD_OPTIMIZER} with learning rate \(\eta\) = {TBD_LR} and weight decay \(\lambda\) = {TBD_WEIGHT_DECAY}.
*   **Protocol:** {TBD_EPOCHS} epochs, batch size {TBD_BATCH_SIZE}, random seed {TBD_RANDOM_SEED}. We use a deterministic split stratified by angle ({TBD_SPEECH_SPLIT_RULE}), with counts {TBD_SPLIT_TRAIN} / {TBD_SPLIT_VAL} / {TBD_SPLIT_TEST}.

**Ablations (Fig. 4b).** We report three ablations with all other components held fixed. **No-transformer** replaces the transformer router with {TBD_ABL_NO_TRANSFORMER_DEF}. **Fixed heuristic** replaces learned routing with {TBD_ABL_FIXED_HEURISTIC_DEF}. **Dense routing** uses {TBD_ABL_DENSE_ROUTING_DEF}. Each ablation uses the same dictionary construction and evaluation protocol as the full model.

Noise robustness experiments add zero-mean white noise at SNR levels {TBD_SNR_LEVELS_DB} in the time domain:
$$
v_\mathrm{noisy} = v + \alpha \xi,\quad \xi\sim\mathcal{N}(0,1),\quad \mathrm{SNR}=10\log_{10}\frac{\lVert v\rVert_2^2}{\lVert \alpha \xi\rVert_2^2}.
$$
The SNR sweep in Fig. 4 uses the same feature extraction and dictionary construction for each noise level.

**Evaluation metrics.** We report top-1 accuracy \(\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}[\hat\theta_i=\theta_i]\) on the discrete angle grid. When reporting angular error in degrees, we compute the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\) and report \(\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\).

### Fingerprint Similarity Analysis
To quantify the uniqueness and stability of the physical encoding (Fig. 1), we compute the Pearson correlation coefficient between feature vectors \(a,b\in\mathbb{R}^F\):
$$
\rho(a,b)=\frac{(a-\bar a)^\top (b-\bar b)}{\lVert a-\bar a\rVert_2\,\lVert b-\bar b\rVert_2},
\qquad \bar a=\frac{1}{F}\sum_{k=1}^{F} a[k].
$$
**Within-angle similarity** is defined as
$$
\rho_\mathrm{within}=\frac{1}{E}\sum_{e=1}^{E}\;\frac{2}{R_e(R_e-1)}\sum_{1\le j<k\le R_e} \rho(\tilde y_{e,j},\tilde y_{e,k}),
$$
and **between-angle similarity** is defined as the uniform average over unordered angle pairs,
$$
\rho_\mathrm{between}=\frac{2}{E(E-1)}\sum_{1\le e<e'\le E}\;\frac{1}{R_eR_{e'}}\sum_{j=1}^{R_e}\sum_{k=1}^{R_{e'}} \rho(\tilde y_{e,j},\tilde y_{e',k}).
$$
We report mean ± s.d. over the per-angle (within) and per-angle-pair (between) aggregates to avoid overweighting angles with more trials.

### Statistics
We report mean ± s.d. over \(n\) = {TBD_N_REPEATS} independent replicates as defined per experiment: {TBD_REPLICATE_DEFINITION}. Significance was assessed via two-sided t-tests with threshold {TBD_PVALUE_THRESHOLD} and multiple-comparison handling {TBD_MULTIPLE_COMPARISON_POLICY}.

### Parameters
| Component | Parameter | Symbol | Value |
|---|---|---:|---|
| Acquisition | Sampling rate | \(f_s\) | {TBD_FS_HZ} |
| Acquisition | Loudspeaker radius | \(R\) | {TBD_RADIUS_M} |
| Acquisition | Angle grid size | \(E\) | {TBD_N_ANGLES} |
| Acquisition | Angle step | \(\Delta\theta\) | {TBD_ANGLE_STEP_DEG}° |
| Acquisition | Angle range |  | {TBD_ANGLE_RANGE_DEG} |
| Acquisition | Angle zero reference |  | {TBD_ANGLE_ZERO_REFERENCE} |
| Acquisition | Angle sign convention |  | {TBD_ANGLE_SIGN_CONVENTION} |
| Acquisition | LDV model |  | {TBD_LDV_MODEL} |
| Acquisition | LDV spot location |  | {TBD_LDV_SPOT_LOCATION} |
| Acquisition | Object mounting condition |  | {TBD_OBJECT_MOUNTING_CONDITION} |
| Acquisition | Trials per angle |  | {TBD_WN_CALIB_CLIPS_PER_ANGLE} (WN calib) / {TBD_WN_EVAL_CLIPS_PER_ANGLE} (WN eval) / {TBD_SPEECH_CLIPS_PER_ANGLE} (speech) |
| Acquisition | Trial duration |  | {TBD_TRIAL_DURATION_S} |
| Acquisition | Excitation signal types |  | white noise; speech |
| STFT | FFT size | \(N_\mathrm{FFT}\) | {TBD_NFFT} |
| STFT | Window length | \(N_\mathrm{win}\) | {TBD_STFT_WIN_SAMPLES} |
| STFT | Hop length | \(N_\mathrm{hop}\) | {TBD_STFT_HOP_SAMPLES} |
| STFT | Overlap |  | {TBD_STFT_OVERLAP_PCT} |
| STFT | Band limits | \([f_\min,f_\max]\) | {TBD_FREQ_MIN_HZ}–{TBD_FREQ_MAX_HZ} Hz |
| STFT | Frequency bins | \(F\) | {TBD_F_BINS} |
| Features | Log clamp | \(\epsilon\) | {TBD_EPS} |
| Features | z-score clamp | \(\epsilon_z\) | {TBD_ZSCORE_EPS} |
| SVD | Truncation rank | \(r\) | {TBD_SVD_R} |
| SVD | Energy captured | \(\mathrm{Energy}(r)\) | {TBD_SVD_ENERGY_PCT}% |
| SVD | Inference policy |  | {TBD_SVD_INFERENCE_POLICY} |
| Model | Unrolled stages | \(K\) | {TBD_K_STAGES} |
| Model | Embedding dim. | \(d_\mathrm{model}\) | {TBD_D_MODEL} |
| Model | Attention heads | \(h\) | {TBD_N_HEADS} |
| Model | Encoder layers |  | {TBD_N_LAYERS} |
| Model | Logit construction |  | {TBD_LOGIT_CONSTRUCTION} |
| Training | Optimizer |  | {TBD_OPTIMIZER} |
| Training | Learning rate | \(\eta\) | {TBD_LR} |
| Training | Weight decay | \(\lambda\) | {TBD_WEIGHT_DECAY} |
| Training | Batch size |  | {TBD_BATCH_SIZE} |
| Training | Epochs |  | {TBD_EPOCHS} |
| Training | Seed |  | {TBD_RANDOM_SEED} |
| Noise | SNR levels (Fig. 4a) |  | {TBD_SNR_LEVELS_DB} |

<!-- Remaining placeholders (to be filled before submission):
- Abstract: {TBD_SNR_MIN_DB}.
- Results (Fig. 1): {TBD_WN_EVAL_CLIPS_PER_ANGLE}, {TBD_SIM_WITHIN_MEAN}, {TBD_SIM_WITHIN_SD}, {TBD_SIM_BETWEEN_MEAN}, {TBD_SIM_BETWEEN_SD}.
- Results (Fig. 2): {TBD_SVD_R}, {TBD_SVD_ENERGY_PCT}, {TBD_SVD_INFERENCE_POLICY}.
- Results (Fig. 4): {TBD_SPEECH_N_CLIPS_TOTAL}, {TBD_SPEECH_CLIPS_PER_ANGLE}, {TBD_SPEECH_TOP1_ACC_PCT}, {TBD_SPEECH_MAE_DEG}, {TBD_SPEECH_P95_DEG},
  {TBD_N_INDEP_RUNS}, {TBD_ACC_SNR10_PCT}, {TBD_ACC_SNR5_PCT}, {TBD_ACC_SNR0_PCT}, {TBD_ACC_ABL_NO_TRANSFORMER_PCT},
  {TBD_ACC_ABL_FIXED_HEURISTIC_PCT}, {TBD_ACC_ABL_DENSE_ROUTING_PCT}.
- Results (Fig. 5): {TBD_DIAGONAL_CONC_FACTOR}.
- Results (Fig. 6): {TBD_RMSE_RANGE_DEG}, {TBD_RMSE_OMP_COMPLEX_DEG}.
- Experimental setup: {TBD_RADIUS_M}, {TBD_N_ANGLES}, {TBD_ANGLE_RANGE_DEG}, {TBD_ANGLE_STEP_DEG}, {TBD_ANGLE_ZERO_REFERENCE}, {TBD_ANGLE_SIGN_CONVENTION},
  {TBD_LDV_MODEL}, {TBD_LDV_SPOT_LOCATION}, {TBD_OBJECT_MOUNTING_CONDITION}, {TBD_TRIAL_DURATION_S}.
- Controls: {TBD_CONTACT_SENSOR_TYPE}, {TBD_CONTACT_SENSOR_MASS_G}, {TBD_CONTACT_SENSOR_LOCATION}, {TBD_CONTACT_SENSOR_ATTACHMENT}, {TBD_CONTACT_LOADING_RESULT_SUMMARY}, {TBD_CONTACT_LOADING_PANEL_REF}.
- Calibration/splits: {TBD_WN_CALIB_CLIPS_PER_ANGLE}, {TBD_WN_EVAL_CLIPS_PER_ANGLE}, {TBD_WN_SPLIT_RULE}, {TBD_SPEECH_SPLIT_RULE}, {TBD_SPLIT_TRAIN}, {TBD_SPLIT_VAL}, {TBD_SPLIT_TEST}.
- STFT/features: {TBD_FS_HZ}, {TBD_NFFT}, {TBD_STFT_WIN_SAMPLES}, {TBD_STFT_HOP_SAMPLES}, {TBD_STFT_OVERLAP_PCT}, {TBD_FREQ_MIN_HZ}, {TBD_FREQ_MAX_HZ}, {TBD_F_BINS}, {TBD_EPS}, {TBD_ZSCORE_EPS}.
- Model/training: {TBD_K_STAGES}, {TBD_D_MODEL}, {TBD_N_HEADS}, {TBD_N_LAYERS}, {TBD_LOGIT_CONSTRUCTION}, {TBD_TRAIN_LOSS_DESCRIPTION}, {TBD_OPTIMIZER}, {TBD_LR}, {TBD_WEIGHT_DECAY}, {TBD_BATCH_SIZE}, {TBD_EPOCHS}, {TBD_RANDOM_SEED}.
- Ablations: {TBD_ABL_NO_TRANSFORMER_DEF}, {TBD_ABL_FIXED_HEURISTIC_DEF}, {TBD_ABL_DENSE_ROUTING_DEF}.
- Statistics: {TBD_N_REPEATS}, {TBD_REPLICATE_DEFINITION}, {TBD_PVALUE_THRESHOLD}, {TBD_MULTIPLE_COMPARISON_POLICY}.
- Availability/admin: {TBD_DATA_AVAILABILITY_URL}, {TBD_CODE_AVAILABILITY_URL}, {TBD_GRANT_INFO}.
-->

## Data availability
Data and processed features used in this study are available at {TBD_DATA_AVAILABILITY_URL}.

## Code availability
Code used for data processing, model training, and figure generation is available at {TBD_CODE_AVAILABILITY_URL}.

## Acknowledgements
Supported by {TBD_GRANT_INFO}.

## References
::: {#refs} 
:::
