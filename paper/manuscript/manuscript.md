# Non-contact acoustic sensing via the natural physical encoding of everyday objects

## Abstract
Conventional acoustic direction sensing relies on microphone arrays, limiting miniaturization and deployment in harsh or constrained environments. Here we show that everyday objects can themselves act as direction-dependent physical encoders: incident sound excites dispersive structural vibrations (frequency-dependent wave propagation through the structure) whose superposition produces a single-point spectral fingerprint measurable by a non-contact laser Doppler vibrometer. Modeling the angle–frequency response as a response matrix reveals a rapidly decaying singular-value spectrum, indicating a limited number of dominant encoding channels. This low-rank structure motivates a sparse inverse formulation: given a single measurement, we identify the most likely incidence direction from a calibrated dictionary of angle-specific spectral templates. A physics-guided neural solver replaces heuristic selection rules with learned routing while preserving consistency with the physical dictionary. In speech, we achieve {TBD_SPEECH_TOP1_ACC_PCT}% top-1 accuracy on a {TBD_N_ANGLES}-angle grid spanning {TBD_ANGLE_RANGE_DEG} with {TBD_ANGLE_STEP_DEG}° resolution. Performance remains robust under additive noise down to {TBD_SNR_MIN_DB} dB SNR. Crucially, the learned routing structure spontaneously aligns with the physical angle manifold, providing an interpretable link between data-driven inference and the underlying modal physics. Across five everyday targets, we reproduce direction-dependent fingerprints and maintain low DOA error under per-object calibration and retraining.

## Introduction
Extracting spatial information from wave fields is a recurring challenge spanning acoustics, optics, and seismology. In real environments, structural dispersion, multiple scattering, and boundary reverberation entangle propagation paths and produce seemingly chaotic time-domain signals [@rotter2017complex_media; @kuttruff2025room_acoustics]. Classical DOA estimation therefore relies on **spatial sampling** with sensor arrays and analytical inversion—beamforming and high-resolution estimators [@capon1969fkw], or subspace methods such as MUSIC and ESPRIT [@schmidt1986music; @roy1989esprit]—with standard treatments in array and microphone-array texts [@krim1996array; @vantrees2002optimum; @johnson1993array; @brandstein2001microphone_arrays; @benesty2008microphone_array]. However, the need for an aperture imposes severe constraints on size and placement, limiting integration in compact devices or harsh environments [@grumiaux2022_ssl_survey_deep_learning].

In this work, we propose a paradigm shift inspired by computational wave physics. Instead of fighting structural dispersion and scattering, we ask whether everyday objects can harness them as a **direction-dependent physical encoder**. Can an object’s material dynamics map acoustic incidence direction to a reproducible spectral fingerprint, such that a single non-contact vibration measurement is sufficient to decode DOA? Related ideas appear in time-reversed acoustics [@fink1997time; @draeger1997one], single-pixel imaging [@duarte2008single], structural health monitoring [@ing2008lamb], and transmission-matrix / wavefront-shaping views of complex media [@popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media], where complexity is treated as a computational resource. If valid, the sensing footprint reduces from an array aperture to a single optical spot, enabling compact integration where microphone arrays are infeasible (Fig. 1; Methods: Experimental setup).

Here we show that everyday objects act as reproducible **direction-dependent physical encoders** for sound (Fig. 1). We show that the resulting angle–frequency response is governed by a small number of dominant channels, motivating a **structured sparse inverse problem** on a discrete angle dictionary (Fig. 2). The encoding survives content variation but classical decoding fails catastrophically, localizing the bottleneck in the solver (Fig. 3). We show that a physics-guided unrolled solver with learnable routing overcomes the limitations of classical pursuit and decodes DOA from a single laser-vibrometry spot with high accuracy and robustness in speech conditions (Figs. 4 and 5), and the mechanism generalizes across five structurally diverse everyday objects (Fig. 6).

We employ a laser Doppler vibrometer (LDV) not merely as a convenient readout, but as a methodological necessity to preserve the integrity of the physical encoder. LDV provides non-contact vibration measurement with high bandwidth and sensitivity [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration]. In contrast, contact sensors (e.g., piezoelectric patches or accelerometers) introduce local mass and stiffness perturbations that modify measured frequency-response functions and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Recent studies suggest that a single structural vibration measurement can carry directional cues because acoustic incidence excites structure-borne waves whose response depends on incidence direction [@dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. In parallel, sparse reconstruction offers an alternative DOA viewpoint by discretizing an angular manifold and recovering a sparse angular spectrum from linear measurements [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling]. Convex relaxations such as basis pursuit and \(\ell_1\)-regularized regression provide classical baselines [@chen2001basis_pursuit; @tibshirani1996lasso], while greedy pursuit algorithms (matching pursuit and OMP) offer efficient solvers that retain interpretability [@mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free alternatives also exist in the form of acoustic vector sensors, which infer DOA from collocated pressure–particle-velocity measurements but require specialized instrumentation [@nehorai1994vector_sensor].

A central open question is whether this apparent single-point “fingerprint” reflects a reproducible physical mechanism or merely target-specific idiosyncrasies. Here we treat the target structure as a *physical encoder* that transforms incident direction into a direction-dependent superposition of dispersive modes. This view suggests two falsifiable predictions. First, the angle-to-spectrum mapping should be governed by a small number of dominant physical channels. Second, inference should remain stable under moderate noise and should transfer across structurally distinct targets if the underlying mechanism is universal.

We operationalize these predictions in three steps. First, we translate structural dynamics into an angle–frequency response model and interrogate its effective degrees of freedom via singular value decomposition. Second, we pose DOA inference as a structured sparse inverse problem on an angle-indexed dictionary and use orthogonal matching pursuit as an analytical baseline [@tropp2007omp]. Third, we derive a physics-guided unrolled solver with learnable routing and test noise robustness, routing mechanisms, and cross-material generality across five targets (Figs. 1–6; Supplementary Figs. 1–4).

**Road map.** In Results, Fig. 1 demonstrates that the structure acts as a direction-dependent spectral filter with stable, repeatable fingerprints; Fig. 2 reveals a low-dimensional physical subspace and defines the dictionary; Fig. 3 contrasts white-noise and speech stimuli to show that encoding is preserved under content variation but classical OMP decoding fails catastrophically — localizing the bottleneck in the solver; Fig. 4 introduces a physics-guided unrolled solver that overcomes this limitation; Fig. 5 shows that the learned routing structure spontaneously aligns with the physical angle manifold and remains robust under noise; and Fig. 6 tests universality across materials under per-object calibration and retraining. Confusion matrices and angle-specific routing are presented in Fig. 5d–f; full ablation variants and band-resolved diagnostics are provided in Supplementary Figs. 1–3.

**Minimal notation (measurement).** We denote the out-of-plane displacement field by \(W(x,y,\omega)\) and the laser-measured velocity by \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\). For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\) (Methods: Derivation details and assumptions).

**Minimal notation (fingerprints).** Because the observed fingerprints are magnitude statistics rather than phase, we summarize each clip by a time-averaged velocity power spectrum \(S(\omega_k;\theta)\). We then construct a log-power feature vector \(y=\phi(S)\) and its standardized version \(\tilde y\) (Methods: Signal processing pipeline).

**Dictionary + compression.** White-noise calibration yields an empirical per-angle prototype dictionary \(H=[h_1,\dots,h_E]\) (Methods: Dictionary calibration and data splits). We compress \(H\) by SVD \(H=U\Sigma V^\top\) and work in a rank-\(r\) projected space via \(z=U_r^\top \tilde y\) and \(A=U_r^\top H\) (Methods: SVD compression).

**Sparse inference.** DOA inference is posed as sparse recovery over angles with coefficient vector \(x\) and pursuit depth \(K\) (Methods: Inference algorithms).

## Results

### Everyday objects act as direction-dependent spectral filters (Fig. 1)
Everyday objects transform incident sound into direction-dependent spectral fingerprints measurable from a single non-contact point (Fig. 1b). When driven by broadband white noise, the flat source spectrum is reshaped differently at each incidence angle (Fig. 1c): the structure acts as a direction-dependent spectral filter whose transfer function H(θ, f) imprints a unique spectral signature for each direction [@rotter2017complex_media; @kuttruff2025room_acoustics]. A non-contact LDV provides the single-point readout while preserving the target’s native boundary conditions (Methods: Experimental setup). Prior work has demonstrated that contact-based vibration sensors can also carry directional cues [@dipassio2023doa_single_sensor]; the non-contact approach adopted here avoids mass-loading perturbations that modify the measured frequency-response structure [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

These fingerprints are highly repeatable: independent white-noise recordings at the same angle produce nearly identical spectra (Fig. 1d), with trial-to-trial variability (±1 s.d. shading) far smaller than the between-angle differences. This stability confirms that the spectral fingerprint is a deterministic physical property of the structure rather than a noise artifact (quantitative discriminability analysis in Fig. 3). The directional encoding is frequency-selective: different frequency bands exhibit distinct angular response patterns (Fig. 1e), consistent with the excitation of multiple dispersive structural modes whose relative amplitudes depend on incidence direction (Fig. 2b,c).

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter H(θ, f) — a flat broadband source is transformed into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°), directly demonstrating the filtering predicted in (b).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles. The near-invisible error bands confirm that the spectral fingerprint is a stable structural property; the clear between-angle separation confirms direction-dependent encoding.
e, Frequency-dependent directivity: polar plot of normalized |H(θ, f)| across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

### Spectral fingerprints arise from a low-dimensional physical manifold (Fig. 2)
The measured angle–frequency response exhibits a pronounced low-dimensional structure: the singular spectrum decays rapidly, indicating that only a small number of dominant spectral–spatial channels account for most direction-dependent variability (Fig. 2a–c). For a representative response matrix \(H\in\mathbb{R}^{F\times E}\), the effective rank is \(r\) = {TBD_SVD_R} (Fig. 2a), capturing {TBD_SVD_ENERGY_PCT}% of the singular-value energy. The full angle–frequency structure of \(H\) (Fig. 2d) and the rank-\(r\) reconstruction quality (Fig. 2e) confirm that low-rank compression retains the essential directional information.

This low-rank structure has a physical origin. Under small-amplitude dynamics, an everyday object acts as a linear system that maps incident direction into a frequency-dependent velocity response at the LDV spot. Because the response is a superposition of a limited number of dispersive structural modes (frequency-dependent wave propagation through the structure) — each with its own frequency signature and direction-dependent coupling strength — the resulting angle–frequency map is inherently low-dimensional (Methods: Derivation details and assumptions) [@ewins2000modal; @meirovitch2001fundamentals]. Even after compressing phase into magnitude statistics, the fingerprints retain this low effective dimensionality (Fig. 2a–c). The inter-angle correlation matrix (Fig. 2f) reveals a smooth manifold structure, where nearby angles share similar fingerprints — a geometric property exploited by the learned router (Fig. 5b).

This concentration motivates a compressed inference space. We construct an angle-indexed dictionary \(H\) by stacking empirical prototype fingerprints estimated from white-noise calibration — one column per candidate direction (Fig. 2d; Methods: Dictionary calibration and data splits). Given a new observation, DOA inference reduces to identifying which column of \(H\) best matches the measured fingerprint — a structured sparse inverse problem that we solve with pursuit-based algorithms (Methods: Inference algorithms). Projecting into the dominant SVD subspace focuses inference on the physically significant channels while suppressing noise (Fig. 2a–b; Methods: SVD compression).

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes; cumulative energy and DOA capacity curves quantify the information concentration.
b, Frequency-selective spectra \(|u_r(f)|\) for modes 1–3 (overlaid), showing distinct spectral peaks for each dominant channel.
c, Direction-selective polar patterns \(v_r(\theta)\) for modes 1–3 (overlaid), forming virtual directional sensing channels.
d, Full angle–frequency heatmap of the dictionary \(H\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, Rank-\(r\) reconstruction quality: original versus reconstructed fingerprint at a representative angle for ranks 3, 5, and 10, quantifying the information retained by low-rank compression.
f, Inter-angle correlation matrix of \(H\), revealing the smooth structure of the physical angle manifold — nearby angles share similar spectral fingerprints, providing the geometric foundation exploited by the learned router (Fig. 5b).

### Encoding is preserved under content variation but classical decoding fails (Fig. 3)
Before introducing the learned solver, we ask whether the directional encoding established in Figs. 1–2 survives the transition from controlled white-noise calibration to realistic speech signals — and whether a classical solver can exploit it.

Under white-noise excitation, within-angle Pearson correlation is near-perfect (r̄ = 1.000) and well separated from between-angle correlation (r̄ = 0.724; Cohen’s d = 2.83; Fig. 3a). Under speech, content variation reduces the within-angle correlation to r̄ = 0.907 but the separation remains highly significant (d = 1.95, p < 10⁻⁴; Fig. 3b). To quantify this more precisely, we compute a per-angle discriminability margin (within r − between r) for both stimulus types: white noise yields a mean margin of 0.28, while speech retains a positive margin of 0.11 at every angle (Fig. 3c). The encoding is therefore degraded but not destroyed by content variation.

Yet classical OMP decoding (\(g = |D^\top y|\), argmax over experts) fails catastrophically: 83.8% accuracy on white noise versus only 1.7% on speech — below the 2.7% chance level (Fig. 3d). The split-triangle pairwise similarity matrix (Fig. 3e) visualizes the underlying cause: white noise produces a near-identity matrix (clean separation amenable to greedy pursuit), whereas speech produces a diffuse but structured manifold that requires a geometry-aware solver. A dose-response analysis across SNR levels (Fig. 3f) confirms that content variation is the causal factor: OMP accuracy declines monotonically from 84% (pure white noise) to 4% (0 dB SNR) for the white-noise signal, and from 44% to 4% for the speech signal with babble noise (5-seed mean ± SEM). Together, these results localize the bottleneck: encoding is robust, but classical pursuit cannot exploit the residual discriminability under content variation. This motivates a physics-guided solver that can learn to navigate the angle manifold (Fig. 4).

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Encoding is preserved under content variation but classical decoding fails catastrophically.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11); speech margin is reduced but positive at all angles.
d, OMP per-angle accuracy: white noise 83.8% versus speech 1.7%, revealing a catastrophic decoding gap despite preserved encoding.
e, Split-triangle pairwise similarity matrix: lower-left = white noise (near-identity), upper-right = speech (diffuse but structured manifold).
f, Dose-response curves: OMP accuracy versus SNR for white-noise signal (blue) and speech signal with babble noise (orange, 5-seed mean ± SEM), both declining monotonically with increasing noise.

### Physics-guided routing resolves dispersive ambiguity (Fig. 4)
Dispersive fingerprints are informative but intrinsically ambiguous: multiple candidate angles can produce partially overlapping spectral responses. A classical greedy solver — orthogonal matching pursuit (OMP) — selects the dictionary entry most correlated with the current measurement residual [@tropp2007omp; @tropp2004greed]. As shown in Fig. 3d, single-shot OMP achieves near-chance accuracy on speech (1.7%), despite the preserved encoding discriminability established in Fig. 3a–c. The dose-response analysis (Fig. 3f) confirms that content variation is the causal factor. In complex media, content-induced spectral variation causes OMP to spread selection mass off-diagonal (Fig. 5c,e), motivating a solver that can learn from the structure of the angle manifold.

We address this by deriving a physics-guided neural solver that unrolls the iterative pursuit algorithm into a trainable network (Fig. 4a; Methods: Inference algorithms) [@gregor2010lista; @monga2021unrolling]. At each stage, the solver computes how well each dictionary entry matches the remaining unexplained signal — a physically grounded correlation step. Instead of selecting the single best match (as OMP does), a learned attention mechanism produces a soft weighting over all candidate directions, informed by the full dictionary structure. This weighting gates a sparse update, after which the explained portion is subtracted to maintain consistency with the physical dictionary (Methods: Attention routing equations). The number of iterative selection steps \(K\) (the pursuit depth) provides a controlled residual-correction budget under noise and feature mismatch, and is shared between OMP and the learned solver for fair comparison.

Training converges stably within 20 epochs (Fig. 4b). A systematic ablation at clean SNR (Fig. 4c) confirms that each component — transformer routing, type bias, and physics-grounded gating — contributes measurably, with performance degrading progressively as components are removed. The solver achieves near-uniform accuracy across the full angular grid (Fig. 4d), indicating no systematic blind spots.

When the learned weights concentrate on a single entry, the update reduces to classical greedy selection; when they spread across nearby angles, the solver exploits the smooth structure of the angle manifold to regularize its choices. Evidence that this routing mechanism aligns with physical structure — rather than learning an arbitrary pattern — is presented in Fig. 5. Sustained accuracy under noise (Fig. 5a) and sharper selection statistics (Fig. 5b,c,e; Supplementary Figs. 1–3) confirm that the learned routing provides a principled advantage over heuristic selection. This sparse-reconstruction viewpoint parallels classical sparse DOA methods that discretize a propagation manifold and solve for a sparse angular spectrum [@malioutov2005sparse_doa; @baraniuk2007compressive_sensing].

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | Physics-guided deep unrolled network with attention-based gating.**
a, Architecture: at stage \(t\), the residual \(r_t\) is correlated with the physical dictionary \(A\). A transformer encoder generates routing weights that gate sparse updates \(\Delta x_t\), enforcing residual consistency \(r_{t+1}=r_t-A\Delta x_t\).
b, Training convergence: total and classification loss decrease steadily over 20 epochs; the vertical dashed line marks the best validation epoch.
c, Clean-condition ablation: strip chart comparing the full model against six ablation variants (individual seeds shown as dots, horizontal bars indicate means), demonstrating that the transformer routing and type-bias components are essential for high accuracy.
d, Per-angle accuracy profile: the 37-point bar chart confirms near-uniform performance across the full angular grid (mean accuracy {TBD_MEAN_ACC}), with angles below the mean highlighted.

### Learned routing aligns with physical structure and resists noise (Fig. 5)
The central finding of this work is that the learned attention router recovers structure already present in the physical dictionary, rather than learning an arbitrary pattern. We demonstrate this alignment and its functional consequences through three complementary analyses.

The physical angle manifold exhibits characteristic correlations in the dictionary \(H\): nearby angles share spectral features, producing a banded correlation structure. Strikingly, the learned query–key (QK) correlation of the attention router mirrors this physical structure with even sharper diagonal locality (Fig. 5b). The Pearson correlation between the upper-triangular entries of the \(H\)-correlation and QK-correlation matrices is \(r\) = {TBD_HCORR_QK_CORR} (\(p\) < {TBD_HCORR_QK_P}), confirming that the router has learned to respect the geometry of the physical angle manifold rather than fitting an unconstrained attention pattern. This alignment is the core interpretability result: it shows that the network’s internal routing reflects genuine physical structure.

The consequences of this alignment are visible in the selection statistics (Fig. 5c). Across the full angle grid, OMP spreads selection mass over a persistent off-diagonal set of dictionary entries, reflecting the spectral overlap that confuses greedy pursuit. By contrast, the physics-aware model concentrates selection probability sharply along the diagonal, indicating that it selects the correct direction for each incidence angle with high consistency. This concentration mirrors the sharpened QK structure and demonstrates that manifold-aligned routing translates directly into cleaner direction selection.

Furthermore, this structural advantage confers noise robustness (Fig. 5a). On speech recordings, the physics-aware model achieves {TBD_SPEECH_TOP1_ACC_PCT}% top-1 accuracy at clean SNR, with graceful degradation to {TBD_ACC_SNR10_PCT}% / {TBD_ACC_SNR5_PCT}% / {TBD_ACC_SNR0_PCT}% at 10 / 5 / 0 dB SNR (averaged over {TBD_N_INDEP_RUNS} independent runs). Removing the transformer degrades accuracy to {TBD_ACC_ABL_NO_TRANSFORMER_PCT}%, while the analytical OMP baseline falls further behind under noise (Fig. 5a; Methods). Detailed ablation results and architectural comparisons are provided in Methods. Confusion matrices (Fig. 5d), angle-specific routing distributions (Fig. 5e), and per-angle improvement quantification (Fig. 5f) demonstrate the routing effect at both macro and micro scales. Full ablation variants and band-resolved diagnostics further corroborate the physical mechanism (Supplementary Figs. 1–3).

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | The learned router mirrors physical structure and maintains robust decoding under noise.**
a, SNR degradation curves for the physics-aware model, no-transformer ablation, and analytical OMP baseline, showing graceful degradation under additive noise.
b, Correlation structure of the physical dictionary \(H\) (top) and the learned QK attention map (bottom), revealing that the router recovers the geometry of the angle manifold — the core interpretability finding.
c, All-angle selection-probability heatmaps comparing OMP (diffuse off-diagonal mass, top) with the physics-aware model (sharply diagonal, bottom), demonstrating that structure-aligned routing concentrates selection on the correct direction.
d, Confusion matrices for the baseline model (left) and no-transformer ablation (right), normalized to row-wise probabilities. The baseline exhibits a sharply diagonal pattern, confirming that the learned router assigns the correct direction for nearly all angles.
e, Angle-specific routing distributions at two representative directions (55° and 100°): the baseline concentrates mass on the correct atom and suppresses off-axis peaks, whereas the no-transformer ablation shows broader, less decisive distributions.
f, Per-angle diagonal concentration: P(correct) for each of the 37 angles comparing the baseline with the no-transformer ablation, quantifying the fraction of angles that benefit from transformer routing. The shaded region highlights the per-angle improvement.

The angle-specific evidence in panels d–f complements the macro-level structure alignment in panels b–c: together they demonstrate that manifold-aligned routing operates consistently from the global angle manifold down to individual directions. Additional ablation variants and band-resolved diagnostics are provided in Supplementary Figs. 1–3. A natural question is whether this mechanism — low-rank modal encoding decoded by manifold-aligned routing — is specific to a single object or reflects a universal physical principle.

### Physical encoding is universal across everyday materials (Fig. 6)
Across five targets with distinct material properties (damping, stiffness) and geometries—acrylic, paper cup, wood, cardboard, and a laptop shell (Fig. 6a)—we consistently observe direction-dependent dispersion signatures (Fig. 6b). Under a per-object calibrate-and-retrain protocol (Methods: Dictionary calibration and data splits), the physics-aware model maintains low DOA error (Fig. 6c). The RMSE is {TBD_RMSE_RANGE_DEG}°. In contrast, analytical OMP degrades substantially on the most complex targets. Its RMSE exceeds {TBD_RMSE_OMP_COMPLEX_DEG}° (Fig. 6c).

**Mechanism of Universality.** While specific material parameters (stiffness \(D_p\), density \(\rho\)) shift resonance frequencies and mode shapes, the fundamental **linear superposition principle** governs all targets. The per-band SVD spectra (Fig. 6d) show consistent rapid singular-value decay across frequency bands, confirming that the low-rank structure is a shared physical property rather than a frequency-specific artifact. Band-resolved routing accuracy (Fig. 6e) further demonstrates that the encoding mechanism is frequency-distributed: the physics-aware model outperforms OMP in every frequency band. Calibration yields an object-specific dictionary \(H\) that captures each object’s dispersive fingerprints, and retraining learns an inverse mapping that exploits the shared sparse-superposition logic. This supports the interpretation that the solver learns the physical logic encoded by \(H\) rather than memorizing individual trials (Fig. 6b–e).

![](../figures/fig06_universality.jpg)

**Fig. 6 | Universal physical encoding across diverse materials.**
a, The five target objects spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell).
b, Representative dictionary heatmaps for each material, showing shared dispersion-signature structure despite differing physical properties.
c, Cross-material RMSE comparison: the physics-aware model maintains low DOA error across all materials, while analytical OMP degrades on complex targets.
d, Per-band SVD spectra: normalized singular-value decay across five frequency bands (full band + four sub-bands), demonstrating consistent low-rank structure regardless of frequency range — a physics-level (not accuracy-level) indicator of universality.
e, Band-resolved routing consistency: diagonal accuracy per band comparing OMP versus physics-aware AI, confirming that the encoding mechanism is frequency-distributed rather than tied to a single resonance artifact.

## Discussion
These results establish a complementary sensing paradigm in which passive everyday structures function as naturally randomized **physical encoders** for sound. A single, non-contact vibration readout contains enough information to decode DOA because the object converts incidence direction into a reproducible spectral fingerprint (Fig. 1), shifting the sensing footprint from an array aperture to a single optical spot.

**Mechanism (physics).** Directional information arises because an incident acoustic field excites structure-borne waves whose coupling depends on incidence direction and boundary conditions. In the modal view, direction modulates the participation weights \(\alpha_m(\theta)\), consistent with the interpretive separable form \(Y(\omega;\theta)\approx \sum_m s_m(\omega)\alpha_m(\theta)\) introduced in Results (Fig. 2b). This step assumes approximately linear time-invariant dynamics over each analysis window and is used here to explain the observed fingerprints rather than to identify individual modes. White-noise calibration operationalizes this mapping by estimating per-angle prototypes \(h_e\), which serve as empirical “atoms” that encode how the object’s dispersive dynamics respond to each direction (Fig. 1b).

**Mechanism (low effective rank).** The rapid decay of the singular spectrum (Fig. 2a) indicates that the measured angle–frequency response is governed by a small number of dominant channels, consistent with eigenchannel interpretations of complex media. This low effective rank explains why projecting into the dominant subspace improves robustness: it retains the physically informative variation while suppressing directions that contribute weakly or inconsistently. As damping increases, modal overlap grows and directional identifiability can degrade because distinct directions become less separable in the compressed channel space [@ewins2000modal; @inman2013engineering_vibration].

**Mechanism (routing on an angle manifold).** Greedy pursuit can fail when content-induced spectral overlap causes spurious off-axis correlations (Fig. 5c,e). The physics-guided unrolled solver addresses this by learning a soft routing distribution \(w_t\) that regularizes selection using context across the entire angle manifold, while preserving residual-consistent updates anchored to the physical dictionary (Fig. 4a). The emergent near-diagonal learned structure (Fig. 5b), the sharply diagonal all-angle selection maps (Fig. 5c), the confusion matrices showing near-perfect diagonal concentration (Fig. 5d), the stronger correct-atom peaks at representative directions (Fig. 5e), and the band-resolved persistence of the same mechanism (Fig. 6e; Supplementary Figs. 1–3) together indicate that the solver exploits manifold smoothness—nearby angles share similar dispersive structure—to stabilize inference under noise and nonstationary speech (Fig. 5a).

**Scientific frontiers and outlook.** Our current pipeline requires per-object calibration to construct an object-specific dictionary \(H\) and retraining to learn the inverse mapping (Fig. 6), highlighting a frontier question: which invariances of the angle–frequency manifold are shared across objects, and which are inherently object-specific? A second frontier is to quantify information-theoretic limits of single-point decoding as angle fingerprints become less separable (Fig. 2a), and to determine how mounting conditions and LDV spot placement trade off robustness versus sensitivity. Extending the framework to multi-source and moving-source regimes will require modeling how fingerprints superpose and evolve across time windows, and identifying when the residual-correction depth \(K\) captures meaningful additional structure versus overfitting.

## Methods
### Software and hardware
All processing and learning were implemented in Python. Neural models were implemented in PyTorch and trained on Apple Silicon using Metal Performance Shaders (MPS) in the `trl-training` conda environment. Unless otherwise stated, random seeds were fixed to {TBD_RANDOM_SEED} (NumPy and PyTorch).

### Experimental setup
Experiments were conducted in a controlled acoustic environment with a single loudspeaker source at a radius of {TBD_RADIUS_M} m. We define \(\theta=0^\circ\) as {TBD_ANGLE_ZERO_REFERENCE} and increase \(\theta\) in {TBD_ANGLE_SIGN_CONVENTION} direction. The source angle was varied on a {TBD_N_ANGLES}-point grid covering {TBD_ANGLE_RANGE_DEG} with step {TBD_ANGLE_STEP_DEG}°. We measured out-of-plane surface velocity using a {TBD_LDV_MODEL} laser Doppler vibrometer (LDV). All downstream processing uses waveforms resampled to {TBD_FS_HZ} Hz. The LDV spot location was {TBD_LDV_SPOT_LOCATION} and the object mounting/boundary condition was {TBD_OBJECT_MOUNTING_CONDITION}.

We used two excitation regimes: (i) white-noise playback for dictionary calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end DOA decoding and robustness experiments. Unless otherwise stated, each clip duration is {TBD_TRIAL_DURATION_S} s.

**Object Selection.** We selected five objects (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell) to span a range of **Q-factors** (damping) and structural complexity, testing the limits of the encoding mechanism.

### Dictionary calibration and data splits
The angle dictionary \(H\) is constructed exclusively from white-noise calibration recordings. For each target object, we acquire a white-noise dataset organized by angle and partition it into (i) a **calibration subset** used solely to estimate \(H\) and (ii) a disjoint **evaluation subset** used to quantify fingerprint repeatability/separability (Fig. 1) and verify dictionary validity. The calibration subset contains {TBD_WN_CALIB_CLIPS_PER_ANGLE} clips per angle selected by {TBD_WN_SPLIT_RULE}; the remaining {TBD_WN_EVAL_CLIPS_PER_ANGLE} clips per angle are reserved for evaluation. Calibration clips are never used for training or evaluation of the DOA decoder.

For speech decoding (Fig. 5; Supplementary Figs. 1–3), we train the decoder on a speech dataset split into train/validation/test sets stratified by angle using {TBD_SPEECH_SPLIT_RULE} (counts: {TBD_SPLIT_TRAIN} / {TBD_SPLIT_VAL} / {TBD_SPLIT_TEST} clips). All speech results use the fixed \(H\) estimated from white-noise calibration only.

For cross-material experiments (Fig. 6), we repeat the full pipeline per target: acquire object-specific white-noise calibration data, estimate that object’s \(H\), and retrain a separate model for that object. Universality is therefore assessed at the level of a shared physical mechanism and inference recipe, not by reusing weights across objects.

### Signal processing pipeline
**Definition ladder (waveform → dictionary).** For a clip at direction \(\theta\), the LDV provides an out-of-plane velocity waveform \(v[n]\) (m/s). We compute an STFT \(V[k,t]\), form a time-averaged power spectrum estimate \(\widehat S(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2\), and restrict it to a frequency band to obtain a band-limited fingerprint. Features are defined as \(y=\phi(\widehat S)\), standardized to \(\tilde y\), and averaged across calibration trials to form per-angle prototypes \(h_e\) and the dictionary \(H=[h_1,\dots,h_E]\).

To extract stable spectral features from the measured velocity time series \(v[n]\) (sampled at rate \(f_s\) = {TBD_FS_HZ} Hz), we use a fixed short-time Fourier transform (STFT) pipeline [@allen1977stft]. With a Hann window \(w[m]\) of length \(N_\mathrm{win}\) = {TBD_STFT_WIN_SAMPLES} samples and hop size \(N_\mathrm{hop}\) = {TBD_STFT_HOP_SAMPLES} samples, we compute
$$
V[k,t] = \sum_{m=0}^{N_\mathrm{win}-1} v[tN_\mathrm{hop}+m]\, w[m]\, e^{-i2\pi km/N_\mathrm{FFT}},
$$
where \(N_\mathrm{FFT}\) is the FFT size (set to {TBD_NFFT}) and \(k\) indexes frequency bins. We form a time-averaged power spectrum
$$
P[k] = \frac{1}{T}\sum_{t=1}^{T} \left|V[k,t]\right|^2,
$$
apply a band mask \(f_k\in[f_\min,f_\max]\) with \(f_\min\) = {TBD_FREQ_MIN_HZ} Hz and \(f_\max\) = {TBD_FREQ_MAX_HZ} Hz to obtain \(P_\mathrm{band}\in\mathbb{R}^{F}\) (F = {TBD_F_BINS}). We interpret \(P_\mathrm{band}[k]\) as a finite-sample estimator of the band-limited power fingerprint \(S(\omega_k;\theta)\) used in Results. The feature vector is
$$
\widehat S(\omega_k;\theta) := P_\mathrm{band}[k],
\qquad
y[k] = \phi\!\left(\widehat S(\omega_k;\theta)\right) = \log_{10}\!\left(\widehat S(\omega_k;\theta) + \epsilon\right).
$$
with log clamp \(\epsilon\) = {TBD_EPS}.
This definition makes explicit the link between the physical complex response \(Y(\omega;\theta)\) and the observable fingerprint: we use a time-averaged magnitude statistic \(\widehat S\) (no phase) to obtain stable, repeatable fingerprints. In Results, we write \(S\) for the underlying power-spectrum fingerprint, with \(\widehat S\) denoting its finite-sample estimator.

We estimate per-frequency standardization statistics \((\mu_k,\sigma_k)\) from the white-noise calibration subset and define the standardized feature
$$
\tilde y[k] = \frac{y[k]-\mu_k}{\sigma_k+\epsilon_z}.
$$
with z-score clamp \(\epsilon_z\) = {TBD_ZSCORE_EPS}.
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
Given the projected feature \(z\in\mathbb{R}^r\) and projected dictionary \(A\in\mathbb{R}^{r\times E}\), we solve the sparse inverse problem \(z\approx Ax\) under a \(\lVert x\rVert_0\le K\) constraint using OMP as an analytical baseline and an unrolled, physics-guided neural solver (Results, Fig. 4).

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

**Physics-guided unrolling.** We unroll \(K\) pursuit stages into a network, replace the discrete selection rule with learnable attention-based routing, and retain residual-consistent updates (Results, Fig. 4). This yields a differentiable solver whose routing weights can be inspected as mechanistic evidence (Results, Fig. 5b,c; Supplementary Figs. 1–3).

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

**Attention routing equations (Fig. 4).** In the rank-\(r\) projected space, at stage \(t\) we start from residual \(r_t\in\mathbb{R}^r\) and compute correlations \(g_t=A^\top r_t\in\mathbb{R}^E\). We parameterize routing weights over atoms via dot-product attention [@vaswani2017attention]. A query is computed from the current state,
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
The physics-guided unrolled network was implemented in PyTorch [@paszke2019pytorch]. The architecture consists of \(K\) unrolled pursuit stages with a transformer router (embedding dimension \(d_\mathrm{model}\), \(h\) attention heads, and {TBD_N_LAYERS} encoder layers; Results, Fig. 4). We use \(K\) = {TBD_K_STAGES}, \(d_\mathrm{model}\) = {TBD_D_MODEL}, and \(h\) = {TBD_N_HEADS}. At stage \(t\), the current residual \(r_t\) is correlated with the dictionary to form a physical match \(g_t=A^\top r_t\). The router uses scaled dot-product attention to produce routing weights \(w_t\), and the stage update uses residual-consistent subtraction \(r_{t+1}=r_t-A\Delta x_t\). Expert-level logits are obtained by {TBD_LOGIT_CONSTRUCTION}, and the predicted direction is \(\hat\theta=\arg\max_e p[e]\).

**Training.** We trained the unrolled network using {TBD_TRAIN_LOSS_DESCRIPTION}, optimized with {TBD_OPTIMIZER} (learning rate \(\eta\) = {TBD_LR}, weight decay \(\lambda\) = {TBD_WEIGHT_DECAY}) for {TBD_EPOCHS} epochs with batch size {TBD_BATCH_SIZE} and random seed {TBD_RANDOM_SEED}. Speech clips were split deterministically stratified by angle ({TBD_SPEECH_SPLIT_RULE}; {TBD_SPLIT_TRAIN} / {TBD_SPLIT_VAL} / {TBD_SPLIT_TEST} clips).

**Ablations (Fig. 5a).** We report three ablations with all other components held fixed. **No-transformer** replaces the transformer router with {TBD_ABL_NO_TRANSFORMER_DEF}. **Fixed heuristic** replaces learned routing with {TBD_ABL_FIXED_HEURISTIC_DEF}. **Dense routing** uses {TBD_ABL_DENSE_ROUTING_DEF}. Each ablation uses the same dictionary construction and evaluation protocol as the full model.

Noise robustness experiments add zero-mean white noise at SNR levels {TBD_SNR_LEVELS_DB} in the time domain:
$$
v_\mathrm{noisy} = v + \alpha \xi,\quad \xi\sim\mathcal{N}(0,1),\quad \mathrm{SNR}=10\log_{10}\frac{\lVert v\rVert_2^2}{\lVert \alpha \xi\rVert_2^2}.
$$
The SNR sweep in Fig. 5a uses the same feature extraction and dictionary construction for each noise level.

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
We report mean ± s.d. over \(n\) = {TBD_N_REPEATS} independent replicates (definition per experiment: {TBD_REPLICATE_DEFINITION}). When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure/panel.

## Data availability
Data and processed features used in this study are available at {TBD_DATA_AVAILABILITY_URL}.

## Code availability
Code used for data processing, model training, and figure generation is available at {TBD_CODE_AVAILABILITY_URL}.

## Acknowledgements
Supported by {TBD_GRANT_INFO}.

## References
::: {#refs} 
:::
