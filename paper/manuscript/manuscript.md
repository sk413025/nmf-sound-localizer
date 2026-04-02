# Object-specific calibration enables single-point vibrometric readout of sound direction in an acrylic plate

## Abstract
Matched calibration over a 37-angle half-plane benchmark (0-180° in 5° steps) shows that a passive acrylic plate preserves sound-direction information in a single non-contact laser Doppler vibrometer (LDV) readout. The measured vibration fingerprints are repeatable and occupy a compact angle-ordered structure, with neighboring calibrated directions remaining locally similar. Replacing the white-noise calibration probe with held-out speech weakens this structure but does not erase it: within-angle similarity remains higher than between-angle similarity across the benchmark. The main readout difficulty therefore arises when nearby calibrated directions overlap, because speech broadens that local ambiguity and destabilizes immediate one-angle commitment. Readout is strongest when nearby evidence is pooled before commitment, consistent with the measured local ordering of the calibrated fingerprints. In an exploratory five-object screen, structured directional fingerprints and low-rank continuity recur under matched object-specific calibration, but the data support only bounded recurrence within this small sample. These results show that calibrated single-point vibrometric direction readout is possible in this acrylic-plate benchmark and identify local-overlap handling as the operational constraint.

## Introduction
Incident sound direction can be encoded in how a passive object vibrates under directional forcing. That observation motivates the question studied here: after matched calibration, can one fixed vibrometric readout from an acrylic plate serve as a reliable directional measurement? In this framing, the object itself preserves directional information in a measurable vibration response.

Our working hypothesis is that direction changes how sound couples into an object's vibration modes, and that this change can be read as a repeatable spectrum at one fixed point. Incident sound first produces an object response, the LDV records that response as a measured fingerprint, matched calibration organizes those fingerprints into a template matrix \(H\), and readout must then distinguish nearby calibrated directions when they overlap. We test that pathway in a controlled benchmark and ask two linked questions: does the calibrated response occupy a compact measured structure, and can that structure still be read once source content changes?

Several lines of prior work motivate this possibility, but they do not yet show that ordinary objects provide a reproducible calibrated measurement pathway. Single-sensor studies have reported direction-dependent transfer functions in engineered scatterers and in structure-borne measurements [@elbadawy2018lego_doa; @dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Sparse reconstruction offers one useful language for inference once a calibrated angle grid has been defined [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling; @chen2001basis_pursuit; @tibshirani1996lasso; @mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free localization is also possible with acoustic vector sensors, but those require specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. What remains unresolved is whether everyday targets themselves can supply repeatable directional fingerprints that are systematic enough to support matched calibration and stable readout.

To test that possibility without mechanically perturbing the target, we use a laser Doppler vibrometer (LDV) as a non-contact readout. LDV provides high-bandwidth, high-sensitivity vibration measurement [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration], whereas piezoelectric patches and accelerometers can alter local mass, stiffness, and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. The broader conceptual backdrop comes from wave systems in which scattering and disorder are harnessed rather than suppressed, including time-reversal acoustics, single-pixel imaging, transmission-matrix optics, and randomized metamaterials [@fink1997time; @draeger1997one; @duarte2008single; @ing2008lamb; @gigan2022imaging_computing_disorder; @popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media; @hoang2021single_pixel_doa; @jiang2020randomized_metamaterial]. Here we ask whether an ordinary object can supply enough direction-dependent vibration structure for calibrated single-point readout.

## Results

### The acrylic plate yields repeatable direction-dependent vibration fingerprints (Fig. 1)
We begin with the acrylic plate used throughout the main benchmark. Under broadband white-noise excitation, the measured LDV spectrum changes systematically with source angle: the same flat input is reshaped into distinct output spectra at different directions (Fig. 1c). Incident sound therefore drives direction-dependent plate response, and the LDV samples that response at one fixed point. Because the readout is optical and non-contact, it preserves the target's native boundary conditions and avoids sensor loading that could otherwise alter the measured response [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

These fingerprints are also repeatable. Independent white-noise recordings at the same angle produce nearly identical spectra (Fig. 1d), and their trial-to-trial variation is much smaller than the differences across angles. Different frequency bands then show different angular patterns (Fig. 1e), consistent with multiple dispersive vibration modes contributing with angle-dependent weights (Fig. 2b,c) [@rotter2017complex_media; @kuttruff2025room_acoustics]. The next question is whether these repeatable angle-specific spectra occupy an organized measured space rather than a collection of unrelated traces.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°-180° for four frequency bands (0.3-0.5, 0.5-1, 1-2, 2-3 kHz), showing that each band carries a distinct directional response pattern.

### Calibration fingerprints form a compact angle-ordered space (Fig. 2)
We next ask whether those repeatable fingerprints share a lower-dimensional organization. The singular spectrum of the centered-magnitude analysis matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|
$$

decays rapidly (Fig. 2a-c): the first six modes capture 80.3% of the energy and eight capture 85.1%. Across the full angle-frequency heatmap in Fig. 2d, the calibrated fingerprints vary systematically with angle. The rank-truncation curves in Fig. 2e show that a small retained basis already reconstructs them well, and the inter-angle similarity matrix in Fig. 2f places neighboring directions close in fingerprint space. The measured fingerprints therefore occupy a compact angle-ordered space with strong local continuity across nearby directions.

A simple modal interpretation is consistent with that organization. Under small-amplitude dynamics, the single-point velocity response can be written as a superposition of dispersive structural modes, each with its own spectral pattern \(s_m(\omega)\) and direction-dependent coupling \(\alpha_m(\theta)\):

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the complex frequency response at the LDV measurement point for incidence direction \(\theta\), and the sum runs over a limited set of contributing modes (Supplementary Methods 1) [@ewins2000modal; @meirovitch2001fundamentals]. In this view, changing direction mainly changes the coupling weights \(\alpha_m(\theta)\) across a limited set of dispersive modes, so nearby angles reuse related spectral components. The mode spectra in Fig. 2b,c and the near-diagonal similarity structure in Fig. 2f are consistent with that local reuse.

This low-rank observation gives a simple consequence for readout. Once calibration has organized the measured fingerprints into the template matrix \(H\) on a discrete angle grid, ambiguity among nearby directions should be local rather than global. We therefore next test whether held-out speech preserves that local structure, and if so, whether a decoder can still read those calibrated fingerprints when nearby directions overlap.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. The cumulative curve rises quickly across the 37-angle grid: six modes capture 80.3% of the energy and eight capture 85.1%. The overlaid direction-decoding trace is included as a visual comparison.
b, Frequency-selective spectra \(|u_r(f)|\) for representative Modes 1, 2, and 6. These traces show three reusable spectral patterns in the compressed representation.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative Modes 1, 2, and 6, showing how those same modes vary across 0°-180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-mode regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that neighboring angles remain close in fingerprint space, revealing the local angle ordering later compared with the guided neighborhood map in Fig. 5c.

### Speech preserves directional structure but destabilizes immediate commitment (Fig. 3)
We next ask whether the compact structure identified in Figs. 1 and 2 survives a change in source content. White noise is the calibration probe used to estimate the template matrix \(H\), so it reveals the direction-dependent vibration response with minimal source structure. Held-out speech poses the harder test because it introduces structured spectrotemporal variation that was absent during calibration. Keeping the calibrated reference templates fixed therefore asks whether directional structure remains readable when the source waveform changes.

Under white-noise excitation, within-angle Pearson correlation is near-perfect and clearly separated from between-angle correlation (\(\bar{r} = 1.000\) versus 0.724; \(d = 2.83\); Fig. 3a). Under speech, within-angle similarity drops to 0.907, but it remains higher than between-angle similarity (0.798), with \(d = 1.95\) (Fig. 3b). The per-angle discriminability margin stays positive at every angle, although its mean decreases from 0.28 under white noise to 0.11 under speech (Fig. 3c). Speech therefore weakens the directional structure without erasing it.

The failure appears at the first hard choice. The correlation-based greedy diagnostic performs well on most white-noise fingerprints but drops to near chance on held-out speech clips (Fig. 3d). The pairwise similarity map explains why: white noise yields a near-identity pattern, whereas speech spreads similarity across neighboring calibrated angles while retaining local ordering (Fig. 3e). Once several nearby reference templates become jointly plausible, immediate one-angle commitment becomes unstable even though the directional code is still present.

Noise worsens the same failure mode. Greedy accuracy declines monotonically as additive noise increases in both the white-noise and speech conditions (Fig. 3f). Figure 3 therefore sets the hinge for the remainder of the paper: speech does not remove directional information from measured fingerprint space, but it exposes a local-overlap readout problem that must be resolved before one direction can be reported reliably.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Directional structure persists under speech, but first-choice matching fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based first-choice diagnostic traces for white noise and speech, shown with light clip-level uncertainty bands: the diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response curves: correlation-based greedy diagnostic accuracy versus SNR for white-noise signal (blue, clip-level standard error of the mean (SEM) shading) and speech signal with babble noise (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading), both declining monotonically with increasing noise.

### One local update resolves nearby overlap and reduces the residual (Fig. 4)
Figure 4 illustrates the overlap constraint exposed by speech at the level of one shared 70° exemplar. The initial match is broad across nearby calibrated directions, consistent with the local ambiguity seen in Fig. 3e, but one guided local update concentrates that support around the correct neighborhood and leaves less unexplained signal [@gregor2010lista; @monga2021unrolling]. The compact callouts summarize the same shift numerically: the fraction of the match that falls within 15° of the true direction rises from 0.18 to 0.98, while the remaining residual falls from 1.00 to 0.48.

The panels trace a local-overlap mechanism directly: the readout moves from a broad neighborhood response to a more concentrated one. They show how a readout can exploit the measured local ordering of the calibrated fingerprints instead of forcing an immediate isolated choice. The clean-condition comparison in Fig. 4d is consistent with that interpretation: the model that pools local evidence before commitment is the most accurate in this restricted setting.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | One local update resolves nearby overlap and reduces the residual.**
a, Mechanism strip illustrating the broad-match -> local-gate -> cleaner-residual sequence.
b, Shared 70° exemplar: the initial broad match is concentrated into one local neighborhood around 70°.
c, After one local step, less signal remains once the overlapping neighborhood has been updated. Compact callouts report the same shift numerically: the fraction of the match within 15° of the true direction rises from 0.18 to 0.98, while the residual falls from 1.00 to 0.48.
d, Clean-condition comparison across the guided solver, router-bypass, OMP baseline, and dense routing variants over the same five-seed sweep (individual seeds shown as dots; points with horizontal bars indicate mean \(\pm\) s.e.m.). The model that pools local evidence before commitment is most accurate in this restricted setting.

### Overlap-aware readout follows the measured local structure across the benchmark (Fig. 5)
We next test whether benchmark-scale performance follows the same local structure seen in the calibrated fingerprints. The measured template matrix \(H\) retains a banded correlation pattern, and the learned neighborhood map places most of its weight along that same local angle ordering rather than across distant angles (Fig. 5c). The prediction summaries follow the same pattern: outputs stay most concentrated near the correct neighborhood when nearby evidence is pooled before commitment, both in the confusion-family block and in representative conditional profiles (Fig. 5b,d).

The same pattern appears across noise levels and across all 37 angles: the overlap-aware readout remains the most accurate within this comparison, whereas immediate or weakly constrained choices degrade more sharply (Fig. 5a,e). Figure 5 therefore provides benchmark-scale support for the view that overlap-aware readout matches the measured local structure of the calibrated fingerprints. The decoder comparison is supportive because it tests that consequence across the full benchmark.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | Overlap-aware readout follows the measured local structure across the benchmark.**
a, SNR degradation curves comparing four decoder variants across additive-noise levels; the overlap-aware readout degrades least as noise increases.
b, Row-normalized confusion-family block across the four decoders. The guided solver stays most concentrated near the diagonal, whereas the less locally constrained alternatives show broader off-axis leakage or collapse toward a preferred output mode.
c, Correlation structure of the measured template matrix \(H\) (top) and the guided neighborhood map (bottom), showing that the learned map places most weight near the same diagonal angle ordering seen in the measured fingerprints.
d, Angle-specific conditional output distributions at four representative directions (55°, 70°, 95°, and 100°): the guided solver produces tighter local prediction profiles, whereas router-bypass shows broader off-axis leakage.
e, Per-angle decoder accuracy: five-seed clean mean \(P(\mathrm{correct})\) across the 37 measured angles, shown as a 3-angle centered moving-average display with light \(\pm 1\) s.e.m. shading, comparing the guided solver, router-bypass, OMP baseline, and dense routing. The overlap-aware readout retains the highest clean mean accuracy overall, whereas dense routing remains near chance across almost the entire angle set.

### A five-object screen shows bounded recurrence under matched calibration (Fig. 6)
We finally ask a narrower breadth question: does matched calibration reveal the same response-to-readout structure beyond the acrylic plate in a small five-object screen? Each object yields structured angle-frequency fingerprints in its own calibrated template matrix \(H\) (Fig. 6a,b). Applying the same centered-magnitude SVD view used in Fig. 2 again shows early energy capture across the five objects (Fig. 6c). The extension is therefore bounded: matched calibration can reveal compact directional structure beyond one plate, but the paper's primary evidence remains the acrylic benchmark.

The same screen also shows why the breadth claim should remain secondary. Each object remains above chance, but normalized response energy does not monotonically track Top-1 accuracy across the five objects (Fig. 6d). The per-object frequency summaries further show that the informative band can shift across objects (Fig. 6e). This five-object screen therefore supports recurrence under matched calibration, not transfer across objects or a validated material law.

![](../figures/fig06_universality.jpg)

**Fig. 6 | Matched calibration reveals bounded recurrence across an exploratory five-object screen.**
a, Five target objects in screening order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects despite different response patterns.
c, Low-rank continuity. Applying the same centered-magnitude SVD view used in Fig. 2 to each object's template matrix again shows early energy capture across the five-object screen.
d, Screening performance across the five objects. All five objects remain above chance under matched calibration, and the energy-versus-accuracy comparison shows that Top-1 screening accuracy does not monotonically track overall response energy across this sample. The accompanying Top-1 confidence intervals summarize screening uncertainty while preserving that mismatch across objects.
e, Frequency structure across objects. Per-object spectra and directional band profiles indicate that the informative frequency band can shift across objects.

## Discussion
The main advance here is that matched calibration turns the acrylic plate's direction-dependent response into a readable single-point measurement pathway. At one fixed LDV position, the plate produces repeatable measured fingerprints, those fingerprints organize into a compact calibrated template matrix \(H\), and held-out speech still preserves directional structure even as it broadens local overlap [@gigan2022imaging_computing_disorder; @rotter2017complex_media]. The bounded conclusion is therefore clear: in this acrylic-plate benchmark, sound direction can be read from one calibrated vibrometric measurement, provided the readout resolves neighboring calibrated directions before forcing a single-angle choice.

The five-object screen serves a narrower role. Under matched object-specific calibration, each object again yields structured fingerprints, early energy capture in the centered-magnitude SVD view, and above-chance screening performance. At the same time, normalized response energy, informative bands, and Top-1 performance differ across objects, so this extension remains an exploratory breadth result and supports bounded recurrence under matched calibration only.

These findings connect to studies that use engineered disorder or metasurfaces as computational wave encoders [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa], but here matched calibration exposes usable direction-dependent vibration structure in an unmodified acrylic plate. The work also complements single-sensor localization with embedded microphones and contact-mounted piezoelectric sensors [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming], because non-contact LDV readout preserves native boundary conditions and avoids sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Several limitations bound the present claims. The benchmark uses a half-plane grid (0-180°) at 5° spacing, a controlled acoustic environment, a single static source, and per-object matched calibration. The LDV is a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. Future work should test whether the observed local angle ordering can be predicted or adapted with less calibration, how performance changes under stronger damping, reverberation, multiple simultaneous sources, or moving emitters, and whether compact optical or MEMS readouts can preserve enough sensitivity for practical deployment. More broadly, passive structural complexity may aid single-point wave sensing beyond this benchmark, but whether that broader promise survives reduced calibration and wider object variation remains open.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for template-matrix calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end direction decoding and robustness experiments. The four frequency bands in Fig. 1e (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
Each clip is reduced to a standardized log-power fingerprint built from the LDV-measured out-of-plane surface velocity \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), where \(W\) is the displacement field and \((x_L, y_L)\) denotes the fixed laser measurement location. For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). For each recorded clip, a short-time Fourier transform (Hann window, 2,048 samples, hop 512) is computed and collapsed into a time-averaged power spectrum:

$$\widehat{S}(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2, \qquad (3)$$

where \(V[k,t]\) is the complex STFT coefficient at frequency bin \(k\) and frame \(t\), and \(T\) is the number of retained frames. The spectrum is restricted to [300, 3,000] Hz (\(F = 346\) bins). Because the observed fingerprints are magnitude statistics, each clip is summarized by a log-power feature vector \(y[k]=\log_{10}(\widehat{S}+\epsilon)\), then standardized per frequency bin using white-noise calibration statistics to yield a normalized feature \(\tilde{y}[k]\) (Supplementary Methods 2). These normalized log-power fingerprints are the observables used for downstream inference.

### Physical dictionary formulation and centered-magnitude SVD
The normalized features serve as the basis for constructing the angle-indexed template matrix used in inference. Per-angle prototypes are averaged across calibration trials to form columns \(h_e\), where \(e\) indexes one of the \(E\) candidate directions, yielding the standardized fingerprint matrix \(H=[h_1,\dots,h_{37}]\in\mathbb{R}^{F\times E}\), with \(E=37\). For the Fig. 2 analysis, we then form the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|.
$$

Singular-value decomposition (SVD) of \(H_{\mathrm{fig}}\) shows rapid early saturation (Fig. 2a). Six modes capture 80.3% of the centered-magnitude energy. Extending to eight modes raises the cumulative fraction to 85.1%.

For intuition, one may also form a reduced-order surrogate in a retained singular subspace; this is the reduced-order picture referenced by Eq. 2. The reported decoding uses the full \(F=346\) standardized feature space without PCA/SVD preprocessing.

### Inference algorithms and network architecture
For decoding, we keep the full standardized feature space and use a grouped dictionary \(D=[d_{e,m}]\); no projection onto the reduced SVD basis is applied at inference time. The inference pipeline follows the measurement sequence itself: the object response is recorded as a standardized fingerprint \(\tilde y\), white-noise calibration organizes those fingerprints into the template matrix \(H\), and decoding must resolve overlap among nearby calibrated directions. The reduced surrogate \((z, A, x)\) below is used only to express that local-overlap geometry in compact form, whereas the implemented inference operates directly on the grouped dictionary \(D\) and the routing stages that follow. The formulation rests on three working assumptions: (1) over each analysis window the target behaves approximately as a linear time-invariant system, so direction-dependent responses superpose in the frequency domain; (2) the calibrated template matrix \(H\) provides a sparse prototype approximation in standardized fingerprint space; and (3) discretizing the angle ordering supplies a useful single-source surrogate over nearby directional templates. Supplementary Methods 1 gives a representative Kirchhoff-Love plate operator and Green's function derivation for assumption (1).

To make that local-overlap picture explicit, we also use a reduced-order surrogate in which a fingerprint is approximated by one dominant calibrated direction plus a small number of nearby corrections:

$$z \approx A\,x, \quad \text{subject to} \quad \lVert x \rVert_0 \le K, \qquad (2)$$

where \(z\) is the reduced fingerprint, \(A\) is the reduced template matrix, \(x\) is a sparse coefficient vector whose dominant support identifies the source direction, and \(K\) is a small residual-correction budget for local overlap and noise. This surrogate is used to motivate the inference problem procedurally: if nearby calibrated directions reuse related reference templates, the ambiguity that must be resolved should be local rather than global.

Once direction is discretized onto the measured angle grid, hard OMP provides the classical greedy sparse-recovery baseline for this surrogate. Supplementary Methods 3 gives the recursion explicitly. Figure 3 uses the corresponding stage-0 correlation profile as a diagnostic of how strongly a fingerprint concentrates on one calibrated angle before any residual refitting.

The guided solver keeps the same \(K\)-stage residual-correction scaffold but replaces the hard one-angle selection with learned local routing. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient vector as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the physical match score between the residual and every grouped template is

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual. A transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary configuration) uses the current residual to query the grouped templates and produces expert-level routing scores

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. In the reported configuration, routed selection uses hard Gumbel gating in the forward pass, so the routing step acts as a discrete local gate over neighboring angle hypotheses. The routed weights then gate the physical match scores to produce a sparse update:

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication. The sparse coefficient vector is accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter, and the residual is corrected by template-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\). Direction supervision and final prediction come from expert-level routing scores:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t^{(\mathrm{exp})}[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). Supplementary Methods 4 provides the complete grouped formulation.

The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the expert-level readout \(\bar s[e]\). In the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The executed loss weights are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization uses Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Calibration protocol and outer-fold cross-validation
White-noise calibration builds the fixed template matrix \(H\): for each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over all three calibration clips, and those clips are never reused for training or evaluation of the direction decoder.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech results use the fixed \(H\) estimated from white-noise calibration only, and scalar performance metrics are reported from held-out outer-test predictions.

For cross-material screening experiments (Fig. 6), object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). For Fig. 6c, each material-specific template matrix is then analyzed with the same row-wise mean-centered magnitude SVD used for Fig. 2, that is, on \(|H| - \mathrm{mean}_{\theta}(|H|)\). For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) screening tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration, rather than cross-object weight sharing.

### Discriminability analysis
Before evaluating any learned solver, Fig. 3 quantifies whether directional structure survives a change in source content. White-noise and speech stimuli are compared using the same calibrated matrix \(H\). Because white noise is the probe used to construct \(H\), whereas speech introduces structured source variation absent from calibration, this contrast tests whether discriminability remains tied to the measured directional response rather than to one source waveform. For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the measured fingerprints preserve directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

### Dose-response and noise robustness
For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise. For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and template matrix for each level.

### Solver ablations
Figures 4 and 5 compare four stable decoder families: *guided solver*, *router-bypass*, *OMP baseline*, and *dense routing*. Figure 3 reports the correlation-based greedy diagnostic separately because immediate single-choice matching already degrades under content variation before full residual correction is applied.

### Evaluation metrics and statistics
Top-1 accuracy is reported on the discrete angle grid. Angular error is the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\), and root-mean-square error (RMSE) = \(\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\). Unless noted otherwise, scalar speech-decoder metrics are reported as mean ± s.d. across the five outer test folds. In Fig. 3f, the white-noise signal curve uses clip-level SEM within each synthetic SNR dataset, whereas the speech-plus-babble curve uses SEM across the five precomputed sweep runs. Mechanistic visualizations such as training curves, confusion matrices, and routing maps are shown for representative folds, whereas aggregate performance statements in the main text refer to held-out outer-fold summaries. When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure panel. Effect sizes are reported as Cohen's \(d\); distribution comparisons use the two-sided Mann-Whitney \(U\) test.

## Data availability
Data and processed features used in this study are available at {USER_TBD: data repository URL}.

## Code availability
Code used for data processing, model training, and figure generation is available at {USER_TBD: code repository URL}.

## Acknowledgements
{USER_TBD: acknowledgements}.

## Funding
{USER_TBD: funding statement}.

## Author contributions
{USER_TBD: author contributions statement}.

## Competing interests
{USER_TBD: competing interests statement}.

## References
::: {#refs}
:::
