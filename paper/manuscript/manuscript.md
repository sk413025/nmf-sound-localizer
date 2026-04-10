# A recurring local directional code emerges across passive objects in single-point vibrometry

## Abstract
Sound direction is usually measured with arrays or specialized directional sensors.
Here we show that ordinary passive objects already carry a directional code in their vibrations that can be read from one fixed laser spot.
In an acrylic plate, that code first appears as a compact local organization rather than as a set of unrelated angle templates.
Held-out speech preserves that code, but broadens overlap first among neighboring directions.
Readout therefore fails when one angle is chosen before evidence shared across that neighborhood has been consolidated.
Matched calibration makes that neighborhood readable, and a physics-guided solver succeeds only when it preserves the measured neighborhood before subtraction.
Across an acrylic plate, paper cup, cardboard box, wooden board, and laptop shell, the same locally ordered directional code recurs, showing that passive structure itself can act as a directional encoding resource rather than only as an obstacle to sensing.

## Introduction
Sound direction is usually engineered by separating sensors in space. Arrays and specialized directional sensors are designed so that direction is measurable at the sensor, while the vibrating or scattering object is usually treated as a nuisance. That default picture leaves a broader possibility mostly unexplored: an ordinary passive object may already separate direction in how it vibrates. If so, directional sensing need not be designed only into the transducer.

Prior work makes that possibility plausible, but not yet general. Single-sensor studies have reported direction-dependent transfer structure in engineered scatterers and structure-borne measurements [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming], while sparse angle-grid inverse methods provide the natural comparison class once nearby directions overlap [@malioutov2005sparse_doa; @chen2001basis_pursuit; @pati1993omp; @tropp2007omp]. Array-free localization is also possible with acoustic vector sensors, but those approaches still depend on specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. The broader physical claim remains open: whether ordinary passive objects themselves carry a reusable directional code, and what governs whether that code stays readable.

Our hypothesis is that direction changes how sound couples into an object's vibration modes, so one fixed point records repeatable spectral reweighting across angle. If nearby directions excite overlapping structural responses, then realistic source variation should broaden ambiguity locally rather than erase the code. We test that idea with a non-contact laser Doppler vibrometer (LDV), whose surface-velocity readout avoids the loading introduced by attached piezoelectric patches or accelerometers [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration; @ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

The acrylic plate is the clearest first revealer of that behavior, but not the endpoint of the paper. The stronger question is whether the same local directional organization recurs across ordinary passive objects and whether recoverability is governed by local separability rather than by response energy alone. More broadly, the study sits alongside wave systems that treat complex scattering as an encoding resource rather than as noise [@fink1997time; @duarte2008single; @gigan2022imaging_computing_disorder; @rotter2017complex_media; @jiang2020randomized_metamaterial]. The paper therefore asks whether ordinary passive objects share a recurring local directional code, and whether local separability is the variable that decides when that code remains readable.

## Results

### A reference acrylic plate exposes directional coding at one measurement point (Fig. 1)
In the acrylic reference object, broadband white-noise excitation establishes the phenomenon that drives the rest of the paper: one flat acoustic input is converted into direction-dependent vibration fingerprints in a single LDV readout.
Representative spectra differ strongly across angle (Fig. 1c), and the full angle-frequency map shows that this is a structured pattern across the full 37-angle grid rather than a trace-level artifact (Fig. 1d).
A component-decomposition view further shows that angle mainly redistributes weight across a small set of reusable centered-magnitude spectral patterns (Fig. 1b).
At one fixed measurement point, the acrylic plate therefore turns direction into a reproducible spectral code. Because the readout is optical and non-contact, it preserves the target's native boundary conditions and avoids sensor loading that could otherwise alter the measured response [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Different frequency bands emphasize different directional sectors (Fig. 1e), indicating that multiple dispersive vibration pathways contribute with angle-dependent weights [@rotter2017complex_media; @kuttruff2025room_acoustics]. The next question is whether this code has an ordered geometry that can be read, or whether it is only a collection of different spectra.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | A passive acrylic plate exposes directional coding at one measurement point.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Component-decomposition view. Three reusable centered-magnitude component spectra are shown together with their relative weight shares at representative angles (0°, 90°, 180°), illustrating how direction reweights the same shared spectral patterns across angle.
c, Broadband spectral reshaping under matched calibration. The flat white-noise source spectrum (grey dashed) is redistributed differently at five representative angles (0°, 45°, 90°, 135°, 180°), showing that direction changes the measured output spectrum.
d, Full angle-frequency heatmap of the mean white-noise calibration response across the 37 measured angles (0°–180°). The response varies systematically across both angle and frequency, making the directional fingerprint visible as organized full-band structure.
e, Frequency-dependent directivity across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz). Different bands emphasize different directional sectors, showing that directional encoding is distributed unevenly across frequency.

### Matched calibration reveals a locally ordered directional code (Fig. 2)
Matched calibration shows that these fingerprints are not a set of unrelated angle templates.
In the full angle-frequency matrix, neighboring directions reuse related structure. Reconstruction error drops sharply once a small retained basis is kept, the inter-angle similarity map is strongest near the diagonal, and the compact 2D embedding shows the same angle-ordered geometry in a lower-dimensional view (Fig. 2d-f). Six components capture 80.3% of the centered-magnitude energy, and eight capture 85.1% (Fig. 2a-c). The directional code is therefore compact before any decoder enters the story.

That compactness matters because it is what we expect if changing direction mainly reweights a limited set of structural responses instead of generating fully independent fingerprints at every angle. Nearby directions therefore remain related without becoming interchangeable. Matched calibration is exposing a local directional code, not a bag of unrelated templates.

That observation changes the form of the decoding problem. Once the measured fingerprints are placed on a discrete angle grid, direction recovery is no longer a search over 37 unrelated references. It becomes a local separability problem: neighboring templates can become jointly plausible before a final angle is chosen. The next question is whether realistic source variation leaves that code intact or turns it into local ambiguity.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Matched calibration reveals a compact local directional code.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. Energy accumulates rapidly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. Most measured directional structure therefore sits in a small component set, and the same early components carry most of the angle ordering.
b, Representative component spectra \(|u_r(f)|\) for components 1, 2, and 6. Together they show three reusable spectral patterns in the centered-magnitude decomposition.
c, Matching half-plane polar profiles \(v_r(\theta)\) for components 1, 2, and 6. They show how those same components vary across 0°-180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing that neighboring directions reuse related spectral structure across the full measured grid.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE drops sharply in the same six-component regime highlighted in panel a, showing that the local code is captured early.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that nearby angles remain related rather than interchangeable. Inset, a 2D embedding of the same centered-\(|H|\) geometry gives a compact visual summary of the curved angle-ordered trajectory implied by that local similarity structure.

### Speech preserves the code but turns decoding into a local ambiguity problem (Fig. 3)
Held-out speech does not erase the locally ordered code revealed by calibration. It changes the problem: because speech samples the same calibrated structure unevenly in time-frequency, overlap broadens first among neighboring directions.

The observed statistics match that local-overlap picture. Under white-noise excitation, within-angle Pearson correlation is near-perfect and clearly separated from between-angle correlation (\(\bar{r} = 1.000\) versus 0.724; \(d = 2.83\); Fig. 3a). Under speech, within-angle similarity drops to 0.907 and still exceeds between-angle similarity (0.798), with \(d = 1.95\) (Fig. 3b). The per-angle discriminability margin stays positive at every angle, although its mean decreases from 0.28 under white noise to 0.11 under speech (Fig. 3c). Speech therefore preserves the code while broadening overlap among nearby calibrated angles.

The first hard angle choice is where that broader local overlap starts to hurt readout. The correlation-based first-choice diagnostic isolates that initial greedy decision before any residual correction. It succeeds on most white-noise fingerprints, where the pairwise similarity map is close to identity, but it collapses on held-out speech, where similarity spreads across neighboring calibrated angles while remaining locally ordered (Fig. 3d,e). Once several adjacent templates become jointly plausible, immediate one-angle commitment becomes unstable even though the code itself is still present.

Added noise pushes the same ambiguity outward. In both synthetic noisy white-noise datasets and a separate five-seed speech-plus-babble sweep, the stage-0 diagnostic falls monotonically with SNR (Fig. 3f).

The code survives. The remaining question is what kind of readout can preserve that neighborhood long enough to recover direction.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Speech preserves the code but broadens nearby-angle overlap.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); within-angle similarity exceeds between-angle similarity despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is smaller but remains above zero at all angles.
d, Stage-0 correlation-based first-choice diagnostic across angle for white noise and held-out speech, shown with light clip-level uncertainty bands. The diagnostic is strong on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Stage-0 correlation-based first-choice diagnostic versus SNR. The white-noise curve (blue, clip-level standard error of the mean (SEM) shading) is recomputed on synthetic noisy white-noise datasets. The speech-plus-babble curve (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading) comes from a separate five-seed sweep. Both curves decline monotonically as noise increases, so noise enlarges the same local-overlap failure already exposed by speech.

### Readout succeeds only when the measured neighborhood is preserved (Figs. 4 and 5)
Figures 4 and 5 answer one bounded question. Once local overlap is the governing difficulty, what readout remains scientifically admissible? The answer is narrow: subtraction helps only after the measured neighborhood has been preserved [@gregor2010lista; @monga2021unrolling].

Figure 4 shows that requirement directly against measured structure. In a representative 70° validation clip, broad matching initially excites several nearby calibrated angles above the physically plausible band in \(H\) (Fig. 4a,b), and the local gate then contracts support around the correct neighborhood (Fig. 4c). After the first guided step, the residual drops from 1.00 to 0.48 while the fraction of update mass within 15° rises from 0.18 to 0.98 (Fig. 4d). The correction is therefore not just smaller. It has collapsed into the neighborhood that the measurement itself says is plausible.

The clean decoder comparison isolates the same requirement in simpler form. Guided decoding stays highest because it preserves that neighborhood before subtraction, whereas router-bypass, OMP, and dense routing progressively break the measured local order (Fig. 4e). Once overlap is local, early commitment is the wrong operation.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | Preserving the measured neighborhood makes readout reliable.**
a, Architecture and physics correspondence. The measured local band in \(H\) is shown beside the staged broad-match, local-gate, and local-update profiles on the shared angle axis, defining the neighborhood that should be preserved before subtraction.
b, Broad initial match for a representative 70° validation clip. Broad residual-to-dictionary matching excites several nearby calibrated angles above the measured local band in \(H\).
c, Local gate convergence for the same representative clip. Learned local pooling contracts support around the correct neighborhood and keeps the resulting update confined there.
d, Residual purification. Left, relative residual norm across the guided steps for the representative clip. Right, validation-wide cumulative update mass within radius before and after one local step across 1,924 clips; the mass within 15° rises from 0.18 to 0.98. Nearly all correction therefore collapses into the physically plausible neighborhood before subtraction proceeds.
e, Clean-condition Top-1 comparison of four overlap-handling rules on the same fixed dictionary (individual seeds shown as dots; large markers and error bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing. Preserving the local angle band is what keeps subtraction accurate.

Figure 5 asks the remaining admissibility question: does that readout stay aligned with the measured neighborhood it exploits? It does. The calibrated template matrix \(H\) and the guided neighborhood-emphasis map share the same coarse near-diagonal organization on the same angle frame (Fig. 5d,e). Matrix-level agreement reaches \(r = 0.47\), and the per-angle local-band summaries reach \(r = 0.46\) with a mean absolute gap of 0.23 (Fig. 5f). Those values do not prove perfect recovery. They show that the learned geometry stays anchored to the same neighborhood that calibration exposed as relevant for decoding.

The same alignment explains the prediction behavior. Under clean evaluation, guided decoding stays near the diagonal, whereas the OMP baseline fractures into broader off-diagonal errors after committing too early (Fig. 5c). Across angle, delayed commitment performs best across most of the 37 measured directions, and across SNR it degrades least as added noise increases (Fig. 5a,b). The admissibility question is therefore closed. The remaining question is whether this local code is peculiar to acrylic or general across ordinary passive objects.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | The admissible readout follows the measured neighborhood.**
a, Five-seed SNR sweep for the same four overlap-handling rules. The delayed-commitment rule degrades least as noise increases, consistent with preserving local overlap before subtraction.
b, Per-angle clean accuracy across the 37 measured angles. After light 3-angle smoothing, the delayed-commitment rule remains highest across most angles, whereas dense routing stays near chance across much of the grid; light shading shows \(\pm 1\) s.e.m. around the five-seed mean \(P(\mathrm{correct})\). The neighborhood therefore matters across the grid, not only in one average metric.
c, Row-normalized clean confusion comparison of the OMP baseline and guided solver. Guided decoding concentrates near the diagonal, whereas the OMP baseline shows broader off-diagonal fracture after early commitment.
d, Measured local structure in the calibrated fingerprint space. The near-diagonal band summarizes the physical neighborhood that nearby angles share.
e, Learned neighborhood-emphasis map from the guided solver on the same angle frame and correlation scale as panel d. The learned map keeps a similar coarse near-diagonal ordering instead of dispersing weight broadly across the grid.
f, Quantitative structure alignment. Top, normalized local-band summaries from the measured and learned maps across angle. Bottom, concordance scatter of the same per-angle scores. The full-matrix agreement reaches \(r = 0.47\), and the per-angle summaries reach \(r = 0.46\) (mean absolute gap \(= 0.23\)). The learned map therefore remains anchored to the measured neighborhood.

### The same locally ordered directional code recurs across ordinary passive objects (Fig. 6)
Across an acrylic plate, paper cup, cardboard box, wooden board, and laptop shell, the same locally ordered directional code reappears. Each object shows structured angle-frequency fingerprints, a finite neighborhood of positive local ordering, and above-chance single-point readout under matched calibration (Fig. 6a-c). This directional organization is therefore a broader passive-object phenomenon rather than a special property of one plate.

Cross-object readout then clarifies what governs how usable that code is. Matched object-specific calibration keeps Top-1 readout above chance across the full set (Fig. 6d). Across objects, mean Top-1 aligns more closely with mean top-3 subspace overlap burden than with normalized overall \(|H|\) energy alone. The paper cup provides the clearest contrast: it carries the strongest overall energy, yet its neighboring angles overlap more broadly, so it reads out worse than cardboard. Local separability therefore matters more than raw energy for directional decoding.

The same logic carries into frequency. A shared contrast-selection rule identifies a different informative window for each object, and the corresponding band-limited directional codes differ across those object-specific bands (Fig. 6e). What recurs across the tested set is not one universal spectrum. It is one local directional principle expressed through object-specific informative bands.

![](../figures/fig06_universality.jpg)

**Fig. 6 | A recurring directional code appears across ordinary passive objects.**
a, Measured response-regime map across the five target objects. Each object is placed by the zero crossing of its mean centered-\(|H|\) correlation-decay curve and by the effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means exceed chance across the five objects. Object mean Top-1 is plotted against mean top-3 subspace overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Response energy alone does not rank the objects. Local separability does.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. The informative band changes from object to object, but the directional principle does not.

## Discussion
Directional encoding is not only something engineers build into arrays and specialized sensors. Across the tested objects, passive structural vibration itself organizes sound direction into a finite local code that one fixed vibrometric point can read after matched calibration. The object is therefore not merely something a directional sensor probes. It participates in the encoding.

That result also changes what governs recoverability. The decisive variable is not raw response energy or global template separability. It is local separability inside the calibrated neighborhood. In the acrylic reference object, held-out speech preserves the code but broadens overlap among neighboring directions [@gigan2022imaging_computing_disorder; @rotter2017complex_media]. Decoding therefore fails when subtraction happens before evidence shared across that neighborhood has been preserved. The guided solver matters because it respects the measured neighborhood long enough for subtraction to sharpen, and its learned structure mirrors the same local geometry revealed by calibration.

This view links the paper to several literatures. Studies of engineered disorder and metasurfaces treat complex scattering as a computational wave encoder [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa]. Single-sensor localization studies show that direction-dependent transfer structure can be exploited without large arrays [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Here the same logic appears in unmodified passive objects, while non-contact LDV readout preserves native boundary conditions and avoids sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. Passive structural complexity is therefore not something to average away. It is the encoding resource.

These conclusions are bounded to the tested half-plane grid (0-180° at 5° spacing), the controlled acoustic environment, the single static source, and per-object matched calibration. The LDV is still a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. Wider object variation, reduced calibration, stronger damping, reverberation, multiple simultaneous sources, moving emitters, and compact optical or MEMS readouts will therefore require new measurements.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for template-matrix calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end direction decoding and robustness experiments. The four frequency bands in Fig. 1e (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
At the physical level, we interpret the single-point response as a superposition of dispersive structural contributions whose direction-dependent weights change with incidence angle:

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the single-point velocity response, \(s_m(\omega)\) are modal spectral patterns, and \(\alpha_m(\theta)\) are direction-dependent participation weights. In this view, changing direction mainly reweights a limited set of structural responses rather than generating fully independent fingerprints at every angle.

Each clip is reduced to a standardized log-power fingerprint built from the LDV-measured out-of-plane surface velocity \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), where \(W\) is the displacement field and \((x_L, y_L)\) denotes the fixed laser measurement location. For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). For each recorded clip, a short-time Fourier transform (Hann window, 2,048 samples, hop 512) is computed and collapsed into a time-averaged power spectrum:

$$\widehat{S}(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2, \qquad (3)$$

where \(V[k,t]\) is the complex STFT coefficient at frequency bin \(k\) and frame \(t\), and \(T\) is the number of retained frames. The spectrum is restricted to [300, 3,000] Hz (\(F = 346\) bins). Because the observed fingerprints are magnitude statistics, each clip is summarized by a log-power feature vector \(y[k]=\log_{10}(\widehat{S}+\epsilon)\), then standardized per frequency bin using white-noise calibration statistics to yield a normalized feature \(\tilde{y}[k]\) (Supplementary Methods 2). These normalized log-power fingerprints are the observables used for downstream inference.

### Calibration object and centered-magnitude representation
Matched calibration produces the angle-indexed template matrix used throughout the readout analysis. Per-angle prototypes are averaged across calibration trials to form columns \(h_e\), where \(e\) indexes one of the \(E\) candidate directions, yielding the standardized fingerprint matrix \(H=[h_1,\dots,h_{37}]\in\mathbb{R}^{F\times E}\), with \(E=37\). For Fig. 2 and the descriptor analyses in Fig. 6, we also form the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|.
$$

Singular-value decomposition (SVD) of \(H_{\mathrm{fig}}\) shows rapid early saturation (Fig. 2a). Six components capture 80.3% of the centered-magnitude energy. Extending to eight components raises the cumulative fraction to 85.1%.

The calibrated template matrix \(H\) is the measured map of how the structural response changes across direction. The centered-magnitude matrix \(H_{\mathrm{fig}}\) is the representation used for compactness and local-order analyses. The reported readout itself operates in the full \(F=346\) standardized feature space through grouped templates derived from the calibrated fingerprints.

### Inference formulations and algorithms
The implemented readout operates directly on a grouped dictionary \(D=[d_{e,m}]\) built from the full standardized fingerprints, with no projection onto the reduced SVD basis at inference time. The formulation rests on three working assumptions: (1) over each analysis window the target behaves approximately as a linear time-invariant system, so direction-dependent responses superpose in the frequency domain; (2) the calibrated template matrix \(H\) provides a sparse prototype approximation in standardized fingerprint space; and (3) discretizing the angle ordering supplies a useful single-source surrogate over nearby directional templates. Supplementary Methods 1 gives a representative Kirchhoff-Love plate operator and Green's function derivation for assumption (1).

To make the local-overlap geometry explicit, we also use a reduced-order surrogate in which a fingerprint is approximated by one dominant calibrated direction plus a small number of nearby corrections:

$$z \approx A\,x, \quad \text{subject to} \quad \lVert x \rVert_0 \le K, \qquad (2)$$

where \(z\) is the reduced fingerprint, \(A\) is the reduced template matrix, \(x\) is a sparse coefficient vector whose dominant support identifies the source direction, and \(K\) is a small residual-correction budget for local overlap and noise. In this surrogate, nearby calibrated directions reuse related reference templates, so the ambiguity to be resolved is local rather than global.

Once direction is discretized onto the measured angle grid, hard OMP provides the classical greedy sparse-recovery baseline for this local-overlap surrogate. Supplementary Methods 3 gives the recursion explicitly. Figure 3 uses the corresponding stage-0 correlation profile as a diagnostic of how strongly a fingerprint concentrates on one calibrated angle before any residual refitting.

The guided solver uses a \(K\)-stage residual-correction scaffold and replaces hard one-angle selection with learned local gating, preserving the contiguous angle neighborhood implied by \(H\) long enough for subtraction to act on a physically plausible local support. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient vector as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the physical match score between the residual and every grouped template is

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual. In the reported implementation, the local gate is parameterized with a compact transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary configuration) that uses the current residual to query the grouped templates and produces direction-group scores

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. This score concentrates the existing physical match into one locally plausible neighborhood before subtraction. In the reported implementation, selection uses a straight-through Gumbel approximation in the forward pass, so this step acts as a discrete local gate over neighboring angle hypotheses. The gated weights then modulate the physical match scores to produce a sparse update:

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication. The sparse coefficient vector is accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter, and the residual is corrected by template-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\). This update focuses broad local evidence onto one physically plausible neighborhood before subtraction. Direction supervision and final prediction come from the same direction-group scores:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t^{(\mathrm{exp})}[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). Supplementary Methods 4 provides the complete grouped formulation.

### Training and optimization
The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the final readout \(\bar s[e]\). In the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The executed loss weights are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization uses Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Evaluation protocols
White-noise calibration builds the fixed template matrix \(H\): for each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over all three calibration clips, and those clips are never reused for training or evaluation of the direction readout.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech analyses use the fixed \(H\) estimated from white-noise calibration. The mechanism panels in Fig. 4b,c draw from validation routing outputs in the selected fold, and Fig. 5b,d use representative clean displays from selected runs. For Fig. 4d and the sweep-derived panels in Fig. 5a,e, we ran five independent clean or SNR-sweep evaluations and summarized the checkpoints selected on the validation fold (`results/figure4_data.json` and `results/fig05_panel_f_no_type_bias_clean_seed_means/summary.npz`). Held-out outer-test predictions are reported only where noted.

For the five-object comparison in Fig. 6, object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). The panel-a descriptor map and panel-c decay curves are computed from the row-wise mean-centered magnitude matrix \(|H| - \mathrm{mean}_{\theta}(|H|)\): for each object, we form the mean inter-angle correlation as a function of angular separation and record the first separation at which that mean correlation becomes non-positive. Effective rank is taken from the singular-value spectrum of the same centered-\(|H|\) matrix. For panel e, the representative frequency window is defined by the peak of a smoothed across-angle contrast profile \(\mathrm{std}_{\theta}(|H|)/\mathrm{mean}_{\theta}(|H|)\), and the corresponding band-limited directional code is obtained by averaging \(|H|\) across that three-bin window. For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) readout tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration.

### Diagnostic analyses
Figure 3 quantifies whether directional structure survives a change in source content. White-noise and speech stimuli are compared using the same calibrated matrix \(H\). Because white noise is the probe used to construct \(H\), whereas speech is sparse and non-stationary in time-frequency, this contrast tests whether discriminability stays tied to the measured directional response when short windows sample that calibration map unevenly across frequency. For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the measured fingerprints preserve directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise.
Panel \(d\) uses held-out clean speech and white-noise fingerprints from the matched calibration environment.
Panel \(f\) instead combines synthetic noisy white-noise datasets with a separate precomputed five-seed speech-plus-babble sweep.
The two curves are aligned as stage-0 first-choice diagnostics, not as one continuous held-out-speech surface.
For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and template matrix for each level.

Figures 4 and 5 hold the matched-calibration dictionary fixed and compare how different overlap-handling rules change residual cleanup, prediction structure, and noise robustness once local ambiguity appears. *Guided solver* first pools support within the locally matching neighborhood, *router-bypass* keeps the same staged updates without that pooling, *OMP baseline* applies the soft-OMP comparison without learned local routing, and *dense routing* spreads activation across experts without sparse local concentration. Figure 3 reports the correlation-based greedy diagnostic separately because immediate single-choice matching already degrades under content variation before full residual correction is applied.

### Evaluation metrics and statistics
Top-1 accuracy is reported on the discrete angle grid. Angular error is the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\), and root-mean-square error (RMSE) = \(\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\). Unless noted otherwise, scalar speech-readout metrics from the outer-fold protocol are reported as mean ± s.d. across the five held-out test folds. Figure 4d and the sweep-derived panels in Fig. 5a,e instead summarize five independent sweep runs selected by validation performance, with mean ± s.e.m. across sweep seeds; Fig. 5b,d are representative clean displays. In Fig. 3f, the white-noise signal curve uses clip-level SEM within each synthetic SNR dataset, whereas the speech-plus-babble curve uses SEM across the five precomputed sweep runs. When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure panel. Effect sizes are reported as Cohen's \(d\); distribution comparisons use the two-sided Mann-Whitney \(U\) test.

## Data availability
{USER_TBD: data availability statement}.

## Code availability
{USER_TBD: code availability statement}.

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
