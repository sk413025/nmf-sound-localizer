# Sound direction survives in a single vibrometric readout of a passive acrylic plate after matched calibration

## Abstract
Passive objects can encode sound direction in their vibration patterns, but it is unclear whether one fixed vibrometric measurement can recover that direction after calibration. Here, matched calibration on a 37-angle half-plane (0-180° in 5° steps) shows that a passive acrylic plate produces repeatable directional fingerprints in a single non-contact laser Doppler vibrometer (LDV) readout. Held-out speech preserves that directional structure, but its sparse and non-stationary time-frequency content samples the calibrated response unevenly across short windows, consistent with broader overlap among nearby angles. The main readout failure therefore appears when several neighboring directions remain plausible and the decoder commits too early to one of them. A physics-guided solver that preserves local evidence before subtraction stabilizes clean-condition readout. Using the acrylic plate as a reference object, a five-object comparison shows that matched calibration can reveal analogous object-conditioned directional readout in other ordinary passive objects. Together, these results identify matched calibration as a pathway to calibrated single-point structural-acoustic sensing in passive objects, with nearby-angle overlap defining the central readout constraint.

## Introduction
Sound direction is usually measured with arrays or specialized vector sensors, but passive structures may already encode it in how they vibrate under directional forcing. The question here is whether one fixed vibrometric measurement, taken at one point on the object, can recover that direction after object-specific calibration. In this framing, the object itself becomes the sensing element, and the main readout challenge is whether nearby directions remain distinct enough to be separated from one another.

Our working hypothesis is that direction changes how sound couples into an object's vibration modes, and that a fixed measurement point can record those changes as repeatable spectra. Incident sound excites the object, the LDV records the resulting fingerprint, and matched calibration organizes those fingerprints into a template matrix \(H\). If nearby directions reuse related spectral structure, then source-content changes should broaden ambiguity mainly within local angle neighborhoods rather than erase the directional code altogether.

That hypothesis makes two concrete predictions. First, calibration should reveal a compact measured space in which neighboring directions remain locally ordered. Second, white noise and speech should interrogate that space differently. White noise fills the analysis band at each short window and should excite the available resonant pathways densely, whereas speech is sparse and non-stationary and should sample the same calibration map unevenly across short windows. If nearby directions reuse related spectral structure, that uneven sampling should blur separation first within local angle neighborhoods. Stable decoding would then depend on pooling local evidence across neighboring directions before a final angle is committed.

Several lines of prior work motivate that possibility, but they stop short of the present setting. Single-sensor studies have reported direction-dependent transfer functions in engineered scatterers and in structure-borne measurements [@elbadawy2018lego_doa; @dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Calibrated inverse formulations on a discrete angle grid provide one useful comparison class once nearby overlap must be resolved [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling; @chen2001basis_pursuit; @tibshirani1996lasso; @mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free localization is also possible with acoustic vector sensors, but those approaches require specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. What remains unclear is whether an unmodified passive structure can provide a reproducible calibrated pathway for single-point directional readout.

We test that possibility with a non-contact laser Doppler vibrometer (LDV). LDV provides high-bandwidth, high-sensitivity vibration measurement [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration], whereas piezoelectric patches and accelerometers can alter local mass, stiffness, and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. More broadly, the study sits alongside wave systems that treat complex scattering as an encoding resource rather than as noise [@fink1997time; @duarte2008single; @gigan2022imaging_computing_disorder; @rotter2017complex_media; @jiang2020randomized_metamaterial]. We therefore use the acrylic plate as a reference object for dissecting the mechanism and deriving the readout constraint, then test across five ordinary passive objects whether the same style of locally ordered directional readout extends beyond one reference structure.

## Results

### The acrylic plate yields repeatable direction-dependent vibration fingerprints (Fig. 1)
In the acrylic plate used throughout the main reference-object study, the measured LDV spectrum changes systematically with source angle under broadband white-noise excitation: the same flat input is reshaped into distinct output spectra at different directions (Fig. 1c). Incident sound therefore drives direction-dependent plate response, and the LDV samples that response at one fixed point. Because the readout is optical and non-contact, it preserves the target's native boundary conditions and avoids sensor loading that could otherwise alter the measured response [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

These fingerprints are also repeatable. Independent white-noise recordings at the same angle produce nearly identical spectra (Fig. 1d), and their trial-to-trial variation is much smaller than the differences across angles. Different frequency bands then show different angular patterns (Fig. 1e), consistent with multiple dispersive vibration modes contributing with angle-dependent weights (Fig. 2b,c) [@rotter2017complex_media; @kuttruff2025room_acoustics]. These repeatable angle-specific spectra therefore suggest an organized measured space rather than a collection of unrelated traces.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°-180° for four frequency bands (0.3-0.5, 0.5-1, 1-2, 2-3 kHz), showing that each band carries a distinct directional response pattern.

### Calibration fingerprints compress into a locally ordered measured space (Fig. 2)
Matched calibration reveals a compact measured space. Neighboring angles remain close in fingerprint space, and a small retained basis already reconstructs most of the structure (Fig. 2d-f). For the centered-magnitude view used in Fig. 2,

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|
$$

the spectrum decays rapidly (Fig. 2a-c): the first six components capture 80.3% of the energy and eight capture 85.1%. The measured fingerprints therefore occupy a compact angle-ordered space with strong local continuity across nearby directions.

That organization is consistent with a low-dimensional physical response. Under small-amplitude dynamics, the single-point velocity response can be written as a superposition of dispersive structural modes, each with its own spectral pattern \(s_m(\omega)\) and direction-dependent coupling \(\alpha_m(\theta)\):

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the complex frequency response at the LDV measurement point for incidence direction \(\theta\), and the sum runs over a limited set of contributing modes (Supplementary Methods 1) [@ewins2000modal; @meirovitch2001fundamentals]. In this view, changing direction mainly reweights a limited set of dispersive responses, so neighboring angles reuse related spectral structure rather than generating fully independent fingerprints. The centered-magnitude components should therefore be read as empirical summaries of shared spectral structure, not as resolved physical eigenmodes. The inter-angle similarity map shows that this shared structure remains concentrated among neighboring angles.

Calibration therefore sets the form of the decoding problem. Once the measured fingerprints are arranged on a discrete angle grid, direction recovery is no longer a choice among 37 unrelated references. The main ambiguity is local: neighboring templates can remain jointly plausible before a final angle is chosen. Held-out speech next tests how far that local overlap expands under source-content variation.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. The cumulative curve rises quickly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. The overlaid auxiliary angle-ordering proxy, derived from the same decomposition, follows the same rapid early accumulation.
b, Frequency-selective spectra \(|u_r(f)|\) for representative components 1, 2, and 6. These traces summarize three reusable spectral patterns in the centered-magnitude decomposition.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative components 1, 2, and 6, showing how those same components vary across 0°-180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-component regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that neighboring angles remain close in fingerprint space, revealing a local angle ordering across nearby directions.

### Speech preserves directional structure while broadening nearby-angle overlap (Fig. 3)
Held-out speech tests whether the compact local structure identified in Figs. 1 and 2 survives a change in excitation. White noise is the calibration probe used to estimate the template matrix \(H\), and because it fills the analysis band in each short window it samples the plate's directional response densely. Speech probes the same structure differently. Its energy is sparse and non-stationary in time-frequency, concentrated in moving harmonics and formants rather than spread uniformly across the band, so short windows need not sample the calibrated response uniformly across frequency. Because Fig. 2f showed that neighboring angles already reuse related spectral structure, that uneven sampling should blur separation first within local angle neighborhoods rather than erase similarity globally.

The observed statistics match that local-overlap picture. Under white-noise excitation, within-angle Pearson correlation is near-perfect and clearly separated from between-angle correlation (\(\bar{r} = 1.000\) versus 0.724; \(d = 2.83\); Fig. 3a). Under speech, within-angle similarity drops to 0.907, but it remains higher than between-angle similarity (0.798), with \(d = 1.95\) (Fig. 3b). The per-angle discriminability margin stays positive at every angle, although its mean decreases from 0.28 under white noise to 0.11 under speech (Fig. 3c). Speech therefore preserves the directional code while broadening overlap among nearby calibrated angles.

The first discrete commitment is where that broader local overlap becomes operationally costly. The correlation-based first-choice diagnostic isolates that initial greedy decision before any residual correction. It succeeds on most white-noise fingerprints, where the pairwise similarity map is close to identity, but it collapses on held-out speech, where similarity spreads across neighboring calibrated angles while remaining locally ordered (Fig. 3d,e). Once several adjacent templates become jointly plausible, immediate one-angle commitment becomes unstable even though directional structure remains present.

Added noise worsens the same kind of overlap problem, but panel f is a separate surface from panel d: the white-noise curve is recomputed on synthetic noisy white-noise datasets, whereas the speech curve comes from a separate five-seed speech-plus-babble sweep. Within those two matched stage-0 diagnostic surfaces, first-choice accuracy declines monotonically as noise increases (Fig. 3f). Figure 3 therefore does not by itself identify one algorithm, but it does set a concrete readout requirement: local evidence among neighboring calibrated angles must be preserved long enough to be resolved before subtraction begins. The next two figures stay within the acrylic reference-object study and test that requirement directly.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Speech preserves directional structure while broadening nearby-angle overlap.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); within-angle similarity remains higher than between-angle similarity despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based first-choice diagnostic traces for white noise and held-out speech, shown with light clip-level uncertainty bands: the stage-0 diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response curves for stage-0 correlation-based first-choice matching versus SNR. The white-noise signal curve (blue, clip-level standard error of the mean (SEM) shading) is recomputed on synthetic noisy white-noise datasets, whereas the speech-plus-babble curve (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading) comes from a separate five-seed sweep; within those two matched surfaces, both decline monotonically with increasing noise.

### Resolving nearby-angle ambiguity before subtraction stabilizes decoding (Fig. 4)
Figure 3 sets that readout requirement in operational form: once speech broadens overlap among neighboring templates, hard one-angle selection discards the measured local continuity too early. We therefore use a physics-guided solver that writes that local continuity into each residual-correction step by scoring the residual against the calibrated dictionary, concentrating the match within a local angle neighborhood, and only then subtracting [@gregor2010lista; @monga2021unrolling].

In a representative 70° validation exemplar, the initial match spans several neighboring calibrated angles, then contracts around the correct neighborhood and leaves a cleaner residual after one update (Fig. 4b,c). Figure 4a sketches the same sequence. The 15° mass summary is validation-wide across 1,924 validation clips, rising from 0.18 to 0.98, whereas the residual drop from 1.00 to 0.48 belongs to the representative exemplar shown here.

Panel d is a restricted clean-condition check on the same point. Accuracy is highest for the guided solver (0.98), lower for router-bypass (0.58), lower again for the OMP baseline (0.44), and near chance for dense routing (0.03) (Fig. 4d). On this secondary surface, the family that narrows nearby-angle ambiguity before subtraction is also the family that decodes most accurately.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | A physics-guided solver concentrates nearby overlap into the correct neighborhood.**
a, Sequence summary of broad physical match, learned local gating, and residual cleanup after one step.
b, Representative 70° validation exemplar: the initial physical match spans nearby calibrated directions, the learned local gate concentrates that support around the correct neighborhood, and the resulting local update remains confined to that band.
c, Residual profile before and after one refinement step for the same exemplar. The 15° mass callout summarizes the corresponding validation-wide aggregate shift across 1,924 validation clips (0.18 to 0.98), whereas the residual drop (1.00 to 0.48) belongs to the representative exemplar shown in the panel.
d, Secondary restricted clean-condition check across the four readout families over the same five-seed sweep (individual seeds shown as dots; points with horizontal bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing.

### Prediction structure stays organized around the measured local band (Fig. 5)
Figure 5 then examines the prediction structure that remains once local ambiguity is handled before subtraction. Panel c suggests the answer is still local. The calibrated template matrix \(H\) shows a near-diagonal band of local overlap, and the guided solver's neighborhood-emphasis map retains a coarse near-diagonal band (Fig. 5c).

The representative prediction displays show the consequence of that local organization. In the confusion maps and angle profiles, the guided solver remains concentrated near the correct neighborhood. Router-bypass spreads more broadly around the same targets, the OMP baseline breaks into off-diagonal errors, and dense routing collapses toward a preferred output mode instead of preserving angle-conditioned neighborhoods (Fig. 5b,d).

The sweep summaries in Fig. 5a,e then show the performance consequence. Across noise levels and across clean per-angle accuracy, the guided solver degrades least and remains highest overall, whereas the more immediate or more diffuse alternatives deteriorate more sharply. Delaying commitment until local overlap has narrowed therefore improves not only the update shown in Fig. 4, but also the broader decoding performance measured here.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | Prediction structure stays organized around the measured local band.**
a, Five-seed SNR sweep across four readout families. The guided solver degrades least as noise increases.
b, Row-normalized representative confusion display across the four compared families. The guided solver stays nearest the diagonal, whereas the OMP baseline shows fractured off-diagonal leakage, router-bypass remains broader around the target neighborhood, and dense routing collapses toward a preferred output mode.
c, Measured local structure (top) and neighborhood-emphasis map from the guided solver (bottom). The lower map retains a coarse near-diagonal angle ordering similar to the calibrated fingerprints.
d, Angle-specific prediction profiles at four representative directions (55°, 70°, 95°, and 100°): the guided solver produces tighter local prediction profiles, whereas router-bypass shows broader off-axis leakage around the same targets.
e, Per-angle clean accuracy: five-seed mean \(P(\mathrm{correct})\) across the 37 measured angles, shown as a 3-angle centered moving-average display with light \(\pm 1\) s.e.m. shading, comparing the four readout families on the clean sweep. The guided solver retains the highest clean mean accuracy overall, whereas dense routing remains near chance across almost the entire angle set.

### Matched calibration extends object-conditioned directional readout beyond the acrylic reference object (Fig. 6)
The acrylic plate is the reference object used to dissect the local-overlap problem. Figure 6 then asks whether the same style of object-conditioned directional readout also appears in other passive objects. Across five ordinary objects, each object again yields an angle-frequency fingerprint matrix \(H\) with visible directional structure (Fig. 6a,b). Each object also remains above chance under its own matched calibration (Fig. 6d), and the compactness summary in Fig. 6c is consistent with the same kind of organized fingerprint space. The informative band, however, shifts across objects (Fig. 6e), so the extension is carried by object-conditioned structure rather than by one fixed spectral template.

Each object still requires its own calibration because response energy, informative band, and Top-1 accuracy differ across objects. Even so, the recurrence of structured \(H\) matrices, compactness, and above-chance readout across five materials shows that the acrylic plate is not the only passive structure that supports locally ordered directional readout under matched calibration. Figure 6 therefore extends object-conditioned directional readout beyond the acrylic plate while keeping the directional code tied to each object's own structural response.

![](../figures/fig06_universality.jpg)

**Fig. 6 | Matched calibration extends object-conditioned directional readout beyond the acrylic reference object.**
a, Five target objects in display order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects despite different response patterns.
c, Centered-magnitude SVD summary across objects. Cumulative centered-\(|H|\) energy and the corresponding rank-90 and rank-95 markers provide a supporting compactness comparison for the five calibrated template matrices.
d, Object-conditioned readout across the five objects. All five objects remain above chance under matched object-specific calibration, and the energy-versus-accuracy comparison shows that Top-1 accuracy does not monotonically track overall response energy across this sample. The accompanying Top-1 confidence intervals summarize uncertainty while preserving that mismatch across objects.
e, Frequency structure across objects. Per-object spectra and directional band profiles indicate that the informative frequency band can shift across objects, so the matched-calibration readout remains object-conditioned.

## Discussion
Matched calibration turns the acrylic plate's direction-dependent response into a calibrated single-point readout that remains readable under source change. At one fixed LDV position, the plate produces repeatable measured fingerprints, those fingerprints organize into a compact template matrix \(H\), and held-out speech preserves their local angle ordering even as it broadens overlap among neighboring calibrated directions [@gigan2022imaging_computing_disorder; @rotter2017complex_media]. In that setting, the operational readout problem is not whether directional structure remains, but how to resolve overlap among nearby calibrated directions without destroying that local ordering. Together, these results identify the readout logic that makes calibrated single-point structural-acoustic sensing feasible in this setting: a passive object can provide object-specific directional information, and nearby-angle overlap becomes the central constraint on decoding it.

The five-object comparison extends that observation beyond the acrylic plate. After each tested object is calibrated on its own, structured directional fingerprints and above-chance readout appear in cardboard, wood, paper-cup, and laptop-shell objects as well. Normalized response energy, informative bands, and Top-1 performance remain object-specific, but the recurrence of compact, locally organized directional structure across these materials shows that the acrylic plate is a reference object rather than a singular exception. Matched calibration therefore reveals the same style of object-conditioned directional readout across the tested passive structures.

These findings connect to studies that use engineered disorder or metasurfaces as computational wave encoders [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa], but here matched calibration exposes usable direction-dependent vibration structure in an unmodified acrylic plate. The work also complements single-sensor localization with embedded microphones and contact-mounted piezoelectric sensors [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming], because non-contact LDV readout preserves native boundary conditions and avoids sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Several limitations bound the present claims. The present study uses a half-plane grid (0-180°) at 5° spacing, a controlled acoustic environment, a single static source, and per-object matched calibration. The LDV is a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. Future work should test whether the observed local angle ordering can be predicted or adapted with less calibration, how performance changes under stronger damping, reverberation, multiple simultaneous sources, or moving emitters, and whether compact optical or MEMS readouts can preserve enough sensitivity for practical deployment. More broadly, passive structural complexity may aid single-point wave sensing beyond the tested setting, but whether that broader promise survives reduced calibration and wider object variation remains open.

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
The normalized features serve as the basis for constructing the angle-indexed template matrix used in readout. Per-angle prototypes are averaged across calibration trials to form columns \(h_e\), where \(e\) indexes one of the \(E\) candidate directions, yielding the standardized fingerprint matrix \(H=[h_1,\dots,h_{37}]\in\mathbb{R}^{F\times E}\), with \(E=37\). For the Fig. 2 analysis, we then form the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|.
$$

Singular-value decomposition (SVD) of \(H_{\mathrm{fig}}\) shows rapid early saturation (Fig. 2a). Six components capture 80.3% of the centered-magnitude energy. Extending to eight components raises the cumulative fraction to 85.1%.

In the paper's logic, \(H\) is the scientific object: it is the measured calibration map that records how the structural response changes across direction. For intuition, one may also form a reduced-order surrogate in a retained singular subspace; this is the reduced-order picture referenced by Eq. 2. The reported readout does not infer in that reduced space. Instead, it keeps the full \(F=346\) standardized feature space and uses grouped templates derived from the calibrated fingerprints.

### Inference algorithms and network architecture
To keep the inference layers distinct, we use three linked representations. First, the template matrix \(H\) is the measured calibration object introduced above. Second, the reduced surrogate \((z, A, x)\) below is an intuition layer that makes local overlap explicit when nearby calibrated fingerprints reuse related structure. Third, the implemented readout operates directly on a grouped dictionary \(D=[d_{e,m}]\) built from the full standardized fingerprints, with no projection onto the reduced SVD basis at inference time. The pipeline follows the measurement sequence itself: the object response is recorded as a standardized fingerprint \(\tilde y\), white-noise calibration organizes those fingerprints into \(H\), and readout must then resolve overlap among nearby calibrated directions. The formulation rests on three working assumptions: (1) over each analysis window the target behaves approximately as a linear time-invariant system, so direction-dependent responses superpose in the frequency domain; (2) the calibrated template matrix \(H\) provides a sparse prototype approximation in standardized fingerprint space; and (3) discretizing the angle ordering supplies a useful single-source surrogate over nearby directional templates. Supplementary Methods 1 gives a representative Kirchhoff-Love plate operator and Green's function derivation for assumption (1).

To make that local-overlap picture explicit, we also use a reduced-order surrogate in which a fingerprint is approximated by one dominant calibrated direction plus a small number of nearby corrections:

$$z \approx A\,x, \quad \text{subject to} \quad \lVert x \rVert_0 \le K, \qquad (2)$$

where \(z\) is the reduced fingerprint, \(A\) is the reduced template matrix, \(x\) is a sparse coefficient vector whose dominant support identifies the source direction, and \(K\) is a small residual-correction budget for local overlap and noise. This surrogate is used only to motivate the readout geometry: if nearby calibrated directions reuse related reference templates, the ambiguity that must be resolved should be local rather than global.

Once direction is discretized onto the measured angle grid, hard OMP provides the classical greedy sparse-recovery baseline for this local-overlap surrogate. Supplementary Methods 3 gives the recursion explicitly. Figure 3 uses the corresponding stage-0 correlation profile as a diagnostic of how strongly a fingerprint concentrates on one calibrated angle before any residual refitting.

The guided solver keeps the same \(K\)-stage residual-correction scaffold but replaces the hard one-angle selection with learned local gating. Its role is narrow: preserve the contiguous angle neighborhood implied by \(H\) long enough for subtraction to act on a physically plausible local support. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient vector as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the physical match score between the residual and every grouped template is

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual. In the reported implementation, this local gate is parameterized with a compact transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary configuration) that uses the current residual to query the grouped templates and produces direction-group scores

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. In the reported implementation, selection uses a straight-through Gumbel approximation in the forward pass, so this step acts as a discrete local gate over neighboring angle hypotheses. The gated weights then modulate the physical match scores to produce a sparse update:

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication. The sparse coefficient vector is accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter, and the residual is corrected by template-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\). Direction supervision and final prediction come from the same direction-group scores:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t^{(\mathrm{exp})}[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). Supplementary Methods 4 provides the complete grouped formulation.

The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the final readout \(\bar s[e]\). In the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The executed loss weights are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization uses Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Calibration protocol and outer-fold cross-validation
White-noise calibration builds the fixed template matrix \(H\): for each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over all three calibration clips, and those clips are never reused for training or evaluation of the direction readout.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech readout surfaces use the fixed \(H\) estimated from white-noise calibration only. Three manuscript-facing evaluation surfaces are then kept distinct: representative mechanism panels use validation routing outputs from the selected fold, Fig. 5b,d use representative clean displays from selected runs, and Fig. 4d together with the sweep-derived panels in Fig. 5a,e use governed five-seed clean or SNR sweep artifacts built from best-validation checkpoints (`results/figure4_data.json` and `results/fig05_panel_f_no_type_bias_clean_seed_means/summary.npz`). Held-out outer-test predictions are used only where that surface is stated explicitly.

For the five-object comparison in Fig. 6, object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). For Fig. 6c, each material-specific template matrix is then analyzed with the same row-wise mean-centered magnitude SVD used for Fig. 2, that is, on \(|H| - \mathrm{mean}_{\theta}(|H|)\). For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) readout tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration, rather than cross-object weight sharing.

### Discriminability analysis
Before evaluating any learned readout family, Fig. 3 quantifies whether directional structure survives a change in source content. White-noise and speech stimuli are compared using the same calibrated matrix \(H\). Because white noise is the probe used to construct \(H\), whereas speech is sparse and non-stationary in time-frequency, this contrast tests whether discriminability remains tied to the measured directional response when short windows sample that calibration map unevenly across frequency. For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the measured fingerprints preserve directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

### Dose-response and noise robustness
For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise. Panel \(d\) and panel \(f\) are therefore kept distinct: panel \(d\) uses held-out clean speech and white-noise fingerprints from the matched calibration environment, whereas panel \(f\) combines synthetic noisy white-noise datasets with a separate precomputed five-seed speech-plus-babble sweep. The two curves are aligned as stage-0 first-choice diagnostics, not as one continuous held-out-speech surface. For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and template matrix for each level.

### Readout family comparisons
Figures 4 and 5 display four readout surfaces that share the same matched-calibration dictionary but differ in how they handle overlap among nearby calibrated directions: *guided solver* first concentrates support within the locally matching neighborhood, *router-bypass* keeps the same staged updates without that concentration, *OMP baseline* applies the paper-facing soft-OMP family without learned local routing, and *dense routing* spreads activation across experts without sparse local concentration. Figure 3 reports the correlation-based greedy diagnostic separately because immediate single-choice matching already degrades under content variation before full residual correction is applied.

### Evaluation metrics and statistics
Top-1 accuracy is reported on the discrete angle grid. Angular error is the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\), and root-mean-square error (RMSE) = \(\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\). Unless noted otherwise, scalar speech-readout metrics from the outer-fold protocol are reported as mean ± s.d. across the five held-out test folds. Figure 4d and the sweep-derived panels in Fig. 5a,e instead report governed five-seed sweep summaries from best-validation checkpoints, with mean ± s.e.m. across sweep seeds; Fig. 5b,d are representative clean displays. In Fig. 3f, the white-noise signal curve uses clip-level SEM within each synthetic SNR dataset, whereas the speech-plus-babble curve uses SEM across the five precomputed sweep runs. When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure panel. Effect sizes are reported as Cohen's \(d\); distribution comparisons use the two-sided Mann-Whitney \(U\) test.

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
