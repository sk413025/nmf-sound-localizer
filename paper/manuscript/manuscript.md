# Matched calibration reveals a recurring locally ordered directional code in passive objects from single-point vibrometry

## Abstract
Passive objects can encode sound direction in their vibration patterns. The open question is whether one fixed vibrometric measurement can recover that direction after calibration and whether the same directional organization persists across materially different objects. Here, matched calibration on a 37-angle half-plane (0-180° in 5° steps) shows that a passive acrylic plate produces repeatable directional fingerprints in a single non-contact laser Doppler vibrometer (LDV) readout. Held-out speech preserves that structure while its sparse and non-stationary time-frequency content samples the calibrated response unevenly across short windows and broadens overlap among nearby angles. Readout fails mainly when several neighboring directions become jointly plausible and the decoder commits too early to one of them. A physics-guided solver that preserves local evidence before subtraction stabilizes clean-condition readout and recapitulates the measured local neighborhood in its learned representation. Across five materially different passive objects, structured angle-frequency fingerprints, finite neighborhoods of positive local ordering, and above-chance readout recur under matched calibration, with object-conditioned informative bands and overlap widths. Together, these results show that matched calibration can recover single-point structural-acoustic direction readout from passive objects because a recurring locally ordered directional code persists across the tested structures.

## Introduction
Sound direction is usually measured with arrays or specialized vector sensors. Passive structures may already encode it in how they vibrate under directional forcing. The question here is whether one fixed vibrometric measurement, taken at one point on the object, can recover that direction after object-specific calibration. In this framing, the object itself becomes the sensing element, and the main readout challenge is whether nearby directions stay distinct enough to be separated from one another.

Our working hypothesis is that direction changes how sound couples into an object's vibration modes, and that a fixed measurement point can record those changes as repeatable spectra. Incident sound excites the object, the LDV records the resulting fingerprint, and matched calibration organizes those fingerprints into a template matrix \(H\). If nearby directions reuse related spectral structure, then source-content changes should broaden ambiguity mainly within local angle neighborhoods instead of erasing the directional code altogether.

That hypothesis makes two concrete predictions. First, calibration should reveal a compact measured space in which neighboring directions stay locally ordered. Second, white noise and speech should interrogate that space differently. White noise fills the analysis band at each short window and should excite the available resonant pathways densely. Speech is sparse and non-stationary and should sample the same calibration map unevenly across short windows. If nearby directions reuse related spectral structure, that uneven sampling should blur separation first within local angle neighborhoods. Stable decoding would then depend on pooling local evidence across neighboring directions before a final angle is committed. The resulting readout problem is staged rather than instantaneous: evidence must accumulate across a locally coherent neighborhood before subtraction can safely sharpen the estimate.

Several lines of prior work motivate that possibility, but they stop short of the present setting. Single-sensor studies have reported direction-dependent transfer functions in engineered scatterers and in structure-borne measurements [@elbadawy2018lego_doa; @dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Calibrated inverse formulations on a discrete angle grid provide one useful comparison class once nearby overlap must be resolved [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling; @chen2001basis_pursuit; @tibshirani1996lasso; @mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Within that comparison class, greedy pursuit methods are especially brittle when neighboring calibrated fingerprints are locally coherent and short speech windows sample them unevenly, because the first orthogonal selection step must isolate one template before any residual refitting can recover coupled local evidence. Array-free localization is also possible with acoustic vector sensors, but those approaches require specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. The open question is whether an unmodified passive structure can provide a reproducible calibrated pathway for single-point directional readout.

We test that possibility with a non-contact laser Doppler vibrometer (LDV). LDV provides high-bandwidth, high-sensitivity vibration measurement [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration], whereas piezoelectric patches and accelerometers can alter local mass, stiffness, and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. More broadly, the study sits alongside wave systems that treat complex scattering as an encoding resource rather than as noise [@fink1997time; @duarte2008single; @gigan2022imaging_computing_disorder; @rotter2017complex_media; @jiang2020randomized_metamaterial]. We use the acrylic plate as a reference structure to dissect the overlap mechanism and derive the readout constraint, then test whether residual updates that preserve local evidence before commitment satisfy that constraint. The paper-level question is broader: across ordinary passive objects, does matched calibration reveal one recurring locally ordered directional code, or is the acrylic plate an isolated special case?

## Results

### A passive plate turns sound direction into structured single-point vibration fingerprints (Fig. 1)
In the acrylic plate used throughout the main reference-object study, broadband white-noise excitation produces direction-dependent vibration fingerprints in a single LDV readout. The same flat input is redistributed into distinct output spectra at different directions (Fig. 1c), and the full angle-frequency map shows that this redistribution stays organized across the 37-angle grid instead of appearing only in selected traces (Fig. 1d). An empirical component bridge further shows that the measured response can be summarized by a small set of reusable centered-magnitude spectral patterns whose relative weights change across angle (Fig. 1b). Incident sound therefore drives a structured plate response at different directions, and the LDV samples that response at one fixed point. Because the readout is optical and non-contact, it preserves the target's native boundary conditions and avoids sensor loading that could otherwise alter the measured response [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Different frequency bands then emphasize different directional sectors (Fig. 1e), consistent with multiple dispersive vibration pathways contributing with angle-dependent weights (Fig. 2b,c) [@rotter2017complex_media; @kuttruff2025room_acoustics]. Together, the representative spectra, the full angle-frequency map, and the band-limited directivity patterns show that direction is encoded as structured single-point vibration fingerprints. The next question is whether matched calibration compresses those fingerprints into a compact measured space that makes the directional code readable.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | A passive plate turns sound direction into structured single-point vibration fingerprints.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Empirical component bridge. Three reusable centered-magnitude component spectra are shown together with their relative weight shares at representative angles (0°, 90°, 180°), illustrating how direction changes the mixture of shared spectral patterns across angle.
c, Broadband spectral reshaping under matched calibration. The flat white-noise source spectrum (grey dashed) is redistributed differently at five representative angles (0°, 45°, 90°, 135°, 180°), showing that direction changes the measured output spectrum.
d, Full angle-frequency heatmap of the mean white-noise calibration response across the 37 measured angles (0°–180°). The response varies systematically across both angle and frequency, making the directional fingerprint visible as organized full-band structure.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz). Different bands emphasize different directional sectors, indicating that the directional response is frequency-dependent.

### Matched calibration compresses those fingerprints into a compact local directional map (Fig. 2)
Matched calibration compresses the measured fingerprints into a locally ordered geometry. In the full angle-frequency matrix, neighboring directions retain related structure; reconstruction error falls sharply once a small retained basis is kept; and the inter-angle similarity map is strongest near the diagonal, with a compact 2D embedding inset providing the same angle-ordered geometry in a lower-dimensional view (Fig. 2d-f). For the centered-magnitude view used in Fig. 2,

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|
$$

the spectrum decays rapidly (Fig. 2a-c): the first six components capture 80.3% of the energy and eight capture 85.1%. The measured fingerprints therefore occupy a compact angle-ordered space with strong local continuity across nearby directions.

That organization is consistent with a low-dimensional physical response. Under small-amplitude dynamics, the single-point velocity response can be written as a superposition of dispersive structural modes, each with its own spectral pattern \(s_m(\omega)\) and direction-dependent coupling \(\alpha_m(\theta)\):

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the complex frequency response at the LDV measurement point for incidence direction \(\theta\), and the sum runs over a limited set of contributing modes (Supplementary Methods 1) [@ewins2000modal; @meirovitch2001fundamentals]. In this view, changing direction mainly reweights a limited set of dispersive responses, so neighboring angles reuse related spectral structure instead of generating fully independent fingerprints. The centered-magnitude components should therefore be read as empirical summaries of shared spectral structure, not as resolved physical eigenmodes. The inter-angle similarity map shows that this shared structure stays concentrated among neighboring angles.

Calibration therefore sets the form of the decoding problem. Once the measured fingerprints are arranged on a discrete angle grid, direction recovery is no longer a choice among 37 unrelated references. The main ambiguity is local: neighboring templates can become jointly plausible before a final angle is chosen. Held-out speech next tests whether that local organization survives a realistic source change or whether the directional code collapses once the excitation becomes sparse and non-stationary.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Matched calibration compresses directional fingerprints into a compact local code.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. Energy accumulates rapidly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. The auxiliary angle-ordering proxy rises over the same early-component regime.
b, Frequency-selective spectra \(|u_r(f)|\) for representative components 1, 2, and 6. These traces summarize three reusable spectral patterns in the centered-magnitude decomposition.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative components 1, 2, and 6, showing how those same components vary across 0°-180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing that neighboring directions reuse related spectral structure across the full measured grid.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-component regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that local angle ordering stays concentrated among neighboring directions. Inset, a 2D embedding of the same centered-\(|H|\) geometry gives a compact visual summary of the curved angle-ordered trajectory implied by that local similarity structure.

### Speech preserves directional structure while broadening nearby-angle overlap (Fig. 3)
Held-out speech tests whether the compact local structure identified in Figs. 1 and 2 survives a change in excitation. White noise is the calibration probe used to estimate the template matrix \(H\), and because it fills the analysis band in each short window it samples the plate's directional response densely. Speech probes the same structure differently: its energy is sparse and non-stationary in time-frequency, concentrated in moving harmonics and formants instead of spread uniformly across the band, so short windows need not sample the calibrated response uniformly across frequency. Because Fig. 2f showed that neighboring angles already reuse related spectral structure, that uneven sampling should blur separation first within local angle neighborhoods instead of erasing similarity globally.

The observed statistics match that local-overlap picture. Under white-noise excitation, within-angle Pearson correlation is near-perfect and clearly separated from between-angle correlation (\(\bar{r} = 1.000\) versus 0.724; \(d = 2.83\); Fig. 3a). Under speech, within-angle similarity drops to 0.907 and still exceeds between-angle similarity (0.798), with \(d = 1.95\) (Fig. 3b). The per-angle discriminability margin stays positive at every angle, although its mean decreases from 0.28 under white noise to 0.11 under speech (Fig. 3c). Speech therefore preserves the directional code while broadening overlap among nearby calibrated angles.

The first discrete commitment is where that broader local overlap becomes operationally costly. The correlation-based first-choice diagnostic isolates that initial greedy decision before any residual correction. It succeeds on most white-noise fingerprints, where the pairwise similarity map is close to identity, but it collapses on held-out speech, where similarity spreads across neighboring calibrated angles while staying locally ordered (Fig. 3d,e). Once several adjacent templates become jointly plausible, immediate one-angle commitment becomes unstable even though directional structure persists.

Added noise worsens the same overlap problem. The two dose-response curves come from separate matched stage-0 diagnostic surfaces: the white-noise curve is recomputed on synthetic noisy white-noise datasets, whereas the speech curve comes from a separate five-seed speech-plus-babble sweep. Within those matched surfaces, first-choice accuracy declines monotonically as noise increases (Fig. 3f). Figure 3 therefore does not by itself identify one algorithm; it isolates the readout requirement instead: local evidence among neighboring calibrated angles must be preserved long enough for subtraction to act on the correct neighborhood.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Speech preserves directional structure while broadening nearby-angle overlap.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); within-angle similarity stays higher than between-angle similarity despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is reduced but positive at all angles.
d, Stage-0 correlation-based first-choice diagnostic across angle for white noise and held-out speech, shown with light clip-level uncertainty bands. The diagnostic stays strong on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response of the same stage-0 correlation-based first-choice diagnostic versus SNR. The white-noise signal curve (blue, clip-level standard error of the mean (SEM) shading) is recomputed on synthetic noisy white-noise datasets, and the speech-plus-babble curve (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading) comes from a separate five-seed sweep; within those matched surfaces, both decline monotonically with increasing noise.

### Local evidence must be preserved before subtraction can sharpen directional readout (Fig. 4)
Speech broadens evidence across neighboring calibrated angles without erasing the directional code. The guided solver therefore treats those adjacent angles as a coupled local band while it scores the residual, pools that support within the matched neighborhood, and only then applies the residual-correction step [@gregor2010lista; @monga2021unrolling].

Figure 4a anchors that mechanism against the measured local structure itself: the same target-centered band in \(H\) that defines the physically plausible neighborhood is shown beside the staged broad-match, local-gate, and local-update profiles. In a representative 70° validation exemplar, broad matching initially excites multiple nearby calibrated angles above that measured band (Fig. 4b), whereas the local gate contracts the support around the correct neighborhood before the update stays confined there (Fig. 4c). The residual then drops from 1.00 to 0.48 after the first guided step and to 0.47 after the second, while the validation-wide fraction of update mass within 15° rises from 0.18 before the local step to 0.98 after it (Fig. 4d).

That same constraint appears in the clean-condition decoder comparison on the fixed dictionary. Router-bypass keeps staged residual updates but removes local pooling, OMP commits to a single support element while the local band is still shared, and dense routing distributes activity without preserving the measured angle order. The clean-condition means follow the same ordering: 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing (Fig. 4e). In this setting, subtraction works best when it acts on the measured local band before the solver commits to one angle or discards spatial continuity. The next question is whether that algorithmic gain reflects the physical neighborhood revealed by calibration or merely an internal structure that happens to score well.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | Local evidence must be preserved before subtraction can sharpen directional readout.**
a, Architecture and physics correspondence. The measured local band in \(H\) is shown beside the staged broad-match, local-gate, and local-update profiles on the shared angle axis, defining the neighborhood that should be preserved before subtraction.
b, Broad initial match for a representative 70° validation clip. Broad residual-to-dictionary matching excites several nearby calibrated angles above the measured local band in \(H\).
c, Local gate convergence for the same representative clip. Learned local pooling contracts support around the correct neighborhood and the resulting update stays confined there.
d, Residual purification. Left, relative residual norm across the guided steps for the representative clip. Right, validation-wide cumulative update mass within radius before and after one local step across 1,924 clips; the mass within 15° rises from 0.18 to 0.98.
e, Clean-condition Top-1 comparison of four overlap-handling rules on the same fixed dictionary (individual seeds shown as dots; large markers and error bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing, consistent with accurate subtraction requiring preservation of the local angle band.

### The guided solver recapitulates the measured local neighborhood in its learned representation (Fig. 5)
Having established that delayed commitment stabilizes decoding, we next test whether the guided solver preserves the physical neighborhood structure seen in measurement. The calibrated template matrix \(H\) shows a near-diagonal band of local overlap, and the guided-solver neighborhood-emphasis map retains the same coarse angle ordering on that shared angle frame (Fig. 5d,e). Quantitatively, the full-matrix agreement is substantial at \(r = 0.47\), and the min-max normalized per-angle local-band profiles align at \(r = 0.46\) with a mean absolute gap of 0.23 (Fig. 5f). The learned readout therefore recapitulates the measured local band at the level of coarse neighborhood structure even though it does not reproduce every matrix entry exactly.

That preserved local organization shapes the predictions themselves. Under clean evaluation, guided decoding stays concentrated near the diagonal, whereas the OMP baseline fractures into broader off-diagonal errors once it commits too early (Fig. 5c). The same ordering appears in the per-angle accuracy profiles: delayed commitment retains the highest clean accuracy across most of the 37 measured directions, while router-bypass loses more broadly and dense routing stays near chance (Fig. 5b).

Across SNR, the same local-band organization continues to matter. The guided solver degrades least as added noise increases (Fig. 5a), consistent with delayed commitment preserving the local evidence needed for stable subtraction before a final one-angle decision. The structural fidelity established here lets us treat the guided solver as a calibrated probe and ask whether the same locally ordered directional code recurs beyond the acrylic reference object.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | The guided solver recapitulates the measured physical neighborhood in its learned representation.**
a, Five-seed SNR sweep for the same four overlap-handling rules. The delayed-commitment rule degrades least as noise increases.
b, Per-angle clean accuracy: five-seed mean \(P(\mathrm{correct})\) across the 37 measured angles, displayed as a 3-angle centered moving-average view with light \(\pm 1\) s.e.m. shading for the same four overlap-handling rules on the clean sweep. The delayed-commitment rule retains the highest clean mean accuracy overall, and dense routing stays near chance across almost the entire angle set.
c, Row-normalized clean confusion comparison of the OMP baseline and guided solver. Guided decoding stays concentrated near the diagonal, whereas the OMP baseline shows broader off-diagonal fracture after early commitment.
d, Measured local structure in the calibrated fingerprint space. The near-diagonal band summarizes the physical neighborhood that nearby angles share.
e, Learned neighborhood-emphasis map from the guided solver on the same angle frame and correlation scale as panel d. The learned map keeps a similar coarse near-diagonal ordering instead of dispersing weight broadly across the grid.
f, Quantitative structure alignment. Top, min-max normalized local-band profiles from the measured and learned maps across angle. Bottom, concordance scatter of the same per-angle local-band scores. The full-matrix agreement is substantial at \(r = 0.47\), and the per-angle local-band profiles align at \(r = 0.46\) (mean absolute gap \(= 0.23\)).

### A recurring locally ordered directional code spans materially different passive objects under matched calibration (Fig. 6)
With the guided solver acting as a calibrated probe of local overlap, we can now ask whether the same directional organization is specific to the acrylic plate or recurs across other passive objects. Across five materially different passive objects, structured angle-frequency fingerprints, finite neighborhoods of positive local ordering, and above-chance readout all reappear under matched calibration. The five objects occupy different response regimes when placed by correlation-decay width and effective rank (Fig. 6a), yet each still yields an angle-frequency fingerprint matrix \(H\) with visible directional structure (Fig. 6b). The shared regularity is that, for every object, the centered-\(|H|\) correlation stays positive over a finite neighborhood of nearby angles before decaying, even though that decay width differs across objects (Fig. 6c).

Matched object-specific calibration keeps Top-1 readout above chance across the full set (Fig. 6d). Across objects, mean Top-1 aligns more informatively with mean top-3 subspace overlap burden than with normalized overall \(|H|\) energy alone. The paper cup, for example, carries the strongest normalized overall energy yet still sits at broader overlap burden and lower readout than cardboard. A shared contrast-selection rule also identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands (Fig. 6e). What recurs across the tested set is therefore not one shared spectral template, but one recurring locally ordered organization expressed in different object-conditioned frequency regimes.

Each object therefore requires its own calibration because overlap width, compactness, and informative band are object-specific. Even so, the recurrence of a finite locally ordered neighborhood, structured \(H\) matrices, and above-chance readout across five materials shows that the acrylic plate is not a singular exception within the tested object set. What recurs across materials is a bounded but robust directional-code structure that a matched decoder can exploit under matched calibration.

![](../figures/fig06_universality.jpg)

**Fig. 6 | A recurring locally ordered directional code spans materially different passive objects under matched calibration.**
a, Measured response-regime map across the five target objects. Each object is placed by the zero crossing of its mean centered-\(|H|\) correlation-decay curve and by the effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means stay above chance across the five objects. Object mean Top-1 is plotted against mean top-3 subspace overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Response energy alone does not order the objects; the overlap-burden metric provides the more informative comparison axis.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. Informative frequency structure is object-conditioned across the set instead of collapsing to one shared band.

## Discussion
Matched calibration reveals that materially different passive objects can carry a recurring locally ordered directional code in a single-point vibrometric readout. Across the tested set, that code appears as structured angle-frequency fingerprints, finite neighborhoods of positive local ordering, and above-chance Top-1 readout after object-specific calibration. The acrylic plate establishes the mechanism-level readout story for one reference structure. The broader result is that the same directional organization reappears across objects whose overlap widths, compactness, and informative bands differ substantially.

That cross-object recurrence sharpens how the acrylic results should be interpreted. At one fixed LDV position, the same flat input is redistributed into angle-dependent spectra, the resulting centered-magnitude fingerprints occupy a compact locally ordered space, and held-out speech preserves that ordering even as it broadens overlap among neighboring calibrated directions [@gigan2022imaging_computing_disorder; @rotter2017complex_media]. The central decoding problem in this setting is resolving shared local support among nearby calibrated angles before subtraction destroys it. The guided solver matters because it improves accuracy on the acrylic plate and preserves the measured local neighborhood strongly enough to expose the same directional-code structure across the wider object set.

These findings connect to studies that use engineered disorder or metasurfaces as computational wave encoders [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa]. Here, matched calibration exposes usable direction-dependent vibration structure in unmodified passive objects instead of a purpose-built scatterer. The work also complements single-sensor localization with embedded microphones and contact-mounted piezoelectric sensors [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming], because non-contact LDV readout preserves native boundary conditions and avoids sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. More broadly, the recurrence of a locally ordered code suggests that passive structural complexity can act as a usable directional encoding resource instead of an obstacle to be averaged away.

These conclusions are bounded to the tested half-plane grid (0-180° at 5° spacing), the controlled acoustic environment, the single static source, and per-object matched calibration. The LDV is still a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. Extending the result to wider object variation, reduced calibration, stronger damping, reverberation, multiple simultaneous sources, moving emitters, or compact optical and MEMS readouts will require new measurements.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for template-matrix calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end direction decoding and robustness experiments. The four frequency bands in Fig. 1e (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
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

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech analyses use the fixed \(H\) estimated from white-noise calibration. The mechanism panels in Fig. 4b,c draw from validation routing outputs in the selected fold, Fig. 5b,d use representative clean displays from selected runs, and Fig. 4d together with the sweep-derived panels in Fig. 5a,e use governed five-seed clean or SNR sweep artifacts built from best-validation checkpoints (`results/figure4_data.json` and `results/fig05_panel_f_no_type_bias_clean_seed_means/summary.npz`). Held-out outer-test predictions are reported only where noted.

For the five-object comparison in Fig. 6, object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). The panel-a descriptor map and panel-c decay curves are computed from the row-wise mean-centered magnitude matrix \(|H| - \mathrm{mean}_{\theta}(|H|)\): for each object, we form the mean inter-angle correlation as a function of angular separation and record the first separation at which that mean correlation becomes non-positive. Effective rank is taken from the singular-value spectrum of the same centered-\(|H|\) matrix. For panel e, the representative frequency window is defined by the peak of a smoothed across-angle contrast profile \(\mathrm{std}_{\theta}(|H|)/\mathrm{mean}_{\theta}(|H|)\), and the corresponding band-limited directional code is obtained by averaging \(|H|\) across that three-bin window. For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) readout tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration.

### Diagnostic analyses
Figure 3 quantifies whether directional structure survives a change in source content. White-noise and speech stimuli are compared using the same calibrated matrix \(H\). Because white noise is the probe used to construct \(H\), whereas speech is sparse and non-stationary in time-frequency, this contrast tests whether discriminability stays tied to the measured directional response when short windows sample that calibration map unevenly across frequency. For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the measured fingerprints preserve directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise. Panel \(d\) and panel \(f\) are kept distinct: panel \(d\) uses held-out clean speech and white-noise fingerprints from the matched calibration environment, whereas panel \(f\) combines synthetic noisy white-noise datasets with a separate precomputed five-seed speech-plus-babble sweep. The two curves are aligned as stage-0 first-choice diagnostics, not as one continuous held-out-speech surface. For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and template matrix for each level.

Figures 4 and 5 hold the matched-calibration dictionary fixed and compare how different overlap-handling rules change residual cleanup, prediction structure, and noise robustness once local ambiguity appears. *Guided solver* first pools support within the locally matching neighborhood, *router-bypass* keeps the same staged updates without that pooling, *OMP baseline* applies the soft-OMP comparison without learned local routing, and *dense routing* spreads activation across experts without sparse local concentration. Figure 3 reports the correlation-based greedy diagnostic separately because immediate single-choice matching already degrades under content variation before full residual correction is applied.

### Evaluation metrics and statistics
Top-1 accuracy is reported on the discrete angle grid. Angular error is the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\), and root-mean-square error (RMSE) = \(\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\). Unless noted otherwise, scalar speech-readout metrics from the outer-fold protocol are reported as mean ± s.d. across the five held-out test folds. Figure 4d and the sweep-derived panels in Fig. 5a,e instead report governed five-seed sweep summaries from best-validation checkpoints, with mean ± s.e.m. across sweep seeds; Fig. 5b,d are representative clean displays. In Fig. 3f, the white-noise signal curve uses clip-level SEM within each synthetic SNR dataset, whereas the speech-plus-babble curve uses SEM across the five precomputed sweep runs. When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure panel. Effect sizes are reported as Cohen's \(d\); distribution comparisons use the two-sided Mann-Whitney \(U\) test.

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
