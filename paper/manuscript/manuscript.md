# Non-contact acoustic direction sensing via physical encoding in everyday objects

## Abstract
Everyday objects can act as acoustic direction encoders: incoming sound excites direction-dependent structural vibrations, and a single non-contact laser Doppler vibrometer can read the resulting spectral fingerprint from one measurement point rather than from a microphone array. In a calibrated benchmark on a 37-angle half-plane grid (0-180° in 5° steps), these fingerprints were repeatable and organized on a low-dimensional manifold, indicating that directional information is concentrated in a small number of dominant channels rather than in 37 unrelated templates. This structure motivated a calibration-based sparse surrogate. A physics-guided routed solver then decoded held-out speech down to 0 dB in our additive-noise benchmark, whereas a classical greedy baseline collapsed under content variation. The learned routing weights aligned with the smooth physical angle manifold, linking solver behavior back to the measured structural response. In an exploratory five-object screen, direction-dependent fingerprints and low-rank continuity recurred under object-specific calibration, while differences between objects remained consistent with shifts in where usable directional information was concentrated across frequency.

## Introduction
Acoustic direction sensing usually works by comparing what arrives at multiple microphones. That strategy underlies beamforming, MUSIC, ESPRIT, and related array-processing methods [@capon1969fkw; @schmidt1986music; @roy1989esprit; @krim1996array; @vantrees2002optimum; @johnson1993array; @brandstein2001microphone_arrays; @benesty2008microphone_array]. It is powerful, but it ties performance to aperture size and sensor placement, which can be difficult to accommodate in wearables, structural monitoring systems, and other compact or harsh settings [@grumiaux2022_ssl_survey_deep_learning]. The central question of this study is whether directional information can instead be encoded by the object being observed, so that direction can be read from a single measurement point rather than from a spatial array.

Our working hypothesis is simple. When sound strikes an everyday object, the object does not respond as a rigid point detector; it supports direction-dependent mixtures of dispersive structural vibrations. If those mixtures are reproducible, then a single-point vibration measurement should carry a spectral fingerprint of direction. This framing turns the object from passive obstacle into active encoder: direction is first written into structural dynamics, then read out optically as a spectrum. The paper tests that hypothesis in a calibrated benchmark and asks whether the resulting encoding is physically compressible and practically decodable.

Several lines of prior work motivate this possibility, but they do not yet establish a general mechanism in ordinary objects. Single-sensor studies have shown direction-dependent transfer functions in engineered scatterers and in structure-borne measurements [@elbadawy2018lego_doa; @dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Sparse reconstruction provides a natural inference language once direction is discretized onto an angular manifold [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling; @chen2001basis_pursuit; @tibshirani1996lasso; @mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free localization is also possible with acoustic vector sensors, but those require specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. What remains unresolved is whether everyday targets themselves provide a reproducible, physically compressible encoding that can be calibrated systematically rather than exploited case by case.

To interrogate that question without mechanically altering the encoder, we use a laser Doppler vibrometer (LDV) as a non-contact readout. LDV provides high-bandwidth, high-sensitivity vibration measurement [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration], whereas piezoelectric patches and accelerometers can perturb local mass, stiffness, and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. The broader conceptual backdrop comes from wave systems in which scattering and disorder are harnessed rather than suppressed, including time-reversal acoustics, single-pixel imaging, transmission-matrix optics, and randomized metamaterials [@fink1997time; @draeger1997one; @duarte2008single; @ing2008lamb; @gigan2022imaging_computing_disorder; @popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media; @hoang2021single_pixel_doa; @jiang2020randomized_metamaterial]. Here we ask whether ordinary objects already supply enough structural complexity to play that role.

## Results

### Everyday objects act as direction-dependent spectral filters (Fig. 1)
As schematized in Fig. 1b and demonstrated experimentally in Fig. 1c-e, everyday objects transform incident sound into direction-dependent spectral fingerprints measurable from a single non-contact point. When driven by broadband white noise, the flat source spectrum is reshaped differently at each incidence angle (Fig. 1c): the structure acts as a direction-dependent spectral filter whose continuous angle-frequency response \(\mathcal H(\theta, f)\) imprints a unique spectral signature for each direction [@rotter2017complex_media; @kuttruff2025room_acoustics]. A non-contact LDV provides the single-point readout while preserving the target's native boundary conditions (Methods: Experimental setup and data acquisition), ensuring that the measured frequency-response structure reflects the object's intrinsic dynamics rather than sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

These fingerprints are highly repeatable. Independent white-noise recordings at the same angle produce nearly identical spectra (Fig. 1d), with trial-to-trial variability far smaller than the between-angle differences. The spectral fingerprint is therefore a stable physical signature of the structure rather than a noise artifact; quantitative discriminability analysis is provided in Fig. 3. The directional encoding is also frequency-selective: different frequency bands exhibit distinct angular response patterns (Fig. 1e), consistent with the excitation of multiple dispersive structural modes whose relative amplitudes depend on incidence direction (Fig. 2b,c). Figure 1 therefore establishes the effect itself and frames the next question: are these angle-specific spectra organized enough to be described by a smaller set of reusable physical channels?

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°-180° for four frequency bands (0.3-0.5, 0.5-1, 1-2, 2-3 kHz), showing that each band carries a distinct directional response pattern.

### Spectral fingerprints arise from a low-dimensional physical manifold (Fig. 2)
Figure 1 established that direction leaves a stable spectral trace. Figure 2 asks whether those traces are organized or merely enumerated. The singular spectrum of the centered-magnitude analysis matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|
$$

decays rapidly (Fig. 2a-c): the first six modes capture 80.3% of the energy and eight capture 85.1%. The full angle-frequency heatmap in Fig. 2d shows what is being compressed, the rank-truncation fidelity curves in Fig. 2e show that a small retained basis already reconstructs the centered fingerprints well, and the inter-angle similarity matrix in Fig. 2f shows that neighboring directions remain close in fingerprint space rather than scattering arbitrarily. The directional code therefore lies on a smooth low-dimensional manifold rather than on 37 unrelated templates.

A simple modal picture explains why this compression is plausible. Under small-amplitude dynamics, the single-point velocity response of an object can be written as a superposition of dispersive structural modes, each with its own spectral pattern \(s_m(\omega)\) and direction-dependent coupling \(\alpha_m(\theta)\):

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the complex frequency response at the LDV measurement point for incidence direction \(\theta\), and the sum runs over a limited set of contributing modes (Supplementary Methods 1) [@ewins2000modal; @meirovitch2001fundamentals]. In this view, changing direction mainly changes the coupling weights \(\alpha_m(\theta)\) across a limited set of dispersive modes, so nearby angles reuse related spectral building blocks rather than generating wholly independent fingerprints. That interpretation is consistent with the mode spectra and polar patterns in Fig. 2b,c and with the near-diagonal structure in Fig. 2f.

Because the fingerprints vary smoothly rather than arbitrarily, a compact sparse surrogate is reasonable. In a retained singular subspace, the standardized fingerprint and calibration dictionary reduce to \(z\) and \(A\), and direction-of-arrival inference becomes a structured sparse inverse problem:

$$z \approx A\,x, \quad \text{subject to} \quad \lVert x \rVert_0 \le K, \qquad (2)$$

where \(z\) is the reduced fingerprint, \(A\) is the reduced dictionary, \(x\) is a sparse coefficient vector whose dominant support identifies the source direction, and \(K\) is the pursuit depth, that is, a small residual-correction budget for local overlap and noise rather than an assumption of multiple simultaneous sources (Methods: Inference algorithms and network architecture). Equation 2 is only a conceptual surrogate for this calibration geometry; the reported decoders below operate directly on the full standardized fingerprints. In practical terms, Fig. 2 shows that nearby angles reuse related spectral building blocks, so later decoders should exploit neighborhood structure rather than treat the 37 directions as unrelated classes.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Physical encoding via spectral-spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum across the full 37-mode basis. Cumulative fraction is shown on the left axis and singular values on the right, emphasizing the early saturation of the centered-magnitude dictionary across the calibrated angle manifold. The first six modes capture 80.3% of the cumulative energy. Extending to eight modes raises the cumulative fraction to 85.1%, and the cumulative direction-of-arrival capacity curve closely tracks the same rise.
b, Frequency-selective spectra \(|u_r(f)|\) for representative Modes 1, 2, and 6 (overlaid). These profiles highlight a dominant broadside-like channel (Mode 1), an edge-weighted grazing-angle-like channel (Mode 2), and an end-fire-like channel with a distinct higher-frequency shoulder (Mode 6).
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) across 0°-180° for representative Modes 1, 2, and 6 (overlaid), showing three physically interpretable directional couplings that define distinct virtual sensing channels.
d, Full angle-frequency heatmap of the magnitude dictionary \(|H|\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation: per-angle centered-magnitude RMSE in the \(H_{\mathrm{fig}}\) representation for ranks 3, 5, and 6, showing that reconstruction fidelity becomes strong within the same six-mode regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band indicates that neighboring angles share similar spectral fingerprints, revealing the smooth physical angle manifold later exploited by the learned router (Fig. 5c).

### Encoding survives content variation but classical decoding fails (Fig. 3)
A useful encoder must survive the move from calibration stimuli to realistic signals. We therefore next ask whether the directional structure identified in Figs. 1 and 2 persists when the controlled white-noise input used for calibration is replaced by held-out speech, and whether a classical sparse decoder can still exploit it.

Under white-noise excitation, within-angle Pearson correlation is near-perfect (\(\bar{r} = 1.000\)) and well separated from between-angle correlation (\(\bar{r} = 0.724\); Cohen's \(d = 2.83\); Fig. 3a). Under speech, content variation reduces the within-angle correlation to \(\bar{r} = 0.907\) but the separation remains highly significant (\(d = 1.95\), \(p < 10^{-4}\); Fig. 3b). To quantify this more precisely, we compute a per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for both stimulus types: white noise yields a mean margin of 0.28, while speech retains a positive margin of 0.11 at every angle (Fig. 3c, with light bootstrap uncertainty shading). The encoding is therefore degraded but not destroyed by content variation.

The failure appears when decoding must commit to a direction. A classical greedy diagnostic on the same calibration dictionary performs strongly on most white-noise fingerprints but collapses to near-chance levels on held-out speech clips (Fig. 3d, with clip-level uncertainty shading). The split-triangle similarity map in Fig. 3e explains why: white noise produces a near-identity structure that favors single-template matching, whereas speech broadens local overlap along the calibration manifold. Once neighboring templates are jointly plausible, immediate one-angle commitment becomes unstable.

Noise worsens the same failure mode. Greedy diagnostic accuracy declines monotonically as additive noise increases in both the white-noise and speech conditions (Fig. 3f), with clip-level uncertainty shown for white noise and a five-seed mean \(\pm\) SEM band shown for speech-babble. The combined evidence therefore points to a decoder bottleneck rather than a loss of encoding: the directional information remains present, but the solver must accumulate neighborhood evidence before collapsing to a single direction. The remaining decoder comparisons therefore use four stable families: guided solver, router-bypass, OMP baseline, and dense routing, while Fig. 3 remains separate as a correlation-based greedy diagnostic. Figure 4 tests that design criterion.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Encoding survives content variation but classical decoding fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based greedy diagnostic traces for white noise and speech, shown with light clip-level uncertainty bands: the greedy diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (diffuse but structured manifold), with the diagonal masked to separate the two regimes.
f, Dose-response curves: correlation-based greedy diagnostic accuracy versus SNR for white-noise signal (blue, clip-level standard error of the mean (SEM) shading) and speech signal with babble noise (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading), both declining monotonically with increasing noise.

### Guided solver resolves dispersive ambiguity (Fig. 4)
Figure 3 identifies the design criterion: speech broadens overlap along the physical angle manifold, so nearby evidence must be integrated before a final directional commitment. In the guided solver (Fig. 4a; Methods: Inference algorithms and network architecture) [@gregor2010lista; @monga2021unrolling], that integration gathers the broad physical match into the same neighborhood so residual correction proceeds on a cleaner ambiguity landscape.

Figure 4 visualizes that physical response on a shared 70° validation exemplar. The initial physical match is broad across neighboring directions (Fig. 4b), nearby evidence concentrates into the same neighborhood, and the resulting update becomes correspondingly localized before residual subtraction (Fig. 4c). Panel d then summarizes the same stagewise localization and update behavior on the angle axis.

Residual correction then changes the next correlation landscape in a dictionary-consistent way (Fig. 4c). The clean-condition ablation in Fig. 4e shows the practical payoff: among the four decoder families defined above, the guided solver outperforms router-bypass, the OMP baseline, and dense routing. Figure 5 asks whether that same advantage persists across the full held-out benchmark and noise sweeps.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | The guided solver sharpens local ambiguity before residual correction.**
a, Architecture: at stage \(t\), the residual \(r_t\) is correlated with the physical dictionary \(D\) to form \(g_t = D^\top r_t\). A transformer encoder converts QK scores \(q_tK_t^\top\) into routed weights that gate the sparse update \(\Delta x_t\), accumulate \(x_{t+1}=x_t+\eta\Delta x_t\), and enforce residual consistency \(r_{t+1}=r_t-D(\eta\Delta x_t)\).
b, Routing formation: angle-conditioned validation summaries at a shared 70° exemplar trace the stage-0 sequence from the broad physical correlation \(g_t(\theta)\), through the angle-aggregated QK score \((q_tK_t^\top)(\theta)\), to the routed weight \(w_t(\theta)\).
c, Gated update and residual correction: the shared 70° exemplar shows that the gated update \(\Delta x_t(\theta)\) localizes mass much more strongly than the raw physical correlation \(g_t(\theta)\). The first residual-consistent update then changes \(g_t = D^\top r_t\) to \(g_{t+1} = D^\top r_{t+1}\); the lower summaries quantify validation-wide localization and residual-norm descent across the unrolled stages.
d, Aggregation bridge: the shared 70° exemplar makes the symbol mapping in panel a explicit by showing the mode-resolved routing tensor \(w_{t,\theta,m}\) and gated-update tensor \(|\eta\hat{\Delta x}_{t,\theta,m}|\), together with the angle-level reductions \(w_t(\theta)=\sum_m w_{t,\theta,m}\) and \(\Delta x_t(\theta)=\|\eta\hat{\Delta x}_{t,\theta,\cdot}\|_2\) used in panels b-c.
e, Routing-mechanism ablation: compact clean-condition comparison across the guided solver, router-bypass, OMP baseline, and dense routing families over the same five-seed sweep (individual seeds shown as dots, points with horizontal bars indicate mean \(\pm\) s.e.m.).

### Guided solver mirrors physical structure and resists noise (Fig. 5)
Figure 4 established the routing mechanism on a representative example. Figure 5 asks whether that same physical idea scales to the full held-out benchmark. Across signal-to-noise ratio levels, the guided solver remains the most robust of the four decoder families, while the OMP baseline and router-bypass degrade substantially and dense routing remains near chance (Fig. 5a). The confusion-family block shows the same ranking geometrically: the guided solver stays close to the diagonal, the OMP baseline spills off axis, router-bypass leaks locally, and dense routing collapses toward a preferred output mode (Fig. 5b).

That improvement remains tied to the measured physics rather than to arbitrary attention patterns. The dictionary \(H\) has a banded correlation structure because nearby angles share related fingerprints, and the guided solver mirrors that same smooth manifold while sharpening its local diagonal concentration (Fig. 5c). The physical manifold therefore remains the primary structure, with the learned response following it rather than replacing it with an unconstrained classifier.

Panels d and e return to prediction behavior on the angle axis. At representative directions, the guided solver sharpens the conditional output around the correct neighborhood relative to router-bypass (Fig. 5d), and across all 37 angles it preserves the same ranking seen in the noise benchmark (Fig. 5e). Together, Figs. 4 and 5 support a simple practical message: decoding improves when evidence is aggregated along the physical manifold before committing to one direction.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | The learned router mirrors physical structure and maintains robust decoding under noise.**
a, SNR degradation curves for the guided solver, router-bypass, OMP baseline, and dense routing, showing graceful degradation under additive noise.
b, Unified row-normalized confusion-family block across the four decoders. The OMP baseline shows broader fragmented off-axis spillover, the guided solver remains sharply diagonal, dense routing collapses toward a single preferred output mode, and router-bypass retains limited local structure with broader leakage.
c, Correlation structure of the physical dictionary \(H\) (top) and the learned router's QK attention map (bottom), showing that learned routing mirrors the physical angle manifold.
d, Angle-specific conditional output distributions at four representative directions (55°, 70°, 95°, and 100°): the guided solver sharpens the prediction profile around the correct angle and its local neighborhood, whereas router-bypass shows broader off-axis leakage.
e, Per-angle decoder accuracy: five-seed clean mean \(P(\mathrm{correct})\) across the 37 measured angles, shown as a 3-angle centered moving-average display with light \(\pm 1\) s.e.m. shading, comparing the guided solver, router-bypass, OMP baseline, and dense routing. The guided solver remains strongest overall, whereas dense routing remains near chance across almost the entire angle set.

### Direction-dependent encoding and low-rank continuity recur in an exploratory five-object screen (Fig. 6)
Figure 5 established that manifold-aligned routing improves decoding on the acrylic benchmark. Figure 6 asks, in an exploratory five-object screen, whether the encoding principle itself recurs across structurally different objects under object-specific calibration. It does. Across the cardboard box, wooden board, acrylic plate, paper cup, and laptop shell (Fig. 6a), object-specific calibration still produces structured angle-frequency transfer functions (Fig. 6b) and above-chance direction recovery (Fig. 6d), so the encoding principle is not confined to one favorable surface.

The same row-wise mean-centered magnitude SVD used in Fig. 2 shows that a modest number of modes capture most centered energy for every object (Fig. 6c), indicating that low-rank continuity also recurs across the five-object screen. The remaining statistics are supportive rather than primary: normalized response energy does not rank the objects in the same order as direction-recovery accuracy, and the angle-level MAE matrix does not reveal detectable cross-material heterogeneity in this committed sample (Friedman chi-square(4) = 1.427136, \(p = 0.839465\)).

The frequency-band summaries in Fig. 6e are therefore exploratory rather than explanatory. They suggest that object-to-object differences may depend on where usable directional information concentrates across frequency, but the current five-object screen does not establish a transferable material law.

![](../figures/fig06_universality.jpg)

**Fig. 6 | Direction-dependent encoding and low-rank continuity recur across materials.**
a, Five target objects spanning a broad range of material and geometric complexity, shown here in screening order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Cross-material \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across all materials despite different resonance patterns.
c, Low-rank continuity. The same row-wise mean-centered magnitude SVD used in Fig. 2 was applied separately to each material-specific dictionary. The resulting cumulative-energy curves and rank90/rank95 summaries show that the encoder remains low-dimensional across the full object set, extending the low-rank physical-dictionary view from Fig. 2 beyond the single acrylic plate.
d, Screening consequence. The left comparison plots normalized overall response energy and Top-1 accuracy for each material, showing that stronger overall response energy does not guarantee higher Top-1 screening accuracy. The adjacent Top-1 and MAE summaries report 95% bootstrap confidence intervals, while the separate angle-level MAE matrix used for the omnibus heterogeneity test does not reveal detectable cross-material heterogeneity in this committed sample.
e, Material frequency structure. Each material is shown as a three-level card: normalized spectral envelope, angular-contrast spectrum, and the representative directional code recovered from that material's selected informative band.

## Discussion
The main point of this benchmark is simple: ordinary objects can help tell us where sound came from. Incoming sound writes direction into the object's structural vibration pattern, and a single non-contact LDV readout can recover that information from one optical spot rather than from a microphone array. The contribution is therefore a sensing principle first and a decoder result second: spatial information can be encoded by the object itself and then read out optically [@gigan2022imaging_computing_disorder; @rotter2017complex_media].

Taken together, the data are consistent with a coherent physical picture. Direction changes the mixture of dispersive modes excited in the object, that response compresses into a low-rank angle-frequency manifold with smooth local geometry, and a solver that respects that geometry decodes ambiguous speech more reliably than classical greedy pursuit (Figs. 1-5). The cross-material screen extends the same picture cautiously: within five sampled objects, direction-dependent fingerprints and low-rank continuity recurred under object-specific calibration, while screening differences were consistent with shifts in where usable directional information was concentrated across frequency rather than with a loss of the encoding principle itself (Fig. 6). At present, however, that interpretation remains descriptive rather than predictive; the data support it as a candidate explanation, not as a universal law across materials or mounting conditions.

These findings connect to, and differ from, several existing lines of work. Prior studies used engineered disorder or metasurfaces as computational wave encoders [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa], whereas our results indicate that unmodified everyday objects can already supply useful structural complexity in the acoustic domain. Single-sensor localization has also been demonstrated with embedded microphones and contact-mounted piezoelectric sensors [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. The present study is complementary rather than competitive with those modalities. Contact and embedded sensors are appealing for integrated devices, whereas non-contact readout is especially useful when the goal is to preserve the object's native response and avoid sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Several limitations bound the present claims. The benchmark uses a half-plane grid (0-180°) at 5° spacing, a controlled acoustic environment, a single static source, and per-object matched calibration. The LDV is a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. The cross-material analysis is a five-object screen, so it supports recurrence within this sample but not transfer across arbitrary objects. Future work should test whether manifold structure can be predicted or adapted with less calibration, how performance changes under stronger damping, reverberation, multiple simultaneous sources, or moving emitters, and whether compact optical or MEMS readouts can preserve enough sensitivity for practical deployment. More broadly, the idea that passive structural complexity can encode wave-field information may prove useful beyond this benchmark, including in vibration sensing and other array-limited settings, but those applications remain to be demonstrated.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for dictionary calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end DOA decoding and robustness experiments. The four frequency bands in Fig. 1e (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
The LDV measures out-of-plane surface velocity \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), where \(W\) is the displacement field and \((x_L, y_L)\) denotes the fixed laser measurement location. For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). For each recorded clip, a short-time Fourier transform (Hann window, 2,048 samples, hop 512) is computed and collapsed into a time-averaged power spectrum:

$$\widehat{S}(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2, \qquad (3)$$

where \(V[k,t]\) is the complex STFT coefficient at frequency bin \(k\) and frame \(t\), and \(T\) is the number of retained frames. The spectrum is restricted to [300, 3,000] Hz (\(F = 346\) bins). Because the observed fingerprints are magnitude statistics, each clip is summarized by a log-power feature vector \(y[k]=\log_{10}(\widehat{S}+\epsilon)\), then standardized per frequency bin using white-noise calibration statistics to yield a normalized feature \(\tilde{y}[k]\) (Supplementary Methods 2). These normalized log-power fingerprints are the observables used for downstream inference. Accordingly, Eq. 2 should be read as a feature-space surrogate derived from the physical response in Eq. 1: the complex response \(Y(\omega;\theta)\) is converted into magnitude statistics, discretized over frequency and angle, and standardized before sparse matching. It is therefore not a literal complex-domain superposition law or a direct power-additivity statement.

### Physical dictionary formulation and centered-magnitude SVD
The normalized features serve as the basis for constructing the angle-indexed dictionary used in inference. Per-angle prototypes are averaged across calibration trials to form columns \(h_e\), where \(e\) indexes one of the \(E\) candidate directions, yielding the standardized fingerprint dictionary \(H=[h_1,\dots,h_{37}]\in\mathbb{R}^{F\times E}\), with \(E=37\). Figure 2 does not decompose the ideal complex response \(\mathcal H\) directly. Instead, it analyzes the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|.
$$

Singular-value decomposition (SVD) of \(H_{\mathrm{fig}}\) shows rapid early saturation (Fig. 2a). Six modes capture 80.3% of the centered-magnitude energy. Extending to eight modes raises the cumulative fraction to 85.1%.

For intuition, one may also form a reduced-order surrogate in a retained singular subspace; this is the reduced-order picture referenced by Eq. 2, while the reported decoding uses the full \(F=346\) standardized feature space without PCA/SVD preprocessing.

### Inference algorithms and network architecture
Using the sparse inverse viewpoint introduced in Results, DOA inference uses the full standardized feature space with a grouped dictionary \(D=[d_{e,m}]\) and no PCA/SVD projection. The formulation rests on three working assumptions: (1) over each analysis window the target behaves as a linear time-invariant system, so direction-dependent responses superpose in the frequency domain; (2) \(Hx\) is a sparse prototype approximation in standardized fingerprint space, not a literal power-additivity law; and (3) discretizing the angle manifold provides a useful single-source surrogate over nearby directional templates. The representative Kirchhoff-Love plate operator and Green's function derivation supporting assumption (1) are given in Supplementary Methods 1.

Supplementary Methods 3 gives the exact analytical hard-OMP recursion for the reduced-order surrogate. In that mathematical baseline, the dictionary atom most correlated with the current residual is iteratively selected, added to the support set, and its contribution removed by least-squares refit and residual subtraction. After \(K=2\) stages, the predicted direction is \(\hat\theta=\theta_{\arg\max_e |x[e]|}\). The displayed Fig. 3 traces, however, are correlation-aggregated greedy diagnostics on the same calibration dictionary rather than literal visualizations of every least-squares refit step.

The physics-guided solver is an OMP-inspired differentiable routed pursuit rather than an exact OMP unrolling. It shares the same \(K\)-stage residual-correction scaffold and physical match score, but replaces hard support selection and least-squares refitting with learned routed gating. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient vector as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the physical match score between the residual and every grouped dictionary atom is computed as:

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual and each entry of \(g_t\) quantifies how well a candidate direction explains the unexplained observation through the physical dictionary. A transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary guided-solver configuration) uses the current residual to query the dictionary and produces expert-level routing scores from scaled dot-product interactions:

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. In the reported configuration, routed selection uses hard Gumbel gating in the forward pass. The routing step therefore behaves as a manifold-aware discrete gate over neighboring angle hypotheses rather than as a purely soft attention weight. These routed weights gate the physical match scores to produce a sparse update:

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication. The sparse coefficient vector is accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter that scales each stage-wise sparse update before residual correction, and the residual is corrected by dictionary-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\), ensuring that each stage explains a portion of the observation through the physical dictionary. Direction supervision and final prediction are read out from expert-level routing scores rather than from the final coefficient energy:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t^{(\mathrm{exp})}[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). The routing weights \(w_t\) can be directly inspected as model-internal support for manifold-aligned selection (Fig. 5c); Supplementary Methods 4 provides the complete grouped formulation.

The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the expert-level readout \(\bar s[e]\). Equation 9 gives the main loss; in the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The loss weights in the executed main objective are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization is performed with Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Calibration protocol and outer-fold cross-validation
The dictionary \(H\) described above is constructed exclusively from white-noise calibration recordings, using all 3 clips per angle. For each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over those calibration clips. Calibration clips are never used for training or evaluation of the DOA decoder.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech results use the fixed \(H\) estimated from white-noise calibration only, and scalar performance metrics are reported from held-out outer-test predictions.

For cross-material screening experiments (Fig. 6), object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). For Fig. 6c, each material-specific dictionary is then analyzed with the same row-wise mean-centered magnitude SVD used for Fig. 2, that is, on \(|H| - \mathrm{mean}_{\theta}(|H|)\). For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) screening tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration, rather than cross-object weight sharing.

### Discriminability analysis
Before evaluating the learned solver, encoding quality is quantified independently of any inference algorithm (Fig. 3). White-noise and speech stimuli are compared using the same dictionary \(H\). For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the encoding preserves directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

### Dose-response and noise robustness
For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise. For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and dictionary for each level.

### Solver ablations
Decoder comparisons in Figs. 4 and 5 use four stable families: *guided solver*, *router-bypass*, *OMP baseline*, and *dense routing*. The OMP baseline is distinct from the correlation-based greedy diagnostic in Fig. 3, which is reported separately to isolate the failure of immediate single-choice matching under content variation.

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
