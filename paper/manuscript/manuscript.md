# Everyday objects as calibrated acoustic direction encoders

## Abstract
After object-specific calibration, an everyday object can encode sound direction in its vibration spectrum, and a single non-contact laser Doppler vibrometer point can read that encoding. In a 37-angle half-plane benchmark (0-180° in 5° steps), an acrylic plate produced repeatable spectral fingerprints that occupied a compact angle-ordered space with strong local reuse across neighboring directions. Held-out speech preserved that directional structure, but immediate one-angle matching became unstable once neighboring calibrated angles overlapped. A neighborhood-aware guided solver then decoded directions more reliably than the tested baselines and remained effective down to 0 dB additive noise. Across an exploratory five-object screen, structured fingerprints and low-rank continuity recurred under object-specific calibration. These results indicate that the object establishes the directional encoding, while the solver determines how reliably that encoding can be read out.

## Introduction
Acoustic direction sensing usually works by comparing what arrives at multiple microphones. That strategy underlies beamforming, MUSIC, ESPRIT, and related array-processing methods [@capon1969fkw; @schmidt1986music; @roy1989esprit; @krim1996array; @vantrees2002optimum; @johnson1993array; @brandstein2001microphone_arrays; @benesty2008microphone_array]. It is powerful, but it ties performance to aperture size and sensor placement, which can be difficult to accommodate in wearables, structural monitoring systems, and other compact or harsh settings [@grumiaux2022_ssl_survey_deep_learning]. The central question of this study is whether directional information can instead be encoded by the object being observed, so that direction can be read from a single measurement point rather than from a spatial array.

Our working hypothesis is simple. When sound strikes an everyday object, the object responds through direction-dependent mixtures of dispersive structural vibrations. If those mixtures are reproducible, then a single-point vibration measurement should carry a spectral fingerprint of direction. In that view, the object becomes part of the measurement pathway: direction-dependent structural response is established in the object and then sampled optically as a spectrum. We test that hypothesis in a calibrated benchmark and ask whether the resulting fingerprints are physically compressible and practically decodable.

Several lines of prior work motivate this possibility, but they do not yet establish a general mechanism in ordinary objects. Single-sensor studies have shown direction-dependent transfer functions in engineered scatterers and in structure-borne measurements [@elbadawy2018lego_doa; @dipassio2022_audio_capture_structural_sensors; @dipassio2023_waspaa_wake_word; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Sparse reconstruction provides one useful inference language once a calibrated angle grid has been defined [@malioutov2005sparse_doa; @donoho2006compressed_sensing; @candes2006robust_uncertainty; @candes2008compressive_sampling; @chen2001basis_pursuit; @tibshirani1996lasso; @mallat1993matching_pursuits; @pati1993omp; @tropp2007omp]. Array-free localization is also possible with acoustic vector sensors, but those require specialized collocated pressure-particle-velocity instrumentation [@nehorai1994vector_sensor]. What remains unresolved is whether everyday targets themselves provide a reproducible, physically compressible encoding that can be calibrated systematically rather than exploited case by case.

To interrogate that question without mechanically altering the encoder, we use a laser Doppler vibrometer (LDV) as a non-contact readout. LDV provides high-bandwidth, high-sensitivity vibration measurement [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration], whereas piezoelectric patches and accelerometers can perturb local mass, stiffness, and boundary conditions [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors]. The broader conceptual backdrop comes from wave systems in which scattering and disorder are harnessed rather than suppressed, including time-reversal acoustics, single-pixel imaging, transmission-matrix optics, and randomized metamaterials [@fink1997time; @draeger1997one; @duarte2008single; @ing2008lamb; @gigan2022imaging_computing_disorder; @popoff2010transmission_matrix; @mosk2012complex_media; @rotter2017complex_media; @hoang2021single_pixel_doa; @jiang2020randomized_metamaterial]. Here we ask whether ordinary objects already supply enough structural complexity to play that role.

## Results

### Everyday objects act as direction-dependent spectral filters (Fig. 1)
We first establish the core effect on the acrylic plate. After calibration, incident sound is converted into stable direction-dependent spectral fingerprints that a single non-contact point can read. When driven by broadband white noise, the flat source spectrum is reshaped differently at each incidence angle (Fig. 1c): the structure acts as a direction-dependent spectral filter whose continuous angle-frequency response \(\mathcal H(\theta, f)\) imprints an angle-specific spectral signature [@rotter2017complex_media; @kuttruff2025room_acoustics]. A non-contact LDV provides the single-point readout while preserving the target's native boundary conditions (Methods: Experimental setup and data acquisition), ensuring that the measured frequency-response structure reflects the object's intrinsic dynamics rather than sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

These fingerprints are highly repeatable. Independent white-noise recordings at the same angle produce nearly identical spectra (Fig. 1d), with trial-to-trial variability far smaller than the between-angle differences. The spectral fingerprint is therefore a stable physical signature of the structure rather than a noise artifact; quantitative discriminability analysis is provided in Fig. 3. The directional encoding is also frequency-selective: different frequency bands exhibit distinct angular response patterns (Fig. 1e), consistent with the excitation of multiple dispersive structural modes whose relative amplitudes depend on incidence direction (Fig. 2b,c). This result frames the next question: are these angle-specific spectra organized enough to be summarized by a smaller set of reusable patterns?

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°-180° for four frequency bands (0.3-0.5, 0.5-1, 1-2, 2-3 kHz), showing that each band carries a distinct directional response pattern.

### Calibration fingerprints occupy a compact angle-ordered space (Fig. 2)
We next ask whether those stable fingerprints collapse to a small reusable basis. The singular spectrum of the centered-magnitude analysis matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|
$$

decays rapidly (Fig. 2a-c): the first six modes capture 80.3% of the energy and eight capture 85.1%. Across the full angle-frequency heatmap in Fig. 2d, the calibrated fingerprints vary systematically with angle; the rank-truncation fidelity curves in Fig. 2e indicate that a small retained basis already reconstructs them well, and the inter-angle similarity matrix in Fig. 2f places neighboring directions close in fingerprint space. The calibrated fingerprints thus occupy a compact angle-ordered space with local continuity across nearby directions.

A simple modal interpretation is consistent with that compression. Under small-amplitude dynamics, the single-point velocity response of an object can be written as a superposition of dispersive structural modes, each with its own spectral pattern \(s_m(\omega)\) and direction-dependent coupling \(\alpha_m(\theta)\):

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the complex frequency response at the LDV measurement point for incidence direction \(\theta\), and the sum runs over a limited set of contributing modes (Supplementary Methods 1) [@ewins2000modal; @meirovitch2001fundamentals]. In this view, changing direction mainly changes the coupling weights \(\alpha_m(\theta)\) across a limited set of dispersive modes, so nearby angles reuse related spectral building blocks. The mode spectra in Fig. 2b,c and the near-diagonal similarity structure in Fig. 2f follow that same organization and support a compact measured description of the calibrated fingerprints.

This low-rank observation yields a bounded engineering readout. After calibration converts the measured response into standardized fingerprints on the discrete angle grid, a held-out fingerprint can be described by one dominant calibrated direction plus a small number of nearby corrections:

$$z \approx A\,x, \quad \text{subject to} \quad \lVert x \rVert_0 \le K, \qquad (2)$$

where \(z\) is the reduced fingerprint, \(A\) is the reduced dictionary, \(x\) is a sparse coefficient vector whose dominant support identifies the source direction, and \(K\) is a small residual-correction budget for local overlap and noise (Methods: Inference algorithms and network architecture). In this reduced view, the dominant support identifies the source direction and the remaining support captures overlap among neighboring calibrated templates. Equation 2 therefore carries the modal picture in Eq. 1 into measured fingerprint space, where reused spectral patterns become a local-overlap code on the calibrated angle grid. The benchmark decoders below operate on the full standardized fingerprints. We next test whether held-out speech broadens that local overlap enough to require explicit neighborhood pooling.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint dictionary. The cumulative curve rises quickly across the 37-angle grid: six modes capture 80.3% of the energy and eight capture 85.1%. The overlaid direction-decoding trace is included as a visual comparison.
b, Frequency-selective spectra \(|u_r(f)|\) for representative Modes 1, 2, and 6. These traces show three reusable spectral patterns in the compressed representation.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative Modes 1, 2, and 6, showing how those same modes vary across 0°-180°.
d, Full angle-frequency heatmap of the magnitude dictionary \(|H|\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-mode regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that neighboring angles remain close in fingerprint space, revealing the local angle ordering later compared with the guided neighborhood map in Fig. 5c.

### Encoding survives content variation but classical decoding fails (Fig. 3)
We next test whether the compact angle ordering identified in Figs. 1 and 2 survives realistic content variation. White noise and speech play distinct roles in that test. White noise is the broadband, near-flat calibration probe used to estimate \(H\), so angular differences primarily expose the object's directional filtering with minimal source structure. Held-out speech then asks the harder question of whether the same object-tied directional geometry remains readable when the source carries structured spectrotemporal variation. We therefore keep the calibration dictionary fixed, replace the calibration probe with held-out speech, and ask two linked questions: does the measured encoding survive content variation, and if it does, what kind of decoder can still read it?

Under white-noise excitation, within-angle Pearson correlation is near-perfect (\(\bar{r} = 1.000\)) and well separated from between-angle correlation (\(\bar{r} = 0.724\); Cohen's \(d = 2.83\); Fig. 3a). Under speech, content variation reduces the within-angle correlation to \(\bar{r} = 0.907\) but the separation remains highly significant (\(d = 1.95\), \(p < 10^{-4}\); Fig. 3b). To quantify this more precisely, we compute a per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for both stimulus types: white noise yields a mean margin of 0.28, while speech retains a positive margin of 0.11 at every angle (Fig. 3c, with light bootstrap uncertainty shading). The encoding is therefore degraded but not destroyed by content variation.

The failure appears at the commitment step. A classical greedy diagnostic on the same calibration dictionary performs strongly on most white-noise fingerprints but collapses to near-chance levels on held-out speech clips (Fig. 3d, with clip-level uncertainty shading). In the split-triangle similarity map in Fig. 3e, white noise produces a near-identity structure that favors single-template matching, whereas speech broadens local overlap across neighboring calibrated angles. Once structured speech makes neighboring templates jointly plausible, immediate one-angle commitment becomes unstable.

Noise worsens the same failure mode. Greedy diagnostic accuracy declines monotonically as additive noise increases in both the white-noise and speech conditions (Fig. 3f), with clip-level uncertainty shown for white noise and a five-seed mean \(\pm\) SEM band shown for speech-babble. Taken together, these results define the decoder problem. Directional information remains present in measured fingerprint space, but a useful decoder now has to pool nearby evidence before making a hard directional choice. The correlation-based greedy diagnostic isolates that early-commitment failure on the calibration dictionary. The next step is therefore to test a local-pooling update and then evaluate its benchmark consequence against the stable decoder families.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Encoding survives content variation but classical decoding fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (\(d = 2.83\), within \(\bar{r} = 1.000\)).
b, Speech stimulus: same analysis (\(d = 1.95\), within \(\bar{r} = 0.907\)); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within \(\bar{r}\) minus between \(\bar{r}\)) for white noise (\(\Delta \bar{r} = 0.28\)) and speech (\(\Delta \bar{r} = 0.11\)), shown with light bootstrap uncertainty bands; the speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based greedy diagnostic traces for white noise and speech, shown with light clip-level uncertainty bands: the greedy diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (diffuse but still angle-ordered), with the diagonal masked to separate the two regimes.
f, Dose-response curves: correlation-based greedy diagnostic accuracy versus SNR for white-noise signal (blue, clip-level standard error of the mean (SEM) shading) and speech signal with babble noise (orange, 5-seed mean \(\pm\) standard error of the mean (SEM) shading), both declining monotonically with increasing noise.

### Local neighborhood routing sharpens ambiguous matches (Fig. 4)
That diagnosis suggests the next engineering move: keep the physical match score broad, pool it locally, and then subtract the resolved component from the residual [@gregor2010lista; @monga2021unrolling]. In the 70° exemplar, speech broadens overlap across neighboring calibrated angles, so the guided solver first concentrates a broad physical match into one local neighborhood and then updates the residual. The update sequence moves from a broad match to local concentration and then to a cleaner residual.

On the shared 70° validation exemplar, the broad physical match \(g_t(\theta)\), the local gate \(w_t(\theta)\), and the localized update \(\Delta x_t(\theta)\) show the broad-to-local transition on one shared axis. After that local step, the next correlation profile \(g_{t+1}(\theta)\) is reduced around the updated neighborhood, while compact callouts report the corresponding inward mass shift and first residual drop.

The clean five-seed comparison in panel d keeps that illustration grounded. In this local view, the guided solver sits highest, router-bypass and the OMP baseline trail, and dense routing performs worst. That local ranking motivates testing the same update rule across the full benchmark.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | Local neighborhood routing sharpens ambiguous matches.**
a, Decoder strip illustrating the broad-match -> local-concentration -> cleaner-residual sequence.
b, Shared 70° example: broad physical match, local gate, and localized update on one axis.
c, Residual cleanup: after the first local update, the next correlation profile \(g_{t+1} = D^\top r_{t+1}\) is reduced near the updated neighborhood. Compact callouts report the corresponding 0-15° inward mass shift and first residual-norm drop.
d, Clean-condition comparison across the guided solver, router-bypass, OMP baseline, and dense routing families over the same five-seed sweep (individual seeds shown as dots, points with horizontal bars indicate mean \(\pm\) s.e.m.).

### Neighborhood-aware decoding remains most robust across the benchmark (Fig. 5)
We then ask whether the same local-pooling rule remains advantageous across the full held-out benchmark. The dictionary \(H\) has a banded correlation pattern because nearby angles share related fingerprints, and the guided neighborhood map places most weight near that same diagonal ordering (Fig. 5c). This alignment indicates that the solver is pooling evidence along the same local angle structure present in the measured fingerprints, rather than imposing an unrelated decoding pattern.

The local prediction views match that picture. In the confusion-family block, the guided solver stays closest to the diagonal, while the OMP baseline spills off axis, router-bypass leaks locally, and dense routing collapses toward a preferred output mode (Fig. 5b). At representative directions, the guided solver also produces tighter local output profiles than router-bypass (Fig. 5d).

That advantage persists across the full benchmark. Across signal-to-noise ratio levels, the guided solver remains the strongest of the four decoder families, while the OMP baseline and router-bypass degrade more sharply and dense routing remains near chance (Fig. 5a). Across all 37 angles, the same ranking persists in the clean-condition accuracy summary (Fig. 5e). Together, these benchmark comparisons indicate that, in this benchmark, decoding is strongest when the solver pools evidence across the local angle structure before committing to one direction.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | Neighborhood-aware decoding remains most robust across the benchmark.**
a, SNR degradation curves for the guided solver, router-bypass, OMP baseline, and dense routing across additive-noise levels.
b, Row-normalized confusion-family block across the four decoders. The guided solver stays most concentrated near the diagonal, whereas the OMP baseline and router-bypass show broader off-axis leakage and dense routing collapses toward a preferred output mode.
c, Correlation structure of the measured physical dictionary \(H\) (top) and the guided neighborhood map (bottom), showing that the learned map places most weight near the same diagonal angle ordering seen in the measured dictionary.
d, Angle-specific conditional output distributions at four representative directions (55°, 70°, 95°, and 100°): the guided solver produces tighter local prediction profiles, whereas router-bypass shows broader off-axis leakage.
e, Per-angle decoder accuracy: five-seed clean mean \(P(\mathrm{correct})\) across the 37 measured angles, shown as a 3-angle centered moving-average display with light \(\pm 1\) s.e.m. shading, comparing the guided solver, router-bypass, OMP baseline, and dense routing. The guided solver remains strongest overall, whereas dense routing remains near chance across almost the entire angle set.

### Direction-dependent fingerprints recur across an exploratory five-object screen (Fig. 6)
Object-specific calibration also yields structured angle-frequency fingerprints in five structurally different objects (Fig. 6a,b). Applying the same centered-magnitude SVD view used in Fig. 2 to each object's dictionary again shows early energy capture (Fig. 6c). Within this sample, both the directional encoding and its compact local continuity recur beyond the acrylic plate.

Across the five-object screen, each object remains above chance, while normalized response energy does not monotonically track Top-1 accuracy across materials (Fig. 6d). The per-object frequency summaries further show that the informative band shifts across objects (Fig. 6e). Together, these exploratory comparisons indicate that object-specific calibration can repeatedly yield readable directional structure within this small sample, while cross-material transfer laws remain open.

![](../figures/fig06_universality.jpg)

**Fig. 6 | Direction-dependent fingerprints recur across an exploratory five-object screen.**
a, Five target objects in screening order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Cross-material \(H\). Angle-frequency maps show structured directional encoding across all materials despite different resonance patterns.
c, Low-rank continuity. Applying the same centered-magnitude SVD view used in Fig. 2 to each material-specific dictionary again shows early energy capture for every object.
d, Screening performance across the five objects. All objects remain above chance in this screen, and the energy-versus-accuracy comparison indicates that Top-1 screening accuracy does not monotonically track overall response energy across this sample. The accompanying Top-1 confidence intervals summarize screening uncertainty while preserving that mismatch across objects.
e, Frequency structure across objects. Material-specific spectra and directional band profiles indicate that the informative frequency band can shift across objects.

## Discussion
After object-specific calibration, ordinary objects can carry measurable cues about where sound came from. A single non-contact LDV readout recovers those cues from one optical spot while avoiding the aperture and placement requirements of a microphone array. In this benchmark, the object supplies the direction-dependent fingerprint, and the solver determines how reliably that fingerprint can be read out [@gigan2022imaging_computing_disorder; @rotter2017complex_media].

Object-specific calibration writes a compact, locally ordered fingerprint into the measured vibration spectrum. Held-out speech preserves that structure while broadening local overlap in measured feature space. Pooling evidence across that local structure then decodes direction more reliably than the tested baselines. This sequence links the physical response in Figs. 1-2 to the decoder benchmark in Figs. 3-5 and supports a modal interpretation in which nearby angles reuse related spectral patterns. The cross-object screen extends this picture within a small sample: direction-dependent fingerprints and low-rank continuity recur under object-specific calibration, while the informative frequency band shifts across objects. A general law across materials or mounting conditions remains for future work.

These findings connect to, and differ from, several existing lines of work. Prior studies used engineered disorder or metasurfaces as computational wave encoders [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa], whereas our results indicate that unmodified everyday objects can already supply useful structural complexity in the acoustic domain. Single-sensor localization has also been demonstrated with embedded microphones and contact-mounted piezoelectric sensors [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. The present study complements those modalities. Contact and embedded sensors are appealing for integrated devices, whereas non-contact readout is especially useful when the goal is to preserve the object's native response and avoid sensor-induced loading [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Several limitations bound the present claims. The benchmark uses a half-plane grid (0-180°) at 5° spacing, a controlled acoustic environment, a single static source, and per-object matched calibration. The LDV is a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. The cross-material analysis is a five-object screen, so it supports recurrence within this sample but not transfer across arbitrary objects. Future work should test whether the observed angle ordering can be predicted or adapted with less calibration, how performance changes under stronger damping, reverberation, multiple simultaneous sources, or moving emitters, and whether compact optical or MEMS readouts can preserve enough sensitivity for practical deployment. More broadly, the idea that passive structural complexity can encode wave-field information may prove useful beyond this benchmark, including in vibration sensing and other array-limited settings, but those applications remain to be demonstrated.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for dictionary calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end direction decoding and robustness experiments. The four frequency bands in Fig. 1e (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
Each clip is reduced to a standardized log-power fingerprint built from the LDV-measured out-of-plane surface velocity \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), where \(W\) is the displacement field and \((x_L, y_L)\) denotes the fixed laser measurement location. For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). For each recorded clip, a short-time Fourier transform (Hann window, 2,048 samples, hop 512) is computed and collapsed into a time-averaged power spectrum:

$$\widehat{S}(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2, \qquad (3)$$

where \(V[k,t]\) is the complex STFT coefficient at frequency bin \(k\) and frame \(t\), and \(T\) is the number of retained frames. The spectrum is restricted to [300, 3,000] Hz (\(F = 346\) bins). Because the observed fingerprints are magnitude statistics, each clip is summarized by a log-power feature vector \(y[k]=\log_{10}(\widehat{S}+\epsilon)\), then standardized per frequency bin using white-noise calibration statistics to yield a normalized feature \(\tilde{y}[k]\) (Supplementary Methods 2). These normalized log-power fingerprints are the observables used for downstream inference. Equation 2 carries Eq. 1 into this measured space: the complex response is summarized as standardized fingerprints on the calibrated angle grid, and a held-out fingerprint is then modeled by one dominant calibrated template plus a small local correction set. In that representation, local support tracks overlap among neighboring measured templates, and the model remains anchored to the calibrated fingerprint geometry.

### Physical dictionary formulation and centered-magnitude SVD
The normalized features serve as the basis for constructing the angle-indexed dictionary used in inference. Per-angle prototypes are averaged across calibration trials to form columns \(h_e\), where \(e\) indexes one of the \(E\) candidate directions, yielding the standardized fingerprint dictionary \(H=[h_1,\dots,h_{37}]\in\mathbb{R}^{F\times E}\), with \(E=37\). For the Fig. 2 analysis, we then form the centered-magnitude matrix

$$
H_{\mathrm{fig}}[k,e] = |H[k,e]| - \frac{1}{E}\sum_{e'=1}^{E}|H[k,e']|.
$$

Singular-value decomposition (SVD) of \(H_{\mathrm{fig}}\) shows rapid early saturation (Fig. 2a). Six modes capture 80.3% of the centered-magnitude energy. Extending to eight modes raises the cumulative fraction to 85.1%.

For intuition, one may also form a reduced-order surrogate in a retained singular subspace; this is the reduced-order picture referenced by Eq. 2, while the reported decoding uses the full \(F=346\) standardized feature space without PCA/SVD preprocessing.

### Inference algorithms and network architecture
For inference, we keep the full standardized feature space and use a grouped dictionary \(D=[d_{e,m}]\). We do not project onto the reduced SVD basis before decoding. The formulation rests on three working assumptions: (1) over each analysis window the target behaves as a linear time-invariant system, so direction-dependent responses superpose in the frequency domain; (2) \(Hx\) serves as a sparse prototype approximation in standardized fingerprint space; and (3) discretizing the angle ordering provides a useful single-source surrogate over nearby directional templates. The representative Kirchhoff-Love plate operator and Green's function derivation supporting assumption (1) are given in Supplementary Methods 1.

Once direction is discretized onto the measured angle grid, hard OMP provides the standard classical greedy sparse-recovery baseline for the surrogate above. Supplementary Methods 3 gives that recursion explicitly. Figures 4 and 5 summarize that OMP baseline family. Figure 3 shows how correlation mass is distributed across the calibrated angle grid before residual refitting: sharp concentration marks locally separable fingerprints, whereas broad spread across neighboring angles reveals the ambiguity that later destabilizes single-choice decoding.

The physics-guided solver is an OMP-inspired differentiable routed pursuit. In plain terms, OMP repeatedly picks the single best-matching atom and refits on the active set. The guided solver keeps the same \(K\)-stage residual-correction scaffold and physical match score, then replaces that hard one-angle choice with learned routing across nearby angle groups. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient vector as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the physical match score between the residual and every grouped dictionary atom is computed as:

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual and each entry of \(g_t\) quantifies how well a candidate direction explains the unexplained observation through the physical dictionary. A transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary guided-solver configuration) uses the current residual to query the dictionary and produces expert-level routing scores from scaled dot-product interactions:

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. In the reported configuration, routed selection uses hard Gumbel gating in the forward pass, so the routing step acts as a discrete gate over neighboring angle hypotheses. These routed weights gate the physical match scores to produce a sparse update:

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication. The sparse coefficient vector is accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter that scales each stage-wise sparse update before residual correction, and the residual is corrected by dictionary-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\), ensuring that each stage explains a portion of the observation through the physical dictionary. Direction supervision and final prediction come from expert-level routing scores:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t^{(\mathrm{exp})}[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t^{(\mathrm{exp})}[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). The routing weights \(w_t\) can be directly inspected as model-internal support for neighborhood-aware selection (Fig. 5c); Supplementary Methods 4 provides the complete grouped formulation.

The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the expert-level readout \(\bar s[e]\). Equation 9 gives the main loss; in the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The loss weights in the executed main objective are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization is performed with Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Calibration protocol and outer-fold cross-validation
White-noise calibration builds the fixed dictionary \(H\): for each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over all three calibration clips, and those clips are never reused for training or evaluation of the direction decoder.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech results use the fixed \(H\) estimated from white-noise calibration only, and scalar performance metrics are reported from held-out outer-test predictions.

For cross-material screening experiments (Fig. 6), object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). For Fig. 6c, each material-specific dictionary is then analyzed with the same row-wise mean-centered magnitude SVD used for Fig. 2, that is, on \(|H| - \mathrm{mean}_{\theta}(|H|)\). For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) screening tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration, rather than cross-object weight sharing.

### Discriminability analysis
Before evaluating the learned solver, encoding quality is quantified independently of any inference algorithm (Fig. 3). White-noise and speech stimuli are compared using the same dictionary \(H\). Because white noise is the calibration probe used to construct \(H\), whereas speech introduces structured source variation absent from calibration, this contrast tests whether discriminability remains tied to the object's measured directional response rather than to a single source waveform. For each stimulus type, within-angle Pearson correlation (mean pairwise \(\rho\) among clips at the same angle) and between-angle correlation (mean pairwise \(\rho\) across different angles) are computed. The discriminability margin is defined as within \(\bar{r}\) minus between \(\bar{r}\); a positive margin indicates that the encoding preserves directional information (Fig. 3c). Effect sizes are reported as Cohen's \(d\) and significance via the two-sided Mann-Whitney \(U\) test. Formal definitions are given in Supplementary Methods 5.

### Dose-response and noise robustness
For the dose-response analysis (Fig. 3f), noise levels are swept in two conditions: (i) white-noise signal with additive speech-spectrum noise, and (ii) speech signal with additive babble noise. For noise-robustness evaluation, zero-mean white noise is added at SNR levels \(\infty\), 30, 20, 15, 10, 5, and 0 dB in the time domain, using the same feature extraction and dictionary for each level.

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
