# A recurring locally ordered directional code emerges across passive objects in single-point vibrometry

## Abstract
Sound direction is usually measured with arrays or specialized directional sensors.
Here we show that ordinary passive objects can themselves carry directional information in their vibration response at one fixed measurement point.
In an acrylic plate, direction appears as a compact local code: nearby angles reuse related spectral structure instead of forming unrelated fingerprints.
Held-out speech preserves that code but broadens overlap among neighboring angles, so the main difficulty becomes local ambiguity rather than loss of directional structure.
Readout therefore succeeds only when it preserves that measured neighborhood long enough for subtraction to sharpen it.
Across acrylic, paper, cardboard, wood, and a laptop shell, the same locally ordered directional code recurs, and cross-object readout is governed more by local overlap burden than by response energy alone.
Ordinary passive objects therefore act as directional sensing substrates: some directional front-end information can already reside in passive structures present in a scene.

## Introduction
Sound direction is usually engineered by separating sensors in space. Arrays and specialized directional sensors are designed so that direction is measurable at the sensor, while the vibrating or scattering object is usually treated as a nuisance. That default picture leaves a broader possibility mostly unexplored: an ordinary passive object may already separate direction in how it vibrates. If so, directional sensing need not be designed only into the transducer.

Prior work makes that possibility plausible, but not yet general. Some single-sensor studies already show direction-dependent transfer structure in engineered scatterers and structure-borne measurements [@elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. Sparse angle-grid inverse methods supply the natural comparison class once nearby directions overlap [@malioutov2005sparse_doa; @chen2001basis_pursuit; @pati1993omp; @tropp2007omp]. Acoustic vector sensors show that array-free localization is possible when the sensing hardware itself is specialized [@nehorai1994vector_sensor]. What remains open is the broader physical claim: whether ordinary passive objects themselves carry a reusable directional code, and what governs whether that code stays readable.

Our hypothesis is that direction changes how sound couples into an object's vibration modes, so one fixed point records repeatable spectral reweighting across angle. If nearby directions excite overlapping structural responses, then realistic source variation should broaden ambiguity locally rather than erase the code. We test that idea with a non-contact laser Doppler vibrometer (LDV), whose surface-velocity readout avoids the loading introduced by attached piezoelectric patches or accelerometers [@rothberg2017ldv; @castellini2006ldv; @wagner2021_laser_microphone_calibration; @ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

The acrylic plate offers the clearest first view of this behavior. We then ask whether the same local directional organization recurs across ordinary passive objects, and whether local separability rather than response energy determines when that organization remains readable. This question sits alongside wave systems that treat complex scattering as an encoding resource rather than as noise [@fink1997time; @duarte2008single; @gigan2022imaging_computing_disorder; @rotter2017complex_media; @jiang2020randomized_metamaterial]. Our aim is therefore to establish a recurring locally ordered directional code across ordinary passive objects and to identify the variable that governs its recoverability. If that organization recurs, then part of directional sensing resides in passive structures already present in a scene [@iravantchi2023sawsense; @wang2024gpms].

## Results

### A reference acrylic plate exposes directional coding at one measurement point (Fig. 1)
In the acrylic reference object, broadband white-noise excitation establishes the phenomenon that drives the rest of the paper: one flat acoustic input is converted into direction-dependent vibration fingerprints in a single LDV readout.
The physical intuition is shown first. The schematic in Fig. 1b, corresponding to Supplementary Methods 1, illustrates the governing mechanism: changing input direction changes how the plate excites and combines structural modes, so the same fixed LDV point records a different single-point spectral fingerprint.
That mechanism is visible directly in the data. In Fig. 1c, one flat white-noise source is redistributed into angle-dependent output spectra, and repeated calibration trials at each representative angle collapse onto the same mean fingerprint rather than wandering across trials.
At one fixed measurement point, the acrylic plate therefore turns direction into a reproducible spectral code. Because the readout is optical and non-contact, it preserves the target's native boundary conditions and avoids sensor loading that could otherwise alter the measured response [@ewins2000modal; @bi2013transducer_mass_loading; @nassif2005ldv_contact_sensors].

Different frequency bands emphasize different directional sectors (Fig. 1d), indicating that multiple dispersive vibration pathways contribute with angle-dependent weights [@rotter2017complex_media; @kuttruff2025room_acoustics]. The inter-angle similarity matrix in Fig. 1e then shows the first direct sign that these fingerprints are not isolated templates: nearby directions remain more similar to one another than distant ones. The next question is how compact that local geometry is, and whether the full calibrated dictionary occupies a readable reduced space.

![](../figures/fig01_paradigm-shift.jpg)

**Fig. 1 | A passive acrylic plate exposes directional coding at one measurement point.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Physical-principle schematic corresponding to Supplementary Methods 1. Changing input direction alters how the acrylic plate excites and combines structural modes, so one fixed LDV point records a different single-point spectral fingerprint even though the measurement location does not move.
c, Broadband spectral reshaping with repeatability under matched calibration. The flat white-noise source spectrum (grey dashed) is redistributed differently at five representative angles (0°, 45°, 90°, 135°, 180°); light traces show repeated calibration trials and bold traces show the angle-wise mean response.
d, Frequency-dependent directivity across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz). Different bands emphasize different directional sectors, showing that directional encoding is distributed unevenly across frequency.
e, Inter-angle fingerprint similarity matrix of the calibrated acrylic dictionary \(H\). The near-diagonal high-similarity band shows that nearby directions remain related without becoming interchangeable, providing the first direct evidence that the fingerprints trace a local geometry.

### The directional code is compact and locally ordered (Fig. 2)
The local geometry exposed in Fig. 1e becomes sharper after shared magnitude offsets are removed. In the centered representation used for the compactness analysis, mean inter-angle correlation decays from strong nearest-angle values to a first nonpositive mean near 25° (Fig. 2d), so the acrylic code occupies a finite local neighborhood rather than a field of unrelated references. The same early low-rank regime already captures most of the centered-magnitude energy (Fig. 2a-e): six components account for 80.3% and eight account for 85.1%.

The pattern has a direct physical meaning. Changing direction mainly reweights a limited set of shared structural responses instead of generating a new fingerprint at every angle. Nearby directions therefore remain related without becoming interchangeable. The directional code is already compact and locally ordered before any decoder enters the story. The physical argument and measured dictionary construction behind that interpretation are developed in Supplementary Methods 1 and 2.

A complementary graph view makes the same point from the neighborhood side. In the two-dimensional spectral embedding of the positive centered-neighborhood graph (Fig. 2f), the calibrated fingerprints follow a curved, ordered trajectory rather than fragmenting into disconnected clusters. The code is therefore not only locally correlated after centering; it also remains coherent when that centered neighborhood graph is viewed in two dimensions.

That observation changes the decoding problem itself. Once the measured fingerprints are placed on the discrete angle grid, direction recovery is no longer a search over 37 unrelated references. It becomes a local separability problem: neighboring templates can become jointly plausible before a final angle is chosen. The next question is whether realistic source variation preserves that local code or turns it into local ambiguity.

![](../figures/fig02_svd-physical-dictionary.jpg)

**Fig. 2 | The directional code is compact and locally ordered.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. Energy accumulates rapidly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. Most measured directional structure therefore sits in a small component set.
b, Representative component spectra \(|u_r(f)|\) for components 1, 2, and 6. Together they show three reusable spectral patterns in the centered-magnitude decomposition.
c, Matching half-plane polar profiles \(v_r(\theta)\) for components 1, 2, and 6. They show how those same components vary across 0°-180°.
d, Local-ordering decay in centered-\(|H|\). Mean inter-angle correlation is plotted against angular separation for the acrylic reference object, showing a finite positive local neighborhood that decays toward a first nonpositive mean near 25°.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE drops sharply in the same six-component regime highlighted in panel a, showing that the local code is captured early.
f, Two-dimensional spectral embedding of the positive centered-neighborhood graph derived from centered-\(|H|\). The measured angle trajectory stays curved and locally ordered rather than fragmenting into isolated clusters, so the finite neighborhood quantified in panel d remains coherent in a graph-based view.

### Speech preserves the code but turns decoding into a local ambiguity problem (Fig. 3)
The local code from Fig. 2 survives speech, but it creates a new difficulty precisely because it is local. When neighboring directions share structure, a realistic source does not erase that code. It makes nearby directions harder to tell apart. The ambiguity is therefore structured, not random.

The speech-side summaries show that the same compact structure remains present. When the angle-conditioned speech fingerprints are placed on the same centered representation family used in Fig. 2, their cumulative energy still saturates early (Fig. 3a): four components already capture 79.4% of the speech-side angle-mean energy, and six capture 87.3%. Speech therefore does not destroy the code. It preserves a compact shared response while changing how that response is sampled across clips.

What changes is the neighborhood width. On the same centered summary surface, mean inter-angle correlation stays positive for nearby angles under speech and decays more slowly than under calibration (Fig. 3b), while the split-triangle similarity map keeps the same coarse angle ordering even as the near-diagonal band broadens (Fig. 3c). The local code therefore survives realistic content variation, but it survives in a wider neighborhood than the one exposed by matched calibration. Supplementary Methods 5 defines the compactness and centered-neighborhood statistics behind those summaries.

That broader neighborhood is exactly where the first hard choice starts to fail. On the frozen stage-0 grouped-match surface, white noise concentrates more mass in the exact angle bin, whereas speech retains only modest exact support and instead spreads substantial mass across the first local radius bands (Fig. 3d). The angle-resolved exact first-choice traces then show where immediate commitment collapses under speech (Fig. 3e), and the speech exact-versus-local comparison confirms that within-10° tolerance remains well above exact-match success across the grid (Fig. 3f). The code does not disappear. What fails is the assumption that one angle can be chosen immediately once several neighboring templates become jointly plausible. Supplementary Methods 3 gives that hard-commitment recursion in full.

The code therefore survives realistic source variation, but it survives in a form that demands a different readout. The next question is no longer whether the code exists. It is what kind of readout can preserve the measured neighborhood long enough to act on it.

![](../figures/fig03_fingerprint-discriminability.jpg)

**Fig. 3 | Speech preserves the code but broadens local overlap.**
a, Mirrored compactness on the centered angle-conditioned summary surface. Cumulative energy saturation remains early under both calibration and speech, showing that the speech-side directional code still occupies a compact low-rank space.
b, Speech-side local-ordering decay. Mean inter-angle correlation is plotted against angular separation for calibration and speech on matched centered summary surfaces. Speech retains a positive nearest-angle neighborhood but broadens it relative to calibration.
c, Neighborhood-coherence map in split-triangle form: lower-left = calibration centered-neighborhood similarity, upper-right = speech-side centered similarity. Speech preserves the coarse angle ordering and near-diagonal structure while widening the local band.
d, Stage-0 local separability on the grouped-match surface. Mean cumulative match mass within radius \(r\) is plotted for white noise and held-out speech. Speech retains substantial local support within nearby angle bands even as exact support weakens.
e, Angle-resolved exact first-choice collapse. Stage-0 exact-match accuracy across the calibrated grid remains high for white noise but drops sharply for held-out speech.
f, Speech exact versus local tolerance. On the same stage-0 grouped-match family, within-10° support remains well above exact-match success across the grid, showing that the failure under speech is local rather than random.

### Local-neighborhood preservation determines whether readout sharpens or fractures (Figs. 4 and 5)
Once speech broadens overlap locally, the next scientific question is what a readout must preserve in order to act on the code without breaking it. Figures 4 and 5 answer that question with a narrow claim: subtraction remains admissible only when the calibrated neighborhood is preserved before commitment [@gregor2010lista; @monga2021unrolling]. The guided solver is useful here only because it makes that requirement visible.

Figure 4 makes that requirement explicit. On one shared local angle frame, the measured neighborhood serves only as the physical band that the first guided step must preserve: stage-0 support remains broad across nearby calibrated angles, whereas the routed update contracts that support back into the admissible neighborhood (Fig. 4a). The same transition is visible in one representative 70° validation clip, shown first before the guided step (Fig. 4b) and then after contraction (Fig. 4c). The important change is not only that the update becomes sharper. It becomes local in the same way as the measured structure.

That contraction is not anecdotal. Validation-wide cumulative mass-within-radius curves shift sharply inward after the first guided step (Fig. 4d). At the operative 15° neighborhood the same gain remains positive across the directional grid (Fig. 4e), and the clip-level gain distribution shows that this inward shift is positive for nearly all validation clips rather than being driven by a small subset (Fig. 4f). The admissible readout is therefore the one that preserves and contracts the broadened neighborhood exposed in Fig. 3 before subtraction is allowed to act. The next question is whether that same radius-based neighborhood metric survives all the way to final prediction and still separates decoder families. The full neighborhood-preserving formulation is given in Supplementary Methods 4.

![](../figures/fig04_solver-dynamics.jpg)

**Fig. 4 | Preserving the broadened neighborhood keeps subtraction admissible.**
a, Admissible neighborhood contraction. On one shared local angle frame, the measured neighborhood is used only as the physical reference band while stage-0 broad support and the contracted support after the first guided step are overlaid on the same axis.
b, Representative broad support for a 70° validation clip. Before the first guided step, support remains spread across several nearby calibrated angles rather than collapsing immediately to one direction.
c, Representative contracted support for the same 70° validation clip. After one guided step, that same support contracts back into the physically plausible neighborhood.
d, Validation-wide neighborhood contraction. Cumulative update mass within radius is plotted before and after one guided step across the validation set. The full curve shifts inward, showing that the first routed update sharpens support toward the local band exposed by calibration.
e, Angle-resolved within-15° contraction. Mass inside the operative 15° neighborhood is plotted before and after the first guided step for each target angle, showing that the inward shift remains positive across the directional grid.
f, Clip-level within-15° gain CDF. The empirical cumulative distribution of per-clip gain inside the operative 15° neighborhood shows that the first guided step increases local mass for nearly all validation clips.

Figure 5 follows that same neighborhood metric to the decision surface. On one shared radius axis, speech stage-0 matching carries little exact support but substantial local mass, the first guided step contracts that same mass sharply inward, and final guided predictions remain almost entirely inside the same neighborhood (Fig. 5a). The same metric then orders decoder families at final prediction: guided decoding preserves the most local mass, router-bypass remains partially local, the OMP baseline broadens further, and dense routing largely loses neighborhood concentration (Fig. 5b). That locality ordering also predicts exact clean consequence on the same coherent family surface: families that retain more local mass achieve higher exact clean accuracy, whereas dense routing remains poor on both quantities (Fig. 5c). The same hierarchy is visible directly in four-family final-prediction locality maps, where guided predictions remain tightly concentrated near the diagonal, router-bypass broadens modestly, the OMP baseline fractures further, and dense routing disperses broadly (Fig. 5d). The measured neighborhood geometry then provides the common reference surface for those family differences (Fig. 5e). When each family's final confusion is compared back to that measured neighborhood, the strongest decoders retain the highest local-band agreement, whereas the secondary full-matrix correlation is weaker and does not by itself recover the same ordering (Fig. 5f). Noise robustness follows the same hierarchy under babble degradation (Fig. 5g). Final prediction succeeds here because the admissible readout keeps the measured neighborhood local all the way to the decision surface.

If that requirement held only for one acrylic plate, it would still be a narrow single-object result. Figure 6 asks whether the same local directional organization recurs across a broader set of passive structures.

![](../figures/fig05_performance-structure.jpg)

**Fig. 5 | Final prediction succeeds by preserving the measured neighborhood.**
a, Neighborhood-preservation cascade on one shared radius axis. The row-normalized mass within a given angular radius is plotted for speech stage-0 grouped support, after one guided step, and at final guided prediction. Early speech support is broad but still local; the guided update contracts that same support, and the final predictions remain inside the same neighborhood.
b, Family neighborhood preservation. The same cumulative mass-within-radius statistic is computed from the final clean family runs reconstructed on the governed babble protocol. Decoder success is ordered by how much local mass each family retains.
c, Exact-versus-local consequence. Exact clean accuracy is plotted against within-15° local mass for the same four decoder families. Families that preserve more local support also achieve stronger exact clean prediction.
d, Final prediction locality by family. Clean row-normalized confusion maps for guided solver, router-bypass, OMP baseline, and dense routing show how the accepted hierarchy appears directly at the decision surface: guided predictions remain tightly concentrated near the diagonal, router-bypass broadens modestly, OMP fractures further, and dense routing disperses broadly.
e, Measured neighborhood geometry. The angle-angle correlation map of calibrated \(H\) shows the near-diagonal physical neighborhood that nearby angles share.
f, Family-to-measured neighborhood alignment. Bars show a local-band agreement score for each decoder family, defined by retained within-15° local mass multiplied by a bounded anglewise profile-agreement factor relative to the measured neighborhood profile. Open circles show the corresponding full-matrix correlation to the measured neighborhood as a secondary reference. Families that keep support both local and correctly ordered score highest on the primary local-band metric.
g, Noise robustness consequence. Five-seed babble SNR sweeps for the same four overlap-handling rules reconstructed from the governed babble sweep families show that the same neighborhood-preserving hierarchy remains visible under noisy evaluation.

### The same locally ordered directional code recurs across ordinary passive objects (Fig. 6)
Across five ordinary passive objects—an acrylic plate, paper cup, cardboard box, wooden board, and laptop shell—the same locally ordered directional code reappears. Figure 6a places those objects by two executed response descriptors derived from centered \(|H|\): the first separation at which mean inter-angle correlation becomes non-positive, and the entropy-based effective rank of the same centered-magnitude spectrum. These targets are better understood as structurally distinct everyday archetypes than as five nominal material labels. They differ in stiffness, damping, anisotropy, shell-versus-plate geometry, layering, and cavity structure, yet each still shows structured angle-frequency fingerprints, a finite neighborhood of positive local ordering, and above-chance single-point readout under matched calibration (Fig. 6a-c).

The cross-object comparison also clarifies what governs usability. Matched object-specific calibration keeps Top-1 readout above chance across the full set (Fig. 6d), but object performance does not follow normalized overall \(|H|\) energy alone. It tracks a mean top-3 overlap burden more closely: the material-wise average of pairwise squared canonical-correlation overlap between each object's leading centered-\(|H|\) subspace and those of the other objects. The paper cup makes the contrast explicit: it has the strongest overall energy, yet its neighboring directions overlap more broadly, so it reads out worse than cardboard. Across this tested archetype set, local separability therefore matters more than raw response energy for directional readout.

The same point appears in frequency. A shared contrast-selection rule identifies a different informative window for each object, and the corresponding band-limited directional codes differ across those object-specific bands (Fig. 6e). What recurs is not one universal spectrum. It is one local directional principle expressed through different structural bands. On that bounded reading, directional front-end information resides partly in the passive structure itself rather than only in dedicated directional hardware. Supplementary Methods 6 and Supplementary Table 1 provide the supporting cross-object interpretation.

![](../figures/fig06_universality.jpg)

**Fig. 6 | A recurring directional code appears across ordinary passive objects.**
a, Measured response-regime map across the five target objects. Each object is placed by the first non-positive separation of its mean centered-\(|H|\) correlation-decay curve and by the entropy-equivalent effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means exceed chance across the five objects. Object mean Top-1 is plotted against mean top-3 subspace overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Local separability, not response energy alone, is what orders the objects.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. The informative band changes from object to object, but the directional principle does not.

## Discussion
Directional encoding need not be designed only into arrays and specialized sensors. Across the tested passive-object archetypes, one fixed vibrometric point repeatedly records a finite local directional code after matched calibration (Fig. 6). Passive structure is therefore part of the sensing substrate, not merely a nuisance surrounding the sensor.

The same results identify what governs recoverability. The decisive variable is not raw response energy or global template separability. It is local separability within the calibrated neighborhood. In the acrylic reference object, held-out speech preserves the code but broadens overlap among neighboring directions (Fig. 3) [@gigan2022imaging_computing_disorder; @rotter2017complex_media]. Figures 4 and 5 then show why readout fails or sharpens: subtraction breaks down when it outruns that neighborhood and succeeds when it stays inside it. The guided solver matters here only as one readout that keeps subtraction aligned with the measured local geometry long enough for that requirement to become visible. The neighborhood-preserving logic is developed in Supplementary Methods 4, and the cross-object boundary is developed in Supplementary Methods 6 and Supplementary Table 1.

Placed in context, this finding links the paper to literatures that treat complex scattering as an encoding resource and to single-sensor localization studies that exploit direction-dependent transfer structure without large arrays [@jiang2020randomized_metamaterial; @hoang2021single_pixel_doa; @elbadawy2018lego_doa; @dipassio2023doa_single_sensor; @rutowski2024reverb_single_sensor; @rutowski2025_structural_beamforming]. The difference here is that the same logic appears in unmodified passive objects rather than in engineered structures or specialized directional hardware. Recurrence across objects therefore supports a broader inference: directional front-end information can already reside in existing surfaces and structures when matched calibration exposes the local code they carry.

These conclusions are bounded to the tested half-plane grid (0-180° at 5° spacing), the controlled acoustic environment, the single static source, and per-object matched calibration. The LDV is still a laboratory instrument rather than a field-ready sensor, and generalization to an external speech corpus has not yet been tested. Wider object variation, reduced calibration, stronger damping, reverberation, multiple simultaneous sources, moving emitters, and compact optical or MEMS readouts will therefore require new measurements.

## Methods

### Experimental setup and data acquisition
All processing and learning were implemented in Python with PyTorch and trained on Apple Silicon (MPS). Unless otherwise stated, NumPy and PyTorch random seeds were fixed to 42.

Experiments used a single loudspeaker scanned over a half-plane angular grid comprising 37 incidence directions from 0° to 180° in 5° increments. Out-of-plane surface velocity was measured at a fixed LDV spot on each object while the mounting configuration was held fixed throughout calibration and evaluation for that object. All downstream processing used waveforms resampled to 16,000 Hz, and each clip was approximately 3 s long.

Two excitation regimes were used: (i) broadband white-noise playback for template-matrix calibration and fingerprint repeatability diagnostics, and (ii) speech recordings for end-to-end direction decoding and robustness experiments. The four frequency bands in Fig. 1d (0.3-0.5, 0.5-1, 1-2, and 2-3 kHz) tile the analysis band and were analyzed separately to probe frequency-dependent dispersive behavior. Five objects were selected to span a range of damping and structural complexity: acrylic plate, paper cup, wooden board, cardboard box, and laptop shell.

### Signal processing and feature extraction
At the physical level, we interpret the single-point response as a superposition of dispersive structural contributions whose direction-dependent weights change with incidence angle:

$$Y(\omega;\theta) \approx \sum_{m} s_m(\omega)\,\alpha_m(\theta), \qquad (1)$$

where \(Y(\omega;\theta)\) is the single-point velocity response, \(s_m(\omega)\) are modal spectral patterns, and \(\alpha_m(\theta)\) are direction-dependent participation weights. In this view, changing direction mainly reweights a limited set of structural responses rather than generating fully independent fingerprints at every angle.

Each clip is reduced to a standardized log-power fingerprint built from the LDV-measured out-of-plane surface velocity \(V(x_L,y_L,\omega)=i\omega W(x_L,y_L,\omega)\), where \(W\) is the displacement field and \((x_L, y_L)\) denotes the fixed laser measurement location. For incidence direction \(\theta\), the complex single-point response is \(Y(\omega;\theta)=V(x_L,y_L,\omega)\). For each recorded clip, a short-time Fourier transform (Hann window, 2,048 samples, hop 512) is computed and collapsed into a time-averaged power spectrum:

$$\widehat{S}(\omega_k;\theta)=\frac{1}{T}\sum_t |V[k,t]|^2, \qquad (3)$$

where \(V[k,t]\) is the complex STFT coefficient at frequency bin \(k\) and frame \(t\), and \(T\) is the number of retained frames. The spectrum is restricted to [300, 3,000] Hz (\(F = 346\) bins). Because the observed fingerprints are magnitude statistics, each clip is summarized by a log-power feature vector \(y[k]=\log_{10}(\widehat{S}+\epsilon)\), then standardized per frequency bin using white-noise calibration statistics to yield a normalized feature \(\tilde{y}[k]\) (Supplementary Methods 2). These normalized log-power fingerprints are the observables used for downstream inference. Supplementary Methods 2 restores the explicit angle and trial indices when the same calibration construction is written in full.

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

The guided solver keeps the same \(K\)-stage residual-correction scaffold but changes the commitment rule. Instead of collapsing the residual onto one direction immediately, it first consolidates evidence across the measured local neighborhood and only then applies subtraction. The residual is initialized as \(r_0 = \tilde y\) and the sparse coefficient state as \(x_0 = 0\). At each stage \(t = 1,\dots,K\), the residual is compared with the grouped dictionary to form the physical match score

$$g_t = D^\top r_t, \qquad (4)$$

where \(r_t\) is the current residual and \(g_t\) measures how strongly that residual aligns with the grouped calibrated templates. A compact transformer encoder (embedding dimension \(d_\mathrm{model}=128\), 2 attention heads, 1 encoder layer in the reported primary configuration) then uses the same residual to produce a direction-level routing score

$$s_t[e] = \frac{\langle q_t, k_e \rangle}{\sqrt{d_k}}, \qquad (5)$$

where \(q_t\) is a learned query derived from the current residual, \(k_e\) is the learned key associated with direction group \(e\), and \(d_k\) is the key dimension. This score does not replace the physical match. It decides where that match should be concentrated before subtraction, so the update remains tied to one locally plausible neighborhood rather than to a prematurely sharpened support. In the reported implementation, selection uses a straight-through Gumbel approximation in the forward pass, so the routing acts as a discrete local gate over neighboring angle hypotheses. At the paper-facing level, the resulting gated update is written compactly as

$$\Delta x_t = w_t \odot g_t, \qquad (6)$$

where \(\odot\) denotes element-wise multiplication and the direction-level gate \(w_t\) is understood to broadcast across the grouped local match within each retained neighborhood. Supplementary Methods 4 writes that same grouped construction explicitly at atom resolution. The sparse coefficient state is then accumulated as \(x_{t+1} = x_t + \eta\,\Delta x_t\), where \(\eta\) is a learned step-size parameter, and the residual is corrected by template-consistent subtraction:

$$r_{t+1} = r_t - D\,(\eta\,\Delta x_t), \qquad (7)$$

equivalently, \(r_{t+1} = \tilde y - D x_{t+1}\). In this form, the routed update preserves a broad local band long enough for subtraction to act on the measured directional code instead of destroying it. Direction supervision and final prediction come from the same direction-level scores:

$$\bar s[e] = \frac{1}{K_{\mathrm{sup}}}\sum_{t=1}^{K_{\mathrm{sup}}} s_t[e], \qquad \hat\theta = \theta_{\arg\max_e \bar s[e]}, \qquad (8)$$

where \(s_t[e]\) denotes the per-direction expert score at stage \(t\), and the reported configuration uses \(K_{\mathrm{sup}}=1\). Supplementary Methods 4 provides the complete grouped formulation.

### Training and optimization
The network is trained with a composite loss containing reconstruction, monotonicity, and classification terms:

$$\mathcal{L} = \alpha\,\mathcal{L}_\mathrm{rec} + \beta\,\mathcal{L}_\mathrm{mono} + \gamma\,\mathcal{L}_\mathrm{cls}, \qquad (9)$$

where \(\mathcal{L}_\mathrm{rec} = \lVert r_K \rVert_2^2\), \(\mathcal{L}_\mathrm{mono}\) encourages stagewise residual descent, and \(\mathcal{L}_\mathrm{cls}\) is the cross-entropy loss over the final readout \(\bar s[e]\). In the reported primary run, training also includes an auxiliary teacher-warmup cross-entropy term during the first 10 epochs. The executed loss weights are \((\alpha,\beta,\gamma)=(1.0,\,0.2,\,0.5)\). Optimization uses Adam (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) for 20 epochs with batch size 32.

### Evaluation protocols
White-noise calibration builds the fixed template matrix \(H\): for each angle, the prototype \(h_e\) is the mean standardized log-power fingerprint over all three calibration clips, and those clips are never reused for training or evaluation of the direction readout.

For speech decoding (Figs. 4-5), each angle contributes 260 speech clips (9,620 total). Clips are assigned deterministically to five outer folds by the rule \(\mathrm{fold} = \mathrm{clip\_id} \bmod 5\), producing 52 clips per angle per fold. For outer fold \(f\), fold \(f\) serves as the held-out test set, fold \((f+1) \bmod 5\) serves as the validation set, and the remaining three folds form the training set, yielding 5,772/1,924/1,924 train/validation/test samples in total. Model architecture, loss weights, optimizer, and the 20-epoch training schedule are fixed across folds; best-epoch selection uses only the validation fold. All speech analyses use the fixed \(H\) estimated from white-noise calibration. The representative mechanics panels in Fig. 4b,c draw from validation routing outputs in the selected fold, and the validation-wide neighborhood summaries in Fig. 4d,e are derived from governed checkpoint replay on the same validation bundle (`results/fig04_stepwise_mechanics.npz`). Figure 5a reuses that governed first-step contraction summary and places it on the same radius axis as stage-0 speech support and final guided prediction locality. Figure 5b-d,g reconstruct the clean and noisy family consequences directly from the governed babble sweep families, so the family-backed bridge panels share one coherent legacy protocol. Figure 5e provides the measured neighborhood reference, and Fig. 5f compares each family's final clean confusion back to that reference through a local-band agreement score together with the corresponding full-matrix correlation. Held-out outer-test predictions are reported only where noted.

For the five-object comparison in Fig. 6, object-specific white-noise calibration data are acquired for each target and used to estimate the corresponding \(H\). The panel-a descriptor map and panel-c decay curves are computed from the row-wise mean-centered magnitude matrix \(|H| - \mathrm{mean}_{\theta}(|H|)\): for each object, we form the mean inter-angle correlation as a function of angular separation and record the first separation at which that mean correlation becomes non-positive. Effective rank is taken from the singular-value spectrum of the same centered-\(|H|\) matrix. For panel e, the representative frequency window is defined by the peak of a smoothed across-angle contrast profile \(\mathrm{std}_{\theta}(|H|)/\mathrm{mean}_{\theta}(|H|)\), and the corresponding band-limited directional code is obtained by averaging \(|H|\) across that three-bin window. For each object, an original-side basis is learned from the matched normalized reference clips, and downstream evaluation is performed on the corresponding normalized material-side clips with the same routed OMP-family evaluator. Per-material Top-1 and within-10\(^\circ\) readout tallies are tested against the 1/37 and grid-aware within-10\(^\circ\) nulls, respectively, using exact binomial tests with Holm correction across the five materials. Angle-level MAE heterogeneity is then assessed on the 37-angle MAE matrix with a Friedman test across materials; pairwise follow-up is performed only if the omnibus test is significant. These cross-material analyses quantify recurrence of structured direction-dependent encoding under object-specific calibration.

### Diagnostic analyses
Figure 3 quantifies whether the compact local code from calibration survives a change in source content. White-noise calibration and held-out speech are compared on two aligned analysis families. Panels a-c use centered angle-conditioned summary surfaces to measure compactness, local-ordering decay, and neighborhood coherence after source-dependent averaging across clips at each angle. Panels d-f then return to the frozen stage-0 grouped-match surface \(g_0^{(\mathrm{grp})}\) to ask how much support remains exact versus local before any residual correction occurs. Formal definitions of these compactness, neighborhood, and local-tolerance summaries are given in Supplementary Methods 5.

Figures 4 and 5 hold the matched-calibration dictionary fixed and compare how different overlap-handling rules act once local ambiguity appears. Figure 4 isolates the admissibility question on the guided solver itself: whether the first routed update preserves and contracts support inside the measured neighborhood before subtraction. Figure 5 then asks how decoder families differ in locality preservation, exact consequence, robustness, and structure alignment once that neighborhood-preserving requirement is imposed. *Guided solver* first pools support within the locally matching neighborhood, *router-bypass* keeps the same staged updates without that pooling, *OMP baseline* applies the soft-OMP comparison without learned local routing, and *dense routing* spreads activation across experts without sparse local concentration. Figure 3 reports the ungated stage-0 diagnostic separately because immediate single-choice matching already degrades under content variation before full residual correction is applied.

### Evaluation metrics and statistics
Top-1 accuracy is reported on the discrete angle grid. Angular error is the minimal angular difference \(\Delta(\hat\theta,\theta)=\min_{k\in\mathbb{Z}}|\hat\theta-\theta+360k|\), and root-mean-square error (RMSE) = \(\sqrt{\frac{1}{N}\sum_i \Delta(\hat\theta_i,\theta_i)^2}\). Unless noted otherwise, scalar speech-readout metrics from the outer-fold protocol are reported as mean ± s.d. across the five held-out test folds. Figure 4c summarizes first-step operating-point recovery at exact, 5°, 10°, and 15° thresholds, and Fig. 4d-f summarize governed validation-wide neighborhood-contraction statistics derived from checkpoint replay. Figure 5a-c place stage-0 support, first-step contraction, final family locality, and exact-versus-local clean consequence on the same neighborhood-preservation logic. Figure 5d makes that hierarchy visible directly as four-family final confusion morphology. Figure 5f reports a family-to-measured local-band agreement score defined by retained within-15° mass multiplied by a bounded anglewise profile-agreement factor relative to the measured neighborhood profile; open-circle annotations on the same panel show the corresponding full-matrix correlation to the measured neighborhood as a secondary reference. Figure 5g reports the coherent five-seed babble SNR sweep with mean ± s.e.m. across sweep seeds. For Fig. 6, correlation-decay width, entropy-equivalent effective rank, and mean top-3 overlap burden are treated as executed response descriptors rather than intrinsic material constants. When statistical hypothesis tests are used, the test, sidedness, and multiple-comparison correction are stated alongside the corresponding figure panel.

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
