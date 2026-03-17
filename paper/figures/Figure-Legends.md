# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the six main-paper figures.

## Fig. 1 (5 panels)

**Fig. 1 | From chaotic acoustic scattering to sparse physical order in complex-media sensing.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)); inset shows a representative single-point vibration waveform exhibiting complex, seemingly chaotic fluctuations.
b, Conceptual schematic illustrating that different incidence directions excite distinct combinations of a small number of structural modes, whose spectral superposition yields direction-specific single-point "spectral fingerprints".
c, Time-domain vibration waveforms at three representative angles (0, 90, 180 degrees) appear visually similar due to dispersion and reverberation, motivating spectral analysis.
d, Spectral fingerprints for the same three angles reveal distinct, direction-dependent frequency structures — the core phenomenon underlying single-point DOA sensing.
e, Contact-loading control: spectral distinctiveness (1 - mean inter-angle Pearson r) is substantially reduced under contact readout compared with non-contact LDV, confirming that preserving the target's native boundary conditions is essential for directional encoding.

## Fig. 2 (7 panels: a, b-d, e, f, g)

**Fig. 2 | Physical encoding via spectral-spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes; cumulative energy and DOA capacity curves quantify the information concentration.
b-d, Modal decomposition of modes 1-3 into frequency-selective spectra (left) and direction-selective polar patterns (right), forming virtual directional sensing channels.
e, Full angle-frequency heatmap of the dictionary H (37 angles x 346 frequency bins), showing systematic spectral variation across directions.
f, Rank-r reconstruction quality: original versus reconstructed fingerprint at a representative angle for ranks 3, 5, and 10, quantifying the information retained by low-rank compression.
g, Inter-angle correlation matrix of H, revealing the smooth structure of the physical angle manifold — nearby angles share similar spectral fingerprints.

## Fig. 3 (4 panels)

**Fig. 3 | Physics-guided deep unrolled network with attention-based gating for sparse DOA inference.**
a, Architecture: at stage t, the residual is correlated with the static physics dictionary D to form a physical match; a transformer encoder outputs attention weights that gate sparse updates, followed by residual update. Unrolling across stages accumulates a sparse vector mapped to the final DOA estimate.
b, Training convergence: total and classification loss decrease steadily over 20 epochs; the vertical dashed line marks the best validation epoch.
c, Clean-condition ablation: strip chart comparing the full model against six ablation variants (individual seeds shown as dots, horizontal bars indicate means), demonstrating that the transformer routing and type-bias components are essential for high accuracy.
d, Per-angle accuracy profile: 37-point bar chart confirms near-uniform performance across the full angular grid, with below-mean angles highlighted.

## Fig. 4 (5 panels)

**Fig. 4 | Spectral fingerprints are statistically discriminable and repeatable across the angle grid.**
a, Angle-frequency heatmap of the calibration dictionary H, showing direction-dependent spectral structure across the 37-angle grid.
b, Violin plot of pairwise Pearson correlations partitioned into within-angle and between-angle pairs, with Mann-Whitney U test significance and Cohen's d effect size annotated.
c, Per-angle fingerprint repeatability: mean within-angle Pearson r (+/- SEM) for each direction, demonstrating consistent encoding fidelity across the angular range.
d, Full pairwise similarity matrix (37 x 37): mean Pearson r between all angle pairs computed from trial-level data, revealing the smooth manifold structure of the angle space.
e, Prediction confidence distribution: histograms of maximum softmax confidence for correct versus incorrect trials, showing well-calibrated high-confidence correct predictions.

## Fig. 5 (6 panels)

**Fig. 5 | The learned router mirrors physical structure and maintains robust decoding under noise.**
a, SNR degradation curves for the physics-aware model, no-transformer ablation, and analytical OMP baseline, showing graceful degradation under additive noise.
b, Correlation structure of the physical dictionary H (top) and the learned QK attention map (bottom), revealing that the router recovers the geometry of the angle manifold — the core interpretability finding.
c, All-angle selection-probability heatmaps comparing OMP (diffuse off-diagonal mass, top) with the physics-aware model (sharply diagonal, bottom), demonstrating that structure-aligned routing concentrates selection on the correct direction.
d, Confusion matrices (baseline versus no-transformer ablation, normalized to row-wise probabilities): the baseline exhibits a sharply diagonal pattern, confirming effective direction assignment.
e, Angle-specific routing distributions at 55 and 100 degrees: the baseline concentrates mass on the correct atom and suppresses off-axis peaks, whereas the no-transformer ablation shows broader distributions.
f, Per-angle diagonal concentration: P(correct) for each of the 37 angles comparing baseline with no-transformer ablation. The shaded region quantifies the per-angle improvement from transformer routing.

## Fig. 6 (5 panels)

**Fig. 6 | Universal physical encoding across diverse materials and robust cross-material performance.**
a, Targets spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell).
b, Representative dictionary/response heatmaps for each material, highlighting shared dispersion-signature structure despite differing physical properties.
c, DOA estimation error (RMSE) across materials comparing analytical OMP and physics-aware AI, showing degradation of OMP under increasing complexity and stable low error for the physics-aware model.
d, Per-band SVD spectra: normalized singular-value decay across five frequency bands, demonstrating consistent low-rank structure regardless of frequency range — a physics-level indicator of universality.
e, Band-resolved routing consistency: diagonal accuracy per band comparing OMP versus physics-aware AI, confirming that the encoding mechanism is frequency-distributed rather than tied to a single resonance artifact.
