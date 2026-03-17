# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the six main-paper figures.

## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup: a loudspeaker excites an acrylic sensor plate while a laser Doppler vibrometer (LDV) records single-point surface vibrations at varying incidence angles.
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter H(θ, f) — a flat broadband source is transformed into angle-specific spectral fingerprints by the structural transfer function.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°), directly demonstrating the filtering predicted in (b).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of the same five angles. The near-invisible error bands confirm that the spectral fingerprint is a stable physical property of the structure, while the clear separation between angles confirms direction-dependent encoding.
e, Frequency-dependent directivity: polar plot of the normalized transfer-function amplitude H(θ, f) across the full 0°–180° angular range for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

## Fig. 2 (6 panels)

**Fig. 2 | Low-rank modal structure of the physical dictionary.**
a, Singular-value spectrum of H with cumulative energy and DOA capacity curves, showing that the structural response is dominated by a small number of modes.
b, Frequency profiles of the three leading SVD modes, revealing frequency-selective spectral channels.
c, Polar patterns of the same three modes, showing orthogonal direction-selective angular responses.
d, Full angle-frequency heatmap of the dictionary H (37 angles × 346 frequency bins), displaying systematic spectral variation across the angular grid.
e, Rank-r reconstruction quality at a representative angle (90°) for ranks 3, 5, and 10, quantifying the information retained by low-rank approximation.
f, Inter-angle correlation matrix of H, revealing smooth manifold structure — nearby angles share similar fingerprints while distant angles decorrelate progressively.

## Fig. 3 (6 panels)

**Fig. 3 | Encoding is preserved under content variation but classical decoding fails catastrophically.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (Mann-Whitney U test, Cohen's d = 2.83), showing near-perfect fingerprint discriminability in the absence of content variation (mean within-angle r = 1.000).
b, Speech stimulus: same analysis reveals that discriminability remains statistically significant (d = 1.95, p < 10⁻⁴) despite content variation (mean within-angle r = 0.907).
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11). The speech margin is reduced but positive at all angles, confirming that directional information survives content variation.
d, OMP per-angle accuracy: classical OMP achieves 83.8% on white noise but only 1.7% on speech (below chance at 2.7%), demonstrating that the decoding bottleneck lies in the solver, not the encoding.
e, Split-triangle pairwise similarity matrix: lower-left triangle shows white noise (near-identity, clean separation), upper-right shows speech (diffuse but structured manifold). The contrast directly visualizes why OMP succeeds for white noise but fails for speech — the manifold structure exists but requires a geometry-aware solver.
f, OMP dose-response curves: accuracy as a function of SNR for white-noise signal (blue, starting at 84%) and speech signal with babble noise (orange, starting at 44%, 5-seed mean ± SEM). Both curves decline monotonically with increasing noise, converging near chance at 0 dB, establishing content variation as the causal factor.

## Fig. 4 (4 panels)

**Fig. 4 | Physics-guided deep unrolled network with attention-based gating for sparse DOA inference.**
a, Architecture: at each unrolling stage the residual is correlated with the static physics dictionary D to form a physical match; a transformer encoder outputs attention weights that gate sparse updates, followed by a residual update.
b, Training convergence: total and classification loss decrease steadily over 20 epochs; the vertical dashed line marks the best validation epoch.
c, Clean-condition ablation: strip chart comparing the full model against ablation variants (individual seeds as dots, horizontal bars indicate means), demonstrating that transformer routing and type-bias components are essential for high accuracy.
d, Per-angle accuracy profile: 37-point bar chart confirms near-uniform performance across the full angular grid, with below-mean angles highlighted.

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
