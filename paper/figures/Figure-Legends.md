# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the six main-paper figures.
Canonical source: `paper/manuscript/manuscript.md`. This file must stay in sync with the inline legends in the manuscript.
Naming contract: `paper/manuscript/FIGURE_NAMING_CONTRACT.md`.

## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter H(θ, f) — a flat broadband source is transformed into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized |H(θ, f)| across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

## Fig. 2 (6 panels)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum showing rapid decay, indicating that the measured structural response is dominated by a small set of modes; cumulative energy and DOA capacity curves quantify the information concentration.
b, Frequency-selective spectra \(|u_r(f)|\) for modes 1–3 (overlaid), showing distinct spectral peaks for each dominant channel.
c, Direction-selective polar patterns \(v_r(\theta)\) for modes 1–3 (overlaid), forming virtual directional sensing channels.
d, Full angle–frequency heatmap of the dictionary \(H\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation: per-angle fingerprint RMSE for ranks 3, 5, and 10, showing that low-rank compression preserves directional fingerprints across the calibrated angle manifold.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band indicates that neighboring angles share similar spectral fingerprints, revealing the smooth physical angle manifold later exploited by the learned router (Fig. 5b).

## Fig. 3 (6 panels)

**Fig. 3 | Encoding survives content variation but classical decoding fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11); speech margin is reduced but positive at all angles.
d, Stacked angle-resolved OMP traces for white noise and speech, showing high white-noise performance across most calibrated angles but near-chance speech performance across the grid (white noise 83.8% versus speech 1.7%).
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (diffuse but structured manifold), with the diagonal masked to separate the two regimes.
f, Dose-response curves: OMP accuracy versus SNR for white-noise signal (blue) and speech signal with babble noise (orange, 5-seed mean ± SEM), both declining monotonically with increasing noise.

## Fig. 4 (4 panels)

**Fig. 4 | Physics-guided deep unrolled network with attention-based gating.**
a, Architecture: at stage \(t\), the residual \(r_t\) is correlated with the physical dictionary \(A\). A transformer encoder generates routing weights that gate sparse updates \(\Delta x_t\), enforcing residual consistency \(r_{t+1}=r_t-A\Delta x_t\).
b, Training convergence: total and classification loss decrease steadily over 20 epochs; the vertical dashed line marks the best validation epoch.
c, Clean-condition ablation: strip chart comparing the full physics-aware solver against a local ablation family centered on the router-bypass condition and related routing variants (individual seeds shown as dots, horizontal bars indicate means), demonstrating that transformer routing and the type-bias component are essential for high accuracy.
d, Per-angle accuracy profile: the 37-point bar chart confirms near-uniform performance across the full angular grid (mean accuracy 0.946), with angles below the mean highlighted.

## Fig. 5 (6 panels)

**Fig. 5 | The learned router mirrors physical structure and maintains robust decoding under noise.**
Panel logic: a benchmarks task robustness against both the analytical OMP baseline and the router-bypass ablation; b-c provide physical and classical reference context; d-f form a matched-ablation block that isolates learned routing.
a, SNR degradation curves for the physics-aware solver, router-bypass ablation, and analytical OMP baseline, showing graceful degradation under additive noise.
b, Correlation structure of the physical dictionary \(H\) (top) and the learned router's QK attention map (bottom), showing that learned routing mirrors the physical angle manifold.
c, Row-normalized confusion maps contrasting the analytical OMP baseline (broader, fragmented structure; top) with the physics-aware solver (sharply diagonal; bottom), providing the classical decoding context for the ablation panels.
d, Confusion matrices for the full physics-aware solver (top) and the router-bypass ablation (bottom), normalized to row-wise probabilities.
e, Angle-specific conditional output distributions at two representative directions (55° and 100°): the full physics-aware solver sharpens the prediction profile around the correct angle and its local neighborhood, whereas the router-bypass ablation shows broader off-axis leakage.
f, Per-angle routing gain: P(correct) for each of the 37 angles comparing the full physics-aware solver with the router-bypass ablation, quantifying the fraction of angles that benefit from learned routing. The shaded region highlights the per-angle improvement.

## Fig. 6 (5 panels)

**Fig. 6 | Universal physical encoding across diverse materials.**
a, The five target objects spanning a broad spectrum of material and geometric complexity (acrylic plate, paper cup, wooden board, cardboard box, and a laptop shell).
b, Representative dictionary heatmaps for each material, showing shared dispersion-signature structure despite differing physical properties.
c, Cross-material RMSE comparison: the physics-aware solver maintains low DOA error across all materials, while the analytical OMP baseline degrades on complex targets.
d, Per-band SVD spectra: normalized singular-value decay across five frequency bands (full band + four sub-bands), demonstrating consistent low-rank structure regardless of frequency range — a physics-level (not accuracy-level) indicator of universality.
e, Band-resolved routing consistency: diagonal accuracy per band comparing the analytical OMP baseline versus the physics-aware solver, confirming that the encoding mechanism is frequency-distributed rather than tied to a single resonance artifact.
