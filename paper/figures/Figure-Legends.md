# Figure legends (Nature Communications style)

This document provides concise manuscript-ready legends for the six main-paper figures.
Canonical source: `paper/manuscript/manuscript.md`. This file must stay in sync with the inline legends in the manuscript.
Naming contract: `paper/manuscript/FIGURE_NAMING_CONTRACT.md`.

## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

## Fig. 2 (6 panels)

**Fig. 2 | Physical encoding via spectral–spatial modes and construction of a structured dictionary.**
a, Singular-value spectrum across the full 37-mode basis. Cumulative fraction is shown on the left axis and singular values on the right, emphasizing the early saturation of the centered-magnitude dictionary across the calibrated angle manifold. The first six modes capture 80.3% of the cumulative energy. Extending to eight modes raises the cumulative fraction to 85.1%, and the cumulative direction-of-arrival capacity curve closely tracks the same rise.
b, Frequency-selective spectra \(|u_r(f)|\) for representative Modes 1, 2, and 6 (overlaid). These profiles highlight a dominant broadside-like channel (Mode 1), an edge-weighted grazing-angle-like channel (Mode 2), and an end-fire-like channel with a distinct higher-frequency shoulder (Mode 6).
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) across 0°–180° for representative Modes 1, 2, and 6 (overlaid), showing three physically interpretable directional couplings that define distinct virtual sensing channels.
d, Full angle–frequency heatmap of the magnitude dictionary \(|H|\) (37 angles × 346 frequency bins), showing systematic spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation: per-angle centered-magnitude RMSE in the \(H_{\mathrm{fig}}\) representation for ranks 3, 5, and 6, showing that reconstruction fidelity becomes strong within the same six-mode regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band indicates that neighboring angles share similar spectral fingerprints, revealing the smooth physical angle manifold.

## Fig. 3 (6 panels)

**Fig. 3 | Encoding survives content variation but classical decoding fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11), shown with light bootstrap uncertainty bands; speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based greedy diagnostic traces for white noise and speech, shown with light clip-level uncertainty bands: the greedy diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (diffuse but structured manifold), with the diagonal masked to separate the two regimes.
f, Dose-response curves: correlation-based greedy diagnostic accuracy versus SNR for white-noise signal (blue, clip-level SEM shading) and speech signal with babble noise (orange, 5-seed mean ± SEM shading), both declining monotonically with increasing noise.

## Fig. 4 (5 panels)

**Fig. 4 | The guided solver sharpens local ambiguity before residual correction.**
a, Architecture: at each stage, the residual is compared with the physical dictionary to form a match score. A gate concentrates that score into nearby directions, updates the sparse estimate, and refreshes the residual.
b, Routing formation: at a shared 70° exemplar, the broad physical match narrows into a local peak.
c, Gated update and residual correction: the shared 70° exemplar shows that the gated update localizes mass much more strongly than the raw physical match. The next residual then shifts the match landscape accordingly.
d, Aggregation bridge: the shared 70° exemplar links the angle-axis summaries to the underlying stagewise localization and update fields used in panels b-c.
e, Routing-mechanism ablation: compact clean-condition comparison across the guided solver, router-bypass, OMP baseline, and dense routing families over the same five-seed sweep (individual seeds shown as dots, points with horizontal bars indicate mean \(\pm\) s.e.m.).

## Fig. 5 (5 panels)

**Fig. 5 | The guided solver mirrors physical structure and maintains robust decoding under noise.**
a, SNR degradation curves for the guided solver, router-bypass, OMP baseline, and dense routing, showing graceful degradation under additive noise.
b, Unified row-normalized confusion-family block across the four decoders. The OMP baseline shows broader fragmented off-axis spillover, the guided solver remains sharply diagonal, dense routing collapses toward a single preferred output mode, and router-bypass retains limited local structure with broader leakage.
c, Correlation structure of the physical dictionary \(H\) (top) and the guided solver's local similarity map (bottom), showing that the learned structure mirrors the physical angle manifold.
d, Angle-specific conditional output distributions at four representative directions (55°, 70°, 95°, and 100°): the guided solver sharpens the prediction profile around the correct angle and its local neighborhood, whereas router-bypass shows broader off-axis leakage.
e, Per-angle decoder accuracy: five-seed clean mean P(correct) across the 37 measured angles, shown as a 3-angle centered moving-average display with light +/-1 s.e.m. shading, comparing the guided solver, router-bypass, OMP baseline, and dense routing. The guided solver remains strongest overall, whereas dense routing remains near chance across almost the entire angle set.

## Fig. 6 (5 panels)

**Fig. 6 | Direction-dependent encoding and low-rank continuity recur in an exploratory five-object screen.**
a, Five target objects spanning a broad range of material and geometric complexity, shown here in screening order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Cross-material \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across all materials despite different resonance patterns.
c, Low-rank continuity. The same row-wise mean-centered magnitude SVD used in Fig. 2 was applied separately to each material-specific dictionary. The resulting cumulative-energy curves and rank90/rank95 summaries show that the encoder remains low-dimensional across the full object set, extending the low-rank physical-dictionary view from Fig. 2 beyond the single acrylic plate.
d, Exploratory screen consequence. All five objects remain above chance. The left comparison plots normalized overall response energy and Top-1 accuracy for each material, showing that stronger overall response energy does not guarantee higher Top-1 screening accuracy. The adjacent Top-1 and MAE summaries report 95% bootstrap confidence intervals, while the separate angle-level MAE matrix used for the omnibus heterogeneity test does not reveal detectable cross-material heterogeneity in this committed sample.
e, Material frequency structure. Each material is shown as a three-level card: normalized spectral envelope, angular-contrast spectrum, and the representative directional code recovered from that material's selected informative band.
