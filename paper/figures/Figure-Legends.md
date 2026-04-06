## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

## Fig. 2 (6 panels)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. The cumulative curve rises quickly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. The overlaid auxiliary angle-ordering proxy, derived from the same decomposition, follows the same rapid early accumulation.
b, Frequency-selective spectra \(|u_r(f)|\) for representative components 1, 2, and 6. These traces summarize three reusable spectral patterns in the centered-magnitude decomposition.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative components 1, 2, and 6, showing how those same components vary across 0°–180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing structured spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-component regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that neighboring angles remain close in fingerprint space, revealing a local angle ordering across nearby directions.

## Fig. 3 (6 panels)

**Fig. 3 | Speech preserves directional structure while broadening nearby-angle overlap.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); within-angle similarity remains higher than between-angle similarity despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11), shown with light bootstrap uncertainty bands; speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based first-choice diagnostic traces for white noise and held-out speech, shown with light clip-level uncertainty bands: the stage-0 diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response curves for stage-0 correlation-based first-choice matching versus SNR. The white-noise signal curve (blue, clip-level SEM shading) is recomputed on synthetic noisy white-noise datasets, whereas the speech-plus-babble curve (orange, 5-seed mean ± SEM shading) comes from a separate five-seed sweep; within those two matched surfaces, both decline monotonically with increasing noise.

## Fig. 4 (4 panels)

**Fig. 4 | Preserving measured local continuity before subtraction stabilizes readout.**
a, Measured local-overlap structure from the fixed dictionary \(H\), shown as an angle-by-angle similarity map around the representative 70° target.
b, Representative 70° validation exemplar: residual-to-dictionary matching initially spans nearby calibrated directions, local gating concentrates that support around the correct neighborhood, and the resulting update remains confined to that band.
c, Validation-wide cumulative update mass within radius before and after one local step across 1,924 clips. The mass within 15° rises from 0.18 to 0.98, showing that the local step concentrates the update inside the supported neighborhood.
d, Clean-condition comparison of four overlap-handling rules on the same fixed dictionary (individual seeds shown as dots; large markers and error bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing, consistent with accurate subtraction requiring preservation of the local angle band.

## Fig. 5 (5 panels)

**Fig. 5 | Prediction structure stays organized around the measured local band.**
a, Five-seed SNR sweep for the same four overlap-handling rules. The delayed-commitment rule degrades least as noise increases.
b, Row-normalized representative confusion display for the same four overlap-handling rules. The guided solver stays nearest the diagonal, whereas the OMP baseline shows fractured off-diagonal leakage, router-bypass remains broader around the target neighborhood, and dense routing collapses toward a preferred output mode.
c, Measured local structure (top) and neighborhood-emphasis map from the guided solver (bottom). The lower map retains a coarse near-diagonal angle ordering similar to the calibrated fingerprints.
d, Angle-specific prediction profiles at four representative directions (55°, 70°, 95°, and 100°): the guided solver produces tighter local prediction profiles, whereas router-bypass shows broader off-axis leakage around the same targets.
e, Per-angle clean accuracy: five-seed mean P(correct) across the 37 measured angles, shown as a 3-angle centered moving-average display with light +/-1 s.e.m. shading for the same four overlap-handling rules on the clean sweep. The delayed-commitment rule retains the highest clean mean accuracy overall, whereas dense routing remains near chance across almost the entire angle set.

## Fig. 6 (5 panels)

**Fig. 6 | A finite locally ordered directional code recurs across passive objects under matched calibration.**
a, Measured response-regime map across the five target objects. Each object is placed by the zero crossing of its mean centered-\(|H|\) correlation-decay curve and by the effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. The upper row shows per-angle Top-1 distributions with object mean \(\pm\) 95% confidence intervals. The lower row plots object mean Top-1 against correlation-decay width, with marker area proportional to normalized overall \(|H|\) energy. This five-object comparison is descriptive rather than a fitted law, but it shows that response energy alone does not order the objects and that overlap width provides a more informative comparison axis.
e, Object-conditioned contrast bands and recovered directional codes. The upper row shows the selected frequency window that maximizes across-angle contrast for each object, and the lower row shows the corresponding band-limited directional code. The same selection rule therefore picks different informative bands across objects, and the recovered band-limited codes differ across those object-specific bands.
