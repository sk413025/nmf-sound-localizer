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

**Fig. 4 | A physics-guided solver concentrates nearby overlap into the correct neighborhood.**
a, Sequence summary of broad physical match, learned local gating, and residual cleanup after one step.
b, Representative 70° validation exemplar: the initial physical match spans nearby calibrated directions, the learned local gate concentrates that support around the correct neighborhood, and the resulting local update remains confined to that band.
c, Residual profile before and after one refinement step for the same exemplar. The 15° mass callout summarizes the corresponding validation-wide aggregate shift across 1,924 validation clips (0.18 to 0.98), whereas the residual drop (1.00 to 0.48) belongs to the representative exemplar shown in the panel.
d, Secondary restricted clean-condition check across the four readout families over the same five-seed sweep (individual seeds shown as dots; points with horizontal bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing.

## Fig. 5 (5 panels)

**Fig. 5 | Prediction structure stays organized around the measured local band.**
a, Five-seed SNR sweep across four readout families. The guided solver degrades least as noise increases.
b, Row-normalized representative confusion display across the four compared families. The guided solver stays nearest the diagonal, whereas the OMP baseline shows fractured off-diagonal leakage, router-bypass remains broader around the target neighborhood, and dense routing collapses toward a preferred output mode.
c, Measured local structure (top) and neighborhood-emphasis map from the guided solver (bottom). The lower map retains a coarse near-diagonal angle ordering similar to the calibrated fingerprints.
d, Angle-specific prediction profiles at four representative directions (55°, 70°, 95°, and 100°): the guided solver produces tighter local prediction profiles, whereas router-bypass shows broader off-axis leakage around the same targets.
e, Per-angle clean accuracy: five-seed mean P(correct) across the 37 measured angles, shown as a 3-angle centered moving-average display with light +/-1 s.e.m. shading, comparing the four readout families on the clean sweep. The guided solver retains the highest clean mean accuracy overall, whereas dense routing remains near chance across almost the entire angle set.

## Fig. 6 (5 panels)

**Fig. 6 | Matched calibration reveals object-conditioned directional structure beyond the acrylic reference object.**
a, Five target objects in display order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects despite different response patterns.
c, Centered-magnitude SVD summary across objects. Cumulative centered-\(|H|\) energy and the corresponding rank-90 and rank-95 markers provide a supporting compactness comparison for the five calibrated template matrices.
d, Object-conditioned readout across the five objects. All five objects remain above chance under matched object-specific calibration, and the energy-versus-accuracy comparison shows that Top-1 accuracy does not monotonically track overall response energy across this sample. The accompanying Top-1 confidence intervals summarize uncertainty while preserving that mismatch across objects.
e, Frequency structure across objects. Per-object spectra and directional band profiles indicate that the informative frequency band can shift across objects, so the matched-calibration readout remains object-conditioned.
