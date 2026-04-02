## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Input-to-output spectral shaping: the flat white-noise source spectrum (grey dashed) is reshaped differently at five representative angles (0°, 45°, 90°, 135°, 180°).
d, Trial repeatability: mean spectra (±1 s.d. shading) from three independent white-noise recordings at each of five angles.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz), showing that each band carries a distinct directional response pattern.

## Fig. 2 (6 panels)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. The cumulative curve rises quickly across the 37-angle grid: six modes capture 80.3% of the energy and eight capture 85.1%. The overlaid direction-decoding trace is included as a visual comparison.
b, Frequency-selective spectra \(|u_r(f)|\) for representative Modes 1, 2, and 6. These traces show three reusable spectral patterns in the compressed representation.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative Modes 1, 2, and 6, showing how those same modes vary across 0°–180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing structured spectral variation across directions.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-mode regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that neighboring angles remain close in fingerprint space, revealing the local angle ordering later compared with the guided neighborhood map in Fig. 5c.

## Fig. 3 (6 panels)

**Fig. 3 | Directional structure persists under speech, but first-choice matching fails.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); encoding remains significant despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11), shown with light bootstrap uncertainty bands; speech margin is reduced but positive at all angles.
d, Stacked angle-resolved correlation-based first-choice diagnostic traces for white noise and speech, shown with light clip-level uncertainty bands: the diagnostic performs strongly on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response curves: correlation-based greedy diagnostic accuracy versus SNR for white-noise signal (blue, clip-level SEM shading) and speech signal with babble noise (orange, 5-seed mean ± SEM shading), both declining monotonically with increasing noise.

## Fig. 4 (4 panels)

**Fig. 4 | Nearby overlap narrows after one refinement step.**
a, Sequence summary illustrating the broad-match -> neighborhood-concentration -> lower-residual progression.
b, Shared 70° case: the initial broad match is concentrated into one local neighborhood around 70°.
c, After one refinement step, less signal remains once the overlapping neighborhood has been consolidated. Compact callouts report the same shift numerically: the fraction of the match within 15° of the true direction rises from 0.18 to 0.98, while the residual falls from 1.00 to 0.48.
d, Clean-condition comparison across the four benchmarked approaches over the same five-seed sweep (individual seeds shown as dots; points with horizontal bars indicate mean \(\pm\) s.e.m.). The most locally concentrated comparison is most accurate in this restricted setting.

## Fig. 5 (5 panels)

**Fig. 5 | Benchmark predictions remain aligned with the measured local structure.**
a, SNR degradation curves comparing four benchmarked approaches across additive-noise levels; the most locally concentrated comparison degrades least as noise increases.
b, Row-normalized benchmark comparison across the four benchmarked approaches. The most locally concentrated comparison stays nearest the diagonal, whereas the less locally constrained alternatives show broader off-axis leakage or collapse toward a preferred output mode.
c, Correlation structure of the measured template matrix \(H\) (top) and the corresponding neighborhood-emphasis map (bottom), showing that the lower map places most of its weight near the same diagonal angle ordering seen in the measured fingerprints.
d, Angle-specific prediction profiles at four representative directions (55°, 70°, 95°, and 100°): the most locally concentrated comparison produces tighter local prediction profiles, whereas the less locally constrained alternative shows broader off-axis leakage.
e, Per-angle readout accuracy: five-seed clean mean P(correct) across the 37 measured angles, shown as a 3-angle centered moving-average display with light +/-1 s.e.m. shading, comparing the four benchmarked approaches. The most locally concentrated comparison retains the highest clean mean accuracy overall, whereas the least locally constrained comparison remains near chance across almost the entire angle set.

## Fig. 6 (5 panels)

**Fig. 6 | Matched calibration reveals bounded recurrence across an exploratory five-object screen.**
a, Five target objects in screening order: cardboard box, wooden board, acrylic plate, paper cup, and laptop shell.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects despite different response patterns.
c, Low-rank continuity. Applying the same centered-magnitude SVD view used in Fig. 2 to each object's template matrix again shows early energy capture across the five-object screen.
d, Screening performance across the five objects. All five objects remain above chance under matched calibration, and the energy-versus-accuracy comparison shows that Top-1 screening accuracy does not monotonically track overall response energy across this sample. The accompanying Top-1 confidence intervals summarize screening uncertainty while preserving that mismatch across objects.
e, Frequency structure across objects. Per-object spectra and directional band profiles indicate that the informative frequency band can shift across objects.
