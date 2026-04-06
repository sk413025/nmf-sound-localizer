## Fig. 1 (5 panels)

**Fig. 1 | Direction-dependent structural filtering revealed by single-point laser vibrometry.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Conceptual schematic: the structure acts as a direction-dependent spectral filter \(\mathcal H(\theta, f)\) and transforms a flat broadband source into angle-specific spectral fingerprints.
c, Broadband spectral reshaping under matched calibration. The flat white-noise source spectrum (grey dashed) is redistributed differently at five representative angles (0°, 45°, 90°, 135°, 180°), showing that direction changes the measured output spectrum.
d, Full angle-frequency heatmap of the mean white-noise calibration response across the 37 measured angles (0°–180°). The response varies systematically across both angle and frequency, making the directional fingerprint visible as organized full-band structure.
e, Frequency-dependent directivity: polar plot of normalized \(|\mathcal H(\theta, f)|\) across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz). Different bands emphasize different directional sectors, indicating that the directional response is frequency-dependent.

## Fig. 2 (6 panels)

**Fig. 2 | Calibration fingerprints occupy a compact angle-ordered space.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. Energy accumulates rapidly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. The auxiliary angle-ordering proxy rises over the same early-component regime.
b, Frequency-selective spectra \(|u_r(f)|\) for representative components 1, 2, and 6. These traces summarize three reusable spectral patterns in the centered-magnitude decomposition.
c, Direction-selective half-plane polar patterns \(v_r(\theta)\) for representative components 1, 2, and 6, showing how those same components vary across 0°–180°.
d, Full angle-frequency heatmap of the template matrix \(|H|\) (37 angles × 346 frequency bins), showing that neighboring directions reuse related spectral structure rather than forming unrelated fingerprints.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE falls markedly by the same six-component regime highlighted in panel a.
f, Inter-angle fingerprint similarity matrix of \(H\). The near-diagonal high-similarity band shows that local angle ordering remains concentrated among neighboring directions.

## Fig. 3 (6 panels)

**Fig. 3 | Speech preserves directional structure while broadening nearby-angle overlap.**
a, White-noise stimulus: violin plot of within-angle versus between-angle Pearson correlations (d = 2.83, within r̄ = 1.000).
b, Speech stimulus: same analysis (d = 1.95, within r̄ = 0.907); within-angle similarity remains higher than between-angle similarity despite content variation.
c, Per-angle discriminability margin (within r − between r) for white noise (Δr̄ = 0.28) and speech (Δr̄ = 0.11), shown with light bootstrap uncertainty bands; speech margin is reduced but positive at all angles.
d, Stage-0 correlation-based first-choice diagnostic across angle for white noise and held-out speech, shown with light clip-level uncertainty bands. The diagnostic remains strong on most white-noise fingerprints but drops to near chance across the calibrated grid for held-out speech clips.
e, Split-triangle pairwise fingerprint similarity map: lower-left = white noise (near-identity), upper-right = speech (broader local overlap but retained angle ordering), with the diagonal masked to separate the two regimes.
f, Dose-response of the same stage-0 correlation-based first-choice diagnostic versus SNR. The white-noise signal curve (blue, clip-level SEM shading) is recomputed on synthetic noisy white-noise datasets, whereas the speech-plus-babble curve (orange, 5-seed mean ± SEM shading) comes from a separate five-seed sweep; within those matched surfaces, both decline monotonically with increasing noise.

## Fig. 4 (5 panels)

**Fig. 4 | Preserving measured local continuity before subtraction stabilizes readout.**
a, Architecture and physics correspondence. The measured local band in \(H\) is shown beside the staged broad-match, local-gate, and local-update profiles on the shared angle axis, defining the neighborhood that should be preserved before subtraction.
b, Broad initial match for a representative 70° validation clip. Broad residual-to-dictionary matching excites several nearby calibrated angles above the measured local band in \(H\).
c, Local gate convergence for the same representative clip. Learned local pooling contracts support around the correct neighborhood and the resulting update remains confined there.
d, Residual purification. Left, relative residual norm across the guided steps for the representative clip. Right, validation-wide cumulative update mass within radius before and after one local step across 1,924 clips; the mass within 15° rises from 0.18 to 0.98.
e, Clean-condition Top-1 comparison of four overlap-handling rules on the same fixed dictionary (individual seeds shown as dots; large markers and error bars indicate mean \(\pm\) s.e.m.). The displayed clean means are 0.98 for the guided solver, 0.58 for router-bypass, 0.44 for the OMP baseline, and 0.03 for dense routing, consistent with accurate subtraction requiring preservation of the local angle band.

## Fig. 5 (6 panels)

**Fig. 5 | Prediction structure stays organized around the measured local band.**
a, Five-seed SNR sweep for the same four overlap-handling rules. The delayed-commitment rule degrades least as noise increases.
b, Per-angle clean accuracy: five-seed mean P(correct) across the 37 measured angles, displayed as a 3-angle centered moving-average view with light +/-1 s.e.m. shading for the same four overlap-handling rules on the clean sweep. The delayed-commitment rule retains the highest clean mean accuracy overall, whereas dense routing remains near chance across almost the entire angle set.
c, Row-normalized clean confusion comparison of the OMP baseline and guided solver. Guided decoding stays concentrated near the diagonal, whereas the OMP baseline shows broader off-diagonal fracture after early commitment.
d, Measured local structure in the calibrated fingerprint space. The near-diagonal band summarizes the physical neighborhood that nearby angles share.
e, Guided-solver neighborhood-emphasis map on the same angle frame and correlation scale as panel d. The learned map retains a similar coarse near-diagonal ordering rather than dispersing weight broadly across the grid.
f, Quantitative structure alignment. Top, min-max normalized local-band profiles from the measured and learned maps across angle. Bottom, concordance scatter of the same per-angle local-band scores. The full-matrix agreement remains positive (r = 0.47), and the per-angle local-band profiles remain positively aligned (r = 0.46; mean absolute gap = 0.23).

## Fig. 6 (5 panels)

**Fig. 6 | A finite locally ordered directional code recurs across passive objects under matched calibration.**
a, Measured response-regime map across the five target objects. Each object is placed by the zero crossing of its mean centered-\(|H|\) correlation-decay curve and by the effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means remain above chance across the five objects. A descriptive comparison places object mean Top-1 against mean top-3 subspace overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Response energy alone does not order the objects, whereas the overlap-burden metric provides the more informative comparison axis.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. Informative frequency structure therefore remains object-conditioned rather than collapsing to one shared band.
