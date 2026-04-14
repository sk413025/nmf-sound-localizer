## Fig. 1 (5 panels)

**Fig. 1 | A passive acrylic plate exposes directional coding at one measurement point.**
a, Photograph of the experimental setup (loudspeaker excitation, acrylic sensor plate and laser Doppler vibrometer (LDV)).
b, Physical-principle schematic corresponding to Supplementary Methods 1. Changing input direction alters how the acrylic plate excites and combines structural modes, so one fixed LDV point records a different single-point spectral fingerprint even though the measurement location does not move.
c, Broadband spectral reshaping with repeatability under matched calibration. The flat white-noise source spectrum (grey dashed) is redistributed differently at five representative angles (0°, 45°, 90°, 135°, 180°); light traces show repeated calibration trials and bold traces show the angle-wise mean response.
d, Frequency-dependent directivity across 0°–180° for four frequency bands (0.3–0.5, 0.5–1, 1–2, 2–3 kHz). Different bands emphasize different directional sectors, showing that directional encoding is distributed unevenly across frequency.
e, Inter-angle fingerprint similarity matrix of the calibrated acrylic dictionary \(H\). The near-diagonal high-similarity band shows that nearby directions remain related without becoming interchangeable, providing the first direct evidence that the fingerprints trace a local geometry.

## Fig. 2 (6 panels)

**Fig. 2 | The directional code is compact and locally ordered.**
a, Singular-value spectrum of the centered-magnitude fingerprint matrix. Energy accumulates rapidly across the 37-angle grid: six components capture 80.3% of the energy and eight capture 85.1%. Most measured directional structure therefore sits in a small component set.
b, Representative component spectra \(|u_r(f)|\) for components 1, 2, and 6. Together they show three reusable spectral patterns in the centered-magnitude decomposition.
c, Matching half-plane polar profiles \(v_r(\theta)\) for components 1, 2, and 6. They show how those same components vary across 0°-180°.
d, Local-ordering decay in centered-\(|H|\). Mean inter-angle correlation is plotted against angular separation for the acrylic reference object, showing a finite positive local neighborhood that decays toward a first nonpositive mean near 25°.
e, All-angle reconstruction fidelity under rank-\(r\) truncation. Per-angle centered-magnitude RMSE drops sharply in the same six-component regime highlighted in panel a, showing that the local code is captured early.
f, Two-dimensional spectral embedding of the positive centered-neighborhood graph derived from centered-\(|H|\). The measured angle trajectory stays curved and locally ordered rather than fragmenting into isolated clusters, so the finite neighborhood quantified in panel d remains coherent in a graph-based view.

## Fig. 3 (6 panels)

**Fig. 3 | Speech preserves the code but broadens local overlap.**
a, Mirrored compactness on the centered angle-conditioned summary surface. Calibration and speech both show early cumulative-energy saturation, so the speech-side directional code remains compact rather than dissolving into a high-rank response family.
b, Speech-side local-ordering decay. Mean inter-angle correlation is plotted against angular separation for calibration and speech on matched centered summary surfaces. Speech keeps a positive nearest-angle neighborhood but broadens it relative to calibration.
c, Neighborhood-coherence map in split-triangle form: lower-left = calibration centered-neighborhood similarity, upper-right = speech-side centered similarity. Speech preserves the coarse angle ordering and near-diagonal structure while widening the local band.
d, Stage-0 local separability on the frozen grouped-match surface. Mean cumulative match mass within radius \(r\) is plotted for white noise and held-out speech. Speech retains substantial local support within nearby angle bands even as exact support weakens.
e, Angle-resolved exact first-choice collapse. Stage-0 exact-match accuracy across the calibrated grid remains high for white noise but drops sharply for held-out speech.
f, Speech exact versus local tolerance. On the same stage-0 grouped-match family, within-10° tolerance remains well above exact-match success across the grid, showing that the failure under speech is local rather than random.

## Fig. 4 (5 panels)

**Fig. 4 | Preserving the broadened neighborhood keeps subtraction admissible.**
a, Neighborhood admissibility strip. The measured local neighborhood in \(H\) is shown beside the representative broad-match, local-gate, and local-update profiles on the shared angle axis, defining the neighborhood that should be preserved before subtraction.
b, Broad initial match for a representative 70° validation clip. Stage-0 matching excites several nearby calibrated angles above the measured local band.
c, Local contraction for the same representative clip. The routed update contracts that broad support into the physically plausible neighborhood before subtraction proceeds.
d, Validation-wide neighborhood contraction. Cumulative update mass within radius is plotted before and after one guided step across the validation set. The entire curve shifts inward, showing that the first routed update sharpens support toward the local band exposed by calibration.
e, Within-15° mass gain across validation clips. Per-clip before/after summaries show that the first guided step consistently increases mass inside the physically plausible neighborhood.

## Fig. 5 (6 panels)

**Fig. 5 | The admissible readout follows the measured neighborhood.**
a, Five-seed SNR sweep for the same four overlap-handling rules. The delayed-commitment rule degrades least as noise increases, consistent with preserving local overlap before subtraction.
b, Per-angle clean accuracy across the 37 measured angles. After light 3-angle smoothing, the delayed-commitment rule remains highest across most angles, whereas dense routing stays near chance across much of the grid; light shading shows \(\pm 1\) s.e.m. around the five-seed mean \(P(\mathrm{correct})\). The neighborhood therefore governs readout across the grid, not only in one average metric.
c, Clean confusion comparison (row-normalized) of the OMP baseline and guided solver. Guided decoding concentrates near the diagonal, whereas the OMP baseline shows broader off-diagonal fracture after early commitment.
d, Measured local structure in the calibrated fingerprint space. The near-diagonal band summarizes the physical neighborhood that nearby angles share.
e, Learned neighborhood-emphasis map from the guided solver on the same angle frame and correlation scale as panel d. The learned map keeps a similar coarse near-diagonal ordering instead of dispersing weight broadly across the grid.
f, Quantitative structure alignment. Top, normalized local-band summaries from the measured and learned maps across angle. Bottom, concordance scatter of the same per-angle scores. The learned map follows the same measured neighborhood identified by calibration: the full-matrix agreement reaches \(r = 0.47\), the per-angle summaries reach \(r = 0.46\), and the mean absolute gap is 0.23. Both correlations exceed the corresponding 95th-percentile permutation nulls (\(r = 0.41\) globally and \(r = 0.32\) for the local-band profile), so the alignment is stronger than shuffled angle structure would permit.

## Fig. 6 (5 panels)

**Fig. 6 | A recurring directional code appears across ordinary passive objects.**
a, Measured response-regime map across the five target objects. Each object is placed by the zero crossing of its mean centered-\(|H|\) correlation-decay curve and by the effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means exceed chance across the five objects. Object mean Top-1 is plotted against mean top-3 overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Local separability, not response energy alone, is what orders the objects.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. The informative band changes from object to object, but the directional principle does not.
