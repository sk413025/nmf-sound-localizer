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

## Fig. 4 (6 panels)

**Fig. 4 | Preserving the broadened neighborhood keeps subtraction admissible.**
a, Admissibility synthesis. On one local angle frame, the broad first-step physical match, its contracted post-routing profile, and one representative 70° validation clip are plotted against the measured-neighborhood band. The admissible update therefore contracts broad local support back into the measured band rather than sharpening outside it.
b, Routing carves a local update. For the same representative clip, the broad physical match, learned routing cue, routed weight profile, and resulting gated update are plotted on one shared angle frame. Routing concentrates support from within the broad physical match rather than inventing a new neighborhood.
c, First-step operating-point recovery. Guided-only, one-step before-versus-after mass is summarized at exact, 5°, 10°, and 15° operating thresholds. The first guided step sharply recovers exact commitment while keeping local commitment inside the operative neighborhood nearly saturated.
d, Validation-wide neighborhood contraction. Cumulative update mass within radius is plotted before and after one guided step across the validation set. The full curve shifts inward, showing that the first routed update sharpens support toward the local band exposed by calibration.
e, Angle-resolved within-15° contraction. Mass inside the operative 15° neighborhood is plotted before and after the first guided step for each target angle, showing that the inward shift remains positive across the directional grid.
f, Clip-level within-15° gain CDF. The empirical cumulative distribution of per-clip gain inside the operative 15° neighborhood shows that the first guided step increases local mass for nearly all validation clips.

## Fig. 5 (7 panels)

**Fig. 5 | Final prediction succeeds by preserving the measured neighborhood.**
a, Neighborhood-preservation cascade on one shared radius axis. Row-normalized mass within a given angular radius is plotted for speech stage-0 grouped support, after one guided step, and at final guided prediction. Early speech support is broad but still local; the guided update contracts that same support, and the final predictions remain inside the same neighborhood.
b, Family neighborhood preservation. The same cumulative mass-within-radius statistic is computed from the final clean family runs reconstructed on the governed babble protocol. Decoder success is ordered by how much local mass each family retains.
c, Exact-versus-local consequence. Exact clean accuracy is plotted against within-15° local mass for the same four decoder families. Families that preserve more local support also achieve stronger exact clean prediction.
d, Final prediction locality by family. Clean row-normalized confusion maps for guided solver, router-bypass, OMP baseline, and dense routing show how the accepted hierarchy appears directly at the decision surface: guided predictions remain tightly concentrated near the diagonal, router-bypass broadens modestly, OMP fractures further, and dense routing disperses broadly.
e, Measured neighborhood geometry. The angle-angle correlation map of calibrated \(H\) shows the near-diagonal physical neighborhood that nearby angles share.
f, Family-to-measured neighborhood alignment. Bars show a local-band agreement score for each decoder family, defined by retained within-15° local mass multiplied by a bounded anglewise profile-agreement factor relative to the measured neighborhood profile. Open circles show the corresponding full-matrix correlation to the measured neighborhood as a secondary reference. Families that keep support both local and correctly ordered score highest on the primary local-band metric.
g, Noise robustness consequence. Five-seed babble SNR sweeps for the same four overlap-handling rules show that the neighborhood-preserving hierarchy remains visible under noisy evaluation.

## Fig. 6 (5 panels)

**Fig. 6 | A recurring directional code appears across ordinary passive objects.**
a, Measured response-regime map across the five target objects. Each object is placed by the first non-positive separation of its mean centered-\(|H|\) correlation-decay curve and by the entropy-equivalent effective rank of its centered-\(|H|\) matrix, so the panel summarizes response width and compactness without assigning unmeasured material constants.
b, Per-object template matrices \(H\). Shared-normalization heatmaps show structured angle-frequency encoding across the five objects; the colored horizontal band on each heatmap marks the object-specific frequency region that carries the strongest directional contrast.
c, Local-ordering decay across objects. Mean centered-\(|H|\) correlation is plotted against angular separation, with open markers denoting the first zero crossing for each object. Every object retains a finite neighborhood of positive local correlation, but the decay width differs across materials.
d, Object-conditioned readout versus overlap burden. Per-angle Top-1 distributions and object means exceed chance across the five objects. Object mean Top-1 is plotted against mean top-3 subspace overlap burden, with marker area proportional to normalized overall \(|H|\) energy. Local separability, not response energy alone, is what orders the objects.
e, Selected bands and recovered directional codes. A shared contrast-selection rule identifies different informative frequency windows across objects, and the corresponding band-limited directional codes differ across those object-specific bands. The informative band changes from object to object, but the directional principle does not.
