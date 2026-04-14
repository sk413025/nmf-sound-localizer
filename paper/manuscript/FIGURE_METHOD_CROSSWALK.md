# Figure-Method Crosswalk

This document is the canonical panel-to-method crosswalk for the active
main-paper figures. It links each panel in `Fig. 1-6` to its scientific job,
its panel type, the quantity it encodes, and the main-text / supplementary
methods surface that makes the panel interpretable.

It is not a second figure legend, a second naming contract, or a second review
workflow. Its job is narrower: keep the `figure spine`, the `method spine`, and
the `supplementary spine` on one explicit map so figure redesign can be judged
against scientific role rather than visual taste alone.

## Whole-Paper Spine

- `Fig. 1` opens the phenomenon: a passive acrylic plate converts direction into
  repeatable single-point spectral fingerprints.
- `Fig. 2` shows that those fingerprints occupy a compact, finite, coherent
  local neighborhood rather than a set of unrelated templates.
- `Fig. 3` shows that held-out speech preserves the same code but broadens it
  into structured local ambiguity on the stage-0 grouped-match surface.
- `Fig. 4` defines the admissible first-step rule: broad local support must be
  preserved and contracted before subtraction.
- `Fig. 5` follows the same locality metric to final prediction, decoder-family
  hierarchy, and noisy consequence.
- `Fig. 6` asks whether the same locally ordered directional code recurs across
  a broader set of passive-object archetypes.

Current diagnosis: the Results spine is coherent, but the figure-to-method
traceability is uneven. `Fig. 4` is strongly anchored to the routed-update
equations; `Fig. 2f`, `Fig. 5f`, and parts of `Fig. 6` are more weakly anchored
and rely more heavily on prose explanation.

Legend for `Panel type`:

- `observation`
- `derived statistic`
- `mechanism instantiation`
- `consequence / validation`

Legend for `Linkage type`:

- `direct equation-backed`
- `method-defined statistic`
- `observation-led with method context`

Legend for `Linkage verdict`:

- `strong`
- `adequate`
- `weak`

Legend for `Action`:

- `keep`
- `tighten prose`
- `tighten methods anchor`
- `redesign candidate`

## Fig. 1

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `1a` Setup photograph | Show the real single-point LDV acquisition geometry | observation | none | raster setup image | One fixed LDV point reads a passive plate under directional excitation | Introduction setup paragraph; Experimental setup context | Supp. Methods 1 physical context only | observation-led with method context | adequate | low | keep |
| `1b` Physical-principle schematic | State the physical mechanism: direction changes modal weighting at one point | mechanism instantiation | none | schematic mode-combination flow | Changing direction changes how structural modes combine at the measured point | Results opener for Fig. 1; Methods Eq. (1) | Supp. Methods 1, Eqs. (S1-S9) | direct equation-backed | strong | low | keep |
| `1c` Spectral shaping with repeatability | Show that direction yields repeatable angle-specific fingerprints | observation | frequency (kHz) | normalized amplitude | Flat white noise is reshaped differently at each angle, reproducibly across trials | Results Fig. 1 paragraph on reproducible spectral code; Methods Eq. (3) and standardized fingerprints | Supp. Methods 2, Eqs. (S10-S15) | observation-led with method context | strong | low | keep |
| `1d` Frequency-dependent directivity | Show that directional encoding is uneven across frequency bands | derived statistic | angle (polar half-plane) | band-limited directivity magnitude | Different bands emphasize different directional sectors | Results Fig. 1 paragraph on band-specific directivity | Supp. Methods 2 measured fingerprints; no single unique panel equation | method-defined statistic | adequate | low | keep |
| `1e` Inter-angle fingerprint similarity | Show the first direct evidence of local geometry in calibrated `H` | derived statistic | angle (deg) | angle-angle similarity in calibrated `H` | Nearby angles remain related without becoming interchangeable | Results transition from Fig. 1 to Fig. 2; Methods `H` construction | Supp. Methods 2, Eq. (S15); Supp. Methods 5, Eq. (S44) | method-defined statistic | strong | low | keep |

**Figure verdict**

- Chapter role: phenomenon opener
- Method linkage quality: strong overall because `1b` and `1e` already anchor the
  physical model and the measured dictionary
- Main risk: `1d` is useful but rests on descriptive band summaries rather than
  a named methods statistic

## Fig. 2

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `2a` Singular-value spectrum | Show rapid early saturation of centered-magnitude structure | derived statistic | component index `r` | cumulative energy / singular value | Most directional structure sits in a small component set | Results compactness paragraph; centered-magnitude section | Supp. Methods 2, Eqs. (S16-S17) | method-defined statistic | strong | low | keep |
| `2b` Representative component spectra | Show reusable spectral patterns in the centered decomposition | derived statistic | frequency (kHz) | relative loading | A few reusable spectral patterns explain the code | Results compactness paragraph | Supp. Methods 2, Eq. (S17) | method-defined statistic | adequate | low | keep |
| `2c` Angle-selective loadings | Show that those component patterns vary systematically across angle | derived statistic | polar angle | relative loading | The same components map onto ordered directional profiles | Results compactness paragraph | Supp. Methods 2, Eq. (S17) | method-defined statistic | adequate | low | keep |
| `2d` Local-ordering decay | Quantify the finite positive neighborhood in centered `|H|` | derived statistic | angular separation (deg) | mean centered-`|H|` correlation | The code occupies a finite positive local neighborhood | Results finite-neighborhood paragraph | Supp. Methods 5, Eq. (S45) | method-defined statistic | strong | low | keep |
| `2e` Reconstruction fidelity | Show that the same early component regime reconstructs all angles well | consequence / validation | angle (deg) | RMSE under rank truncation | The local code is captured early, not only compact in abstract | Results compactness paragraph | Supp. Methods 2, Eqs. (S16-S17) | method-defined statistic | strong | low | keep |
| `2f` Centered-neighborhood graph embedding | Give a descriptive coherence view of the same local neighborhood | derived statistic | graph axis 1 | graph axis 2, color = angle | The finite neighborhood remains coherent in a graph view | Results coherence paragraph | Supp. Methods 5, Eq. (S45) plus graph-Laplacian construction after it | method-defined statistic | adequate | moderate with `2d` | tighten prose |

**Figure verdict**

- Chapter role: compact finite neighborhood discovery
- Method linkage quality: strong for `2a/d/e`; weaker for `2f`
- Main risk: `2f` is scientifically useful but needs constant prose bounding so
  it reads as a descriptive complement rather than a second latent-space thesis

## Fig. 3

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `3a` Mirrored compactness | Show that speech preserves compactness on a matched centered summary surface | derived statistic | component index `r` | cumulative energy fraction | Speech keeps the directional code compact | Results speech-side compactness paragraph | Supp. Methods 5, Eqs. (S46-S48) | method-defined statistic | strong | low | keep |
| `3b` Speech-side local-ordering decay | Show that speech preserves but broadens the local neighborhood | derived statistic | angular separation (deg) | mean centered correlation | The speech-side neighborhood stays positive but widens | Results neighborhood-width paragraph | Supp. Methods 5, Eq. (S49) | method-defined statistic | strong | low | keep |
| `3c` Neighborhood-coherence map | Show calibration-vs-speech neighborhood structure on one matched similarity family | derived statistic | angle (deg) | angle-angle centered similarity | Coarse angle ordering survives under speech | Results neighborhood-width paragraph | Supp. Methods 5, Eq. (S49) | method-defined statistic | strong | low | keep |
| `3d` Stage-0 local separability | Show that speech weakens exact support while retaining local mass before any correction | consequence / validation | neighborhood radius (deg) | stage-0 mass within radius | Speech retains local support even as exact support weakens | Results stage-0 ambiguity paragraph; main-text Eq. (2) surrogate | Supp. Methods 3, Eq. (S29); Supp. Methods 5, Eqs. (S50-S51) | direct equation-backed | strong | low | keep |
| `3e` Exact first-choice collapse | Show where immediate exact commitment fails under speech | consequence / validation | true angle (deg) | exact Top-1 rate | Exact first choice collapses under held-out speech | Results stage-0 ambiguity paragraph | Supp. Methods 5, Eq. (S52) | method-defined statistic | strong | low | keep |
| `3f` Speech exact vs local tolerance | Show that failure under speech is local rather than random | consequence / validation | true angle (deg) | speech match rate | Within-10 deg tolerance stays above exact success | Results stage-0 ambiguity paragraph | Supp. Methods 5, Eq. (S53) plus Eq. (S52) comparison | method-defined statistic | strong | low | keep |

**Figure verdict**

- Chapter role: speech broadens the code into local ambiguity
- Method linkage quality: strong and well-partitioned
- Main risk: low; the two surface families are distinct and already anchored to
  different supplementary methods blocks

## Fig. 4

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `4a` Admissibility synthesis | Define the admissible first-step rule on one local frame | mechanism instantiation | local angle window (deg) | normalized support | Broad support must contract inside the measured neighborhood before subtraction | Results admissibility paragraph; main-text Eqs. (4), (6), (7) | Supp. Methods 4, Eqs. (S31), (S37), (S38) | direct equation-backed | strong | low | keep |
| `4b` Routing carves a local update | Show how the learned cue and routed weight carve an update from broad physical support | mechanism instantiation | local angle window (deg) | normalized profiles for support, cue, weight, update | The update is carved from within the broad support, not invented outside it | Results mechanism paragraph; main-text Eqs. (4), (5), (6) | Supp. Methods 4, Eqs. (S31-S37) | direct equation-backed | strong | low | keep |
| `4c` First-step operating-point recovery | Show the immediate consequence of one guided step at exact / 5 / 10 / 15 deg thresholds | consequence / validation | local commitment threshold | mass within threshold | The first step sharply recovers exact and local commitment | Results operating-point paragraph; main-text Eq. (8) in consequence language | Supp. Methods 4 neighborhood-preserving update plus Fig. 5 locality statistic paragraph | method-defined statistic | adequate | moderate with `5c` | tighten methods anchor |
| `4d` Validation-wide neighborhood contraction | Show the main population-level inward shift over radius | consequence / validation | neighborhood radius (deg) | cumulative mass within radius | The first guided step shifts support inward validation-wide | Results contraction paragraph | Supp. Methods 4 final locality statistic paragraph | method-defined statistic | strong | low | keep |
| `4e` Angle-resolved within-15 deg contraction | Show that the same inward shift holds across the directional grid | consequence / validation | true angle (deg) | mass inside 15 deg | The 15 deg gain stays positive across angle | Results contraction paragraph | Supp. Methods 4 locality statistic at operative radius | method-defined statistic | strong | low | keep |
| `4f` Clip-level within-15 deg gain CDF | Show that the effect is not driven by a minority of clips | consequence / validation | gain in mass inside 15 deg | cumulative fraction of clips | Nearly all clips gain local mass after one step | Results contraction paragraph | Supp. Methods 4 locality statistic at clip level | method-defined statistic | strong | low | keep |

**Figure verdict**

- Chapter role: admissible first-step contraction pivot
- Method linkage quality: strongest in the paper
- Main risk: `4c` has a valid role, but its operating-point statistic should be
  kept explicitly step-level and guided-only so it does not drift into `Fig. 5c`

## Fig. 5

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `5a` Neighborhood-preservation cascade | Carry one locality metric from stage-0 support to first-step contraction to final guided prediction | consequence / validation | neighborhood radius (deg) | row-normalized mass within radius | The same locality metric survives to final prediction | Results Fig. 5 opening; main-text Eq. (8) in final prediction context | Supp. Methods 4 locality statistic paragraph after Eqs. (S39-S42) | method-defined statistic | strong | low | keep |
| `5b` Family neighborhood preservation | Order decoder families by retained local mass at final prediction | consequence / validation | neighborhood radius (deg) | cumulative mass within radius | Decoder success is ordered by local mass retention | Results family locality paragraph | Supp. Methods 4 locality statistic paragraph | method-defined statistic | strong | low | keep |
| `5c` Exact-versus-local consequence | Show that local mass predicts exact clean consequence across families | consequence / validation | within-15 deg local mass | exact clean accuracy | Families that preserve more local support predict better exactly | Results exact-vs-local paragraph | Supp. Methods 4 locality statistic paragraph plus coherent family protocol note | method-defined statistic | adequate | moderate with `4c` | tighten methods anchor |
| `5d` Final prediction locality by family | Make the family hierarchy visible directly as confusion morphology | consequence / validation | predicted angle (deg) | true angle (deg), row-normalized confusion | The same hierarchy is visible in final morphology | Results morphology paragraph; main-text Eq. (8) at final prediction level | Supp. Methods 4 readout and locality extension paragraph | method-defined statistic | adequate | low | keep |
| `5e` Measured neighborhood geometry | Reintroduce the measured reference surface for final-family explanation | derived statistic | angle (deg) | angle-angle correlation of calibrated `H` | The measured neighborhood is the common reference surface | Results reference-surface paragraph; centered-`H` description | Supp. Methods 2, Eq. (S15); Supp. Methods 5, Eq. (S44) | method-defined statistic | strong | low | keep |
| `5f` Family-to-measured neighborhood alignment | Compare each family back to the measured neighborhood | consequence / validation | family / decoder | local-band agreement bar plus full-matrix correlation point | Stronger decoders retain the highest neighborhood agreement | Results alignment paragraph | Supp. Methods 4 final note on Fig. 5f alignment summary and permutation null | method-defined statistic | weak | moderate with `5e` | tighten methods anchor |
| `5g` Noise robustness consequence | Show that the same hierarchy persists under babble degradation | consequence / validation | babble SNR (dB) | Top-1 accuracy | Noise robustness follows the same hierarchy | Results noise-robustness paragraph | coherent family sweep provenance; no single main-text equation | method-defined statistic | adequate | low | keep |

**Figure verdict**

- Chapter role: final family consequence under one locality metric
- Method linkage quality: strong for `5a-b-e`; weaker for `5f`
- Main risk: `5f` has a strong paper role but a weak visible methods definition in
  the main manuscript, so it depends heavily on supplementary anchoring

## Fig. 6

| Panel / title | Scientific job | Panel type | X-axis | Y / encoding | Legend claim | Main-text anchor | Supplementary anchor | Linkage type | Linkage verdict | Overlap risk | Action |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `6a` Measured response-regime map | Summarize each object's response width and compactness on one descriptor map | derived statistic | correlation-decay width (deg) | effective rank | Response geometry differs across objects without invoking intrinsic constants | Results cross-object opening paragraph; centered-`H` descriptor text | Supp. Methods 6 descriptor interpretation | method-defined statistic | weak | moderate with `2a/2d` | tighten methods anchor |
| `6b` Per-object template matrices `H` | Show structured angle-frequency fingerprints across objects | observation | angle (deg) | frequency (kHz), color = log10 `|H|` | Each object retains structured angle-frequency encoding | Results cross-object opening paragraph | Supp. Methods 2 measured fingerprints; Supp. Methods 6 object reading | observation-led with method context | adequate | low | keep |
| `6c` Local-ordering decay across objects | Show that every object retains a finite positive local neighborhood | derived statistic | angular separation (deg) | mean centered-`|H|` correlation | Local ordering recurs across structurally distinct objects | Results cross-object opening paragraph | Supp. Methods 5, Eq. (S45); Supp. Methods 6 descriptor interpretation | method-defined statistic | strong | low | keep |
| `6d` Object-conditioned readout vs overlap burden | Show that recoverability tracks overlap burden more than response energy | consequence / validation | mean top-3 overlap burden | Top-1 mean, plus per-angle distribution and marker area = normalized `|H|` energy | Local separability, not energy alone, orders the objects | Results cross-object consequence paragraph | Supp. Methods 6 descriptor interpretation; Supplementary Table 1 | method-defined statistic | adequate | moderate with `6a` | tighten methods anchor |
| `6e` Selected bands and recovered directional codes | Show that recurrence does not require one universal informative band | derived statistic | selected contrast band (top) / angle (bottom) | recovered band code | The informative band changes across objects but the local directional principle does not | Results cross-object frequency paragraph | Supp. Methods 6 plus Supplementary Table 1 | method-defined statistic | adequate | low | keep |

**Figure verdict**

- Chapter role: bounded cross-object recurrence cash-out
- Method linkage quality: adequate but more descriptor-heavy than earlier figures
- Main risk: `6a` and `6d` need stronger explicit methods anchoring so they read
  as measurement-derived descriptors rather than free-floating summary graphics

## Redesign Priority Ledger

### Prose / legend only

- `2f`: keep its role explicitly descriptive; do not let it sound like a second
  reduced-order model.
- `5f`: make the methods definition of the local-band agreement score more
  visible when the panel is discussed in paper-facing prose.
- `6a`, `6d`: keep saying these are response-level descriptors, not intrinsic
  material constants.

### Methods / supplementary anchoring needed

- `4c`: tie the operating-point recovery panel more explicitly to the locality
  statistic inherited by `Fig. 5`, so it is visibly a step-level bridge rather
  than an isolated scorecard.
- `5f`: promote its current supplementary-only definition into a clearer
  manuscript-facing methods anchor.
- `6a`, `6d`: add clearer mention that the descriptor axes are executed response
  summaries defined in Supplementary Methods 6 and Supplementary Table 1.

### Panel redesign candidates

- none are immediate blockers on whole-paper coherence.
- the current priority is traceability tightening, not large-scale panel
  replacement.

## Working Conclusion

The current active panel map is not principally wrong. The main architecture
problem is traceability unevenness:

- `Fig. 1-4` are already fairly well tied to methods and supplementary logic.
- `Fig. 5f` and parts of `Fig. 6` are the weakest links.
- the paper now needs one stable canonical surface that makes these panel-to-
  method links explicit before any new large-scale figure redesign is attempted.
