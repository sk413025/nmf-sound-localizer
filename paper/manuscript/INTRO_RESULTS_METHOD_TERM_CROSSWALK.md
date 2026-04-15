# Introduction-Results-to-Methods Term Crosswalk

This document is the canonical term-level crosswalk for Methods-relevant
language already used in the `Introduction` and `Results` sections of the main
manuscript. Its job is narrower than a glossary and narrower than a Methods
rewrite. It maps paper-facing term families onto the exact Methods surfaces
that define, quantify, or operationalize them, so terminology drift can be
judged against one explicit ledger.

It is not a second Methods section, a second figure legend, or a manuscript
management checklist. It exists to keep the `front-door prose`, the `Results
spine`, and the `Methods spine` on one shared terminology map.

## Scope And Exclusions

- Scope: terms in `Introduction` and `Results` that imply a representation,
  statistic, inference object, protocol, or evaluation quantity in `Methods`.
- Scope: paper-facing terms that act as bridges between the physical narrative
  and the implemented analysis surface.
- Exclusion: generic content nouns such as `speech`, `white noise`, `object`,
  `angle`, or `LDV` unless they appear as part of a methods-defined construct.
- Exclusion: figure-only labels that do not carry an independent Methods
  meaning.

Status legend:

- `explicitly defined`
- `partially defined`
- `distributed across sections`
- `paper-facing only`

## Whole-Manuscript Diagnosis

- The manuscript already has a stable `local-neighborhood` narrative, but that
  narrative spans two distinct quantity families: `centered-|H|` descriptive
  structure and `radius-based support/locality` metrics for readout behavior.
- The strongest Methods anchors already exist for `centered representation`,
  `template/dictionary` objects, `guided solver` updates, and `mass-within-radius`
  locality metrics.
- The terms most at risk of drift are the paper-facing bridge terms
  `directional code`, `measured neighborhood`, `stage-0 grouped-match surface`,
  and `overlap burden`, because each compresses several lower-level quantities.
- `Introduction` uses some scientifically useful umbrella terms on purpose. The
  crosswalk should keep them, but it should make clear when they are narrative
  labels rather than one-to-one Methods objects.

## Crosswalk

| Term family | Representative manuscript phrases | Section role in Introduction / Results | Primary Methods anchor | Secondary anchor | Current status | Risk / note |
|---|---|---|---|---|---|---|
| `local neighborhood` | `finite local neighborhood`; `broadened neighborhood`; `measured neighborhood`; `operative 15° neighborhood` | Core bridge from the compactness chapter to the admissible-readout and final-family chapters | `Calibration object and centered-magnitude representation`; `Diagnostic analyses`; `Evaluation metrics and statistics` | `Evaluation protocols` | `distributed across sections` | This is the most important umbrella term in the paper, but it names two different surfaces: the descriptive neighborhood in centered `|H|` and the readout-side radius metric. The prose should keep naming the surface shift when moving between them. |
| `local separability / local ambiguity / local overlap` | `local separability problem`; `local ambiguity`; `nearby directions become jointly plausible`; `speech broadens overlap locally` | Names the scientific problem created once the code survives but exact commitment weakens | `Inference formulations and algorithms` | `Diagnostic analyses` | `partially defined` | The surrogate sparse-recovery language and the stage-0 diagnostic define the mechanism, but the paper-facing phrase `local ambiguity` remains a narrative compression rather than one single formula name. |
| `directional code / local directional organization` | `reusable directional code`; `locally ordered directional code`; `local directional organization`; `the code survives speech` | Paper protagonist and front-door discovery label | `Signal processing and feature extraction`; `Calibration object and centered-magnitude representation` | `Diagnostic analyses` | `paper-facing only` | This is intentionally a paper-level term, not a Methods variable. It cashes out through fingerprints, centered `|H|`, neighborhood decay, and readout locality rather than through one object called `directional code`. |
| `fingerprint / spectral fingerprint` | `single-point spectral fingerprint`; `direction-dependent vibration fingerprints`; `calibrated fingerprints`; `speech fingerprints` | Front-door observable that carries directional information before any decoder enters | `Signal processing and feature extraction` | `Experimental setup and data acquisition` | `explicitly defined` | This family is well grounded: the Methods section defines the standardized log-power fingerprint from the LDV response and keeps it as the observable used downstream. |
| `centered representation / centered-|H| / centered-magnitude` | `centered representation`; `centered-magnitude energy`; `centered summary surface`; `centered-neighborhood statistics` | Main descriptive surface for compactness, local ordering, and cross-object descriptors | `Calibration object and centered-magnitude representation` | `Diagnostic analyses`; `Evaluation protocols` | `explicitly defined` | This is one of the cleanest term families in the manuscript. The main consistency requirement is to keep distinguishing centered descriptive summaries from grouped-match or replay surfaces. |
| `compactness / low-rank structure / effective rank` | `compact shared response`; `low-rank space`; `cumulative energy still saturates early`; `effective rank` | Quantifies that the code lives in a limited structural subspace rather than a field of unrelated templates | `Calibration object and centered-magnitude representation`; `Evaluation protocols` | `Diagnostic analyses` | `explicitly defined` | `Compactness` in the acrylic and speech chapters is well defined through SVD-style summaries. `Effective rank` in the cross-object chapter is also defined, but it is a descriptor derived from the same centered surface rather than a new independent object. |
| `local ordering / neighborhood coherence / graph embedding` | `locally ordered`; `mean inter-angle correlation decays`; `neighborhood coherence`; `positive centered-neighborhood graph`; `spectral embedding` | Describes the shape of the neighborhood after centering, especially in Figs. 2 and 3 | `Calibration object and centered-magnitude representation` | `Diagnostic analyses` | `partially defined` | Correlation-decay statistics are clearly grounded. The graph-embedding view is weaker as a primary Methods object and should remain a descriptive complement rather than a second reduced-order thesis. |
| `template matrix H / calibrated dictionary / grouped dictionary / grouped templates` | `calibrated dictionary H`; `template matrix H`; `grouped dictionary`; `grouped templates` | Connects measured calibration fingerprints to the inference surface used for readout | `Calibration object and centered-magnitude representation`; `Inference formulations and algorithms` | `Evaluation protocols` | `distributed across sections` | The manuscript uses `H` for the measured calibration matrix and `D` for the grouped inference dictionary. That distinction exists in Methods, but the Results prose can still drift if `dictionary`, `template`, and `grouped` are used too loosely. |
| `stage-0 grouped-match surface` | `frozen stage-0 grouped-match surface`; `stage-0 local separability`; `stage-0 support`; `stage-0 matching` | Defines the pre-correction diagnostic surface on which speech broadening first appears | `Inference formulations and algorithms`; `Diagnostic analyses` | `Evaluation metrics and statistics` | `partially defined` | This term is operationally grounded by the stage-0 match and the grouped support diagnostics, but the exact manuscript label is still a compact phrase rather than a canonical Methods subsection name. |
| `guided solver / routing / routed update / subtraction` | `guided solver`; `routing`; `learned cue`; `routed weight`; `gated update`; `subtraction remains admissible` | Makes the admissibility rule visible once local ambiguity appears | `Inference formulations and algorithms` | `Training and optimization`; `Diagnostic analyses` | `explicitly defined` | This family is strongly anchored by Eqs. (4)-(8) and the grouped update logic. The main prose risk is not missing definition but accidentally letting the solver become the paper protagonist. |
| `exact support / local support / mass-within-radius / within-15° local mass` | `exact support`; `local support`; `mass within radius`; `within-10° tolerance`; `within-15° local mass` | Shared evaluation vocabulary for the Fig. 3 diagnostic and the Fig. 4-5 locality bridge | `Diagnostic analyses`; `Evaluation metrics and statistics` | `Evaluation protocols` | `explicitly defined` | This family is method-defined and reusable across figures. The main requirement is to keep `exact` and `local` consequence distinct rather than treating them as interchangeable notions of success. |
| `local-band agreement / anglewise profile agreement / full-matrix correlation` | `local-band agreement score`; `bounded anglewise profile-agreement factor`; `full-matrix correlation` | Gives the final-family alignment metric back to the measured neighborhood in Fig. 5f | `Evaluation protocols`; `Evaluation metrics and statistics` | `Diagnostic analyses` | `explicitly defined` | The Methods section now states the primary bar metric and the secondary full-matrix reference clearly enough. The paper-facing risk is role drift: the secondary full-matrix correlation should not sound like the primary ranking quantity. |
| `overlap burden / canonical-correlation overlap` | `mean top-3 overlap burden`; `pairwise squared canonical-correlation overlap`; `neighboring directions overlap more broadly` | Cross-object descriptor that explains why recoverability does not follow response energy alone | `Evaluation protocols` | `Evaluation metrics and statistics` | `partially defined` | The formal descriptor is defined, but the Results phrase `overlap burden` still compresses both the averaging procedure and the top-3 subspace construction. Keep it reading as an executed response descriptor, not an intrinsic material constant. |
| `contrast-selection rule / band-limited directional code` | `shared contrast-selection rule`; `informative window`; `band-limited directional code`; `object-specific bands` | Cross-object frequency-side cash-out showing recurrence without a universal band | `Evaluation protocols` | `Signal processing and feature extraction` | `partially defined` | The executed rule is present, but this family is more descriptor-heavy than the earlier figures. The prose should keep saying that the band is selected by an executed contrast rule, not by post hoc visual choice. |

## Priority Harmonization Ledger

### Highest priority

- `local neighborhood`
  This is the paper's main bridge term, but it spans both centered descriptive
  geometry and radius-based readout locality. Later manuscript cleanup should
  keep those surfaces distinct whenever the paper moves from `Fig. 2-3` into
  `Fig. 4-5`.

- `template matrix H / grouped dictionary D`
  The measured calibration object and the grouped inference object are both
  present in Methods, but the manuscript can still blur them in Results if
  `dictionary`, `template`, and `grouped support` are used without the object
  name.

- `stage-0 grouped-match surface`
  The concept is real and operationally grounded, but the exact phrase is still
  a compressed narrative label. If this term gains more salience, it may need a
  tighter local bridge sentence in Methods or Diagnostic analyses.

### Medium priority

- `directional code`
  Keep it as the paper protagonist, but do not let it pretend to be a single
  Methods variable.

- `overlap burden`
  Keep the descriptor language explicit so the cross-object result does not read
  as though a hidden intrinsic constant had been measured.

- `contrast-selection rule / band-limited directional code`
  Keep the executed rule visible whenever the selected band is discussed in
  paper-facing prose.

### Lower priority

- `graph embedding`
  The current manuscript already treats it as a descriptive complement. It
  becomes risky only if later prose starts making it sound like the paper's
  central reduced-order representation.

## Working Conclusion

The current terminology map is usable and mostly coherent. The strongest method
anchors already exist for the observable fingerprints, centered representations,
guided updates, and locality metrics. The remaining cleanup pressure is mainly
about `quantity discipline`: making sure umbrella terms such as `local
neighborhood`, `directional code`, and `overlap burden` stay connected to the
right analysis surface at the right point in the Results spine.
