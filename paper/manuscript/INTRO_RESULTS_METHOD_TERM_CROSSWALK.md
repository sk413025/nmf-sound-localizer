# Introduction-Results-to-Methods Term Crosswalk

This document is the canonical terminology-reduction ledger for
Methods-relevant language in the `Introduction` and `Results` sections of the
main manuscript.

Its job is narrower than a glossary and narrower than a Methods rewrite. It
decides which paper-facing terms should remain in `Introduction` / `Results`,
which formal labels belong only in `Methods` or `Supplementary Methods`, and
which near-synonyms should be retired from the main text so cross-disciplinary
readers do not have to learn a second internal vocabulary before they can
follow the science.

It is not a second Methods section, a second naming contract, or a second
governance vocabulary. It exists to keep the `front-door prose`, the `Results
spine`, and the `Methods spine` on one shared term map.

## Scope And Exclusions

- Scope: terms in `Introduction` and `Results` that imply a representation,
  statistic, inference object, protocol, or evaluation quantity in `Methods`.
- Scope: paper-facing terms that bridge physical intuition to a formal Methods
  surface.
- Exclusion: generic content nouns such as `speech`, `white noise`, `object`,
  `angle`, or `LDV` unless they appear as part of a methods-defined construct.
- Exclusion: figure-only labels that do not carry an independent Methods
  meaning.

## Reduction Policy

- Keep one default paper-facing term per concept in `Introduction` / `Results`.
- Keep formal labels, symbols, and internal surface names in `Methods` /
  `Supplementary Methods` unless the formal label itself is the scientific point
  of the sentence.
- If a paper-facing umbrella term spans more than one formal surface, keep the
  umbrella term but require a short bridge that tells the reader which surface
  is active in that paragraph.
- Do not create a second paper-facing synonym once a canonical term is chosen.

## Canonical Main-Text Vocabulary

- `directional code`
- `fingerprint`
- `local neighborhood`
- `local separability`
- `local support`
- `exact support`
- `guided solver`
- `cross-object subspace overlap`
- `object-specific informative band`

## Reduction Ledger

| Term family | Canonical paper-facing term | Formal term(s) allowed only in Methods / Supplementary | Terms to retire from Introduction / Results | First-use bridge rule |
|---|---|---|---|---|
| Recurring directional structure across angles | `directional code` | centered `\|H\|` descriptors; grouped support statistics; routed locality metrics | `local directional organization`; `directional encoding` as a protagonist synonym | First use should say the code is carried by `fingerprints` or by the `single-point readout`, not leave `code` floating as an abstract label. |
| Single-point observable | `fingerprint` | standardized log-power fingerprint; `\tilde y`; prototype matrix `H` | none by default | First use should connect `fingerprint` to the single-point LDV readout. |
| Calibration-side nearby-angle geometry | `local neighborhood` | centered-neighborhood similarity; centered `\|H\|` correlation matrix; correlation-decay width | `measured neighborhood` as a generic catch-all; `finite positive local neighborhood` when the simpler phrase works | When the paragraph is on calibration-side geometry, say `local neighborhood` and make clear that nearby angles remain related on the calibrated fingerprint surface. |
| Readout difficulty under nearby-angle competition | `local separability` | local-overlap surrogate; grouped stage-0 summaries; neighborhood-tolerant success definitions | `local ambiguity` as a peer term unless the sentence is explicitly contrasting ambiguity with exact commitment | First use should say that nearby candidates become jointly plausible before a final angle is chosen. |
| Readout-side nearby-angle retention | `local support` | mass-within-radius; within-10° success; within-15° mass; locality metric | `operative 15° neighborhood`; `radius-based neighborhood metric`; `local-band metric` in main text | When the paragraph is on decoder behavior, stop saying only `neighborhood`; say `local support` if the active quantity is retained nearby-angle mass. |
| Immediate exact-angle concentration | `exact support` | exact-match success; radius-zero case; Top-1 exact first choice | `exact commitment` when it is being used as a metric label rather than a consequence verb | Pair it with `local support` when both are compared. |
| Active learned decoder family | `guided solver` | routed update; direction-level routing score; grouped dictionary recursion; Gumbel gate | `routing` or `routed update` as a family name; `physics-guided structured readout` as the default family label in Results | In Results, use `guided solver` for the family and reserve routing internals for Fig. 4 mechanism sentences or Methods. |
| Cross-object descriptor for recoverability | `cross-object subspace overlap` | mean top-3 overlap burden; pairwise squared canonical correlations between top-3 centered-`\|H\|` subspaces | `overlap burden` as the first paper-facing name | First use should explain that the descriptor summarizes how strongly each object's leading directional-response subspace overlaps with those of the other objects. |
| Cross-object frequency-side descriptor | `object-specific informative band` | contrast-selection rule; smoothed contrast profile; band-averaged directional code construction | `contrast-selection rule` as a first main-text label; `band-limited directional code` as the default main-text phrase | First use should say that each object has its own informative band, then point to Methods for how that band is selected. |
| Descriptive compactness language | `compact` or `compact structure` | centered-magnitude energy; cumulative singular-value energy; effective rank | `centered representation family`; `compactness analysis surface` in main text | Keep the sentence on what the structure does before naming the formal compactness statistic. |
| Descriptive coherence language | `locally ordered` or `neighborhood coherence` | centered-neighborhood graph; spectral embedding; graph Laplacian | `positive centered-neighborhood graph` and `spectral embedding` as the main sentence subject in Results | Keep graph language descriptive and secondary to the main local-neighborhood claim. |
| Calibration vs inference objects | `calibrated fingerprints` or `calibrated dictionary` when needed | prototype matrix `H`; grouped dictionary `D`; grouped atom set | loose alternation among `dictionary`, `template`, `grouped templates`, and `references` without object name | When both objects are in play, name them explicitly as `calibrated dictionary H` and `grouped dictionary D` in formal surfaces, but keep main-text prose on what they do rather than on the letters themselves. |
| Pre-update diagnostic under speech | `pre-update grouped match` | `ungated stage-0 grouped-match surface`; `g_0^{(grp)}` | `stage-0 grouped-match surface` as the default reader-facing phrase | First use should describe it as the grouped match before any routing or refitting acts, then point to Supplementary Methods 3 for the formal label. |
| Final family alignment descriptor | `local support aligned with the measured geometry` | local-band agreement score; anglewise profile-agreement factor; full-matrix correlation | `local-band agreement score`; `anglewise profile-agreement factor`; `full-matrix correlation` as peer main-text ranking terms | In Results, describe the ranking consequence in words and reserve the exact score construction for Methods and Supplementary. |

## Priority Harmonization Ledger

### Highest priority

- `directional code`
  Keep as protagonist, but never let it stand in for one formal matrix, score,
  or decoder surface.

- `local neighborhood`
  Keep for calibration-side geometry. When the paragraph has moved to decoder
  behavior, switch to `local support` rather than overloading `neighborhood`.

- `cross-object subspace overlap`
  Promote this as the paper-facing descriptor and demote `overlap burden` to
  Methods / Supplementary language.

- `pre-update grouped match`
  Use as the reader-facing bridge for the formal stage-0 grouped diagnostic.

- `object-specific informative band`
  Promote this as the paper-facing descriptor and demote `contrast-selection
  rule` and `band-limited directional code` to formal surfaces.

### Lower priority

- `graph embedding`
  Keep only as a descriptive complement.

- `full-matrix correlation`
  Keep only as a secondary Methods-side reference, not as a main-text ranking
  label.

## Working Conclusion

The manuscript does not need more names. It needs fewer paper-facing names and
clearer division of labor between paper-facing intuition and formal Methods
labels. The canonical set above should carry `Introduction` and `Results`; the
retired terms should remain available only where the formal construction itself
must be stated.
