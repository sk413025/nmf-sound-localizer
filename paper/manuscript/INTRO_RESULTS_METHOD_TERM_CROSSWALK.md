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
- Keep one canonical head term across `Introduction`, `Results`, legends,
  `Methods`, and `Supplementary Methods`.
- Reopen each concept family in `Methods` / `Supplementary Methods` with that
  same canonical head term before attaching notation, equations, or exact score
  names.
- Concentrate each required surface bridge at the first shift where it becomes
  necessary, then refer back to that landing site rather than re-explaining the
  same mapping in later paragraphs.
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
- `pre-update grouped match`
- `shared-response overlap across objects`
- `object-specific informative band`

## Current Strict-Pass Families

The following concept families now pass the branch's strict lexical
no-translation test: one canonical head term appears across main text,
legends, `Methods`, and `Supplementary Methods`, while formal labels remain
second-layer precision rather than competing reader-facing names.

- `directional code`
- `fingerprint`
- `local neighborhood`
- `local separability`
- `local support`
- `exact support`
- `guided solver`
- `pre-update grouped match`
- `shared-response overlap across objects`
- `object-specific informative band`

## Reduction Ledger

| Term family | Canonical paper-facing term | Formal term(s) allowed only in Methods / Supplementary | Terms to retire from Introduction / Results | First-use bridge rule |
|---|---|---|---|---|
| Recurring directional structure across angles | `directional code` | centered `\|H\|` descriptors; grouped support statistics; routed locality metrics | `local directional organization`; `directional encoding` as a protagonist synonym | First use should say the code is carried by `fingerprints` or by the `single-point readout`, not leave `code` floating as an abstract label. |
| Single-point observable | `fingerprint` | standardized log-power fingerprint; `\tilde y`; calibrated dictionary `H` | none by default | First use should connect `fingerprint` to the single-point LDV readout. |
| Calibration-side nearby-angle geometry | `local neighborhood` | centered-neighborhood similarity; centered `\|H\|` correlation matrix; correlation-decay width | `measured neighborhood` as a generic catch-all; `finite positive local neighborhood` when the simpler phrase works | When the paragraph is on calibration-side geometry, say `local neighborhood` and make clear that nearby angles remain related on the calibrated fingerprint surface. |
| Readout difficulty under nearby-angle competition | `local separability` | local-overlap surrogate; grouped stage-0 summaries; exact-versus-local support definitions | `local ambiguity` as a peer term unless the sentence is explicitly contrasting ambiguity with exact commitment | First use should say that nearby candidates become jointly plausible before a final angle is chosen. |
| Readout-side nearby-angle retention | `local support` | mass-within-radius; within-10° success; within-15° mass; locality metric | `operative 15° neighborhood`; `radius-based neighborhood metric`; `local-band metric` in main text | When the paragraph is on decoder behavior, stop saying only `neighborhood`; say `local support` if the active quantity is retained nearby-angle mass. |
| Immediate exact-angle concentration | `exact support` | exact-match success; radius-zero case; Top-1 exact-support statistic | `exact commitment` when it is being used as a metric label rather than a consequence verb | Pair it with `local support` when both are compared. |
| Active learned decoder family | `guided solver` | routed update; direction-level routing score; grouped dictionary recursion; Gumbel gate | `routing` or `routed update` as a family name; `physics-guided structured readout` as the default family label in Results | In Results, use `guided solver` for the family, keep at most one formal bridge per `Fig. 4` / `Fig. 5` subsection, and reserve routing internals for Fig. 4 mechanism sentences or Methods. |
| Cross-object descriptor for recoverability | `shared-response overlap across objects` | mean top-3 overlap burden; pairwise squared canonical correlations between top-3 centered-`\|H\|` subspaces | `overlap burden` as the first paper-facing name; `cross-object subspace overlap` once the plainer phrase has landed | First use should explain that the descriptor summarizes how strongly each object's leading directional-response subspace overlaps with those of the other objects. |
| Cross-object frequency-side descriptor | `object-specific informative band` | contrast-selection rule; smoothed contrast profile; band-averaged directional code construction | `contrast-selection rule` as a first main-text label; `band-limited directional code` as the default main-text phrase | First use should say that each object has its own informative band, then point to Methods for how that band is selected. |
| Descriptive compactness language | `compact` | compactness summaries; centered-magnitude energy; cumulative singular-value energy; effective rank | `compact structure` as a parallel head term; `centered representation family`; `compactness analysis surface` in main text | Keep `compact` as the descriptive modifier in prose and let `compactness` name the formal summary only when the statistic itself is the point. |
| Descriptive coherence language | `locally ordered` | graph view; centered-neighborhood graph; spectral embedding; graph Laplacian | `neighborhood coherence` as a parallel reader-facing label; `positive centered-neighborhood graph` and `spectral embedding` as the main sentence subject in Results | Keep `locally ordered` as the descriptive claim; when the 2D visualization itself matters, reopen it as a `graph view` and reserve `graph embedding` for the formal construction. |
| Calibration vs inference objects | `calibrated fingerprints` in paper-facing prose; `calibrated dictionary` when the matrix `H` is active | calibrated dictionary `H`; grouped dictionary `D`; grouped atom set | loose alternation among `dictionary`, `template`, `prototype`, `grouped templates`, and `references` without object name | In prose, prefer `calibrated fingerprints`; when the formal object is active, name it directly as `calibrated dictionary H`, and keep `grouped dictionary D` for the refined readout object built from the same calibrated fingerprint space. |
| Matched speech-versus-calibration summary surface | `angle-conditioned summaries` | centered summary matrix `M^{(c)}`; mirrored compactness curve; split-triangle similarity matrix | `centered summary surface` or `angle-conditioned centered summary surface` as a parallel reader-facing name | In Results, use `angle-conditioned summaries` for the matched speech-versus-calibration surface and reserve `M^{(c)}` plus the exact summary constructions for Methods and Supplementary. |
| Pre-update diagnostic under speech | `pre-update grouped match` | `ungated stage-0 grouped-match surface`; `g_0^{(grp)}` | `stage-0 grouped-match surface` as the default reader-facing phrase | First use should describe it as the grouped match before any routing or refitting acts, then point to Supplementary Methods 3 for the formal label. |
| Guided-solver routing mechanism | `direction-level routing score` | atom-level routing score `s_t^{(\mathrm{atom})}[e,m]`; direction-level score `s_t[e]`; Gumbel gate | `routing logits`; `expert score` as a competing reader-facing name when the formal score itself is the point | Use this name only when the guided solver's formal routing mechanism is the sentence subject; otherwise keep the paper-facing focus on the guided solver and local support. |
| Final family alignment descriptor | `local support aligned with the measured geometry` | local-band agreement score; anglewise profile-agreement factor; whole-map correlation | `local-band agreement score`; `anglewise profile-agreement factor`; `whole-map correlation` as peer main-text ranking terms | In Results, describe the ranking consequence in words and reserve the exact score construction for Methods and Supplementary. |

## Priority Harmonization Ledger

### Already Harmonized

- `directional code`
  Landed as the shared protagonist term. Keep it as the paper-facing name and
  let `compact` / `locally ordered` remain descriptive modifiers rather than
  competing head terms.

- `fingerprint`
  Landed as the shared observable name. Keep `standardized fingerprint` and
  `single-point spectral fingerprint` as descriptive refinements, not
  alternative head terms.

- `local neighborhood`
  Landed as the shared calibration-side geometry term. Keep width, matrix, and
  profile language as formal second-layer definitions of the same local
  neighborhood.

- `local separability`
  Landed as the shared paper-facing name for nearby-angle competition. Keep it
  paired with `exact support` and `local support` rather than reopening `local
  ambiguity` as a parallel head term.

- `local support`
  Landed as the shared decoder-side retention term. Keep radius-based or
  within-15° constructions as formal second-layer definitions of local support.

- `exact support`
  Landed as the shared exact-angle concentration term. Keep radius-zero or
  exact-match constructions as formal second-layer definitions of exact
  support.

- `guided solver`
  Landed as the shared decoder-family name. Keep routing and grouped-update
  language as formal mechanism descriptions rather than competing family names.

- `pre-update grouped match`
  Landed as the shared reader-facing bridge for the stage-0 grouped diagnostic.
  Keep `g_0^{(grp)}` and the ungated grouped surface as formal second-layer
  labels only.

- `shared-response overlap across objects`
  Landed as the shared paper-facing cross-object descriptor. Keep `overlap
  burden` and squared-canonical-correlation language in formal surfaces only.

- `object-specific informative band`
  Landed as the shared paper-facing frequency-side descriptor. Keep
  `contrast-selection rule` and band-construction language in formal surfaces
  only.

### Still Secondary

- `compact`
  Keep as a descriptive modifier that supports the directional-code and
  local-neighborhood claims. Use `compactness` only when the formal summary
  statistic itself is the point.

- `locally ordered`
  Keep as a descriptive modifier that supports the local-neighborhood claim.
  Use `graph view` for the visual complement and reserve `graph embedding` for
  the formal construction.

- `calibrated fingerprints` / `calibrated dictionary`
  Keep `calibrated fingerprints` as the paper-facing observable family and
  reopen the formal object as `calibrated dictionary H` only when the matrix is
  active. Keep `grouped dictionary D` as the refined readout object built from
  the same calibrated fingerprint space.

- `angle-conditioned summaries`
  Keep as the matched speech-versus-calibration summary surface for Fig. 3.
  Reopen the formal construction as the centered summary matrix `M^{(c)}` only
  when the summary surface itself is being defined.

- `direction-level routing score`
  Keep only as the formal guided-solver mechanism label. It should support the
  guided-solver story, not become a second reader-facing decoder family name.

- `graph embedding`
  Keep only as the formal construction behind the graph view of the local
  neighborhood. It should not become a second protagonist or second naming
  spine.

- `whole-map correlation`
  Keep only as a secondary Methods-side reference, not as a main-text ranking
  label.

## Working Conclusion

The manuscript does not need more names. It needs fewer paper-facing names,
one stable landing site for each bridge, and clearer division of labor between
paper-facing intuition and formal Methods labels. The canonical set above
should carry `Introduction` and `Results`; the retired terms should remain
available only where the formal construction itself must be stated. In
practice, `Fig. 4-5` should read in paper-facing terms as `admissibility`,
`local support`, and `family consequence`, with decoder internals appearing
only as bounded formal bridges and not as a second narrative spine.
