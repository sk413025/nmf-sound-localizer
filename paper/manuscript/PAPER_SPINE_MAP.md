# Whole-Paper Spine Map and Architecture Audit

This document is the active whole-paper spine map for the main manuscript. Its
job is narrower than the figure-method crosswalk: it aligns the paper-level
`old-world belief -> new-world belief` shift, the Results section jobs, the
paper pivot, and the current rewrite priorities on one manuscript-facing
surface.

Use it for `cross-section` or `whole-manuscript` planning rounds. Do not use it
as a second figure-method ledger; panel-level traceability remains canonical in
[`FIGURE_METHOD_CROSSWALK.md`](./FIGURE_METHOD_CROSSWALK.md).

## Current Round Status

- `Architecture scope:` `cross-section`
- `Current use:` whole-paper operating-model lock, cross-section block rewrites, and closeout planning
- `Not yet claiming:` whole-paper architecture fully landed
- `Why:` the paper spine is explicit, but the active round still needs to verify
  that `Fig. 4-6`, main Methods, and Supplementary Methods `2-5` now share one
  reader contract instead of splitting between discovery-first and
  framework-stabilization-first prose

## Whole-Paper Spine Map

- `Old-world belief`
  - sound direction must be designed into arrays or specialized directional
    hardware; passive objects are mostly nuisance structure around the sensor
- `New-world belief`
  - ordinary passive objects can themselves carry a recurring, locally ordered
    directional code in single-point vibrometry, and readout succeeds when it
    preserves that measured neighborhood rather than collapsing it too early
- `Paper protagonist`
  - the recurring local directional code carried by passive structure
- `Supporting actors`
  - matched calibration
  - the acrylic reference object
  - the guided solver and its comparator families
  - the five-object passive-structure archetype set
- `Results section jobs`
  - `Fig. 1`: phenomenon opener at one vibrometric point
  - `Fig. 2`: compact finite neighborhood discovery
  - `Fig. 3`: speech broadens the code into structured local ambiguity
  - `Fig. 4`: admissible first-step contraction of that broadened support
  - `Fig. 5`: final decoder-family consequence under the same locality metric
  - `Fig. 6`: recurrence across passive-object archetypes and bounded
    cross-object cash-out
- `Pivot section`
  - `Fig. 4`
- `Pivot sentence`
  - "The scientific point is therefore not the update machinery by itself. It
    is that a usable readout must preserve the measured local neighborhood
    before subtraction is allowed to sharpen it."
- `Discovery cash-out section`
  - `Fig. 6`
- `Tool role`
  - matched calibration and the guided solver reveal, preserve, and test the
    measured local geometry; they are supporting devices rather than the paper's
    main contribution
- `Reference-object role`
  - the acrylic plate is the clearest first view of the phenomenon, not the
    paper's end point
- `Discovery-vs-tool weight budget`
  - target approximately `70:30` in favor of the discovery and its governing
    principle over calibration/decoder mechanics
- `Operating-model rule`
  - main text changes the reader's model of the system, main Methods formalizes
    already-stable actors, and Supplementary Methods proves those actors without
    renegotiating their identity
- `Second-layer discovery`
  - passive structure is part of the sensing substrate rather than only a
    nuisance surrounding the sensor
- `Broader-significance status`
  - `second-layer earned`
- `Broader-implication trunk`
  - some directional front-end information can already reside in passive
    structures present in a scene
- `Downstream-consequence branch`
  - not promoted in the active spine; keep downstream branches out unless they
    remain clearly subordinate to the trunk
- `Optional leaf consequence`
  - none in the active spine
- `Redundancy / breathing risks`
  - `Fig. 2f` can sound like a second latent-space thesis if left unbounded
  - `Fig. 4` carries strong method linkage and can become solver-heavy if its
    mechanism panels are not kept subordinate to admissibility
  - `Fig. 5f` now has an explicit primary-versus-secondary metric hierarchy,
    but that distinction must remain visible in prose and legend language
  - `Fig. 6a` and `Fig. 6d` are now explicitly anchored as executed response
    descriptors, but they remain more descriptor-heavy than the central acrylic
    chapters
- `Worldview-shift sentence`
  - directional sensing need not be designed only into arrays and specialized
    sensors; passive structure can already carry a recurring local directional
    code that functions as part of the sensing substrate

## Current Architecture Evidence Map

### Front door

- `Manuscript title`
  - "A recurring locally ordered directional code emerges across passive objects in
    single-point vibrometry"
- `Metadata title`
  - "A recurring locally ordered directional code emerges across passive objects
    in single-point vibrometry"
- `Verdict`
  - unified around the recurring locally ordered directional code as
    protagonist; matched calibration remains a supporting actor rather than the
    title's main subject

### Abstract trunk

- `Current trunk sentence`
  - "Directional sensing therefore need not be designed only into dedicated
    hardware: some directional front-end information can already reside in
    passive structures present in a scene."
- `Verdict`
  - strong and aligned with the current `second-layer earned` status

### Introduction preload

- `Current preload sentence`
  - "Our aim is therefore to establish a recurring directional code across
    ordinary passive objects and to identify the condition that governs its
    recoverability: readout must preserve the measured local neighborhood
    before it sharpens it."
- `Verdict`
  - aligned with the abstract trunk and the current broader-significance status

### Pivot

- `Current pivot sentence`
  - "The scientific point is therefore not the update machinery by itself. It
    is that a usable readout must preserve the measured local neighborhood
    before subtraction is allowed to sharpen it."
- `Verdict`
  - strong and well aligned with the `Fig. 3 -> Fig. 4 -> Fig. 5` bridge

### Discovery cash-out

- `Current Results cash-out sentence`
  - "What recurs across objects is not one universal spectrum, but one
    recurring directional code expressed through different passive structures.
    That cross-object recurrence is where the paper's broader inference becomes
    unavoidable: passive structure is part of the sensing substrate, not merely
    surrounding hardware."
- `Verdict`
  - good Results-side cash-out; this is where the paper-level discovery becomes
    unavoidable before Discussion restates its broader meaning

### Discussion opening

- `Current worldview-shift sentence`
  - "Directional sensing need not be designed only into arrays and specialized
    sensors. Across the tested passive-object archetypes, one fixed vibrometric
    point repeatedly records a recurring directional code with a finite local
    neighborhood after matched calibration. Passive structure is therefore part
    of the sensing front end, not merely a nuisance surrounding the sensor."
- `Verdict`
  - strong and claim-forward; it lands the worldview shift cleanly

## Current Whole-Paper Verdict

### What is already working

- the Results spine is coherent and monotonic:
  `Fig. 1 -> Fig. 2 -> Fig. 3 -> Fig. 4 -> Fig. 5 -> Fig. 6`
- the protagonist is stable:
  the paper is about the recurring local directional code, not about the solver
- the pivot is legible:
  `Fig. 4` now reads as the admissibility chapter rather than a generic solver
  tour
- the paper already supports a `second-layer earned` broader-significance
  status, and the abstract, Introduction, Results cash-out, and Discussion all
  point toward the same trunk

### What is still uneven

- the front door is now unified across `manuscript.md` and `metadata.yaml`, but
  the unified discovery-first framing still needs to stay stable as later
  figure and prose rounds reopen local sections
- the main-text `Fig. 4 -> Fig. 5 -> Fig. 6` arc now lands the intended
  discovery-first weighting, but it still needs one verification pass to make
  sure the chapter now reads as one continuous readout-to-cash-out block rather
  than as local section patches
- the method spine is now explicit at its previously weakest panels:
  `Fig. 5f`, `Fig. 6a`, and `Fig. 6d` are no longer weakly defined, but they
  still require discipline so their descriptor language does not drift away
  from the underlying constructions
- the largest remaining readability risk is now split reader contract rather
  than a missing Results chapter: the supplement still needs to read as formal
  backing rather than as repeated concept stabilization

### What does not currently justify a large redraw

- no figure is presently a whole-paper architectural blocker on its own
- the immediate problem is traceability unevenness and front-door alignment, not
  a missing Results chapter
- further panel redesign should only begin after the title/front-door and
  weakest methods-anchor surfaces are either tightened or proven insufficient

## Priority Rewrite Packets

### Packet A — Front-door unification

- `Why`
  - completed in the current round; the front door now uses one protagonist and
    one title framing across manuscript and metadata
- `Target surfaces`
  - `paper/manuscript/manuscript.md`
  - `paper/manuscript/metadata.yaml`
- `Decision rule`
  - the final title must keep the recurring locally ordered directional code as
    protagonist and keep matched calibration as a supporting actor rather than
    the title's main subject
- `Acceptance`
  - one unified title
  - unchanged `second-layer earned` trunk across abstract, Introduction, Results
    cash-out, and Discussion opening

### Packet B — Methods-anchor tightening

- `Why`
  - active in the current round; the next task is to rewrite the weakest
    methods-anchor surfaces as one aligned formal bridge instead of as separate
    local fixes
- `Target surfaces`
  - `Fig. 5f`
  - `Fig. 6a`
  - `Fig. 6d`
  - secondary check on `Fig. 2f` and `Fig. 4c`
- `Decision rule`
  - prefer prose / Methods / Supplementary anchoring first; do not open panel
    redesign unless traceability remains weak after the anchor is tightened
- `Acceptance`
  - each target panel can be explained in one sentence that names its primary
    quantity and its method/supplementary construction without relying on figure
    intuition alone

### Packet C — Fig. 4 stabilization

- `Why`
  - `Fig. 4` is now the pivot chapter and therefore carries disproportionate
    architecture risk if it drifts solver-heavy or visually overpacked
- `Target surfaces`
  - active `Fig. 4` asset
  - local Results paragraph
  - local Methods / evaluation wording
- `Decision rule`
  - keep `Fig. 4` only if it still reads as
    `admissible contraction of broadened local support` rather than
    `guided-solver internals`
- `Acceptance`
  - `a-c` remain a bounded admissibility sentence
  - `d-f` remain robustness-of-contraction rather than a second discovery block
  - no new family-comparison or passive-object-recurrence role leaks into
    `Fig. 4`

### Packet D — Architecture closeout packet

- `Why`
  - after Packets A-C, the branch still needs one explicit verdict on whether
    whole-paper architecture has actually landed
- `Target surfaces`
  - this spine map
  - `FIGURE_METHOD_CROSSWALK.md`
  - manuscript front door and Discussion opening
- `Decision rule`
  - only report whole-paper architecture landed if front door, pivot, cash-out,
    and weakest method-anchor panels are all aligned at once
- `Acceptance`
  - updated architecture evidence map
  - narrowed list of remaining panel redesign candidates
  - explicit `landed / not yet landed` verdict

## Immediate Next-Step Recommendation

Start with **Packet C**, not more figure expansion.

- `Packet B` has already removed the clearest weak-link panels from the
  manuscript-side method spine without reopening the Results architecture.
- `Packet C` should now decide whether the active `Fig. 4` is genuinely stable
  as the admissibility pivot or still carries avoidable visual or narrative
  overweight.
