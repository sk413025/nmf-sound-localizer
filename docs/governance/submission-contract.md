# Submission Contract

Use this contract for Nature Communications compliance, figure/table packaging, source-data expectations, and final submission-facing assets.

## Applies to

- figure export and validation rules
- tables and metadata
- submission package requirements
- data and code availability sections
- paper-facing asset review

## Core rules

- The canonical Nature Communications requirements file is the source of truth.
- Legacy local figure notes are redirects, not authority.
- Paper-facing asset review must remain subordinate to manuscript-first branch priorities.
- When local implementation conflicts with current Nature guidance, update the canonical document and implementation together.
- Figure style must follow the branch visual grammar derived from the canonical Nature requirements.
- Typography must use the branch type scale:
  - panel labels about `8 pt` bold
  - most other figure text within `5–7 pt`
- Figure color must follow paper-wide semantic roles rather than ad hoc per-figure choices, unless the figure is encoding a genuinely local scientific variable such as mode identity.
- If a figure, panel, generator, or composition path is found to bypass the governed visual grammar or semantic-color contract, remove the bypass-causing implementation rather than documenting an exception around it. Rebuild that asset through the governed contract path before the figure can pass review.
- In-figure narration must stay restrained; slide-style master titles, oversized internal headers, and colored explanatory text are not acceptable defaults.
- Multi-panel layouts should minimize white space and size panels by claim importance rather than force equal-size symmetry when the content does not warrant it.
- Every multi-panel figure must preserve one final paper-facing composite plus a set of split top-level panel assets and a panel manifest under `figures/output/`.
- Split panel assets are internal reproducibility and review artifacts; the Nature-facing submission asset remains the final composite figure.
- `paper/figures/` is a canonical paper-facing asset surface and the literal manuscript-facing asset surface for final paper figures. Only declared manuscript assets and governed paper-facing sidecars such as `paper/figures/*.layout.json` belong there.
- Generator intermediates belong under `figures/output/` unless a contract explicitly declares them as paper-facing assets.
- For recomposed or geometry-sensitive paper figures, preserve a matching realized-layout sidecar under `paper/figures/*.layout.json`.
- Figures with stacked, densely packed, or crowding-prone layouts require both visual review and a passing quantitative clearance check. Thresholds belong in executable checks or figure-specific layout contracts, not in this file.
- Every paper-facing figure judgment must begin with visual inspection of the actual asset.
- `jpg` and `png` assets must be reviewed directly; `pdf` assets must first be converted page-by-page into PNG previews for visual inspection.
- Generated or data-backed figures are not submission-ready unless the visual asset has been reconciled against both its generator or composition code and its upstream evidence or provenance sources.
- Paper-facing explanatory prose in legends, captions, availability sections, and submission-facing notes inherits the same paper-facing explanation standard as the broader branch model: clear subject, strong verb, explicit consequence, low noun-stack friction, natural scientific English where precision permits, and supporting-actor language that does not let figures or tools replace the paper's discovery as protagonist.

## Required outputs

- submission-ready manuscript assets under `paper/`
- realized manuscript-layout sidecars where the figure workflow requires them
- validated figure outputs and review artifacts
- split panel assets and panel manifests for every multi-panel figure
- required submission sections and metadata
- explicit handling of data and code availability

## Acceptance criteria

- paper asset review workflow has been run when applicable
- manuscript includes the required branch-level submission sections
- figure files and references are consistent with branch policy
- `paper/figures/` contains only declared manuscript assets and governed sidecars
- canonical Nature requirements remain reachable from main entrypoints
- main-paper figures do not rely on oversized internal headers or presentation-board narration
- typography and semantic colors are consistent with the branch visual grammar
- no accepted figure or review path still depends on a bypass around the governed style or review contract
- every multi-panel figure has a matching panel manifest whose panel order agrees with the composite
- realized layout sidecars exist where the figure workflow requires them
- geometry and clearance checks pass where the figure workflow requires them
- figure acceptance does not rely on filenames, registry prose, or manuscript text alone when visual inspection or provenance backtrace would change the interpretation
- paper-facing legends, captions, and availability prose do not regress into caption-choreography, formal-register compression, or unexplained number clusters simply because they sit outside the main manuscript body

## Executable gates

- `make paper-review-assets`
- `make paper-review-gate`
- `make -C figures validate`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/check_figure_layout_clearance.py <paper/figures/*.layout.json>` when the figure workflow declares a clearance audit
- `python scripts/paper/check_governance_links.py`
