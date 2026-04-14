# Paper Asset Review Workflow

This document defines the manuscript-first, Codex-governed workflow for paper-facing figure review in this repository.

## Why this exists

The goal is not merely to generate compliant artwork. The goal is to decide whether a figure is suitable for this manuscript, in this journal context, for the specific claim it is meant to support.

Therefore:

- `figures/` owns generation and preview preparation
- `scripts/paper/` owns paper-facing review entrypoints
- Codex multimodal review is the primary acceptance judge
- AGENTS and local skills define the governance contract
- Paper-facing figure review is visual-first: figure meaning, panel identity, and suitability cannot be inferred from filenames or metadata alone

## Entry points

```bash
# Build or refresh paper-facing assets
make -C figures all

# Prepare review bundles
python scripts/paper/review_paper_assets.py prepare

# After Codex reviews are recorded
python scripts/paper/review_paper_assets.py gate
```

## Required role split

Every paper-facing figure review uses three explicit roles:

1. `visual-reviewer`
   - judges readability, hierarchy, spacing, overload, typography compliance, semantic color consistency, and narration restraint
2. `manuscript-fit-reviewer`
   - judges whether the figure supports the intended claim, belongs in the intended paper role, delegates explanation to the caption instead of overexplaining inside the artwork, and remains consistent with the canonical panel-to-method crosswalk at `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`
3. `supervisor`
   - consolidates both role reports into the final `review.json`

## Review bundle contract

Each bundle under `figures/review_artifacts/<figure_id>/` contains:

- `preview.png`
- `preview_overlay.png`
- `context.json`
- `workflow.json`
- `codex_review_prompt.md`
- `reviews/visual-reviewer.template.json`
- `reviews/manuscript-fit-reviewer.template.json`
- `reviews/supervisor.template.json`

The final accepted artifact is:

- `review.json`

`context.json` is expected to distinguish five asset layers when applicable:

- `evidence source`: results/data files that support the claim
- `generator output`: clean code-produced figure asset
- `split panel assets`: top-level manuscript panels (`a/b/c/...`) stored as internal recomposition assets
- `paper-facing asset`: final paper-facing composite that is actually reviewed
- `manuscript layout sidecar`: realized manuscript geometry such as `paper/figures/*.layout.json` used for overlay and clearance audits

For single-panel manual figures, only the paper-facing asset may exist. For data-backed figures,
the paper-facing asset must remain traceable to the upstream evidence and generator outputs.
For multi-panel figures in this branch, split top-level panel assets are required internal assets even though the journal-facing release target remains the final composite.
When a manuscript layout sidecar exists, it is part of the review context rather than an optional debugging extra.

Review must follow this inspection order:

1. Inspect the paper-facing asset visually.
2. If a manuscript layout sidecar exists, inspect it before deciding whether spacing and geometry are acceptable.
3. If the paper-facing asset or any reviewed upstream figure asset is a `pdf`, inspect PNG page previews for every page before interpreting the asset.
4. Inspect split panel assets or upstream figure assets listed in `context.json`.
5. Inspect the generator or composition code that produced the figure.
6. Inspect the evidence or provenance sources that the figure claims to summarize.
7. For active main-paper figures, inspect the corresponding entries in `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`.
8. Only then decide manuscript fit, lineage, and Nature suitability.

## Acceptance rules

A paper-facing figure is not releasable unless:

- geometry checks pass
- any required layout-sidecar-backed clearance audit passes
- `review.json` exists
- `review.json` declares `skill_used = paper-asset-review`
- all three reviewer roles are recorded
- `overall_verdict = pass`
- the role fit matches the target role
- no split or supplementary move is recommended for a main-paper figure
- the review does not identify typography-band violations, semantic-color drift, or presentation-style narration overload as unresolved blockers
- for active main-paper figures, the review agrees with the canonical crosswalk on panel scientific job, method/supplement anchor, and overlap verdict, or records the mismatch as a blocker

## Required visual rubric

Every `visual-reviewer` pass must explicitly judge:

- whether non-panel text stays within the branch `5–7 pt` band
- whether panel labels stay near the branch `8 pt` bold standard
- whether semantic colors are consistent with the paper-wide visual grammar
- whether the figure uses internal headers or narration that should live in the caption instead
- whether the panel layout minimizes white space and makes the main claim visually dominant
- whether any spacing or crowding risk requires layout-sidecar-backed clearance review
- whether any weakness is intrinsic to a split panel or introduced only by the final recomposition
- whether the spacing judgment agrees with manuscript layout metadata when such metadata exists
- whether the reviewer inspected the actual asset rather than inferring content from filenames or manuscript text
- whether any PDF assets were reviewed through page-by-page PNG previews
- whether the visual story agrees with the generator or composition code and the listed evidence sources

Every `manuscript-fit-reviewer` pass for an active main-paper figure must explicitly judge:

- whether each panel's scientific job still matches the canonical entry in `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`
- whether the panel's method anchor and supplementary anchor remain adequate for the paper-facing claim
- whether any overlap risk recorded in the crosswalk has reappeared in the current paper-facing asset

## Governance principle

If code and governance diverge, governance wins:

- update AGENTS / skills / workflow docs first
- then update code to match

Do not allow a new figure-review code path to bypass these contracts.

If review discovers a governance bypass, do not accept a local patch that merely papers over the symptom.
Remove the bypass-causing implementation and rewrite the affected figure, panel, generator, or composition path so it again flows through the governed style and review contract.
