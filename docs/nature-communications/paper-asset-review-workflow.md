# Paper Asset Review Workflow

This document defines the manuscript-first, Codex-governed workflow for paper-facing figure review in this repository.

## Why this exists

The goal is not merely to generate compliant artwork. The goal is to decide whether a figure is suitable for this manuscript, in this journal context, for the specific claim it is meant to support.

Therefore:

- `figures/` owns generation and preview preparation
- `scripts/paper/` owns manuscript-facing review entrypoints
- Codex multimodal review is the primary acceptance judge
- AGENTS and local skills define the governance contract

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
   - judges readability, hierarchy, spacing, and overload
2. `manuscript-fit-reviewer`
   - judges whether the figure supports the intended claim and belongs in the intended paper role
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

`context.json` is expected to distinguish three asset layers when applicable:

- `evidence source`: results/data files that support the claim
- `generator output`: clean code-produced figure asset
- `manuscript asset`: final paper-facing composite that is actually reviewed

For manual figures, only the manuscript asset may exist. For data-backed figures,
the manuscript asset must remain traceable to the upstream evidence and generator outputs.

## Acceptance rules

A paper-facing figure is not releasable unless:

- geometry checks pass
- `review.json` exists
- `review.json` declares `skill_used = paper-asset-review`
- all three reviewer roles are recorded
- `overall_verdict = pass`
- the role fit matches the target role
- no split or supplementary move is recommended for a main-paper figure

## Governance principle

If code and governance diverge, governance wins:

- update AGENTS / skills / workflow docs first
- then update code to match

Do not allow a new figure-review code path to bypass these contracts.
