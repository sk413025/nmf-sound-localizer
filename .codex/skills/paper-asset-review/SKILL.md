---
name: paper-asset-review
description: Use this skill when reviewing paper-facing figures or tables with Codex multimodal judgment, multi-agent role separation, and manuscript-first acceptance criteria in this repository.
---

# Paper Asset Review

Use this skill for:

- main-paper figure review
- supplementary or extended-data figure review
- manual figure asset review
- future table review tasks that need manuscript-first visual judgment

## Required first step

Open and follow:

- [docs/nature-communications/nature-communications-submission-requirements.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/nature-communications-submission-requirements.md)
- [docs/nature-communications/paper-asset-review-workflow.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/paper-asset-review-workflow.md)

Treat this workflow as manuscript-first governance. `figures/` prepares assets and bundles; `scripts/paper/` owns the review entrypoint.

## Command surface

- `python scripts/paper/review_paper_assets.py prepare`
- `python scripts/paper/review_paper_assets.py gate`

## Required roles

Every paper-facing asset review must include these roles:

1. `visual-reviewer`
2. `manuscript-fit-reviewer`
3. `supervisor`

## Workflow

1. Run `python scripts/paper/review_paper_assets.py prepare`.
2. Open the target bundle under `figures/review_artifacts/<figure_id>/`.
3. Read `context.json`, `workflow.json`, and `codex_review_prompt.md`.
4. `visual-reviewer` inspects `preview.png` and `preview_overlay.png`, then writes `reviews/visual-reviewer.json`.
5. `manuscript-fit-reviewer` checks whether the asset supports the intended claim and paper role, then writes `reviews/manuscript-fit-reviewer.json`.
6. `supervisor` consolidates both role reports into `review.json`.
7. Enforce with `python scripts/paper/review_paper_assets.py gate`.

## Guardrails

- Do not treat a generated bundle as sufficient evidence; the actual Codex multimodal review must be recorded.
- Do not skip the manuscript-fit role just because the figure looks readable.
- Do not pass a main-paper figure if the review recommends splitting it or moving it to supplementary.
- Keep code support thin; use Codex to make the paper-level visual judgment.
