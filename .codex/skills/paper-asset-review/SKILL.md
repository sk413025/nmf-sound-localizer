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

- `START_HERE_AGENT.md`
- `docs/governance/submission-contract.md`
- `docs/agent-ops/TASK_PACKETS.md`
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
4. Visually inspect the manuscript asset first. If any reviewed asset is a `pdf`, inspect the generated PNG page previews for every page before drawing conclusions.
5. If `context.json` lists split panel assets or upstream generator assets, inspect them before deciding whether a problem is panel-local or caused by recomposition.
6. Inspect the generator or composition code path and the evidence or provenance paths referenced by the bundle context before deciding figure meaning, lineage, or claim support.
7. `visual-reviewer` records the completed visual and backtrace checks, then writes `reviews/visual-reviewer.json`.
8. `manuscript-fit-reviewer` checks whether the asset supports the intended claim and paper role after reconciling the visual asset with code and evidence, then writes `reviews/manuscript-fit-reviewer.json`.
9. `supervisor` consolidates both role reports into `review.json`.
10. Enforce with `python scripts/paper/review_paper_assets.py gate`.

The visual review must explicitly judge:

- typography against the branch type scale
- semantic color consistency across figures
- whether explanation has been pushed into the caption instead of oversized internal headers
- whether panel hierarchy matches the claim importance

## Guardrails

- Do not treat a generated bundle as sufficient evidence; the actual Codex multimodal review must be recorded.
- Do not interpret a paper-facing figure from filenames, registry prose, or manuscript text alone.
- Do not skip generator or evidence backtrace for generated or data-backed figures.
- Do not treat a PDF as reviewed until its pages have been converted to image previews and visually inspected.
- Do not skip the manuscript-fit role just because the figure looks readable.
- Do not pass a main-paper figure if the review recommends splitting it or moving it to supplementary.
- Do not pass a figure that still reads like a presentation board rather than a journal figure.
- Do not treat ad hoc local color choices as acceptable if they drift from the paper-wide semantic palette without justification.
- Keep code support thin; use Codex to make the paper-level visual judgment.
