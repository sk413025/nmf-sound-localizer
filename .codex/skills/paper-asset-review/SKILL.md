---
name: paper-asset-review
description: Use this skill when reviewing paper-facing figures or tables, explaining what a figure or panel is showing, comparing a figure against a critique or claim, or tracing figure meaning from manuscript text to the visual asset, generator, and evidence in this repository.
---

# Paper Asset Review

Use this skill for:

- main-paper figure review
- supplementary or extended-data figure review
- manual figure asset review
- future table review tasks that need manuscript-first visual judgment
- explaining what a figure or panel is actually showing
- checking whether a figure supports the intended manuscript claim
- comparing the current figure against reviewer-style critique or rewrite proposals
- tracing panel meaning from manuscript text to asset, generator, and provenance

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/submission-contract.md`
- `docs/agent-ops/TASK_PACKETS.md`
- [docs/nature-communications/nature-communications-submission-requirements.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/nature-communications-submission-requirements.md)
- [docs/nature-communications/paper-asset-review-workflow.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/paper-asset-review-workflow.md)

Treat this workflow as manuscript-first governance. `figures/` prepares assets and bundles; `scripts/paper/` owns the review entrypoint.

If the user is asking an interpretive question rather than a formal review-gate task, also read:

- [references/figure-claim-backtrace.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/paper-asset-review/references/figure-claim-backtrace.md)

## Command surface

- `python scripts/paper/review_paper_assets.py prepare`
- `python scripts/paper/review_paper_assets.py gate`

## Required roles

Every paper-facing asset review must include these roles:

1. `visual-reviewer`
2. `manuscript-fit-reviewer`
3. `supervisor`

## Workflow

1. Classify the request as either:
   - formal asset review or gate work
   - interpretive figure or panel backtrace
   - figure-vs-claim or figure-vs-critique comparison
2. For formal review or gate work, run `python scripts/paper/review_paper_assets.py prepare`.
3. Open the target bundle under `figures/review_artifacts/<figure_id>/` when one exists.
4. Read `context.json`, `workflow.json`, and `codex_review_prompt.md` when present.
5. Visually inspect the manuscript asset first. If any reviewed asset is a `pdf`, inspect the generated PNG page previews for every page before drawing conclusions.
6. If `context.json` lists split panel assets or upstream generator assets, inspect them before deciding whether a problem is panel-local or caused by recomposition.
7. Inspect the generator or composition code path and the evidence or provenance paths referenced by the bundle context before deciding figure meaning, lineage, or claim support.
8. For interpretive or comparison tasks, answer in this order:
   - what the figure or panel is intended to show
   - what the visual asset actually shows
   - whether manuscript text, asset, generator, and evidence agree
   - which parts are supported, missing, or overstated
9. For formal review tasks, `visual-reviewer` records the completed visual and backtrace checks, then writes `reviews/visual-reviewer.json`.
10. For formal review tasks, `manuscript-fit-reviewer` checks whether the asset supports the intended claim and paper role after reconciling the visual asset with code and evidence, then writes `reviews/manuscript-fit-reviewer.json`.
11. For formal review tasks, `supervisor` consolidates both role reports into `review.json`.
12. Enforce formal review outputs with `python scripts/paper/review_paper_assets.py gate`.

Common trigger phrases:

- "what is this figure or panel showing"
- "what is the logic behind this panel"
- "does this figure support the claim"
- "what gap exists between this figure and this critique"
- "trace this figure from manuscript to generator to evidence"

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
- Do not answer "what this figure means" from caption prose alone when the panel layout, generator, or evidence trail could change the interpretation.
- Do not pass a main-paper figure if the review recommends splitting it or moving it to supplementary.
- Do not pass a figure that still reads like a presentation board rather than a journal figure.
- Do not treat ad hoc local color choices as acceptable if they drift from the paper-wide semantic palette without justification.
- Keep code support thin; use Codex to make the paper-level visual judgment.
