---
name: paper-asset-review
description: Use this skill when reviewing paper-facing figures or tables, explaining what a figure or panel is showing, comparing a figure against a critique or claim, or tracing figure meaning from manuscript text to the visual asset, generator, and evidence in this repository.
---

# Paper Asset Review

Use this skill for:

- main-paper figure review
- supplementary or extended-data figure review
- manual figure asset review
- future table review tasks that need paper-facing asset judgment under manuscript-first branch priorities
- explaining what a figure or panel is actually showing
- checking whether a figure supports the intended manuscript claim
- comparing the current figure against reviewer-style critique or rewrite proposals
- tracing panel meaning from manuscript text to asset, generator, and provenance

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/submission-contract.md`
- `docs/governance/scientific-voice-guide.md`
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- `docs/agent-ops/TASK_PACKETS.md`
- [docs/nature-communications/nature-communications-submission-requirements.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/nature-communications-submission-requirements.md)
- [docs/nature-communications/paper-asset-review-workflow.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/paper-asset-review-workflow.md)

Treat this workflow as paper-facing asset governance inside a manuscript-first branch. `figures/` prepares assets and bundles; `scripts/paper/` owns the review entrypoint.

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
9. If the output includes paper-facing explanation or review-note prose, first identify the prose role:
   - what paper protagonist the asset is serving
   - whether the asset belongs before, at, or after the paper pivot
   - which `Results section job` the asset is serving when the figure affects whole-paper sequence
   - whether the asset strengthens the `discovery cash-out` or dilutes it into a late extension
   - whether the explanation risks making the figure, reference object, or tool sound like the paper's main character
   - whether the asset's current role increases tool-weight relative to discovery
10. If the output includes paper-facing explanation or review-note prose, run a sentence-energy pass before finalizing:
   - reduce noun stacks
   - add causal glue where numbers or contrasts need interpretation
   - normalize diction toward natural scientific English
11. For formal review tasks, `visual-reviewer` records the completed visual and backtrace checks, then writes `reviews/visual-reviewer.json`.
12. For formal review tasks, `manuscript-fit-reviewer` checks whether the asset supports the intended claim and paper role after reconciling the visual asset with code and evidence, then writes `reviews/manuscript-fit-reviewer.json`.
13. For formal review tasks, `supervisor` consolidates both role reports into `review.json`.
14. Enforce formal review outputs with `python scripts/paper/review_paper_assets.py gate`.
15. For any asset revision round, include a concise `visual delta summary` in the round `Delivery evidence` that states what changed in the visible asset and whether the delta closes the targeted review issue.

## Reviewer subset and acceptance surface

Use `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source.
The canonical stack supplements the required formal-review roles here; it does not replace `visual-reviewer`, `manuscript-fit-reviewer`, or `supervisor`.

Default to the minimal reviewer subset that matches the figure risk:

- `figure-science-readability reviewer` for almost every paper-facing figure or table decision
- `cross-disciplinary-readability reviewer` when the deliverable includes interpretive prose, legend advice, or reviewer-facing explanation that may later flow into the paper
- `statistics-evidence reviewer` when the figure may overclaim relative to the supporting evidence
- `sparse-inverse-problem-comparator reviewer` when comparator families, baselines, or panel-to-panel comparison logic could be read as unfair or silently shifting
- `cognitive-load reviewer` or `narrative-flow reviewer` when panel density, panel order, or caption burden makes the figure hard to use in the paper sequence

Acceptance surface for this skill:

- the figure has a clear scientific job in the paper
- panel logic, labels, and emphasis make the intended comparison readable
- evidence support and comparator logic match the manuscript claim
- any split, simplify, revise, or move-to-supplementary recommendation is made explicit
- review notes and interpretive writeups keep the paper protagonist clear and do not let the figure, panel, or supporting tool become the paper's main character
- architecture-sensitive figure reviews name the supported `Results section job`, whether the figure belongs before, at, or after the pivot, whether it strengthens the discovery cash-out, and whether it creates tool overweight
- review notes and interpretive writeups use active voice, simple cause-effect phrasing, low noun-stack density, and natural scientific English when prose is needed
- when a prose problem is identified, the review can name the closest `SV#` exemplar plus sentence-friction type (`noun-stack`, `causal-gap`, `number-without-meaning`, `formal-register`, `static-verb`)

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
