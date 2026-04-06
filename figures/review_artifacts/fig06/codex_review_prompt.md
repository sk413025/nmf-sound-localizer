# Codex Paper Asset Review

Review target: `fig06`

Inputs:

- `preview.png`
- `preview_overlay.png`
- `context.json`
- `workflow.json`
- `reviews/visual-reviewer.template.json`
- `reviews/manuscript-fit-reviewer.template.json`
- `reviews/supervisor.template.json`

Use the `paper-asset-review` skill and the canonical Nature Communications requirements file from the context.

Required roles:

1. `visual-reviewer`
   - inspect the actual visual asset first; do not infer content from filenames, manuscript prose, or registry text
   - if any reviewed asset is a PDF, inspect the generated PNG previews for every page before drawing conclusions
   - inspect visual readability, hierarchy, spacing, overload, typography compliance, semantic color consistency, and narration restraint
   - inspect whether the figure or any of its code paths bypass the governed visual grammar; if so, report that as a failure and require removal of the bypass-causing implementation rather than a local exception
   - compare the current target directly against the current manuscript assets for Figs. 2, 3, 5, and 6 when judging family consistency
   - inspect the active generator and composition paths for raw local style bypasses such as ad hoc hex colors, non-token line-width literals, or manual panel styling that does not flow through the governed contract
   - fill every checklist item in `checklist_results` with `pass`, `fail`, or `n/a`, plus a concrete reason and concrete evidence anchors
   - write `reviews/visual-reviewer.json`
2. `manuscript-fit-reviewer`
   - inspect manuscript fit, claim support, caption delegation, and whether the asset belongs in the intended paper role
   - reconcile the figure's visual content with its generator or composition code and its evidence or provenance sources before concluding what the figure means
   - inspect whether any manuscript-facing acceptance claim depends on a governance bypass in the figure, generator, composition path, or review path; if so, fail the review until that bypass is removed and the asset is rewritten through the governed path
   - fill every checklist item in `checklist_results` with `pass`, `fail`, or `n/a`, plus a concrete reason and concrete evidence anchors
   - write `reviews/manuscript-fit-reviewer.json`
3. `supervisor`
   - reject reviews that skip visual inspection, PDF page conversion, or code/provenance reconciliation
   - reject reviews that do not explicitly audit governance-bypass risk and whether bypass-causing implementation was removed
   - reject reviews that leave checklist items without a verdict, reason, and concrete evidence anchors
   - consolidate both role reports
   - write final `review.json`

Asset model reminder:

- `context.json` distinguishes the manuscript-facing review asset from any upstream generator outputs and evidence sources.
- For multi-panel figures, inspect the split top-level panel assets listed in `context.json` to separate panel-local problems from recomposition problems.
- Use `registry_path`, `manuscript_path`, and the figure metadata in `context.json` to locate the generator or composition code path when needed.
- For `data_backed_*` provenance modes, judge the final manuscript asset as the release candidate, but use the upstream evidence references to detect provenance gaps or slide-style recomposition mistakes.
- If the final asset appears to discard or distort the data-backed upstream figure, call that out explicitly in the role reports.
- Apply the branch visual grammar: panel labels about 8 pt, most other figure text within 5–7 pt, restrained internal narration, and stable semantic colors across the paper.
- Apply the governance-bypass rule: if a figure, panel, generator, composition path, or review path bypasses the governed visual grammar or review contract, require removal of the bypass-causing implementation and a rewrite through the governed path before the figure can pass.

Checklist items that must be answered with concrete evidence:
- `C1` Panel hierarchy matches claim importance.
- `C2` Panel a reads as a low-burden support strip rather than a foreign explainer sidebar.
- `C3` Panel c has one scientific job and does not mix multiple semantics.
- `C4` Line weights and stroke grammar are consistent with governed tokens and the surrounding figure family.
- `C5` Typography scale is consistent with branch type rules and nearby main-paper figures.
- `C6` Semantic colors are consistent with the paper-wide visual grammar and do not drift into local ad hoc choices.
- `C7` Whitespace and density are efficient for a Nature Communications main-paper figure.
- `C8` The local manuscript text and legend accurately describe the current visible figure.
- `C9` The manuscript page placement is clean and readable at proof scale.
- `C10` The current figure is visually consistent with Figs. 2, 3, 5, and 6.
- `G1` Governance contract paths were inspected directly against the current figure and code.
- `G2` No active figure path still relies on raw local style bypasses such as ad hoc hex colors or non-token line widths.

Bundle hash:

- `b3c5484184caed007c3bb1c1229dba2e60bf9f36a911eb9dca939928930f0113`
