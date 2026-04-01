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
   - write `reviews/visual-reviewer.json`
2. `manuscript-fit-reviewer`
   - inspect manuscript fit, claim support, caption delegation, and whether the asset belongs in the intended paper role
   - reconcile the figure's visual content with its generator or composition code and its evidence or provenance sources before concluding what the figure means
   - write `reviews/manuscript-fit-reviewer.json`
3. `supervisor`
   - reject reviews that skip visual inspection, PDF page conversion, or code/provenance reconciliation
   - consolidate both role reports
   - write final `review.json`

Asset model reminder:

- `context.json` distinguishes the manuscript-facing review asset from any upstream generator outputs and evidence sources.
- For multi-panel figures, inspect the split top-level panel assets listed in `context.json` to separate panel-local problems from recomposition problems.
- Use `registry_path`, `manuscript_path`, and the figure metadata in `context.json` to locate the generator or composition code path when needed.
- For `data_backed_*` provenance modes, judge the final manuscript asset as the release candidate, but use the upstream evidence references to detect provenance gaps or slide-style recomposition mistakes.
- If the final asset appears to discard or distort the data-backed upstream figure, call that out explicitly in the role reports.
- Apply the branch visual grammar: panel labels about 8 pt, most other figure text within 5–7 pt, restrained internal narration, and stable semantic colors across the paper.

Bundle hash:

- `e0e274ed00886790871cdcddee9654bc94b9b9a5a100470650b5606a7c561c9a`
