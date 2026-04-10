---
name: experiment-results
description: Use this skill for interpreting executed runs, identifying which computed factors or metrics matter most, discovering which quantities can be computed now from committed artifacts, comparing candidate explanatory variables, or checking experiment provenance and reproducibility in this repository.
---

# Experiment Results

Use this skill for:

- executed run interpretation
- results-commit analysis
- provenance-backed manuscript support
- failure analysis grounded in committed artifacts
- factor-importance analysis grounded in executed bundles
- metric discovery from existing artifacts
- cross-material or cross-run geometry audits
- scorecards that compare candidate explanatory variables

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/experiment-contract.md`
- `docs/governance/scientific-voice-guide.md`
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- `docs/agent-ops/TASK_PACKETS.md`

If the task is about "what matters most", "which factors can we compute now", "candidate universal equation", or "cross-material geometry", also read:

- [references/cross-material-geometry-audit.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/experiment-results/references/cross-material-geometry-audit.md)

## Workflow

1. Classify the task as:
   - run interpretation
   - reproducibility or provenance check
   - factor or metric audit
   - figure-support analysis for manuscript use
2. Start from committed artifacts under `results/<run_name>/`.
3. Extract numbers, logs, provenance inputs, and failure signals.
4. If multiple metric definitions are possible, freeze the representation before comparing candidates. State whether the analysis uses raw `H`, `|H|`, centered magnitude, projected low-rank coordinates, or another representation.
5. If a run artifact is being used as a paper-facing figure or as support for a paper-facing figure, inspect the visual asset first. For `pdf` artifacts, convert every page to PNG previews before review.
6. When a figure interpretation is involved, inspect the generator or composition code and reconcile it with the upstream run artifacts before writing conclusions.
7. For factor or metric audits, separate outputs into:
   - stable findings
   - rejected or weak candidates
   - current best candidate mechanism or equation form
   - missing evidence or validation still needed
8. If the output will be promoted into paper-facing explanation such as manuscript, legend, rebuttal, or figure-support prose, also identify the paper-facing narrative role:
   - what old-world belief this result helps replace
   - what protagonist it is supporting
   - whether it is evidence for the pivot, the governing principle, the broader implication, or the discovery cash-out
   - which `Results section job` this result is supporting when it is being promoted into manuscript architecture
   - how the result stays subordinate to discovery rather than becoming a tool-centered detour
   - whether using this result risks repeated explanation or tool overweight instead of advancing the paper spine
9. If the output will be promoted into paper-facing explanation such as manuscript, legend, rebuttal, or figure-support prose, run a sentence-energy pass:
   - name the result
   - state the consequence
   - keep one main causal move per sentence
   - replace compressed formal diction with natural scientific English where precision is unchanged
10. When you create a new analysis, write it as a reproducible run bundle under `results/<run_name>/` with script, command, and machine-readable outputs.
11. Write analysis in the language required by the experiment contract.
12. Escalate if the requested claim outruns the available evidence.

## Reviewer subset and acceptance surface

Use `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source when results analysis is feeding manuscript claims or figure interpretation.
Default to the minimal reviewer subset that matches the analysis risk:

- `cross-disciplinary-readability reviewer` when the result summary itself is intended to be paper-facing explanation rather than only an internal factor audit
- `statistics-evidence reviewer` for nearly all manuscript-supporting results summaries, audits, and scorecards
- `physical-mechanism reviewer` when the analysis is being used to justify mechanism language rather than only descriptive performance language
- `sparse-inverse-problem-comparator reviewer` when the result compares solver families, baselines, or ablations whose fairness depends on setup choices
- `acoustics-doa reviewer` when conclusions depend on geometry, acoustics framing, or DOA plausibility

Acceptance surface for this skill:

- findings are tied to committed artifacts, fixed metric definitions, and named provenance paths
- descriptive findings, candidate mechanism, and unsupported interpretation remain clearly separated
- comparator fairness and physical plausibility risks are surfaced when they matter
- paper-facing follow-up is routed onward when the output now requires prose or figure revision rather than more analysis
- paper-facing follow-up does not silently drift into method-as-protagonist language when the analysis is only supporting evidence for the discovery
- architecture-sensitive paper-facing follow-up names the supported `Results section job`, whether the result serves the `discovery cash-out`, and whether it risks overweighting tool or repeated explanation
- paper-facing explanation uses active voice, simple cause-effect phrasing, low noun-stack density, and natural scientific English; do not bury causality in nominalizations or leave important numbers unexplained

Common trigger phrases:

- "which factors matter most"
- "what can we compute right now"
- "can we do feature selection or statistical tests"
- "what is the candidate universal equation"
- "which metric explains performance better"
- "make a scorecard or factor audit"

## Guardrails

- Do not treat unexecuted code as evidence.
- Do not fabricate metrics, artifacts, or causal explanations.
- Do not promote a run artifact into manuscript evidence based on filenames or logs alone when a rendered figure, generator path, and provenance trail are available.
- Do not compare candidate metrics across mixed definitions of `H` or mixed preprocessing without naming and justifying the chosen representation first.
- Do not present a descriptive correlation audit as an established physical law.
- Keep code subordinate to executed artifacts and reproducibility records.
