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
8. When you create a new analysis, write it as a reproducible run bundle under `results/<run_name>/` with script, command, and machine-readable outputs.
9. Write analysis in the language required by the experiment contract.
10. Escalate if the requested claim outruns the available evidence.

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
