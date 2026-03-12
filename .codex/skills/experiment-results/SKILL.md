---
name: experiment-results
description: Use this skill for interpreting executed runs, writing results-grounded analysis, or checking experiment provenance and reproducibility in this repository.
---

# Experiment Results

Use this skill for:

- executed run interpretation
- results-commit analysis
- provenance-backed manuscript support
- failure analysis grounded in committed artifacts

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/experiment-contract.md`
- `docs/agent-ops/TASK_PACKETS.md`

## Workflow

1. Start from committed artifacts under `results/<run_name>/`.
2. Extract numbers, logs, provenance inputs, and failure signals.
3. If a run artifact is being used as a paper-facing figure or as support for a paper-facing figure, inspect the visual asset first. For `pdf` artifacts, convert every page to PNG previews before review.
4. When a figure interpretation is involved, inspect the generator or composition code and reconcile it with the upstream run artifacts before writing conclusions.
5. Write analysis in the language required by the experiment contract.
6. Escalate if the requested claim outruns the available evidence.

## Guardrails

- Do not treat unexecuted code as evidence.
- Do not fabricate metrics, artifacts, or causal explanations.
- Do not promote a run artifact into manuscript evidence based on filenames or logs alone when a rendered figure, generator path, and provenance trail are available.
- Keep code subordinate to executed artifacts and reproducibility records.
