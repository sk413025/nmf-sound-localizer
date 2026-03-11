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
3. Write analysis in the language required by the experiment contract.
4. Escalate if the requested claim outruns the available evidence.

## Guardrails

- Do not treat unexecuted code as evidence.
- Do not fabricate metrics, artifacts, or causal explanations.
- Keep code subordinate to executed artifacts and reproducibility records.
