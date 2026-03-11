---
name: results-interpretation
description: Use this skill when converting executed results, logs, metrics, and artifacts into experiment-contract analysis or manuscript-supporting interpretation in this repository.
---

# Results Interpretation

Use this skill for:

- interpreting executed runs
- writing result summaries for commits
- turning metrics into analysis language
- identifying success factors and failure modes from actual artifacts

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/experiment-contract.md`
- `docs/agent-ops/task-packets/results-interpretation-packet.md`

## Workflow

1. Gather logs, metrics, and artifacts from the run directory.
2. Extract key numbers and normal-state signals.
3. Interpret them using the experiment contract.
4. Escalate if evidence is incomplete or contradictory.

## Guardrails

- No unexecuted-code speculation as evidence.
- No fabricated numbers.
- Code is substrate; run artifacts are primary.
