---
name: agent-orchestrator
description: Use this skill when coordinating multiple agents, defining role boundaries, choosing task packets, managing review loops, or routing ambiguous manuscript-first requests to the right existing core skill in this repository.
---

# Agent Orchestrator

Use this skill for:

- supervisor-led task decomposition
- multi-agent coordination
- review and escalation planning
- routing tasks to roles and skills

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/codex-collaboration-contract.md`
- `docs/agent-ops/README.md`
- `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
- `docs/agent-ops/TASK_PACKETS.md`
- `docs/agent-ops/REVIEW_AND_ESCALATION.md`

## Workflow

1. Classify the task by manuscript impact and role complexity.
2. Choose the right task packet, role, and core skill.
3. Define review, handoff, and escalation requirements.
4. Keep the human at milestone approval boundaries unless the task requires earlier intervention.

Quick routing defaults:

- "what is this figure or panel showing", "does this figure support the claim", or "what gap exists between this figure and this critique" -> `paper-asset-review`
- "which factors matter most", "what can we compute now", "which metric explains performance better", or "can we do a factor audit" -> `experiment-results`
- "write this as manuscript prose", "explain this for cross-disciplinary readers", or "rewrite this in plain language without losing rigor" -> `paper-submission`

## Guardrails

- Do not drift into a code-first coordination model.
- Do not create new roles when the role catalog already fits.
- Do not add new governance layers before proving a concrete workflow gap.
- Prefer manuscript objectives over implementation-centric decomposition.
