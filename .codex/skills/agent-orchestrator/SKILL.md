---
name: agent-orchestrator
description: Top-level parent orchestration for this repository. Use when Codex is the first agent receiving a task in this branch, or when coordinating child agents, defining role boundaries, choosing task packets, managing review loops, or routing manuscript-first requests to the right child-worker skill. The top-level agent must stay a parent orchestrator, prepare context-aware handoffs, and delegate execution to child agents.
---

# Agent Orchestrator

Use this skill for:

- top-level parent orchestration
- supervisor-led task decomposition
- child-agent coordination
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

## Parent-only rule

Act as the top-level parent orchestrator, not a worker.

Do:

- classify the task
- choose the child role, skill, and task packet
- write `Relevant conversation context`
- choose `Context mode`
- decide whether `fork_context` is necessary
- spawn child agents
- review outputs and synthesize results

Do not:

- perform the specialist execution step yourself
- collapse into the child-worker execution step yourself
- dump irrelevant thread history into a child prompt

Apply this parent-only rule in both Default mode and Plan mode.

## Workflow

1. Classify the task by manuscript impact and role complexity.
2. Treat this repository's default operating mode as standing authorization for sub-agent use.
3. Choose the right child task packet, role, and core skill.
4. Write a task packet with `Objective`, `Relevant conversation context`, `Source of truth`, `Constraints`, `Expected outputs`, `Escalate when`, and `Context mode`.
5. Summarize only the task-relevant conversation history.
6. Use `Context mode: summary-only` by default.
7. Upgrade to `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
8. Define review, handoff, and escalation requirements.
9. Keep the human at milestone approval boundaries unless the task requires earlier intervention.

## Plan mode

- Keep the same parent-orchestrator routing in Plan mode.
- Use child agents in Plan mode only for planning, exploration, checking, and review.
- Keep parent and child work non-mutating and plan-safe until execution mode.

Quick routing defaults:

- "what is this figure or panel showing", "does this figure support the claim", or "what gap exists between this figure and this critique" -> `paper-asset-review`
- "which factors matter most", "what can we compute now", "which metric explains performance better", or "can we do a factor audit" -> `experiment-results`
- "write this as manuscript prose", "explain this for cross-disciplinary readers", or "rewrite this in plain language without losing rigor" -> `paper-submission`

## Guardrails

- Do not drift into a code-first coordination model.
- Do not create new roles when the role catalog already fits.
- Do not add new governance layers before proving a concrete workflow gap.
- Prefer manuscript objectives over implementation-centric decomposition.
- Do not let the parent do child-worker execution.
- Do not omit `Relevant conversation context` from a child handoff.
- Do not use `summary+fork_context` when `summary-only` is sufficient.
