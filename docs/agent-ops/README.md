# Agent Operations

This directory defines how the `paper/nature-comm` worktree operates as an agent-first paper system.

## Purpose

Use this layer when the question is not just "what is allowed?" but "how should agents actually work here?"

This layer treats:

- manuscript as the product
- evidence and submission readiness as operating objectives
- code, figures, and results as substrate used by agents
- humans as milestone approvers by default

## Read in this order

1. `SUPERVISOR_OPERATING_MODEL.md`
2. `ROLE_CATALOG.md`
3. `TASK_PACKETS.md`
4. `REVIEW_AND_ESCALATION.md`

## Main files

- `SUPERVISOR_OPERATING_MODEL.md`
- `ROLE_CATALOG.md`
- `TASK_PACKETS.md`
- `REVIEW_AND_ESCALATION.md`

## Relationship to governance

- `AGENTS.md` is the constitution
- `docs/governance/` defines contracts
- `docs/agent-ops/` defines execution and handoff
- project-local skills implement repeated task workflows

## Codex-native baseline

Treat this branch as Codex-native only when the workflow is grounded in actual repository primitives:

- `AGENTS.md` as persistent branch memory
- project-local skills as reusable work modes
- `docs/governance/` as contracts
- `docs/agent-ops/` as supervisor, role, and task routing
- `make paper-check` and `scripts/paper/` as executable gates

Do not rebuild a parallel platform around these primitives unless a real workflow gap remains after simplification.

## Default operating model

1. human gives a goal
2. supervisor frames the task when needed
3. specialists work through one of the core skills and one of the unified task packets
4. reviewer or red-team agents challenge outputs when required
5. supervisor consolidates
6. human approves or redirects at milestone boundaries
