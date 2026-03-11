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
3. `TASK_LIFECYCLE.md`
4. `HANDOFF_RULES.md`
5. `REVIEW_AND_ESCALATION.md`
6. the matching task packet under `task-packets/`

## Main files

- `SUPERVISOR_OPERATING_MODEL.md`
- `ROLE_CATALOG.md`
- `TASK_LIFECYCLE.md`
- `HANDOFF_RULES.md`
- `REVIEW_AND_ESCALATION.md`
- `task-packets/`

## Relationship to governance

- `AGENTS.md` is the constitution
- `docs/governance/` defines contracts
- `docs/agent-ops/` defines execution and handoff
- project-local skills implement repeated task workflows

## Default operating model

1. human gives a goal
2. supervisor frames the task
3. specialist agents work through role-appropriate task packets
4. reviewer or red-team agents challenge outputs when required
5. supervisor consolidates
6. human approves or redirects at milestone boundaries
