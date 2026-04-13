# Agent Operations

This directory is the on-demand operating layer for the `paper/nature-comm` worktree.

It is not part of the mandatory first-pass load for Codex. The top-level agent should first read:

1. `AGENTS.md`
2. `.codex/memory/CURRENT_BRANCH_MEMORY.md`
3. `.codex/skills/agent-orchestrator/SKILL.md`

Open files in this directory only after routing says they are needed.

## What lives here

- `ROLE_CATALOG.md`
  - role names and responsibilities
- `NATURE_REVIEWER_STACK.md`
  - canonical reviewer lenses for paper-facing review
- `TASK_PACKETS.md`
  - packet schema and task templates
- `ROUND_GOVERNANCE_SCHEMA.md`
  - the only machine-readable field inventory for `high-risk` broader-significance rounds
- `REVIEW_AND_ESCALATION.md`
  - warning codes and escalation triggers
- `ROUND_CLOSEOUT_TEMPLATE.md`
  - reusable closeout ledger template

## How to use this layer

- use `agent-orchestrator` for the top-level routing decision
- open `.codex/skills/agent-orchestrator/references/supervisor-operating-model.md` only when deeper supervision notes are needed after routing
- open `TASK_PACKETS.md` when delegation or explicit ownership boundaries are needed
- open `NATURE_REVIEWER_STACK.md` when paper-facing critique or red-team review is in scope
- open `ROUND_GOVERNANCE_SCHEMA.md` only for `high-risk` broader-significance rounds
- open `REVIEW_AND_ESCALATION.md` or `ROUND_CLOSEOUT_TEMPLATE.md` only when review, escalation, or closeout detail is the active problem

## Codex-native baseline

Treat this branch as Codex-native only when the workflow stays grounded in local primitives:

- `AGENTS.md` as the constitution
- `.codex/memory/CURRENT_BRANCH_MEMORY.md` as the derived branch-memory brief
- project-local skills as reusable work modes
- `docs/governance/` as contracts
- `docs/agent-ops/` as on-demand operating references
- `make paper-check` and `scripts/paper/` as executable gates
- `results/<round_name>/governance_round.yaml` plus `make paper-governance-gate ROUND_DIR=results/<round_name>` for `high-risk` broader-significance rounds
