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
3. `NATURE_REVIEWER_STACK.md`
4. `TASK_PACKETS.md`
5. `REVIEW_AND_ESCALATION.md`

## Main files

- `SUPERVISOR_OPERATING_MODEL.md`
- `ROLE_CATALOG.md`
- `NATURE_REVIEWER_STACK.md`
- `TASK_PACKETS.md`
- `REVIEW_AND_ESCALATION.md`
- `ROUND_CLOSEOUT_TEMPLATE.md`

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
2. the top-level supervisor acts as the parent orchestrator
3. the parent chooses the child role, specialist skill, and unified task packet
4. the parent writes `Relevant conversation context` and chooses `Context mode`
5. for manuscript-facing hardening, the parent selects the applicable reviewer roles and evaluation goals from `NATURE_REVIEWER_STACK.md`
6. child specialists work through one of the core skills and one of the unified task packets
7. reviewer or red-team agents challenge outputs when required, using the canonical reviewer stack instead of ad hoc reviewer personas
8. the packet classifies the round as `high-risk` or `non-high-risk`
9. for `high-risk` rounds, the packet names both a review owner and a verification owner; compressed ownership requires an explicit compression rationale
10. for `non-high-risk` rounds, compressed ownership requires a non-high-risk rationale
11. the parent supervisor consolidates only against the packet's owned plan items, acceptance surface, delivery evidence, and verification target
12. if delivered scope is narrower than planned scope, the parent discloses the downgrade explicitly rather than closing out the original round as complete
13. use the closeout-integrity governance contract plus `ROUND_CLOSEOUT_TEMPLATE.md` for dialogue closeouts when a reusable ledger helps, but do not treat the template as a mandatory repo artifact
14. the human approves or redirects at milestone boundaries
