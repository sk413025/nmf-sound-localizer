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
- `docs/governance/scientific-voice-guide.md` is the canonical positive-style reference for manuscript voice
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
2. the top-level supervisor routes through `agent-orchestrator` and decides whether to execute directly or delegate
3. for manuscript-facing work, the top-level supervisor extracts the supported discovery sentence before caveat-hardening the prose and uses the canonical `SV#` exemplars to guide high-salience rewrites
4. when delegating, the top-level supervisor chooses the child role, specialist skill, and unified task packet
5. the round records `Relevant conversation context`, `Context mode`, risk level, and ownership boundaries
6. for manuscript-facing hardening, the top-level supervisor selects the applicable reviewer roles and evaluation goals from `NATURE_REVIEWER_STACK.md`
7. direct work or delegated child work proceeds through one of the core skills
8. reviewer or red-team agents challenge outputs when required, using the canonical reviewer stack instead of ad hoc reviewer personas
9. the packet or closeout classifies the round as `high-risk` or `non-high-risk`
10. for `high-risk` rounds, the round names both a review owner and a verification owner; compressed ownership requires an explicit compression rationale
11. for `non-high-risk` rounds, compressed ownership requires a non-high-risk rationale
12. the top-level supervisor consolidates only against the round's owned plan items, acceptance surface, delivery evidence, and verification target
13. if delivered scope is narrower than planned scope, the top-level supervisor discloses the downgrade explicitly rather than closing out the original round as complete
14. use the closeout-integrity governance contract plus `ROUND_CLOSEOUT_TEMPLATE.md` for dialogue closeouts when a reusable ledger helps, but do not treat the template as a mandatory repo artifact
15. the human approves or redirects at milestone boundaries
