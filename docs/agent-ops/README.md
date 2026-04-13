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
5. `ROUND_GOVERNANCE_SCHEMA.md`
6. `REVIEW_AND_ESCALATION.md`

## Main files

- `SUPERVISOR_OPERATING_MODEL.md`
- `ROLE_CATALOG.md`
- `NATURE_REVIEWER_STACK.md`
- `TASK_PACKETS.md`
- `ROUND_GOVERNANCE_SCHEMA.md`
- `REVIEW_AND_ESCALATION.md`
- `ROUND_CLOSEOUT_TEMPLATE.md`

## Relationship to governance

- `AGENTS.md` is the constitution
- `docs/governance/` defines contracts
- `docs/governance/scientific-voice-guide.md` is the canonical positive-style reference for paper-facing explanation, including narrative architecture, sentence-energy, and natural-scientific-English defaults
- `docs/agent-ops/` defines execution and handoff
- project-local skills implement repeated task workflows

## Codex-native baseline

Treat this branch as Codex-native only when the workflow is grounded in actual repository primitives:

- `AGENTS.md` as the constitution
- `.codex/memory/CURRENT_BRANCH_MEMORY.md` as the derived branch-memory brief
- project-local skills as reusable work modes
- `docs/governance/` as contracts
- `docs/agent-ops/` as supervisor, role, and task routing
- `make paper-check` and `scripts/paper/` as executable gates
- `make paper-governance-gate ROUND_DIR=results/<round_name>` plus `results/<round_name>/governance_round.yaml` as the blocking semantic gate for high-risk broader-significance rounds

Do not rebuild a parallel platform around these primitives unless a real workflow gap remains after simplification.

## Default operating model

1. human gives a goal
2. the top-level supervisor routes through `agent-orchestrator` and decides whether to execute directly or delegate
3. for paper-facing work, the top-level supervisor first classifies `Architecture scope` as `local-salience`, `cross-section`, or `whole-manuscript`
4. `cross-section` and `whole-manuscript` rounds must carry the drafting-time architecture bundle required by `TASK_PACKETS.md`, `manuscript-contract.md`, and `scientific-voice-guide.md`
5. when a `cross-section` or `whole-manuscript` round claims architecture landed, the closeout must carry an `Architecture evidence map`
6. when delegating, the top-level supervisor chooses the child role, specialist skill, and unified task packet
7. the round records `Relevant conversation context`, `Context mode`, risk level, and ownership boundaries
8. for paper-facing explanation and paper-facing hardening, the top-level supervisor selects the applicable reviewer roles and evaluation goals from `NATURE_REVIEWER_STACK.md`
9. direct work or delegated child work proceeds through one of the core skills
10. reviewer or red-team agents challenge outputs when required, using the canonical reviewer stack instead of ad hoc reviewer personas
11. the packet or closeout classifies the round as `high-risk` or `non-high-risk`
12. for `high-risk` rounds, the round names both a review owner and a verification owner; compressed ownership requires an explicit compression rationale
13. for `non-high-risk` rounds, compressed ownership requires a non-high-risk rationale
14. the top-level supervisor consolidates only against the round's owned plan items, acceptance surface, delivery evidence, and verification target
15. if delivered scope is narrower than planned scope, the top-level supervisor discloses the downgrade explicitly rather than closing out the original round as complete
16. use the closeout-integrity governance contract plus `ROUND_CLOSEOUT_TEMPLATE.md` for dialogue closeouts when a reusable ledger helps, but do not treat the template as a mandatory repo artifact
17. the human approves or redirects at milestone boundaries

For broader-significance manuscript rounds, treat `second-layer discovery` as the mandatory bridge between core discovery and any downstream application. Applications that do not descend from that bridge are out of scope for Nature-facing hardening in this branch.
Use `ROUND_GOVERNANCE_SCHEMA.md` as the only canonical machine-readable field inventory for `high-risk` broader-significance rounds, and require `results/<round_name>/governance_round.yaml` plus a passing `make paper-governance-gate ROUND_DIR=results/<round_name>` before closeout may report that broader significance landed.
