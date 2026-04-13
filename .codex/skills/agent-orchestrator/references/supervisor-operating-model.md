# Supervisor Operating Model

This file is a deeper reference for top-level supervision inside `agent-orchestrator`.

The runnable top-level workflow lives in `../SKILL.md`. Use this file only when you need extra supervision guidance after routing.

## Mission

Convert high-level goals into agent-safe work units while preserving two things at once:

- evidence discipline
- claim-floor legibility

Do not protect the first by sacrificing the second.

## Codex-native baseline

Define `codex-native` from real primitives in this branch:

- `AGENTS.md`
- `.codex/memory/CURRENT_BRANCH_MEMORY.md`
- the four project-local skills
- `docs/governance/`
- `docs/agent-ops/`
- `scripts/paper/` and `make paper-check`

Do not propose an orchestration layer that duplicates these primitives without a concrete gap.

## Complexity discipline

Treat governance complexity as an explicit supervision risk.

Use the anti-complexity discipline from:

- `docs/governance/codex-collaboration-contract.md`
- `.codex/skills/agent-orchestrator/SKILL.md`

Before adding a new field, verdict, artifact, or workflow branch, confirm:

- it is not already canonical elsewhere
- it is not reviewer judgment disguised as checker work
- it removes or collapses an older duplicated surface

If those checks do not produce a strong answer, prefer simplification or reuse over expansion.

## Execution-or-delegation rule

The top-level agent is the default coordinator for routed work. It may execute directly or delegate after an explicit routing decision.

The canonical execution-or-delegation rule, unfamiliarity bootstrap, handoff rules, and supervision loop live in `.codex/skills/agent-orchestrator/SKILL.md`.

Use this file only for deeper supervisory judgment such as:

- whether a round needs separate reviewer and verifier ownership
- whether a multi-surface request should be decomposed before execution
- whether a scope downgrade must be disclosed explicitly in closeout
- whether a proposed simplification actually deleted duplicated surface

## High-risk reminders

For `high-risk` rounds:

- keep implementer, reviewer, and verifier ownership explicit
- do not treat reviewer approval as verification
- require explicit scope-downgrade disclosure when delivery is narrower than planned

For `high-risk` broader-significance rounds:

- use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md`
- create `results/<round_name>/governance_round.yaml`
- require `make paper-governance-gate ROUND_DIR=results/<round_name>` before reporting that broader significance landed

## When this file is worth opening

- manuscript claim changes
- submission-readiness changes
- governance changes
- tasks where wording, evidence, and figures must remain aligned
- closeout-sensitive rounds where ownership or scope truthfulness is in doubt
