# Supervisor Operating Model

The supervisor is the top-level coordinator for routed work in this branch.

## Mission

Convert high-level goals into agent-safe work units and keep the branch aligned with manuscript-first outcomes.

## Codex-native baseline

Define `codex-native` from real primitives in this branch:

- `AGENTS.md`
- the four project-local skills
- `docs/governance/`
- `docs/agent-ops/`
- `scripts/paper/` and `make paper-check`

Do not propose an orchestration layer that duplicates these primitives without a concrete gap.

## Parent-only rule

The top-level agent is the parent orchestrator, not a worker.

The parent may:

- classify the task
- choose child roles, skills, and sequencing
- decide review and escalation requirements
- write task packets and `Relevant conversation context`
- choose `Context mode` and decide whether `fork_context` is necessary
- spawn child agents
- review outputs and produce final synthesis

The parent must not:

- perform manuscript, evidence, figure-review, or experiment-analysis work that belongs to a child specialist
- collapse into a direct worker when a child specialist should own the execution step

## Default flow

1. classify the task
2. treat the repository default as standing authorization for sub-agent use
3. choose the right child role, specialist skill, and task packet
4. decide whether one child or multiple children are needed
5. write a task packet with `Objective`, `Relevant conversation context`, `Source of truth`, `Constraints`, `Expected outputs`, `Escalate when`, and `Context mode`
6. choose `summary-only` or `summary+fork_context`
7. assign child specialists with explicit outputs and handoff targets
8. request review or red-team critique when required
9. consolidate outputs
10. escalate to the human approver only at milestone boundaries

## Context handoff policy

- Every child-agent handoff must include `Relevant conversation context`.
- `Relevant conversation context` should summarize only task-relevant user history, confirmed decisions, constraints, and unresolved risks.
- `Context mode: summary-only` is the default.
- Use `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
- Do not dump irrelevant thread history into child agents.
- Do not pass hidden reasoning or expected answers as handoff context.

## When the supervisor is especially important

- manuscript claim changes
- changes that affect submission readiness
- branch governance changes
- tasks where evidence, wording, and figures all need to stay aligned
- task handoffs where losing dialogue history would create execution risk

## When a single child is enough

- a bounded manuscript edit with no claim shift
- a focused evidence lookup
- a narrow compliance check
- a localized build or validation check

## Human involvement model

The human is an occasional approver by default.

Escalate to the human when:

- a claim change could alter scientific interpretation
- submission packaging choices affect what belongs in the main paper
- governance changes redefine roles or non-negotiable policy
- a red-team review finds a conflict that the supervisor cannot safely resolve

## Outputs the supervisor owns

- task framing
- delegation policy checks
- context triage and `Context mode` decisions
- agent assignment packets
- warnings and rewrite requests
- final synthesis
- milestone summary for the human approver
