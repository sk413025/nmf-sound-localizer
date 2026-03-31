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
- monitor active child-agent progress and latest outputs
- review outputs and produce final synthesis

The parent must not:

- perform manuscript, evidence, figure-review, or experiment-analysis work that belongs to a child specialist
- collapse into a direct worker when a child specialist should own the execution step
- close a child agent solely because it feels slow

This parent-only rule applies in both Default mode and Plan mode.

## Default flow

1. classify the task
2. treat the repository default as standing authorization for sub-agent use
3. decide whether the request is a true single-child task or must be decomposed into multiple child tasks
4. choose the right child role, specialist skill, and task packet for each child task
5. write a task packet with `Objective`, `Relevant conversation context`, `Source of truth`, `Constraints`, `Expected outputs`, `Escalate when`, and `Context mode`
6. choose `summary-only` or `summary+fork_context`
7. assign child specialists with explicit outputs and handoff targets
8. monitor active child agents and inspect status before interrupting, redirecting, or closing them
9. request review or red-team critique when required
10. consolidate outputs
11. escalate to the human approver only at milestone boundaries

## Decomposition threshold

Use a single child only when all of the following are true:

- the work maps to one core skill
- the work has one main output bundle
- the work does not require an independent review or red-team pass as a separate child task
- the acceptance criteria can be satisfied by one bounded specialist task packet

Decompose into multiple child tasks when any of the following are true:

- the request spans more than one core skill or specialist role
- the request combines execution work with a separate review, audit, or red-team step
- the request contains parallelizable subproblems with different source-of-truth sets or output bundles
- one child packet would otherwise contain multiple independent acceptance criteria or loosely related asks
- evidence, figure, manuscript, or governance work must be coordinated but should not be executed by the same child

## Plan mode behavior

- Keep the same parent-orchestrator routing in Plan mode.
- The parent may spawn child agents for planning, exploration, checking, and review.
- In Plan mode, both parent and child work must remain non-mutating and plan-safe.
- Do not use child agents in Plan mode to implement repo-tracked changes.

## Supervision loop

- After spawning a child agent, either do non-overlapping parent work or check the child's progress explicitly.
- Before interrupting or closing a child agent, inspect its current status or latest output first.
- Close a child agent only when it has completed, the user has cancelled the work, the task has been superseded, or the parent has reviewed the status and decided on an explicit redirect.
- Do not shut down a child agent solely because elapsed time feels long.

## Context handoff policy

- Every child-agent handoff must include `Relevant conversation context`.
- `Relevant conversation context` should summarize only task-relevant user history, confirmed decisions, constraints, and unresolved risks.
- `Context mode: summary-only` is the default.
- Use `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
- Do not dump irrelevant thread history into child agents.
- Do not pass hidden reasoning or expected answers as handoff context.
- Treat current child status and latest output as part of the required supervision context before any redirect or shutdown.

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
- child-status checks before interruption or shutdown
- warnings and rewrite requests
- final synthesis
- milestone summary for the human approver
