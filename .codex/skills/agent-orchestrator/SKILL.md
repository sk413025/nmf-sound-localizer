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
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
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
- monitor active child-agent progress and latest outputs
- review outputs and synthesize results

Do not:

- perform the specialist execution step yourself
- collapse into the child-worker execution step yourself
- dump irrelevant thread history into a child prompt
- close a child agent solely because it feels slow

Apply this parent-only rule in both Default mode and Plan mode.

## Workflow

1. Classify the task by manuscript impact and role complexity.
2. Treat this repository's default operating mode as standing authorization for sub-agent use.
3. Decide whether the request is a valid single-child task or must be decomposed into multiple child tasks.
4. Choose the right child task packet, role, and core skill for each child task.
5. Write a task packet with `Objective`, `Relevant conversation context`, `Source of truth`, `Constraints`, `Expected outputs`, `Escalate when`, and `Context mode`.
6. For manuscript-facing planning, writing, or review tasks, state the prose acceptance rule explicitly in the packet or review request: prefer active voice, simple cause-effect relations, and lower noun-stack friction when scientific accuracy permits.
7. Summarize only the task-relevant conversation history.
8. Use `Context mode: summary-only` by default.
9. Upgrade to `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
10. After spawning a child agent, monitor its status and latest output before deciding on interruption, redirect, or shutdown.
11. Define review, handoff, and escalation requirements.
12. Keep the human at milestone approval boundaries unless the task requires earlier intervention.

## Reviewer-stack routing

For manuscript-facing work, treat `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source.
Do not invent ad hoc reviewer personas when the stack already covers the risk.

Use the minimal reviewer subset that matches the acceptance surface:

- `handling-editor-scope reviewer` and `reviewer-routing reviewer` for editorial fit, paper-level framing, or likely reviewer-community routing
- `cross-disciplinary-readability reviewer`, `narrative-flow reviewer`, and `cognitive-load reviewer` for manuscript prose, whole-paper flow, reader-burden risk, or sentence-level friction from passive phrasing, nominalization, and noun stacking
- `physical-mechanism reviewer`, `acoustics-doa reviewer`, `sparse-inverse-problem-comparator reviewer`, and `statistics-evidence reviewer` for interpretation, plausibility, comparator fairness, or evidence sufficiency risk
- `figure-science-readability reviewer` for figure science, panel logic, caption burden, or main-vs-supplementary judgment

When writing a child packet or review request, name:

- the selected reviewer roles
- the acceptance surface they are judging
- any follow-up owner if a reviewer finding must be routed to another skill

Parent acceptance on reviewer-stack use:

- the packet cites the canonical reviewer-stack doc
- the reviewer subset is minimal and task-matched
- the acceptance surface is explicit rather than implied
- manuscript-facing packets explicitly state the active-voice, simple-causality, and noun-stack-friction standard when prose quality is in scope
- review findings are consolidated at the parent layer instead of left as disconnected comments

## Plan mode

- Keep the same parent-orchestrator routing in Plan mode.
- Use child agents in Plan mode only for planning, exploration, checking, and review.
- Keep parent and child work non-mutating and plan-safe until execution mode.

## Decomposition decision

Use a single child only when the request fits one core skill, one main output bundle, and one bounded acceptance surface.

Decompose into multiple child tasks when the request:

- spans multiple skills or specialist roles
- mixes execution with a separate review, audit, or red-team pass
- contains parallelizable subproblems with different source-of-truth sets or outputs
- would force one child packet to satisfy multiple independent acceptance criteria

## Supervision loop

- After spawning a child agent, either continue non-overlapping parent work or check the child's progress explicitly.
- Before interrupting or closing a child agent, inspect its current status or latest output first.
- Close a child agent only after completion, user cancellation, clear supersession, or an explicit redirect decision grounded in observed status.
- Do not close a child agent solely because elapsed time feels long.

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
- Do not interrupt or close a child agent without first checking status or latest output.
