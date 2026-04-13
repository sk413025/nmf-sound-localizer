---
name: agent-orchestrator
description: Top-level orchestration for this repository. Use when Codex is the first agent receiving a task in this branch, or when coordinating child agents, defining role boundaries, choosing task packets, managing review loops, or routing manuscript-first requests.
---

# Agent Orchestrator

Use this skill for:

- top-level routing
- execution-versus-delegation decisions
- unfamiliarity management before acting
- child-agent handoff and supervision
- review, escalation, and closeout routing

## Preflight

Before using this skill, the top-level agent should already have read:

- `AGENTS.md`
- `.codex/memory/CURRENT_BRANCH_MEMORY.md`
- `START_HERE_AGENT.md`

Do not bulk-load the rest of the governance stack. Load only the smallest additional surface needed for the routed task.

## Progressive disclosure

Open these only when needed:

- `docs/agent-ops/TASK_PACKETS.md`
  - when delegating or when a direct round still needs explicit ownership and acceptance surfaces
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
  - when paper-facing review or red-team critique is in scope
- `docs/governance/scientific-voice-guide.md`
  - when drafting or judging paper-facing prose
- `docs/governance/codex-collaboration-contract.md`
  - when changing governance or machine-checkable workflow rules
- `docs/governance/closeout-integrity-contract.md`
  - when closeout posture, owner separation, or reporting truthfulness is the active problem
- `docs/governance/runtime-substrate-contract.md`
  - when work touches active runtime code, scripts outside `scripts/paper/`, tests, or package metadata
- `references/supervisor-operating-model.md`
  - when deeper supervision guidance is needed after the top-level routing decision

## Unfamiliarity bootstrap

When the environment, task surface, or artifact lineage is not already familiar, complete this bootstrap before any mutating action:

1. classify the task surface and likely source-of-truth layer
2. inspect the environment surface you will rely on
3. inspect the smallest memory, contract, or artifact that can reduce uncertainty
4. name the main unknowns, risks, and missing evidence
5. choose the next epistemic action before the first mutating action

Epistemic actions include:

- inspect repo structure or relevant files
- inspect figure assets, logs, schemas, or API surfaces
- inspect the runtime substrate contract
- run dry checks or read-only validation
- narrow scope before acting

Treat unfamiliarity as a routing signal, not only a stop sign.

## Core routing decision

Make an explicit execution-or-delegation decision before specialist work begins.

Choose direct execution when:

- the work is bounded
- delegation would not improve correctness, throughput, or role separation
- required reviewer and verifier ownership can still remain explicit

Choose delegation when:

- the work spans more than one specialist surface
- review or red-team critique should remain separate from implementation
- child isolation improves source-of-truth handling or throughput

If delegating, decide whether the request is:

- a true single-child task
- or a multi-child task that must be decomposed before execution begins

## Handoff rules

Every child-agent handoff must include:

- `Relevant conversation context`
- `Context mode`
- the owned acceptance surface
- the owned output bundle

`Relevant conversation context` should summarize only task-relevant history, confirmed decisions, constraints, and unresolved risks.

`Context mode: summary-only` is the default.
Use `summary+fork_context` only when exact wording or non-compressible dialogue constraints matter.

Do not dump irrelevant thread history into child prompts.

## Packet discipline

Open `docs/agent-ops/TASK_PACKETS.md` when delegation or explicit round ownership is needed.

Use packet fields to record:

- source of truth
- constraints
- acceptance surface
- out-of-scope surfaces
- delivery evidence
- review owner
- verification owner

For unfamiliar or runtime-touching tasks, make sure the packet records:

- which environment surface was inspected first
- which unknowns remain open
- which epistemic action must happen before mutation if uncertainty is still high

## Paper-facing routing

For manuscript, supplementary, legend, caption, review-note, or analysis-summary prose:

- load `docs/governance/scientific-voice-guide.md`
- keep the paper's discovery as protagonist
- preserve the supported claim floor before adding the evidence boundary
- classify `Architecture scope` as `local-salience`, `cross-section`, or `whole-manuscript`

If paper-facing review or critique is in scope, load `docs/agent-ops/NATURE_REVIEWER_STACK.md` and use the smallest reviewer subset that fits.

## High-risk rounds

Treat a round as `high-risk` when it can change manuscript claims, governance posture, acceptance status, or closeout state in a way that could be overstated in the final report.

For `high-risk` rounds:

- keep implementer, reviewer, and verifier ownership explicit
- do not treat reviewer approval as verification
- require scope-downgrade disclosure if the delivered surface is narrower than planned

For `high-risk` broader-significance rounds:

- use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md`
- create `results/<round_name>/governance_round.yaml`
- require `make paper-governance-gate ROUND_DIR=results/<round_name>` before reporting that broader significance landed

## Supervision loop

After spawning a child agent:

- continue non-overlapping parent work, or
- check the child's status explicitly

Before interrupting or closing a child agent:

- inspect its current status or latest output first

Close a child agent only after:

- completion
- user cancellation
- clear supersession
- or an explicit redirect grounded in observed status

Do not close a child agent solely because it feels slow.

## Guardrails

- Do not invent a parallel workflow when the existing skills and contracts already fit.
- Do not rebuild packet or schema logic inside a child prompt.
- Do not let governance caution leak into paper-facing prose.
- Do not treat archive notes as current source of truth.
- Do not let simplification add a second governance vocabulary.
