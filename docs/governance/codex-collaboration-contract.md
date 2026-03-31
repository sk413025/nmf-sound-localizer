# Codex Collaboration Contract

Use this contract for any task about Codex-native workflow, multi-agent organization, governance design, AGENTS usage, or local skills.

## Applies to

- `AGENTS.md`
- `START_HERE_AGENT.md`
- project-local skills under `.codex/skills/`
- `docs/agent-ops/`
- supervisor, specialist, and red-team coordination

## Core rules

- Define `codex-native` from actual Codex capabilities and local repository primitives, not from taste alone.
- Prefer existing repository primitives such as `AGENTS.md`, local skills, and `scripts/paper/` over parallel systems.
- Keep collaboration manuscript-first.
- Make code subordinate to manuscript, evidence, and submission workflows.
- Distinguish branch-local source of truth from archive material.
- Route the top-level agent through `agent-orchestrator` and treat it as a parent orchestrator, not a worker.
- Multi-agent recommendations must include explicit acceptance criteria and ownership boundaries.
- The top-level parent must make an explicit decomposition decision before execution: single child only for genuinely single-scope work, otherwise split the request into multiple child tasks.
- Require every child-agent handoff to include task framing plus `Relevant conversation context`.
- Default child-agent handoff to `Context mode: summary-only`; use `summary+fork_context` only when exact task-relevant dialogue cannot be safely compressed.
- Do not pass irrelevant thread history, hidden reasoning, or expected answers to child agents.
- In this repository's default operating mode, treat the human as providing standing authorization for sub-agent use and let the top-level parent decide when child agents are needed.
- Require the top-level parent to monitor active child agents and inspect status before interrupting or closing them.
- Do not close a child agent solely because elapsed time feels long.
- For paper-related figures, use Codex multimodal capability on the real asset rather than metadata-only inference.
- Require image inspection for `jpg` and `png`, and page-by-page PDF-to-PNG conversion before figure interpretation when the asset is a `pdf`.
- For generated or data-backed figures, require a three-layer check: visual asset, generator or composition code, and upstream evidence or provenance artifact.

## Required outputs

- clear routing for human, agent, and supervisor roles
- reusable skills and unified task packets for repeated work
- task packets that record `Relevant conversation context` and a `Context mode` decision
- evidence-backed recommendations
- executable checks where policy is high value and low ambiguity
- a stable agent operating model for task decomposition, handoff, and review

## Acceptance criteria

- main entrypoints clearly route to the right canonical docs
- local skills align with branch governance and stay limited to the core branch workflows
- top-level routing goes through `agent-orchestrator`
- task packets and role definitions are discoverable from the main governance path
- decomposition guidance distinguishes valid single-child tasks from tasks that must be split
- task packets include `Relevant conversation context` and explicit `Context mode`
- supervision guidance requires child-status checks before interruption or shutdown
- governance checks confirm the key files and links exist
- Codex-native orchestration guidance remains discoverable from the main governance path
- paper-related figure workflows do not allow metadata-only acceptance when visual inspection or provenance backtrace is required

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `make paper-check`
- `docs/agent-ops/README.md`
