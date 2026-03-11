# Codex Collaboration Contract

Use this contract for any task about Codex-native workflow, multi-agent organization, governance design, AGENTS usage, or local skills.

## Applies to

- `AGENTS.md`
- `START_HERE_AGENT.md`
- project-local skills under `.codex/skills/`
- `docs/agent-ops/`
- Codex-native assessment workflows
- supervisor, specialist, and red-team coordination

## Core rules

- Define `codex-native` from actual Codex capabilities and local repository primitives, not from taste alone.
- Prefer existing repository primitives such as `AGENTS.md`, local skills, and `scripts/paper/` over parallel systems.
- Keep collaboration manuscript-first.
- Make code subordinate to manuscript, evidence, and submission workflows.
- Distinguish branch-local source of truth from archive material.
- Multi-agent recommendations must include explicit acceptance criteria and ownership boundaries.

## Required outputs

- clear routing for human, agent, and supervisor roles
- reusable role packets or skills for repeated work
- evidence-backed recommendations
- executable checks where policy is high value and low ambiguity
- a stable agent operating model for task decomposition, handoff, and review

## Acceptance criteria

- main entrypoints clearly route to the right canonical docs
- local skills align with branch governance
- agent task packets and role definitions are discoverable from the main governance path
- governance checks confirm the key files and links exist
- Codex-native assessment pack remains discoverable from the main governance path

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `make paper-check`
- `docs/codex-native-assessment/README.md`
- `docs/agent-ops/README.md`
