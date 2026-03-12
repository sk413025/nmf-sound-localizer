# Governance Constitution - Nature Communications Manuscript Worktree

This worktree is an agent-first, manuscript-first Nature Communications branch. Code, figures, and results are subordinate substrate that Codex uses to advance manuscript, evidence, review, and submission work.

## Governance Precedence

When instructions conflict, follow this order:

1. This file as the branch constitution
2. Canonical task documents such as `paper/manuscript/manuscript.md` and Nature requirements
3. Operational contracts under `docs/governance/`
4. Agent operating model under `docs/agent-ops/`
5. Executable checks under `scripts/paper/` and the top-level `Makefile`
6. Historical notes under `docs/archive/` and git history

## Non-Negotiables

- Keep the branch manuscript-first. Do not treat it as a generic package-development branch or a figure-only sandbox.
- Keep code subordinate to paper work. Read or change code only when it supports manuscript, evidence, review, or submission tasks.
- For any paper-related figure task, inspect the actual figure visually before interpreting, comparing, replacing, renumbering, or approving it.
- For `jpg` and `png` paper assets, inspect the image directly. For `pdf` paper assets, convert every page to viewable PNG previews before judging content or suitability.
- For any generated or data-backed paper figure, inspect the visual asset, the generator or composition code, and the upstream evidence or provenance source before concluding what the figure shows or how it should be used in the manuscript.
- Keep experiment and results commits executable, reproducible, and grounded in real runs with real artifacts.
- Keep outputs under declared subdirectories such as `results/<run_name>/`; do not write run artifacts to the repository root.
- Keep fail-fast behavior. Do not add silent fallbacks, coercions, or best-effort recovery to experiment-critical paths.
- Keep all project-tracked content in English.
- Prefer simplification. Remove redundant files, skills, and documents before adding new process.

## Canonical Entry Points

- Human quickstart: [START_HERE_HUMAN.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_HUMAN.md)
- Agent quickstart: [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- Branch overview: [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
- Governance contracts: [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
- Agent operating model: [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Core Contracts

- Experiment and results work: `docs/governance/experiment-contract.md`
- Manuscript writing and claim/evidence discipline: `docs/governance/manuscript-contract.md`
- Submission packaging and Nature compliance: `docs/governance/submission-contract.md`
- Codex-native routing, skills, and orchestration: `docs/governance/codex-collaboration-contract.md`
- Active code and package substrate: `docs/governance/runtime-substrate-contract.md`
- Asset classification: `docs/governance/ASSET_CLASSES.md`

## Skill Routing

Use only the minimal skill set for repeated work:

- `paper-submission`
- `paper-asset-review`
- `agent-orchestrator`
- `experiment-results`

If a task does not clearly fit one of these skills, route through `agent-orchestrator` first instead of inventing a parallel workflow.

## Command Surface

- `make paper-build`
- `make paper-check`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Current Operating Model

- The human sets direction and approves milestones.
- The supervisor or orchestrator decomposes multi-agent work.
- Specialists execute bounded paper-facing tasks.
- Review and red-team loops are mandatory when claims, governance, or submission posture could shift.

## Asset Boundaries

- Evidence docs may support provenance and paper evidence, but they are not branch entrypoints.
- Working notes may inform future work, but they are never canonical source of truth by themselves.
- Quarantined assets must remain outside the main workflow.
- Active runtime substrate is limited to TF + USM + soft-OMP support. Legacy pipeline, DT, oracle, and reconstruction paths belong under `legacy/runtime/` or `legacy/tests/`.

## Historical Material

Package-era notes are no longer operational source of truth for this branch. Use `docs/archive/`, `CHANGELOG.md`, `nmf_localizer/README.md`, and git history only when deliberate historical context is needed.
