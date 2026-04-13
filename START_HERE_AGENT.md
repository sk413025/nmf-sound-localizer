# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Mandatory first pass

Only three top-level surfaces are mandatory before routing:

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [.codex/memory/CURRENT_BRANCH_MEMORY.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/memory/CURRENT_BRANCH_MEMORY.md)
3. [.codex/skills/agent-orchestrator/SKILL.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/agent-orchestrator/SKILL.md)

Load other docs only after the top-level routing step says they are needed.

## Top-level route

1. Identify the task surface:
   - manuscript or submission
   - paper asset review
   - experiment or results interpretation
   - governance or orchestration
   - runtime substrate maintenance
2. Route through `agent-orchestrator`.
3. After routing, load only the smallest on-demand surface that fits:
   - `paper-submission`
   - `paper-asset-review`
   - `experiment-results`
   - one task-packet section
   - one task-specific contract if needed

## Unfamiliarity bootstrap

When the environment, artifact lineage, or runtime surface is not already familiar, do this before any mutating action:

1. classify the task surface and likely source-of-truth layer
2. inspect the environment surface you will rely on
3. inspect the minimum memory, contract, or artifact needed for that surface
4. name the main unknowns, risks, and missing evidence
5. choose the next epistemic action before choosing the first mutating action

Valid epistemic actions include:

- inspect repo structure or relevant files
- inspect figure assets, logs, schemas, or API surfaces
- inspect the runtime substrate contract
- run dry checks or read-only validation
- narrow scope before acting

Do not treat unfamiliarity as a reason to freeze. Treat it as a reason to gather the next missing fact on purpose.

## On-demand surfaces

Load these only when the task requires them:

- `docs/governance/scientific-voice-guide.md` for paper-facing prose or critique
- `docs/agent-ops/TASK_PACKETS.md` when delegating or when a direct round still needs explicit ownership and acceptance surfaces
- `docs/agent-ops/NATURE_REVIEWER_STACK.md` when paper-facing review, hardening, or red-team critique is in scope
- `docs/governance/runtime-substrate-contract.md` when the task touches active runtime code, scripts outside `scripts/paper/`, tests, or package metadata
- `docs/governance/codex-collaboration-contract.md` when changing governance, routing, or machine-checkable workflow rules
- `docs/governance/closeout-integrity-contract.md` when a round can change claims, acceptance status, or closeout posture

## Working rules

- Code is substrate. Read or change code only when it supports manuscript, evidence, review, or submission work.
- For manuscript, governance, strategy, and other branch-shaping tasks, read the branch memory brief first and open archive notes only when the brief points there.
- If a task includes a substantive `.codex/memory` update, first open `.codex/memory/README.md`, then run a weak-agent anti-log review before the first mutating action on that memory file.
- Child agents may propose memory candidates, but only the top-level agent may promote content into `CURRENT_BRANCH_MEMORY.md` or `archive/`.
- Context mode: `summary-only` is the default for child handoff. Upgrade to `summary+fork_context` only when exact dialogue history cannot be safely compressed.
- Before making any paper-figure judgment, inspect the real asset. For `pdf`, inspect PNG previews for every page first.
- For generated or data-backed paper figures, inspect the visual asset, generator or composition code, and upstream evidence or provenance source before drawing conclusions.
- Do not use older package-era README or CONTRIBUTING text, `NATURE_FIGURE_GUIDELINES.md`, or archive notes as current source of truth.

## Common commands

- `make paper-build`
- `make paper-check`
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`
