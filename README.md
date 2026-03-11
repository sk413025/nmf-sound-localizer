# Nature Communications Manuscript Worktree

This branch is a manuscript-first workspace for writing, validating, and packaging a Nature Communications submission on non-contact acoustic sensing with everyday objects as physical encoders. Code, figures, and results are substrate that Codex and other agents use to serve paper work.

## Start here

- Human quickstart: [START_HERE_HUMAN.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_HUMAN.md)
- Agent quickstart: [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- Branch constitution: [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
- Manuscript workspace: [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)
- Governance contracts: [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
- Agent operating model: [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Canonical documents

- Nature Communications requirements:
  - `docs/nature-communications/nature-communications-submission-requirements.md`
- Manuscript source:
  - `paper/manuscript/manuscript.md`
- Figure registry:
  - `figures/FIGURE_REGISTRY.md`
- Task routing:
  - `docs/agent-ops/TASK_PACKETS.md`
- Asset classification:
  - `docs/governance/ASSET_CLASSES.md`

## Command surface

```bash
make paper-build
make paper-check
make manuscript
make paper-review-assets
make paper-review-gate
```

## Workspace map

- `paper/`: manuscript source, references, templates, and final paper assets
- `figures/`: generator, validator, review-bundle, and deployment pipeline
- `scripts/paper/`: manuscript build and paper-facing checks
- `docs/governance/`: operational contracts
- `docs/agent-ops/`: supervisor model, roles, task packets, and review loop
- `docs/evidence/`: evidence, lineage, and provenance notes that still support current work
- `docs/working-notes/`: exploratory notes, prompts, and analysis that are not canonical
- `nmf_localizer/`, `doa_rl/`, `scripts/`, `tests/`: active runtime substrate governed by `docs/governance/runtime-substrate-contract.md`
- `docs/archive/`: historical material that is not current source of truth
- `legacy/`: quarantined scripts and assets outside the main workflow

## Historical material

This worktree evolved out of a broader acoustic-localization toolkit repository. Use `docs/archive/`, `CHANGELOG.md`, and `nmf_localizer/README.md` only when historical context is genuinely needed.
