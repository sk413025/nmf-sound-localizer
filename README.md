# Nature Communications Manuscript Worktree

This branch is a manuscript-first workspace for writing, validating, and packaging a Nature Communications submission on non-contact acoustic sensing with everyday objects as physical encoders.

## Start here

- Human quickstart: [START_HERE_HUMAN.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_HUMAN.md)
- Agent quickstart: [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- Branch constitution: [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
- Manuscript workspace: [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)
- Governance contracts: [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)

## Canonical documents

- Nature Communications requirements:
  - `docs/nature-communications/nature-communications-submission-requirements.md`
- Manuscript source:
  - `paper/manuscript/manuscript.md`
- Figure registry:
  - `figures/FIGURE_REGISTRY.md`
- Codex-native assessment pack:
  - `docs/codex-native-assessment/README.md`

## Command surface

```bash
# Build the manuscript DOCX only
make paper-build

# Run manuscript and governance checks
make paper-check

# Rebuild figures, then rebuild the manuscript
make manuscript

# Prepare manuscript-facing asset review bundles
make paper-review-assets

# Enforce paper asset review decisions
make paper-review-gate
```

## Workspace map

- `paper/`: manuscript source, references, templates, and final paper assets
- `figures/`: generator, validator, review-bundle, and deployment pipeline
- `scripts/paper/`: manuscript build and paper-facing checks
- `docs/governance/`: operational contracts for experiments, manuscript work, submission work, and Codex collaboration
- `docs/archive/`: historical or package-era materials that are not source of truth for this branch

## Historical material

This worktree evolved out of a broader acoustic-localization toolkit repository. That history still matters for provenance and legacy experiments, but it is not the primary identity of this branch.

- Historical notes: `docs/archive/`
- Package-era module docs: `nmf_localizer/README.md`
- Historical package changelog: `CHANGELOG.md`
