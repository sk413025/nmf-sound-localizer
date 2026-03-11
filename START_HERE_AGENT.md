# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
3. [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)
4. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)

## Task routing

- Manuscript or submission task:
  - Read `docs/nature-communications/nature-communications-submission-requirements.md`
  - Use `.codex/skills/nature-communications-submission/SKILL.md`
- Paper-facing asset review:
  - Use `.codex/skills/paper-asset-review/SKILL.md`
  - Run `python scripts/paper/review_paper_assets.py prepare`
- Codex-native or multi-agent governance task:
  - Use `.codex/skills/codex-native-assessment/SKILL.md`
  - Read `docs/codex-native-assessment/README.md`
- Experiment or results task:
  - Follow `docs/governance/experiment-contract.md`

## Common commands

- `make paper-build`
- `make paper-check`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Do not use as source of truth

- `README.md` content from older package-era history in git
- `CONTRIBUTING.md` from older package-era history in git
- `NATURE_FIGURE_GUIDELINES.md` as an authoritative policy source
- archived notes under `docs/archive/`
