# Start Here for Humans

This worktree exists to write, validate, and prepare a Nature Communications manuscript. It is not primarily a package-development branch and not primarily a figure-only branch.

The intended operating model is agent-first: Codex and other agents do the detailed manuscript, evidence, and review work; the human mainly sets direction and approves milestones.

## Read in this order

1. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
2. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
3. [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)
4. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
5. [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
6. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Use these depending on task

- Writing or revising the manuscript:
  - `paper/manuscript/manuscript.md`
  - `docs/governance/manuscript-contract.md`
  - `docs/governance/scientific-voice-guide.md`
  - use the voice guide for both paper-level claim posture and sentence-level readability (`noun stacking`, `causal glue`, and natural scientific English)
- Nature submission or artwork compliance:
  - `docs/nature-communications/nature-communications-submission-requirements.md`
  - `docs/governance/submission-contract.md`
  - paper-facing legends, captions, and submission prose still inherit the same sentence-energy defaults
- Experiments, results commits, or reproducibility:
  - `docs/governance/experiment-contract.md`
- Active code or package substrate:
  - `docs/governance/runtime-substrate-contract.md`
  - `docs/governance/ASSET_CLASSES.md`
- Codex and multi-agent workflow design:
  - `docs/governance/codex-collaboration-contract.md`
  - `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`

## Common commands

- `make paper-build`
- `make paper-check`
- `make paper-audit-voice`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Historical material

Older package-era and toolkit-era material has been removed from the main entry path. Use `docs/archive/` only when you intentionally need historical context.
