# Contributing to the Nature Communications Manuscript Worktree

This branch is governed as a manuscript-first research worktree. Contributions should improve manuscript quality, submission readiness, reproducibility, paper-facing assets, or Codex collaboration on those tasks.

## Read first

1. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
2. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
3. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
4. [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
5. [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)

## Contribution modes

- Manuscript edits:
  - Follow `docs/governance/manuscript-contract.md`
  - Use `docs/governance/scientific-voice-guide.md` as the canonical positive-style reference for high-salience prose
- Submission or figure compliance work:
  - Follow `docs/governance/submission-contract.md`
- Experiments, validations, and results commits:
  - Follow `docs/governance/experiment-contract.md`
- Runtime substrate or package-facing changes:
  - Follow `docs/governance/runtime-substrate-contract.md`
  - Default to maintaining only TF + USM + soft-OMP support paths
- Codex or multi-agent workflow changes:
  - Follow `docs/governance/codex-collaboration-contract.md`

## Required branch-local behavior

- Keep the branch manuscript-first; do not optimize the repo around figure generation alone.
- Use real data for validation when touching experiment or results paths.
- Keep paper-facing artifacts and outputs under the branch's established locations, especially `results/<run_name>/`.
- Prefer updating canonical docs and executable checks together when policy changes.
- Keep all repo-tracked content in English.

## Minimum validation before proposing completion

```bash
make paper-check
```

Run additional commands as needed:

```bash
make paper-build
make paper-audit-voice
make manuscript
make paper-review-assets
make paper-review-gate
```

## Historical package-era guidance

Older package-oriented contribution guidance is no longer the main operating model for this branch. Historical context is preserved under `docs/archive/` and git history.
