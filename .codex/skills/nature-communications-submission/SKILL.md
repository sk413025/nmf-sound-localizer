---
name: nature-communications-submission
description: Use this skill when editing manuscript text, figure/table formatting, artwork export rules, submission metadata, or journal-facing documentation for Nature Communications in this repository.
---

# Nature Communications Submission

Use this skill for tasks involving:

- `paper/` manuscript edits
- `figures/` generators, validators, and export settings
- figure legends, tables, source-data prep, or submission checklists
- Nature Communications compliance questions for this repository

Do not use this skill as the primary workflow for manuscript-facing figure acceptance review. For that, use:

- [paper-asset-review](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/paper-asset-review/SKILL.md)

## Required first step

Open and follow:

- [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- [docs/nature-communications/nature-communications-submission-requirements.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/nature-communications/nature-communications-submission-requirements.md)
- [docs/governance/submission-contract.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/submission-contract.md)
- [docs/agent-ops/task-packets/submission-readiness-packet.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/task-packets/submission-readiness-packet.md)

Treat that file as the canonical local reference for official requirements and precedence.

## Workflow

1. Read the canonical requirements file before proposing changes.
2. If the task affects figures, inspect:
   - [figures/style.py](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/figures/style.py)
   - [figures/validate.py](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/figures/validate.py)
   - [figures/review.py](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/figures/review.py)
3. If the task affects manuscript sections, inspect:
   - [paper/manuscript/manuscript.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/manuscript/manuscript.md)
4. For paper-facing figure review, hand off to the `paper-asset-review` skill instead of doing an ad hoc single-agent review here.
5. When local code or older notes conflict with the canonical requirements file, surface the conflict explicitly and update both the implementation and the documentation if needed.
6. If the user asks for the latest Nature guidance, verify with official Nature / Nature Communications web pages before finalizing changes.

## Guardrails

- Do not treat [NATURE_FIGURE_GUIDELINES.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/NATURE_FIGURE_GUIDELINES.md) as authoritative.
- Keep manuscript-facing edits consistent with the repository's Nature Communications writing style in [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md).
- Prefer simplifying local rules when they encode stale assumptions from older Nature guidance.
- Let Codex multimodal review make the paper-level visual judgment; keep code support focused on preview generation, geometry extraction, and gating artifacts.
