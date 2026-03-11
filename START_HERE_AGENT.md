# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
3. [paper/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/paper/README.md)
4. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
5. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Task routing

- Manuscript or submission task:
  - Read `docs/nature-communications/nature-communications-submission-requirements.md`
  - Use `.codex/skills/nature-communications-submission/SKILL.md`
- Manuscript revision task:
  - Use `.codex/skills/manuscript-revision/SKILL.md`
  - Start from `docs/agent-ops/task-packets/manuscript-revision-packet.md`
- Claim or evidence audit:
  - Use `.codex/skills/claim-evidence-audit/SKILL.md`
  - Start from `docs/agent-ops/task-packets/claim-audit-packet.md`
- Results interpretation:
  - Use `.codex/skills/results-interpretation/SKILL.md`
  - Start from `docs/agent-ops/task-packets/results-interpretation-packet.md`
- Paper-facing asset review:
  - Use `.codex/skills/paper-asset-review/SKILL.md`
  - Run `python scripts/paper/review_paper_assets.py prepare`
- Codex-native or multi-agent governance task:
  - Use `.codex/skills/agent-orchestrator/SKILL.md` or `.codex/skills/codex-native-assessment/SKILL.md`
  - Read `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
- Experiment or results task:
  - Follow `docs/governance/experiment-contract.md`

## Agent-first operating model

- Code is substrate. Read code when needed to support a paper task, not as the default starting point.
- Prefer a role packet and a skill before improvising a workflow.
- Use supervisor-led orchestration for tasks that affect manuscript claims, submission posture, or branch governance.
- Treat the human as an occasional approver unless the task packet says otherwise.

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
