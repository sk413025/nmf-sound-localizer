# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
3. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
4. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Route in five steps

1. Identify the task type:
   - manuscript or submission
   - paper asset review
   - experiment or results interpretation
   - orchestration or governance
   - runtime substrate maintenance
2. Choose one skill only:
   - `.codex/skills/paper-submission/SKILL.md`
   - `.codex/skills/paper-asset-review/SKILL.md`
   - `.codex/skills/experiment-results/SKILL.md`
   - `.codex/skills/agent-orchestrator/SKILL.md`
3. If the task touches `nmf_localizer/`, `doa_rl/`, `scripts/` outside `scripts/paper/`, `tests/`, or package metadata, open `docs/governance/runtime-substrate-contract.md`.
4. Open the matching section in `docs/agent-ops/TASK_PACKETS.md`.
5. If the task could shift claims, governance, or submission posture, route through the supervisor model first.

## Agent-first operating model

- Code is substrate. Read code when needed to support a paper task, not as the default starting point.
- Prefer a task packet and a skill before improvising a workflow.
- Use supervisor-led orchestration for tasks that affect manuscript claims, submission posture, or branch governance.
- Treat the human as an occasional approver unless the task packet says otherwise.

## Common commands

- `make paper-build`
- `make paper-check`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Do not use as source of truth

- older package-era README or CONTRIBUTING text from git history
- `NATURE_FIGURE_GUIDELINES.md` as an authoritative policy source
- archived notes under `docs/archive/`
