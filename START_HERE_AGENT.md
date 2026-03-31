# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
3. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
4. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Route in six steps

1. Identify the task type:
   - manuscript or submission
   - paper asset review
   - experiment or results interpretation
   - orchestration or governance
   - runtime substrate maintenance
2. At top level, start with `.codex/skills/agent-orchestrator/SKILL.md`.
3. Let the parent orchestrator choose exactly one specialist skill for each child worker:
   - `.codex/skills/paper-submission/SKILL.md`
   - `.codex/skills/paper-asset-review/SKILL.md`
   - `.codex/skills/experiment-results/SKILL.md`
4. If the task touches `nmf_localizer/`, `doa_rl/`, `scripts/` outside `scripts/paper/`, `tests/`, or package metadata, open `docs/governance/runtime-substrate-contract.md` and assume only TF + USM + soft-OMP support is active unless proven otherwise.
5. Open the matching section in `docs/agent-ops/TASK_PACKETS.md`.
6. If the task could shift claims, governance, or submission posture, stay in the supervisor model and hand the execution step to a child worker.

## Agent-first operating model

- Code is substrate. Read code when needed to support a paper task, not as the default starting point.
- Prefer a task packet and a skill before improvising a workflow.
- The top-level agent is the parent orchestrator, not a worker.
- Even a single bounded task should be handed to at least one child agent.
- Use supervisor-led orchestration for tasks that affect manuscript claims, submission posture, or branch governance.
- Before spawning a child agent, write a task packet with `Relevant conversation context`.
- Context mode: `summary-only` by default; switch to `summary+fork_context` only when task-relevant dialogue history cannot be safely compressed.
- In this repository's default operating mode, treat the human as providing standing authorization for sub-agent use and let the parent decide when child agents are needed.
- Apply this parent-orchestrator policy in both Default mode and Plan mode.
- In Plan mode, delegated child work must stay non-mutating and limited to planning, exploration, checking, or review.
- Treat the human as an occasional approver unless the task packet says otherwise.
- Before making any paper-figure judgment, visually inspect the actual figure asset. If the asset is a PDF, convert every page to PNG previews first.
- For generated or data-backed paper figures, trace the figure through its generator or composition code and upstream evidence sources before deciding panel identity, lineage, claim support, or Nature suitability.

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
