# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
3. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
4. [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
5. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Route in six steps

1. Identify the task type:
   - manuscript or submission
   - paper asset review
   - experiment or results interpretation
   - orchestration or governance
   - runtime substrate maintenance
2. At top level, start with `.codex/skills/agent-orchestrator/SKILL.md`.
3. Use that top-level routing step to decide whether to execute directly or delegate. If delegating, choose the smallest specialist skill set that fits:
   - `.codex/skills/paper-submission/SKILL.md`
   - `.codex/skills/paper-asset-review/SKILL.md`
   - `.codex/skills/experiment-results/SKILL.md`
4. If the task touches `nmf_localizer/`, `doa_rl/`, `scripts/` outside `scripts/paper/`, `tests/`, or package metadata, open `docs/governance/runtime-substrate-contract.md` and assume only TF + USM + soft-OMP support is active unless proven otherwise.
5. Open the matching section in `docs/agent-ops/TASK_PACKETS.md`.
6. If the task is paper-facing hardening or could shift claims, governance, or submission posture, open `docs/agent-ops/NATURE_REVIEWER_STACK.md`, select the applicable reviewer roles and evaluation goals, and stay in the supervisor model for routing.

## Scientific voice default

- Write paper-facing explanation to make the paper's actual discovery legible to an editor on first read.
- Lead with the strongest supported claim, then add the evidence boundary.
- Do not import closeout, verifier, or governance caution language into paper-facing sentences.
- Avoid the false tradeoff `rigorous = timid`. In this branch, rigor means precise support, not self-erasure.
- If a sentence sounds safer only because it lowers the claim floor, rewrite it.
- Use this salience order by default: `discovery -> evidence -> implication -> boundary`.
- Use this sentence order by default: `clear subject -> strong verb -> explicit consequence`.
- If one sentence carries more than one major causal relation, split it.
- Keep noun stacks short enough that a broad scientific reader can parse the sentence in one pass.
- Translate important numbers into meaning. If the sentence gives a rise, drop, or contrast, also state what that change means.
- Use lab-meeting English for any paper-facing explanation prose, not only the main manuscript. If a strong PhD student would not naturally say the sentence aloud, rewrite it.
- Before touching high-salience prose or any paper-facing explanation surface, read the canonical examples in `docs/governance/scientific-voice-guide.md` and use the closest `SV#` exemplar as the rewrite target.

## Agent-first operating model

- Code is substrate. Read code when needed to support a paper task, not as the default starting point.
- Prefer a task packet and a skill before improvising a workflow.
- The top-level agent routes through orchestration first, then may execute directly or delegate.
- Delegate even a bounded task only when delegation improves scope control, review separation, or execution safety.
- If delegation is chosen, use a single child only when the request is genuinely single-scope; otherwise split it into multiple child tasks before execution starts.
- Use supervisor-led orchestration for tasks that affect manuscript claims, submission posture, or branch governance.
- For paper-facing review and hardening, use the canonical reviewer stack in `docs/agent-ops/NATURE_REVIEWER_STACK.md` instead of inventing review personas ad hoc.
- Before spawning a child agent, write a task packet with `Relevant conversation context`.
- After spawning a child agent, monitor it until completion, explicit redirect, or a justified shutdown.
- Inspect a child agent's current status or latest output before interrupting or closing it.
- Do not close a child agent just because it feels slow.
- Context mode: `summary-only` by default; switch to `summary+fork_context` only when task-relevant dialogue history cannot be safely compressed.
- In this repository's default operating mode, treat the human as providing standing authorization for sub-agent use and let the top-level agent decide when delegation is needed.
- Apply this execution-or-delegation policy in both Default mode and Plan mode.
- In Plan mode, both direct and delegated work must stay non-mutating and limited to planning, exploration, checking, or review.
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
