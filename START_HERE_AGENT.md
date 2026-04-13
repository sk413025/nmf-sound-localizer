# Start Here for Agents

This worktree is a manuscript-first Nature Communications branch. Do not treat it as a generic package repo or a figure-only sandbox.

## Read in this order

1. [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md)
2. [.codex/memory/CURRENT_BRANCH_MEMORY.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/memory/CURRENT_BRANCH_MEMORY.md)
3. [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
4. [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
5. [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
6. [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

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

- Before drafting or reviewing paper-facing prose, classify `Architecture scope`:
  - `local-salience` for a local high-salience rewrite with no section reweighting
  - `cross-section` for a round that changes more than one section, section bridges, or discovery-versus-tool weight
  - `whole-manuscript` for a full-paper restructuring or any rewrite that re-architects the Results spine
- Write paper-facing explanation to make the paper's actual discovery legible to an editor on first read.
- Before drafting, identify the paper's `old-world belief`, `new-world belief`, and one stable `paper protagonist`.
- Default the protagonist to the phenomenon or organizing principle, not to the solver, calibration procedure, or reference object.
- Identify the paper pivot before revising Results flow. If no section clearly updates the reader's model of the system, the story is still under-architected.
- For any `cross-section` or `whole-manuscript` rewrite, write a `Paper spine map` before editing. Include the Results section jobs, the pivot sentence, the discovery cash-out section, and the discovery-versus-tool weight budget.
- Keep tool sections subordinate to discovery. A solver or assay may reveal the phenomenon, but it should not occupy more narrative weight than the paper-level finding.
- Do not narrate the paper in experiment time order. Default to `old intuition -> surprising observation -> governing principle -> broader implication`.
- If the Results outline still sounds like `and then we tested`, the spine is still wrong even if each paragraph is locally clear.
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
- For title, abstract, Results framing, and Discussion lead, also use the architecture exemplars there to keep protagonist, pivot, and worldview shift explicit.
- For whole-manuscript architecture work, do not stop at sentence polish. Check where the old-world belief breaks, where the pivot lands, where the discovery cashes out, and whether any tool section has taken more mass than the finding it serves.

## Agent-first operating model

- Code is substrate. Read code when needed to support a paper task, not as the default starting point.
- For manuscript, governance, strategy, and other branch-shaping tasks, read the current branch memory brief first and expand into archive notes only when the brief points there.
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
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

For any `high-risk` round with broader significance or cross-disciplinary consequence in scope, also create:

- `results/<round_name>/governance_round.yaml`

and treat `make paper-governance-gate ROUND_DIR=results/<round_name>` as the blocking semantic gate for promotion, demotion, and closeout coherence.

## Do not use as source of truth

- older package-era README or CONTRIBUTING text from git history
- `NATURE_FIGURE_GUIDELINES.md` as an authoritative policy source
- archived notes under `docs/archive/`
