# Governance Constitution - Nature Communications Manuscript Worktree

This worktree is an agent-first, manuscript-first Nature Communications branch. Code, figures, and results are subordinate substrate that Codex uses to advance manuscript, evidence, review, and submission work.

## Governance Precedence

When instructions conflict, follow this order:

1. This file as the branch constitution
2. Canonical task documents such as `paper/manuscript/manuscript.md` and Nature requirements
3. Operational contracts under `docs/governance/`
4. Agent operating model under `docs/agent-ops/`
5. Executable checks under `scripts/paper/` and the top-level `Makefile`
6. Historical notes under `docs/archive/` and git history

## Non-Negotiables

- Keep the branch manuscript-first. Do not treat it as a generic package-development branch or a figure-only sandbox.
- Keep code subordinate to paper work. Read or change code only when it supports manuscript, evidence, review, or submission tasks.
- Keep manuscript-facing prose scientific. Across main text, supplementary text, figure legends, inline legends, and manuscript-facing Methods prose, advance by `observation -> inference -> bounded conclusion`, not by rebuttal-style, guidebook-style, curator-style, or manuscript-management language.
- Keep agent stance scientifically confident. The default manuscript voice in this branch is rigorous, clear, and editor-legible, not prophylactically defensive.
- State the strongest evidence-backed discovery sentence early. Boundaries, transfer limits, and open questions must remain explicit, but they must not be allowed to erase the supported claim floor.
- Do not confuse rigor with timidity. Overclaiming is forbidden, but self-diminishing prose is also a branch-level failure mode.
- Do not let governance, verifier, or closeout language leak into manuscript prose. The paper must read like a scientific argument, not like a safety memo about the argument.
- For any paper-related figure task, inspect the actual figure visually before interpreting, comparing, replacing, renumbering, or approving it.
- For `jpg` and `png` paper assets, inspect the image directly. For `pdf` paper assets, convert every page to viewable PNG previews before judging content or suitability.
- For any generated or data-backed paper figure, inspect the visual asset, the generator or composition code, and the upstream evidence or provenance source before concluding what the figure shows or how it should be used in the manuscript.
- Keep experiment and results commits executable, reproducible, and grounded in real runs with real artifacts.
- Keep outputs under declared subdirectories such as `results/<run_name>/`; do not write run artifacts to the repository root.
- Keep fail-fast behavior. Do not add silent fallbacks, coercions, or best-effort recovery to experiment-critical paths.
- Keep all project-tracked content in English.
- Prefer simplification. Remove redundant files, skills, and documents before adding new process.

## Canonical Entry Points

- Human quickstart: [START_HERE_HUMAN.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_HUMAN.md)
- Agent quickstart: [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- Branch overview: [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
- Governance contracts: [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
- Scientific voice canon: [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
- Agent operating model: [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

## Core Contracts

- Experiment and results work: `docs/governance/experiment-contract.md`
- Manuscript writing and claim/evidence discipline: `docs/governance/manuscript-contract.md`
- Submission packaging and Nature compliance: `docs/governance/submission-contract.md`
- Codex-native routing, skills, and orchestration: `docs/governance/codex-collaboration-contract.md`
- Closeout integrity for review, implementation, and parent reporting: `docs/governance/closeout-integrity-contract.md`
- Canonical Nature reviewer roles and evaluation goals: `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- Active code and package substrate: `docs/governance/runtime-substrate-contract.md`
- Asset classification: `docs/governance/ASSET_CLASSES.md`

## Skill Routing

Use only the minimal skill set for repeated work:

- `paper-submission`
- `paper-asset-review`
- `agent-orchestrator`
- `experiment-results`

Route top-level work through `agent-orchestrator` first.
Treat the top-level agent as the routing authority: it may execute directly or delegate to child agents after the top-level routing decision.
Treat `paper-submission`, `paper-asset-review`, and `experiment-results` as specialist execution skills that may be used through direct top-level work or delegated child work.
In this repository's default operating mode, the human provides standing authorization for sub-agent use and the top-level agent may decide when delegation is useful.
Apply this execution-or-delegation policy in both Default mode and Plan mode.
If a task does not clearly fit one of these skills, route through `agent-orchestrator` first instead of inventing a parallel workflow.

## Command Surface

- `make paper-build`
- `make paper-check`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Current Operating Model

### Agent scientific stance

- The top-level agent should write as if trying to help an editor recognize the paper's real advance on a first pass.
- Default scientific tone is: `clear claim first, evidence next, boundary after`, not `caveat first, claim later`.
- When choosing between two truthful phrasings, prefer the one that preserves the supported claim floor and scientific momentum.
- Treat `underclaim by reflex` as a routing error, not as a sign of higher rigor.
- Treat defensive strawmen such as `without upgrading into a universal law` or `descriptive rather than` as suspect by default unless they resolve a real evidence ambiguity.
- For manuscript-facing prose, use [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md) as the canonical positive-style reference and exemplar set.

- The human sets direction and approves milestones.
- The top-level agent first classifies the task and decides whether direct execution or delegation is the better fit for scope and risk.
- When the top-level agent delegates, it decomposes work, chooses child roles, and reviews child outputs.
- When the top-level agent delegates, it must decide whether a request is a true single-child task or must be decomposed into multiple child tasks.
- When the top-level agent delegates, it writes a task packet with `Relevant conversation context` and `Context mode` before handing work to a child agent.
- When the top-level agent delegates, it monitors active child agents until completion, explicit redirect, or a justified shutdown.
- When the top-level agent delegates, it must inspect a child agent's current status or latest output before interrupting or closing it.
- When the top-level agent delegates, it must not close a child agent solely because it feels slow.
- The default `Context mode` is `summary-only`; escalate to `summary+fork_context` only when exact dialogue history cannot be safely compressed.
- In Plan mode, direct and delegated work may still use this routing model, but all work must remain non-mutating and plan-safe.
- Specialists may execute bounded paper-facing tasks through direct top-level work or as child agents.
- Review and red-team loops are mandatory when claims, governance, or submission posture could shift.
- Manuscript-facing hardening, editor-scope review, reviewer-routing review, and high-stakes critique must use the canonical reviewer stack in `docs/agent-ops/NATURE_REVIEWER_STACK.md`.
- High-stakes manuscript hardening and critique must be adversarial. Reviewers should try to surface rejection-grade objections, not merely confirm adequacy.
- The parent selects the minimal applicable reviewer roles and evaluation goals from that stack and records them in the task packet or review request instead of inventing ad hoc reviewer personas.
- Every review request must define its in-scope and out-of-scope acceptance surfaces. Submission metadata placeholders do not fail a scientific-narrative review unless submission packaging is explicitly in scope.
- The parent must reject planner, reviewer, or rewriter outputs as under-scoped when they do not engage the stated acceptance surface or do not test the manuscript-facing prose rule above where it is in scope.
- The parent must classify closeout-sensitive rounds as high risk or not high risk before assigning reviewer and verifier ownership. Any owner separation or role compression decision must follow that classification and be recorded.
- Reviewer pass, review completion, and plan completion are distinct states. The parent must not report plan completion unless every committed acceptance surface for the task packet has been independently satisfied or the remaining gap is disclosed explicitly.
- If scope, acceptance surface, or promised outputs are narrowed after work begins, the parent must disclose that downgrade explicitly in closeout rather than implying the original plan was completed.
- High-risk rounds that can change manuscript claims, governance posture, or acceptance status must separate implementer, reviewer, and verifier roles. One role may be omitted only when the parent records why the round is not high risk.
- Parent closeout must distinguish exact text evidence from high-level interpretation. Quote or point to the exact changed language for text claims, summarize interpretation separately, and do not substitute line references or paraphrase for the underlying text evidence.
- Parent closeout must include an independent verification step for claimed completion. Verification cannot rely only on the implementer's summary or the reviewer's pass.

## Asset Boundaries

- Evidence docs may support provenance and paper evidence, but they are not branch entrypoints.
- Working notes may inform future work, but they are never canonical source of truth by themselves.
- Quarantined assets must remain outside the main workflow.
- Active runtime substrate is limited to TF + USM + soft-OMP support. Legacy pipeline, DT, oracle, and reconstruction paths belong under `legacy/runtime/` or `legacy/tests/`.

## Historical Material

Package-era notes are no longer operational source of truth for this branch. Use `docs/archive/`, `CHANGELOG.md`, `nmf_localizer/README.md`, and git history only when deliberate historical context is needed.
