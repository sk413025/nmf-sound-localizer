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
- Keep paper-facing explanation scientific. Across main text, supplementary text, figure legends, inline legends, captions, review-note prose, analysis summaries that may flow into the paper, and main-manuscript Methods prose, advance by `observation -> inference -> bounded conclusion`, not by rebuttal-style, guidebook-style, curator-style, or manuscript-management language.
- Keep agent stance scientifically confident. The default paper voice in this branch is rigorous, clear, and editor-legible, not prophylactically defensive.
- State the strongest evidence-backed discovery sentence early. Boundaries, transfer limits, and open questions must remain explicit, but they must not be allowed to erase the supported claim floor.
- Do not confuse rigor with timidity. Overclaiming is forbidden, but self-diminishing prose is also a branch-level failure mode.
- Keep paper-facing explanation architected around one cognitive shift. A Nature Communications paper in this branch must move the reader from an `old-world belief` to a `new-world belief`, not merely report a sequence of technically correct observations.
- Classify `Architecture scope` before paper-facing hardening begins. Use `local-salience` only for local high-salience rewrites with no section reweighting, no section-bridge changes, and no Results-spine changes; use `cross-section` for rounds that change more than one section, section bridges, or discovery-versus-tool weight; use `whole-manuscript` for full-paper restructuring or any round that re-architects the Results spine.
- Keep one stable paper protagonist. The default protagonist is the discovery, organizing principle, or physical phenomenon. Reference objects, calibration schemes, solvers, and analysis blocks are supporting actors unless the packet explicitly declares a method paper.
- Keep one explicit paper pivot. At least one Results transition must function as the turning point that updates the reader's model of the system, not just the next item in an experiment log.
- Keep discovery weight higher than tool-validation weight. Tool sections may reveal, test, or preserve the discovery, but they must not carry more narrative mass than the paper-level finding they support.
- Keep one explicit whole-paper spine map for any cross-section or whole-manuscript hardening round. The map must name the old-world belief, new-world belief, protagonist, supporting actors, Results section jobs, pivot, discovery cash-out section, tool role, and discovery-versus-tool weight budget before prose is treated as architected.
- Keep architecture artifacts phase-correct. `Paper spine map` belongs to drafting-time packets and rewrite planning; `Architecture evidence map` belongs to closeout-time verification when a cross-section or whole-manuscript round claims architecture landed.
- Do not write in experiment time order. Default to `old intuition -> surprising observation -> governing principle -> broader implication`, even when that differs from the order in which the work was done.
- Treat architecture as part of rigor. A paper that is locally precise but narratively mis-centered is still scientifically under-explained.
- Keep paper-facing explanation reader-first at the sentence level. A scientifically literate generalist should not have to decode syntax, noun stacks, or governance-shaped abstractions before seeing the science.
- Default to `clear subject -> strong verb -> explicit consequence`. If a sentence carries more than one main causal move, split it.
- Prefer spoken-natural scientific English over compressed formal register. If a strong PhD student would not say the sentence aloud in lab meeting, treat that as a rewrite trigger.
- Translate numbers and comparisons into consequence. When a sentence reports a strong rise, drop, separation, or contrast, state what that change means for the scientific claim.
- Do not let governance, verifier, or closeout language leak into paper-facing explanation. The paper must read like a scientific argument, not like a safety memo about the argument.
- For any paper-related figure task, inspect the actual figure visually before interpreting, comparing, replacing, renumbering, or approving it.
- For `jpg` and `png` paper assets, inspect the image directly. For `pdf` paper assets, convert every page to viewable PNG previews before judging content or suitability.
- For any generated or data-backed paper figure, inspect the visual asset, the generator or composition code, and the upstream evidence or provenance source before concluding what the figure shows or how it should be used in the manuscript.
- Keep experiment and results commits executable, reproducible, and grounded in real runs with real artifacts.
- Keep outputs under declared subdirectories such as `results/<run_name>/`; do not write run artifacts to the repository root.
- Keep fail-fast behavior. Do not add silent fallbacks, coercions, or best-effort recovery to experiment-critical paths.
- Keep all project-tracked content in English.
- Prefer simplification. Remove redundant files, skills, and documents before adding new process.
- Do not create a second governance vocabulary when one canonical artifact or schema already exists.
- Treat governance complexity as a branch-level failure mode. New fields, verdicts, or artifacts must justify why existing primitives are insufficient and what duplication will be removed in exchange.

## Canonical Entry Points

- Human quickstart: [START_HERE_HUMAN.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_HUMAN.md)
- Agent quickstart: [START_HERE_AGENT.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/START_HERE_AGENT.md)
- Current branch memory brief: [.codex/memory/CURRENT_BRANCH_MEMORY.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/memory/CURRENT_BRANCH_MEMORY.md)
- Branch overview: [README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/README.md)
- Governance contracts: [docs/governance/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/README.md)
- Scientific voice canon: [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md)
- Agent operating model: [docs/agent-ops/README.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/agent-ops/README.md)

Treat `.codex/memory/` as the derived branch-memory layer, not as a second constitution. It may summarize stable branch state and session-derived lessons, but it must not override `AGENTS.md`, manuscript text, or canonical governance contracts.

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
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Current Operating Model

### Agent scientific stance

- The top-level agent should write as if trying to help an editor recognize the paper's real advance on a first pass.
- For manuscript, governance, strategy, or other branch-shaping work, the top-level agent should read the current branch memory brief before deeper routing so recent stable lessons are loaded without reopening raw sessions.
- The top-level agent should first ask what old belief the paper is replacing and what new belief the reader should leave with.
- The top-level agent should identify the paper protagonist before revising prose. If the protagonist silently shifts between phenomenon, method, reference object, and workflow, treat that as a drafting failure.
- The top-level agent should identify the paper pivot before drafting or approving a Results sequence. If every section speaks at the same narrative volume, treat that as a story-architecture failure.
- The top-level agent should write down the Results section jobs before restructuring or approving a whole-paper rewrite. If the section list still reads like experiment chronology or analysis-bundle order, treat that as an architecture failure.
- The top-level agent should keep reference objects and solvers subordinate to the discovery unless the packet explicitly declares a method-first paper.
- The top-level agent should identify where the paper actually cashes out its discovery. If the discovery appears only as a final extension after long tool-validation buildup, treat that as narrative mis-centering.
- Default scientific tone is: `clear claim first, evidence next, boundary after`, not `caveat first, claim later`.
- When choosing between two truthful phrasings, prefer the one that preserves the supported claim floor and scientific momentum.
- Treat `underclaim by reflex` as a routing error, not as a sign of higher rigor.
- Treat `reader must decode the sentence before seeing the claim` as a drafting failure, not as acceptable technical density.
- Use one main causal move per sentence by default. When a sentence reports both the measurement and its interpretation, make the link explicit with direct causal language or split the sentence.
- Keep technical noun stacks short unless the phrase is an established term of art. Prefer clauses, verbs, and prepositional phrases over front-loaded label chains.
- Prefer dynamic scientific verbs such as `concentrates`, `exceeds`, `holds`, `tracks`, `reveals`, and `limits` over static filler verbs such as `remains` or `stays` when the stronger verb preserves the truth value.
- Treat defensive strawmen such as `without upgrading into a universal law` or `descriptive rather than` as suspect by default unless they resolve a real evidence ambiguity.
- For paper-facing explanation, use [docs/governance/scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md) as the canonical positive-style reference and exemplar set.

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
- The parent must reject planner, reviewer, or rewriter outputs as under-scoped when they do not engage the stated acceptance surface or do not test the paper-facing explanation rule above where it is in scope.
- The parent must also reject paper-facing planning or review as under-scoped when it does not identify the `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, and `tool role` for high-salience manuscript rounds.
- The parent must reject whole-manuscript or cross-section hardening as under-architected when it lacks a `Paper spine map`, `Results section jobs`, `Discovery-vs-tool weight budget`, or a named `Discovery cash-out section`.
- The parent must classify closeout-sensitive rounds as high risk or not high risk before assigning reviewer and verifier ownership. Any owner separation or role compression decision must follow that classification and be recorded.
- High-risk rounds with broader significance or cross-disciplinary consequence in scope must create `results/<round_name>/governance_round.yaml` and pass `make paper-governance-gate ROUND_DIR=results/<round_name>` before parent closeout may report that broader significance landed.
- Reviewer pass, review completion, and plan completion are distinct states. The parent must not report plan completion unless every committed acceptance surface for the task packet has been independently satisfied or the remaining gap is disclosed explicitly.
- If scope, acceptance surface, or promised outputs are narrowed after work begins, the parent must disclose that downgrade explicitly in closeout rather than implying the original plan was completed.
- High-risk rounds that can change manuscript claims, governance posture, or acceptance status must separate implementer, reviewer, and verifier roles. One role may be omitted only when the parent records why the round is not high risk.
- Governance-changing rounds must name their complexity risk, why existing primitives were insufficient, what duplicated surface was removed, and what remains canonical after the round.
- Parent closeout must distinguish exact text evidence from high-level interpretation. Quote or point to the exact changed language for text claims, summarize interpretation separately, and do not substitute line references or paraphrase for the underlying text evidence.
- Parent closeout must include an independent verification step for claimed completion. Verification cannot rely only on the implementer's summary or the reviewer's pass.

## Asset Boundaries

- Evidence docs may support provenance and paper evidence, but they are not branch entrypoints.
- Working notes may inform future work, but they are never canonical source of truth by themselves.
- Quarantined assets must remain outside the main workflow.
- Active runtime substrate is limited to TF + USM + soft-OMP support. Legacy pipeline, DT, oracle, and reconstruction paths belong under `legacy/runtime/` or `legacy/tests/`.

## Historical Material

Package-era notes are no longer operational source of truth for this branch. Use `docs/archive/`, `CHANGELOG.md`, `nmf_localizer/README.md`, and git history only when deliberate historical context is needed.
