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

Use only the minimal repeated skill set:

- `agent-orchestrator`
- `paper-submission`
- `paper-asset-review`
- `experiment-results`

Route top-level work through `agent-orchestrator` first.
Treat the top-level agent as the routing authority: it may execute directly or delegate after the top-level routing decision.
Treat the specialist skills as on-demand execution surfaces, not mandatory first-pass reads.
If a task does not clearly fit one of these skills, route through `agent-orchestrator` first instead of inventing a parallel workflow.

## Agent Operating Model

- The top-level agent should write as if helping an editor recognize the paper's actual advance on first pass.
- For manuscript, governance, strategy, and other branch-shaping work, read the current branch memory brief before deeper routing.
- Use the smallest top-level surface that fits: constitution, branch memory, then `agent-orchestrator`.
- Treat unfamiliarity as a reason to inspect the next missing source-of-truth surface, not as a reason to improvise or freeze.
- Keep paper-facing prose claim-forward: `clear claim first, evidence next, boundary after`.
- Route manuscript hardening, editorial critique, and reviewer-routing critique through the canonical reviewer stack in `docs/agent-ops/NATURE_REVIEWER_STACK.md`.
- Treat review, verification, and plan completion as separate decisions.
- For `high-risk` rounds with broader significance or cross-disciplinary consequence in scope, require `results/<round_name>/governance_round.yaml` plus `make paper-governance-gate ROUND_DIR=results/<round_name>` before closeout may report that broader significance landed.
- Parent closeout must distinguish exact text evidence from high-level interpretation and must not imply full completion after a scope downgrade.

## Command Surface

- `make paper-build`
- `make paper-check`
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
- `make manuscript`
- `make paper-review-assets`
- `make paper-review-gate`

## Asset Boundaries

- Evidence docs may support provenance and paper evidence, but they are not branch entrypoints.
- Working notes may inform future work, but they are never canonical source of truth by themselves.
- Quarantined assets must remain outside the main workflow.
- Active runtime substrate is limited to TF + USM + soft-OMP support. Legacy pipeline, DT, oracle, and reconstruction paths belong under `legacy/runtime/` or `legacy/tests/`.

## Historical Material

Package-era notes are no longer operational source of truth for this branch. Use `docs/archive/`, `CHANGELOG.md`, `nmf_localizer/README.md`, and git history only when deliberate historical context is needed.
