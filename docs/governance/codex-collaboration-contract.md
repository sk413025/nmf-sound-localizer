# Codex Collaboration Contract

Use this contract for Codex-native workflow, local skills, task routing, governance design, and machine-checkable collaboration rules.

## Applies to

- `AGENTS.md`
- `START_HERE_AGENT.md`
- project-local skills under `.codex/skills/`
- `docs/agent-ops/`
- executable governance checks under `scripts/paper/`

## Core principles

- Define `codex-native` from real repository primitives, not from taste alone.
- Prefer existing primitives such as `AGENTS.md`, local skills, task packets, and executable checks over parallel systems.
- Keep collaboration manuscript-first. Code, figures, and results remain substrate.
- Prefer simplification over accretion. A governance change must delete or collapse duplicated surface before it claims simplification.
- Do not let one concept acquire multiple canonical homes. One rule, one schema, and one status surface should each have one canonical home.
- Repo-local branch memory may summarize stable branch state, but it remains a derived layer. `.codex/memory/` must not create new policy, new canonical schemas, or a second governance vocabulary.
- Do not let the checker over-model reviewer judgment. Executable gates should enforce low-ambiguity consistency and ownership boundaries, not infer paper-level scientific verdicts.
- Keep paper-facing workflow claim-forward and reader-first. Closeout rigor belongs in routing, review, and verification, not in the paper's sentence tone.
- Keep governance and memory language positive and compressive. State the supported rule, posture, or constraint directly rather than storing anticipated objections or reviewer-prebuttal phrasing as if they were policy.
- When a governance or memory sentence feels too broad, narrow the rule or scope directly instead of rewriting it as `not X`, `should not be read as`, or other defensive explanation.
- For manuscript-routing and prose-hardening work, treat bidirectional term traceability as a required workflow check: agents should test `Introduction/Results -> Methods` and `Methods/Supplementary -> Introduction/Results` and reject rounds that still force readers to build a private translation map.
- For manuscript-routing and prose-hardening work, enforce single-head term propagation across manuscript, legends, `Methods`, and `Supplementary Methods`: one concept family should keep one canonical reader-facing name, with formal labels added as precision rather than as a second naming layer.
- In workflow terms, treat `Methods` / `Supplementary Methods` first-mention lock as part of the same check: when a formal surface is reopened, the first mention should reuse the canonical head term before notation or score construction appears.

## Canonical homes

- constitution and non-negotiables:
  - `AGENTS.md`
- top-level quickstart:
  - `START_HERE_AGENT.md`
- derived branch-memory brief:
  - `.codex/memory/CURRENT_BRANCH_MEMORY.md`
- canonical panel-to-method crosswalk for active main-paper figures:
  - `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`
- top-level execution, delegation, supervision, and unfamiliarity bootstrap:
  - `.codex/skills/agent-orchestrator/SKILL.md`
- task packet schema and per-task templates:
  - `docs/agent-ops/TASK_PACKETS.md`
- reviewer lenses for paper-facing critique:
  - `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- closeout truthfulness and owner separation:
  - `docs/governance/closeout-integrity-contract.md`
- only machine-readable field inventory for `high-risk` broader-significance rounds:
  - `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md`
- only machine-readable round artifact for those rounds:
  - `results/<round_name>/governance_round.yaml`

Do not restate operational detail from these homes in parallel documents. Point to the canonical home instead.
For active main-paper figures, panel-method traceability must live in `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`, not in branch memory, ad hoc working notes, or duplicated contract tables.

## Anti-duplication rule

Before adding a new field, verdict, artifact, or workflow branch, answer all four:

1. why existing primitives were insufficient
2. what duplicated surface will be removed
3. what remains canonical after the change
4. what complexity risk stays open

If those answers are weak, simplify or reuse instead of expanding the system.

Default complexity symptoms:

- duplicated schema or field inventory
- derived state presented as canonical machine state
- operational instructions duplicated across constitution, quickstart, skill, and packet docs
- checker logic standing in for reviewer judgment
- a tutorial or worked example starting to act like a second contract

## High-risk broader-significance rounds

For any `high-risk` round with broader significance or cross-disciplinary consequence in scope:

- use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md` as the only machine-readable field inventory
- create `results/<round_name>/governance_round.yaml`
- treat `governance_round.yaml` as the blocking machine-readable source of truth for promotion, demotion, verification, and closeout-ready status
- require `make paper-governance-gate ROUND_DIR=results/<round_name>` before reporting that broader significance landed

## Executable gates

- `python scripts/paper/check_governance_links.py`
- `python scripts/paper/check_round_governance_semantics.py --round-dir results/<round_name>`
- `make paper-governance-gate ROUND_DIR=results/<round_name>`
- `make paper-check`
