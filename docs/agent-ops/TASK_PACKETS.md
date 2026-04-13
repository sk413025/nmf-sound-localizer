# Task Packets

Use these templates to turn a high-level request into a bounded delegated task, or into a checklist for a direct round that still needs explicit ownership and acceptance surfaces.

This file owns the packet schema. It does not restate the full supervisor workflow. Top-level routing, unfamiliarity bootstrap, and child supervision live in `.codex/skills/agent-orchestrator/SKILL.md`.

## Required fields

Every packet should include:

- `Objective`
- `Relevant conversation context`
- `Source of truth`
- `Constraints`
- `Risk level`
- `Architecture scope`
- `Acceptance surface`
- `Out-of-scope surfaces`
- `Plan items owned`
- `Delivery evidence required`
- `Review owner`
- `Verification owner`
- `Verification target`
- `Scope downgrade rule`
- `Expected outputs`
- `Escalate when`
- `Context mode`

## Field notes

- `Relevant conversation context`
  - summarize only task-relevant history, confirmed decisions, constraints, and unresolved risks
- `Source of truth`
  - name the files, artifacts, or runtime surfaces the task is allowed to rely on
- `Constraints`
  - name what must be preserved, what must be inspected first, and any epistemic action required before mutation on unfamiliar work
- `Risk level`
  - use `high-risk` or `non-high-risk`
- `Architecture scope`
  - use `local-salience`, `cross-section`, or `whole-manuscript`
- `Delivery evidence required`
  - name the exact text, file, artifact, visual inspection, or command evidence needed before claiming completion
- `Context mode`
  - `summary-only` by default; use `summary+fork_context` only when exact wording history matters

Use packet fields to record decisions, not to create a second schema or a second supervision checklist.

## Unfamiliar or runtime-touching work

For unfamiliar tasks, runtime-touching tasks, or artifact lineages that are not already clear, the packet should explicitly record:

- which environment surface was inspected first
- which unknowns still remain open
- which epistemic action must happen before mutation if uncertainty is still high

Do this inside existing fields. Do not add a new unfamiliarity field.

## Paper-facing additions

For any packet whose outputs may be promoted into manuscript, supplementary, legend, caption, review-note, or analysis-summary prose, also include:

- `Claim floor`
- `Claim ceiling`
- `Evidence boundary`
- `Editor readout sentence`
- `Relevant voice exemplar(s)`
- `Bridge question`

For any local high-salience, `cross-section`, or `whole-manuscript` manuscript round, carry the architecture bundle required by:

- `docs/governance/manuscript-contract.md`
- `docs/governance/scientific-voice-guide.md`

Do not recreate that bundle here.

## High-risk broader-significance rounds

For any `high-risk` round with broader significance or cross-disciplinary consequence in scope:

- create `results/<round_name>/governance_round.yaml`
- use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md` as the only machine-readable field inventory
- include a passing `make paper-governance-gate ROUND_DIR=results/<round_name>` run in `Delivery evidence required`

## Governance-changing rounds

For governance-changing rounds, also record:

- `Complexity risk`
- `Why existing primitives were insufficient`
- `What duplicated surface was removed`
- `What remains canonical after this round`

## Manuscript and submission packet

- Role:
  - `manuscript-reviser`, `claim-auditor`, or `submission-auditor`
- Skill:
  - `paper-submission`
- Source of truth:
  - `paper/manuscript/manuscript.md`
  - `docs/governance/manuscript-contract.md`
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/nature-communications-submission-requirements.md`
- Constraints:
  - preserve claim and evidence integrity
  - inspect neighboring paragraphs and section logic before treating a local rewrite as complete
  - record visual inspection plus generator and provenance backtrace for any figure-dependent interpretation
- Delivery evidence required:
  - revised text or audit findings
  - anchored file locations
  - figure/provenance notes when relevant
- Expected outputs:
  - revised text, audit findings, or compliance gaps
  - the paper-facing additions listed above
- Escalate when:
  - the change could alter scientific interpretation
  - main-versus-supplement placement may change
  - Nature-facing compliance conflicts with current manuscript structure

## Experiment and results packet

- Role:
  - `experiment-results-analyst`
- Skill:
  - `experiment-results`
- Source of truth:
  - `docs/governance/experiment-contract.md`
  - `results/<run_name>/`
  - `DATA_PROVENANCE.md`
- Constraints:
  - ground conclusions in committed logs, metrics, and provenance artifacts
  - keep reproducibility expectations explicit
- Delivery evidence required:
  - artifact paths
  - logs and metrics
  - reproduction commands when needed
- Expected outputs:
  - artifact-grounded interpretation
  - reproduction surface
  - figure-facing backtrace when a run artifact supports paper evidence
- Escalate when:
  - logs, metrics, or provenance are missing
  - the run is not reproducible from committed artifacts
  - a paper-facing claim would rely on weak or partial evidence

## Paper asset review packet

- Role:
  - `paper-asset-reviewer`
- Skill:
  - `paper-asset-review`
- Source of truth:
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/paper-asset-review-workflow.md`
  - `figures/review_artifacts/<figure_id>/`
- Constraints:
  - inspect the real visual asset
  - check generator and evidence layers when the asset is generated or data-backed
- Delivery evidence required:
  - visual inspection confirmation
  - generator, provenance, and review-gate artifacts when relevant
- Expected outputs:
  - keep, revise, split, or move recommendation
  - manuscript-fit justification
  - explicit confirmation that visual and provenance checks were performed
- Escalate when:
  - visual content, generator code, and provenance disagree
  - the recommendation implies main-versus-supplement reclassification

## Governance and orchestration packet

- Role:
  - `supervisor` or `red-team-reviewer`
- Skill:
  - `agent-orchestrator`
- Source of truth:
  - `docs/governance/codex-collaboration-contract.md`
  - `docs/governance/closeout-integrity-contract.md`
  - `docs/agent-ops/REVIEW_AND_ESCALATION.md`
  - `docs/agent-ops/ROUND_CLOSEOUT_TEMPLATE.md`
- Constraints:
  - keep the task bounded to the named workflow or governance surface
  - record the owned acceptance surface and ownership boundaries explicitly
  - use the unfamiliarity bootstrap when the routed surface is not already familiar
- Delivery evidence required:
  - packet text, routing decisions, warning disposition, and any cited artifact or status checks needed for closeout
- Expected outputs:
  - task framing
  - routing or decomposition decisions
  - warnings, rewrites, or milestone summary
- Escalate when:
  - the proposal adds a new governance layer or duplicate skill
  - the workflow shifts toward code-first operation
  - a milestone-level decision requires human approval
