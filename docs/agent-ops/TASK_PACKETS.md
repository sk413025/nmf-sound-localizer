# Task Packets

Use these packet templates to turn a high-level request into a bounded agent task. Every packet should name the role, the skill, the source-of-truth files, the required outputs, and the escalation threshold.

## Manuscript and submission packet

- Role: `manuscript-reviser`, `claim-auditor`, or `submission-auditor`
- Skill: `paper-submission`
- Source of truth:
  - `paper/manuscript/manuscript.md`
  - `docs/governance/manuscript-contract.md`
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/nature-communications-submission-requirements.md`
- Required outputs:
  - revised text, audit findings, or compliance gap list
  - figure and Methods anchors for every claim-level change
  - visual inspection notes plus generator and provenance backtrace for any figure-dependent interpretation
  - explicit unresolved issues if evidence is weak
- Escalate when:
  - a change could alter scientific interpretation
  - a figure or table might move between main paper and supplementary
  - Nature-facing compliance conflicts with the current manuscript structure

## Experiment and results packet

- Role: `experiment-results-analyst`
- Skill: `experiment-results`
- Source of truth:
  - `docs/governance/experiment-contract.md`
  - `results/<run_name>/`
  - `DATA_PROVENANCE.md`
- Required outputs:
  - executed-run interpretation grounded in logs and artifacts
  - reproduction commands and artifact paths
  - figure-facing backtrace when a run artifact is being promoted into manuscript evidence
  - success, failure, and next-step analysis written in contract language
- Escalate when:
  - logs, metrics, or provenance inputs are missing
  - the run is not reproducible from committed artifacts
  - a manuscript-facing claim is being inferred from weak or partial evidence

## Paper asset review packet

- Role: `paper-asset-reviewer`
- Skill: `paper-asset-review`
- Source of truth:
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/paper-asset-review-workflow.md`
  - figure review bundle under `figures/review_artifacts/<figure_id>/`
- Required outputs:
  - keep, revise, split, or move recommendation
  - manuscript-fit justification
  - recorded confirmation that the figure was visually inspected and that generator and evidence layers were checked when applicable
  - review JSON artifacts required by the review gate
- Escalate when:
  - a figure looks acceptable visually but weakens the paper-level claim
  - visual content, generator code, and provenance source disagree about what the figure is showing
  - the review implies main-paper versus supplementary reclassification

## Governance and orchestration packet

- Role: `supervisor` or `red-team-reviewer`
- Skill: `agent-orchestrator`
- Source of truth:
  - `docs/governance/codex-collaboration-contract.md`
  - `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
  - `docs/agent-ops/ROLE_CATALOG.md`
  - `docs/agent-ops/REVIEW_AND_ESCALATION.md`
- Required outputs:
  - task framing
  - role assignments or review findings
  - warnings, rewrites, or milestone summary
- Escalate when:
  - a proposal introduces new governance layers or duplicate skills
  - a workflow shifts the branch toward code-first operation
  - a milestone decision requires human approval
