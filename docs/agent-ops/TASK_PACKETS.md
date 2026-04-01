# Task Packets

Use these packet templates to turn a high-level request into a bounded child-agent task.

Every packet should include:

- `Objective`
- `Relevant conversation context`
- `Source of truth`
- `Constraints`
- `Expected outputs`
- `Escalate when`
- `Context mode`

`Relevant conversation context` should summarize only the task-relevant parts of the current interaction.
`Context mode: summary-only` is the default.
Use `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.

Before issuing any packet, the parent must make a decomposition decision.
Use one child packet only when the request fits one core skill, one main output bundle, and one bounded acceptance surface.
Split the request into multiple child packets when it spans multiple skills or roles, mixes execution with separate review work, or contains independent acceptance criteria that can be delegated separately.

## Manuscript and submission packet

- Role: `manuscript-reviser`, `claim-auditor`, or `submission-auditor`
- Skill: `paper-submission`
- Objective:
  - revise manuscript text, audit claim/evidence alignment, or check Nature-facing compliance
- Relevant conversation context:
  - include the requested change, confirmed style or audience constraints, already approved language, and unresolved evidence risks
  - omit unrelated thread history
- Source of truth:
  - `paper/manuscript/manuscript.md`
  - `docs/governance/manuscript-contract.md`
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/nature-communications-submission-requirements.md`
- Constraints:
  - preserve claim and evidence integrity
  - default to cross-disciplinary readability for Nature-facing prose
  - inspect neighboring paragraphs and section logic before treating a local rewrite as complete
  - record visual inspection notes plus generator and provenance backtrace for any figure-dependent interpretation
- Expected outputs:
  - revised text, audit findings, or compliance gap list
  - figure and Methods anchors for every claim-level change
  - coherence or terminology notes when surrounding transitions had to change to keep the manuscript natural
  - visual inspection notes plus generator and provenance backtrace for any figure-dependent interpretation
  - explicit unresolved issues if evidence is weak
- Escalate when:
  - a change could alter scientific interpretation
  - a figure or table might move between main paper and supplementary
  - Nature-facing compliance conflicts with the current manuscript structure
- Context mode:
  - start with `summary-only`
  - upgrade to `summary+fork_context` when exact wording history or reviewer instructions must be preserved

## Experiment and results packet

- Role: `experiment-results-analyst`
- Skill: `experiment-results`
- Objective:
  - interpret committed run artifacts and turn them into manuscript-safe analysis language
- Relevant conversation context:
  - include the metric, comparison, or hypothesis the user actually asked about
  - include any already agreed caution language or reproducibility constraints
- Source of truth:
  - `docs/governance/experiment-contract.md`
  - `results/<run_name>/`
  - `DATA_PROVENANCE.md`
- Constraints:
  - ground conclusions in committed logs, metrics, and provenance artifacts
  - keep fail-fast expectations and reproducibility requirements explicit
- Expected outputs:
  - executed-run interpretation grounded in logs and artifacts
  - reproduction commands and artifact paths
  - figure-facing backtrace when a run artifact is being promoted into manuscript evidence
  - success, failure, and next-step analysis written in contract language
- Escalate when:
  - logs, metrics, or provenance inputs are missing
  - the run is not reproducible from committed artifacts
  - a manuscript-facing claim is being inferred from weak or partial evidence
- Context mode:
  - start with `summary-only`
  - upgrade to `summary+fork_context` when the task depends on precise dialogue about metrics, caveats, or comparison scope

## Paper asset review packet

- Role: `paper-asset-reviewer`
- Skill: `paper-asset-review`
- Objective:
  - decide whether a figure or table should stay, be revised, be split, or move to supplementary
- Relevant conversation context:
  - include the figure or panel under review, the manuscript claim or critique that triggered the review, and any already agreed layout or claim constraints
- Source of truth:
  - `docs/governance/submission-contract.md`
  - `docs/nature-communications/paper-asset-review-workflow.md`
  - figure review bundle under `figures/review_artifacts/<figure_id>/`
- Constraints:
  - inspect the real visual asset
  - check generator and evidence layers when the asset is generated or data-backed
- Expected outputs:
  - keep, revise, split, or move recommendation
  - manuscript-fit justification
  - recorded confirmation that the figure was visually inspected and that generator and evidence layers were checked when applicable
  - review JSON artifacts required by the review gate
- Escalate when:
  - a figure looks acceptable visually but weakens the paper-level claim
  - visual content, generator code, and provenance source disagree about what the figure is showing
  - the review implies main-paper versus supplementary reclassification
- Context mode:
  - start with `summary-only`
  - upgrade to `summary+fork_context` when precise claim wording or prior review instructions materially affect the judgment

## Governance and orchestration packet

- Role: `supervisor` or `red-team-reviewer`
- Skill: `agent-orchestrator`
- Objective:
  - frame a task, choose child roles, challenge a workflow proposal, or review a coordinated result set
- Relevant conversation context:
  - include the requested workflow, confirmed delegation policy, unresolved governance risks, and any already chosen defaults
- Source of truth:
  - `docs/governance/codex-collaboration-contract.md`
  - `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
  - `docs/agent-ops/ROLE_CATALOG.md`
  - `docs/agent-ops/REVIEW_AND_ESCALATION.md`
- Constraints:
  - the top-level agent remains the parent orchestrator, not a worker
  - assume repository-default standing authorization for sub-agent use unless a higher-level constraint blocks delegation
- Expected outputs:
  - task framing
  - role assignments, decomposition decisions, review findings, or `Context mode` decisions
  - warnings, rewrites, or milestone summary
- Escalate when:
  - a proposal introduces new governance layers or duplicate skills
  - a workflow shifts the branch toward code-first operation
  - a milestone decision requires human approval
- Context mode:
  - start with `summary-only`
  - upgrade to `summary+fork_context` when the child reviewer must reconstruct exact dialogue history to judge the workflow safely
