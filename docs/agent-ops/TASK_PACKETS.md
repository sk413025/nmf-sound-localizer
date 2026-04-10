# Task Packets

Use these packet templates to turn a high-level request into a bounded delegated task, or into a checklist for a direct top-level execution round when explicit ownership and acceptance surfaces still need to be recorded.

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

Use packet fields to record decisions, not to recreate a second schema.

`Relevant conversation context` should summarize only the task-relevant parts of the current interaction.
`Context mode: summary-only` is the default.
Use `Context mode: summary+fork_context` only when exact wording, multi-turn decisions, or non-compressible constraints matter to the child task.
If the packet includes paper-facing review or hardening, name the applicable reviewer roles and evaluation goals from `docs/agent-ops/NATURE_REVIEWER_STACK.md`.

For any paper-facing explanation packet whose outputs may be promoted into manuscript, supplementary, legend, caption, review-note, or analysis-summary prose, the minimum human-facing additions are:

- `Claim floor`
- `Claim ceiling`
- `Evidence boundary`
- `Editor readout sentence`
- `Relevant voice exemplar(s)`
- `Bridge question`

Add sentence-friction, diction, or local cleanup notes only when they are genuinely part of the owned acceptance surface.

For any local high-salience, `cross-section`, or `whole-manuscript` manuscript round, carry the architecture bundle required by `docs/governance/manuscript-contract.md` and `docs/governance/scientific-voice-guide.md` instead of restating the full field inventory here.

For any `high-risk` round with broader significance or cross-disciplinary consequence in scope:

- create `results/<round_name>/governance_round.yaml`
- treat `governance_round.yaml` as the only machine-readable source of truth for proposed, reviewed, final, and verified status
- use `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md` as the canonical field inventory
- include a passing `make paper-governance-gate ROUND_DIR=results/<round_name>` run in `Delivery evidence required`

Anti-complexity rule:

- do not add a new packet field, verdict, or artifact unless you can first explain why the existing canonical surfaces are insufficient
- any governance expansion should also name what duplicated surface is removed or collapsed in exchange

For governance-changing rounds, also record these reflection fields; otherwise omit them:

- `Complexity risk`
- `Why existing primitives were insufficient`
- `What duplicated surface was removed`
- `What remains canonical after this round`

Before issuing any packet, the top-level agent must make an execution-or-delegation decision.
If delegation is chosen, use one child packet only when the request fits one core skill, one main output bundle, and one bounded acceptance surface.
If delegation is chosen, split the request into multiple child packets when it spans multiple skills or roles, mixes execution with separate review work, or contains independent acceptance criteria that can be delegated separately.
`Plan items owned` must name the exact subset the delegated child or direct implementer is accountable for closing.
`Delivery evidence required` must identify the text, file, or artifact evidence needed before the top-level agent may claim completion.
`Risk level` must classify the round as `high-risk` or `non-high-risk`.
`Architecture scope` must be one of `local-salience`, `cross-section`, or `whole-manuscript`.
Use `local-salience` only for local high-salience rewrites with no section reweighting, no section-bridge changes, and no Results-spine changes.
Use `cross-section` when the round changes more than one section, changes section bridges, or redistributes discovery-versus-tool weight without rebuilding the full paper.
Use `whole-manuscript` when the round restructures the Results spine or whole-paper story architecture.
For `high-risk` rounds, `Review owner` and `Verification owner` must both be named.
If ownership is compressed, record either a `non-high-risk rationale` or a `compression rationale`.
`Verification target` must identify who or what verifies delivery against evidence.
`Scope downgrade rule` must say how to report a narrowed landing without overstating the round as complete.

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
  - state the supported claim floor clearly before adding the evidence boundary
  - default to cross-disciplinary readability for Nature-facing prose
  - inspect neighboring paragraphs and section logic before treating a local rewrite as complete
  - record visual inspection notes plus generator and provenance backtrace for any figure-dependent interpretation
- Risk level:
  - classify as `high-risk` when interpretation, submission posture, or claim scope could shift; otherwise classify as `non-high-risk`
- Acceptance surface:
  - the named manuscript or submission surfaces the child is allowed to close
- Out-of-scope surfaces:
  - neighboring manuscript, figure, or packaging surfaces the child may mention but not claim as completed
- Plan items owned:
  - the exact rewrite, audit, or compliance items assigned in this packet, whether to a delegated child or a direct implementer
- Delivery evidence required:
  - revised text, anchored file locations, and any figure/provenance notes needed to justify closeout
- Review owner:
  - name the reviewer, or record a `non-high-risk rationale` or `compression rationale` if ownership is compressed
- Verification owner:
  - name the verifier; `high-risk` packets must not leave this blank
- Verification target:
  - parent or verifier confirms that the delivered text and anchors satisfy the packet without expanding scope
- Scope downgrade rule:
  - if only part of the requested manuscript surface lands, close out only that subset and list the remaining items explicitly
- Expected outputs:
  - revised text, audit findings, or compliance gap list
  - explicit `Claim floor`, `Claim ceiling`, `Evidence boundary`, `Editor readout sentence`, and `Bridge question` entries for the owned manuscript surface
  - anchored claim-evidence support, including figure or Methods anchors when relevant
  - `Relevant voice exemplar(s)` plus any required architecture or broader-significance bundle by reference to the canonical contracts, not by recreating a second field inventory
  - visual inspection notes plus generator/provenance backtrace for any figure-dependent interpretation
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
- Risk level:
  - classify as `high-risk` when the analysis may affect paper-facing interpretation; otherwise classify as `non-high-risk`
- Acceptance surface:
  - the named experiment question, comparison, or evidence interpretation to be closed
- Out-of-scope surfaces:
  - unrun comparisons, speculative mechanisms, or manuscript claims not supported by the committed artifacts
- Plan items owned:
  - the exact analysis items delegated from the round plan
- Delivery evidence required:
  - artifact paths, logs, metrics, and reproduction commands sufficient to support the reported interpretation
- Review owner:
  - name the reviewer, or record a `non-high-risk rationale` or `compression rationale` if ownership is compressed
- Verification owner:
  - name the verifier; `high-risk` packets must not leave this blank
- Verification target:
  - parent or verifier checks the cited artifacts and reproduction surface before accepting the closeout
- Scope downgrade rule:
  - if only a subset of the analysis lands, disclose the missing items and do not roll the whole experiment round up as complete
- Expected outputs:
  - executed-run interpretation grounded in logs and artifacts
  - reproduction commands and artifact paths
  - figure-facing backtrace when a run artifact is being promoted into manuscript evidence
  - success, failure, and next-step analysis written in contract language
- Escalate when:
  - logs, metrics, or provenance inputs are missing
  - the run is not reproducible from committed artifacts
  - a paper-facing claim is being inferred from weak or partial evidence
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
- Risk level:
  - classify as `high-risk` when the recommendation could affect paper-level claim support or asset placement; otherwise classify as `non-high-risk`
- Acceptance surface:
  - the named figure or table decision and its manuscript-fit judgment
- Out-of-scope surfaces:
  - broader manuscript rewrites, undelegated panels, or unreviewed assets
- Plan items owned:
  - the exact asset-review decisions assigned in this packet, whether to a delegated child or a direct implementer
- Delivery evidence required:
  - visual inspection confirmation plus any generator, provenance, and review-gate artifacts required for the recommendation
- Review owner:
  - name the reviewer, or record a `non-high-risk rationale` or `compression rationale` if ownership is compressed
- Verification owner:
  - name the verifier; `high-risk` packets must not leave this blank
- Verification target:
  - parent or verifier checks that the reviewed asset and cited evidence match the recommendation
- Scope downgrade rule:
  - if only some panels or decisions were reviewed, disclose the narrowed coverage explicitly
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
  - frame a task, choose direct execution or child roles, challenge a workflow proposal, or review a coordinated result set
- Relevant conversation context:
  - include the requested workflow, confirmed delegation policy, unresolved governance risks, and any already chosen defaults
- Source of truth:
  - `docs/governance/codex-collaboration-contract.md`
  - `docs/governance/closeout-integrity-contract.md`
  - `docs/agent-ops/SUPERVISOR_OPERATING_MODEL.md`
  - `docs/agent-ops/ROLE_CATALOG.md`
  - `docs/agent-ops/REVIEW_AND_ESCALATION.md`
  - `docs/agent-ops/ROUND_CLOSEOUT_TEMPLATE.md`
- Constraints:
  - the top-level agent must make an explicit execution-or-delegation decision before specialist work begins
  - assume repository-default standing authorization for sub-agent use unless a higher-level constraint blocks delegation
- Risk level:
  - classify as `high-risk` when the packet can change governance posture, closeout claims, or milestone reporting; otherwise classify as `non-high-risk`
- Acceptance surface:
  - the specific workflow, packet, review, or closeout surface under judgment
- Out-of-scope surfaces:
  - unrelated governance layers, role redesign, or non-requested process rewrites
- Plan items owned:
  - the exact orchestration, review, or hardening items assigned in this packet, whether to a delegated child or a direct implementer
- Delivery evidence required:
  - packet text, routing decisions, warning disposition, and any cited status or artifact checks needed for closeout
- Review owner:
  - name the reviewer, or record a `non-high-risk rationale` or `compression rationale` if ownership is compressed
- Verification owner:
  - name the verifier; `high-risk` packets must not leave this blank
- Verification target:
  - parent or verifier checks that the reported outcome matches the owned packet scope
- Scope downgrade rule:
  - if the packet lands only a narrowed governance subset, the closeout must name the downgrade and remaining plan items
- Expected outputs:
  - task framing
  - execution-versus-delegation decisions, role assignments, decomposition decisions, review findings, or `Context mode` decisions
  - warnings, rewrites, or milestone summary
- Escalate when:
  - a proposal introduces new governance layers or duplicate skills
  - a workflow shifts the branch toward code-first operation
  - a milestone decision requires human approval
- Context mode:
  - start with `summary-only`
  - upgrade to `summary+fork_context` when the child reviewer must reconstruct exact dialogue history to judge the workflow safely
