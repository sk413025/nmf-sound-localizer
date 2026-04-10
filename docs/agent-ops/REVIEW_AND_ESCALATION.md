# Review and Escalation

This file defines when work must be reviewed and when it must be escalated.

## Review requirements

Reviewer involvement is mandatory for:

- manuscript claim changes
- movement of assets between main paper and supplementary
- governance or role changes
- delegation-gate or context-handoff policy changes
- large rewrites of Results, Discussion, or Abstract
- paper-figure layout/readability disputes that require paper-asset-review or geometry-audit confirmation
- closeouts for large rounds when delivered scope may be narrower than planned scope

## Red-team requirements

Red-team review is mandatory for:

- new workflows or role systems
- changes that increase process complexity
- changes to top-level execution-or-delegation policy or child-context policy
- proposals that redefine `codex-native` or `agent-native`
- proposals that blur reviewer and verifier duties or let parent closeout claims outrun landed evidence

## Standard warning levels

- `WARN`: fixable issue; work may continue after correction
- `NEEDS_REWRITE`: the output is not safe to use as-is
- `ESCALATE_HUMAN`: milestone-level decision required from the human approver

## Common warning codes

- `WARN_SCOPE`: the task drifted away from manuscript-first outcomes
- `WARN_CAPABILITY`: the proposal assumes Codex features that are not grounded in local primitives
- `WARN_ACTIONABILITY`: the output cannot be handed to another agent without new decisions
- `WARN_CONTEXT`: the child-agent handoff is missing relevant conversation context or leaks irrelevant history
- `WARN_DECOMPOSITION`: the parent kept a multi-scope request in one child task without a valid single-child rationale
- `WARN_SUPERVISION`: the parent interrupted or closed a child agent without first checking current status or latest output
- `WARN_DUPLICATION`: the output duplicates an existing role, skill, or document path
- `WARN_OVERMODELED_GOVERNANCE`: the change introduces a field, verdict, artifact, or checker rule that duplicates an existing canonical surface or hard-codes reviewer judgment into the machine layer
- `WARN_EVIDENCE`: claims or recommendations are not tied to files, commands, or artifacts
- `WARN_SCOPE_DOWNGRADE_CONCEALED`: delivered scope is narrower than planned scope but the closeout implies full completion
- `WARN_NO_TEXT_EVIDENCE`: the parent closeout lacks the text, file, or artifact evidence required by the packet
- `WARN_REVIEW_VERIFICATION_CONFUSION`: reviewer approval is being used as delivery verification without the required evidence check
- `WARN_PARENT_OVERCLAIM`: the parent claims completion beyond the packet's owned plan items or acceptance surface
- `WARN_MISSING_HIGH_RISK_VERIFIER`: a `high-risk` round does not name a verification owner
- `WARN_CLAIM_FLOOR_BURIED`: the supported discovery is present in the evidence but buried under qualifiers, setup detail, or caveat-led wording
- `WARN_CAVEAT_LEADS`: the highest-salience sentence leads with a limitation, pathway, or constraint instead of the supported advance
- `WARN_MOMENTUM_COLLAPSE`: repeated hedge-heavy sentence order prevents the manuscript from carrying discovery momentum across sections
- `WARN_PROTAGONIST_DRIFT`: the paper's main character shifts between discovery, tool, reference object, or workflow
- `WARN_MISSING_PIVOT`: the Results sequence never clearly updates the reader's model of the system
- `WARN_TOOL_OVERWEIGHT`: tool-validation has taken more narrative mass than the discovery it is meant to support
- `WARN_WORLDVIEW_RECAP`: the Discussion opening restates Results instead of saying what understanding has changed
- `WARN_SECTION_JOB_DRIFT`: Results sections have local findings but no stable whole-paper job map
- `WARN_DISCOVERY_CASHOUT_BURIED`: the paper-level discovery appears only as a late extension rather than as the story's cash-out
- `WARN_SECOND_LAYER_DISCOVERY_MISSING`: broader significance appears only as applications or Discussion add-on language rather than as an evidence-earned second-layer discovery
- `WARN_SECOND_LAYER_NOT_EARNED`: the round promotes a second-layer discovery that is not yet strong enough to survive conservative rewrite or reviewer routing
- `WARN_BRANCH_NOT_EARNED`: the round promotes a downstream branch that the evidence does not yet support as a bounded consequence
- `WARN_TRUNK_BRANCH_COLLAPSE`: downstream consequence has become more prominent than the broader-implication trunk that should support it
- `WARN_BOLT_ON_IMPLICATION`: a broader implication reads like a newly introduced topic or literature thread rather than a consequence of the paper's own discovery actor
- `WARN_FRONT_DOOR_PRELOAD_MISSING`: the second-layer discovery appears in Discussion or late Results only, without an Abstract-tail or Introduction-ending preload
- `WARN_TWO_TAKEAWAY_FAILURE`: the paper supports two takeaways in principle but the broader one is not memorable enough for a skimming editor to retain
- `WARN_OPTIONAL_LEAF_OVERWEIGHT`: a weaker optional consequence is competing with the trunk or branch instead of remaining visibly subordinate
- `WARN_LEAF_SHOULD_DROP`: a leaf consequence is not worth the added literature thread or reader burden and should be removed
- `WARN_NO_BOLT_ON_TEST_MISSING`: the round claims broader significance landed but never checks whether the branch collapses when the trunk is removed
- `WARN_BREATHING_COLLAPSE`: repeated dense explanation leaves no reset sentence or memorable take-home shift
- `WARN_ARCHITECTURE_SCOPE_MISROUTED`: the round changes section order, section bridges, discovery-versus-tool weight, or jointly reframes title/abstract/introduction/discussion but is still routed as `local-salience`
- `WARN_NOUN_STACK`: the sentence asks the reader to decode a dense technical noun chain before seeing the scientific action
- `WARN_CAUSAL_GAP`: facts or numbers are reported without telling the reader what they mean
- `WARN_FORMAL_REGISTER`: the wording is formally correct but unnatural for broad scientific prose
- `WARN_STATIC_VERB_DRAG`: low-energy holding verbs flatten a stronger scientific action that the evidence already supports

When `WARN_CLAIM_FLOOR_BURIED`, `WARN_CAVEAT_LEADS`, `WARN_MOMENTUM_COLLAPSE`, `WARN_PROTAGONIST_DRIFT`, `WARN_MISSING_PIVOT`, `WARN_TOOL_OVERWEIGHT`, `WARN_WORLDVIEW_RECAP`, `WARN_SECTION_JOB_DRIFT`, `WARN_DISCOVERY_CASHOUT_BURIED`, `WARN_SECOND_LAYER_DISCOVERY_MISSING`, `WARN_SECOND_LAYER_NOT_EARNED`, `WARN_BRANCH_NOT_EARNED`, `WARN_TRUNK_BRANCH_COLLAPSE`, `WARN_BOLT_ON_IMPLICATION`, `WARN_FRONT_DOOR_PRELOAD_MISSING`, `WARN_TWO_TAKEAWAY_FAILURE`, `WARN_OPTIONAL_LEAF_OVERWEIGHT`, `WARN_LEAF_SHOULD_DROP`, `WARN_NO_BOLT_ON_TEST_MISSING`, `WARN_BREATHING_COLLAPSE`, `WARN_ARCHITECTURE_SCOPE_MISROUTED`, `WARN_NOUN_STACK`, `WARN_CAUSAL_GAP`, `WARN_FORMAL_REGISTER`, or `WARN_STATIC_VERB_DRAG` is raised on paper-facing explanation, cite the closest `SV#` exemplar from `docs/governance/scientific-voice-guide.md` so the rewrite target is concrete rather than abstract.

## Closeout integrity checks

Before a round is reported as complete, confirm all of the following:

- the closeout names the packet's acceptance surface and out-of-scope surfaces
- the closeout names the packet's risk level, review owner, and verification owner
- the closeout accounts for owned plan items as landed, deferred, or dropped
- any scope downgrade is stated explicitly rather than folded into a broad completion claim
- delivery evidence required by the packet is cited in text
- review findings and verification findings are distinguished when both roles exist
- a `high-risk` round does not omit verifier ownership
- any round that changes section order, section bridges, discovery-versus-tool weight, or title/abstract/introduction/discussion together is not still classified as `local-salience`
- any `cross-section` or `whole-manuscript` round that claims architecture landed includes an `Architecture evidence map`
- whole-paper architecture remains open when reviewer findings still flag protagonist drift, missing pivot, tool overweight, buried discovery cash-out, missing second-layer discovery, trunk/branch collapse, bolt-on implication, or breathing collapse
- whole-paper broader-significance promotion remains open when review still flags unearned trunk promotion, unearned branch promotion, or a leaf that should be dropped
- governance simplification remains open when review still flags duplicated schema surfaces, checker semantic overreach, or a missing statement of what now remains canonical

Escalate with `ESCALATE_HUMAN` when any of the following persist after one rewrite attempt:

- `WARN_SCOPE_DOWNGRADE_CONCEALED`
- `WARN_NO_TEXT_EVIDENCE`
- `WARN_REVIEW_VERIFICATION_CONFUSION`
- `WARN_PARENT_OVERCLAIM`
- `WARN_MISSING_HIGH_RISK_VERIFIER`
- `WARN_ARCHITECTURE_SCOPE_MISROUTED`

## Human gates

Escalate to the human approver at these boundaries:

- strategic direction changes
- scientific claim changes with interpretation risk
- delegation policy changes that affect top-level agent behavior
- submission packaging choices with paper-level consequences
- final approval for a coordinated multi-agent change set

When figure risk is primarily layout/readability rather than claim validity,
route first through paper-asset-review plus any required geometry audit before
escalating toward manuscript rewrite.
