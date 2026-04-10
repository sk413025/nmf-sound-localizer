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
- `WARN_EVIDENCE`: claims or recommendations are not tied to files, commands, or artifacts
- `WARN_SCOPE_DOWNGRADE_CONCEALED`: delivered scope is narrower than planned scope but the closeout implies full completion
- `WARN_NO_TEXT_EVIDENCE`: the parent closeout lacks the text, file, or artifact evidence required by the packet
- `WARN_REVIEW_VERIFICATION_CONFUSION`: reviewer approval is being used as delivery verification without the required evidence check
- `WARN_PARENT_OVERCLAIM`: the parent claims completion beyond the packet's owned plan items or acceptance surface
- `WARN_MISSING_HIGH_RISK_VERIFIER`: a `high-risk` round does not name a verification owner
- `WARN_CLAIM_FLOOR_BURIED`: the supported discovery is present in the evidence but buried under qualifiers, setup detail, or caveat-led wording
- `WARN_CAVEAT_LEADS`: the highest-salience sentence leads with a limitation, pathway, or constraint instead of the supported advance
- `WARN_MOMENTUM_COLLAPSE`: repeated hedge-heavy sentence order prevents the manuscript from carrying discovery momentum across sections

When `WARN_CLAIM_FLOOR_BURIED`, `WARN_CAVEAT_LEADS`, or `WARN_MOMENTUM_COLLAPSE` is raised on manuscript-facing prose, cite the closest `SV#` exemplar from `docs/governance/scientific-voice-guide.md` so the rewrite target is concrete rather than abstract.

## Closeout integrity checks

Before a round is reported as complete, confirm all of the following:

- the closeout names the packet's acceptance surface and out-of-scope surfaces
- the closeout names the packet's risk level, review owner, and verification owner
- the closeout accounts for owned plan items as landed, deferred, or dropped
- any scope downgrade is stated explicitly rather than folded into a broad completion claim
- delivery evidence required by the packet is cited in text
- review findings and verification findings are distinguished when both roles exist
- a `high-risk` round does not omit verifier ownership

Escalate with `ESCALATE_HUMAN` when any of the following persist after one rewrite attempt:

- `WARN_SCOPE_DOWNGRADE_CONCEALED`
- `WARN_NO_TEXT_EVIDENCE`
- `WARN_REVIEW_VERIFICATION_CONFUSION`
- `WARN_PARENT_OVERCLAIM`
- `WARN_MISSING_HIGH_RISK_VERIFIER`

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
