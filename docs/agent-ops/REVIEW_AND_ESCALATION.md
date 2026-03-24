# Review and Escalation

This file defines when work must be reviewed and when it must be escalated.

## Review requirements

Reviewer involvement is mandatory for:

- manuscript claim changes
- movement of assets between main paper and supplementary
- governance or role changes
- large rewrites of Results, Discussion, or Abstract
- paper-figure layout/readability disputes that require paper-asset-review or geometry-audit confirmation

## Red-team requirements

Red-team review is mandatory for:

- new workflows or role systems
- changes that increase process complexity
- proposals that redefine `codex-native` or `agent-native`

## Standard warning levels

- `WARN`: fixable issue; work may continue after correction
- `NEEDS_REWRITE`: the output is not safe to use as-is
- `ESCALATE_HUMAN`: milestone-level decision required from the human approver

## Common warning codes

- `WARN_SCOPE`: the task drifted away from manuscript-first outcomes
- `WARN_CAPABILITY`: the proposal assumes Codex features that are not grounded in local primitives
- `WARN_ACTIONABILITY`: the output cannot be handed to another agent without new decisions
- `WARN_DUPLICATION`: the output duplicates an existing role, skill, or document path
- `WARN_EVIDENCE`: claims or recommendations are not tied to files, commands, or artifacts

## Human gates

Escalate to the human approver at these boundaries:

- strategic direction changes
- scientific claim changes with interpretation risk
- submission packaging choices with paper-level consequences
- final approval for a coordinated multi-agent change set

When figure risk is primarily layout/readability rather than claim validity,
route first through paper-asset-review plus any required geometry audit before
escalating toward manuscript rewrite.
