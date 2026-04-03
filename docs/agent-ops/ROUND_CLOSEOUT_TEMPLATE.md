# Round Closeout Template

Use this template for dialogue closeouts when a round needs explicit accounting.
It is a reusable closeout ledger, not a mandatory repo artifact for every round.

## Closeout ledger

- Round or packet:
- Objective:
- Risk level:
- Acceptance surface:
- Out-of-scope surfaces:
- Plan items owned:
- Plan completion:
  - `complete`, `partial`, or `not complete`
- Delivered items:
- Deferred or dropped items:
- Unresolved promised joints:
  - list promised links, handoffs, or dependencies that did not land cleanly
- Scope-downgrade disclosure:
  - `none`, or state the narrowed landing explicitly
- Delivery evidence:
  - cite the text, files, artifacts, or commands required by the packet
- Review verdict:
  - named review owner, plus disposition
- Verification verdict:
  - named verification owner, plus evidence check outcome
- Warnings:
  - `none`, or list warning codes
- Escalation:
  - `none`, `NEEDS_REWRITE`, or `ESCALATE_HUMAN`
- Parent closeout statement:
  - state only what landed inside the owned acceptance surface

## Required closeout rules

- Do not mark the round complete if `Deferred or dropped items` is non-empty unless the closeout explicitly scopes completion down to the delivered subset.
- Do not omit `Delivery evidence`.
- Do not omit named review and verification ownership on `high-risk` rounds.
- Do not merge `Review verdict` and `Verification verdict` unless the packet explicitly assigned both duties to one owner and the closeout states a `non-high-risk rationale` or `compression rationale`.
- Do not let the `Parent closeout statement` exceed the packet's owned plan items or acceptance surface.
