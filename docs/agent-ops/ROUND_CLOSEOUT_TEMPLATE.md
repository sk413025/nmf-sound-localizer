# Round Closeout Template

Use this template for dialogue closeouts when a round needs explicit accounting.
It is a reusable closeout ledger, not a mandatory repo artifact for every round.
For any `high-risk` round with broader significance or cross-disciplinary consequence in scope, the blocking repo artifact is `results/<round_name>/governance_round.yaml`.

## Closeout ledger

- Round or packet:
- Objective:
- Risk level:
- Architecture scope:
- Acceptance surface:
- Out-of-scope surfaces:
- Plan items owned:
- Plan completion:
  - `complete`, `partial`, or `not complete`
- Delivered items:
- Deferred or dropped items:
- Architecture verdict:
  - state whether the landed surface preserves the intended protagonist, pivot, tool role, and worldview shift
- Protagonist preserved:
  - `yes` or `no`
- Pivot landed:
  - `yes` or `no`
- Tool role preserved:
  - `yes` or `no`
- Worldview shift explicit:
  - `yes` or `no`
- Second-layer discovery explicit:
  - `yes` or `no`
- Broader-significance status:
  - `core-only`, `second-layer earned`, `branch earned`, or `leaf allowed`
- Final status:
  - mirror `closeout.final_status` from `results/<round_name>/governance_round.yaml`
- Status change note:
  - `none`, or a short note if closeout demoted the reviewer-approved level
- Front-door preload sentence landed:
  - `yes`, `no`, or `not required`
- Two-takeaway editor readout landed:
  - `yes`, `no`, or `not required`
- No-bolt-on test passed:
  - `yes` or `no`
- Architecture evidence map:
  - tie the intended spine to title, the `front-door preload sentence` when required, pivot sentence, discovery cash-out sentence, and Discussion opening
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
- Do not report broader significance as landed on a `high-risk` round unless `results/<round_name>/governance_round.yaml` exists and `make paper-governance-gate ROUND_DIR=results/<round_name>` passes.
- Do not let closeout freehand a second governance vocabulary. For `high-risk` broader-significance rounds, treat `governance_round.yaml` as the canonical status record and summarize it here rather than inventing new field names.
- Do not treat sentence polish as architectural completion. When the packet required protagonist, pivot, tool role, worldview shift, or second-layer discovery work, fill the architecture fields explicitly.
- Do not claim whole-paper architecture completion without an `Architecture evidence map` when the round touched cross-section or whole-manuscript structure.
- Do not merge `Review verdict` and `Verification verdict` unless the packet explicitly assigned both duties to one owner and the closeout states a `non-high-risk rationale` or `compression rationale`.
- Do not let the `Parent closeout statement` exceed the packet's owned plan items or acceptance surface.
