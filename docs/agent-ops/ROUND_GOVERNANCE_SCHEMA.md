# Round Governance Schema

Use this schema only for `high-risk` rounds where broader significance or cross-disciplinary consequence is in scope.

The canonical machine-readable artifact is:

- `results/<round_name>/governance_round.yaml`

Human-readable memos may accompany the round, but they do not replace this YAML file for gating.

## Purpose

Keep one executable decision spine for Nature-level broader significance:

- `core discovery`
- `earned second-layer discovery`
- `bounded consequence`
- `optional removable leaf`

The semantic gate checks only:

- artifact shape
- status monotonicity
- owner separation or justified compression
- downgrade disclosure
- verifier and closeout agreement

It does **not** try to infer reviewer judgment from human-facing verdict fields.

## Canonical path

- `results/<round_name>/governance_round.yaml`

Run the blocking semantic gate with:

- `make paper-governance-gate ROUND_DIR=results/<round_name>`

## Status ladder

Use only these values:

- `core-only`
- `second-layer earned`
- `branch earned`
- `leaf allowed`

Status may stay the same or move downward across the round:

- `reviewed_status <= proposed_status`
- `final_status <= reviewed_status`
- `verified_status == final_status`

Any downgrade must include a non-empty `status_change_note` at the stage where it happens.

## Role handling

High-risk owner handling must be explicit:

- `role_mode: separated`
  - implementer, reviewer, and verifier are three distinct owners
  - `role_compression_reason: none`
- `role_mode: compressed`
  - at least one role is shared
  - `role_compression_reason` explains why compression is unavoidable

## Minimal schema

```yaml
risk_level: high-risk
architecture_scope: local-salience|cross-section|whole-manuscript
broader_significance_in_scope: true
role_mode: separated|compressed
role_compression_reason: none|"..."

task_packet:
  core_discovery: "..."
  second_layer_discovery: "..."
  trunk: "..."
  branch: "..."
  leaf: "..."
  proposed_status: core-only|second-layer earned|branch earned|leaf allowed
  front_door_preload_sentence: "..."
  two_takeaway_editor_readout:
    - "core discovery sentence"
    - "second-layer discovery sentence"
  no_bolt_on_test: "..."

review_verdict:
  reviewer_owner: "..."
  reviewed_status: core-only|second-layer earned|branch earned|leaf allowed
  status_change_note: none|"..."

verification_verdict:
  verifier_owner: "..."
  verified_status: core-only|second-layer earned|branch earned|leaf allowed
  independent_verification_complete: yes|no
  closeout_ready: yes|no

closeout:
  implementer_owner: "..."
  final_status: core-only|second-layer earned|branch earned|leaf allowed
  status_change_note: none|"..."
  remaining_gap_disclosed: "..."
```

Status-dependent requirements are enforced by the checker:

- `core-only`
  - only `core_discovery` and `proposed_status` stay populated in `task_packet`
- `second-layer earned`
  - `second_layer_discovery`, `trunk`, `front_door_preload_sentence`, and `two_takeaway_editor_readout` are required
  - `branch`, `leaf`, and `no_bolt_on_test` stay empty
- `branch earned`
  - all trunk fields plus `branch` and `no_bolt_on_test` are required
  - `leaf` stays empty
- `leaf allowed`
  - trunk and branch fields remain required
  - `leaf` becomes required

## Gate behavior

`check_round_governance_semantics.py` blocks when:

- the YAML shape is incomplete
- a required field is empty for the chosen status
- status moves upward across the round
- a downgrade happens without `status_change_note`
- `role_mode: separated` does not have distinct owners
- `role_mode: compressed` lacks a real compression reason
- verifier and closeout disagree about the final status

Reviewer judgments such as `Front-door preload sentence landed`, `Downstream consequence bounded`, `No-bolt-on test passed`, or closeout narrative verdicts such as `Second-layer discovery explicit` remain useful in human review and closeout prose, but they are not part of the canonical YAML interface and are not machine-validated here.

## Examples

### `second-layer earned`

```yaml
risk_level: high-risk
architecture_scope: cross-section
broader_significance_in_scope: true
role_mode: separated
role_compression_reason: none

task_packet:
  core_discovery: "A recurring physical code is present."
  second_layer_discovery: "Passive structure itself becomes part of the sensing substrate."
  trunk: "Passive structure becomes part of the sensing substrate."
  branch: ""
  leaf: ""
  proposed_status: second-layer earned
  front_door_preload_sentence: "This discovery relocates part of sensing into passive structure."
  two_takeaway_editor_readout:
    - "A recurring physical code is present."
    - "Passive structure becomes part of the sensing substrate."
  no_bolt_on_test: ""

review_verdict:
  reviewer_owner: "Reviewer"
  reviewed_status: second-layer earned
  status_change_note: none

verification_verdict:
  verifier_owner: "Verifier"
  verified_status: second-layer earned
  independent_verification_complete: yes
  closeout_ready: yes

closeout:
  implementer_owner: "Implementer"
  final_status: second-layer earned
  status_change_note: none
  remaining_gap_disclosed: none
```

### Downgrade example

```yaml
task_packet:
  proposed_status: leaf allowed
  branch: "Coarse neighborhoods support bounded front-end filtering."
  leaf: "A weaker farther implication."

review_verdict:
  reviewed_status: branch earned
  status_change_note: "Drop the optional leaf to preserve trunk memory."

verification_verdict:
  verified_status: branch earned

closeout:
  final_status: branch earned
  status_change_note: none
```
