# Round Governance Schema

Use this schema for any `high-risk` round where broader significance or cross-disciplinary consequence is in scope.

The canonical machine-readable artifact is:

- `results/<round_name>/governance_round.yaml`

Human-readable memos may still accompany the round, but they do not replace this YAML file for gating.

## Purpose

This artifact keeps one executable decision spine for Nature-level broader significance:

- `core discovery`
- `earned second-layer discovery`
- `bounded consequence`
- `optional removable leaf`

The semantic gate does not track handwritten demotion codes anymore. It only checks:

- proposed status
- reviewed status
- final closeout status
- verified final status
- whether any downgrade is explicitly disclosed with a short note

That keeps the gate blocking, but much easier to maintain.

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

The statuses may stay the same or move downward across the round:

- `reviewed_status <= proposed_status`
- `final_status <= reviewed_status`
- `verified_status == final_status`

Any downgrade must include a non-empty `status_change_note` at the stage where it happens.

## Role handling

High-risk owner handling must be explicit:

- `role_mode: separated`
  - implementer, reviewer, and verifier must be three distinct owners
  - `role_compression_reason` stays `none`
- `role_mode: compressed`
  - at least one role is shared
  - `role_compression_reason` must explain why compression is unavoidable

## Required top-level fields

```yaml
risk_level: high-risk
architecture_scope: local-salience|cross-section|whole-manuscript
broader_significance_in_scope: true
role_mode: separated|compressed
role_compression_reason: none|"..."

task_packet: {...}
review_verdict: {...}
verification_verdict: {...}
closeout: {...}
```

## Required packet fields

```yaml
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
```

Status-dependent expectations:

- `core-only`
  - only `core_discovery` and `proposed_status` stay populated
  - `second_layer_discovery`, `trunk`, `branch`, `leaf`, `front_door_preload_sentence`, `two_takeaway_editor_readout`, and `no_bolt_on_test` stay empty
- `second-layer earned`
  - `second_layer_discovery`, `trunk`, `front_door_preload_sentence`, and `two_takeaway_editor_readout` are required
  - `branch`, `leaf`, and `no_bolt_on_test` stay empty
- `branch earned`
  - all trunk fields plus `branch` and `no_bolt_on_test` are required
  - `leaf` stays empty
- `leaf allowed`
  - trunk and branch fields remain required
  - `leaf` becomes required

## Required review verdict fields

```yaml
review_verdict:
  reviewer_owner: "..."
  reviewed_status: core-only|second-layer earned|branch earned|leaf allowed
  status_change_note: none|"..."
  front_door_landed: yes|no|not required
  consequence_bounded: yes|no|not required
  no_bolt_on_passed: yes|no|not required
```

Meaning:

- `status_change_note`
  - use `none` when review keeps the packet status unchanged
  - use a short explanation when review demotes the packet one or more levels
- `front_door_landed`
  - `not required` for `core-only`
  - `yes` for every promoted status
- `consequence_bounded` and `no_bolt_on_passed`
  - `not required` for `core-only` and `second-layer earned`
  - `yes` for `branch earned` and `leaf allowed`

## Required verification verdict fields

```yaml
verification_verdict:
  verifier_owner: "..."
  verified_status: core-only|second-layer earned|branch earned|leaf allowed
  independent_verification_complete: yes|no
  closeout_ready: yes|no
```

## Required closeout fields

```yaml
closeout:
  implementer_owner: "..."
  final_status: core-only|second-layer earned|branch earned|leaf allowed
  status_change_note: none|"..."
  second_layer_explicit: yes|no
  branch_retained: yes|no
  leaf_retained: yes|no
  remaining_gap_disclosed: "..."
```

Meaning:

- `status_change_note`
  - use `none` when closeout keeps the reviewed status unchanged
  - use a short explanation when closeout demotes the round below the reviewer status
- `second_layer_explicit`
  - `no` only for `core-only`
- `branch_retained`
  - `yes` only for `branch earned` and `leaf allowed`
- `leaf_retained`
  - `yes` only for `leaf allowed`

## Minimal legal examples

### `second-layer earned`

```yaml
risk_level: high-risk
architecture_scope: cross-section
broader_significance_in_scope: true
role_mode: separated
role_compression_reason: none

task_packet:
  core_discovery: "The system carries a recurring physical code."
  second_layer_discovery: "The code relocates part of sensing into passive structure."
  trunk: "Passive structure itself becomes part of the sensing substrate."
  branch: ""
  leaf: ""
  proposed_status: second-layer earned
  front_door_preload_sentence: "This discovery relocates part of sensing into passive structure."
  two_takeaway_editor_readout:
    - "The system carries a recurring physical code."
    - "Passive structure itself becomes part of the sensing substrate."
  no_bolt_on_test: ""

review_verdict:
  reviewer_owner: "Reviewer"
  reviewed_status: second-layer earned
  status_change_note: none
  front_door_landed: yes
  consequence_bounded: not required
  no_bolt_on_passed: not required

verification_verdict:
  verifier_owner: "Verifier"
  verified_status: second-layer earned
  independent_verification_complete: yes
  closeout_ready: yes

closeout:
  implementer_owner: "Implementer"
  final_status: second-layer earned
  status_change_note: none
  second_layer_explicit: yes
  branch_retained: no
  leaf_retained: no
  remaining_gap_disclosed: none
```

### `branch earned`

```yaml
risk_level: high-risk
architecture_scope: cross-section
broader_significance_in_scope: true
role_mode: separated
role_compression_reason: none

task_packet:
  core_discovery: "The system carries a recurring physical code."
  second_layer_discovery: "The code relocates part of sensing into passive structure."
  trunk: "Passive structure itself becomes part of the sensing substrate."
  branch: "Coarse neighborhoods can support bounded front-end filtering."
  leaf: ""
  proposed_status: branch earned
  front_door_preload_sentence: "This discovery relocates part of sensing into passive structure."
  two_takeaway_editor_readout:
    - "The system carries a recurring physical code."
    - "Passive structure itself becomes part of the sensing substrate."
  no_bolt_on_test: "Remove the trunk and the branch collapses immediately."

review_verdict:
  reviewer_owner: "Reviewer"
  reviewed_status: branch earned
  status_change_note: none
  front_door_landed: yes
  consequence_bounded: yes
  no_bolt_on_passed: yes

verification_verdict:
  verifier_owner: "Verifier"
  verified_status: branch earned
  independent_verification_complete: yes
  closeout_ready: yes

closeout:
  implementer_owner: "Implementer"
  final_status: branch earned
  status_change_note: none
  second_layer_explicit: yes
  branch_retained: yes
  leaf_retained: no
  remaining_gap_disclosed: none
```

### `leaf allowed` reviewed down to `branch earned`

```yaml
review_verdict:
  reviewed_status: branch earned
  status_change_note: "Drop the optional leaf to preserve trunk memory."

verification_verdict:
  verified_status: branch earned

closeout:
  final_status: branch earned
  status_change_note: none
  leaf_retained: no
```

## Gate behavior

The gate is blocking. It fails when:

- the YAML artifact is missing
- status-specific fields do not match the claimed promotion level
- review status exceeds the packet status
- closeout status exceeds the reviewed status
- verified status disagrees with closeout
- a downgrade happens without a `status_change_note`
- owner separation is collapsed without an explicit compression rationale
- branch or leaf appears without the required trunk
- closeout claims branch or leaf retention at a lower final status

Use this schema prospectively. Historical rounds do not need retroactive backfill.
