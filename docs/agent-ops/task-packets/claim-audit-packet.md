# Task Packet: Claim Audit

## Objective

Check whether a manuscript claim is adequately supported by figures, citations, methods, and artifacts.

## Source-of-truth files

- `paper/manuscript/manuscript.md`
- `figures/FIGURE_REGISTRY.md`
- `DATA_PROVENANCE.md`
- `docs/governance/manuscript-contract.md`

## Required skill

- `claim-evidence-audit`

## Role

- `claim-auditor`

## Non-goals

- rewriting the paper directly
- inventing missing evidence

## Required outputs

- supported claims
- weak or unsupported claims
- concrete recommendations

## Acceptance criteria

- each flagged issue cites evidence
- each recommendation names the next role

## Escalation conditions

- contradictory evidence
- major figure/manuscript mismatch

## Handoff target

- `manuscript-reviser` or `supervisor`
