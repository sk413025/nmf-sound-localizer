# Supervisor Playbook

This file defines how the supervisor agent should run the Codex-native assessment.

## Mission

Run the assessment as a manuscript-first Codex collaboration exercise. The supervisor is responsible for correctness, scope control, deduplication, and final synthesis.

## Phase 0: Preflight

Before any specialist starts:

1. Read `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`.
2. Confirm it still matches the local Codex environment.
3. Refresh the local CLI facts if needed:
   - `codex --help`
   - `codex features list`
4. Send the baseline and `docs/codex-native-assessment/SHARED_RUBRIC.md` to every specialist.

No specialist work should begin until this preflight is complete.

## Launch order

Launch these four specialist roles in parallel first:

- manuscript workflow
- agent onboarding
- traceability
- automation and validation

Launch the governance and red-team role only after the first four reports have passed the initial quality gate.

## Warning codes

- `WARN_SCOPE`: the report treats the branch as figure-first or generic code-first
- `WARN_EVIDENCE`: findings are not tied to concrete repository evidence
- `WARN_ACTIONABILITY`: recommendations are too vague to implement
- `WARN_GOVERNANCE`: no acceptance criteria or ownership model is proposed
- `WARN_DUPLICATION`: the report duplicates another role without adding value
- `WARN_CAPABILITY_MISMATCH`: the report assumes Codex abilities that are not grounded in the baseline
- `WARN_NATIVE_MISSED`: the report ignores AGENTS, local skills, or other existing Codex primitives
- `WARN_PRODUCT_DRIFT`: the report talks about agent-native behavior without staying specific to Codex

## First quality gate

A specialist report passes only if all conditions hold:

- cites the capability baseline
- contains at least 3 evidence-backed findings
- contains at least 3 actionable recommendations
- distinguishes quick wins from structural changes
- includes at least 1 manuscript workflow recommendation
- includes at least 1 recommendation that can later be validated automatically

If any condition fails, return the report once for rewrite.

## Cross-review

After the first four reports pass:

- manuscript workflow reviews automation and traceability
- agent onboarding reviews governance recommendations
- governance and red-team reviews all accepted reports

The goal is conflict detection and simplification, not restarting discovery.

## Final outputs

The supervisor must publish:

- a warning and adjudication log
- a consolidated readiness report
- a prioritized roadmap using `P0`, `P1`, and `P2`

## Final synthesis checklist

The final synthesis is not complete until it includes:

- the current Codex-native gap map
- the current manuscript-native gap map
- the minimum viable first implementation set
- the dependency order for follow-on implementation work
- clear ownership boundaries for future execution agents
