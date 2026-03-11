# Codex Native Assessment Pack

This folder is the launch kit for evaluating how this branch can become more `codex-native` and `agent-native` without drifting away from its primary role as a manuscript-first Nature Communications workspace.

## Purpose

Use this pack when you want multiple Codex agents to evaluate:

- whether the branch structure fits Codex's native workflow model
- how well the branch supports multi-agent manuscript work
- what should change in `AGENTS.md`, local skills, manuscript workflow, evidence tracking, and paper automation

This pack is for **assessment and planning**, not for direct paper editing.

## Required reading order

1. `docs/codex-native-assessment/CODEX_CAPABILITY_BASELINE.md`
2. `docs/codex-native-assessment/SHARED_RUBRIC.md`
3. `docs/codex-native-assessment/SUPERVISOR_PLAYBOOK.md`
4. the role packet under `docs/codex-native-assessment/roles/`
5. `docs/codex-native-assessment/REPORT_TEMPLATE.md`

## Source-of-truth repository context

All agents using this pack should inspect these files before drawing conclusions:

- `AGENTS.md`
- `paper/manuscript/manuscript.md`
- `paper/manuscript/REVISION_CHECKLIST.md`
- `figures/FIGURE_REGISTRY.md`
- `docs/nature-communications/nature-communications-submission-requirements.md`
- `scripts/paper/build_docx.sh`
- `scripts/paper/resolve_tbd.py`
- `scripts/paper/verify_provenance.py`

## Launch sequence

1. Supervisor completes the preflight capability baseline check.
2. Supervisor sends the shared rubric and one role packet to each specialist.
3. Four specialist agents run in parallel:
   - manuscript workflow
   - agent onboarding
   - traceability
   - automation and validation
4. Supervisor performs the first quality gate.
5. Governance and red-team agent reviews the accepted reports.
6. Supervisor publishes the consolidated readiness report and prioritized roadmap.

## Expected outputs

The full assessment should produce:

- one capability baseline
- four specialist reports
- one governance and red-team report
- one supervisor warning log
- one consolidated readiness report
- one prioritized implementation roadmap

## Operating principle

The branch should be judged against **Codex-native collaboration for manuscript work**, not against a generic automation ideal and not against a figure-only workflow.
