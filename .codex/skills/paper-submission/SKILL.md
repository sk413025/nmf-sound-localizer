---
name: paper-submission
description: Use this skill for manuscript revision, claim-evidence auditing, Nature-facing submission compliance, or other paper-native editing tasks in this repository.
---

# Paper Submission

Use this skill for:

- revising manuscript sections
- auditing claim and evidence alignment
- checking Nature-facing submission compliance
- editing paper legends, tables, and manuscript-facing assets

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/manuscript-contract.md`
- `docs/governance/submission-contract.md`
- `docs/agent-ops/TASK_PACKETS.md`

If the task involves official Nature guidance, also open:

- `docs/nature-communications/nature-communications-submission-requirements.md`

## Workflow

1. Classify the task as revision, audit, or submission check.
2. If a change depends on figure meaning, panel identity, lineage, or paper placement, visually inspect the figure asset first. For `pdf` assets, inspect PNG previews for every page before proceeding.
3. For generated or data-backed figures, inspect the generator or composition code and the upstream evidence or provenance artifacts before changing claims, legends, or numbering.
4. Anchor every claim-level change to figures, Methods, or committed artifacts.
5. Keep Results interpretive and Methods procedural.
6. Route paper-facing figure acceptance to `paper-asset-review` instead of improvising a visual review here.

## Guardrails

- Do not invent evidence or silently strengthen claims.
- Do not revise manuscript figure claims from filenames or captions alone.
- Do not treat a generated figure as understood until the visual asset, code path, and evidence path agree.
- Do not turn this branch into a code-first editing workflow.
- Prefer fewer, clearer steps and fewer, clearer docs when simplifying the paper workflow.
