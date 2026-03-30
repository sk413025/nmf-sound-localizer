---
name: paper-submission
description: Use this skill for manuscript revision, claim-evidence auditing, Nature-facing submission compliance, turning executed results into manuscript-ready scientific narrative, or explaining paper logic in cross-disciplinary or plain-language form in this repository.
---

# Paper Submission

Use this skill for:

- revising manuscript sections
- auditing claim and evidence alignment
- checking Nature-facing submission compliance
- editing paper legends, tables, and manuscript-facing assets
- translating executed analysis into manuscript-ready claims
- rewriting technical findings into cross-disciplinary scientific prose
- explaining results in plain language without overstating certainty

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/manuscript-contract.md`
- `docs/governance/submission-contract.md`
- `docs/agent-ops/TASK_PACKETS.md`

If the task involves official Nature guidance, also open:

- `docs/nature-communications/nature-communications-submission-requirements.md`

If the task is about converting results into stronger manuscript logic, cross-disciplinary explanation, or plain-language scientific narrative, also read:

- [references/results-to-narrative.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/paper-submission/references/results-to-narrative.md)

## Workflow

1. Classify the task as revision, audit, submission check, or results-to-narrative translation.
2. If a change depends on figure meaning, panel identity, lineage, or paper placement, visually inspect the figure asset first. For `pdf` assets, inspect PNG previews for every page before proceeding.
3. For generated or data-backed figures, inspect the generator or composition code and the upstream evidence or provenance artifacts before changing claims, legends, or numbering.
4. Anchor every claim-level change to figures, Methods, or committed artifacts.
5. When translating analysis into prose, keep three layers distinct:
   - what the executed results directly support
   - what is a current best mechanism or candidate explanation
   - what remains a frontier or open question
6. If the user asks for cross-disciplinary or plain-language explanation, simplify the language without upgrading the evidence level.
7. Keep Results interpretive and Methods procedural.
8. Route paper-facing figure acceptance to `paper-asset-review` instead of improvising a visual review here.

Common trigger phrases:

- "write this as Nature Communications prose"
- "turn these results into a stronger scientific narrative"
- "explain this for cross-disciplinary readers"
- "explain this in plain language"
- "rewrite this so non-specialists can still follow it"

## Guardrails

- Do not invent evidence or silently strengthen claims.
- Do not revise manuscript figure claims from filenames or captions alone.
- Do not treat a generated figure as understood until the visual asset, code path, and evidence path agree.
- Do not turn a candidate mechanism, factor audit, or descriptive trend into a settled law just because the prose sounds cleaner.
- Do not turn this branch into a code-first editing workflow.
- Prefer fewer, clearer steps and fewer, clearer docs when simplifying the paper workflow.
