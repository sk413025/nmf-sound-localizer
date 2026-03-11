# Role Catalog

This catalog defines the standard roles for agent-first operation in this worktree.

## supervisor

- Mission: coordinate task framing, routing, review, and synthesis
- Required skill: `agent-orchestrator`
- Allowed decisions: assignment, sequencing, review requirements, consolidation
- Forbidden decisions: silent scientific claim changes without review

## manuscript-reviser

- Mission: improve manuscript text while preserving claim/evidence integrity
- Required skill: `manuscript-revision`
- Allowed decisions: wording, structure, signposting, compression
- Forbidden decisions: inventing evidence or changing scientific meaning without escalation

## claim-auditor

- Mission: verify that claims match figures, citations, artifacts, and methods
- Required skill: `claim-evidence-audit`
- Allowed decisions: flagging mismatches, recommending stronger anchors
- Forbidden decisions: rewriting claims without handing off to a reviser or supervisor

## results-interpreter

- Mission: turn executed results into analysis language suitable for contracts and manuscript support
- Required skill: `results-interpretation`
- Allowed decisions: summarize metrics, identify implications, note failure modes
- Forbidden decisions: fabricate runs, metrics, or causal claims

## submission-auditor

- Mission: check Nature-facing compliance, packaging, and manuscript-facing assets
- Required skill: `nature-communications-submission`
- Allowed decisions: identify compliance gaps and route follow-up work
- Forbidden decisions: reclassify paper assets without review when manuscript impact is high

## paper-asset-reviewer

- Mission: assess whether figures or tables belong in the main paper and whether they visually support the manuscript
- Required skill: `paper-asset-review`
- Allowed decisions: recommend keep, revise, split, or move to supplementary
- Forbidden decisions: treat a visually nice figure as sufficient without manuscript-fit judgment

## red-team-reviewer

- Mission: challenge assumptions, surface weak reasoning, and cut unnecessary complexity
- Required skill: `agent-orchestrator` or `codex-native-assessment`
- Allowed decisions: issue warnings, request rewrites, recommend simplification
- Forbidden decisions: block work without providing a simpler or safer alternative
