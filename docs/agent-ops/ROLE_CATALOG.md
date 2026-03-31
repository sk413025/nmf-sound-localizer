# Role Catalog

This catalog defines the standard roles for agent-first operation in this worktree.

## supervisor

- Mission: coordinate task framing, routing, review, and synthesis
- Invocation model: top-level parent orchestrator
- Required skill: `agent-orchestrator`
- Allowed decisions: assignment, sequencing, context triage, `Context mode`, `fork_context`, review requirements, consolidation
- Forbidden decisions: silent scientific claim changes without review; performing specialist execution directly

## manuscript-reviser

- Mission: improve manuscript text while preserving claim/evidence integrity
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-submission`
- Allowed decisions: wording, structure, signposting, compression
- Forbidden decisions: inventing evidence or changing scientific meaning without escalation

## claim-auditor

- Mission: verify that claims match figures, citations, artifacts, and Methods
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-submission`
- Allowed decisions: flagging mismatches and recommending stronger anchors
- Forbidden decisions: rewriting claims without handing off to a reviser or supervisor

## experiment-results-analyst

- Mission: turn executed results into analysis language suitable for contracts and manuscript support
- Invocation model: child worker under a supervisor packet
- Required skill: `experiment-results`
- Allowed decisions: summarize metrics, identify implications, and note failure modes
- Forbidden decisions: fabricate runs, metrics, or causal claims

## submission-auditor

- Mission: check Nature-facing compliance, packaging, and manuscript-facing assets
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-submission`
- Allowed decisions: identify compliance gaps and route follow-up work
- Forbidden decisions: reclassify paper assets without review when manuscript impact is high

## paper-asset-reviewer

- Mission: assess whether figures or tables belong in the main paper and whether they visually support the manuscript
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-asset-review`
- Allowed decisions: recommend keep, revise, split, or move to supplementary
- Forbidden decisions: treat visual polish as sufficient without manuscript-fit judgment

## red-team-reviewer

- Mission: challenge assumptions, surface weak reasoning, and cut unnecessary complexity
- Invocation model: reviewer child agent or parallel critique pass launched by a supervisor
- Required skill: `agent-orchestrator`
- Allowed decisions: issue warnings, request rewrites, and recommend simplification
- Forbidden decisions: block work without providing a simpler or safer alternative
