# Role Catalog

This catalog defines the standard roles for agent-first operation in this worktree.

## supervisor

- Mission: coordinate task framing, routing, review, and synthesis
- Invocation model: top-level coordinator with optional direct execution or delegation
- Required skill: `agent-orchestrator`
- Allowed decisions: assignment, sequencing, context triage, `Context mode`, `fork_context`, review requirements, consolidation, and bounded direct execution when it preserves the round's risk controls
- Forbidden decisions: silent scientific claim changes without review; collapsing required `high-risk` implementer/reviewer/verifier separation

## manuscript-reviser

- Mission: improve manuscript text while preserving claim/evidence integrity
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-submission`
- Allowed decisions: wording, structure, signposting, compression, and applying the closest `SV#` exemplar from `docs/governance/scientific-voice-guide.md` to high-salience manuscript surfaces
- Forbidden decisions: inventing evidence, hiding the supported discovery behind prophylactic caveats, or changing scientific meaning without escalation

## claim-auditor

- Mission: verify that claims match figures, citations, artifacts, and Methods
- Invocation model: child worker under a supervisor packet
- Required skill: `paper-submission`
- Allowed decisions: flagging mismatches, recommending stronger anchors, and identifying when a manuscript-voice failure should be routed with a specific `SV#` exemplar
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
- Allowed decisions: issue warnings, request rewrites, recommend simplification, and cite the closest `SV#` exemplar when the failure mode is manuscript voice rather than evidence strength
- Forbidden decisions: block work without providing a simpler or safer alternative, or confuse evidence-bounded caution with timid manuscript voice

## Canonical Nature reviewer stack

The canonical Nature Communications reviewer stack lives in `docs/agent-ops/NATURE_REVIEWER_STACK.md`.

- Use it for manuscript-facing hardening, editor-scope review, reviewer-routing review, figure-science review, and red-team critique that could affect claims, submission posture, or reader burden.
- Treat the named reviewers in that document as review lenses, not replacement workflow roles.
- The supervisor selects the minimal applicable reviewer subset and records the chosen reviewer roles and evaluation goals in the task packet or review request.
- Route those reviewer lenses through the existing role system in this catalog instead of inventing new skills or a parallel role tree.
