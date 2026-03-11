# Shared Rubric

All specialist reports must score the branch on a `0-5` scale for the categories below and justify each score with concrete repository evidence.

## Score meanings

- `0`: absent
- `1`: ad hoc
- `2`: partially present but unreliable
- `3`: usable with friction
- `4`: strong
- `5`: decision-complete and agent-ready

## Scored categories

### C1 Codex-native alignment
How well the branch structure matches actual Codex capabilities and workflow primitives.

### C2 AGENTS leverage
How effectively the branch uses persistent repository instructions instead of leaving behavior implicit.

### C3 Skill fit
Whether repeated high-value tasks should be encoded as reusable local skills, and whether current skill usage is discoverable.

### C4 Multi-agent fit
How well the work can be split across multiple agents with minimal collision, rework, or ambiguity.

### C5 Manuscript-first alignment
Whether the repository clearly treats manuscript work as the main target and figure generation as supporting infrastructure.

### C6 New-agent discoverability
How quickly a fresh agent can find the right entrypoints, constraints, and source-of-truth files.

### C7 Claim and evidence traceability
How reliably an agent can map a scientific claim to figures, artifacts, provenance, and manuscript text.

### C8 Deterministic commands and validation
Whether the branch provides stable commands or checks for build, validation, and submission-facing tasks.

### C9 Governance and acceptance criteria
Whether agent work can be reviewed against explicit acceptance gates instead of taste-based feedback.

### C10 Parallel-agent safety
Whether multiple agents can work concurrently without accidental overlap or contradictory changes.

## Report requirements

Every specialist report must include:

- at least 3 evidence-backed findings
- at least 3 actionable recommendations
- at least 1 recommendation affecting manuscript workflow directly
- at least 1 recommendation that can later be automated
- 3 quick wins
- 3 structural changes
- blockers
- confidence level

## Evidence standard

Evidence may cite:

- repository files
- command surfaces
- local scripts
- documented workflow gaps
- Codex capability baseline items

Recommendations without evidence should be treated as weak and subject to supervisor warning.
