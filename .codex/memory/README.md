# Branch Memory

This directory holds the repo-local derived memory layer for the Nature Communications manuscript branch.

## Purpose

- keep a short, reusable branch-state brief for top-level agent routing
- preserve stable lessons from recent high-salience rounds without reopening raw session transcripts
- point agents to the right deeper notes only when the current task needs them

## Precedence

This directory is not a second constitution.

When instructions or claims conflict, follow:

1. `AGENTS.md`
2. canonical task documents such as `paper/manuscript/manuscript.md` and Nature requirements
3. contracts under `docs/governance/`
4. operating model docs under `docs/agent-ops/`
5. `.codex/memory/`
6. raw sessions, ad hoc notes, and archive material

Memory files here may summarize branch state, but they must not create new policy, new canonical schemas, or a second governance vocabulary.

## Read Rule

- top-level agents read `CURRENT_BRANCH_MEMORY.md` first for manuscript, governance, strategy, and other branch-shaping tasks
- open archive notes only when `CURRENT_BRANCH_MEMORY.md` links them as relevant to the current task
- child agents should receive memory-relevant context through task packets or parent summaries rather than bulk-loading the archive

## Update Rule

- keep `CURRENT_BRANCH_MEMORY.md` short enough for one-pass reading
- move session-derived detail into `archive/` rather than letting the current brief expand into a ledger
- record only stable lessons, current branch priorities, and unresolved fronts that are still active
- do not paste long session transcripts into this directory
