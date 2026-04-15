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
- write memory in positive branch-state language when possible: store the supported posture, active constraint, or stable failure mode directly instead of narrating anticipated objections
- when scope needs tightening, narrow the stored lesson itself rather than phrasing the memory as `not X`, `should not be read as`, or other reviewer-prebuttal language
- when a stored manuscript lesson concerns term routing or reader burden, prefer recording it as a bidirectional no-translation posture: the paper should read naturally from `Introduction/Results` into `Methods` and back again without forcing readers to invent a private synonym map

## Memory Update Review

- child agents may propose memory candidates, but only the top-level agent may decide what is written under `.codex/memory/`
- trigger a weak-agent anti-log review before any substantive update to `CURRENT_BRANCH_MEMORY.md` or any new or rewritten note under `archive/`
- do not trigger that review for typo fixes, link fixes, title changes, or format-only cleanup

Give the weak agent only:

- the current memory file or files being considered
- the candidate lines or bullets proposed for storage
- a short summary of the completed round that produced those candidates

The weak agent is an advisory classifier, not a policy authority. Its output should stay limited to:

- `Keep as memory`
- `Rewrite as branch state`
- `Do not store in memory`
- placement advice: `current`, `archive`, or `nowhere`

Use these fixed questions when reviewing a candidate:

- is this line recording branch state, or only a chronology of what happened
- would this still help the next top-level agent route or judge the branch one week from now
- if dates, step order, and session sequencing disappear, does the line still hold

## Promotion And Placement

- only promote conclusions that came from a completed round, closeout, or high-salience review and are already stable enough to recur
- do not promote live speculation, unresolved hunches, or unverified discussion fragments
- only promote a candidate when it affects branch posture, routing, an active front, or a stable failure mode and still holds after chronology is removed
- `CURRENT_BRANCH_MEMORY.md` stores only the current live branch state
- `archive/` stores only distilled stable lessons from a session cluster that may recur later
- if a candidate fits neither `current` nor `archive`, do not store it anywhere under `.codex/memory/`

When rewriting a chronology-heavy candidate, compress it into one of these forms:

- `stable lesson`
- `active front`
- `failure mode`
- `open unresolved item`

If the line cannot survive that compression without preserving the timeline, drop it.

When rewriting a defensive or objection-shaped candidate, compress it into one of these forms:

- `affirmative branch posture`
- `scope-containing constraint`
- `stable failure mode`

If the line only works as an anticipated rebuttal, rewrite it or drop it.

## Supersession And Cleanup

- treat `CURRENT_BRANCH_MEMORY.md` as live state, not as branch history
- when a new stable lesson replaces an older one, update the current brief by superseding the old state instead of stacking both versions
- on every substantive current-brief update, remove open items that are no longer open, active fronts that are no longer active, and failure modes that no longer drive routing
- if the current brief is losing one-pass readability, compact it before adding more bullets
- if the branch undergoes a major reframe, prefer rewriting `CURRENT_BRANCH_MEMORY.md` from the new live posture instead of patching the old worldview line by line
