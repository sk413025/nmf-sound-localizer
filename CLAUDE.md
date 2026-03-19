# CLAUDE.md — Quick-start rules for Claude on this branch

This is a Nature Communications manuscript worktree. The primary deliverable is `paper/manuscript/manuscript.md`.

## Read these before any manuscript edit

1. **AGENTS.md** — Branch constitution and governance precedence.
2. **docs/governance/manuscript-contract.md** — Manuscript editing rules, including the "Nature Communications prose discipline" section that codifies style constraints derived from published Nature Comm articles.
3. **docs/nature-communications/nature-communications-submission-requirements.md** — Journal-specific formatting, word limits, and figure specifications.

## Top-level constraints

- This branch is manuscript-first. Code and figures are subordinate.
- Main text structure: Introduction → Results → Discussion → Methods.
- Target: main text ~5,000 words, Methods ≤ 3,000 words, Abstract ≤ 200 words, ≤ 70 references, ≤ 10 display items.

## Most common AI writing pitfalls (details in manuscript-contract.md)

These are patterns that AI agents repeatedly introduce unless explicitly warned:

- **No bold emphasis in body text.** Bold is only for figure legend titles and Methods sub-section headings.
- **No pseudo-subheadings in Discussion.** Nature Comm does not permit Discussion subheadings.
- **Display math for core equations.** Key formulas in Results get numbered display equations, not inline-only math.
- **Define symbols at first use in Results.** Every symbol gets an inline "where" clause where it first appears; do not assume the reader reads Methods first.
- **Vary contribution verbs.** Do not repeat "We show" consecutively; alternate with demonstrate, establish, reveal.
- **No dramatic tone words.** Avoid catastrophically, strikingly, crucially, remarkably in body text and section titles.
- **Limit em dashes** to one per paragraph; prefer commas or separate sentences.
- **Limit parenthetical asides** to ~10 words; rewrite longer ones as proper clauses.
- **Limit numbers to 3 per sentence.** Point extra values to figure panels.
- **Cross-section literature consistency.** Every reference in the Discussion must be set up in the Introduction.
- **No standalone Road map paragraph.** Fold paper-organization cues into the contribution statement.

## Governance hierarchy

When instructions conflict: AGENTS.md > manuscript-contract.md > submission-requirements.md > local scripts.
