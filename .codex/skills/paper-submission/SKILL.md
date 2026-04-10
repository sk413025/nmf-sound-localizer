---
name: paper-submission
description: Use this skill for manuscript revision, claim-evidence auditing, Nature-facing submission compliance, turning executed results into manuscript-ready scientific narrative, or explaining paper logic in cross-disciplinary or plain-language form in this repository. Use it when Codex must preserve whole-manuscript logic, terminology consistency, and natural narrative flow rather than polishing sentences in isolation.
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
- improving whole-manuscript coherence, transitions, and narrative flow

## Required first step

Open and follow:

- `START_HERE_AGENT.md`
- `docs/governance/manuscript-contract.md`
- `docs/governance/scientific-voice-guide.md`
- `docs/governance/submission-contract.md`
- `docs/agent-ops/NATURE_REVIEWER_STACK.md`
- `docs/agent-ops/TASK_PACKETS.md`

If the task involves official Nature guidance, also open:

- `docs/nature-communications/nature-communications-submission-requirements.md`

If the task is about converting results into stronger manuscript logic, cross-disciplinary explanation, or plain-language scientific narrative, also read:

- [references/results-to-narrative.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/.codex/skills/paper-submission/references/results-to-narrative.md)

## Workflow

1. Classify the task as revision, audit, submission check, results-to-narrative translation, or coherence pass.
2. Before editing locally, identify the paragraph or section's role in the paper-level argument and inspect the surrounding text it must connect to.
3. If a change depends on figure meaning, panel identity, lineage, or paper placement, visually inspect the figure asset first. For `pdf` assets, inspect PNG previews for every page before proceeding.
4. For generated or data-backed figures, inspect the generator or composition code and the upstream evidence or provenance artifacts before changing claims, legends, or numbering.
5. Anchor every claim-level change to figures, Methods, or committed artifacts.
6. When translating analysis into prose, keep three layers distinct:
   - what the executed results directly support
   - what is a current best mechanism or candidate explanation
   - what remains a frontier or open question
7. For each revised paragraph, explicitly identify the `claim floor`, `claim ceiling`, and `evidence boundary` before drafting the final prose.
8. Lead with the supported claim floor. Add the evidence boundary after the reader has already learned what the evidence does support.
9. For high-salience surfaces, also write one `editor readout sentence`: the sentence a handling editor should still remember after skimming the section once.
10. For high-salience rewrites, name the closest `SV#` exemplar from `docs/governance/scientific-voice-guide.md` before drafting. Use it as a positive rewrite target, not just as a warning label.
11. Default to cross-disciplinary scientific readability for Nature-facing prose, even when the user only asks for a rewrite.
12. Simplify language without upgrading the evidence level. Prefer scientific-inference prose that moves by `observation -> inference -> bounded conclusion`, with active voice, verb-led clauses, and direct cause-effect phrasing over dense nominalization, front-loaded noun stacks, or manuscript-management language.
13. Check terminology, comparator labels, and mechanism language against the surrounding sections before finalizing.
14. Perform a coherence pass on transitions, paragraph openings, and paragraph endings so the edited text reads as part of one natural manuscript rather than a local patch.
15. Keep Results interpretive and Methods procedural.
16. Route paper-facing figure acceptance to `paper-asset-review` instead of improvising a visual review here.

## Required output bundle

When this skill proposes or performs a manuscript-facing revision, the output must be verifier-ready.

Include:

- `Exact revised text:` with the exact replacement prose, not a summary
- `Before anchor:` and `After anchor:` quoting the neighboring manuscript text that brackets the change
- `Editor readout sentence:` stating the one-sentence discovery a skimming editor should retain from the revised surface
- `Applied exemplar(s):` naming the `SV#` pair or pairs that guided the rewrite on high-salience surfaces
- `Unresolved promised joints:` listing any requested transition, claim linkage, or downstream manuscript connection that remains unfinished
- `Verifier mode:` stating how the verifier should check the change; use `text-diff` when the task is local prose replacement and name any stronger mode when figure, evidence, or claim support must also be re-checked

If no manuscript text was changed, state that explicitly in `Delivered items:` or `Deferred or dropped items:` instead of implying completion through plan language alone.

## Reviewer subset and acceptance surface

Use `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source.
Default to the minimal reviewer subset that matches the manuscript change:

- `cross-disciplinary-readability reviewer`, `narrative-flow reviewer`, and `cognitive-load reviewer` for most Nature-facing prose revision, explanation, and coherence passes
- `handling-editor-scope reviewer` and `reviewer-routing reviewer` when title, abstract, Results framing, Discussion framing, or paper-level positioning could change editorial fit
- `physical-mechanism reviewer`, `sparse-inverse-problem-comparator reviewer`, and `statistics-evidence reviewer` when wording changes touch mechanism language, comparator logic, or evidence strength

Acceptance surface for this skill:

- the revised text remains legible to cross-disciplinary readers
- the revised text uses scientific inference rather than rebuttal, guidebook, curator, or manuscript-management phrasing
- the revised text uses active voice and simple cause-effect sentence structure where scientifically appropriate, without translation-like noun stacking
- paragraph and section flow still reads as one manuscript-level argument
- mechanism, comparator, and evidence wording stay within the support shown by figures, Methods, and artifacts
- unresolved reviewer-stack risks are named and escalated instead of hidden inside cleaner prose

## Manuscript-management blacklist

Treat the following as rewrite triggers on manuscript-facing surfaces unless a narrow procedural context truly requires them:

- `This section asks...`
- `We use Fig. X...`
- `Panel X shows...` when the scientific actor, observation, or intervention can carry the sentence
- `used in the paper`
- `summarized in manuscript Fig...`
- repo or provenance bookkeeping in manuscript prose such as `committed`, `executed`, or code-switch framing between manuscript and repository state

Replace these patterns with evidence-led phrasing that states what was observed, what changed, what was compared, and what bounded conclusion follows.

Common trigger phrases:

- "write this as Nature Communications prose"
- "turn these results into a stronger scientific narrative"
- "explain this for cross-disciplinary readers"
- "explain this in plain language"
- "rewrite this so non-specialists can still follow it"
- "make this read naturally"
- "the logic feels jumpy"
- "improve the flow between these paragraphs"

Additional self-diminishing trigger phrases:

- `without upgrading`
- `descriptive rather than`
- `remains positive` without comparator context
- abstract endings led by `pathway`, `constraint`, or equivalent caveat-first framing
- section openings that recap the previous result instead of posing the next scientific question or stating the next supported finding

## Guardrails

- Do not invent evidence or silently strengthen claims.
- Do not revise manuscript figure claims from filenames or captions alone.
- Do not treat a generated figure as understood until the visual asset, code path, and evidence path agree.
- Do not turn a candidate mechanism, factor audit, or descriptive trend into a settled law just because the prose sounds cleaner.
- Do not optimize one paragraph in isolation if it breaks the surrounding logic.
- Do not leave terminology drift, comparator drift, or abrupt transitions after a local rewrite.
- Do not assume a Nature Communications reader shares the subfield's shorthand or unstated background.
- Do not preserve nominalization-heavy or front-loaded noun-stack phrasing just because the terminology itself is correct.
- Do not preserve rebuttal-style, guidebook-style, curator-style, panel-choreography, or manuscript-management language just because the sentence is now active voice.
- Do not turn this branch into a code-first editing workflow.
- Prefer fewer, clearer steps and fewer, clearer docs when simplifying the paper workflow.
