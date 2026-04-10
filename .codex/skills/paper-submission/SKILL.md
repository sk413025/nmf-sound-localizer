---
name: paper-submission
description: Use this skill for manuscript revision, claim-evidence auditing, Nature-facing submission compliance, turning executed results into manuscript-ready scientific narrative, or explaining paper logic in cross-disciplinary or plain-language form in this repository. Use it when Codex must preserve whole-manuscript logic, terminology consistency, and natural narrative flow rather than polishing sentences in isolation.
---

# Paper Submission

Use this skill for:

- revising manuscript sections and other paper-facing explanation surfaces
- auditing claim and evidence alignment
- checking Nature-facing submission compliance
- editing paper legends, tables, and paper-facing explanatory assets
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

## Repo-local stance for broader significance

When broader significance is in scope, keep these defaults explicit:

- `promotion-conservative`: do not upgrade a branch into a trunk just because the sentence sounds smoother
- `demotion-forward`: drop or demote weak leaves instead of protecting them through extra explanation
- `editor-memory-first`: preserve the one trunk sentence an editor should remember after one skim
- `reviewer-routing aware`: keep only the broader implication that survives likely reviewer routing
- `anti-slogan`: ban generic “smart environments”, “AI”, or similar utility slogans unless they descend from an earned trunk

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
8. Classify `Architecture scope` before local drafting:
   - `local-salience` for a local high-salience rewrite with no section reweighting
   - `cross-section` for a round that changes more than one section, section bridges, or discovery-versus-tool weight
   - `whole-manuscript` for a full-paper restructuring or any rewrite that re-architects the Results spine
9. For `local-salience` main-manuscript work, run the local architecture pass before drafting. Identify the `old-world belief`, `new-world belief`, `paper protagonist`, `pivot`, `tool role`, `reference-object role`, and target `worldview-shift sentence`.
10. For `cross-section` or `whole-manuscript` work, run the full architecture pass before drafting. Identify the `old-world belief`, `new-world belief`, `paper protagonist`, `supporting actors`, `paper spine map`, `Results section jobs`, `pivot`, `pivot sentence`, `discovery cash-out section`, `tool role`, `reference-object role`, `discovery-vs-tool weight budget`, `second-layer discovery`, `broader-significance status`, `promotion rationale`, `demotion trigger`, `broader-implication trunk`, `downstream-consequence branch`, `optional leaf consequence`, `reviewer-routing survival`, `leaf deletion rule`, `redundancy / breathing risks`, and target `worldview-shift sentence`.
11. Lead with the supported claim floor. Add the evidence boundary after the reader has already learned what the evidence does support.
12. When broader significance is in scope, classify its promotion level before drafting. Use only `core-only`, `second-layer earned`, `branch earned`, or `leaf allowed`.
13. When broader significance is in scope, state it first as a `second-layer discovery` earned by the paper's own evidence before drafting any downstream application sentence.
14. When broader significance is in scope, apply the `Earned-discovery test`, `Boundary-pressure test`, and `Reviewer-routing survival test` before keeping the trunk in any high-salience surface.
15. When an optional leaf is in scope, apply the `Leaf deletion test` before preserving it.
16. For any `high-risk` round with broader significance or cross-disciplinary consequence in scope, create `results/<round_name>/governance_round.yaml` as the canonical machine-readable round artifact and treat `make paper-governance-gate ROUND_DIR=results/<round_name>` as a blocking closeout gate.
17. For local high-salience surfaces, also write one `editor readout sentence`: the sentence a handling editor should still remember after skimming the section once.
18. When broader significance is in scope, also write one `two-takeaway editor readout`: the core discovery sentence plus the second-layer discovery sentence a skimming editor should retain.
19. For local high-salience rewrites, name the closest macro `SV#` exemplar, the closest micro sentence-craft `SV#` exemplar, and the closest architecture `SV#` exemplar from `docs/governance/scientific-voice-guide.md` before drafting. Use them as positive rewrite targets, not just as warning labels.
20. After claim-floor extraction, do a sentence-skeleton pass. For each key sentence, identify the subject, strongest available verb, and explicit consequence.
21. Then do a sentence-energy pass. Split sentences with more than one main causal move unless scientific precision requires them to stay together.
22. Translate important numerical changes into meaning. If a sentence gives a rise, drop, or contrast, state what that change means for the scientific point.
23. Default to cross-disciplinary scientific readability for Nature-facing prose, even when the user only asks for a rewrite.
24. Simplify language without upgrading the evidence level. Prefer scientific-inference prose that moves by `observation -> inference -> bounded conclusion`, with active voice, verb-led clauses, direct cause-effect phrasing, and low noun-stack friction over dense nominalization, front-loaded noun stacks, or manuscript-management language.
25. Run a lab-meeting English test on paper-facing explanation. If a strong PhD student would not naturally say the sentence aloud, rewrite it in more natural scientific English and then restore only the technical precision that matters.
26. Check terminology, comparator labels, mechanism language, protagonist stability, and broader-implication trunk stability against the surrounding sections before finalizing.
27. Perform a coherence pass on transitions, paragraph openings, paragraph endings, and section jobs so the edited text reads as one manuscript rather than a local patch.
28. If `Architecture scope` is `cross-section` or `whole-manuscript`, verify that section order, discovery cash-out, tool-versus-discovery weight, second-layer discovery, and the chosen broader-significance status are carried consistently across the in-scope surfaces before closeout.
29. Keep Results interpretive and Methods procedural.
30. Route paper-facing figure acceptance to `paper-asset-review` instead of improvising a visual review here.

## Required output bundle

When this skill proposes or performs a paper-facing explanation revision, the output must be verifier-ready.

Include:

- `Architecture scope:` set to `local-salience`, `cross-section`, or `whole-manuscript`
- `Architecture ledger:` naming the `old-world belief`, `new-world belief`, `paper protagonist`, and any additional architecture fields required by the chosen `Architecture scope`
- `Broader-significance status:` when broader significance is in scope, naming whether the prose is `core-only`, `second-layer earned`, `branch earned`, or `leaf allowed`
- `Exact revised text:` with the exact replacement prose, not a summary
- `Before anchor:` and `After anchor:` quoting the neighboring manuscript text that brackets the change
- `Editor readout sentence:` stating the one-sentence discovery a skimming editor should retain from the revised surface
- `Two-takeaway editor readout:` when broader significance is in scope, stating the core discovery sentence and the second-layer discovery sentence a skimming editor should retain
- `Applied exemplar(s):` naming the `SV#` pair or pairs that guided the rewrite on high-salience surfaces
- `Sentence craft fixes:` naming any noun-stack, causal-glue, diction, or static-verb fixes applied on the in-scope surface
- `Unresolved promised joints:` listing any requested transition, claim linkage, or downstream manuscript connection that remains unfinished
- `Architecture evidence map:` for any `cross-section` or `whole-manuscript` round, tying title, the `front-door preload sentence` when required, pivot sentence, discovery cash-out sentence, and Discussion opening to the intended spine
- `No-bolt-on test:` for any broader-significance round, stating how the verifier should confirm that downstream consequence depends on the second-layer discovery
- `Verifier mode:` stating how the verifier should check the change; use `text-diff` when the task is local prose replacement and name any stronger mode when figure, evidence, or claim support must also be re-checked
- `Canonical round artifact:` for any `high-risk` broader-significance round, naming `results/<round_name>/governance_round.yaml`
- `Semantic gate:` for any `high-risk` broader-significance round, naming `make paper-governance-gate ROUND_DIR=results/<round_name>`

For `high-risk` broader-significance rounds, keep the human-readable output lean and treat `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md` as the only canonical field inventory for the machine-readable artifact.

If no manuscript text was changed, state that explicitly in `Delivered items:` or `Deferred or dropped items:` instead of implying completion through plan language alone.

## Reviewer subset and acceptance surface

Use `docs/agent-ops/NATURE_REVIEWER_STACK.md` as the canonical reviewer-lens source.
Default to the minimal reviewer subset that matches the manuscript change:

- `cross-disciplinary-readability reviewer` for every paper-facing prose-hardening round
- `cross-disciplinary-readability reviewer`, `narrative-flow reviewer`, and `cognitive-load reviewer` for most Nature-facing prose revision, explanation, and coherence passes
- `handling-editor-scope reviewer` and `reviewer-routing reviewer` when title, abstract, Results framing, Discussion framing, or paper-level positioning could change editorial fit
- `physical-mechanism reviewer`, `sparse-inverse-problem-comparator reviewer`, and `statistics-evidence reviewer` when wording changes touch mechanism language, comparator logic, or evidence strength

Acceptance surface for this skill:

- the revised text remains legible to cross-disciplinary readers
- the revised text preserves one stable paper protagonist and keeps tools in a supporting role unless the packet explicitly declares a method paper
- the revised text supports a clear paper pivot and does not read like an experiment log
- the revised text uses scientific inference rather than rebuttal, guidebook, curator, or manuscript-management phrasing
- the revised text uses active voice and simple cause-effect sentence structure where scientifically appropriate, without translation-like noun stacking
- the revised text passes a sentence-energy check: main sentences have clear subjects, direct verbs, explicit consequence, and natural enough diction for one-pass reading
- paragraph and section flow still reads as one manuscript-level argument
- mechanism, comparator, and evidence wording stay within the support shown by figures, Methods, and artifacts
- unresolved reviewer-stack risks are named and escalated instead of hidden inside cleaner prose

## Manuscript-management blacklist

Treat the following as rewrite triggers on paper-facing explanation surfaces unless a narrow procedural context truly requires them:

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
- "this reads like a technical report"
- "the paper still has no pivot"
- "the method feels like the main character"

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
- Do not preserve a solver-, calibration-, or reference-object-centered framing when the discovery should be the protagonist.
- Do not let tool validation consume more narrative mass than the paper-level finding it supports.
- Do not leave terminology drift, comparator drift, or abrupt transitions after a local rewrite.
- Do not assume a Nature Communications reader shares the subfield's shorthand or unstated background.
- Do not preserve nominalization-heavy or front-loaded noun-stack phrasing just because the terminology itself is correct.
- Do not preserve rebuttal-style, guidebook-style, curator-style, panel-choreography, or manuscript-management language just because the sentence is now active voice.
- Do not turn this branch into a code-first editing workflow.
- Prefer fewer, clearer steps and fewer, clearer docs when simplifying the paper workflow.
