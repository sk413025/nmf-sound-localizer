# Results To Narrative

Use this reference when the user asks for:

- manuscript-ready wording from executed analyses
- stronger scientific narrative for Results or Discussion
- cross-disciplinary explanation
- plain-language explanation that remains scientifically honest
- better whole-manuscript flow or coherence
- paper-facing legend, review-note, or analysis-summary prose that may later be promoted into the manuscript

Default audience for Nature-facing prose:

- scientifically literate readers outside the immediate subfield
- readers who should not need to read Methods first to follow the main argument

Keep these layers separate:

1. Directly supported finding
   - what the executed artifact really shows
2. Candidate mechanism
   - the current best explanatory interpretation
3. Frontier or open question
   - what the paper cannot yet claim as established

Architecture layer for full-paper work:

1. Old-world belief
   - what the reader currently assumes
2. New-world belief
   - what the paper wants the reader to understand instead
3. Paper protagonist
   - the discovery, organizing principle, or phenomenon the paper is really about
4. Supporting actors
   - tools, reference objects, calibration schemes, and assays that reveal or test that protagonist
5. Pivot
   - the section or transition where the reader's model of the system actually changes
6. Results section jobs
   - what each Results section contributes to the primary cognitive shift and, when earned, the second-layer discovery
7. Discovery cash-out
   - where the paper-level discovery becomes unavoidable rather than optional
8. Discovery-vs-tool weight budget
   - how much narrative mass the discovery gets versus the tool that serves it
9. Second-layer discovery
   - the broader paper-level inference earned by the discovery itself
10. Broader-significance status
   - whether the broader implication is still `core-only`, has reached `second-layer earned`, has earned one branch, or may carry an optional leaf
11. Promotion rationale
   - why the current evidence earns that level rather than a lower one
12. Demotion trigger
   - what evidence or routing weakness would force a downgrade
13. Broader-implication trunk
   - the shortest sentence that states that second-layer discovery
14. Downstream-consequence branch
   - the nearest bounded consequence that follows from an already earned trunk
15. Optional leaf consequence
   - a weaker, more distant implication that should stay visibly subordinate
16. Reviewer-routing survival
   - whether the promoted trunk or branch survives likely reviewer routing
17. Leaf deletion rule
   - when the leaf should be deleted to protect trunk memorability
18. Front-door preload sentence
   - the abstract-tail or Introduction-ending sentence that plants the trunk before Discussion
19. No-bolt-on test
   - remove the branch and the trunk should still land; remove the trunk and the branch should collapse
20. Two-takeaway editor readout
   - the core discovery sentence plus the second-layer discovery sentence an editor should retain

Promotion gate for broader significance:

1. `Earned-discovery test`
   - can the broader implication still be stated with the paper's own discovery actor as subject
2. `Boundary-pressure test`
   - if you rewrite the implication in the most conservative truthful form, is it still worth front-door or Discussion-trunk placement
3. `Reviewer-routing survival test`
   - would the proposed promotion still read as evidence-earned to the likely reviewer community
4. `Leaf deletion test`
   - if deleting the optional leaf improves memory for the trunk or branch, delete the leaf

Working defaults:

- `promotion-conservative`
- `demotion-forward`
- `editor-memory-first`
- `reviewer-routing aware`
- `anti-slogan`

Scientific voice rule:

- write with earned confidence
- use `docs/governance/scientific-voice-guide.md` as the canonical sentence-shape and exemplar reference
- say the supported discovery sentence plainly before naming the limitation
- if a paragraph sounds safer only because it is weaker, it is probably worse

Editor-first test:

- after drafting, ask what a handling editor would underline in 15 seconds
- if the answer is a caveat, setup detail, or workflow description, rewrite the paragraph with the closest `SV#` exemplar as the model
- if the answer is the discovery sentence, keep going
- if the answer is a tool, workflow, or reference object rather than the discovery, the protagonist has drifted
- if the answer is a distant use case rather than the second-layer discovery, the broader implication is over-promoted

For cross-disciplinary explanation:

- preserve the causal structure
- reduce jargon before removing technical precision
- define the physical role of each quantity before naming the metric
- state the physical question before the model, metric, or architecture label
- prefer short bridge phrases over symbol-heavy prose
- keep one main causal move per sentence
- when you report a number, also report what the number means
- unpack noun stacks into clauses when they slow first-pass reading
- prefer the wording you could say naturally in lab meeting over a more compressed formal synonym
- treat legends, reviewer-facing explanation, and paper-support analysis summaries as part of the same sentence-energy surface when they may later flow into the manuscript

For plain-language explanation:

- keep the scientific hierarchy intact
- do not replace uncertainty with confidence
- do not replace confidence with timidity
- say "suggests" or "currently points to" when the evidence is descriptive or limited

For whole-manuscript coherence:

- identify the old-world belief and the new-world belief before rewriting section flow
- identify the paper protagonist before rewriting local paragraphs
- identify the Results pivot before deciding how much weight to give tool-validation sections
- write the Results section jobs before approving a whole-paper flow
- identify the discovery cash-out section before letting a final Results section read like a mere extension
- compress or merge tool sections that answer one bounded scientific question instead of narrating them as independent discoveries
- identify what logical job the paragraph or section performs
- bridge from the previous known point before introducing the next claim
- keep terminology, comparator names, and mechanism labels stable across sections
- make each paragraph advance one main logical step
- end paragraphs by showing why the point matters for the next step of the paper
- if a sentence only works with hidden local context, rewrite it so the link is explicit
- if a section only works because the reader remembers the experiment chronology, rewrite it around what the reader learns next instead
- if the paper still sounds like a series of analyses rather than one model update, the spine is wrong even if each section is locally strong

Preferred answer pattern:

1. one-sentence conclusion
2. what the result directly supports
3. what mechanism it points toward
4. what it still does not prove

Preferred whole-paper pattern:

1. old-world belief
2. surprising observation
3. governing principle
4. second-layer discovery
5. bounded downstream consequence
6. boundary

Preferred promotion order for broader significance:

1. core-only
2. second-layer earned
3. branch earned
4. leaf allowed

Promote slowly. Demote quickly.

Worked example for this branch:

- `core discovery`
  - ordinary passive objects carry a recurring local directional code
- `second-layer discovery`
  - directional sensing can partly reside in passive structure rather than only in dedicated hardware
- `bounded branch`
  - object-mediated directional neighborhoods can support event-origin triage or front-end filtering
- `optional leaf`
  - farther selective-listening or ambient-intelligence interpretations
- `leaf status`
  - removable without weakening the trunk

For any `high-risk` round that keeps broader significance in scope, record the corresponding promotion level, review-forced demotion, and final landed status in:

- `results/<round_name>/governance_round.yaml`

and treat:

- `make paper-governance-gate ROUND_DIR=results/<round_name>`

as the blocking semantic check before closeout may report that the trunk, branch, or leaf landed.

Preferred rewrite move when prose feels too cautious:

1. shorten the claim to the supported scope
2. strengthen the verb until it matches the evidence
3. move the caveat to the second sentence unless it changes the truth value of the first
4. keep only the limitations that a reviewer truly needs in order to interpret the claim correctly

If the translation depends on figure meaning, first route through `paper-asset-review` or perform the required visual and provenance backtrace inside `paper-submission`.
