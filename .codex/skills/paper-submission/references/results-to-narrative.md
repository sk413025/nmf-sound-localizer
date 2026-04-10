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

Scientific voice rule:

- write with earned confidence
- use `docs/governance/scientific-voice-guide.md` as the canonical sentence-shape and exemplar reference
- say the supported discovery sentence plainly before naming the limitation
- if a paragraph sounds safer only because it is weaker, it is probably worse

Editor-first test:

- after drafting, ask what a handling editor would underline in 15 seconds
- if the answer is a caveat, setup detail, or workflow description, rewrite the paragraph with the closest `SV#` exemplar as the model
- if the answer is the discovery sentence, keep going

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

- identify what logical job the paragraph or section performs
- bridge from the previous known point before introducing the next claim
- keep terminology, comparator names, and mechanism labels stable across sections
- make each paragraph advance one main logical step
- end paragraphs by showing why the point matters for the next step of the paper
- if a sentence only works with hidden local context, rewrite it so the link is explicit

Preferred answer pattern:

1. one-sentence conclusion
2. what the result directly supports
3. what mechanism it points toward
4. what it still does not prove

Preferred rewrite move when prose feels too cautious:

1. shorten the claim to the supported scope
2. strengthen the verb until it matches the evidence
3. move the caveat to the second sentence unless it changes the truth value of the first
4. keep only the limitations that a reviewer truly needs in order to interpret the claim correctly

If the translation depends on figure meaning, first route through `paper-asset-review` or perform the required visual and provenance backtrace inside `paper-submission`.
