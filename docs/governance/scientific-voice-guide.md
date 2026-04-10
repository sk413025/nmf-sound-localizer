# Scientific Voice Guide

Use this guide as the canonical positive-style reference for paper-facing explanation in this branch.

It exists to keep paper voice distinct from review, governance, and closeout voice, and to make the paper's supported discovery legible without overclaiming.

Use these surface definitions across the active branch:

- `paper-facing explanation`: manuscript, supplementary, legends, captions, review-note prose, availability prose, and analysis summaries that may flow into the paper
- `main-manuscript salience`: the highest-salience subset of paper-facing explanation, such as title, abstract, Results openings, and Discussion lead
- `paper-facing asset`: figures, tables, and governed sidecars used to support the paper
- `manuscript-facing`: reserve this narrower label for truly main-manuscript-specific prose rules or literal final manuscript assets

## Voice separation matrix

Use the right voice for the right surface.

| Surface | Default job | Default tone | Failure mode |
| --- | --- | --- | --- |
| Paper-facing explanation | state the discovery, show the evidence, explain the implication, and bound the scope across manuscript, supplementary, legends, captions, review-note prose, and analysis summaries that may flow into the paper | claim-forward, editor-legible, evidence-bounded | self-diminishing prose, caveat-led salience, review-language leakage |
| Main-manuscript salience | carry the highest-salience paper claims in title, abstract, Results openings, Discussion lead, and other core manuscript prose | discovery-first, editor-retentive, consequence-carrying | setup-led framing, procedural transitions, weak section energy |
| Review voice | find objections, scope gaps, and rejection risks | adversarial, objection-seeking, pressure-testing | being mistaken for paper-facing explanation |
| Closeout voice | account for scope, evidence, verification, and delivery state | ledger-like, explicit, completion-disciplined | leaking into the paper as if caution were narrative |

Mode leakage is a governance failure. Paper-facing explanation must not sound like a rebuttal, an audit memo, or a milestone closeout.

## Salience ladder

Default order:

1. Discovery
2. Evidence
3. Implication
4. Boundary

Use that order in the paper's highest-salience surfaces unless the boundary changes the truth value of the discovery sentence.

### What that means in practice

- Abstract ending:
  - land on the paper-level advance or implication
- Results opening:
  - pose the next scientific question or state the supported answer
- Results closing:
  - state why the finding matters for what comes next
- Discussion opening:
  - interpret the paper-level meaning before discussing limitations
- Limitation sentence:
  - name the real boundary directly, without reopening the whole claim

## Sentence-energy defaults

Default paper-facing sentence shape:

1. clear subject
2. strong verb
3. explicit consequence

Use these defaults unless scientific precision genuinely requires a denser construction.

- One main causal move per sentence.
- Keep noun stacks short enough for one-pass reading.
- Translate important numbers into meaning instead of leaving them as raw fact clusters.
- Prefer verbs and clauses over compressed label chains.
- Prefer natural scientific English that a strong PhD student could say aloud in lab meeting.
- Prefer dynamic verbs such as `concentrates`, `tracks`, `limits`, `holds`, and `exceeds` over static filler verbs when the stronger verb preserves the truth value.

## Last-read drafting card

Read this before writing any paper-facing explanation:

- State the discovery before the caveat.
- Use one main causal move per sentence.
- If the sentence has too many nouns in a row, unpack it.
- If the sentence reports numbers, also report what they mean.
- If the sentence sounds like a memo, rewrite it like a scientific explanation.
- If the sentence sounds smarter only because it is denser, simplify it.
- If a strong PhD student would not say it aloud, rewrite it.
- Keep the reader decoding science, not syntax.

## Golden pairs

### SV1: Abstract ending lands on caveat instead of advance

Bad:

> Together, these results identify matched calibration as a pathway to single-point sensing, with nearby-angle overlap defining the central readout constraint.

Good:

> Together, these results reveal that matched calibration makes single-point directional readout possible because passive objects retain a finite locally ordered response neighborhood.

Why:

- The bad version ends on pathway and constraint.
- The good version ends on the advance and keeps the mechanism-facing implication visible.

### SV2: Discovery is buried inside setup detail

Bad:

> A five-object comparison then shows that the local-ordering constraint recurs across ordinary passive structures under matched calibration, even though the informative band remains object-specific.

Good:

> Across five passive objects, the same finite locally ordered directional structure reappears under matched calibration, while its width remains object-specific.

Why:

- The bad version makes the experiment design the grammatical subject.
- The good version makes the discovery the subject and moves the boundary to the tail.

### SV3: Discussion opens with self-negation

Bad:

> The five-object comparison extends that observation without upgrading it into a universal law.

Good:

> The five-object comparison shows that finite local angular ordering recurs across the tested objects despite large differences in material and geometry.

Why:

- The bad version opens by denying a stronger claim nobody required.
- The good version states the supported finding, then leaves room to bound it later.

### SV4: Transition is procedural instead of forward-driving

Bad:

> The guided solver preserves the measured local organization at both the map level and the summary-statistic level.

Good:

> Having established that the guided solver converges reliably, we next test whether its predictions preserve the physical neighborhood structure seen in measurement.

Why:

- The bad version drops the reader into an answer without framing the question.
- The good version turns the prior section into the logical reason for the next one.

### SV5: Strong quantitative result is described weakly

Bad:

> The full-matrix agreement remains positive (r = 0.47).

Good:

> The full-matrix agreement is substantial (r = 0.47), indicating that the learned structure tracks the measured neighborhood well above a null expectation near zero.

Why:

- The bad version makes a meaningful result sound barely acceptable.
- The good version gives the number interpretive context without inflating it.

### SV6: The prose devalues a central descriptive result

Bad:

> The cross-object comparison remains descriptive rather than a fitted law.

Good:

> The cross-object comparison shows the same qualitative decay structure across the tested objects, with object-specific width rather than a single shared scale.

Why:

- The bad version apologizes for not doing a different analysis.
- The good version states what the result does establish and bounds it precisely.

### SV7: Limitations dilute the discussion instead of bounding it

Bad:

> These findings are limited by the half-plane grid, the single static source, the laboratory environment, the restricted object set, and the absence of a broader transfer study, so broader implications should be treated cautiously.

Good:

> These conclusions are bounded to the tested geometry, source configuration, and object set. Extending them to broader environments will require new measurements rather than restating the current result more cautiously.

Why:

- The bad version turns the whole sentence into a warning cloud.
- The good version states the true transfer boundary directly and stops.

### SV8: Section title reports activity instead of interpretation

Bad:

> Prediction structure stays organized around the measured local band

Good:

> The guided solver recapitulates the measured physical neighborhood in its learned representation

Why:

- The bad version sounds observational and low-stakes.
- The good version tells the reader what the observation means.

### SV9: Noun stack hides the sentence's action

Bad:

> A physics-guided solver that preserves local evidence before subtraction stabilizes clean-condition readout and recapitulates the measured local neighborhood in its learned representation.

Good:

> A physics-guided solver preserves local evidence before subtraction, which stabilizes readout under clean conditions. Its learned representation then recapitulates the local neighborhood seen in measurement.

Why:

- The bad version asks the reader to hold too many technical noun phrases at once.
- The good version gives each clause one clear job and lets the action carry the sentence.

### SV10: Facts are listed, but the causal link is missing

Bad:

> The residual drops from 1.00 to 0.48 after the first guided step and the fraction of update mass within 15° rises from 0.18 to 0.98 after the local step.

Good:

> The residual drops from 1.00 to 0.48 after the first guided step, confirming that the solver extracts real directional signal. The local gate then shifts the fraction of update mass within 15° from 0.18 to 0.98, meaning that nearly all correction now lands in the physically plausible neighborhood.

Why:

- The bad version reports numbers but leaves the reader to infer why they matter.
- The good version translates each number into a scientific consequence.

### SV11: A quantitative contrast appears without a "so what"

Bad:

> The paper cup carries the strongest normalized overall energy yet still sits at broader overlap burden and lower readout than cardboard.

Good:

> The paper cup carries the strongest overall signal energy, yet its neighboring angles overlap more broadly, which limits readout relative to cardboard. The result shows that local separability matters more than raw energy for directional decoding.

Why:

- The bad version gives a counterintuitive fact but withholds the insight.
- The good version spells out what the contrast teaches the reader.

### SV12: The diction is formally correct but not natural

Bad:

> White noise and speech interrogate that space differently, and broader overlap becomes operationally costly at the first discrete commitment.

Good:

> White noise and speech probe that space differently, and the broader overlap first hurts when the decoder must make a hard angle choice.

Why:

- The bad version sounds like technical compression rather than natural scientific explanation.
- The good version uses words a strong researcher could say aloud without sacrificing precision.

### SV13: Static verbs flatten a real scientific result

Bad:

> Within-angle similarity stays higher than between-angle similarity, and guided decoding stays concentrated near the diagonal.

Good:

> Within-angle similarity exceeds between-angle similarity, and guided decoding concentrates near the diagonal.

Why:

- The bad version describes persistence rather than scientific structure.
- The good version uses verbs that reveal what the result is doing.

### SV14: Methods or legends compress the pipeline into a noun block

Bad:

> Fig. 4d together with the sweep-derived panels in Fig. 5a,e use governed five-seed clean or SNR sweep artifacts built from best-validation checkpoints.

Good:

> For Figs. 4d, 5a, and 5e, we ran five independent seeds under clean or SNR-sweep conditions and summarized the checkpoints selected on the validation set.

Why:

- The bad version turns pipeline bookkeeping into a hard-to-parse noun cluster.
- The good version keeps the same technical content in direct procedural prose.

## Preferred rewrite moves

When prose feels too cautious, do this in order:

1. Shorten the claim to the supported scope.
2. Strengthen the verb until it matches the evidence.
3. Move the caveat to sentence two unless it changes sentence one truth value.
4. Remove defensive strawmen that only exist to pre-empt reviewer objections.
5. Keep only the limitations a reviewer needs in order to interpret the claim correctly.
6. If the sentence still feels dense, identify the subject, verb, and consequence and rewrite in that order.
7. If the sentence still sounds unnatural, rewrite it in lab-meeting English and then restore only the technical precision that matters.

## Editor-first check

Ask:

> What would a handling editor underline in 15 seconds?

If the answer is:

- a caveat
- a setup detail
- a workflow description
- a constraint sentence

rewrite the surface.

If the answer is the supported discovery sentence, the surface is likely carrying the right paper voice.
