# Scientific Voice Guide

Use this guide as the canonical positive-style reference for manuscript-facing prose in this branch.

It exists to keep manuscript voice distinct from review, governance, and closeout voice, and to make the paper's supported discovery legible without overclaiming.

## Voice separation matrix

Use the right voice for the right surface.

| Surface | Default job | Default tone | Failure mode |
| --- | --- | --- | --- |
| Manuscript voice | state the discovery, show the evidence, explain the implication, bound the scope | claim-forward, editor-legible, evidence-bounded | self-diminishing prose, caveat-led salience, review-language leakage |
| Review voice | find objections, scope gaps, and rejection risks | adversarial, objection-seeking, pressure-testing | being mistaken for manuscript prose |
| Closeout voice | account for scope, evidence, verification, and delivery state | ledger-like, explicit, completion-disciplined | leaking into the paper as if caution were narrative |

Mode leakage is a governance failure. Manuscript prose must not sound like a rebuttal, an audit memo, or a milestone closeout.

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

## Preferred rewrite moves

When prose feels too cautious, do this in order:

1. Shorten the claim to the supported scope.
2. Strengthen the verb until it matches the evidence.
3. Move the caveat to sentence two unless it changes sentence one truth value.
4. Remove defensive strawmen that only exist to pre-empt reviewer objections.
5. Keep only the limitations a reviewer needs in order to interpret the claim correctly.

## Editor-first check

Ask:

> What would a handling editor underline in 15 seconds?

If the answer is:

- a caveat
- a setup detail
- a workflow description
- a constraint sentence

rewrite the surface.

If the answer is the supported discovery sentence, the surface is likely carrying the right manuscript voice.
