# Scientific Voice Guide

Use this guide as the canonical positive-style reference for paper-facing explanation in this branch.

It exists to keep paper voice distinct from review, governance, and closeout voice, and to make the paper's supported discovery legible without overclaiming.

Use these surface definitions across the active branch:

- `paper-facing explanation`: manuscript, supplementary, legends, captions, review-note prose, availability prose, and analysis summaries that may flow into the paper
- `main-manuscript salience`: the highest-salience subset of paper-facing explanation, such as title, abstract, Results openings, and Discussion lead
- `paper-facing asset`: figures, tables, and governed sidecars used to support the paper
- `manuscript-facing`: reserve this narrower label for truly main-manuscript-specific prose rules or literal final manuscript assets

## Narrative architecture layer

Sentence energy is not enough. Paper-facing explanation in this branch must also carry one cognitive shift.

Use these architecture definitions:

- `old-world belief`: the default intuition or field habit the paper must replace
- `new-world belief`: the updated understanding the reader should leave with
- `paper protagonist`: the discovery, organizing principle, or physical phenomenon the paper is really about
- `supporting actor`: reference object, calibration scheme, solver, comparator, or assay that serves the protagonist
- `pivot`: the Results transition where the reader's model of the system should change, not just accumulate another result
- `discovery weight`: the narrative mass given to the paper-level finding
- `tool weight`: the narrative mass given to calibration, decoder, or other support machinery
- `worldview-shift sentence`: the Discussion-opening sentence that says what the paper changes in how the system should be understood

Use these architecture defaults unless the packet explicitly declares a method-first paper:

- Start from the old-world belief, then replace it with the new-world belief.
- Keep one stable paper protagonist from title through Discussion.
- Treat reference objects, calibration routines, and solvers as supporting actors.
- Give the discovery more narrative weight than tool validation.
- Write down a whole-paper spine map for any cross-section or whole-manuscript rewrite. Name the Results section jobs, the pivot sentence, the discovery cash-out section, and the discovery-versus-tool weight budget.
- Do not narrate the paper in experiment time order.
- Reserve the Discussion opening for worldview shift, not recap.

Architecture failure is still a writing failure even when the sentences are locally clear.

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

## Architecture ladder

Default order for a full paper:

1. Old-world belief
2. Surprising observation
3. Governing principle
4. Broader implication
5. Boundary

Use that order to decide what each major section is doing.

### What that means in practice

- Title and abstract opening:
  - break the old intuition before zooming into setup
- Introduction:
  - establish stakes and the open problem before hypothesis-level technical detail
- Results:
  - assign each section one narrative job in the cognitive shift
- Pivot section:
  - mark the point where the reader should understand the system differently, not just know one more result
- Tool section:
  - show what requirement, structure, or physical principle the tool reveals; do not let the tool become the protagonist
- Discussion opening:
  - say what understanding has changed at paper level

## Whole-paper spine map

Use `Architecture scope` to decide when this map is required:

- `local-salience`: use the lighter local architecture bundle for local high-salience work with no section reweighting
- `cross-section`: write down this full map before drafting when the round changes more than one section, section bridges, or discovery-versus-tool weight
- `whole-manuscript`: write down this full map before drafting for full-paper restructuring or any round that re-architects the Results spine

For any `cross-section` or `whole-manuscript` round, write down this map before drafting:

- `Old-world belief`
- `New-world belief`
- `Paper protagonist`
- `Supporting actors`
- `Results section jobs`
- `Pivot section`
- `Pivot sentence`
- `Discovery cash-out section`
- `Tool role`
- `Reference-object role`
- `Discovery-vs-tool weight budget`
- `Redundancy / breathing risks`
- `Worldview-shift sentence`

Use the map to answer four architectural questions before sentence editing begins:

- What changes in the reader's model of the system?
- Where does that change happen?
- Which sections build toward that change and which ones cash it out?
- Which tool sections can be merged or compressed because they answer one bounded scientific question?

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
- If the tool sounds like the protagonist, rewrite the paragraph around the discovery instead.
- If no section feels like a pivot, the paper still reads like an experiment log.
- If the Discussion opening merely restates Results, rewrite it as a worldview shift.

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

### SV15: Abstract reads like compressed Results instead of a changed worldview

Bad:

> We calibrate a 37-angle half-plane, evaluate held-out speech, compare several decoders, and then extend the analysis across five objects.

Good:

> Sound direction is usually measured with arrays or specialized sensors. Here we show that ordinary passive objects already carry a locally ordered directional code in their vibrations, readable from one vibrometric measurement after object-specific calibration.

Why:

- The bad version reports workflow order.
- The good version opens by replacing an old intuition with the paper's new one.

### SV16: Introduction spends stakes on technical predictions too early

Bad:

> Our hypothesis makes two predictions: white noise should sample the band densely, whereas speech should sample it sparsely and non-uniformly across frequency.

Good:

> The open question is not whether direction perturbs structural vibration at all, but whether that perturbation forms a reusable code that can be read from a single passive object. We test that possibility with matched calibration and then ask whether the same code survives more realistic excitation.

Why:

- The bad version spends reader attention on technical consequences before the problem matters.
- The good version establishes stakes first and delays technical detail until the reader has a reason to care.

### SV17: The method becomes the paper's main character

Bad:

> Matched calibration and a guided solver together show that directional information can be recovered from passive objects.

Good:

> Passive objects carry a locally ordered directional code; matched calibration reveals it, and the guided solver succeeds only when it preserves that measured neighborhood.

Why:

- The bad version makes the method the grammatical and conceptual subject.
- The good version keeps the discovery as protagonist and assigns the tool a supporting role.

### SV18: Results advance evenly instead of pivoting

Bad:

> We next examine speech, then evaluate the guided solver, then compare learned and measured structure, and finally test more objects.

Good:

> Speech reveals the paper's turning point: the directional code survives, but the problem becomes local ambiguity. From there the solver sections no longer read as architecture inventory; they test the one requirement that now matters, preserving the measured neighborhood before subtraction.

Why:

- The bad version sounds like an experiment log.
- The good version marks the pivot and reinterprets later sections through that pivot.

### SV19: Tool validation outweighs discovery

Bad:

> Two sections establish that the guided solver converges, outperforms comparators, and learns a stable representation before the paper shows the broader phenomenon.

Good:

> The solver sections are there to show one scientific point: readout works only when the measured neighborhood is preserved long enough. That point serves the broader discovery that the same locally ordered code recurs beyond the reference object.

Why:

- The bad version lets the tool consume the paper's narrative mass.
- The good version assigns the tool one bounded scientific job inside the discovery story.

### SV20: Discussion opens by restating Results instead of shifting worldview

Bad:

> Matched calibration reveals a recurring locally ordered directional code across the tested passive objects.

Good:

> The central implication of this work is that directional encoding need not be designed into a sensor array; it can emerge from the passive structural vibration of ordinary objects and become readable after matched calibration.

Why:

- The bad version repeats a result.
- The good version tells the reader what understanding has changed.

### SV21: Abstract opens with specialized setup instead of breaking the old intuition

Bad:

> Passive objects can encode direction after matched calibration over a 37-angle half-plane grid sampled in 5° steps.

Good:

> Sound direction is usually measured with arrays or dedicated directional sensors. Here we show that ordinary passive objects already carry a readable locally ordered directional code in their vibrations after object-specific calibration.

Why:

- The bad version opens inside setup detail and presumes the reader already cares.
- The good version first breaks the old intuition, then names the discovery.

### SV22: Introduction spends stakes too late and technique too early

Bad:

> Our hypothesis makes two predictions about how white noise and speech should sample the calibration map across frequency.

Good:

> The open question is whether passive structural complexity is merely a nuisance for directional sensing or whether it can itself act as the encoding substrate. We approach that question with matched calibration and then test whether the revealed structure survives more realistic excitation.

Why:

- The bad version spends reader attention on technical predictions before the problem feels important.
- The good version establishes stakes first and keeps the technique in a supporting role.

### SV23: Results sections are individually correct but collectively read like an experiment log

Bad:

> We first characterize the fingerprints, then test speech, then benchmark the solver, then compare learned and measured structure, and finally examine more objects.

Good:

> The Results advance in one spine: calibration reveals a locally ordered code, speech turns that code into a local-ambiguity problem, solver comparisons isolate the one requirement that matters, and the final section shows that the same code recurs beyond the reference object.

Why:

- The bad version narrates the order of analyses.
- The good version narrates what the reader learns next.

### SV24: Tool-validation consumes more narrative mass than the discovery

Bad:

> Two full sections are devoted to convergence, comparator tables, and learned representations before the broader phenomenon is established.

Good:

> Tool sections answer one bounded scientific question: what readout principle preserves the measured neighborhood. Once that point is clear, discovery cash-out must reclaim the paper's narrative center.

Why:

- The bad version lets the tool behave like the main event.
- The good version budgets the tool to one scientific job and restores discovery weight.

### SV25: The paper-level discovery appears only as the final extension

Bad:

> After the solver analyses, we finally test other objects and find a similar pattern.

Good:

> The final section is not an extension but the paper's cash-out: the locally ordered directional code seen in the reference object recurs across materially different passive objects.

Why:

- The bad version makes the broad discovery sound optional.
- The good version marks the final section as the place where the paper's real claim becomes unavoidable.

### SV26: Discussion opening recaps findings instead of updating field understanding

Bad:

> We found that matched calibration reveals a recurring locally ordered code and that guided decoding preserves the measured neighborhood.

Good:

> This work shifts directional sensing from a sensor-design view toward a structural-encoding view: passive objects need not be obstacles that arrays must overcome; they can themselves supply the directional code that matched calibration reads out.

Why:

- The bad version restates what happened in the Results.
- The good version tells the reader what understanding of the field has changed.

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
