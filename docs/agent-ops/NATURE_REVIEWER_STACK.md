# Nature Reviewer Stack

This document is the canonical source of truth for the Nature Communications reviewer stack used in manuscript hardening, red-team review, and submission-facing critique in this branch.

Use it to keep reviewer roles and evaluation goals stable across review rounds.
Do not create ad hoc reviewer personas when this stack already covers the risk.

## Purpose

The reviewer stack is a manuscript-facing review model, not a parallel workflow system.

- The supervisor still owns routing, decomposition, and review planning.
- Existing execution roles in `docs/agent-ops/ROLE_CATALOG.md` still own the work.
- This stack defines the named review lenses and evaluation goals that the supervisor selects when manuscript-facing work needs high-standard critique.

## When to use this stack

Use the canonical reviewer stack when a task could affect:

- Nature Communications editorial fit or paper-level scope
- how likely the paper is to be routed to the right reviewers
- cross-disciplinary readability
- physical-mechanism interpretation
- acoustics or DOA plausibility
- comparator logic or sparse inverse-problem fairness
- statistical support or evidence sufficiency
- figure science, figure readability, or panel logic
- narrative flow across sections
- cognitive load for the reader

For full manuscript hardening, the supervisor should assume this stack is the default review surface and then select the minimal applicable subset.

## How the supervisor should apply it

1. Decide whether the work needs one reviewer pass or multiple parallel reviewer passes.
2. Select the applicable reviewer roles from this document instead of inventing new ones.
3. Record the selected reviewer roles and their evaluation goals in the task packet or review request.
4. Define the in-scope and out-of-scope acceptance surfaces for the review. Do not fail scientific-narrative review on submission metadata placeholders unless packaging is explicitly in scope.
5. Route each reviewer pass through the existing role system and core skills.
6. Consolidate reviewer findings at the supervisor layer.

Reviewer qualification gate:

- When manuscript-facing prose is in scope, treat a reviewer output as unqualified if it does not test for scientific inference versus manuscript-management language, or if it ignores the stated acceptance surface.

These reviewer roles are review lenses, not new top-level workflow roles.

## Canonical reviewer roles

### handling-editor-scope reviewer

Evaluation goal:

- Judge whether the manuscript reads like a Nature Communications paper rather than a narrow lab note or method-only report.
- Check whether the claimed advance is legible at the paper level before technical details are unpacked.
- Check whether the manuscript states its claim floor clearly enough for an editor to register the advance on a first pass, rather than burying it under qualifiers, pathways, or defensive framing.
- Test whether the title, abstract, Results framing, and Discussion together justify editorial interest and scope.
- Flag cases where the manuscript feels technically busy but editorially under-motivated.
- Flag rebuttal-style, guidebook-style, or manuscript-positioning prose that explains the paper instead of advancing the scientific inference.

### reviewer-routing reviewer

Evaluation goal:

- Predict which reviewer communities the manuscript will likely be routed to and whether the framing will survive that routing.
- Check whether the manuscript signals the right disciplinary anchors early enough for reviewers to place the contribution correctly.
- Identify places where wording could trigger the wrong reviewer expectations or create avoidable scope mismatch.
- Flag claims that require a specific expert audience but are currently framed too vaguely or too broadly.

### cross-disciplinary-readability reviewer

Evaluation goal:

- Check whether a scientifically literate reader outside the immediate subfield can follow the claim, setup, and implication without Methods-first reading.
- Remove field-internal shorthand, unexplained jumps, and local jargon that would block a broad Nature audience.
- Verify that important terms, comparators, and mechanism labels are introduced before they are relied on.
- Check that key sentences use active, verb-led phrasing and direct cause-effect relations instead of nominalization-heavy or front-loaded noun-stack constructions.
- Test whether the prose advances by observation, inference, and bounded conclusion rather than by guidebook, rebuttal, or manuscript-management language.
- Flag paragraphs that are technically correct but too compressed to be legible across disciplines.

### physical-mechanism reviewer

Evaluation goal:

- Check whether the manuscript distinguishes observed behavior from mechanism-level interpretation with the right level of caution.
- Test whether mechanism language is supported by evidence rather than by suggestive correlation alone.
- Identify where the text overstates why the method works or blurs performance description with mechanistic explanation.
- Flag places where stronger mechanism language would require additional evidence, controls, or caveats.

### acoustics-doa reviewer

Evaluation goal:

- Check whether acoustics and DOA claims are physically plausible, correctly framed, and consistent with the problem setup.
- Verify that geometry, source conditions, and signal assumptions are not described in ways that domain reviewers would reject immediately.
- Identify manuscript statements that misuse acoustics terminology or overgeneralize beyond the evaluated setting.
- Flag places where the paper needs clearer contact with real acoustic intuition or task constraints.

### sparse-inverse-problem-comparator reviewer

Evaluation goal:

- Check whether comparator families are chosen, labeled, and discussed fairly from a sparse inverse-problem perspective.
- Test whether baseline positioning, ablation logic, and solver comparisons avoid straw-man framing.
- Verify that comparator changes across figures or panels are explicit rather than silently shifting.
- Flag cases where fairness depends on tuning, prior information, or setup choices that are not being surfaced.

### statistics-evidence reviewer

Evaluation goal:

- Check whether every manuscript-facing claim has evidence strength that matches the wording.
- Verify that comparisons, trends, and interpretations are grounded in reported statistics, artifact-backed results, or explicit figure anchors.
- Identify missing uncertainty language, missing controls, or over-general conclusions drawn from partial evidence.
- Flag places where the manuscript needs weaker wording, more provenance, or a more explicit evidence boundary.

### figure-science-readability reviewer

Evaluation goal:

- Check whether each figure supports a real scientific job in the paper rather than functioning as decoration or redundancy.
- Verify that panel order, panel logic, labels, legends, and visual emphasis make the scientific comparison readable.
- Identify figures that are individually attractive but paper-level confusing, overloaded, or weakly matched to the claim.
- Flag cases where a figure should be revised, split, simplified, or moved to supplementary.

### narrative-flow reviewer

Evaluation goal:

- Check whether the manuscript moves through problem, method, evidence, and implication in a sequence that feels inevitable rather than patched together.
- Verify that transitions between sections and figures preserve story logic and do not force the reader to reconstruct missing links.
- Check that each major section opens with either the next scientific question or the supported answer to that question, not with procedural recap or defensive boundary-setting.
- Identify places where the paper's logic is buried inside abstract noun chains rather than explicit actions, results, and consequences.
- Identify abrupt scope changes, repeated explanations, and local rewrites that create whole-manuscript drift.
- Flag defensive `X rather than Y` framing, panel-choreography prose, and manuscript-positioning sentences that explain paper structure more than scientific consequence.
- Flag places where section order or paragraph function needs revision to keep the paper natural and persuasive.

### cognitive-load reviewer

Evaluation goal:

- Check whether the paper asks the reader to track too many new concepts, panel purposes, or comparator families at once.
- Identify where density, notation, or panel complexity overwhelms the amount of progress delivered to the reader.
- Flag nominalization-heavy sentences, overloaded front-loaded noun phrases, and other sentence forms that make the reader decode syntax before science.
- Flag figure-as-actor, guidebook, curator, and manuscript-management phrasing that makes the reader track paper machinery instead of scientific actors and observations.
- Flag hedge density that forces the reader to decode caveats before understanding the supported claim floor.
- Verify that each section and figure earns its complexity and that important takeaways are easy to retain.
- Flag compression patterns that may be efficient for insiders but exhausting for reviewers or editors.

## Routing guidance

Use these reviewer roles through the existing role and skill system:

- `handling-editor-scope reviewer` and `reviewer-routing reviewer`: usually routed through `supervisor` or `red-team-reviewer` with `agent-orchestrator`
- `cross-disciplinary-readability reviewer`, `narrative-flow reviewer`, and `cognitive-load reviewer`: usually routed through `manuscript-reviser` or `claim-auditor` with `paper-submission`
- `physical-mechanism reviewer`, `acoustics-doa reviewer`, `sparse-inverse-problem-comparator reviewer`, and `statistics-evidence reviewer`: usually routed through `claim-auditor` or `experiment-results-analyst`, depending on whether the task is manuscript-facing critique or artifact-facing analysis
- `figure-science-readability reviewer`: usually routed through `paper-asset-reviewer`, with `paper-submission` follow-up when the figure critique requires manuscript rewrites

## Output expectations

Reviewer outputs should name:

- the reviewer role used
- the evaluation goal being applied
- the main findings and failure modes
- the best one-sentence editor readout if the current wording were fixed
- the closest `SV#` exemplar from `docs/governance/scientific-voice-guide.md` when the issue is primarily a manuscript-voice or salience failure
- any required rewrite, evidence, or routing follow-up

## Mandatory reviewer subsets for high-salience prose rounds

For any round that changes the title, abstract, Results subsection openings, section-to-section bridges, figure-to-figure transitions, or the first paragraph of Discussion, the parent must include:

- `handling-editor-scope reviewer`
- `narrative-flow reviewer`
- `cognitive-load reviewer`

Add `statistics-evidence reviewer` when wording changes could alter evidence strength or scope, and add `reviewer-routing reviewer` when the new framing may change likely reviewer community routing.

If multiple reviewer roles are used, the supervisor should consolidate them into one manuscript-facing decision rather than leaving them as disconnected comments.
