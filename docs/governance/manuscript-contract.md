# Manuscript Contract

Use this contract for any task that changes manuscript text, figure legends, paper structure, or the scientific narrative.

For sentence-shape defaults, salience order, and positive `bad -> good` exemplars, use [scientific-voice-guide.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/docs/governance/scientific-voice-guide.md) as the canonical style reference.

This contract governs the main-manuscript subset of the broader `paper-facing explanation` model used across the branch.

## Applies to

- `paper/manuscript/manuscript.md`
- figure legends and claim framing
- Results, Discussion, and Methods boundary decisions
- manuscript consistency and placeholder cleanup

## Core rules

- The branch is manuscript-first.
- Main text structure must remain `Introduction -> Results -> Discussion -> Methods`.
- Results should be assertion-first and evidence-driven.
- Methods carries procedural detail and reproducibility-critical specifications.
- Symbols and advanced concepts must be introduced with bridge sentences and clear physical meaning.
- Paragraphs should end with evidence-anchored take-home claims.
- Nature-facing main text should remain intelligible to scientifically literate readers outside the immediate subfield.
- Local revisions must preserve upstream and downstream logic; inspect neighboring paragraphs and section purpose before treating an edit as complete.
- Terminology, comparator labels, and mechanism language must stay consistent across sections unless the manuscript explicitly marks a shift.
- Each Results subsection should make its local question, evidence, and paper-level consequence explicit.
- The manuscript must identify one stable paper protagonist and keep that protagonist explicit across title, abstract, Results framing, and Discussion.
- The manuscript must identify one paper-level pivot where the reader's model of the system changes; a uniform sequence of equally weighted result summaries is not acceptable architecture.
- The manuscript must give the discovery more narrative weight than the tools that reveal or validate it unless the packet explicitly declares a method paper.
- Any whole-manuscript or cross-section hardening round must write a `Paper spine map` before drafting. That map must include the Results section jobs, pivot sentence, discovery cash-out section, tool role, reference-object role, and a discovery-versus-tool weight budget.
- Reference objects, calibration schemes, and solvers must be framed as supporting actors unless the paper's actual contribution is those objects or methods.
- The manuscript must not be narrated in experiment-production order when a different order is needed to deliver the paper's cognitive shift.
- Any manuscript change that depends on figure meaning, panel mapping, or figure lineage must be grounded in actual visual inspection of the figure asset.
- For `jpg` and `png` figure assets, inspect the image directly. For `pdf` assets, convert every page to PNG previews before interpreting the figure.
- For generated or data-backed figures, reconcile manuscript claims with both the generator or composition code and the upstream data or provenance artifacts, not with filenames or captions alone.
- If a multi-panel figure changes comparator family, evidence role, or panel-block purpose across panels, the Results prose and figure legend must state that panel logic explicitly rather than relying on the reader to infer it.
- Manuscript closeout must distinguish exact changed text from high-level interpretation. When claiming that prose was revised, aligned, tightened, or resolved, provide the exact changed language or exact diff evidence rather than summary alone.
- A manuscript review pass does not by itself establish that every planned manuscript change was implemented. Closeout must distinguish reviewer pass from full plan completion and disclose any deferred or narrowed manuscript surface.

### Nature Communications prose discipline

These rules prevent recurring style errors when AI agents or collaborators draft or revise manuscript text. They are derived from comparative analysis of published Nature Communications articles and codified here to avoid repeated manual correction.

**Mathematics presentation:**

- Core equations in Results must use display math with numbered equations, not inline math only. If a formula defines the paper's central physical model or inference problem, it gets a display equation.
- Every symbol must be defined with an inline "where" clause at its first appearance in Results. Do not assume the reader will read Methods first.

**Sentence and punctuation discipline:**

- Do not use bold emphasis in Introduction, Results, or Discussion body text. Bold is reserved for figure legend titles and Methods sub-section headings.
- Discussion must not contain bold lead-in pseudo-subheadings. Nature Communications does not permit subheadings in the Discussion.
- Limit parenthetical asides to short references or brief qualifiers. If the explanatory text inside parentheses exceeds roughly 10 words, rewrite it as a proper clause or a separate sentence.
- Limit em dashes to at most one per paragraph. Prefer comma-delimited appositives or separate sentences over em-dash insertions.
- Limit numerical values to at most 3 per sentence. Additional values should be directed to figure panels.

**Sentence architecture and causal clarity:**

- Prefer active voice when it preserves scientific accuracy. Make the actor, action, and consequence explicit instead of hiding them inside abstract nouns or passive phrasing.
- Prefer verb-led, cause-effect sentences over dense nominalization. If one mechanism, intervention, or result leads to another, state that relation directly.
- Reduce noun-stack friction. Avoid overloaded front-loaded noun phrases when a clause with a clear subject and verb would read more naturally.
- Keep sentence openings light enough that the reader can identify what is acting before they must unpack multiple modifiers, labels, or stacked technical nouns.
- When drafting or revising a paragraph, check that a cross-disciplinary reader can tell in one pass what changed, what caused it, and why it matters.
- Default to one main causal move per sentence. If a sentence carries multiple independent results, mechanisms, or implications, split it unless scientific accuracy requires them to stay together.
- Treat noun stacks longer than three content words as rewrite triggers unless the phrase is an established term of art, a fixed material name, or a standard mathematical label.
- When a sentence reports a strong numerical comparison, rise, drop, or separation, state the scientific consequence in the same sentence or the next one.
- Prefer spoken-natural scientific English over formal machine-like compression. If a strong graduate student would not naturally say the sentence aloud in lab meeting, rewrite it.
- Prefer dynamic verbs over static holding verbs when the stronger verb preserves the truth value. Replace low-energy forms such as `remains`, `stays`, or `continues` when they only mark persistence rather than scientific action.

**Scientific inference discipline:**

- Manuscript-facing prose must advance by `observation -> inference -> bounded conclusion`, not by document guidance, rebuttal-style positioning, curator narration, or manuscript-management phrasing.
- For each claim-bearing sentence or paragraph, keep three layers distinct: `claim floor`, `claim ceiling`, and `evidence boundary`.
- `Claim floor` is the strongest finding the current evidence does support. State it clearly and early instead of hiding it behind defensive qualifiers.
- `Claim ceiling` is the stronger statement the current evidence does not yet support. Bound it explicitly only when that distinction matters for scientific accuracy.
- `Evidence boundary` is the real scope limit, transfer condition, or open question that constrains the claim. Keep it separate from the lead claim instead of collapsing the paragraph into self-diminishing wording.
- Treat self-diminishing prose as a manuscript failure mode. Do not lower the claim floor with prophylactic negation, straw-man disclaimers, or "safe" wording when the evidence already supports a clearer statement.
- Hold supplementary text, figure legends, inline legends, and main-manuscript Methods prose to the same prose standard as the main text.
- Prefer scientific actors and observations over figure-as-actor wording. Avoid panel-choreography phrasing when the sentence can instead name what was measured, changed, or inferred.
- If a sentence mainly explains how to read the paper, how the manuscript is organized, or how a figure is being used, rewrite it so the sentence states what the evidence implies.
- If a local rewrite removes passive voice but still reads like a guidebook, rebuttal, caption choreography, or repo-provenance note, it is not yet acceptable manuscript prose.
- In Abstract endings, Results subsection openings, and the first paragraph of Discussion, lead with the main supported finding or implication before any caveat, pathway framing, or limitation.
- Phrases such as `without upgrading`, `descriptive rather than`, or `remains positive` without comparator context are presumed rewrite triggers on main-manuscript surfaces unless they serve a real evidentiary distinction.

**Positive construction defaults:**

- Build high-salience sentences in this order when possible: supported finding, evidence anchor, implication, then boundary.
- Build whole-paper architecture in this order when possible: old-world belief, surprising observation, governing principle, broader implication, then boundary.
- Use the canonical `SV#` exemplars in `scientific-voice-guide.md` when a sentence sounds defensively cautious but the evidence remains strong.
- When the evidence is strong, strengthen clarity rather than dampening tone. The correct fix for perceived overstatement is usually a narrower claim, not a weaker verb.
- Distinguish `descriptive` from `insignificant`. A descriptive result may still be central if it organizes the paper's discovery.
- Treat `paper protagonist`, `pivot`, and `tool role` as explicit design choices, not as emergent side effects of paragraph edits.
- Treat `Results section jobs`, `Discovery cash-out section`, and `Discovery-vs-tool weight budget` as explicit architecture choices for any whole-manuscript round, not as formatting afterthoughts.

**Preferred sentence jobs:**

- Abstract final sentence: state the paper-level advance or implication.
- Results opening sentence: state the section's scientific question or supported answer.
- Results closing sentence: state why the result matters for the next section.
- Discussion opening sentence: interpret the advance at paper level before turning to limitations.
- Limitation sentence: name the real evidence boundary in one direct clause without re-litigating the main claim.

**Preferred architecture jobs:**

- Title: break the old intuition and name the protagonist.
- Abstract opening: replace the old-world belief with the new-world belief before setup detail arrives.
- Introduction: establish stakes and the open gap before technical predictions.
- Pivot section: state the turning point that updates the reader's model of the system.
- Tool section: reveal the governing requirement or structure; do not become the story's main character.
- Discovery cash-out section: make the paper-level discovery unavoidable rather than sounding like a final extension.
- Discussion opening: deliver a worldview-shift sentence rather than a Results recap.

**Verb and tone discipline:**

- Vary the verbs in contribution statements. Do not use "We show" more than once in the same paragraph; alternate with demonstrate, establish, further show, reveal.
- Avoid dramatic qualifiers in section titles and body text. Do not use catastrophically, strikingly, crucially, remarkably. Use notably, severely, or importantly as neutral alternatives when emphasis is needed.

**Cross-section consistency:**

- Every reference cited in the Discussion must have a corresponding mention or contextual anchor in the Introduction. Do not introduce new literature threads in the Discussion that lack any Introduction setup.
- Do not include a standalone Road map paragraph. Integrate paper-organization cues into the contribution statement.

**Audience and narrative coherence:**

- Default to cross-disciplinary scientific prose rather than field-internal shorthand.
- Introduce the physical problem or question before naming the metric, model block, or architectural label that addresses it.
- Use paragraph openings to orient the reader and paragraph endings to carry the logic forward.
- Avoid patchwork editing: when a local revision changes emphasis or scope, revise transitions and nearby framing so the manuscript still reads naturally.

## Required outputs

- manuscript text consistent with the branch writing style
- manuscript text coherent at sentence, paragraph, and section level
- correct figure references
- consistent symbols across sections
- consistent terminology and comparator naming across sections
- no unexplained placeholders in final paper-facing output

## Acceptance criteria

- required manuscript sections are present
- figure references are consistent with the manuscript and figure registry
- figure-dependent claim edits are grounded in visual inspection plus generator and provenance backtrace when applicable
- figure-dependent narrative makes comparator and panel-block logic explicit when panel roles change within a figure
- prose is understandable to scientifically literate cross-disciplinary readers without requiring Methods-first reading
- prose uses active voice and direct cause-effect phrasing where scientifically appropriate, without dense nominalization or overloaded noun stacks that increase cognitive load
- prose advances by evidence and inference rather than manuscript-management language, and uses scientific actors instead of figure-as-actor or panel-choreography phrasing when possible
- prose states the supported claim floor clearly before naming the evidence boundary, and does not rely on defensive self-negation to signal rigor
- the in-scope surface passes the editor-first readout test from `scientific-voice-guide.md`, meaning a skimming editor would remember the supported discovery rather than the caveat
- the in-scope surface passes a sentence-energy test: high-salience sentences use clear subjects, explicit causal links, and low enough noun-stack density for one-pass reading
- the in-scope surface passes an architecture test: the protagonist is stable, the pivot is legible, tool sections remain subordinate to discovery, and the Discussion opening upgrades the paper's meaning rather than restating Results
- whole-manuscript and cross-section rounds provide a `Paper spine map` showing Results section jobs, discovery cash-out location, discovery-versus-tool weight, and redundancy or breathing risks
- paragraph and section transitions are natural and preserve the paper-level logic
- terminology, comparator labels, and mechanism language stay consistent across sections unless an explicit shift is introduced
- unresolved placeholders are explicitly tracked or resolved
- manuscript closeout claims about textual changes are backed by exact text or diff evidence, with interpretation labeled separately
- manuscript review pass is not treated as equivalent to full implementation or full plan completion
- paper-facing checks under `make paper-check` pass

## Executable gates

- `python scripts/paper/check_required_sections.py`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/verify_provenance.py`
- `make paper-build`
