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
- The manuscript must identify one primary paper-level pivot where the reader's model of the system changes; a uniform sequence of equally weighted result summaries is not acceptable architecture.
- The manuscript must give the discovery more narrative weight than the tools that reveal or validate it unless the packet explicitly declares a method paper.
- When the evidence supports broader significance, the manuscript must express it first as an endogenous `second-layer discovery` rather than as a free-standing application list.
- The allowed hierarchy for broader significance is `core discovery -> second-layer discovery -> bounded downstream consequence -> boundary`.
- If a paper-facing broader implication cannot be stated as a discovery-level inference earned by the current evidence, downstream applications must be weakened or removed rather than promoted into the paper spine.
- Promotion into broader significance must follow this ladder:
  - `core-only`: the evidence supports only the immediate-field discovery; broader implication stays out of the paper spine
  - `second-layer earned`: the evidence supports one discovery-level inference beyond the immediate field; only the trunk may enter the spine
  - `branch earned`: the evidence supports one bounded downstream consequence in addition to an already earned trunk
  - `leaf allowed`: a weaker, more distant implication may appear briefly only after trunk and branch already land cleanly
- If the broader-significance status is below `second-layer earned`, the manuscript must not preload broader consequence in the Abstract tail, Introduction ending, or Discussion trunk paragraph.
- If the broader-significance status is below `branch earned`, downstream application sentences must be removed or rewritten as frontier language outside the paper spine.
- If an optional leaf cannot survive demotion pressure without stealing memorability from the trunk, it must be dropped rather than preserved as decoration.
- For any `high-risk` manuscript round with broader significance or cross-disciplinary consequence in scope, the round must create `results/<round_name>/governance_round.yaml` and pass `make paper-governance-gate ROUND_DIR=results/<round_name>` before closeout may report that the broader significance landed.
- For those rounds, `docs/agent-ops/ROUND_GOVERNANCE_SCHEMA.md` is the only canonical machine-readable field inventory. Do not recreate a second schema inside manuscript-facing contracts, packets, or closeout prose.
- Any whole-manuscript or cross-section hardening round must write a `Paper spine map` before drafting. That map must include the Results section jobs, pivot sentence, discovery cash-out section, tool role, reference-object role, and a discovery-versus-tool weight budget.
- Reference objects, calibration schemes, and solvers must be framed as supporting actors unless the paper's actual contribution is those objects or methods.
- The manuscript must not be narrated in experiment-production order when a different order is needed to deliver the paper's cognitive shift.
- Any manuscript change that depends on figure meaning, panel mapping, or figure lineage must be grounded in actual visual inspection of the figure asset.
- For `jpg` and `png` figure assets, inspect the image directly. For `pdf` assets, convert every page to PNG previews before interpreting the figure.
- For generated or data-backed figures, reconcile manuscript claims with both the generator or composition code and the upstream data or provenance artifacts, not with filenames or captions alone.
- Active main-paper figure planning must remain traceable through the canonical panel-to-method crosswalk at `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`.
- That crosswalk must record, for every active main-paper panel, the scientific job, panel type, encoded quantity, main-text method anchor, supplementary anchor, overlap verdict, and required action.
- Any `cross-section` or `whole-manuscript` round that changes panel role, panel order, Results handoff, or method linkage for an active main-paper figure must update the canonical crosswalk in the same round.
- Figure planning and manuscript acceptance must not rely on readability, panel density, or visual polish alone when panel scientific job, method anchor, supplementary anchor, or overlap verdict would change the paper-level interpretation.
- If a multi-panel figure changes comparator family, evidence role, or panel-block purpose across panels, the Results prose and figure legend must state that panel logic explicitly rather than relying on the reader to infer it.
- Manuscript closeout must distinguish exact changed text from high-level interpretation. When claiming that prose was revised, aligned, tightened, or resolved, provide the exact changed language or exact diff evidence rather than summary alone.
- A manuscript review pass does not by itself establish that every planned manuscript change was implemented. Closeout must distinguish reviewer pass from full plan completion and disclose any deferred or narrowed manuscript surface.

### Nature Communications prose discipline

These rules prevent recurring style errors when AI agents or collaborators draft or revise manuscript text. They are derived from comparative analysis of published Nature Communications articles and codified here to avoid repeated manual correction.

**Mathematics presentation:**

- Results must make method linkage visible, but they should introduce quantities by scientific role and physical meaning before notation.
- Use equation numbers in Results only as light anchors when they improve traceability. Do not force full symbolic unpacking into high-salience Results sentences when the same linkage can be made more naturally in quantity-first language.
- Full display equations belong in Methods unless the formula itself is a paper-level object in the Results argument, such as the paper's central physical model or inference problem.
- If notation appears in Results, keep it to the minimum needed for reader traceability and define it through an immediate bridge to physical meaning. Do not assume the reader will read Methods first.

**Editor-first reading contract:**

- High-salience `Introduction` and `Results` prose must let a skimming Nature Communications editor or cross-disciplinary reviewer tell, without `Methods`-first reading, what physical phenomenon is being claimed, what observable or quantity carries it in the current paragraph, and why it matters.
- Front-door prose should feel like progressive sharpening of one idea, not repeated renaming. Use one stable paper-facing term per concept and let `Methods` or `Supplementary Methods` carry the formal labels, internal surface names, and exact score constructions.
- Introduce difficult concepts in this order when possible: `physical intuition -> observable consequence -> formal anchor`. Do not make the reader learn an internal label, metric name, or symbolic object before the scientific role of that object is clear.
- If a paper-facing umbrella term spans more than one formal surface, the prose must mark that surface shift in plain language rather than silently reusing the same noun across distinct quantities.
- Treat private-vocabulary pressure as a manuscript failure mode. If a paragraph would require a reader to mentally translate among several near-synonyms or internal labels before understanding the science, simplify the terminology instead of adding more bridge jargon.
- Every high-salience manuscript round should pass a bidirectional no-translation check: `Introduction/Results -> Methods` should read like progressive formalization of the same actors and quantities, and `Methods/Supplementary -> Introduction/Results` should let a reader recover the matching paper-facing concept without building a separate term dictionary.
- Treat one-way traceability as insufficient. It is not enough that `Results` can be defended from `Methods`; the formal surfaces must also point back naturally to the paper-facing language they are instantiating.
- Give each active concept family one canonical head term across main text, legends, `Methods`, and `Supplementary Methods`. Let symbols, equations, and exact score names sharpen that head term rather than replace it with a second reader-facing name.
- In `Methods` and `Supplementary Methods`, reopen a concept family by reusing its canonical head term before notation, statistic names, or surface-specific labels appear. The reader should first recognize the same scientific actor and only then see its formal specification.
- Keep legend phrasing on the same canonical head terms used in the body text. Legends may add local precision, but they should reinforce the manuscript's shared term map rather than introduce a parallel technical name.

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
- `Claim floor` is the strongest finding the current evidence does support. State it clearly, early, and in affirmative scientific language before introducing scope limits.
- `Claim ceiling` is the stronger statement the current evidence does not yet support. Bound it explicitly only when that distinction matters for scientific accuracy.
- `Evidence boundary` is the real scope limit, transfer condition, or open question that constrains the claim. Use it to contain the supported claim, not to pre-weaken it or pre-answer a reviewer objection before the discovery lands.
- Treat self-diminishing prose as a manuscript failure mode. Do not lower the claim floor with prophylactic negation, straw-man disclaimers, or "safe" wording when the evidence already supports a clearer statement.
- Do not write anticipated reviewer misreadings into paper-facing prose. If a sentence feels too broad, narrow the scientific claim itself rather than adding `not X`, `should not be read as`, or other prebuttal phrasing.
- Hold supplementary text, figure legends, inline legends, and main-manuscript Methods prose to the same prose standard as the main text.
- Prefer scientific actors and observations over figure-as-actor wording. Avoid panel-choreography phrasing when the sentence can instead name what was measured, changed, or inferred.
- If a sentence mainly explains how to read the paper, how the manuscript is organized, or how a figure is being used, rewrite it so the sentence states what the evidence implies.
- If a local rewrite removes passive voice but still reads like a guidebook, rebuttal, caption choreography, or repo-provenance note, it is not yet acceptable manuscript prose.
- In Abstract endings, Results subsection openings, and the first paragraph of Discussion, lead with the main supported finding or implication before any caveat, pathway framing, or limitation.
- Phrases such as `without upgrading`, `descriptive rather than`, or `remains positive` without comparator context are presumed rewrite triggers on main-manuscript surfaces unless they serve a real evidentiary distinction.

**Positive construction defaults:**

- Build high-salience sentences in this order when possible: supported finding, evidence anchor, implication, then boundary.
- Build whole-paper architecture in this order when possible: old-world belief, surprising observation, governing principle, second-layer discovery, bounded downstream consequence, then boundary.
- Build broader-significance promotion in this order when possible: prove the core discovery, earn the second-layer discovery, earn one bounded branch, then decide whether any optional leaf is still worth the reader burden.
- Use the canonical `SV#` exemplars in `scientific-voice-guide.md` when a sentence sounds defensively cautious but the evidence remains strong.
- When the evidence is strong, strengthen clarity rather than dampening tone. The correct fix for perceived overstatement is usually a narrower claim, not a weaker verb.
- When scope needs to be narrowed, narrow the actor, condition, or consequence. Do not turn the sentence into a defensive explanation of how it might be misread.
- Distinguish `descriptive` from `insignificant`. A descriptive result may still be central if it organizes the paper's discovery.
- Treat `paper protagonist`, `pivot`, and `tool role` as explicit design choices, not as emergent side effects of paragraph edits.
- Treat `second-layer discovery`, `broader-implication trunk`, and `downstream-consequence branch` as explicit design choices whenever the paper reaches beyond its immediate field.
- Treat broader-significance status as an explicit design choice whenever the paper reaches beyond its immediate field. Record whether the round is `core-only`, `second-layer earned`, `branch earned`, or `leaf allowed` before drafting high-salience prose.
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
- Broader-significance trunk: promote it into the paper spine only when it survives the earned-discovery, boundary-pressure, and reviewer-routing survival tests.
- Downstream branch: promote it only when it remains bounded, subordinate to the trunk, and worth its cognitive cost after demotion pressure.

**Verb and tone discipline:**

- Vary the verbs in contribution statements. Do not use "We show" more than once in the same paragraph; alternate with demonstrate, establish, further show, reveal.
- Avoid dramatic qualifiers in section titles and body text. Do not use catastrophically, strikingly, crucially, remarkably. Use notably, severely, or importantly as neutral alternatives when emphasis is needed.

**Cross-section consistency:**

- Every reference cited in the Discussion must have a corresponding mention or contextual anchor in the Introduction. Do not introduce new literature threads in the Discussion that lack any Introduction setup.
- Do not include a standalone Road map paragraph. Integrate paper-organization cues into the contribution statement.

## Broader-significance promotion gate

Use this gate whenever a manuscript round wants to land broader significance.

- `Earned-discovery test`:
  - The proposed second-layer discovery must still be stated with the paper's own discovery actor as subject.
  - If the sentence only works as an application slogan, future-use promise, or literature-tour justification, it is not earned and must demote.
- `Boundary-pressure test`:
  - Rewrite the broader implication in the narrowest affirmative form that remains fully supported.
  - If that narrower positive version is still worth placing in the Abstract tail, Introduction ending, or Discussion trunk, it may remain promoted.
  - If the conservative version collapses into vague usefulness, demote it to `core-only` or drop it.
- `Reviewer-routing survival test`:
  - Ask whether the proposed trunk would still look evidence-earned if the paper were read first by an immediate-field reviewer rather than a sympathetic adjacent-field reader.
  - Keep the prose discovery-led while making that judgment; do not convert the paper-facing sentence into reviewer-prebuttal language.
  - If the claim would likely be read as scope drift, it cannot be promoted beyond `core-only`.
- `Demotion rule`:
  - If the trunk fails, demote to `core-only` and rewrite the prose accordingly.
  - If a downstream branch fails, demote to `second-layer earned`.
  - Optional leaves that fail the same test must demote to `branch earned` or be dropped rather than defended.

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
- active main-paper figure-dependent claim edits are also reconciled against `paper/manuscript/FIGURE_METHOD_CROSSWALK.md`
- figure-dependent narrative makes comparator and panel-block logic explicit when panel roles change within a figure
- active main-paper panels do not remain unclassified or unanchored at the level of scientific job, panel type, and method/supplement linkage
- prose is understandable to scientifically literate cross-disciplinary readers without requiring Methods-first reading
- a skimming editor can identify, in each high-salience paragraph, the physical phenomenon, the active observable or quantity, and the scientific consequence without translating an internal term map
- the in-scope surface passes a bidirectional no-translation test: a reader can move from `Introduction/Results` into `Methods` and from `Methods/Supplementary` back into `Introduction/Results` while recognizing the same concepts without inventing a private synonym map
- each active concept family keeps one canonical head term across main text, legends, `Methods`, and `Supplementary Methods`, with formal labels appearing as secondary precision rather than competing reader-facing names
- each `Methods` or `Supplementary Methods` first mention reopens the same canonical head term before introducing notation, equation-level labels, or exact statistic names
- prose uses active voice and direct cause-effect phrasing where scientifically appropriate, without dense nominalization or overloaded noun stacks that increase cognitive load
- prose advances by evidence and inference rather than manuscript-management language, and uses scientific actors instead of figure-as-actor or panel-choreography phrasing when possible
- prose states the supported claim floor clearly in affirmative form before naming the evidence boundary, and does not rely on defensive self-negation or prebuttal phrasing to signal rigor
- the in-scope surface passes the editor-first readout test from `scientific-voice-guide.md`, meaning a skimming editor would remember the supported discovery rather than the caveat
- the in-scope surface passes a sentence-energy test: high-salience sentences use clear subjects, explicit causal links, and low enough noun-stack density for one-pass reading
- the in-scope surface passes an architecture test: the protagonist is stable, the pivot is legible, tool sections remain subordinate to discovery, and the Discussion opening upgrades the paper's meaning rather than restating Results
- whole-manuscript and cross-section rounds provide a `Paper spine map` showing Results section jobs, discovery cash-out location, discovery-versus-tool weight, second-layer discovery, broader-implication trunk, downstream-consequence branch, and redundancy or breathing risks
- whole-manuscript and cross-section rounds with broader significance also record a broader-significance status and show why the trunk or branch earned promotion rather than relying on intuition alone
- paragraph and section transitions are natural and preserve the paper-level logic
- terminology, comparator labels, and mechanism language stay consistent across sections unless an explicit shift is introduced
- unresolved placeholders are explicitly tracked or resolved
- manuscript closeout claims about textual changes are backed by exact text or diff evidence, with interpretation labeled separately
- manuscript review pass is not treated as equivalent to full implementation or full plan completion
- paper-facing broader significance passes a `no-bolt-on` test: if the second-layer discovery is removed, downstream consequences lose support immediately; if only the downstream consequence is removed, the manuscript still lands its broader paper-level inference
- paper-facing broader significance passes the `earned-discovery`, `boundary-pressure`, and `reviewer-routing survival` tests at the claimed promotion level
- paper-facing checks under `make paper-check` pass

## Executable gates

- `python scripts/paper/check_required_sections.py`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/verify_provenance.py`
- `python scripts/paper/check_round_governance_semantics.py --round-dir results/<round_name>` for any `high-risk` broader-significance round
- `make paper-governance-gate ROUND_DIR=results/<round_name>` for any `high-risk` broader-significance round
- `make paper-build`
