# Manuscript Contract

Use this contract for any task that changes manuscript text, figure legends, paper structure, or the scientific narrative.

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
- Any manuscript change that depends on figure meaning, panel mapping, or figure lineage must be grounded in actual visual inspection of the figure asset.
- For `jpg` and `png` figure assets, inspect the image directly. For `pdf` assets, convert every page to PNG previews before interpreting the figure.
- For generated or data-backed figures, reconcile manuscript claims with both the generator or composition code and the upstream data or provenance artifacts, not with filenames or captions alone.
- If a multi-panel figure changes comparator family, evidence role, or panel-block purpose across panels, the Results prose and figure legend must state that panel logic explicitly rather than relying on the reader to infer it.

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
- paragraph and section transitions are natural and preserve the paper-level logic
- terminology, comparator labels, and mechanism language stay consistent across sections unless an explicit shift is introduced
- unresolved placeholders are explicitly tracked or resolved
- paper-facing checks under `make paper-check` pass

## Executable gates

- `python scripts/paper/check_required_sections.py`
- `python scripts/paper/check_figure_references.py`
- `python scripts/paper/verify_provenance.py`
- `make paper-build`
