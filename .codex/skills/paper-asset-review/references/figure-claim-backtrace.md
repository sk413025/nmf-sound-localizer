# Figure Claim Backtrace

Use this reference when the user is not asking for a formal gate decision, but instead asks:

- what a figure or panel means
- how a figure compares with a critique, reviewer concern, or rewrite proposal
- whether the current figure really supports a manuscript claim
- why a panel exists and how it connects to the paper logic

Use this order:

1. Locate the manuscript anchors.
   - Read the relevant Results, legend, and Methods lines first.
2. Inspect the real asset.
   - View the paper-facing `jpg` or `png` asset.
   - For `pdf`, inspect every page preview first.
3. Inspect split panels or bundle context when available.
   - Use the review bundle to distinguish panel-local issues from composition issues.
4. Inspect the generator or composition code.
   - Confirm what metric, subset, or transformation each panel actually plots.
5. Inspect the evidence or provenance artifact.
   - Confirm the panel is backed by committed data, logs, or executed result bundles.
6. Reconcile the four layers.
   - manuscript text
   - visual asset
   - generator or composition code
   - provenance or executed evidence

Preferred answer structure:

1. Panel intent
2. What the figure actually shows
3. Agreement or mismatch across manuscript, asset, generator, and evidence
4. Whether the critique is already addressed, partly addressed, or still valid
5. What is still missing for a stronger paper-level claim

When a user asks for a "difference" between a figure and a critique, do not jump straight to recommendations. First separate:

- already absorbed by the current figure
- still missing from the current figure
- stronger-than-evidence claims made by the critique itself

Escalate to formal `paper-asset-review` gate output when:

- the recommendation would move a figure between main paper and supplementary
- the figure may need splitting, replacement, or reclassification
- visual content and provenance disagree
