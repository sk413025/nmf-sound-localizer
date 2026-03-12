# Nature Communications Submission Requirements

Verified against official Nature / Nature Communications pages on 2026-03-11.

This document is the canonical local reference for manuscript, figure, table, and submission-package requirements for this project. It is intended for Codex agents and human contributors who are editing `paper/`, `figures/`, submission metadata, or journal-facing assets.

## Scope

- Use this file for Nature Communications submission requirements.
- Use [AGENTS.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/AGENTS.md) for project workflow, experiment rigor, and manuscript-writing style.
- Treat [NATURE_FIGURE_GUIDELINES.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/NATURE_FIGURE_GUIDELINES.md) as a legacy quick note only.

## Precedence

When requirements conflict, follow this order:

1. Current live Nature Communications submission pages.
2. Current Nature Portfolio research figure guide.
3. Current Nature / Nature Communications policy pages.
4. Nature Communications PDF instructions and checklists.
5. Local project files, helper scripts, and legacy notes.

This matters because local code or older PDFs may preserve superseded rules.

## Official Sources

- Nature Communications article submission requirements:
  `https://www.nature.com/ncomms/submit/article`
- Nature Communications how-to-submit page:
  `https://www.nature.com/ncomms/submit/how-to-submit`
- Nature Portfolio figure guide:
  `https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/`
- Nature Portfolio figure panel and export guidance:
  `https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/`
- Nature Portfolio extended data guidance:
  `https://research-figure-guide.nature.com/figures/extended-data-formatting-guidelines/`
- Nature Communications formatting PDF:
  `https://www.nature.com/documents/ncomms-formatting-instructions.pdf`
- Nature Communications manuscript checklist PDF:
  `https://www.nature.com/documents/ncomms-manuscript-checklist.pdf`
- Springer Nature data availability policy:
  `https://www.springernature.com/gp/open-research/policies/data-availability-statements`
- Nature Portfolio AI policy:
  `https://www.nature.com/nature-portfolio/editorial-policies/ai`
- Nature Portfolio image integrity policy:
  `https://www.nature.com/nature-portfolio/editorial-policies/image-integrity`

## Manuscript Requirements

### First-submission formatting

- There is no strict first-submission template requirement.
- Prioritize readability over visual mimicry of the published journal layout.
- For Word manuscripts, Nature Communications recommends:
  - double spacing
  - single-column text
  - no full justification
  - Arabic page numbering

### Length and structure

- Title: at most 15 words.
- Abstract: at most 200 words and no references.
- Main text: target about 5,000 words.
- Methods: usually below 3,000 words.
- Main display items: at most 10 figures/tables combined.
- Figure legends: at most 350 words each.
- References: guideline target not more than 70 references.

### Required sections and metadata

Expect to provide, at minimum:

- Title
- Abstract
- Main text
- Methods
- References
- Figure legends
- Data availability
- Code availability when custom code is central to the claims
- Acknowledgements
- Funding statement
- Author contributions
- Competing interests

For corresponding authors, ORCID information is expected by the later submission stages.

## Figure Requirements

### Canonical dimensions

- Single-column width: 89 mm.
- Double-column width: 183 mm.
- Recommended maximum main-figure height: 170 mm.
- Submission systems may also ask that figure PDFs fit within a 210 x 276 mm page.

The 170 mm height guidance is the one to follow for this project's main figures, even though older Nature Communications documents mention deeper pages.

### Typography

- Use sans-serif fonts only.
- Prefer Arial or Helvetica.
- Panel labels should be lowercase bold letters, typically 8 pt.
- Most other figure text should be about 5-7 pt.
- Keep text editable in vector files.
- Do not outline or rasterize text unless a format constraint makes it unavoidable.
- Embed fonts in exported vector files.

### File formats and rasterization

- Prefer editable vector files for charts, line art, schematics, and composite scientific figures.
- Preferred deliverables for main figures are vector formats such as PDF, EPS, or AI.
- Use raster exports only when the content is intrinsically raster or photographic.
- For photographic or bitmap content, use high-resolution TIFF.
- Minimum raster resolution is typically:
  - 300 dpi for photographs / halftones
  - 600 dpi for combination artwork in older guidance
  - 1200 dpi for pure line art when vector delivery is not possible

Nature's live figure guide is stricter than some older PDF guidance. When unsure, preserve an editable vector master and export any raster derivative from that master.

### Visual style and accessibility

- Keep backgrounds white and uncluttered.
- Avoid rainbow colormaps.
- Avoid red/green-only encodings.
- Avoid colored text when standard black text works.
- Avoid decorative icons and 3D chart effects.
- Use line widths that remain legible after journal scaling.
- Keep axis symbols, units, and notation consistent with the manuscript.
- For microscopy or scale-dependent imaging, use scale bars rather than magnification labels.

### Project visual grammar for this branch

The official Nature guidance above is translated into the following branch policy for paper-facing figures:

- Panel labels use lowercase bold letters at about `8 pt`.
- Most other figure text should stay within `5–7 pt`.
- Subplot titles are allowed only when they remove real ambiguity, and should stay within the normal non-panel text band.
- Default figure text should be black rather than colored.
- Use a paper-wide semantic palette rather than ad hoc per-figure color choices:
  - baseline / physics / classical methods
  - learned / AI / transformer-driven methods
  - ablation / weakened variants
  - truth / target / emphasis
- Mode-specific colors are allowed only for genuinely mode-encoded figures such as Fig. 2; they are not the paper-wide semantic palette.
- Prefer sequential colormaps for nonnegative quantities and diverging colormaps for signed quantities.
- Avoid internal “master figure” headers, oversized section titles, and presentation-style narration inside figures.
- In multi-panel figures, size panels according to claim importance and minimize white space instead of forcing equal-size symmetry by default.
- Branch-local internal rule: every multi-panel figure must also keep split top-level panel assets plus a panel manifest under `figures/output/` for review and recomposition support. These split panels are internal assets, not the Nature-facing primary submission figures.

## Table Requirements

- Main-text tables should remain editable inside the manuscript file.
- Do not convert main tables into image files.
- Keep styling simple and publication-neutral.
- Use concise table titles.
- Avoid nested table numbering such as `Table 1a` and `Table 1b`.

For extended data tables, Nature's figure guide gives a more explicit layout target:

- width typically 89 mm or 180 mm
- single page where possible
- simple horizontal rules
- sans-serif text around 7 pt

## Submission Package Requirements Often Missed

- `Data availability` is required.
- `Code availability` is expected when custom code materially supports the claims.
- `Author contributions` must be present before publication.
- `Competing interests` must be declared.
- `Funding` should be explicit rather than implied through acknowledgements alone.
- Editors and reviewers may request access to critical custom code during review.
- If figure-level numeric data are required as source data, prepare organized source-data files per figure/table.
- Do not use `data not shown` as a substitute for accessible evidence.
- Ensure third-party figures, photos, or icons have clear reuse rights.

## Policy Constraints

### AI use

- LLMs or other AI systems cannot be listed as authors.
- Any substantive AI-assisted writing or editing should be disclosed per current Nature policy.

### Image integrity

- Do not make selective or misleading image edits.
- Global adjustments must preserve scientific meaning.
- Maintain raw data so image-processing choices can be defended if queried.

## Project Translation Layer

For this repository, the following operational rules apply:

- Any task touching `paper/`, `figures/`, figure exports, figure validators, or submission-facing docs must start by reading this file.
- If local code disagrees with this file, assume the local code may be stale and reconcile it deliberately.
- If Nature's live guidance changes, update this file before or alongside the code that implements the requirement.
- Codex multimodal review is the primary paper-level judge for figure suitability; code support should stay thin and should not be expanded into a heavy heuristic layout engine by default.
- Paper-facing figure decisions must be visual-first: do not infer figure meaning, panel identity, or claim support from filenames, legends, registry prose, or manuscript text alone.
- For `jpg` and `png`, review the image directly. For `pdf`, convert every page to PNG previews before Codex reviews the figure.
- For generated or data-backed figures, reconcile the visual asset against both the generator or composition code and the upstream evidence or provenance source before accepting it for manuscript or submission use.

### Codex-native review workflow

For paper-facing figures, use the review bundle workflow:

```bash
# Build or refresh assets
make -C figures all

# Prepare Codex review bundles
python scripts/paper/review_paper_assets.py prepare

# Review the actual image assets in Codex.
# If any reviewed asset is a PDF, inspect the generated page-by-page PNG previews first.
# Then trace the figure through its generator/composition code and evidence sources
# before writing role reports + review.json into each bundle directory

# Enforce stored review verdicts before release
python scripts/paper/review_paper_assets.py gate
```

Review bundles are written under `figures/review_artifacts/<figure_id>/` and include:

- `preview.png`
- `preview_overlay.png`
- `context.json`
- `workflow.json`
- `codex_review_prompt.md`
- `reviews/*.template.json`
- `review.json` (written by Codex during review)

### Current local mismatches to keep in mind

- [figures/style.py](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/figures/style.py) and [figures/validate.py](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/figures/validate.py) now enforce `MAX_HEIGHT_MM = 170`; expect older figure layouts to fail until they are redesigned.
- The current validator enforces paired `PDF + TIFF` outputs; the official guidance is more nuanced and centers on editable vector masters for main figures.
- The current validator still uses lightweight heuristics for text-size and palette-policy checks; final figure suitability still depends on Codex review.

## Suggested Codex Checklist

Before closing any manuscript or figure task, verify:

- manuscript structure still follows `Introduction -> Results -> Discussion -> Methods`
- title and abstract limits are respected
- display-item count remains within journal guidance
- figure width is 89 mm or 183 mm
- figure height is justified if it approaches 170 mm
- each paper-related figure decision was based on actual visual inspection of the asset
- each reviewed PDF was converted page-by-page into PNG previews before visual judgment
- each generated or data-backed figure was reconciled against its generator/composition code and upstream evidence source
- fonts are sans-serif and embedded
- exported figures preserve editable text
- legends are concise and within limit
- `Data availability`, `Code availability`, `Funding`, `Author contributions`, and `Competing interests` are present when required
- local validators and docs do not silently encode superseded Nature rules

## Notes on Older Local Documents

- [NATURE_FIGURE_GUIDELINES.md](/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/nature-comm-paper/NATURE_FIGURE_GUIDELINES.md) is retained only as a short legacy pointer.
- If it conflicts with this document, this document wins.
