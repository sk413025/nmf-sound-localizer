# Master Figure 3 Lineage Bundle

Date: 2026-03-12

This bundle concentrates the original `Master Figure 3`, its current active descendants, panel-level support PDFs, and nearby figure families that are often confused with the same lineage.

It is intended for side-by-side comparison and manuscript-facing provenance discussion. It is not the canonical source of truth. Canonical registry and workflow metadata remain in:

- `figures/FIGURE_REGISTRY.md`
- `docs/working-notes/master_figure3_lineage_crosswalk.md`
- `docs/working-notes/figure_retention_matrix.md`

## What This Bundle Contains

### `00_original_anchor/`

- `master_figure3_original.png`

The original legacy composite that anchors the lineage discussion.

### `01_current_canonical/`

- `current_fig05_composite.png`

The current paper-facing composite descendant. Visually, this remains the direct descendant of the old three-panel `Master Figure 3` concept:

- panel a: global physics-consistent attention
- panel b: micro-level mechanism
- panel c: macro-level selection robustness

### `02_upstream_pdfs/`

- `fig05_structure_alignment.pdf`
- `fig06_routing_mechanism.pdf`
- `fig05_panel_a_global_attention.pdf`
- `fig05_panel_c_macro_robustness.pdf`

These PDFs show how the original composite was split during the generator era:

- old panel `a + c` flowed into `fig05_structure_alignment.pdf`
- old panel `b` flowed into `fig06_routing_mechanism.pdf`
- current isolated panel assets exist for `a` and `c`

### `03_panel_b_support_results/`

- `fig5_b3_line_fullband_smooth.pdf`
- `fig5_b3_line_fullband_nosmooth.pdf`
- `fig5_b3_line_300_500_nosmooth.pdf`
- `fig5_b3_line_500_1000_nosmooth.pdf`
- `fig5_b3_line_1000_2000_nosmooth.pdf`
- `fig5_b3_line_2000_3000_nosmooth.pdf`

These are support descendants of the old panel `b`, not a separate figure family. They preserve the later band-wise decomposition branch of the micro-mechanism case-study analysis.

### `04_separate_families/`

- `master_figure4_cross_material_legacy.jpg`
- `current_fig06_cross_material.jpg`
- `supp_fig09_heatmaps.pdf`
- `supp_fig09_angle55.pdf`
- `supp_fig09_angle100.pdf`
- `legacy_performance_family.jpg`

These assets are included to prevent a numbering trap. They are nearby in historical numbering, but they do not descend from `Master Figure 3`.

### `05_pdf_previews/`

PNG previews of the PDFs above, generated from page 1, for rapid visual comparison without opening every PDF.

## Core Lineage Conclusions

1. The active descendant of the original `Master Figure 3` is the current `Fig. 5` family.
2. The split-era generator outputs decomposed that lineage into:
   - `fig05_structure_alignment.pdf` for the old `a + c`
   - `fig06_routing_mechanism.pdf` for the old `b`
3. The `Fig5_B3_LINE_*` PDFs are later support descendants of panel `b`.
4. Current `Fig. 6`, the old cross-material family, and the supplementary confusion family are separate lineages.

## Recommended Reading Order

1. `00_original_anchor/master_figure3_original.png`
2. `01_current_canonical/current_fig05_composite.png`
3. `02_upstream_pdfs/`
4. `03_panel_b_support_results/`
5. `04_separate_families/`

## Chinese Companion

A Chinese explanation bundle is kept locally under the ignored path:

- `results/master_figure3_lineage_bundle_20260312/`

That local bundle is for working discussion only. The tracked repository version stays in English to follow branch governance.
