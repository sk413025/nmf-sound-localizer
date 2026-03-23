# Figure Retention Matrix

Date: 2026-03-12

Archived note: this document captures a retired `Fig. 1–9` decision surface
from before the branch was consolidated around the current active six-figure
manuscript. Asset paths below are historical unless explicitly updated.

Purpose: provide a decision-ready inventory of current main figures, supplementary figures, and legacy predecessors so the final paper can keep only the assets that remain scientifically necessary and provenance-safe after the `Master Figure 3` family was expanded into `Figs. 5–8`.

This note is a working decision aid, not a canonical registry. Canonical provenance remains in:

- `figures/FIGURE_REGISTRY.md`
- `figures/conf/review_targets.yaml`
- `figures/conf/experiments.yaml`

## Recommendation rubric

- `Keep main`: keep in the main paper by default.
- `Move to supp`: scientifically useful, but secondary to the main narrative.
- `Needs provenance fix`: relevant enough to keep, but not safe as a final paper asset until provenance is completed.
- `Legacy only`: useful for historical interpretation or internal discussion, but superseded as a final paper figure.
- `Archive`: redundant or superseded and not recommended for the final paper package.

---

## Recommended final shortlist

### Main paper

| Figure | Default recommendation | Reason |
|---|---|---|
| Fig. 1 | Keep main | Introduces the sensing setup and the physical intuition for spectral fingerprints. |
| Fig. 2 | Keep main | Establishes the physical dictionary and modal basis used by the whole paper. |
| Fig. 3 | Keep main | Defines the model architecture and the physics-guided unrolled inference pipeline. |
| Fig. 4 | Keep main | Provides core robustness and ablation evidence. |
| Fig. 5 | Keep main | Restores the direct `Master Figure 3` structure + macro-selection lineage. |
| Fig. 6 | Keep main | Makes the representative-angle routing improvement quantitative. |
| Fig. 7 | Keep main | Keeps the full-band and low-band mechanism diagnostics visible in the main sequence. |
| Fig. 8 | Keep main | Completes the band-resolved mechanism story. |
| Fig. 9 | Needs provenance fix | Strong paper-level claim, but the current registry still marks the upstream evidence layer as incomplete. |

### Supplementary

| Figure | Default recommendation | Reason |
|---|---|---|
| Supp Fig. 9 | Keep as supplementary | Useful final-outcome diagnostics, but no longer needed to replace a main-paper mechanism panel. |

### Legacy

| Legacy asset family | Default recommendation | Reason |
|---|---|---|
| `Master Figure 3` | Legacy only | Historical predecessor of the current `Figs. 5–8` family. |
| Temporary split `Fig. 5 Structure Alignment` + `Fig. 6 Routing Mechanism` | Legacy only | Useful provenance layers for the current mechanism family, but not final manuscript figures. |
| Early Figure 5 atomic panels (`Fig5_A*`, `Fig5_B*`, `Fig5_C*`) | Archive or legacy only | Valuable for traceability, but too granular for the submission package. |
| `Fig5_B3_POLAR_ESTIMATION` | Legacy only | Superseded by the band-wise diagnostic branch now promoted into `Figs. 7–8`. |

---

## Current figures and panel-level mapping

### Fig. 5 — Structure Alignment + Macro Selection Robustness

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | `H`-matrix physical structure + learned `QK` structure | `paper/figures/fig05_structure-macro-selection.png`; split panel `fig05_panel_a_global_attention.pdf`; upstream `fig05_structure_alignment.pdf` | `modal_routing_val.npz` + `dictionary.npz` | `Master Figure 3a` | Keep main |
| b | OMP vs physics-aware all-angle selection probability | `paper/figures/fig05_structure-macro-selection.png`; split panel `fig05_panel_c_macro_robustness.pdf`; upstream `fig05_structure_alignment.pdf` | `modal_routing_val.npz` + `dictionary.npz` | `Master Figure 3c` | Keep main |

### Fig. 6 — Angle-Specific Mechanism Diagnostics

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | `55°` baseline vs no-transformer distribution | `paper/figures/fig06_angle-specific-mechanism.png`; upstream `fig09_angle55.pdf` | baseline + no-transformer `metrics.npz` | `Master Figure 3b` upper block | Keep main |
| b | `100°` baseline vs no-transformer distribution | `paper/figures/fig06_angle-specific-mechanism.png`; upstream `fig09_angle100.pdf` | baseline + no-transformer `metrics.npz` | `Master Figure 3b` upper block | Keep main |

### Fig. 7 — Band-Wise Routing Diagnostics I

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | Full-band smoothed diagnostic | `paper/figures/fig07_bandwise-routing-analysis-part1.png`; copied panel PDF | `Fig5_B3_LINE_300_3000.pdf` smoothed run | `Master Figure 3b` lower replacement branch | Keep main |
| b | Full-band no-smoothing diagnostic | `paper/figures/fig07_bandwise-routing-analysis-part1.png`; copied panel PDF | `Fig5_B3_LINE_300_3000.pdf` corrected run | `Master Figure 3b` lower replacement branch | Keep main |
| c | `300–500 Hz` diagnostic | `paper/figures/fig07_bandwise-routing-analysis-part1.png`; copied panel PDF | `Fig5_B3_LINE_300_500.pdf` corrected run | `Master Figure 3b` lower replacement branch | Keep main |

### Fig. 8 — Band-Wise Routing Diagnostics II

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | `500–1000 Hz` diagnostic | `paper/figures/fig08_bandwise-routing-analysis-part2.png`; copied panel PDF | `Fig5_B3_LINE_500_1000.pdf` corrected run | `Master Figure 3b` lower replacement branch | Keep main |
| b | `1000–2000 Hz` diagnostic | `paper/figures/fig08_bandwise-routing-analysis-part2.png`; copied panel PDF | `Fig5_B3_LINE_1000_2000.pdf` corrected run | `Master Figure 3b` lower replacement branch | Keep main |
| c | `2000–3000 Hz` diagnostic | `paper/figures/fig08_bandwise-routing-analysis-part2.png`; copied panel PDF | `Fig5_B3_LINE_2000_3000.pdf` corrected run | `Master Figure 3b` lower replacement branch | Keep main |

### Fig. 9 — Cross-Material Universality

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | Material exemplars | `paper/figures/fig09_cross-material-universality.jpg`; split panel `fig09_panel_a_material_exemplars.png` | Upstream evidence not yet registered in machine-readable form | Old cross-material figure family | Needs provenance fix |
| b | Cross-material heatmaps | `paper/figures/fig09_cross-material-universality.jpg`; split panel `fig09_panel_b_heatmaps.png` | Upstream evidence not yet registered in machine-readable form | Old cross-material figure family | Needs provenance fix |
| c | RMSE comparison | `paper/figures/fig09_cross-material-universality.jpg`; split panel `fig09_panel_c_rmse_comparison.png` | Upstream evidence not yet registered in machine-readable form | Old cross-material figure family | Needs provenance fix |

### Supp Fig. 9 — Confusion Matrix Family

| Panel | Content | Current asset layer | Evidence source | Legacy predecessor | Default recommendation |
|---|---|---|---|---|---|
| a | Confusion heatmaps | `paper/figures/fig09_heatmaps.pdf`; upstream `fig09_heatmaps.pdf` | baseline + no-transformer `metrics.npz` | Earlier `Supp Fig. 8` | Keep as supplementary |
| b | Angle 55 comparison | `paper/figures/fig09_angle55.pdf`; upstream `fig09_angle55.pdf` | same `metrics.npz` pair | Earlier `Supp Fig. 8` | Keep as supplementary |
| c | Angle 100 comparison | `paper/figures/fig09_angle100.pdf`; upstream `fig09_angle100.pdf` | same `metrics.npz` pair | Earlier `Supp Fig. 8` | Keep as supplementary |

---

## Final keep / move / archive matrix

### Recommended to keep in the main paper

- Fig. 1
- Fig. 2
- Fig. 3
- Fig. 4
- Fig. 5
- Fig. 6
- Fig. 7
- Fig. 8

### Recommended to keep only if provenance is repaired

- Fig. 9

### Recommended to keep in supplementary

- Supp Fig. 9a
- Supp Fig. 9b
- Supp Fig. 9c

### Recommended to keep only as historical provenance or internal reference

- `Master Figure 3`
- Temporary split `Fig. 5 Structure Alignment`
- Temporary split `Fig. 6 Routing Mechanism`
- Historical Figure 5 atomic panels
- `Fig5_B3_POLAR_ESTIMATION`

### Recommended to archive from the active final-paper decision surface

- Redundant atomic-level predecessors once the corresponding current manuscript figure is retained
- Any legacy figure that repeats the same claim with weaker styling, weaker provenance registration, or lower manuscript fit than the current figure family
