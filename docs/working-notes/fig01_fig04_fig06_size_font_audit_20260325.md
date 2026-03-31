# Figure 1, 4, and 6 Size and Font Audit

Date: 2026-03-25

## Scope

- This note audits the current manuscript-facing `paper/figures/fig01_paradigm-shift.jpg`, `paper/figures/fig04_solver-dynamics.jpg`, and `paper/figures/fig06_universality.jpg`.
- Figure and panel `mm` sizes come from the realized manuscript layout sidecars under `paper/figures/*.layout.json`.
- `exact` font sizes come from text-span extraction on the actual PDF assets used by the final manuscript composition.
- `approx` font sizes come from OCR on the manuscript-visible crop after the composer crop/scale step for manual raster panels.
- For manual panels, the final composite panel label is not inherited from the raster source. It is redrawn by the composer at `8.0 pt`.

## Composite Sizes

| Figure | Composite size (mm) |
| --- | --- |
| Fig. 1 | `183.0 x 135.0` |
| Fig. 4 | `183.0 x 128.0` |
| Fig. 6 | `183.0 x 144.0` |

## Panel Sizes

### Fig. 1

| Panel | Title | Size (mm) | Origin `x0,y0` (mm) | Source |
| --- | --- | --- | --- | --- |
| a | Experimental setup | `89.0 x 65.0` | `0.0, 70.0` | `paper/figures/fig01_paradigm-shift.layout.json` |
| b | Physical mechanism schematic | `89.0 x 65.0` | `94.0, 70.0` | `paper/figures/fig01_paradigm-shift.layout.json` |
| c | Input -> output spectral shaping | `43.824 x 45.369` | `10.98, 11.76` | `paper/figures/fig01_paradigm-shift.layout.json` |
| d | Trial repeatability (+/-1 s.d.) | `43.824 x 45.369` | `72.333, 11.76` | `paper/figures/fig01_paradigm-shift.layout.json` |
| e | Directivity | `43.824 x 43.824` | `133.687, 12.533` | `paper/figures/fig01_paradigm-shift.layout.json` |

### Fig. 4

| Panel | Title | Size (mm) | Origin `x0,y0` (mm) | Source |
| --- | --- | --- | --- | --- |
| a | Architecture diagram | `183.0 x 88.0` | `0.0, 40.0` | `paper/figures/fig04_solver-dynamics.layout.json` |
| b | Training convergence | `58.333 x 36.0` | `0.0, 0.0` | `paper/figures/fig04_solver-dynamics.layout.json` |
| c | Ablation comparison | `58.333 x 36.0` | `62.333, 0.0` | `paper/figures/fig04_solver-dynamics.layout.json` |
| d | Per-angle accuracy | `58.333 x 36.0` | `124.667, 0.0` | `paper/figures/fig04_solver-dynamics.layout.json` |

### Fig. 6

| Panel | Title | Size (mm) | Origin `x0,y0` (mm) | Source |
| --- | --- | --- | --- | --- |
| a | Material exemplars | `183.0 x 24.0` | `0.0, 120.0` | `paper/figures/fig06_universality.layout.json` |
| b | Representative heatmaps | `183.0 x 34.0` | `0.0, 82.0` | `paper/figures/fig06_universality.layout.json` |
| c | Cross-material RMSE | `62.0 x 78.0` | `0.0, 0.0` | `paper/figures/fig06_universality.layout.json` |
| d | Per-band SVD spectra | `116.0 x 37.0` | `67.0, 41.0` | `paper/figures/fig06_universality.layout.json` |
| e | Band-resolved routing | `116.0 x 37.0` | `67.0, 0.0` | `paper/figures/fig06_universality.layout.json` |

## Font Inventory

### Fig. 1

| Panel | Visible text role | Size (pt) | Precision | Evidence |
| --- | --- | --- | --- | --- |
| a | Panel label `a` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| a | Main photo labels such as `Sound Source`, `Sensor Plate` | `~6.5-7.0` | approx | OCR on final manuscript-visible crop |
| a | Smaller parenthetical labels such as `(Loudspeaker)`, `(Acrylic Target)`, `Optical Table` | `~5.5-6.0` | approx | OCR on final manuscript-visible crop |
| b | Panel label `b` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| b | Larger schematic words such as `Input`, `Total Plate`, `Single-Point` | `~7.0-7.5` | approx | OCR on final manuscript-visible crop |
| b | Medium schematic labels such as `Direction`, `Mode 1/2/3`, `Fingerprint A` | `~6.5-7.0` | approx | OCR on final manuscript-visible crop |
| b | Smaller helper text such as `Resulting`, `(Unique to B)` | `~5.5-6.0` | approx | OCR on final manuscript-visible crop |
| c | Panel label `c` | `8.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| c | Panel title `Input -> output spectral shaping` | `6.5` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| c | Axis labels | `6.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| c | Tick labels | `7.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| c | Legend text | `4.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| d | Panel label `d` | `8.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| d | Panel title `Trial repeatability (+/-1 s.d.)` | `6.5` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| d | Axis labels | `6.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| d | Tick labels | `7.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| d | Legend text | `5.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| e | Panel label `e` | `8.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| e | Panel title `Directivity` | `6.5` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| e | Angular and radial tick labels | `5.0` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |
| e | Legend text | `4.5` | exact | PDF span extraction from `figures/output/fig01_paradigm_data.pdf` |

### Fig. 4

| Panel | Visible text role | Size (pt) | Precision | Evidence |
| --- | --- | --- | --- | --- |
| a | Panel label `a` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| a | Stage headers such as `Stage t (Detailed View)` | `~9.0-9.5` | approx | OCR on final manuscript-visible crop |
| a | Major block labels such as `Transformer Encoder`, `Physics-Aware Sparse Masking`, `Argmax Index Selection & Mapping` | `~7.5-8.0` | approx | OCR on final manuscript-visible crop |
| a | Medium labels such as `Input Residual`, `Physics Dictionary`, `Sparse Accumulation`, `New Residual` | `~7.0-7.5` | approx | OCR on final manuscript-visible crop |
| a | Smaller formulas and index text | `~5.0-6.5` | approx | OCR on final manuscript-visible crop |
| b | Panel label `b` | `8.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_b_convergence.pdf` |
| b | Axis labels | `6.5` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_b_convergence.pdf` |
| b | Tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_b_convergence.pdf` |
| b | Legend text | `6.5` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_b_convergence.pdf` |
| c | Panel label `c` | `8.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_c_ablation.pdf` |
| c | X-axis label | `6.5` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_c_ablation.pdf` |
| c | X tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_c_ablation.pdf` |
| c | Y category labels such as `guided solver`, `router-bypass` | `6.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_c_ablation.pdf` |
| d | Panel label `d` | `8.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_d_perangle.pdf` |
| d | Axis labels | `6.5` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_d_perangle.pdf` |
| d | Tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig04_solver_dynamics_panels/fig04_panel_d_perangle.pdf` |

### Fig. 6

| Panel | Visible text role | Size (pt) | Precision | Evidence |
| --- | --- | --- | --- | --- |
| a | Panel label `a` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| a | In-panel text after final crop | none stably visible | exact | Final manuscript-visible crop inspection |
| b | Panel label `b` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| b | Repeated heatmap axis text and small tick labels | `~4.0-5.0` | approx | OCR plus visual inspection on final manuscript-visible crop |
| b | Smaller micro text | unresolved below `~4 pt` | approx | OCR on final manuscript-visible crop is unstable |
| c | Panel label `c` | `8.0` | exact | Composer draw token in `scripts/paper/compose_master_figure3_family.py` |
| c | Legend entries `OMP baseline`, `guided solver` | `~5.0-5.5` | approx | OCR plus visual inspection on final manuscript-visible crop |
| c | X-category labels and y tick labels | `~7.5-8.0` | approx | OCR plus visual inspection on final manuscript-visible crop |
| c | Callout annotations such as `Stable, Low Error Globally`, `Falls under Complexity` | `~7.0-8.5` | approx | OCR plus visual inspection on final manuscript-visible crop |
| d | Panel label `d` | `8.0` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| d | X-axis label | `6.5` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| d | Y-axis main math glyphs (`sigma`, slash) | `6.5` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| d | Y-axis math subscript glyphs (`r`, `1`) | `4.2-4.55` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| d | Tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| d | Legend text | `6.5` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf` |
| e | Panel label `e` | `8.0` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf` |
| e | Y-axis label | `6.5` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf` |
| e | X tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf` |
| e | Y tick labels | `6.0` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf` |
| e | Legend text | `6.5` | exact | PDF span extraction from `figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf` |

## Notes

- Fig. 1 `c/d/e` must be read from the composite generator PDF `figures/output/fig01_paradigm_data.pdf`, because the manuscript figure uses the full rendered `cde` strip rather than the split panel PDFs.
- Fig. 1 `c/d/e` actual output does not fully match the shared token table in the manuscript sidecar:
  - axis labels render at `6.0 pt`, not `6.5 pt`
  - tick labels render at `7.0 pt`, not `6.0 pt`
  - legends render at `4.0-5.0 pt`, below the branch nominal `5-7 pt` non-panel band
- Fig. 4 `a` source support art contains extra lower caption content upstream, but that lower caption is cropped out before the manuscript composite and is not counted here.
- Fig. 6 `a/b/c` are raster/provenance-gap support panels. Their font sizes are therefore approximate rather than exact without the upstream editable design source.
