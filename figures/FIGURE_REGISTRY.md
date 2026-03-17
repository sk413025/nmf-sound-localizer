# Figure Registry — Nature Communications Paper

> Master index mapping manuscript figures to experiment sources, generators,
> and reproduction contracts. Numbering now follows the expanded manuscript
> sequence (Figs. 1–9 + Supp Fig. 9).
>
> Asset model used in this branch:
> 1. **Evidence source**: data/results/checkpoints that support the scientific claim
> 2. **Generator output**: clean plotted asset produced from code, when applicable
> 3. **Split panel assets**: internal top-level manuscript panels stored under `figures/output/*_panels/`
> 4. **Manuscript asset**: final paper-facing composite stored under `paper/figures/`
>
> Codex review bundles judge the **manuscript asset**, but multi-panel figures must
> also carry split top-level panel assets in the bundle context so review can separate
> panel-local issues from recomposition.

## Overview

| Figure | Description | Asset model | Upstream contract |
|--------|-------------|-------------|-------------------|
| Fig 1 | Setup photo + spectral fingerprint schematic | Manual manuscript composite with canonical split panels | No generator expected |
| Fig 2 | SVD spectrum + modal decomposition + dictionary | Data-backed manuscript composite | `fig02_svd_spectrum.py` + `h_matrix` |
| Fig 3 | Unrolled network architecture diagram | Manual manuscript asset | No generator expected |
| Fig 4 | Noise robustness + component ablation | Data-backed manuscript composite | `fig04_snr_ablation.py` + `figure4_data.json` |
| Fig 5 | Structure alignment + macro selection robustness | Data-backed manuscript composite | `fig05_structure_alignment.py` + routing run |
| Fig 6 | Angle-specific mechanism diagnostics | Data-backed manuscript composite | `fig09_confusion_matrix.py` + `metrics.npz` |
| Fig 7 | Band-wise routing diagnostics I | Data-backed manuscript composite | `fig5_b3_band_decomposition.py` + committed results PDFs |
| Fig 8 | Band-wise routing diagnostics II | Data-backed manuscript composite | `fig5_b3_band_decomposition.py` + committed results PDFs |
| Fig 9 | Cross-material universality | Manuscript composite with provenance gap | Machine-readable upstream evidence not yet registered |
| Supp Fig 9 | Confusion matrix family | Data-backed deployed generator output | `fig09_confusion_matrix.py` + `metrics.npz` |

---

## Figure Details

### Fig 1 — Experimental Setup + Spectral Fingerprint

- **Asset model:** Manual manuscript asset
- **Panels:** (a) Photograph of LDV measurement setup; (b) Spectral fingerprint schematic
- **Evidence source:** Photograph + illustration
- **Generator output:** None expected
- **Manuscript asset:** `paper/figures/fig01_paradigm-shift.jpg`
- **Split panel assets:** `figures/output/fig01_paradigm_shift_panels/`
- **Panel manifest:** `figures/output/fig01_paradigm_shift_panels/fig01_panel_manifest.json`
- **Provenance note:** The current manuscript asset is a clean panel-first recomposition. The split panel PNGs under `figures/output/fig01_paradigm_shift_panels/` are now treated as the canonical manual panel assets for review and submission support; no generator-backed upstream source is expected.

---

### Fig 2 — SVD Spectrum + Modal Decomposition + Dictionary

- **Asset model:** Data-backed manuscript composite
- **Panels:** (a) Singular-value spectrum; (b) Modal decomposition (freq + polar); (c) Dictionary heatmap block
- **Evidence source:** `h_matrix_normalized_original_to_box.pth` (via `figures/conf/paths.yaml`)
- **Generator:** `figures/generators/fig02_svd_spectrum.py`
- **Experiment branch:** `feature/omp-transformer-modal-viz` @ `15b2981`
- **Generator output:** `figures/output/fig02_svd_spectrum.{pdf,tiff}`
- **Split panel assets:** `figures/output/fig02_svd_spectrum_panels/`
- **Panel manifest:** `figures/output/fig02_svd_spectrum_panels/fig02_panel_manifest.json`
- **Manuscript asset:** `paper/figures/fig02_svd-physical-dictionary.jpg`
- **Review rule:** Codex reviews the manuscript composite, but it must remain visually and scientifically faithful to the upstream generator output.

---

### Fig 3 — Unrolled Network Architecture

- **Asset model:** Manual manuscript asset
- **Panels:** Full-page architecture diagram of the unrolled OMP-Transformer
- **Evidence source:** Architecture illustration
- **Generator output:** None expected
- **Manuscript asset:** `paper/figures/fig04_unrolled-attention-omp.jpg`

---

### Fig 4 — Noise Robustness + Component Ablation

- **Asset model:** Data-backed manuscript composite
- **Panels:** (a) Ablation strip chart (SNR=Inf slice); (b) Multi-variant SNR degradation curves
- **Evidence source:** `results/figure4_data.json`
- **Generator:** `figures/generators/fig04_snr_ablation.py`
- **Data:** Babble Speech260 full sweep — `exp-omp-ablation-snr-rerun-20260128` @ `14feb94`
  - 245 runs: 7 variants × 7 SNR levels × 5 seeds
  - Aggregated: `results/figure4_data.json`
- **Generator output:** `figures/output/fig04_snr_ablation.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig04_noise-robustness-ablation.jpg`
- **Split panel assets:** `figures/output/fig04_snr_ablation_panels/`
- **Panel manifest:** `figures/output/fig04_snr_ablation_panels/fig04_panel_manifest.json`

---

### Fig 5 — Structure Alignment + Macro Selection Robustness

Manuscript title: "Global structure alignment and all-angle macro selection robustness"

- **Asset model:** Data-backed manuscript composite
- **Panels:**
  - (a) Structure alignment: `H`-matrix physical structure and learned `QK` structure
  - (b) Macro selection robustness: all-angle selection probability for traditional OMP and physics-aware AI
- **Evidence source:**
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
- **Generator:** `figures/generators/fig05_structure_alignment.py`
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Generator output:** `figures/output/fig05_structure_alignment.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig05_structure-macro-selection.png`
- **Split panel assets:** `figures/output/fig05_structure_macro_panels/`
- **Panel manifest:** `figures/output/fig05_structure_macro_panels/fig05_panel_manifest.json`
- **Lineage note:** This figure is the direct active descendant of the original `Master Figure 3` panel-a + panel-c family. The earlier manuscript-era confusion-heatmap replacement for Fig. 5c has been retired from the main-paper figure path and restored to the supplementary confusion family only.

---

### Fig 6 — Angle-Specific Mechanism Diagnostics

- **Asset model:** Data-backed manuscript composite
- **Panels:**
  - (a) Baseline vs no-transformer routing distributions at `55°`
  - (b) Baseline vs no-transformer routing distributions at `100°`
- **Evidence source:**
  - `results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz`
  - `results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz`
- **Generator:** `figures/generators/fig09_confusion_matrix.py`
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Generator output:** `figures/output/fig09_angle55.{pdf,tiff}`, `figures/output/fig09_angle100.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig06_angle-specific-mechanism.png`
- **Split panel assets:** `figures/output/fig06_angle_specific_mechanism_panels/`
- **Panel manifest:** `figures/output/fig06_angle_specific_mechanism_panels/fig06_panel_manifest.json`
- **Lineage note:** This figure replaces the old upper micro-mechanism block from `Master Figure 3b` with angle-specific probability distributions that make the same correct-vs-off-axis contrast quantitative at two representative directions.

---

### Fig 7 — Band-Wise Routing Diagnostics I

- **Asset model:** Data-backed manuscript composite
- **Panels:**
  - (a) Full-band routing diagnostic with smoothing
  - (b) Full-band routing diagnostic without smoothing
  - (c) `300–500 Hz` routing diagnostic
- **Evidence source:**
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/fig5_b3_line5_20260202_224349/Fig5_B3_LINE_300_3000.pdf`
  - `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_300_3000.pdf`
  - `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_300_500.pdf`
- **Generator:** `scripts/fig5_b3_band_decomposition.py`
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Generator output:** committed results PDFs copied into `figures/output/fig07_bandwise_routing_part1_panels/`
- **Manuscript asset:** `paper/figures/fig07_bandwise-routing-analysis-part1.png`
- **Split panel assets:** `figures/output/fig07_bandwise_routing_part1_panels/`
- **Panel manifest:** `figures/output/fig07_bandwise_routing_part1_panels/fig07_panel_manifest.json`
- **Lineage note:** These panels are active descendants of the `Master Figure 3b` lower band-decomposition branch. The smoothed full-band and corrected no-smoothing full-band views are both retained because they document the historical transition and the final preferred rendering.

---

### Fig 8 — Band-Wise Routing Diagnostics II

- **Asset model:** Data-backed manuscript composite
- **Panels:**
  - (a) `500–1000 Hz` routing diagnostic
  - (b) `1000–2000 Hz` routing diagnostic
  - (c) `2000–3000 Hz` routing diagnostic
- **Evidence source:**
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_500_1000.pdf`
  - `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_1000_2000.pdf`
  - `results/fig5_b3_line5_nosmooth_20260202_233434/Fig5_B3_LINE_2000_3000.pdf`
- **Generator:** `scripts/fig5_b3_band_decomposition.py`
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Generator output:** committed results PDFs copied into `figures/output/fig08_bandwise_routing_part2_panels/`
- **Manuscript asset:** `paper/figures/fig08_bandwise-routing-analysis-part2.png`
- **Split panel assets:** `figures/output/fig08_bandwise_routing_part2_panels/`
- **Panel manifest:** `figures/output/fig08_bandwise_routing_part2_panels/fig08_panel_manifest.json`
- **Lineage note:** This figure completes the active `Master Figure 3b` band-wise family. The earlier polar-estimation subpanel has been dropped from the manuscript path and retained only in the lineage notes.

---

### Fig 9 — Cross-Material Universality

- **Asset model:** Manuscript composite with provenance gap
- **Panels:**
  - (a) Five target objects (acrylic plate, paper cup, wooden board, cardboard box, laptop shell)
  - (b) Dictionary/response heatmaps per material
  - (c) Cross-material RMSE comparison (OMP vs physics-aware AI)
- **Evidence source:** Per-object calibrate-and-retrain experiments (Methods §Cross-material)
- **Generator output:** Not yet registered in machine-readable form
- **Manuscript asset:** `paper/figures/fig09_cross-material-universality.jpg`
- **Split panel assets:** `figures/output/fig09_cross_material_universality_panels/`
- **Panel manifest:** `figures/output/fig09_cross_material_universality_panels/fig09_panel_manifest.json`
- **Panel-source note:** The current panel-first manuscript composite is rebuilt from legacy split panels copied from `figures/output/fig06_cross_material_universality_panels/`, not from a newly registered generator.
- **Release note:** Fig. 9 should not be treated as a purely manual figure. It currently has a panel-first manuscript asset, but the upstream data-backed generator/evidence contract is still missing and must be fixed before release.
- **Historical note:** This family was previously numbered as the manuscript `Fig. 6`, and before that as temporary `Fig. 7` during the split-era figure workflow.

---

### Supp Fig 9 — Confusion Matrix Family

- **Asset model:** Data-backed deployed generator output
- **Panels:** (a) Confusion matrices (Baseline vs No-Transformer); (b–c) Per-angle distribution comparisons
- **Evidence source:** Baseline + No-Transformer `metrics.npz` (via `figures/conf/paths.yaml`)
- **Generator:** `figures/generators/fig09_confusion_matrix.py`
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Generator output:** `figures/output/fig09_*.{pdf,tiff}`
- **Split panel assets:** `figures/output/supp_fig09_confusion_panels/`
- **Panel manifest:** `figures/output/supp_fig09_confusion_panels/supp_fig09_panel_manifest.json`
- **Manuscript asset:** `paper/figures/fig09_*.pdf` (deployed generator outputs)
- **Role note:** These assets remain supplementary outcome diagnostics. They are no longer reused as the active main-paper macro panel.

---

## Numbering History

The mechanism figure family changed twice during development:

```text
Original combined era              Split era                        Current manuscript
────────────────────              ─────────                        ──────────────────
Fig 1  Setup                      Fig 1  Setup                     Fig 1  Setup
Fig 2  SVD                        Fig 2  SVD                       Fig 2  SVD
Fig 3  Architecture               Fig 3  Architecture              Fig 3  Architecture
Fig 4  Ablation                   Fig 4  Ablation                  Fig 4  Ablation
Master Figure 3                   Fig 5  Structure Alignment       Fig 5  Structure + Macro
  panel a + panel c split           Fig 6  Routing Mechanism       Fig 6  Angle-specific mechanism
  panel b lower branch added                                      Fig 7  Band-wise diagnostics I
                                                                  Fig 8  Band-wise diagnostics II
Fig 6  Cross-material             Fig 7  Cross-material            Fig 9  Cross-material
Supp Fig 8  Confusion             Supp Fig 9  Confusion            Supp Fig 9  Confusion
```

The current manuscript therefore treats the old `Master Figure 3` as an expanded
four-figure mechanism family (`Figs. 5–8`) rather than as a single recombined panel set.

---

## Build Pipeline

```bash
# Generate all automated figures and split-panel manifests
make -C figures generate

# Validate compliance (dimensions, fonts, DPI)
make -C figures validate

# Deploy generator outputs and rebuild the manuscript-facing Figs. 5–9 assets
make -C figures deploy

# Prepare Codex review bundles for manuscript-facing assets
make -C figures review-prepare

# Manuscript-facing review entrypoint
python scripts/paper/review_paper_assets.py prepare

# After Codex writes role reports + final review.json, enforce review gate
python scripts/paper/review_paper_assets.py gate

# Existing build pipeline
make -C figures all

# Release pipeline (requires passing review gate)
make -C figures release
```
