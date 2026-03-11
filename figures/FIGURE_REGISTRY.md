# Figure Registry — Nature Communications Paper

> Master index mapping manuscript figures to experiment sources, generators,
> and reproduction commands. Numbering follows the manuscript (Figs. 1–6 + Supp Fig. 9).

## Overview

| Figure | Description | Status | Asset / Generator |
|--------|-------------|--------|-------------------|
| Fig 1 | Setup photo + spectral fingerprint schematic | MANUAL | `fig01_paradigm-shift.jpg` |
| Fig 2 | SVD spectrum + modal decomposition + dictionary | DONE | `fig02_svd_spectrum.py` |
| Fig 3 | Unrolled network architecture diagram | MANUAL | `fig03_unrolled-attention-omp.jpg` |
| Fig 4 | Noise robustness + component ablation | DONE | `fig04_snr_ablation.py` |
| Fig 5 | Routing mechanism analysis (structure + micro + macro) | DONE | `fig05_structure_alignment.py` + `fig06_routing_mechanism.py` |
| Fig 6 | Cross-material universality | MANUAL | `fig06_cross-material-universality.jpg` |
| Supp Fig 9 | Confusion matrix heatmaps | DONE | `fig09_confusion_matrix.py` |

---

## Figure Details

### Fig 1 — Experimental Setup + Spectral Fingerprint

- **Status:** MANUAL
- **Panels:** (a) Photograph of LDV measurement setup; (b) Spectral fingerprint schematic
- **Asset:** `paper/figures/fig01_paradigm-shift.jpg`
- **Source:** Photograph + illustration (no code pipeline)

---

### Fig 2 — SVD Spectrum + Modal Decomposition + Dictionary

- **Status:** DONE
- **Panels:** (a) Singular-value spectrum; (b) Modal decomposition (freq + polar); (c–e) Dictionary heatmaps
- **Generator:** `figures/generators/fig02_svd_spectrum.py`
- **Data:** `h_matrix_normalized_original_to_box.pth` (via `figures/conf/paths.yaml`)
- **Experiment branch:** `feature/omp-transformer-modal-viz` @ `15b2981`
- **Output:** `figures/output/fig02_svd_spectrum.{pdf,tiff}`

---

### Fig 3 — Unrolled Network Architecture

- **Status:** MANUAL
- **Panels:** Full-page architecture diagram of the unrolled OMP-Transformer
- **Asset:** `paper/figures/fig03_unrolled-attention-omp.jpg`
- **Source:** Illustration (no code pipeline)

---

### Fig 4 — Noise Robustness + Component Ablation

- **Status:** DONE
- **Panels:** (a) Ablation strip chart (SNR=Inf slice); (b) Multi-variant SNR degradation curves
- **Generator:** `figures/generators/fig04_snr_ablation.py`
- **Data:** Babble Speech260 full sweep — `exp-omp-ablation-snr-rerun-20260128` @ `14feb94`
  - 245 runs: 7 variants x 7 SNR levels x 5 seeds
  - Aggregated: `results/figure4_data.json`
- **Output:** `figures/output/fig04_snr_ablation.{pdf,tiff}`

---

### Fig 5 — Routing Mechanism Analysis

Manuscript title: "Deciphering model behaviour across scales: attention structure, micro-mechanism and macro-robustness"

- **Status:** DONE
- **Panels:**
  - (a) Global self-attention map — physics-consistent near-diagonal correlation structure
  - (b) Micro-level case study — OMP vs physics-aware selection heatmaps + band-wise line plots
  - (c) Selection-probability statistics — OMP off-diagonal vs QK sharp diagonal
- **Generators:**
  - `figures/generators/fig05_structure_alignment.py` → panels (a) and (c)
  - `figures/generators/fig06_routing_mechanism.py` → panel (b)
- **Data:** `results/omp_transformer_speech260_trainval_split_full_20251115_082341/`
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Output:**
  - `figures/output/fig05_structure_alignment.{pdf,tiff}`
  - `figures/output/fig06_routing_mechanism.{pdf,tiff}`
- **Note:** The two generators produce separate PDFs. The final manuscript figure
  (`fig05_routing-mechanism-analysis.png`) was composed from these panels.

---

### Fig 6 — Cross-Material Universality

- **Status:** MANUAL
- **Panels:**
  - (a) Five target objects (acrylic plate, paper cup, wooden board, cardboard box, laptop shell)
  - (b) Dictionary/response heatmaps per material
  - (c) Cross-material RMSE comparison (OMP vs physics-aware AI)
- **Asset:** `paper/figures/fig06_cross-material-universality.jpg`
- **Source:** Per-object calibrate-and-retrain experiments (Methods §Cross-material)
- **Note:** This figure was previously numbered Fig 7 in an intermediate code branch
  (`paper/nature-comm-figures`) due to the Fig 5/6 split. The manuscript consolidated
  back to Fig 1–6, making the "Fig 7" entry obsolete.

---

### Supp Fig 9 — Confusion Matrix Heatmaps

- **Status:** DONE
- **Panels:** (a) Confusion matrices (Baseline vs No-Transformer); (b–c) Per-angle distribution comparisons
- **Generator:** `figures/generators/fig09_confusion_matrix.py`
- **Data:** Baseline + No-Transformer `metrics.npz` (via `figures/conf/paths.yaml`)
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Output:** `figures/output/fig09_*.{pdf,tiff}`
- **Note:** Originally numbered Supp Fig 8; renumbered to 9 after the Fig 5/6 split
  (`23def1b`), then kept at 9 when the manuscript consolidated.

---

## Numbering History

The figure numbering changed once during development:

```
Original (be2e8dc)          After Fig 5/6 split (ea89f4c)    Manuscript final
─────────────────           ──────────────────────────        ─────────────────
Fig 1  Setup                Fig 1  Setup                     Fig 1  Setup
Fig 2  SVD                  Fig 2  SVD                       Fig 2  SVD
Fig 3  Architecture         Fig 3  Architecture              Fig 3  Architecture
Fig 4  Ablation             Fig 4  Ablation                  Fig 4  Ablation
Fig 5  Structure+Routing    Fig 5  Structure Alignment       Fig 5  Routing Analysis
       (one mega-figure)    Fig 6  Routing Mechanism                (recombined)
Fig 6  Cross-material       Fig 7  Cross-material (TODO)     Fig 6  Cross-material
Supp Fig 8  Confusion       Supp Fig 9  Confusion            Supp Fig 9  Confusion
```

The intermediate "Fig 7 BLOCKED" in code branches was an artifact of the split.
The manuscript never used Fig 7 — cross-material went directly to Fig 6.

---

## Build Pipeline

```bash
# Generate all automated figures (Fig 2, 4, 5 panels, Supp 9)
make -C figures generate

# Validate compliance (dimensions, fonts, DPI)
make -C figures validate

# Deploy automated assets into paper/figures/
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
