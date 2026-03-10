# Figure Registry — Nature Communications Paper

> Master index mapping manuscript figures to experiment sources, scripts,
> and reproduction commands.

## Overview

| Figure | Description | Status | Generator |
|--------|-------------|--------|-----------|
| Fig 1 | Setup photo + spectral fingerprint schematic | MANUAL | N/A |
| Fig 2 | SVD spectrum + modal decomposition + dictionary | DONE | `figures/generators/fig02_svd_spectrum.py` |
| Fig 3 | Unrolled network architecture diagram | MANUAL | N/A |
| Fig 4 | SNR robustness + component ablation | DONE | `figures/generators/fig04_snr_ablation.py` |
| Fig 5 | Structure alignment (correlation + selection) | DONE | `figures/generators/fig05_structure_alignment.py` |
| Fig 6 | Micro-level routing (heatmaps + band decomp) | DONE | `figures/generators/fig06_routing_mechanism.py` |
| Fig 7 | Cross-material generalization | **BLOCKED** | — |
| Supp Fig 9 | Confusion matrix heatmaps | DONE | `figures/generators/fig09_confusion_matrix.py` |

---

## Figure Details

### Fig 1 — Experimental Setup + Spectral Fingerprint

- **Status:** MANUAL
- **Panels:** (a) Photograph of LDV measurement setup; (b) Spectral fingerprint schematic
- **Source:** Photograph + illustration (no code pipeline)

---

### Fig 2 — SVD Spectrum + Modal Decomposition + Dictionary

- **Status:** DONE
- **Panels:** (a) SVD singular-value spectrum; (b) Modal decomposition polar plot; (c) Dictionary heatmap
- **Generator:** `figures/generators/fig02_svd_spectrum.py`
- **Data:** `h_matrix_normalized_original_to_box.pth` (configured in `figures/conf/paths.yaml`)
- **Experiment branch:** `feature/omp-transformer-modal-viz` @ `15b2981`
- **Output:** `figures/output/fig02_*.{pdf,tiff}`
- **Reproduce:**
  ```bash
  make -C figures generate  # or: python -m figures.build.pipeline generate
  ```

---

### Fig 3 — Unrolled Network Architecture

- **Status:** MANUAL
- **Panels:** Full-page architecture diagram of the unrolled OMP-Transformer
- **Source:** Illustration (no code pipeline)

---

### Fig 4 — Component Ablation + SNR Robustness

- **Status:** DONE
- **Panels:** (a) Component ablation strip chart (SNR=Inf slice); (b) Multi-variant SNR degradation curves (mean +/- std)
- **Generator:** `figures/generators/fig04_snr_ablation.py`
- **Data:** Babble Speech260 full sweep — `exp-omp-ablation-snr-rerun-20260128` @ `14feb94`
  - 245 runs: 7 variants x 7 SNR levels x 5 seeds
  - Aggregated: `results/figure4_data.json`
- **Output:** `figures/output/fig04_snr_ablation.{pdf,tiff}`
- **Reproduce:**
  ```bash
  make -C figures generate
  ```

---

### Fig 5 — Structure Alignment (Correlation + Selection Probability)

- **Status:** DONE
- **Panels:**
  - (a) Global correlation: 2 stacked 37x37 heatmaps (H matrix + QK learned structure)
  - (b) Selection probability: 2 stacked 37x37 heatmaps (OMP + QK)
- **Generator:** `figures/generators/fig05_structure_alignment.py`
- **Data:** `results/omp_transformer_speech260_trainval_split_full_20251115_082341/`
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Output:** `figures/output/fig05_structure_alignment.{pdf,tiff}`
- **Reproduce:**
  ```bash
  make -C figures generate
  ```

---

### Fig 6 — Micro-level Routing Mechanism

- **Status:** DONE
- **Panels:**
  - (a) 2x2 freq x atom heatmaps (Physics true/wrong, QK true/wrong)
  - (b) 5 band line plots (full band + 4 sub-bands: OMP vs QK vs True DoA)
- **Generator:** `figures/generators/fig06_routing_mechanism.py`
- **Data:** `results/omp_transformer_speech260_trainval_split_full_20251115_082341/`
- **Output:** `figures/output/fig06_routing_mechanism.{pdf,tiff}`
- **Reproduce:**
  ```bash
  make -C figures generate
  ```

---

### Fig 7 — Cross-Material Generalization

- **Status:** **BLOCKED** (CRITICAL GAP)
- **Panels:** Transfer-learning accuracy across materials
- **Blocking issue:** Need LDV measurements on 4 additional materials
- **Generator:** Not yet written

---

### Supp Fig 9 — Confusion Matrix Heatmaps

- **Status:** DONE
- **Panels:** Confusion matrices for key ablation configurations
- **Generator:** `figures/generators/fig09_confusion_matrix.py`
- **Data:** Baseline + No-Transformer metrics.npz (configured in `figures/conf/paths.yaml`)
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Output:** `figures/output/fig09_*.{pdf,tiff}`
- **Reproduce:**
  ```bash
  make -C figures generate
  ```

---

## Build Pipeline

```bash
# Generate all figures
make -C figures generate

# Validate compliance (dimensions, fonts, DPI)
make -C figures validate

# Deploy to paper/figures/ (requires validation pass)
make -C figures deploy

# Full pipeline
make -C figures all
```

## Identified Gaps

1. **Fig 7 (CRITICAL):** Cross-material experiments not yet conducted. Need LDV measurements on 4 additional materials.
