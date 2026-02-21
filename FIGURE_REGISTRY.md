# Figure Registry — Nature Communications Paper

> Master index mapping manuscript figures to experiment sources, scripts,
> and reproduction commands.

## Overview

| Figure | Description | Status | Script |
|--------|-------------|--------|--------|
| Fig 1 | Setup photo + spectral fingerprint schematic | MANUAL | N/A |
| Fig 2 | SVD spectrum + modal decomposition + dictionary | DONE | `scripts/generate_fig2.py` |
| Fig 3 | Unrolled network architecture diagram | MANUAL | N/A |
| Fig 4 | SNR robustness + component ablation | DONE | `scripts/generate_fig4.py` |
| Fig 5 | Attention map + routing mechanism | DONE | `scripts/generate_fig5_*.py` |
| Fig 6 | Cross-material generalization | **TODO** | — |
| Supp Fig 8 | Confusion matrix heatmaps | DONE | `scripts/generate_fig8.py` |

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
- **Script:** `scripts/generate_fig2.py`
- **Experiment branch:** `feature/omp-transformer-modal-viz` @ `15b2981`
- **Data path:** SVD/modal artifacts generated during training
- **Reproduce:**
  ```bash
  python scripts/generate_fig2.py
  ```

---

### Fig 3 — Unrolled Network Architecture

- **Status:** MANUAL
- **Panels:** Full-page architecture diagram of the unrolled OMP-Transformer
- **Source:** Illustration (no code pipeline)

---

### Fig 4 — SNR Robustness + Component Ablation

- **Status:** DONE (data complete, script migrated)
- **Panels:** (a) MAE vs SNR curves (babble + white noise); (b) Component ablation bar chart
- **Script:** `scripts/generate_fig4.py` (plotter), `scripts/gather_fig4_data.py` (data aggregation)
- **Experiment sources:**
  - **Babble noise sweep:** `exp-omp-ablation-snr-rerun-20260128` @ `14feb94`
    - Data: 245-run CSV (5 architectures x 7 SNRs x 7 seeds)
  - **White noise sweep:** `exp-omp-ablation-snr-rerun-20260128` @ `8ff03fa`
    - Data: mean/std CSV (5-seed aggregate)
  - **Clean ablation:** `exp/omp-ablation-20251209` @ `88a8940`
    - Data: 5-fold cross-validation results
- **Reproduce:**
  ```bash
  python scripts/gather_fig4_data.py   # aggregate experiment CSVs
  python scripts/generate_fig4.py       # generate figure
  ```
- **Note:** Current script uses hardcoded paths via `figure4_data.json`. Refactoring to use relative paths is planned.

---

### Fig 5 — Attention Map + Routing Mechanism

- **Status:** DONE
- **Panels:** (a) Global attention heatmap; (b) Band selection routing; (c) Band-wise decomposition
- **Scripts:**
  - `scripts/generate_fig5_atomic.py` — Band-wise panels (v10)
  - `scripts/generate_fig5_master.py` — Global attention + selection overview
  - `scripts/fig5_b3_band_decomposition.py` — B3 band decomposition helper
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Reproduce:**
  ```bash
  python scripts/generate_fig5_atomic.py
  python scripts/generate_fig5_master.py
  ```

---

### Fig 6 — Cross-Material Generalization

- **Status:** **TODO** (CRITICAL GAP)
- **Panels:** Transfer-learning accuracy across materials
- **Experiment source:** No experiment data exists yet
- **Blocking issue:** Need LDV measurements on 4 additional materials
- **Script:** Not yet written

---

### Supp Fig 8 — Confusion Matrix Heatmaps

- **Status:** DONE
- **Panels:** Confusion matrices for key ablation configurations
- **Script:** `scripts/generate_fig8.py`
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Reproduce:**
  ```bash
  python scripts/generate_fig8.py
  ```

---

## Identified Gaps

1. **Fig 6 (CRITICAL):** Cross-material experiments not yet conducted. Need LDV measurements on 4 additional materials.
2. **Fig 4 script:** Uses hardcoded paths via `figure4_data.json`. Will migrate as-is; refactoring to relative paths planned.
3. **Fig 5:** Split across 3 scripts. Panel ownership documented above.
