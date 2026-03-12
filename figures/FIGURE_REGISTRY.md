# Figure Registry — Nature Communications Paper

> Master index mapping manuscript figures to experiment sources, generators,
> and reproduction commands. Numbering follows the manuscript (Figs. 1–6 + Supp Fig. 9).
>
> Asset model used in this branch:
> 1. **Evidence source**: data/results/checkpoints that support the scientific claim
> 2. **Generator output**: clean plotted asset produced from code, when applicable
> 3. **Manuscript asset**: final paper-facing composite stored under `paper/figures/`
>
> Codex review bundles judge the **manuscript asset**, but data-backed figures must
> carry their upstream evidence source and generator output in the bundle context.

## Overview

| Figure | Description | Asset model | Upstream contract |
|--------|-------------|-------------|-------------------|
| Fig 1 | Setup photo + spectral fingerprint schematic | Manual manuscript asset | No generator expected |
| Fig 2 | SVD spectrum + modal decomposition + dictionary | Data-backed manuscript composite | `fig02_svd_spectrum.py` + `h_matrix` |
| Fig 3 | Unrolled network architecture diagram | Manual manuscript asset | No generator expected |
| Fig 4 | Noise robustness + component ablation | Data-backed manuscript composite | `fig04_snr_ablation.py` + `figure4_data.json` |
| Fig 5 | Routing mechanism analysis (structure + micro + macro) | Data-backed manuscript composite | `fig05_structure_alignment.py` + `fig06_routing_mechanism.py` |
| Fig 6 | Cross-material universality | Manuscript composite with provenance gap | Machine-readable upstream evidence not yet registered |
| Supp Fig 9 | Confusion matrix heatmaps | Data-backed deployed generator output | `fig09_confusion_matrix.py` + `metrics.npz` |

---

## Figure Details

### Fig 1 — Experimental Setup + Spectral Fingerprint

- **Asset model:** Manual manuscript asset
- **Panels:** (a) Photograph of LDV measurement setup; (b) Spectral fingerprint schematic
- **Evidence source:** Photograph + illustration
- **Generator output:** None expected
- **Manuscript asset:** `paper/figures/fig01_paradigm-shift.jpg`

---

### Fig 2 — SVD Spectrum + Modal Decomposition + Dictionary

- **Asset model:** Data-backed manuscript composite
- **Panels:** (a) Singular-value spectrum; (b) Modal decomposition (freq + polar); (c–e) Dictionary heatmaps
- **Evidence source:** `h_matrix_normalized_original_to_box.pth` (via `figures/conf/paths.yaml`)
- **Generator:** `figures/generators/fig02_svd_spectrum.py`
- **Experiment branch:** `feature/omp-transformer-modal-viz` @ `15b2981`
- **Generator output:** `figures/output/fig02_svd_spectrum.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig02_svd-physical-dictionary.jpg`
- **Review rule:** Codex reviews the manuscript composite, but it must remain visually and scientifically faithful to the upstream generator output.

---

### Fig 3 — Unrolled Network Architecture

- **Asset model:** Manual manuscript asset
- **Panels:** Full-page architecture diagram of the unrolled OMP-Transformer
- **Evidence source:** Architecture illustration
- **Generator output:** None expected
- **Manuscript asset:** `paper/figures/fig03_unrolled-attention-omp.jpg`

---

### Fig 4 — Noise Robustness + Component Ablation

- **Asset model:** Data-backed manuscript composite
- **Panels:** (a) Ablation strip chart (SNR=Inf slice); (b) Multi-variant SNR degradation curves
- **Evidence source:** `results/figure4_data.json`
- **Generator:** `figures/generators/fig04_snr_ablation.py`
- **Data:** Babble Speech260 full sweep — `exp-omp-ablation-snr-rerun-20260128` @ `14feb94`
  - 245 runs: 7 variants x 7 SNR levels x 5 seeds
  - Aggregated: `results/figure4_data.json`
- **Generator output:** `figures/output/fig04_snr_ablation.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig04_noise-robustness-ablation.jpg`

---

### Fig 5 — Routing Mechanism Analysis

Manuscript title: "Deciphering model behaviour across scales: attention structure, micro-mechanism and macro-robustness"

- **Asset model:** Data-backed manuscript composite
- **Panels:**
  - (a) Global self-attention map — physics-consistent near-diagonal correlation structure
  - (b) Micro-level case study — OMP vs physics-aware selection heatmaps + band-wise line plots
  - (c) Selection-probability statistics — OMP off-diagonal vs QK sharp diagonal
- **Evidence source:** `results/omp_transformer_speech260_trainval_split_full_20251115_082341/`
- **Generators:**
  - `figures/generators/fig05_structure_alignment.py` → panels (a) and (c)
  - `figures/generators/fig06_routing_mechanism.py` → panel (b)
- **Experiment branch:** `feature/master-figure-nature-comm` @ `97942ac`
- **Generator output:**
  - `figures/output/fig05_structure_alignment.{pdf,tiff}`
  - `figures/output/fig06_routing_mechanism.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig05_routing-mechanism-analysis.png`
- **Historical alias:** Earlier branches and slide-style composites referred to this figure family as `Master Figure 3` ("Deciphering AI Understanding via Micro-Mechanism & Macro-Robustness"). That historical `panel b` is the predecessor of manuscript Fig. 5b, not a separate manuscript figure.
- **Panel-b provenance note:** The older `Master Figure 3 / panel b` went through at least two code-backed forms before the current manuscript assembly:
  - `scripts/create_master_figure.py` and the pre-registry master-figure workflow rendered an early micro-mechanism case study panel.
  - `generate_figure5_atomic*.py` produced intermediate `B`-series assets such as `Fig5_B3_POLAR_ESTIMATION` and later band-wise line plots.
  - The current manuscript-facing upstream source is `figures/generators/fig06_routing_mechanism.py`, whose deployed output is `figures/output/fig06_routing_mechanism.{pdf,tiff}`.
- **Panel-b asset caveat:** Unlike panels (a) and (c), panel (b) is currently registered via the full upstream generator output `figures/output/fig06_routing_mechanism.{pdf,tiff}` rather than a dedicated split asset under `fig05_routing_mechanism_analysis_panels/`. This is intentional in the current registry, but it means the registry tracks panel-b provenance at the generator-output level, not as an isolated panel PDF.
- **Note:** The two generators produce separate PDFs. The final manuscript figure was composed from these panels, so review must check both composition quality and fidelity to the upstream data-backed outputs.

---

### Fig 6 — Cross-Material Universality

- **Asset model:** Manuscript composite with provenance gap
- **Panels:**
  - (a) Five target objects (acrylic plate, paper cup, wooden board, cardboard box, laptop shell)
  - (b) Dictionary/response heatmaps per material
  - (c) Cross-material RMSE comparison (OMP vs physics-aware AI)
- **Evidence source:** Per-object calibrate-and-retrain experiments (Methods §Cross-material)
- **Generator output:** Not yet registered in machine-readable form
- **Manuscript asset:** `paper/figures/fig06_cross-material-universality.jpg`
- **Release note:** Fig. 6 should not be treated as a purely manual figure. It currently has a manuscript asset but lacks a registered upstream evidence/generator contract, which must be fixed before release.
- **Note:** This figure was previously numbered Fig 7 in an intermediate code branch
  (`paper/nature-comm-figures`) due to the Fig 5/6 split. The manuscript consolidated
  back to Fig 1–6, making the "Fig 7" entry obsolete.

---

### Supp Fig 9 — Confusion Matrix Heatmaps

- **Asset model:** Data-backed deployed generator output
- **Panels:** (a) Confusion matrices (Baseline vs No-Transformer); (b–c) Per-angle distribution comparisons
- **Evidence source:** Baseline + No-Transformer `metrics.npz` (via `figures/conf/paths.yaml`)
- **Generator:** `figures/generators/fig09_confusion_matrix.py`
- **Experiment branch:** `exp/omp-ablation-20251209` @ `88a8940`
- **Generator output:** `figures/output/fig09_*.{pdf,tiff}`
- **Manuscript asset:** `paper/figures/fig09_*.pdf` (deployed generator outputs)
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
