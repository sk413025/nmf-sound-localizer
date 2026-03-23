# Figure Registry — Active Six-Figure Manuscript

This registry tracks only the active six main-paper figures used in `paper/manuscript/manuscript.md`.

Retired material:

- the earlier nine-figure main-paper contract is historical only
- use `docs/archive/figures/README.md` and git history when that legacy state is needed

Asset model used in this branch:

1. Evidence source: data, results, or manual support assets that support the figure claim
2. Generator output: clean plotted asset produced from code, when applicable
3. Split panel assets: internal top-level manuscript panels stored under `figures/output/*_panels/`
4. Manuscript asset: final paper-facing composite stored under `paper/figures/`

Panel-mm design authority: `figures/conf/layout_spec.md`
Realized geometry checks: `figures/output/*.layout.json`

## Overview

| Figure | Manuscript asset | Description | Asset model | Upstream contract |
|--------|------------------|-------------|-------------|-------------------|
| Fig 1 | `paper/figures/fig01_paradigm-shift.jpg` | Setup photo + physical principle + spectral evidence | Mixed manuscript composite | `fig01_paradigm_data.py` + manual panels |
| Fig 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | SVD spectrum + modal decomposition + structured dictionary | Data-backed manuscript composite | `fig02_svd_spectrum.py` + `h_matrix` |
| Fig 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | Discriminability, OMP failure, and dose response | Data-backed manuscript composite | `fig03_fingerprint_discriminability.py` + primary run + white-noise data |
| Fig 4 | `paper/figures/fig04_solver-dynamics.jpg` | Architecture + convergence + clean ablation + per-angle accuracy | Mixed manuscript composite | `fig04_solver_dynamics.py` + manual architecture panel |
| Fig 5 | `paper/figures/fig05_performance-structure.jpg` | Noise robustness, structure alignment, and routing diagnostics | Data-backed manuscript composite | `fig05_performance_structure.py` + routing run |
| Fig 6 | `paper/figures/fig06_universality.jpg` | Cross-material universality with per-band physics evidence | Mixed manuscript composite with partial provenance gap | `fig06_universality.py` + cross-material top panels |

## Figure Details

### Fig 1 — Direction-Dependent Structural Filtering

- Asset model: mixed manuscript composite
- Manuscript asset: `paper/figures/fig01_paradigm-shift.jpg`
- Split panel assets: `figures/output/fig01_paradigm_shift_panels/`
- Panel manifest: `figures/output/fig01_paradigm_shift_panels/fig01_panel_manifest.json`
- Generator output: `figures/output/fig01_paradigm_data.{pdf,tiff}`
- Evidence sources:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
  - `figures/conf/paths.yaml`
- Panel status: `a-b` manual support, `c-e` data-backed
- Provenance note: panels a-b are manual support assets; panels c-e are generator-backed evidence panels. Machine-readable panel status lives in `figures/conf/experiments.yaml`.

### Fig 2 — Structured Physical Dictionary

- Asset model: data-backed manuscript composite
- Manuscript asset: `paper/figures/fig02_svd-physical-dictionary.jpg`
- Split panel assets: `figures/output/fig02_svd_spectrum_panels/`
- Panel manifest: `figures/output/fig02_svd_spectrum_panels/fig02_panel_manifest.json`
- Generator: `figures/generators/fig02_svd_spectrum.py`
- Generator output: `figures/output/fig02_svd_spectrum.{pdf,tiff}`
- Evidence sources:
  - `h_matrix_normalized_original_to_box.pth`

### Fig 3 — Discriminability And Classical OMP Failure

- Asset model: data-backed manuscript composite
- Manuscript asset: `paper/figures/fig03_fingerprint-discriminability.jpg`
- Split panel assets: `figures/output/fig03_fingerprint_discriminability_panels/`
- Panel manifest: `figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_manifest.json`
- Generator: `figures/generators/fig03_fingerprint_discriminability.py`
- Generator output: `figures/output/fig03_fingerprint_discriminability.{pdf,tiff}`
- Evidence sources:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/speech_babble_omp_snr_sweep.json`
  - `figures/conf/paths.yaml`
- Provenance note: fully data-backed figure; some upstream generator inputs are resolved indirectly through `figures/conf/paths.yaml`.

### Fig 4 — Solver Dynamics

- Asset model: mixed manuscript composite
- Manuscript asset: `paper/figures/fig04_solver-dynamics.jpg`
- Manual support asset: `paper/figures/fig04_unrolled-attention-omp.jpg`
- Split panel assets: `figures/output/fig04_solver_dynamics_manuscript_panels/`
- Panel manifest: `figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_manifest.json`
- Generator: `figures/generators/fig04_solver_dynamics.py`
- Generator output: `figures/output/fig04_solver_dynamics.{pdf,tiff}`
- Evidence sources:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz`
  - `results/figure4_data.json`
- Panel status: `a` manual support, `b-d` data-backed
- Provenance note: panel a is a manual architecture asset; panels b-d are generator-backed evidence panels. Machine-readable panel status lives in `figures/conf/experiments.yaml`.

### Fig 5 — Performance And Structure

- Asset model: data-backed manuscript composite
- Manuscript asset: `paper/figures/fig05_performance-structure.jpg`
- Split panel assets: `figures/output/fig05_performance_structure_panels/`
- Panel manifest: `figures/output/fig05_performance_structure_panels/fig05_panel_manifest.json`
- Generator: `figures/generators/fig05_performance_structure.py`
- Generator output: `figures/output/fig05_performance_structure.{pdf,tiff}`
- Evidence sources:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
  - `results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz`
  - `results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz`
- Provenance note: fully data-backed figure; panels d-f use the confusion-matrix metrics resolved through `figures/conf/paths.yaml`.

### Fig 6 — Universality

- Asset model: mixed manuscript composite with partial provenance gap
- Manuscript asset: `paper/figures/fig06_universality.jpg`
- Split panel assets: `figures/output/fig06_universality_manuscript_panels/`
- Panel manifest: `figures/output/fig06_universality_manuscript_panels/fig06_panel_manifest.json`
- Generator: `figures/generators/fig06_universality.py`
- Generator output: `figures/output/fig06_universality.{pdf,tiff}`
- Evidence sources:
  - `h_matrix_normalized_original_to_box.pth`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
- Panel status: `a-c` provenance gap, `d-e` data-backed
- Provenance note: panels d-e are generator-backed; top panels a-c are committed split-panel support assets whose upstream machine-readable cross-material evidence contract still needs strengthening.
