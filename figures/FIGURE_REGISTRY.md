# Figure Registry — Active Six-Figure Manuscript

This file is a navigation summary for the active six main-paper figures used in
`paper/manuscript/manuscript.md`. It is not the executable figure contract.

Primary authorities:

- manuscript-facing semantics: `paper/manuscript/manuscript.md`
- canonical figure naming: `paper/manuscript/FIGURE_NAMING_CONTRACT.md`
- machine-readable figure/panel contract: `figures/conf/experiments.yaml`
- intended panel-mm layout: `figures/conf/layout_spec.md`
- realized layout evidence: `paper/figures/*.layout.json`
- review workflow configuration: `figures/conf/review_targets.yaml`

`paper/figures/` is a final-assets-only manuscript surface. Mixed-figure support
panels live under `figures/output/*_panels/`.

Retired material:

- the earlier nine-figure main-paper contract is historical only
- use `docs/archive/figures/README.md` and git history when that legacy state is needed

## Overview

| Figure | Manuscript asset | Generator / composer | Panel mode | Contract anchor |
|--------|------------------|----------------------|------------|-----------------|
| Fig 1 | `paper/figures/fig01_paradigm-shift.jpg` | `fig01_paradigm_data.py` + `compose_master_figure3_family.py` | mixed | `experiments.yaml → fig01_manual` |
| Fig 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | `fig02_svd_spectrum.py` | data-backed | `experiments.yaml → fig02_svd` |
| Fig 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | `fig03_fingerprint_discriminability.py` | data-backed | `experiments.yaml → fig03_discriminability` |
| Fig 4 | `paper/figures/fig04_solver-dynamics.jpg` | `fig04_solver_dynamics.py` + `compose_master_figure3_family.py` | mixed | `experiments.yaml → fig04_ablation` |
| Fig 5 | `paper/figures/fig05_performance-structure.jpg` | `fig05_performance_structure.py` | data-backed | `experiments.yaml → fig05_performance_structure` |
| Fig 6 | `paper/figures/fig06_universality.jpg` | `fig06_universality.py` + `compose_master_figure3_family.py` | mixed | `experiments.yaml → fig06_universality` |

## Figure Details

### Fig 1 — Direction-Dependent Structural Filtering

- Manuscript asset: `paper/figures/fig01_paradigm-shift.jpg`
- Generator / composer: `figures/generators/fig01_paradigm_data.py` + `scripts/paper/compose_master_figure3_family.py`
- Active panel manifest: `figures/output/fig01_paradigm_shift_panels/fig01_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig01_manual`

### Fig 2 — Structured Physical Dictionary

- Manuscript asset: `paper/figures/fig02_svd-physical-dictionary.jpg`
- Generator: `figures/generators/fig02_svd_spectrum.py`
- Active panel manifest: `figures/output/fig02_svd_spectrum_panels/fig02_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig02_svd`

### Fig 3 — Discriminability And Classical OMP Failure

- Manuscript asset: `paper/figures/fig03_fingerprint-discriminability.jpg`
- Generator: `figures/generators/fig03_fingerprint_discriminability.py`
- Active panel manifest: `figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig03_discriminability`

### Fig 4 — Physics-Guided Solver Mechanism

- Manuscript asset: `paper/figures/fig04_solver-dynamics.jpg`
- Generator / composer: `figures/generators/fig04_solver_dynamics.py` + `scripts/paper/compose_master_figure3_family.py`
- Manual support panel `a`: `figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_a_architecture.jpg`
- Active panel manifest: `figures/output/fig04_solver_dynamics_manuscript_panels/fig04_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig04_ablation`
- Active panel logic: architecture + routing formation + gated-update/residual-correction closure + explicit mode-to-angle aggregation bridge + clean routing-mechanism ablation

### Fig 5 — Performance And Structure

- Manuscript asset: `paper/figures/fig05_performance-structure.jpg`
- Generator: `figures/generators/fig05_performance_structure.py`
- Active panel manifest: `figures/output/fig05_performance_structure_panels/fig05_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig05_performance_structure`

### Fig 6 — Universality

- Manuscript asset: `paper/figures/fig06_universality.jpg`
- Generator / composer: `figures/generators/fig06_universality.py` + `scripts/paper/compose_master_figure3_family.py`
- Manual support panel `a`: `figures/output/fig06_cross_material_universality_panels/fig06_panel_a_material_exemplars.png`
- Active panel manifest: `figures/output/fig06_universality_manuscript_panels/fig06_panel_manifest.json`
- Contract entry: `figures/conf/experiments.yaml → fig06_universality`
- Active panel logic: material exemplars + cross-material H breadth + low-rank continuity + compact screening-consequence closure + per-material frequency-structure cards
