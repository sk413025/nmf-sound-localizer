# Fig. 4-6 Panel Naming Provenance

This note records the active naming and provenance contract for the model-bearing
panels in Figs. 4-6. It is intended to support generator-facing naming
decisions, not to replace manuscript prose.

## Naming Layers

- Artifact keys are the experiment- or results-facing identifiers stored in run
  artifacts and aggregation files.
- Display roles are the paper-facing labels used in figure panels, legends, and
  manifests.
- The active crosswalk from artifact keys or derived comparison roles to display
  roles lives in `figures/conf/model_display_crosswalk.yaml`.

## Fig. 4

- Generator: `figures/generators/fig04_solver_dynamics.py`
- Repo-reachable generator lineage:
  - `6c9b8fe` `figures: enrich generators to 32 panels across 6 figures`
  - `64d832c` `figures: fix visual quality across all generators`
  - `8dce255` `figures: swap Fig 3 (solver) ↔ Fig 4 (discriminability) for assertion-first narrative`
- Upstream artifacts:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz`
    - code-state git head: `3785b1f`
  - `results/figure4_data.json`
    - aggregated clean-condition ablation sweep
    - `figures/conf/experiments.yaml` records experiment commit `14feb94`
- Panel notes:
  - `a`: manual architecture asset, no run-variant label logic
  - `b`: convergence panel, no model-comparison labels
  - `c`: clean-condition ablation panel; artifact keys are `Baseline`,
    `No Type Bias`, `No Transformer`, `Fixed Heuristic`, `G-Fixed`,
    `G-Teacher`, `Dense Routing`
  - `d`: per-angle accuracy profile, no model-comparison labels

## Fig. 5

- Generator: `figures/generators/fig05_performance_structure.py`
- Repo-reachable generator lineage:
  - `db55586` `figures: rename fig04/fig05 generators and add JPG assets for DOCX embedding`
  - `6c9b8fe` `figures: enrich generators to 32 panels across 6 figures`
  - `64d832c` `figures: fix visual quality across all generators`
  - `0cd3667` `figures: add layout spec, fix fig05 height violation, enforce min font size`
  - `b6069f2` `fig05: change panel d from side-by-side to stacked layout`
- Upstream artifacts:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
    - code-state git head: `3785b1f`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
    - code-state git head: `3785b1f`
  - `results/fig05_panel_c_g_routing_seed42_clean/metrics.npz`
    - external sweep lineage: `14feb94` (`ablate_speech260_g_routing_snrInf_seed42_ep20_lr1e-3_babble_speech260_full_20260209_011327`)
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz`
    - code-state git head: `3785b1f`
  - `results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz`
    - code-state git head: `34403c7`
- External reference commits noted in active evidence docs:
  - `88a8940` confusion/ablation analysis family
  - `15b2981` modal visualization family
- Panel notes:
  - `a`: binds artifact keys `Baseline`, `No Transformer`, `Fixed Heuristic`
  - `b`: compares the physical matrix and learned QK structure as the physical-interpretability anchor
  - `c`: compares the clean seed42 `g_routing` confusion matrix against the guided-solver confusion matrix as the classical-reference context panel
  - `d/e/f`: form a matched ablation block comparing the full solver metrics run against the router-bypass ablation run

## Fig. 6

- Generator: `figures/generators/fig06_universality.py`
- Repo-reachable generator lineage:
  - `6c9b8fe` `figures: enrich generators to 32 panels across 6 figures`
  - `64d832c` `figures: fix visual quality across all generators`
- Upstream artifacts:
  - `h_matrix_normalized_original_to_box.pth`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
    - code-state git head: `3785b1f`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
    - code-state git head: `3785b1f`
- Panel notes:
  - `a/b`: top cross-material support assets remain a partial provenance gap
  - `c`: committed split-panel support asset without a tracked generator; the legend text is normalized directly at the PNG layer using the Fig. 6 crosswalk bindings (`OMP baseline` / `guided solver`)
  - `d`: SVD spectra panel has no model-comparison labels
  - `e`: compares derived OMP-like bandwise routing against the full solver's learned routing
