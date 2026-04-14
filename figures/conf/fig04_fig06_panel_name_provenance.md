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
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/model_best.pth`
    - code-state git head: `3785b1f`
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/code_state.json`
    - active architecture / routing arguments for the primary guided solver
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/preprocessing.pth`
    - active physical dictionary used by the guided residual updates
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
    - code-state git head: `3785b1f`
  - `results/figure4_data.json`
    - aggregated clean-condition ablation sweep
    - `figures/conf/experiments.yaml` records experiment commit `14feb94`
  - `results/fig04_stepwise_mechanics.npz`
    - governed derived artifact rebuilt from the active primary run
- Panel notes:
  - `a`: generator-produced architecture/physics correspondence panel; it pairs
    the measured local band in `H` with the staged broad-match, local-gate,
    and local-update profiles on the same angle frame
  - `b`: representative broad-match panel; it shows the active 70° exemplar
    before local pooling, with the nearby co-active atoms displayed explicitly
    inside the measured local neighborhood
  - `c`: local-gate convergence panel; it overlays the broad match, local gate,
    and localized update for the same 70° exemplar so the delayed-commitment
    narrowing is visible on one shared axis
  - `d`: residual-purification panel; left = exemplar residual-norm drop across
    the guided steps, right = validation-wide local-mass concentration before
    versus after the guided local step
  - `e`: clean decoder-comparison panel; the upstream sweep still stores the
    artifact keys `Baseline`, `No Type Bias`, `No Transformer`,
    `Fixed Heuristic`, `G-Fixed`, `G-Teacher`, and `Dense Routing`, but the
    active paper-facing panel collapses them to four families:
    `No Type Bias -> guided solver`, `No Transformer -> router-bypass`,
    `Fixed Heuristic/G-Fixed/G-Teacher -> paper-facing OMP baseline`, and
    `Dense Routing -> dense routing`

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
  - `/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/worktrees/exp-omp-ablation-snr-rerun-20260128/results`
    - governed babble sweep root aggregated directly by the active generator for family-backed panels `b`, `c`, `d`, `f`, and `g`
- External reference commits noted in active evidence docs:
  - `88a8940` confusion/ablation analysis family
  - `15b2981` modal visualization family
- Panel notes:
  - `a`: keeps the guided stage-0 support, first-step contraction, and final guided prediction on one shared radius axis
  - `b`: aggregates the coherent clean family mass-within-radius curves directly from the governed babble sweep families
  - `c`: adds the exact-versus-local family bridge, using the same coherent clean family bundle as panel `b`
  - `d`: expands final prediction locality into a four-family confusion grid so the accepted hierarchy is visible as morphology rather than only as summary statistics
  - `e`: isolates the measured physical local-structure matrix as its own panel
  - `f`: compares each family's final clean confusion against the same measured neighborhood surface using a local-band agreement score, with global matrix correlation retained only as a secondary reference
  - `g`: keeps only the noisy babble sweep as the final consequence panel; the clean consequence burden is carried earlier by panels `b-c`

## Fig. 6

- Generator: `figures/generators/fig06_universality.py`
- Repo-reachable generator lineage:
  - `6c9b8fe` `figures: enrich generators to 32 panels across 6 figures`
  - `64d832c` `figures: fix visual quality across all generators`
- Upstream artifacts:
  - `../ldv-master-reference-audio/results/h_matrix_repro_original_to_cross_materials_20260324_220841/reproduction_report.json`
  - `../ldv-master-reference-audio/results/h_matrix_repro_original_to_cross_materials_20260324_220841/h_matrix_normalized_original_to_{a,p,w,b,m}.pth`
  - `../ldv-master-reference-audio/results/cross_materials_material_selection_20260324_221916/comparison_report.json`
  - `../ldv-master-reference-audio/results/cross_materials_material_selection_20260324_221916/selection_summary.json`
  - `../ldv-master-reference-audio/results/cross_materials_fig06_support_20260325_164051/support_report.json`
  - `../ldv-master-reference-audio/results/cross_materials_fig06_support_20260325_164051/material_performance_summary.csv`
  - `../ldv-master-reference-audio/results/cross_materials_fig06_support_20260325_164051/low_rank_summary.csv`
  - `../ldv-master-reference-audio/results/cross_materials_fig06_support_20260325_164051/per_angle_metrics.csv`
- Panel notes:
  - `a`: retained manual support exemplar strip cropped from the committed legacy support image
  - `b`: isolates the five `Original -> Material` H matrices so breadth of structured angle-frequency encoding reads as its own panel
  - `c`: carries the Fig. 2-aligned centered-magnitude cumulative-energy curves and rank90/rank95 summary so low-rank continuity remains explicit as a separate panel
  - `d`: combines the normalized energy-versus-Top-1 comparison with Top-1 and MAE summaries so the figure closes on an object-level readout-versus-energy comparison
  - `e`: turns each material into its own three-level card by pairing the normalized spectral envelope, the angular-contrast curve, and the representative band-limited directional code selected from the committed `H` matrices
