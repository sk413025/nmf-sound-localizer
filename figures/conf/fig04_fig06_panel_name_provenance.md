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
  - `results/fig05_panel_c_g_routing_seed42_clean/metrics.npz`
    - external sweep lineage: `14feb94` (`ablate_speech260_g_routing_snrInf_seed42_ep20_lr1e-3_babble_speech260_full_20260209_011327`)
  - `results/fig05_dense_routing_repro34403c7_clean/metrics.npz`
    - external sweep lineage: `34403c7` (`snrInf_dense_routing_repro34403c7_20260119_080918`)
  - `results/fig05_panel_f_no_type_bias_clean_seed_means/summary.npz`
    - derived from the clean 5-seed sweep families behind `results/figure4_data.json`, with the guided family sourced from the `No Type Bias` sweep
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz`
    - code-state git head: `3785b1f`
  - `results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz`
    - code-state git head: `34403c7`
- External reference commits noted in active evidence docs:
  - `88a8940` confusion/ablation analysis family
  - `15b2981` modal visualization family
- Panel notes:
  - `a`: binds artifact keys `No Type Bias`, `No Transformer`, `Fixed Heuristic`, `Dense Routing` to the four active paper-facing decoder families
  - `b`: compares five-seed clean mean per-angle accuracy across the guided solver, router-bypass, OMP baseline, and dense routing; the guided curve comes from the `No Type Bias` sweep family rather than the retired `Baseline` sweep key
  - `c`: reduces the confusion-family block to the chapter-defining locality comparison: clean seed42 `g_routing` (OMP baseline) versus the guided solver
  - `d`: isolates the measured physical local-structure matrix as its own panel
  - `e`: isolates the guided neighborhood-emphasis map on the same angle frame and correlation scale as panel `d`
  - `f`: closes the chapter with quantitative structure alignment: normalized local-band profile overlay plus concordance scatter derived from the same measured and learned matrices
- Guided-family alignment note:
  - The guided confusion map inside `panel c` and the guided correlation map inside `panel e` both resolve through `paths.yaml[confusion_matrix.baseline]`.
  - That path currently points to `results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz`.
  - The `20251202` metrics file is byte-identical to `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz` (`sha256=12085706b58eaa346f2d6de9681d986f6e8e02eefb429cc30299abe578db5d5b`), so `panel c/e` already use the same representative guided-solver outputs as the active primary run.
  - The practical consequence is that the Fig. 5 guided-family unification only required source-family changes in the aggregate benchmark panels:
    `panel a` switched from the retired `Baseline` sweep key to `No Type Bias`, and `panel b` now uses `results/fig05_panel_f_no_type_bias_clean_seed_means/summary.npz` for the guided clean-accuracy curve.
  - No separate confusion-artifact swap was needed for the guided entries in `panel c` or `panel e`; those panels were already on the intended `no_type_bias=true` guided family.

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
