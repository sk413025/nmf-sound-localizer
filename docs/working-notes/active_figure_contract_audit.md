# Active Figure Contract Audit

This note is an implementation audit for the active six-figure manuscript. It
is not a canonical governance file.

## Per-Figure Traceback

| Figure | Manuscript asset | Generator / composer | Direct runtime inputs | Semantic authority | Machine-readable authority | Layout / font owners |
|--------|------------------|----------------------|-----------------------|--------------------|----------------------------|----------------------|
| Fig. 1 | `paper/figures/fig01_paradigm-shift.jpg` | `fig01_paradigm_data.py` + `compose_master_figure3_family.py` | `figures/style.py`, `figures/conf/paths.yaml`, `dictionary.npz`, manual top panels | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig01_manual` | `layout_spec.md` + compose constants |
| Fig. 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | `fig02_svd_spectrum.py` | `figures/style.py`, `figures/conf/paths.yaml`, `h_matrix_normalized_original_to_box.pth` | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig02_svd` | `layout_spec.md` + generator constants |
| Fig. 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | `fig03_fingerprint_discriminability.py` | `figures/style.py`, `figures/conf/paths.yaml`, primary run NPZs, SNR sweep JSON | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig03_discriminability` | `layout_spec.md` + generator constants |
| Fig. 4 | `paper/figures/fig04_solver-dynamics.jpg` | `fig04_solver_dynamics.py` + `compose_master_figure3_family.py` | `figures/style.py`, `figures/conf/paths.yaml`, `model_display_crosswalk.yaml`, metrics NPZ, `figure4_data.json`, manual architecture panel | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig04_ablation` | `layout_spec.md` + generator constants + compose constants |
| Fig. 5 | `paper/figures/fig05_performance-structure.jpg` | `fig05_performance_structure.py` | `figures/style.py`, `figures/conf/paths.yaml`, `model_display_crosswalk.yaml`, routing NPZs, confusion metrics | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig05_performance_structure` | `layout_spec.md` + generator constants |
| Fig. 6 | `paper/figures/fig06_universality.jpg` | `fig06_universality.py` + `compose_master_figure3_family.py` | `figures/style.py`, `figures/conf/paths.yaml`, `model_display_crosswalk.yaml`, `h_matrix`, routing NPZs, committed top-row support panels | `manuscript.md`, `FIGURE_NAMING_CONTRACT.md`, `Figure-Legends.md` | `experiments.yaml → fig06_universality` | `layout_spec.md` + generator constants + compose constants |

## Consolidation Notes

- Semantic figure identity now belongs at the manuscript layer, not in review or provenance summaries.
- `figures/conf/experiments.yaml` is now the single machine-readable figure and panel contract for active Fig. 1-6.
- `figures/conf/review_targets.yaml` is review orchestration only.
- `figures/output/*_panel_manifest.json` and `paper/figures/*.layout.json` are derived evidence artifacts.
- Layout and font ownership are still partially duplicated between `layout_spec.md` and generator / compose constants; that is the remaining major consolidation gap.
