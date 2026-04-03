# Active Figure Naming Contract

This document defines the active manuscript-facing figure contract for the current six-figure Nature Communications draft.

Canonical sources:

- `paper/manuscript/manuscript.md`
- `paper/figures/Figure-Legends.md`
- `figures/conf/experiments.yaml` for the active machine-readable figure and panel contract
- `figures/conf/model_display_crosswalk.yaml` for generator-facing display labels in Figs. 4-6
- `figures/conf/fig04_fig06_panel_name_provenance.md` for panel-level code, data, and commit backtraces in Figs. 4-6
- `figures/conf/layout_spec.md` for intended panel sizes in mm
- `paper/figures/*.layout.json` for realized geometry checks

Treat `paper/figures/` as the final manuscript-facing figure surface only.
Committed support panels belong under `figures/output/*_panels/`.

The retired nine-figure main-paper contract is historical only. Do not use it as the naming authority for the current manuscript.

## Active Main-Paper Figures

| Fig. | Manuscript asset | Canonical title | Panels | Layout authority |
|------|------------------|-----------------|--------|------------------|
| 1 | `paper/figures/fig01_paradigm-shift.jpg` | Direction-dependent structural filtering revealed by single-point laser vibrometry | a-e | `figures/conf/layout_spec.md` |
| 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | Calibration fingerprints occupy a compact angle-ordered space | a-f | `figures/conf/layout_spec.md` |
| 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | Directional structure persists under speech, but the first correlation-based choice becomes unstable | a-f | `figures/conf/layout_spec.md` |
| 4 | `paper/figures/fig04_solver-dynamics.jpg` | A physics-guided solver concentrates nearby overlap into the correct neighborhood | a-d | `figures/conf/layout_spec.md` |
| 5 | `paper/figures/fig05_performance-structure.jpg` | Prediction structure stays organized around the measured local band | a-e | `figures/conf/layout_spec.md` |
| 6 | `paper/figures/fig06_universality.jpg` | Matched calibration recurs across a bounded five-object screen | a-e | `figures/conf/layout_spec.md` |

## Canonical Model And Method Terms

| Canonical term | Prose short form | Shared in-panel short label | Use for | Do not use as the default name |
|----------------|------------------|-----------------------------|---------|--------------------------------|
| guided solver | guided solver | guided solver | the active learned decoder family on Figs. 4-5; a physics-guided deep-unrolled residual-correction solver with learned local gating, backed by `no_type_bias=true` runs and the `No Type Bias` clean sweep key | physics-aware solver, full physics-aware solver, baseline model, bare `No Type Bias` |
| OMP baseline | OMP baseline | OMP baseline | the collapsed paper-facing soft-OMP baseline family on active Figs. 4-5; not exact hard OMP by default | fixed heuristic, G-fixed, G-teacher, bare OMP when the paper-facing family is meant |
| router-bypass ablation | router-bypass ablation | router-bypass | the ablation that bypasses the transformer router | no-transformer, identity model |
| dense-routing ablation | dense routing | dense routing | the ablation that disables sparse routing and spreads weight across all experts | dense model, uniform router |
| learned router | learned router | QK learned structure | the routing submodule inside the guided solver | solver, model, attention router as a peer-method name |
| angle-specific conditional output distributions | conditional output distributions | conditional outputs | Fig. 5d style mechanistic panels | routing distributions, mechanism figure, angle-specific mechanism as a whole-figure name |

## Naming Rules

- Use `guided solver`, `OMP baseline`, `router-bypass`, and `dense routing` as the only active paper-facing decoder-family names on Figs. 4-5.
- The figure-facing guided-solver family on Figs. 4-5 is backed by `no_type_bias=true` run artifacts and by the `No Type Bias` clean sweep key in `results/figure4_data.json`; do not display `No Type Bias` as a separate model family.
- The active Figs. 4-5 `OMP baseline` family collapses the upstream artifact keys `Fixed Heuristic`, `G-Fixed`, and `G-Teacher`; keep those names only in provenance or historical metadata.
- Treat the active Figs. 4-5 `OMP baseline` label as a collapsed soft-OMP family name, not as a default claim of exact hard OMP. Name exact hard OMP explicitly only when that narrower algorithm is the actual surface under discussion.
- Treat Fig. 3 as a stage-0 correlation-based first-choice diagnostic surface, not as direct evidence for the full hard-OMP recursion unless that narrower algorithm is explicitly reintroduced.
- Treat Fig. 4 panel `a` as schematic manuscript support, panels `b-c` as the data-backed mechanism surface, and panel `d` as a secondary clean-condition check.
- Use `router-bypass ablation` for the family previously referred to by the experiment-facing key `No Transformer`.
- Treat `figures/conf/model_display_crosswalk.yaml` as the active display-label authority for Fig. 4-6 generators and panel manifests. It maps provenance keys or derived comparison roles to paper-facing labels without renaming upstream artifacts.
- Treat `H physical structure`, `QK learned structure`, and OMP-selection references as structure or derived-comparison labels rather than peer model names.
- Treat `figures/conf/layout_spec.md` as the design authority for panel mm sizes; treat `paper/figures/*.layout.json` as realized-geometry evidence, not as the naming authority.
