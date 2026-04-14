# Active Figure Naming Contract

This document defines the active paper-facing figure contract for the current six-figure Nature Communications draft.

Canonical sources:

- `paper/manuscript/manuscript.md`
- `paper/figures/Figure-Legends.md`
- `figures/conf/experiments.yaml` for the active machine-readable figure and panel contract
- `figures/conf/model_display_crosswalk.yaml` for generator-facing display labels in Figs. 4-6
- `figures/conf/fig04_fig06_panel_name_provenance.md` for panel-level code, data, and commit backtraces in Figs. 4-6
- `figures/conf/layout_spec.md` for intended panel sizes in mm
- `paper/figures/*.layout.json` for realized geometry checks

Treat `paper/figures/` as the final paper-facing figure surface and the literal manuscript-facing delivery surface only.
Committed support panels belong under `figures/output/*_panels/`.

The retired nine-figure main-paper contract is historical only. Do not use it as the naming authority for the current manuscript.

## Active Main-Paper Figures

| Fig. | Paper-facing asset | Canonical title | Panels | Layout authority |
|------|------------------|-----------------|--------|------------------|
| 1 | `paper/figures/fig01_paradigm-shift.jpg` | Direction-dependent structural filtering revealed by single-point laser vibrometry | a-e | `figures/conf/layout_spec.md` |
| 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | Calibration fingerprints occupy a compact angle-ordered space | a-f | `figures/conf/layout_spec.md` |
| 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | Speech preserves the local code while broadening nearby-angle overlap | a-f | `figures/conf/layout_spec.md` |
| 4 | `paper/figures/fig04_solver-dynamics.jpg` | Preserving the broadened neighborhood keeps subtraction admissible | a-e | `figures/conf/layout_spec.md` |
| 5 | `paper/figures/fig05_performance-structure.jpg` | Final prediction succeeds by preserving the measured neighborhood | a-g | `figures/conf/layout_spec.md` |
| 6 | `paper/figures/fig06_universality.jpg` | Matched calibration extends object-conditioned directional readout beyond the acrylic reference object | a-e | `figures/conf/layout_spec.md` |

## Canonical Model And Method Terms

| Canonical term | Prose short form | Shared in-panel short label | Use for | Do not use as the default name |
|----------------|------------------|-----------------------------|---------|--------------------------------|
| guided solver | guided solver | guided solver | the active learned decoder family on Figs. 4-5; a physics-guided residual-correction readout with learned local gating, backed by `no_type_bias=true` runs and the `No Type Bias` clean sweep key | physics-aware solver, full physics-aware solver, baseline model, bare `No Type Bias` |
| OMP baseline | OMP baseline | OMP baseline | the collapsed paper-facing soft-OMP baseline family on active Figs. 4-5; not exact hard OMP by default | fixed heuristic, G-fixed, G-teacher, bare OMP when the paper-facing family is meant |
| router-bypass ablation | router-bypass ablation | router-bypass | the ablation that bypasses the transformer router | no-transformer, identity model |
| dense-routing ablation | dense routing | dense routing | the ablation that disables sparse routing and spreads weight across all experts | dense model, uniform router |
| learned router | learned router | QK learned structure | the routing submodule inside the guided solver | solver, model, attention router as a peer-method name |
| family-to-measured neighborhood alignment | family-to-measured neighborhood alignment | family alignment | Fig. 5f style family-wide measured-neighborhood comparison panels | generic scorecard, guided-only mechanism figure |
| noise robustness consequence | noise robustness consequence | robustness summary | Fig. 5g style noisy family consequence panels | generic benchmark panel, clean consequence board |

## Naming Rules

- Use `guided solver`, `OMP baseline`, `router-bypass`, and `dense routing` as the only active paper-facing decoder-family names on Figs. 4-5.
- The figure-facing guided-solver family on Figs. 4-5 is backed by `no_type_bias=true` run artifacts and the corresponding clean/sweep evaluation summaries; do not display `No Type Bias` as a separate model family.
- The active Figs. 4-5 `OMP baseline` family collapses the upstream artifact keys `Fixed Heuristic`, `G-Fixed`, and `G-Teacher`; keep those names only in provenance or historical metadata.
- Treat the active Figs. 4-5 `OMP baseline` label as a collapsed soft-OMP family name, not as a default claim of exact hard OMP. Name exact hard OMP explicitly only when that narrower algorithm is the actual surface under discussion.
- Treat Fig. 3 as the speech-side continuation of Fig. 2: compactness, broadened local neighborhood, neighborhood coherence, and stage-0 local separability on the frozen grouped-match surface.
- Treat Fig. 4 as a neighborhood-preserving admissibility chapter: panel `a` anchors the measured neighborhood, panels `b-c` show representative broad-match and contraction, and panels `d-e` give validation-wide neighborhood sharpening summaries.
- Treat Fig. 5 as the structure-first bridge from admissibility to consequence: panels `a-c` keep the shared neighborhood logic visible through stage-0 support, first-step contraction, coherent family locality, and exact-versus-local clean consequence; panel `d` shows that hierarchy directly as four-family final-prediction locality; panel `e` anchors the measured neighborhood surface; panel `f` compares each family back to that measured neighborhood; panel `g` is the downstream noisy consequence summary.
- Use `router-bypass ablation` for the family previously referred to by the experiment-facing key `No Transformer`.
- Treat `figures/conf/model_display_crosswalk.yaml` as the active display-label authority for Fig. 4-6 generators and panel manifests. It maps provenance keys or derived comparison roles to paper-facing labels without renaming upstream artifacts.
- Treat `H physical structure`, `QK learned structure`, and OMP-selection references as structure or derived-comparison labels rather than peer model names.
- Treat `figures/conf/layout_spec.md` as the design authority for panel mm sizes; treat `paper/figures/*.layout.json` as realized-geometry evidence, not as the naming authority.
