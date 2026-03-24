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
| 2 | `paper/figures/fig02_svd-physical-dictionary.jpg` | Physical encoding via spectral-spatial modes and construction of a structured dictionary | a-f | `figures/conf/layout_spec.md` |
| 3 | `paper/figures/fig03_fingerprint-discriminability.jpg` | Encoding survives content variation but classical decoding fails | a-f | `figures/conf/layout_spec.md` |
| 4 | `paper/figures/fig04_solver-dynamics.jpg` | Physics-guided deep unrolled network with attention-based gating | a-d | `figures/conf/layout_spec.md` |
| 5 | `paper/figures/fig05_performance-structure.jpg` | The learned router mirrors physical structure and maintains robust decoding under noise | a-f | `figures/conf/layout_spec.md` |
| 6 | `paper/figures/fig06_universality.jpg` | Universal physical encoding across diverse materials | a-e | `figures/conf/layout_spec.md` |

## Canonical Model And Method Terms

| Canonical term | Prose short form | Shared in-panel short label | Use for | Do not use as the default name |
|----------------|------------------|-----------------------------|---------|--------------------------------|
| physics-guided neural solver | physics-aware solver | guided solver | the full learned method | physics-aware AI, physics-aware model, baseline model, bare solver |
| analytical OMP baseline | OMP baseline | OMP baseline | the classical sparse baseline | fixed heuristic, baseline, analytical sparse-recovery baseline when OMP is meant, bare OMP |
| router-bypass ablation | router-bypass ablation | router-bypass | the ablation that bypasses the transformer router | no-transformer, identity model |
| type-bias component | type bias | no type bias | the bias term in the full solver | type-bias model |
| learned router | learned router | QK learned structure | the routing submodule inside the full solver | solver, model, attention router as a peer-method name |
| angle-specific conditional output distributions | conditional output distributions | conditional outputs | Fig. 5e style mechanistic panels | routing distributions, mechanism figure, angle-specific mechanism as a whole-figure name |

## Naming Rules

- Use `physics-guided neural solver` at first mention in Results and Methods, then use `physics-aware solver`.
- Use `analytical OMP baseline` in manuscript prose and legends; use `OMP baseline` when a compact but still scientific label is needed.
- Use `router-bypass ablation` for the family previously referred to by the experiment-facing key `No Transformer`.
- Use the shared in-panel compact labels `guided solver`, `OMP baseline`, and `router-bypass` across Figs. 4-6 wherever the same three model families appear.
- Use `full physics-aware solver` only when a panel or manifest must explicitly distinguish the complete method from local ablation variants.
- Keep experiment-internal labels such as `G-fixed` and `G-teacher` inside ablation metadata only unless the manuscript explicitly needs them.
- Treat `figures/conf/model_display_crosswalk.yaml` as the active display-label authority for Fig. 4-6 generators and panel manifests. It maps provenance keys or derived comparison roles to paper-facing labels without renaming upstream artifacts.
- Do not globally rewrite `Fixed Heuristic` into `analytical OMP baseline`; that equivalence is panel-specific and must be bound explicitly in the crosswalk.
- Treat `no-type-bias ablation`, `fixed-heuristic routing ablation`, `G-fixed routing variant`, `G-teacher routing variant`, and `dense-routing ablation` as Fig. 4 local ablation labels anchored to the guided-solver family rather than paper-wide model families.
- Treat `H physical structure`, `QK learned structure`, and OMP-selection references as structure or derived-comparison labels rather than peer model names.
- Treat `figures/conf/layout_spec.md` as the design authority for panel mm sizes; treat `paper/figures/*.layout.json` as realized-geometry evidence, not as the naming authority.
