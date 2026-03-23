# Active Figure Naming Contract

This document defines the active manuscript-facing figure contract for the current six-figure Nature Communications draft.

Canonical sources:

- `paper/manuscript/manuscript.md`
- `paper/figures/Figure-Legends.md`
- `figures/conf/layout_spec.md` for intended panel sizes in mm
- `figures/output/*.layout.json` for realized geometry checks

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

| Canonical term | Allowed short form | Use for | Do not use as the default name |
|----------------|--------------------|---------|--------------------------------|
| physics-guided neural solver | physics-aware solver | the full learned method | physics-aware AI, physics-aware model, baseline model |
| analytical OMP baseline | OMP baseline | the classical sparse baseline | fixed heuristic, baseline, analytical sparse-recovery baseline when OMP is meant |
| no-transformer ablation | no-transformer | the ablation that bypasses the transformer router | identity model |
| type-bias component | type bias | the bias term in the full solver | type-bias model |
| learned router | attention router | the routing submodule inside the full solver | solver, model |
| angle-specific routing distributions | routing distributions | Fig. 5e style mechanistic panels | mechanism figure, angle-specific mechanism as a whole-figure name |

## Naming Rules

- Use `physics-guided neural solver` at first mention in Results and Methods, then use `physics-aware solver`.
- Use `analytical OMP baseline` when contrasting the learned method with the classical solver.
- Use `full physics-aware solver` when a panel compares the full method with ablations.
- Keep experiment-internal labels such as `G-fixed` and `G-teacher` inside ablation metadata only unless the manuscript explicitly needs them.
- Treat `figures/conf/layout_spec.md` as the design authority for panel mm sizes; treat `figures/output/*.layout.json` as realized-geometry evidence, not as the naming authority.
