# Data Provenance — Active Six-Figure Summary

This document is a human-readable summary of the current provenance posture for
the active six main-paper figures. It is not the executable source of truth.

Primary machine-readable authority:

- `figures/conf/experiments.yaml`

Derived evidence artifacts:

- `figures/output/*_panel_manifest.json`
- `paper/figures/*.layout.json`

Manuscript surface rule:

- `paper/figures/` contains only final manuscript assets, layout sidecars, and figure-facing docs
- committed manual support panels live under `figures/output/*_panels/`

Workflow-only config:

- `figures/conf/review_targets.yaml`

Audit entrypoints:

- `python scripts/paper/verify_provenance.py`
- `python scripts/paper/check_figure_regression.py --baseline-ref <git-ref>`

## Figure-Level Summary

| Figure | Generator / composer | Evidence backbone | Current posture |
|--------|----------------------|-------------------|-----------------|
| Fig. 1 | `fig01_paradigm_data.py` + `compose_master_figure3_family.py` | `dictionary.npz` + `figures/conf/paths.yaml` | Mixed figure with explicit manual support panels `a-b` |
| Fig. 2 | `fig02_svd_spectrum.py` | `h_matrix_normalized_original_to_box.pth` | Fully data-backed |
| Fig. 3 | `fig03_fingerprint_discriminability.py` | `dictionary.npz` + `modal_routing_val.npz` + SNR sweep inputs | Fully data-backed, with some inputs resolved through `paths.yaml` |
| Fig. 4 | `fig04_solver_dynamics.py` + `compose_master_figure3_family.py` | `metrics.npz` + `results/figure4_data.json` | Mixed figure with explicit manual architecture panel `a` under `figures/output/fig04_solver_dynamics_manuscript_panels/` |
| Fig. 5 | `fig05_performance_structure.py` | routing + dictionary + confusion metrics | Fully data-backed |
| Fig. 6 | `fig06_universality.py` + `compose_master_figure3_family.py` | cross-material `Original -> Material` H bundle + material-selection bundle | Mixed figure with manual support panel `a` and data-backed panels `b-d` |

## Panel-Level Provenance Classes

The active branch uses three panel-level provenance classes:

- `data_backed`: panel is traced to committed run/data artifacts and a generator-backed output
- `manual_support`: panel is a committed support asset and intentionally not tied to a plotting run
- `provenance_gap`: panel is tracked and managed, but the upstream machine-readable evidence contract is still incomplete

These classes are defined and checked from `figures/conf/experiments.yaml`.
Panel manifests are treated as derived evidence that should agree with that
contract, not as peer-level authority.

## Current Non-Blocking Warnings

- Commit `15b2981` is not reachable from this repo; active Fig. 2 data artifacts remain intact.
- Commit `88a8940` is not reachable from this repo; related supplementary-era lineage remains non-blocking.
- Fig. 3 still resolves some upstream inputs indirectly through `figures/conf/paths.yaml`.

## Interpretation Rule

Do not treat "managed" as equivalent to "fully data-backed."

- Figs. 2, 3, and 5 are intended to be fully data-backed.
- Figs. 1 and 4 are mixed figures with explicit manual support panels.
- Fig. 6 now uses a mixed contract: panel `a` remains manual support, while panels `b-d` are data-backed against the committed cross-material H-reproduction and screening bundles.
