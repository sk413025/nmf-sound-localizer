# Data Provenance — Active Six-Figure Summary

This document summarizes the current provenance posture for the active six
main-paper figures.

Canonical machine-readable sources:

- `figures/conf/review_targets.yaml`
- `figures/conf/experiments.yaml`
- `figures/conf/panel_assets.yaml`
- `figures/output/*_panel_manifest.json`

Audit entrypoint:

- `python scripts/paper/verify_provenance.py`

## Figure-Level Summary

| Figure | Generator | Evidence backbone | Panel status | Current posture |
|--------|-----------|-------------------|--------------|-----------------|
| Fig. 1 | `figures/generators/fig01_paradigm_data.py` | `dictionary.npz` + `figures/conf/paths.yaml` | `a-b` manual support, `c-e` data-backed | Mixed but managed |
| Fig. 2 | `figures/generators/fig02_svd_spectrum.py` | `h_matrix_normalized_original_to_box.pth` | all panels data-backed | Closed except historical branch-tip warning |
| Fig. 3 | `figures/generators/fig03_fingerprint_discriminability.py` | `dictionary.npz` + `modal_routing_val.npz` + SNR sweep inputs | all panels data-backed | Closed, with some generator inputs resolved through `paths.yaml` |
| Fig. 4 | `figures/generators/fig04_solver_dynamics.py` | `metrics.npz` + `results/figure4_data.json` | `a` manual support, `b-d` data-backed | Mixed but managed |
| Fig. 5 | `figures/generators/fig05_performance_structure.py` | routing + dictionary + confusion metrics | all panels data-backed | Closed after harmonizing the baseline metrics source to `20251202_192153` |
| Fig. 6 | `figures/generators/fig06_universality.py` | `h_matrix` + routing + dictionary | `a-c` provenance gap, `d-e` data-backed | Mixed with explicit partial gap |

## Panel-Level Status

The active branch uses three panel-level provenance classes:

- `data_backed`: panel is traced to committed run/data artifacts and a generator-backed output
- `manual_support`: panel is a committed support asset, intentionally not tied to a plotting run
- `provenance_gap`: panel is tracked and managed, but the upstream machine-readable evidence contract is still incomplete

Canonical status locations:

- `figures/output/*_panel_manifest.json` for split-panel asset status
- `figures/conf/experiments.yaml` for mixed-figure panel provenance notes

### Mixed Figures

#### Fig. 1

- `a`: manual support asset
- `b`: manual support asset
- `c`: data-backed
- `d`: data-backed
- `e`: data-backed

#### Fig. 4

- `a`: manual support asset
- `b`: data-backed
- `c`: data-backed
- `d`: data-backed

#### Fig. 6

- `a`: provenance gap
- `b`: provenance gap
- `c`: provenance gap
- `d`: data-backed
- `e`: data-backed

## Current Non-Blocking Warnings

- Commit `15b2981` is not reachable from this repo; active Fig. 2 data artifacts remain intact.
- Commit `88a8940` is not reachable from this repo; related supplementary-era lineage remains non-blocking.
- Fig. 3 still resolves some upstream inputs indirectly through `figures/conf/paths.yaml`.

## Interpretation Rule

Do not treat “managed” as equivalent to “fully data-backed.”

- Figs. 2, 3, and 5 are intended to be fully data-backed.
- Figs. 1 and 4 are mixed figures with explicit manual support panels.
- Fig. 6 remains an honest partial-gap figure until the top-row cross-material support assets gain a stronger machine-readable upstream contract.
