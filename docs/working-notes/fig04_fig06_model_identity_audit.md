# Fig. 4-6 Model Identity Audit

This note audits the active model-bearing panels in Figs. 4-6 and aligns:

- manuscript-facing names
- in-panel display labels
- artifact keys or derived panel roles
- upstream run lineage
- panel-level performance or comparison quantities

It is an implementation note for later naming cleanup. It is not a canonical
governance contract.

The active manuscript-facing naming authority remains:

- `paper/manuscript/FIGURE_NAMING_CONTRACT.md`
- `paper/figures/Figure-Legends.md`

The active generator-facing naming authority remains:

- `figures/conf/model_display_crosswalk.yaml`

Visual acceptance of the current manuscript-facing figures was validated
previously at tag `validated/paper-fig1-6-coldstart-20260323`. This audit
focuses on code/result identity and naming consistency.

## Source Stack Used For This Audit

- `paper/manuscript/FIGURE_NAMING_CONTRACT.md`
- `paper/figures/Figure-Legends.md`
- `figures/conf/model_display_crosswalk.yaml`
- `figures/conf/fig04_fig06_panel_name_provenance.md`
- `figures/conf/experiments.yaml`
- `figures/generators/fig04_solver_dynamics.py`
- `figures/generators/fig05_performance_structure.py`
- `figures/generators/fig06_universality.py`
- `results/figure4_data.json`
- `results/omp_transformer_speech260_trainval_split_full_20251115_082341/metrics.npz`
- `results/omp_transformer_speech260_trainval_split_full_20251115_082341/modal_routing_val.npz`
- `results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz`
- `results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz`

## Canonical Identity Classes

| Canonical identity | Manuscript-facing default | Panel short label | What it actually is | Backing lineage |
|---|---|---|---|---|
| `full_solver` | `physics-guided neural solver` then `physics-aware solver` | `guided solver` | The full learned method | Primary run `3785b1f` via `scripts/omp-transformer-ldv.py` |
| `no_transformer_ablation` | `router-bypass ablation` | `router-bypass` | The learned model with transformer routing removed | Ablation run `34403c7` in Fig. 5; aggregated sweep in Fig. 4 |
| `analytical_omp_baseline` | `analytical OMP baseline` | `OMP baseline` | The classical sparse baseline | Panel-specific binding in Figs. 5-6 |

These are the only three model-level identities that should remain paper-wide.

## Entities That Are Not Primary Model Names

| Current label family | Correct category | Notes |
|---|---|---|
| `No Type Bias` | ablation | A component-removal ablation inside the learned solver family |
| `Fixed Heuristic` in Fig. 4c | ablation | Not the same semantic role as `Fixed Heuristic` in Fig. 5a |
| `G-Fixed` | ablation/variant | Keep local to Fig. 4 only if still scientifically needed |
| `G-Teacher` | ablation/variant | Keep local to Fig. 4 only if still scientifically needed |
| `Dense Routing` | ablation | Figure-only ablation label, not a paper-wide model family |
| `learned router`, `QK learned structure` | submodule / learned structure | Not a separate model |
| `H physical structure` | physical structure | Not a model |
| `OMP selection reference`, `omp_side`, `omp_bandwise` | derived comparison role | A panel-level comparison role, not a separate trained system |

## Panel-By-Panel Identity Audit

| Figure / panel | Current panel-facing label(s) | Artifact key or derived role | Canonical identity | Panel quantity from active artifacts | Audit judgment | Recommended display family |
|---|---|---|---|---|---|---|
| Fig. 4c | `Baseline` | `Baseline` | `full_solver` | Clean mean `0.8625` | Reasonable mapping to the full learned solver | `guided solver` in-panel; `full physics-aware solver` in panel/manifest text |
| Fig. 4c | `No Transformer` | `No Transformer` | `no_transformer_ablation` | Clean mean `0.5820` | Reasonable mapping | `router-bypass` |
| Fig. 4c | `No Type Bias` | `No Type Bias` | `no_type_bias_ablation` | Clean mean `0.9751` | Panel-specific ablation; should not be promoted to a paper-wide model name | `no type bias` only within Fig. 4 |
| Fig. 4c | `Fixed Heuristic` | `Fixed Heuristic` | `fixed_heuristic_ablation` | Clean mean `0.4402` | Semantically distinct from Fig. 5a `Fixed Heuristic`; keep panel-specific if retained | `fixed heuristic` only within Fig. 4 |
| Fig. 4c | `G-Fixed` | `G-Fixed` | `g_fixed_variant` | Clean mean `0.4402` | High-risk naming point: same current aggregate as `Fixed Heuristic` and `G-Teacher` | `G-fixed` only if the variant truly needs to stay visible |
| Fig. 4c | `G-Teacher` | `G-Teacher` | `g_teacher_variant` | Clean mean `0.4402` | High-risk naming point: same current aggregate as `Fixed Heuristic` and `G-Fixed` | `G-teacher` only if the variant truly needs to stay visible |
| Fig. 4c | `Dense Routing` | `Dense Routing` | `dense_routing_ablation` | Clean mean `0.0270` | Clear local ablation, not a paper-wide model name | `dense routing` only within Fig. 4 |
| Fig. 5a | `Baseline` | `Baseline` | `full_solver` | Clean `0.8625`; `20 dB=0.7858`; `0 dB=0.1235` | Reasonable; same model as Fig. 4c `Baseline` | `guided solver` in-panel; `physics-aware solver` in text |
| Fig. 5a | `No Transformer` | `No Transformer` | `no_transformer_ablation` | Clean `0.5820`; `20 dB=0.4789`; `0 dB=0.0638` | Reasonable; same ablation as Fig. 4c `No Transformer` | `router-bypass` |
| Fig. 5a | `Fixed Heuristic` | `Fixed Heuristic` | `analytical_omp_baseline` | Clean `0.4402`; `20 dB=0.3490`; `0 dB=0.0440` | Reasonable only because the crosswalk binds this panel specifically to OMP | `OMP baseline` in-panel; `analytical OMP baseline` in text |
| Fig. 5b | `H physical structure` / `QK learned structure` | `physical_matrix` / `learned_matrix` | physical structure / learned-router structure | Global structure correlation `r=0.4678` | Not model names; keep out of the main model dictionary | `H physical structure`, `QK learned structure` |
| Fig. 5c | `OMP selection` / `Solver selection` | `omp_side` / `learned_side` | `analytical_omp_baseline` / `full_solver` | Diagonal mean: OMP `0.0172`, solver `0.9459`; off-diagonal mass: OMP `0.9828`, solver `0.0541` | Reasonable; this panel compares a derived OMP reference against the full solver | `OMP baseline` vs `guided solver` |
| Fig. 5d | `Solver` / `No-transformer` | `baseline_cm` / `no_transformer_cm` | `full_solver` / `no_transformer_ablation` | Mean per-angle accuracy: solver `0.9459`, no-transformer `0.6310` | Reasonable | `guided solver` vs `router-bypass` |
| Fig. 5e | `Solver` / `No-transformer` | `baseline_dist` / `no_transformer_dist` | `full_solver` / `no_transformer_ablation` | Target-angle diagonal mass at `55°`: solver `0.5577`, no-transformer `0.4038`; at `100°`: solver `0.6538`, no-transformer `0.5192` | Reasonable; panel compares routing sharpness, not separate model families | `guided solver` vs `router-bypass` |
| Fig. 5f | `Solver` / `No-transformer` | `baseline_line` / `no_transformer_line` | `full_solver` / `no_transformer_ablation` | Improved angles `37/37`; mean improvement `0.3150` | Reasonable | `guided solver` vs `router-bypass` |
| Fig. 6c | `OMP` / `solver` | `omp_boxplot` / `solver_boxplot` | `analytical_omp_baseline` / `full_solver` | Cross-material RMSE comparison; exact boxplot source is not tracked in a generator-backed artifact | Identity mapping is reasonable, but the numeric lineage remains a provenance gap | `OMP baseline` vs `guided solver` |
| Fig. 6e | `OMP` / `solver` | `omp_bandwise` / `learned_bandwise` | `analytical_omp_baseline` / `full_solver` | Full band: OMP `0.0172`, solver `0.3051`; `300-500`: `0.0208` vs `0.2765`; `500-1k`: `0.0431` vs `0.2672`; `1k-2k`: `0.0198` vs `0.4111`; `2k-3k`: `0.0270` vs `0.2287` | Reasonable | `OMP baseline` vs `guided solver` |

## Main Same-Identity Equivalence Classes

### Same full learned model

These all refer to the same full learned method:

- Fig. 4c `Baseline`
- Fig. 5a `Baseline`
- Fig. 5c `learned_side`
- Fig. 5d `baseline_cm`
- Fig. 5e `baseline_dist`
- Fig. 5f `baseline_line`
- Fig. 6c `solver_boxplot`
- Fig. 6e `learned_bandwise`

Recommended paper-wide treatment:

- first mention in prose: `physics-guided neural solver`
- short prose form: `physics-aware solver`
- in-panel short label: `guided solver`

### Same router-bypass ablation

These all refer to the same ablation family:

- Fig. 4c `No Transformer`
- Fig. 5a `No Transformer`
- Fig. 5d `no_transformer_cm`
- Fig. 5e `no_transformer_dist`
- Fig. 5f `no_transformer_line`

Recommended paper-wide treatment:

- prose and legends: `router-bypass ablation`
- in-panel short label: `router-bypass`

### Same analytical OMP baseline

These all refer to the same classical baseline role:

- Fig. 5a `Fixed Heuristic`
- Fig. 5c `omp_side`
- Fig. 6c `omp_boxplot`
- Fig. 6e `omp_bandwise`

Recommended paper-wide treatment:

- prose and legends: `analytical OMP baseline`
- in-panel short label: `OMP baseline`

## High-Risk Naming Points

### 1. `Fixed Heuristic` is not globally stable

`Fixed Heuristic` does not mean the same thing everywhere:

- Fig. 4c binds it to `fixed_heuristic_ablation`
- Fig. 5a binds it to `analytical_omp_baseline`

This must not be normalized with a global text-replace. It requires explicit
panel-aware rewriting.

### 2. Fig. 4c currently over-distinguishes three numerically identical variants

The active aggregated backing data gives the same clean-condition mean for:

- `Fixed Heuristic`
- `G-Fixed`
- `G-Teacher`

All three are `0.4402` in `results/figure4_data.json`. Under the current active
artifact, treating them as three empirically distinct outcomes is difficult to
justify without an additional explanation or a data-source correction.

### 3. Structure panels should not be treated as peer model names

The following should never be elevated to the same naming layer as `guided solver`,
`router-bypass`, or `OMP baseline`:

- `H physical structure`
- `QK learned structure`
- `learned router`

They describe structure or submodule behavior, not separate methods.

### 4. Fig. 6c still has a numeric provenance gap

Fig. 6c currently uses a committed support PNG rather than a tracked generator
with recoverable boxplot data. The identity binding is still reasonable, but
later manuscript cleanup should keep this panel marked as a provenance gap until
the exact numeric source is restored or re-derived.

## Recommended Rename Dictionary For Later Implementation

### Keep globally

- `physics-guided neural solver`
- `physics-aware solver`
- `analytical OMP baseline`
- `router-bypass ablation`

### Keep only as Fig. 4 local ablation labels

- `no-type-bias ablation`
- `fixed-heuristic routing ablation`
- `G-fixed routing variant`
- `G-teacher routing variant`
- `dense-routing ablation`

### Keep only as submodule / structure labels

- `learned router`
- `QK learned structure`
- `H physical structure`

### Remove from default manuscript-facing naming

- `Physics-aware AI`
- bare `Baseline`
- `baseline model`
- `No trans.`

## Bottom-Line Judgment

The active crosswalk is largely defensible for the three paper-wide identities:

- physics-guided neural solver / physics-aware solver
- router-bypass ablation
- analytical OMP baseline

The main remaining naming problem is not the guided-solver / OMP-baseline / router-bypass triad.
It is the Fig. 4 ablation family, especially the unstable role of
`Fixed Heuristic` and the current numerical collapse of `Fixed Heuristic`,
`G-Fixed`, and `G-Teacher`.
