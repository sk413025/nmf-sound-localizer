# Fig. 4 Panel c Model-Semantics Audit

Date: 2026-03-25
Status: audit complete

## Scope

This note audits the active `Fig. 4c` clean-condition ablation panel from the
current Nature Communications worktree. The goal is to trace each displayed
model label through:

1. the manuscript-facing figure asset,
2. the active figure generator and display crosswalk,
3. the aggregated panel data file,
4. the sweep aggregation script,
5. the executed run artifacts, and
6. the historical runtime that produced those artifacts.

The audit is evidence-first. When a physical interpretation is inferred from
runtime structure rather than stated explicitly in a paper document, it is
marked as an inference.

## Key Judgment

- The active `Fig. 4c` panel is a historical seven-variant validation-sweep
  summary, not a comparison among seven stable paper-wide model identities.
- The panel is drawn from `results/figure4_data.json`, which is symlinked to a
  245-run babble Speech260 sweep under
  `worktrees/exp-omp-ablation-snr-rerun-20260128`.
- In that sweep, `No Type Bias` outperforms the displayed `Baseline`
  (`0.9751` vs `0.8625` clean mean), so the current artifact does not support a
  claim that the type-bias component is necessary for strong clean-condition
  performance.
- `Fixed Heuristic`, `G-Fixed`, and `G-Teacher` are numerically identical across
  all five clean seeds and all choose `best_epoch = 1`. Under the executed
  runtime, this is expected: all three use `routing_mode = g`, so expert
  selection is driven by the physical correlation term `g = D^T r` rather than
  the Transformer/QK pathway.
- `Dense Routing` is not merely a softer router. In the executed runtime it
  disables OMP sparsity and performs a dense update over all atoms. This is a
  qualitatively different physical assumption from sparse pursuit.
- `Fig. 4c` uses validation accuracy from a deterministic train/validation split
  (`clip_id % 5`), not held-out outer-fold test accuracy.

## Evidence Chain

### 1. Manuscript-facing asset

The active manuscript surface is:

- `paper/figures/fig04_solver-dynamics.jpg`

The panel was visually inspected directly. Panel `c` currently shows seven rows:

- `guided solver`
- `no type bias`
- `router-bypass`
- `fixed heuristic`
- `G-fixed`
- `G-teacher`
- `dense routing`

### 2. Active generator and crosswalk

The active generator is:

- `figures/generators/fig04_solver_dynamics.py`

For panel `c`, the generator:

- reads `results/figure4_data.json`,
- iterates over the fixed variant list
  `Baseline`, `No Type Bias`, `No Transformer`, `Fixed Heuristic`, `G-Fixed`,
  `G-Teacher`, `Dense Routing`,
- and converts those artifact keys into panel labels through
  `figures/conf/model_display_crosswalk.yaml`.

The active crosswalk binds:

- `Baseline` -> `full_solver` -> `guided solver`
- `No Transformer` -> `no_transformer_ablation` -> `router-bypass`
- `No Type Bias` -> `no_type_bias_ablation` -> `no type bias`
- `Fixed Heuristic` -> `fixed_heuristic_ablation` -> `fixed heuristic`
- `G-Fixed` -> `g_fixed_variant` -> `G-fixed`
- `G-Teacher` -> `g_teacher_variant` -> `G-teacher`
- `Dense Routing` -> `dense_routing_ablation` -> `dense routing`

### 3. Active panel data source

`figures/conf/experiments.yaml` declares that `Fig. 4c` uses:

- `results/figure4_data.json`

That file is a symlink to:

- `worktrees/exp-omp-ablation-snr-rerun-20260128/results/figure4_data.json`

The clean-condition means in the active aggregated file are:

| Artifact key | Clean mean |
| --- | ---: |
| Baseline | 0.8625 |
| No Transformer | 0.5820 |
| No Type Bias | 0.9751 |
| Fixed Heuristic | 0.4402 |
| G-Fixed | 0.4402 |
| G-Teacher | 0.4402 |
| Dense Routing | 0.0270 |

### 4. Aggregation script

The sweep aggregation script in the historical worktree is:

- `worktrees/exp-omp-ablation-snr-rerun-20260128/scripts/gather_fig4_data.py`

It maps variant codes to panel keys as follows:

- `baseline` -> `Baseline`
- `no_transformer` -> `No Transformer`
- `g_routing` -> `Fixed Heuristic`
- `no_type_bias` -> `No Type Bias`
- `disable_sparsity` -> `Dense Routing`
- `g_teacher` -> `G-Teacher`
- `g_fixed_F` -> `G-Fixed`

It scans the 245-run sweep:

- 7 variants x 7 SNR levels x 5 seeds

and stores:

- `ablation`: clean-only slice (`snr == Inf`)
- `snr`: all SNR levels

### 5. Executed runtime provenance

The panel `c` sweep artifacts were executed from historical commit:

- `61b327ad285bccaa82a93a37f6d5fb8962cd9bac`

The representative clean seed-42 run commands are:

- `Baseline`: `--routing_mode qk`
- `No Transformer`: `--routing_mode qk --encoder_identity`
- `No Type Bias`: `--routing_mode qk --no_type_bias`
- `Fixed Heuristic`: `--routing_mode g`
- `Dense Routing`: `--routing_mode qk --disable_omp_sparsity`
- `G-Teacher`: `--routing_mode g --encoder_identity --d_model 346`
- `G-Fixed`: `--routing_mode g --d_model 346`

The corresponding runtime file is not present in the worktree checkout anymore,
but it is recoverable through git from commit `61b327a...`.

### 6. Protocol provenance

In the executed `61b327a...` runtime, the panel sweep:

- creates a deterministic train/validation split via `clip_id % 5`,
- trains on `Y_train`,
- evaluates on `Y_val`,
- tracks `best_accuracy` on `Y_val`,
- and saves `confusion_matrix`, `per_angle_accuracy`, `predictions`, and
  `labels` for that validation subset.

Therefore the `Fig. 4c` scalar values are validation accuracies from that split,
not outer-fold held-out test metrics.

## Variant-By-Variant Semantics

### 1. `guided solver` (`Baseline`)

- Artifact key: `Baseline`
- Representative flags: `routing_mode=qk`, `encoder_identity=false`,
  `no_type_bias=false`, `disable_omp_sparsity=false`
- Clean aggregate: mean `0.8625`, all five clean seeds pick `best_epoch = 20`
- Algorithmic meaning:
  - uses QK-derived routing scores to choose which experts and atoms to update,
  - keeps sparse OMP-like selection,
  - keeps learnable type embeddings `type_R` and `type_D`
- Physical interpretation:
  - this is the most explicitly learned residual-to-dictionary routing condition
    in the panel `c` sweep,
  - the model can reshape selection based on learned token interactions before
    applying the physics-consistent residual update
- Important limit:
  - this panel-`c` baseline is not runtime-identical to the `no_type_bias=true`
    lineage used by other active figure artifacts in Fig. 4/5.

### 2. `router-bypass` (`No Transformer`)

- Artifact key: `No Transformer`
- Representative flags: `routing_mode=qk`, `encoder_identity=true`
- Clean aggregate: mean `0.5820`, all five clean seeds pick `best_epoch = 20`
- Algorithmic meaning:
  - token projections remain,
  - QK scores are still computed,
  - but the Transformer encoder is bypassed, so there is no learned contextual
    reshaping of token representations before QK scoring
- Physical interpretation:
  - the solver still uses a learned scoring interface, but no longer learns a
    manifold-aware routing transform over the residual-plus-dictionary tokens
- Audit judgment:
  - this is a genuine routing ablation with a clear mechanistic meaning.

### 3. `no type bias` (`No Type Bias`)

- Artifact key: `No Type Bias`
- Representative flags: `routing_mode=qk`, `no_type_bias=true`
- Clean aggregate: mean `0.9751`, all five clean seeds pick `best_epoch = 20`
- Algorithmic meaning:
  - removes `type_R` and `type_D` from token construction,
  - residual and dictionary tokens differ only through projected signal content,
    not through additive learnable token-type offsets
- Physical interpretation:
  - this removes a non-physical identity prior that tells the network which
    tokens are residual tokens and which are dictionary tokens
- Audit judgment:
  - the active artifact shows this variant outperforming `Baseline`, so the
    current panel does not support a claim that the type-bias component is
    required for strong clean-condition accuracy.

### 4. `fixed heuristic` (`Fixed Heuristic`)

- Artifact key: `Fixed Heuristic`
- Representative flags: `routing_mode=g`
- Clean aggregate: mean `0.4402`, all five clean seeds pick `best_epoch = 1`
- Algorithmic meaning:
  - expert and atom scores are taken directly from the physical correlation
    term `g = D^T r`,
  - the learned QK/Transformer pathway is still instantiated in code but is not
    used to select scores in this mode
- Physical interpretation:
  - this is physics-first heuristic routing rather than learned routing,
  - it behaves like a soft physics-guided selection rule anchored directly to
    dictionary correlation
- Audit judgment:
  - this is not the same semantic object as the panel-`a`/Fig. 5
    `OMP baseline` naming role, even though both are non-full-solver references.

### 5. `G-fixed` (`G-Fixed`)

- Artifact key: `G-Fixed`
- Representative flags: `routing_mode=g`, `d_model=346`,
  `encoder_identity=false`
- Clean aggregate: mean `0.4402`, all five clean seeds pick `best_epoch = 1`
- Algorithmic meaning:
  - same routing source as `Fixed Heuristic`: `g`
  - larger embedding dimension and active encoder remain instantiated
- Physical interpretation:
  - inference: because `routing_mode=g` bypasses QK-derived selection, the
    extra Transformer/QK capacity is largely dormant with respect to the chosen
    routing scores and sparse update path
- Audit judgment:
  - the panel artifact provides no empirical separation from `Fixed Heuristic`.

### 6. `G-teacher` (`G-Teacher`)

- Artifact key: `G-Teacher`
- Representative flags: `routing_mode=g`, `d_model=346`,
  `encoder_identity=true`
- Clean aggregate: mean `0.4402`, all five clean seeds pick `best_epoch = 1`
- Algorithmic meaning:
  - same physics-correlation routing source as `Fixed Heuristic`
  - encoder is bypassed entirely
- Physical interpretation:
  - inference: this is an even more explicitly non-contextual `g`-family
    condition, but because `g` already determines selection, its observable
    behavior collapses onto the other `g`-family variants in the active sweep
- Audit judgment:
  - the current panel should not be interpreted as showing a distinct mechanism
    relative to `Fixed Heuristic` and `G-Fixed`.

### 7. `dense routing` (`Dense Routing`)

- Artifact key: `Dense Routing`
- Representative flags: `routing_mode=qk`, `disable_omp_sparsity=true`
- Clean aggregate: mean `0.0270`, all five clean seeds pick `best_epoch = 1`
- Algorithmic meaning:
  - in the executed runtime, the sparse routing/update path is bypassed,
  - the model uses an all-ones gate and performs `x = x + eta * g`
    rather than `x = x + eta * (w_all * g)`
- Physical interpretation:
  - this is not just a dense probability distribution over experts,
  - it removes angle-selective sparse pursuit and updates all atoms at once
- Audit judgment:
  - its near-chance behavior is mechanically consistent with destroying the
    sparse-selection prior that the inverse problem relies on.

## Seed-Level Pattern

Across the five clean seeds in the active 245-run sweep:

- `Baseline`, `No Transformer`, and `No Type Bias` all peak at epoch `20`
- `Fixed Heuristic`, `G-Fixed`, `G-Teacher`, and `Dense Routing` all peak at
  epoch `1`

This pattern supports a useful operational distinction:

- the first family is still learning a task-adapted routing policy during the
  20-epoch schedule,
- the latter family behaves like a fixed or nearly fixed routing/update rule
  under this protocol.

## Cross-Figure Identity Tension

`Fig. 4c` is not isolated from the rest of the active figure family.

The current worktree also contains a different solver lineage used by other
active figure artifacts:

- `results/omp_transformer_speech260_trainval_split_full_20251115_082341`
  records `d_model=128`, `routing_mode=qk`, `no_type_bias=true`,
  `score_norm=std`, `score_center_atoms=true`,
  `score_center_expert=true`, and `use_hard_gumbel=true`
- `figures/conf/experiments.yaml` records `no_type_bias: true` in the active
  `fig05_performance_structure` model parameters

This matters because:

- `Fig. 4d` uses the `2025-11-15` primary run metrics
- `Fig. 4c` uses the `2026-02-09` sweep baseline with `no_type_bias=false`
- therefore panel `c` and panel `d` are not backed by a single stable solver
  definition, even though the manuscript-facing labels make them look like a
  unified `guided solver` story

## Claim Risks

### 1. Type-bias claim risk

The active panel shows:

- `No Type Bias` mean `0.9751`
- `Baseline` mean `0.8625`

So the panel does not support statements such as:

- "the type-bias component contributes positively"
- "the type-bias component is essential for strong clean-condition accuracy"

at least not without additional explanation, a different data source, or a
different model definition.

### 2. Over-distinguishing the `g` family

The active panel cannot support treating:

- `Fixed Heuristic`
- `G-Fixed`
- `G-Teacher`

as three empirically distinct solver behaviors. In the current artifact they are
identical at the level of all five clean-seed accuracies and all five clean-seed
`best_epoch` values.

### 3. Protocol mismatch risk

The sweep metrics are validation accuracies from a deterministic
train/validation split, while the active manuscript text describes
held-out outer-fold test reporting. This should be disclosed whenever panel `c`
is used as evidence for a paper-level performance statement.

### 4. Cross-panel solver-identity drift

The panel-`c` `Baseline` and the active `Fig. 4d` / `Fig. 5` full-solver
artifacts do not share the same saved runtime flags. Any claim that treats them
as a single stable model identity should first reconcile that drift.

## Bottom Line

The active `Fig. 4c` panel is most defensible as a local routing-ablation audit
of one historical babble-noise validation sweep. Under that interpretation:

- `router-bypass` is a real learned-routing ablation,
- `dense routing` is a real sparsity-removal collapse condition,
- `fixed heuristic` is a real physics-first `g`-routing condition,
- `G-fixed` and `G-teacher` are not empirically separable from the same
  `g`-routing family in the current artifact,
- and `no type bias` currently behaves as a stronger clean-condition solver than
  the displayed `Baseline`.

This panel should therefore be read as a historically specific ablation surface,
not as a clean decomposition of the current paper-wide solver identity.
