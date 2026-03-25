## Fig. 5f clean 5-seed means with the unified guided-solver family

This directory stores the aggregated clean-condition per-angle summary used by
`Fig. 5f` after unifying the active decoder families across Figs. 4-5.

### Artifact

- `summary.npz`
  - `angles`
  - `guided_mean`, `guided_std`
  - `router_bypass_mean`, `router_bypass_std`
  - `omp_mean`, `omp_std`
  - `dense_mean`, `dense_std`

### Provenance

The summary is derived from the same clean 5-seed babble Speech260 sweep family
that underlies `results/figure4_data.json`, with one family-level change:

- `guided_mean/std` now come from the five `No Type Bias` runs:
  - `ablate_speech260_no_type_bias_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `router_bypass_mean/std` come from:
  - `ablate_speech260_no_transformer_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `omp_mean/std` come from the soft-OMP family:
  - `ablate_speech260_g_routing_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `dense_mean/std` come from:
  - `ablate_speech260_disable_sparsity_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`

The mean of each per-angle mean curve matches the corresponding clean family
mean now intended for the active `Fig. 4c` and `Fig. 5a/f` decoder contract:

- guided solver (`No Type Bias`): `0.975051975051975`
- router-bypass: `0.582016632016632`
- OMP baseline: `0.4402286902286902`
- dense routing: `0.02702702702702703`
