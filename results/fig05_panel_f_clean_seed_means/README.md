## Fig. 5f clean 5-seed means

This directory stores the aggregated clean-condition per-angle summary used by `Fig. 5f`.

### Artifact

- `summary.npz`
  - `angles`
  - `guided_mean`, `guided_std`
  - `router_bypass_mean`, `router_bypass_std`
  - `omp_mean`, `omp_std`
  - `dense_mean`, `dense_std`

### Provenance

The summary is derived from the same clean 5-seed babble Speech260 sweep family that underlies `results/figure4_data.json`:

- `ablate_speech260_baseline_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `ablate_speech260_no_transformer_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `ablate_speech260_g_routing_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`
- `ablate_speech260_disable_sparsity_snrInf_seed{1,2,3,4,42}_ep20_lr1e-3_babble_speech260_full_20260209_011327`

The mean of each per-angle mean curve matches the corresponding clean ablation mean in `results/figure4_data.json`.
