# Ablation Results (Speech260 val-split, seed 42)

Runs (20 epochs, device=mps, dataset `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`, H `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`, W `doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth`):

| Run | Flags | Best acc | Best epoch | Majority angle (deg) | Majority ratio | Per-angle min/max | Non-zero angles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ablate_baseline_speech260_seed42_20251209_202217 | baseline | 0.935 | 20 | 50.0 | 0.033 | 0.808 / 1.000 | 37 |
| ablate_identity_speech260_seed42_20251210_134919 | `--encoder_identity` | 0.631 | 20 | 100.0 | 0.044 | 0.096 / 0.962 | 37 |
| ablate_g_routing_speech260_seed42_20251210_134937 | `--routing_mode g` | 0.017 | 1 | 60.0 | 0.293 | 0.000 / 0.250 | 4 |
| ablate_no_type_bias_speech260_seed42_20251210_134944 | `--no_type_bias` | 0.912 | 20 | 75.0 | 0.041 | 0.577 / 1.000 | 37 |
| ablate_disable_omp_sparsity_speech260_seed42_20251210_135000 | `--disable_omp_sparsity` | 0.027 | 1 | 0.0 | 1.000 | 0.000 / 1.000 | 1 |

Notes:
- Removing sparsity or switching to pure g-routing collapses to a single/few angles (majority ratio 29–100%, per-angle coverage near zero), confirming sparsity and learnable routing are essential.
- Identity encoder hurts but still multi-class (~0.63 acc), showing Transformer nonlinearity contributes meaningfully.
- Removing type embeddings mildly reduces accuracy (~0.91 vs 0.93 baseline) but keeps balanced per-angle coverage.
