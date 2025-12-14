
## Speech260 val-split reproduction (Codex 019adebd-ff0a-7630-b7b1-977d394e40e6)

- Goal: reproduce commit 06bf65de4071bd0cda0211f96e6900442bb67ce7 (val split ≈94.6% on Box 16 kHz) under the current dataset fingerprint.
- Branch/worktree: `exp/omp-speech260-valsplit-20251202` at commit 06bf65de.
- Data paths: H `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`; W `doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth`; Y `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized` with current fingerprint `e23ded2e267115ba6383b83543646857` (differs from historical `f5639848…`).
- Command (tmux `speech260_valsplit_repro`, uses `/Users/jiawei/miniconda3/bin/conda run -n trl-training`, device mps, epochs=20):
  - `RUN_DIR=results/omp_transformer_speech260_trainval_split_full_20251202_$TIME`
  - `python -u scripts/omp-transformer-ldv.py [same args as 06bf65de] --out_dir $RUN_DIR | tee $RUN_DIR/run.log`
  - `python -u scripts/eval_omp_transformer_split.py --run_dir $RUN_DIR --device mps --subset both | tee $RUN_DIR/posthoc_eval.log`
- Current run dir: `results/omp_transformer_speech260_trainval_split_full_20251202_192153`; split diagnostics recorded (train=7696, val=1924, per-angle 208/52). Training still running; run.log/posthoc_eval.log will populate once the job finishes.
- Expected outcome: match the original val accuracy ≈0.946 on the val split; actual may deviate due to dataset fingerprint change (e23ded2e… vs f5639848…); will record final train/val metrics once the run completes.

### Seed tracking (target: 5 repeats)
- Completed val-split runs (20 epochs, mps, dataset MD5 f563984848ae49b4443378c4ef720a51):
  - seed 42: `results/omp_transformer_speech260_trainval_split_full_20251202_192153` — train 0.976, val 0.946.
  - seed 1: `results/omp_transformer_speech260_trainval_split_full_seed1_20251203_105731` — train 0.993, val 0.962.
  - seed 2: `results/omp_transformer_speech260_trainval_split_full_seed2_20251203_105739` — train 0.991, val 0.966.
  - seed 3: `results/omp_transformer_speech260_trainval_split_full_seed3_20251203_202610` — train 0.986, val 0.951.
  - seed 4: `results/omp_transformer_speech260_trainval_split_full_seed4_20251203_202619` — train 0.993, val 0.975.
  - seed 5: `results/omp_transformer_speech260_trainval_split_full_seed5_20251203_202628` — train 0.986, val 0.962.
- Eval fix: `scripts/eval_omp_transformer_split.py` now reuses training `seed` for atom reduction, preventing post-hoc metric mismatch.

## Ablation Runs (Speech260, val split, 20 epochs, mps, seeds 42/1/2/3/4)
Data: H `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`; W `doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth`; Y `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`.

- `--encoder_identity` (No Transformer): seeds 42/1/2/3/4 finished.
  - Acc: 0.631 (42) / 0.623 (1) / 0.727 (2) / 0.640 (3) / 0.678 (4); per-angle coverage = 37/37 for all.
  - Paths: `results/ablate_identity_speech260_seed42_20251210_203029`, `..._seed1_20251211_113851`, `..._seed2_20251211_185200`, `..._seed3_20251212_161859`, `..._seed4_20251213_043329`.

- `--routing_mode g` (Fixed heuristic): seeds 42/1/2/3/4 finished.
  - Acc: 0.017 (42) / 0.014 (1) / 0.019 (2) / 0.027 (3) / 0.030 (4); non-zero angles 3–7; heavy majority collapse (up to 62%).
  - Paths: `results/ablate_g_routing_speech260_seed42_20251210_203034`, `..._seed1_20251211_122649`, `..._seed2_20251211_210227`, `..._seed3_20251212_164010`, `..._seed4_20251213_081907`.

- `--no_type_bias` (No type embeddings): seeds 42/1/2/3 finished; seed4 running (`results/ablate_no_type_bias_speech260_seed4_20251213_142618`).
  - Acc: 0.912 (42) / 0.940 (1) / 0.920 (2) / 0.929 (3) / 0.922 (4); per-angle coverage = 37/37.
  - Paths: `results/ablate_no_type_bias_speech260_seed42_20251210_203042`, `..._seed1_20251211_124027`, `..._seed2_20251212_025820`, `..._seed3_20251212_173046`, `..._seed4_20251213_142618`.

- `--disable_omp_sparsity` (Dense routing): seeds 42/1/2/3/4 finished.
  - Acc: 0.027 across all seeds; predicts single angle (non-zero angles = 1; majority ratio = 1.0).
  - Paths: `results/ablate_disable_omp_sparsity_speech260_seed42_20251210_203049`, `..._seed1_20251211_022038`, `..._seed2_20251211_032717`, `..._seed3_20251211_125643`, `..._seed4_20251211_171758`.

### Status + visualization
- All planned ablation seeds (42/1/2/3/4) have completed.
- Quartile/box plot comparing top-1 accuracies across methods is saved at `results/ablation_accuracy_quartiles.png` (Baseline, No Transformer, G-Routing, No Type Bias, Dense Routing).
