
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
- Completed: seed 42 (run dir `results/omp_transformer_speech260_trainval_split_full_20251202_192153`, val acc ≈0.946, train acc ≈0.976).
- In progress: seed 1 (`tmux` session `valsplit_seed1`, run dir `results/omp_transformer_speech260_trainval_split_full_seed1_20251203_102515`, 20 epochs, mps).
- In progress: seed 2 (`tmux` session `valsplit_seed2`, run dir `results/omp_transformer_speech260_trainval_split_full_seed2_20251203_102524`, 20 epochs, mps).
- Remaining to schedule after these finish: seeds 3 and 4 to reach 5 total repeats.
