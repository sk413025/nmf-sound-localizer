# Acceptance Report Template: E4b — High-Ratio DAgger-lite

Use this template for **E4b**. Fill all placeholders and keep causal language.

---

# Acceptance Report: E4b — High-Ratio DAgger-lite

## 1) Executive Summary

- Run: `<results/run_name>`
- Goal: Increase DAgger signal (ratio=3.0) and coverage (num_clips=3) to widen step range.
- Outcome: `<PASS / PARTIAL / FAIL>`
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = <value>` (target ≤ -0.6)
  - `steps_range = <value>` (target ≥ 0.10)
  - `capture_range = <value>` (target ≥ 0.001)
  - `max(action_change_rate_vs_ref) = <value>`
  - `max(logits_kl_mean_vs_ref) = <value>`

## 2) Experiment Context (REQUIRED)

- Background: E4 corrected monotonicity but `steps_range` was too small.
- Motivation: Increase DAgger ratio and coverage to strengthen lambda-dependent stop behavior.
- Purpose: Validate that E4b increases step range while preserving monotonicity.
- Expected: `lambda_c ↑ ⇒ steps_used_mean ↓` and `steps_range ≥ 0.10`.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Note: DAgger collection runs many least-squares solves on CPU, so MPS utilization can appear low.
- Data roots:
  - MIC: `<path>`
  - LDV: `<path>`
- Subset: `<selection procedure>`
- Fingerprint: `<manifest fingerprint>`
- Params: `Tw=<...>`, `max_lag=<...>`, `K_max=<...>`, `gain=<...>`
- Teacher: `penalty_omp`, `lambda_c_values=<...>`, `min_k=<...>`
- DAgger:
  - `num_clips=3`
  - `dagger_ratio=3.0`
  - `stride=128`
  - rollout policy ckpt: `<path>`

## 4) Exact Commands (REQUIRED)

```bash
# 1) Generate teacher trajectories
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py \
  --mic_root <...> \
  --ldv_root <...> \
  --out_dir results/<run_name>/data \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --variants_per_clip 1 --max_items <...> --all_angles \
  --teacher_mode penalty_omp \
  --lambda_c_values "<...>" \
  --min_k 1 --seed 0

# 2) Collect DAgger data (num_clips=3)
PYTHONPATH=. python -u scripts/h_exploration/run_dagger_collect.py \
  --mic_root <...> \
  --ldv_root <...> \
  --ckpt_path <student_ckpt> \
  --out_dir results/<run_name>/data \
  --lambda_c_values "<...>" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --min_k 1 --num_clips 3 --stride 128 --use_stop_action \
  --rtg1_mode max_k --rtg1_max_k 16 \
  --log_every_windows 25

# 3) Merge datasets (ratio=3.0)
PYTHONPATH=. python -u scripts/h_exploration/merge_dagger_data.py \
  --teacher_pt results/<run_name>/data/lag_trajectories.pt \
  --dagger_pt results/<run_name>/data/dagger_trajectories.pt \
  --out_pt results/<run_name>/data/merged_trajectories.pt \
  --dagger_ratio 3.0 --seed 0

# 4) Train on merged data
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/<run_name>/data/merged_trajectories.pt \
  --out_dir results/<run_name>/model \
  --epochs <...> --batch_size <...> --lr <...> \
  --rtg_dim 2 --rtg_mode lambda_cost \
  --rtg1_mode max_k --rtg1_max_k 16 \
  --lambda_c_values "<...>" \
  --use_stop_action --seed 0

# 5) Free-rollout eval
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root <...> \
  --ldv_root <...> \
  --ckpt_path results/<run_name>/model/dt_freq_aware_best.pth \
  --subset_manifest results/<run_name>/subset_manifest.json \
  --out_dir results/<run_name>/eval \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --lambda_c_values "<...>" \
  --num_clips <...> --use_stop_action \
  --rollout_mode free

# 6) Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid results/<run_name>/eval/lambda_grid.json \
  --out_json results/<run_name>/eval/acceptance_check.json
```

## 5) Artifacts (REQUIRED)

- `results/<run_name>/run.log`
- `results/<run_name>/subset_manifest.json`
- `results/<run_name>/code_state.json`
- `results/<run_name>/data/lag_trajectories.pt`
- `results/<run_name>/data/dagger_trajectories.pt`
- `results/<run_name>/data/merged_trajectories.pt`
- `results/<run_name>/data/merged_trajectories.meta.json`
- `results/<run_name>/model/dt_freq_aware_best.pth`
- `results/<run_name>/train/diagnostics.json`
- `results/<run_name>/eval/lambda_grid.json`
- `results/<run_name>/eval/acceptance_check.json`
- `results/<run_name>/ACCEPTANCE_REPORT.md`

## 6) Results (REQUIRED)

- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_used_mean range = <min> → <max>`
- `final_capture_mean range = <min> → <max>`
- `max(action_change_rate_vs_ref) = <value>`
- `max(logits_kl_mean_vs_ref) = <value>`
- `steps_range = <value>`, `capture_range = <value>`

## 7) Interpretation (REQUIRED)

- State whether `steps_range` increased BECAUSE higher DAgger ratio reduced exposure bias.
- If FAIL, explain DUE TO which factor (label noise, imbalance, STOP collapse).

## 8) Next Steps (REQUIRED)

Choose exactly one:
- increase DAgger ratio to 5.0
- add a second DAgger iteration
- flip RTG0 direction (E3)

## Debug Notes (optional)

- If DAgger appears stalled with no logs, re-run with `--log_every_windows 1`, or send `SIGUSR1` to the Python PID to dump stack traces.
