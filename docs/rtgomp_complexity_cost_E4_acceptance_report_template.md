# Acceptance Report Template: E4 — DAgger-lite

Use this template for **Experiment E4** (DAgger-lite exposure-bias mitigation).  
Fill all placeholders and keep causal language (BECAUSE / THEREFORE).

---

# Acceptance Report: E4 — DAgger-lite (Exposure-Bias Mitigation)

## 1) Executive Summary

- Run: `<results/run_name>`
- Goal: Reduce exposure bias so **free-rollout** monotonicity is correct.
- Outcome: `<PASS/FAIL>`
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = <value>` (target ≤ -0.6)
  - `steps_range = <value>`, `capture_range = <value>`
  - `max(action_change_rate_vs_ref) = <value>`
  - `max(logits_kl_mean_vs_ref) = <value>`

## 2) Experiment Context (REQUIRED)

- Background: `<free-rollout monotonicity inverted in baseline/E1>`
- Motivation: `<E2 confirmed exposure bias>`
- Purpose: `<test DAgger-lite as mitigation>`
- Expected: `<lambda_c ↑ ⇒ steps_used_mean ↓ under free rollout>`

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Data roots:
  - MIC: `<path>`
  - LDV: `<path>`
- Subset: `<selection procedure>`
- Fingerprint: `<manifest fingerprint>`
- Params: `Tw=<...>`, `max_lag=<...>`, `K_max=<...>`, `gain=<...>`
- Teacher: `penalty_omp`, `lambda_c_values=<...>`, `min_k=<...>`
- DAgger:
  - `num_dagger_blocks=<...>`
  - `dagger_ratio=<...>`
  - `rollout_policy=<ckpt path>`

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

# 2) Collect DAgger data
PYTHONPATH=. python -u scripts/h_exploration/run_dagger_collect.py \
  --mic_root <...> \
  --ldv_root <...> \
  --ckpt_path <student_ckpt> \
  --out_dir results/<run_name>/data \
  --lambda_c_values "<...>" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --min_k 1 --seed 0

# 3) Merge teacher + DAgger data
PYTHONPATH=. python -u scripts/h_exploration/merge_dagger_data.py \
  --teacher_pt results/<run_name>/data/lag_trajectories.pt \
  --dagger_pt results/<run_name>/data/dagger_trajectories.pt \
  --out_pt results/<run_name>/data/merged_trajectories.pt \
  --dagger_ratio <...>

# 4) Train on merged data
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/<run_name>/data/merged_trajectories.pt \
  --out_dir results/<run_name>/model \
  --epochs <...> --batch_size <...> --lr <...> \
  --rtg_dim 2 --rtg_mode lambda_cost \
  --rtg1_mode max_k --rtg1_max_k <K_max> \
  --lambda_c_values "<...>" \
  --use_stop_action --seed 0

# 5) Free-rollout eval (no teacher forcing)
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
- `results/<run_name>/model/dt_freq_aware_best.pth`
- `results/<run_name>/train/diagnostics.json`
- `results/<run_name>/eval/lambda_grid.json`
- `results/<run_name>/eval/acceptance_check.json`
- `results/<run_name>/ACCEPTANCE_REPORT.md`

## 6) Results (REQUIRED)

Report the values from `acceptance_check.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_used_mean range = <min> → <max>`
- `final_capture_mean range = <min> → <max>`
- `max(action_change_rate_vs_ref) = <value>`
- `max(logits_kl_mean_vs_ref) = <value>`
- `steps_range = <value>`, `capture_range = <value>`

## 7) Pass/Fail Checklist (E4)

- [ ] `spearman(lambda_c, steps_used_mean) <= -0.6`
- [ ] `steps_range > 0`
- [ ] `capture_range > 0`
- [ ] `max(action_change_rate_vs_ref) >= 0.05`
- [ ] `max(logits_kl_mean_vs_ref) > 0`

## 8) Interpretation (REQUIRED)

- Explain whether monotonicity is corrected BECAUSE DAgger-lite reduced exposure bias.
- If FAIL, identify whether failure is due to poor labels, data imbalance, or STOP collapse.

## 9) Next Steps (REQUIRED)

Choose exactly one:
- Increase DAgger ratio
- Add multiple DAgger iterations
- Flip RTG0 direction
- Modify STOP loss weighting

