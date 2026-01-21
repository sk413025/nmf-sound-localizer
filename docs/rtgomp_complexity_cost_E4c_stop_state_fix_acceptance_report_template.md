# Acceptance Report Template: E4c — STOP-State Alignment Fix

Use this template for **E4c**. Fill all placeholders and keep causal language.

---

# Acceptance Report: E4c — STOP-State Alignment Fix

## 1) Executive Summary

- Run: `<results/run_name>`
- Goal: Fix STOP supervision mismatch by aligning STOP labels to the real stop-decision correlation state.
- Outcome: `<PASS / PARTIAL / FAIL>`
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = <value>` (target ≤ -0.6)
  - `steps_range = <value>` (target ≥ 0.10)
  - `capture_range = <value>` (target ≥ 0.001)
  - `max(action_change_rate_vs_ref) = <value>`
  - `max(logits_kl_mean_vs_ref) = <value>`

## 2) Experiment Context (REQUIRED)

- Background: E4b produced a complete run but free-rollout steps remained saturated near `K_max`.
- Motivation: Stop decisions must be learned from states that match inference-time rollout states.
- Purpose: Validate that aligning STOP labels to real stop-decision states improves monotonicity and range.
- Expected: `lambda_c ↑ ⇒ steps_used_mean ↓` with a visibly larger `steps_range`.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Training data: `<merged_trajectories.pt path>`
- Eval subset: `<subset_manifest path>`
- Lambda sweep: `<...>`
- Params: `Tw=<...>`, `max_lag=<...>`, `K_max=<...>`, `gain=<...>`

## 4) Exact Commands (REQUIRED)

```bash
# Train (reusing existing merged trajectories)
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/<run_name>/data/merged_trajectories.pt \
  --out_dir results/<run_name>/model \
  --rtg_dim 2 --rtg_mode lambda_cost \
  --rtg1_mode max_k --rtg1_max_k 16 \
  --lambda_c_values "<...>" \
  --use_stop_action --seed 0

# Free-rollout eval
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root <...> \
  --ldv_root <...> \
  --ckpt_path results/<run_name>/model/dt_freq_aware_best.pth \
  --subset_manifest results/<run_name>/subset_manifest.json \
  --out_dir results/<run_name>/eval \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --lambda_c_values "<...>" \
  --use_stop_action \
  --rollout_mode free

# Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid results/<run_name>/eval/lambda_grid.json \
  --out_json results/<run_name>/eval/acceptance_check.json
```

## 5) Results (REQUIRED)

- Paste the top-level JSON summary from `acceptance_check.json` and interpret pass/fail per metric.

## 6) Interpretation (REQUIRED)

- Explain whether monotonicity and range improved BECAUSE STOP is now trained on real stop-decision states.
- If FAIL, explain DUE TO what remaining mismatch (teacher rule, lambda sweep, exposure bias).

## 7) Next Step (REQUIRED)

Choose exactly one:
- Increase `--dagger_ratio` to `5.0`, OR
- Add a second DAgger iteration using the E4c model as the rollout policy.

