# Acceptance Report: E4b — High-Ratio DAgger-lite

- Run: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725`
- Outcome: FAIL

## Summary

E4b (DAgger ratio=3.0, num_clips=3, stride=128) does not widen the free-rollout step range: `steps_used_mean` remains saturated near `K_max` across the lambda sweep, so the monotonicity strength target is not met.

## Key Metrics

From `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/eval/acceptance_check.json`:

- `spearman(lambda_c, steps_used_mean) = -0.3` (target ≤ -0.6) → FAIL
- `steps_range = 0.00410` (target ≥ 0.10) → FAIL
- `capture_range = 0.000339` (target ≥ 0.001) → FAIL
- `max(action_change_rate_vs_ref) = 0.21176` (target ≥ 0.05) → PASS
- `max(logits_kl_mean_vs_ref) = 2.27905` (target > 0) → PASS

Per-lambda `steps_used_mean` (near-saturation):
- `1e-4`: `15.17912`
- `3e-4`: `15.17620`
- `1e-3`: `15.17724`
- `3e-3`: `15.17502`
- `1e-2`: `15.17893`

## Reproduction

Exact commands and stdout/stderr are recorded in:
- `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/run.log`

## Artifacts

- Code snapshot: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/code_state.json`
- Teacher data: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/data/lag_trajectories.pt`
- DAgger data: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/data/dagger_trajectories.pt`
- Merged data: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/data/merged_trajectories.pt`
- Model: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/model/dt_freq_aware_best.pth`
- Eval: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/eval/lambda_grid.json`
- Acceptance: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/eval/acceptance_check.json`

## Next Step

Pick exactly one (per E4b template):
- Increase `--dagger_ratio` to `5.0`, OR
- Add a second DAgger iteration using the E4b model as the rollout policy.

