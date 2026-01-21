# Spec: E4b DAgger-lite (High Ratio + Wider Coverage)

This spec extends E4 with **higher DAgger ratio** and **larger DAgger coverage**.
It is intended to **increase the lambda-dependent step range** under free rollout.

Dependencies:
- `docs/rtgomp_complexity_cost_spec.md`
- `docs/rtgomp_complexity_cost_E4_spec.md`
- `docs/rtgomp_complexity_cost_E4b_plan.md`

---

## 1) Required Parameter Changes (from E4)

### 1.1 DAgger ratio
- `--dagger_ratio = 3.0`

### 1.2 DAgger coverage
- `run_dagger_collect.py --num_clips = 3`
 - `run_dagger_collect.py --stride = 128` (runtime cap; must be recorded)
 - Recommended: `run_dagger_collect.py --log_every_windows = 25` (progress visibility; set to 1 for debugging)

All other parameters should match E4:
- `Tw=32`, `max_lag=50`, `K_max=16`, `gain=100`, `min_k=1`
- `rtg1_mode = max_k`

---

## 2) Acceptance Criteria (E4b)

PASS if all:
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`
- `capture_range >= 0.001`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

PARTIAL if:
- monotonicity passes but `steps_range < 0.10`

FAIL if:
- monotonicity is positive or near zero
- steps saturate at `K_max`

---

## 3) Required Artifacts

Same as E4:
- `results/<run>/data/lag_trajectories.pt`
- `results/<run>/data/dagger_trajectories.pt`
- `results/<run>/data/merged_trajectories.pt`
- `results/<run>/data/merged_trajectories.meta.json`
- `results/<run>/model/dt_freq_aware_best.pth`
- `results/<run>/eval/lambda_grid.json`
- `results/<run>/eval/acceptance_check.json`
- `results/<run>/ACCEPTANCE_REPORT.md`

---

## 4) Reporting

`ACCEPTANCE_REPORT.md` must explicitly compare:
- E4 `steps_range` vs E4b `steps_range`
- E4 `capture_range` vs E4b `capture_range`
