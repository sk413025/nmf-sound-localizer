# Plan: E4c — Fix STOP-State Alignment (Lambda-Cost RTG-OMP)

E4b executed end-to-end but failed acceptance because free-rollout `steps_used_mean` stayed saturated near `K_max`, producing:
- weak monotonicity (Spearman ρ too close to 0), and
- near-zero `steps_range` / `capture_range`.

## 1) Hypothesis (Root Cause)

The STOP action is being supervised on an **unrealistic state** in training.

In the current `lambda_cost` dataset construction, STOP is appended using a **dummy all-zero correlation state**, instead of using the **actual student-visited correlation state** at the stop decision time. Therefore the model does not learn to trigger STOP during real free rollout states, even when `valid_len` indicates early stopping is frequent.

## 2) Minimal Fix

For each frequency:
- Use `valid_len = L` (number of selected atoms before stop).
- Train on:
  - `corr_seq = corrs[f, :L+1]` (include the real state at the stop decision step),
  - `act_seq = actions[f, :L] + [STOP_ID]`.

If `L == K_max` (no stop), do not append STOP.

This changes only dataset construction; it does not change the trajectory generators or teacher rules.

## 3) Experiment Design

Re-train + re-evaluate using the exact same E4b merged trajectories and subset:
- Training data: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/data/merged_trajectories.pt`
- Eval subset: `results/rtgomp_lambda_cost_E4b_ratio3_20260121_020725/subset_manifest.json`
- Lambda sweep: same as E4b run log

## 4) Acceptance Criteria (same as E4b)

PASS if all:
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`
- `capture_range >= 0.001`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

## 5) Expected Outcome

Monotonicity and range improve BECAUSE STOP is now trained on the same type of correlation state encountered during free rollout (no dummy state mismatch).

