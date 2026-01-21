# Acceptance Report: E4c — STOP-State Alignment Fix

- Run: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820`
- Outcome: PASS

## Executive Summary

E4c passes the E4b acceptance criteria after fixing STOP supervision to use the **real stop-decision correlation state** instead of a dummy all-zero state. This improves free-rollout monotonicity and significantly increases `steps_range`, consistent with the hypothesis that the prior failure was DUE TO a train/inference state mismatch for STOP.

## Key Metrics

From `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/eval/acceptance_check.json`:

- `spearman(lambda_c, steps_used_mean) = -0.9` (target ≤ -0.6) → PASS
- `steps_range = 0.51954` (target ≥ 0.10) → PASS
- `capture_range = 0.00735` (target ≥ 0.001) → PASS
- `max(action_change_rate_vs_ref) = 0.38297` (target ≥ 0.05) → PASS
- `max(logits_kl_mean_vs_ref) = 3.74263` (target > 0) → PASS

## Reproduction

Exact commands and stdout/stderr are recorded in:
- `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/run.log`

## Artifacts

- Code snapshot: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/code_state.json`
- Training data: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/data/merged_trajectories.pt`
- Merge meta: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/data/merged_trajectories.meta.json`
- Model: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Eval grid: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/eval/lambda_grid.json`
- Acceptance JSON: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/eval/acceptance_check.json`
- Training diagnostics: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/train/diagnostics.json`

## Next Step

Proceed to a results commit (code + artifacts) for E4c, since the acceptance criteria pass.

