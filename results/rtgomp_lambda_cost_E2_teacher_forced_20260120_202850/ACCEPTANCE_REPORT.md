# Acceptance Report: E2 — Teacher-Forced Eval (Exposure Bias Test)

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E2_teacher_forced_20260120_202850`
- Goal: Isolate exposure bias by evaluating the student on teacher-forced state trajectories.
- Outcome: **PASS** for monotonicity under teacher-forced states (diagnostic success).
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = -1.0` (expected ≤ -0.6)
  - `steps_range = 0.6899512195121948`
  - `capture_range = 0.01053686279203836`
  - `max(action_change_rate_vs_ref) = 0.43963514994490605`
  - `max(logits_kl_mean_vs_ref) = 1.8973129178326735`

## 2) Experiment Context (REQUIRED)

- Background: Free rollout shows inverted monotonicity (lambda_c↑ -> steps_used_mean↑).
- Motivation: Test whether state distribution shift (exposure bias) is the dominant cause.
- Purpose: Evaluate the student on teacher-forced residual updates and measure monotonicity.
- Expected: Monotonicity should be correct (lambda_c↑ -> steps_used_mean↓) if exposure bias is the main issue.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: MPS (see `run.log`)
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset: `first 3 clip pairs in dataset order (all angles)`
- Fingerprint: `668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `Tw=32`, `max_lag=50`, `K_max=16`, `gain=100.0`
- Teacher-forced eval:
  - `rollout_mode=teacher_forced`
  - `teacher_min_k=1`
- Model checkpoint (from E1):
  - `results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model/dt_freq_aware_best.pth`

## 4) Exact Commands (REQUIRED)

```bash
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model/dt_freq_aware_best.pth   --subset_manifest results/rtgomp_lambda_cost_E2_teacher_forced_20260120_202850/subset_manifest.json   --out_dir results/rtgomp_lambda_cost_E2_teacher_forced_20260120_202850/eval   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --num_clips 1 --use_stop_action   --rollout_mode teacher_forced --teacher_min_k 1

PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py   --lambda_grid results/rtgomp_lambda_cost_E2_teacher_forced_20260120_202850/eval/lambda_grid.json   --out_json results/rtgomp_lambda_cost_E2_teacher_forced_20260120_202850/eval/acceptance_check.json
```

## 5) Artifacts (REQUIRED)

- `run.log`
- `subset_manifest.json`
- `code_state.json`
- `eval/lambda_grid.json`
- `eval/acceptance_check.json`

## 6) Results (REQUIRED)

- `spearman(lambda_c, steps_used_mean) = -1.0`
- `steps_used_mean range = 2.593171 → 3.283122`
- `final_capture_mean range = 0.400149 → 0.410686`
- `max(action_change_rate_vs_ref) = 0.43963514994490605`
- `max(logits_kl_mean_vs_ref) = 1.8973129178326735`

## 7) Interpretation (REQUIRED)

- The monotonicity is correct under **teacher-forced** states (rho = -1.0), THEREFORE the inverted trend in free rollout is dominated by **exposure bias / state drift**, not RTG1 semantics.
- This implies the next fix should target rollout distribution shift (e.g., DAgger-lite or teacher-forced supervision in training).

## 8) Next Steps (REQUIRED)

- Implement E4 (DAgger-lite) or add a teacher-forced training augmentation to reduce exposure bias.
