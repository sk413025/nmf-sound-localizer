# Acceptance Report: E4 — DAgger-lite (Exposure-Bias Mitigation)

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838`
- Goal: Reduce exposure bias so free-rollout monotonicity is correct.
- Outcome: **PASS** (monotonicity corrected, but near-saturated steps).
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = -0.9` (target ≤ -0.6)
  - `steps_range = 0.038439024390244825`
  - `capture_range = 0.0005000012211683336`
  - `max(action_change_rate_vs_ref) = 0.3884437010144327`
  - `max(logits_kl_mean_vs_ref) = 2.0614291559904068`

## 2) Experiment Context (REQUIRED)

- Background: Free-rollout monotonicity inverted in baseline/E1.
- Motivation: E2 showed monotonicity correct under teacher-forced states, implying exposure bias.
- Purpose: Test DAgger-lite (student-visited states labeled by teacher) to mitigate state drift.
- Expected: `lambda_c ↑ ⇒ steps_used_mean ↓` under free rollout.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: MPS (see `run.log`)
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset: `first 3 clip pairs in dataset order (all angles)`
- Fingerprint: `668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `Tw=32`, `max_lag=50`, `K_max=16`, `gain=100.0`
- Teacher: `penalty_omp`, `lambda_c_values=1e-4,3e-4,1e-3,3e-3,1e-2`, `min_k=1`
- DAgger:
  - `num_dagger_blocks=25` (num_clips=1 for DAgger collection)
  - `dagger_ratio=1.0` (merged with 75 teacher blocks)
  - `rollout_policy=results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model/dt_freq_aware_best.pth`

## 4) Exact Commands (REQUIRED)

```bash
# 1) Generate teacher trajectories
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --out_dir results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --variants_per_clip 1 --max_items 3 --all_angles   --teacher_mode penalty_omp   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --min_k 1 --seed 0

# 2) Collect DAgger data
PYTHONPATH=. python -u scripts/h_exploration/run_dagger_collect.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model/dt_freq_aware_best.pth   --out_dir results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --min_k 1 --num_clips 1 --use_stop_action   --rtg1_mode max_k --rtg1_max_k 16

# 3) Merge datasets
PYTHONPATH=. python -u scripts/h_exploration/merge_dagger_data.py   --teacher_pt results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data/lag_trajectories.pt   --dagger_pt results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data/dagger_trajectories.pt   --out_pt results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data/merged_trajectories.pt   --dagger_ratio 1.0 --seed 0

# 4) Train on merged data
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py   --data_path results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/data/merged_trajectories.pt   --out_dir results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/model   --epochs 5 --batch_size 128 --lr 5e-4   --rtg_dim 2 --rtg_mode lambda_cost   --rtg1_mode max_k --rtg1_max_k 16   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --use_stop_action --seed 0

# 5) Free-rollout eval
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/model/dt_freq_aware_best.pth   --subset_manifest results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/subset_manifest.json   --out_dir results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/eval   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --num_clips 1 --use_stop_action   --rollout_mode free

# 6) Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py   --lambda_grid results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/eval/lambda_grid.json   --out_json results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/eval/acceptance_check.json
```

## 5) Artifacts (REQUIRED)

- `run.log`
- `subset_manifest.json`
- `code_state.json`
- `data/lag_trajectories.pt`
- `data/dagger_trajectories.pt`
- `data/merged_trajectories.pt`
- `model/dt_freq_aware_best.pth`
- `train/diagnostics.json`
- `eval/lambda_grid.json`
- `eval/acceptance_check.json`

## 6) Results (REQUIRED)

- `spearman(lambda_c, steps_used_mean) = -0.9`
- `steps_used_mean range = 15.786146 → 15.824585`
- `final_capture_mean range = 0.996233 → 0.996733`
- `max(action_change_rate_vs_ref) = 0.3884437010144327`
- `max(logits_kl_mean_vs_ref) = 2.0614291559904068`
- `steps_range = 0.038439024390244825`, `capture_range = 0.0005000012211683336`

## 7) Interpretation (REQUIRED)

- Free-rollout monotonicity is corrected (rho = -0.9), THEREFORE DAgger-lite mitigates exposure bias.
- However, steps remain near K_max with a small range; further tuning may be needed to increase the separation across lambda.

## 8) Next Steps (REQUIRED)

- Increase DAgger ratio or add a second DAgger iteration to amplify the lambda-dependent step range.
