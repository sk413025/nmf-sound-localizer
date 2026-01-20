# Acceptance Report: E1 — RTG1 Alignment (Complexity Cost)

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622`
- Goal: Align RTG1 semantics between training and eval (use max_k schedule).
- Outcome: **FAIL**.
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = 1.0` (target ≤ -0.6)
  - `steps_range = 0.004682926829268297`, `capture_range = 7.321996223641491e-05`
  - `max(action_change_rate_vs_ref) = 0.43525035208693746`
  - `max(logits_kl_mean_vs_ref) = 1.8726172322000834`

## 2) Experiment Context (REQUIRED)

- Background: Prior runs showed inverted monotonicity (`lambda_c ↑ → steps_used_mean ↑`) despite teacher monotonicity.
- Motivation: RTG1 mismatch between training and eval was a leading hypothesis for the inversion.
- Purpose: Test whether aligning RTG1 semantics fixes the direction of the steps trend.
- Expected: `spearman(lambda_c, steps_used_mean) <= -0.6` after RTG1 alignment.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: MPS (see `run.log`)
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset: `first 3 clip pairs in dataset order (all angles)`
- Fingerprint: `668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `Tw=32`, `max_lag=50` (`M=101`), `K_max=16`, `gain=100.0`
- Teacher: `penalty_omp`, `lambda_c_values=0.0001, 0.0003, 0.001, 0.003, 0.01`, `min_k=1`
- RTG1 config (E1):
  - `rtg1_mode = max_k`
  - `rtg1_max_k = 16`

## 4) Exact Commands (REQUIRED)

```bash
# 1) Generate trajectories
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --out_dir results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/data   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --variants_per_clip 1 --max_items 3 --all_angles   --teacher_mode penalty_omp   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --min_k 1 --seed 0

# 2) Train (E1: RTG1 max_k)
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py   --data_path results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/data/lag_trajectories.pt   --out_dir results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model   --epochs 5 --batch_size 128 --lr 5e-4   --rtg_dim 2 --rtg_mode lambda_cost   --rtg1_mode max_k --rtg1_max_k 16   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --use_stop_action --seed 0

# 3) Eval (lambda grid)
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/model/dt_freq_aware_best.pth   --subset_manifest results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/subset_manifest.json   --out_dir results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/eval   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --num_clips 1 --use_stop_action

# 4) Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py   --lambda_grid results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/eval/lambda_grid.json   --out_json results/rtgomp_lambda_cost_E1_rtg1maxk_20260120_200622/eval/acceptance_check.json
```

## 5) Artifacts (REQUIRED)

- `run.log`
- `subset_manifest.json`
- `code_state.json`
- `data/lag_trajectories.pt`
- `model/dt_freq_aware_best.pth`
- `train/diagnostics.json`
- `eval/lambda_grid.json`
- `eval/acceptance_check.json`

## 6) Results (REQUIRED)

- `spearman(lambda_c, steps_used_mean) = 1.0`
- `steps_used_mean range = 15.975805 → 15.980488`
- `final_capture_mean range = 0.998490 → 0.998563`
- `max(action_change_rate_vs_ref) = 0.43525035208693746`
- `max(logits_kl_mean_vs_ref) = 1.8726172322000834`
- `steps_range = 0.004682926829268297`, `capture_range = 7.321996223641491e-05`

## 7) Pass/Fail Checklist (E1)

- [x] `rtg1_mode == max_k` (recorded in diagnostics)
- [ ] `spearman(lambda_c, steps_used_mean) <= -0.6`
- [x] `steps_range > 0`
- [x] `capture_range > 0`
- [x] `max(action_change_rate_vs_ref) >= 0.05`
- [x] `max(logits_kl_mean_vs_ref) > 0`

## 8) Interpretation (REQUIRED)

- FAIL: After RTG1 alignment, the steps trend remains **positive** and is nearly saturated near `K_max`.
- The tiny `steps_range` and `capture_range` indicate the model is effectively running to `K_max` for all `lambda_c`.
- This implies RTG1 mismatch is **not sufficient** to fix the inversion; other factors (exposure bias or RTG0 direction) dominate.

## 9) Next Steps (REQUIRED)

- Proceed to **E2 (teacher-forced eval)** to isolate exposure bias effects.
