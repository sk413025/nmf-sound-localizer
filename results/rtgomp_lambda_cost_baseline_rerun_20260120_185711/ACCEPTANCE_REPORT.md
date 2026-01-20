# Acceptance Report: RTG-OMP (Complexity Cost) — Baseline Re-run

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_baseline_rerun_20260120_185711`
- Purpose: Reproduce the known failure (inverted `lambda_c → steps_used_mean`) **without code changes**, to serve as the baseline comparator for follow-up fixes.
- Outcome: **FAIL (expected)**.

Key eval metrics (from `eval/acceptance_check.json`):
- `spearman(lambda_c, steps_used_mean)` = `1.0` (expected ≤ -0.6)
- `steps_used_mean` range: `12.562927 → 13.063024` (higher `lambda_c` → higher steps; inverted)
- `final_capture_mean` range: `0.986024 → 0.989287` (higher `lambda_c` → higher capture; inverted)
- `max(action_change_rate_vs_ref)` = `0.5411425609620731`
- `max(logits_kl_mean_vs_ref)` = `2.0116001574293207`
- `steps_range` = `0.5000975609756093`, `capture_range` = `0.0032627630233764693`

## 2) Background / Motivation / Purpose / Expected

- Background: The RTG-OMP complexity-cost prototype should produce a trade-off where larger `lambda_c` discourages additional selections (earlier STOP).
- Motivation: The previous smoke run failed due to inverted `lambda_c → steps_used_mean`. We need a clean baseline run to compare against subsequent fixes.
- Purpose: Confirm the failure is reproducible with the current code and settings.
- Expected: **FAIL** reproducing the same inverted monotonicity.

## 3) Setup (Real Data)

- Conda env: `trl-training`
- Device: MPS (see `run.log`)
- Data roots:
  - MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset: `first 3 clip pairs in dataset order (all angles)`
- Fingerprint (manifest): `668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `Tw=32`, `max_lag=50` (`M=101`), `K_max=16`, `gain=100.0`
- Teacher mode: `penalty_omp`
- `lambda_c_values`: `0.0001, 0.0003, 0.001, 0.003, 0.01`
- Seed: `0`

## 4) Exact Commands (Copy/Paste)

```bash
# 1) Generate trajectories
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --out_dir results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/data   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --variants_per_clip 1 --max_items 3 --all_angles   --teacher_mode penalty_omp   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --min_k 1 --seed 0

# 2) Train
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py   --data_path results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/data/lag_trajectories.pt   --out_dir results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/model   --epochs 5 --batch_size 128 --lr 5e-4   --rtg_dim 2 --rtg_mode lambda_cost   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --use_stop_action --seed 0

# 3) Eval (lambda grid)
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/model/dt_freq_aware_best.pth   --subset_manifest results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/subset_manifest.json   --out_dir results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/eval   --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --num_clips 1 --use_stop_action

# 4) Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py   --lambda_grid results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/eval/lambda_grid.json   --out_json results/rtgomp_lambda_cost_baseline_rerun_20260120_185711/eval/acceptance_check.json
```

## 5) Artifacts

- `run.log`
- `subset_manifest.json`
- `code_state.json`
- `data/lag_trajectories.pt`
- `model/dt_freq_aware_best.pth`
- `train/diagnostics.json`
- `eval/lambda_grid.json`
- `eval/acceptance_check.json`

## 6) Interpretation / Analysis

- Observed inversion (`lambda_c ↑ → steps_used_mean ↑`) is reproduced, so the failure is not a one-off noise artifact.
- Follow-up experiments should focus on the hypotheses and fixes in `docs/rtgomp_lambda_cost_followup_plan.md` (RTG1 semantics alignment, teacher-forced eval diagnostic, RTG0 direction, and/or DAgger-lite).
