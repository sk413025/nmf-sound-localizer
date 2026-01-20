# Acceptance Report Template: E1 — RTG1 Semantics Alignment

Use this template for **Experiment E1** (align RTG1 semantics between training and eval).  
Fill in all placeholders and keep the causal language (BECAUSE/THEREFORE).

---

# Acceptance Report: E1 — RTG1 Alignment (Complexity Cost)

## 1) Executive Summary

- Run: `<results/run_name>`
- Goal: Align RTG1 semantics between training and eval to remove inverted `lambda_c → steps_used_mean`.
- Outcome: `<PASS/FAIL>`
- Key numbers:
  - `spearman(lambda_c, steps_used_mean) = <value>` (target ≤ -0.6)
  - `steps_range = <value>`, `capture_range = <value>`
  - `max(action_change_rate_vs_ref) = <value>`
  - `max(logits_kl_mean_vs_ref) = <value>`

## 2) Experiment Context (REQUIRED)

- Background: `<what was failing before>`
- Motivation: `<why E1 is the next minimal fix>`
- Purpose: `<what E1 proves>`
- Expected: `<expected monotonicity direction and rationale>`

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
- **RTG1 config (E1)**:
  - `rtg1_mode = max_k`
  - `rtg1_max_k = <value>`

## 4) Exact Commands (REQUIRED)

```bash
# 1) Generate trajectories
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py \
  --mic_root <...> \
  --ldv_root <...> \
  --out_dir results/<run_name>/data \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --variants_per_clip 1 --max_items <...> --all_angles \
  --teacher_mode penalty_omp \
  --lambda_c_values "<...>" \
  --min_k 1 --seed 0

# 2) Train (E1: RTG1 max_k)
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/<run_name>/data/lag_trajectories.pt \
  --out_dir results/<run_name>/model \
  --epochs <...> --batch_size <...> --lr <...> \
  --rtg_dim 2 --rtg_mode lambda_cost \
  --rtg1_mode max_k --rtg1_max_k <K_max> \
  --lambda_c_values "<...>" \
  --use_stop_action --seed 0

# 3) Eval (lambda grid)
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root <...> \
  --ldv_root <...> \
  --ckpt_path results/<run_name>/model/dt_freq_aware_best.pth \
  --subset_manifest results/<run_name>/subset_manifest.json \
  --out_dir results/<run_name>/eval \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --lambda_c_values "<...>" \
  --num_clips <...> --use_stop_action

# 4) Acceptance check
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid results/<run_name>/eval/lambda_grid.json \
  --out_json results/<run_name>/eval/acceptance_check.json
```

## 5) Artifacts (REQUIRED)

- `results/<run_name>/run.log`
- `results/<run_name>/subset_manifest.json`
- `results/<run_name>/code_state.json`
- `results/<run_name>/data/lag_trajectories.pt`
- `results/<run_name>/model/dt_freq_aware_best.pth`
- `results/<run_name>/train/diagnostics.json`
- `results/<run_name>/eval/lambda_grid.json`
- `results/<run_name>/eval/acceptance_check.json`

## 6) Results (REQUIRED)

Report the values from `acceptance_check.json` and `lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_used_mean range = <min> → <max>`
- `final_capture_mean range = <min> → <max>`
- `max(action_change_rate_vs_ref) = <value>`
- `max(logits_kl_mean_vs_ref) = <value>`
- `steps_range = <value>`, `capture_range = <value>`

## 7) Pass/Fail Checklist (E1)

- [ ] `rtg1_mode == max_k` (recorded in train diagnostics)
- [ ] `spearman(lambda_c, steps_used_mean) <= -0.6`
- [ ] `steps_range > 0`
- [ ] `capture_range > 0`
- [ ] `max(action_change_rate_vs_ref) >= 0.05`
- [ ] `max(logits_kl_mean_vs_ref) > 0`

## 8) Interpretation (REQUIRED)

- Explain whether monotonicity is now correct BECAUSE RTG1 semantics are aligned, or still inverted DUE TO other factors.
- If FAIL, state which check failed and hypothesize why.

## 9) Next Steps (REQUIRED)

Choose exactly one next experiment from the follow-up plan and justify it:
- E2: teacher-forced eval (exposure bias test)
- E3: RTG0 direction flip
- E4: DAgger-lite

