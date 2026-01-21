# Acceptance Report Template: E4d — Teacher-Forced Mask Fix + B Rerun

Use this template for the E4d rerun after fixing teacher-forced uniqueness masking. Fill all placeholders. Keep everything in English.

---

# Acceptance Report: E4d — Teacher-Forced Mask Fix + B Rerun

## 1) Executive Summary

- Run: `results/<run_name>/`
- Outcome: `<PASS/FAIL>`
- Compared modes:
  - Free: `B_free/lambda_grid.json`
  - Teacher-forced: `B_teacher_forced/lambda_grid.json`

Key acceptance numbers:
- Teacher-forced @ `lambda_c=1e-4`: `final_capture_mean = <value>` (target ≥ 0.98)
- Teacher-forced monotonicity: `spearman(lambda_c, steps_used_mean) = <value>` (target ≤ -0.6)
- Consistency @ `lambda_c=1e-4`: `capture_free - capture_teacher_forced = <value>` (target ≥ -0.02)

## 2) Experiment Context (REQUIRED)

- Background: E4c extval run `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/` passed A and C but B was not trustworthy because teacher-forced capture collapsed (~0.40), consistent with duplicate-lag selection in teacher mode.\n- Motivation: Fix evaluation correctness so B can be used as a meaningful student-vs-teacher comparison.\n- Purpose: Verify that enforcing teacher uniqueness masking removes the capture collapse and restores sensible lambda-dependent behavior.\n- Expected: Teacher-forced results become comparable to free rollout (high capture at low penalty) BECAUSE duplicate selections are prevented.\n+
## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/<run_name>/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Fixed params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
- Lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`

## 4) Exact Commands (REQUIRED)

Record the exact commands executed (from the E4d spec), including `conda run -n trl-training` and `PYTHONPATH=.`.

## 5) Results (REQUIRED)

### 5.1 Free rollout summary

From `B_free/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_range = <value>`
- `capture_range = <value>`
- `capture(lambda_c=1e-4) = <value>`

### 5.2 Teacher-forced summary

From `B_teacher_forced/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_range = <value>`
- `capture_range = <value>`
- `capture(lambda_c=1e-4) = <value>`

## 6) Acceptance Decision (REQUIRED)

- Teacher-forced capture not collapsed: `<PASS/FAIL>` (≥ 0.98 at `lambda_c=1e-4`)
- Teacher-forced monotonicity: `<PASS/FAIL>` (≤ -0.6)
- Free vs teacher-forced consistency: `<PASS/FAIL>` (free ≥ teacher-forced − 0.02 at `lambda_c=1e-4`)
- Overall: `<PASS/FAIL>`

## 7) Interpretation (REQUIRED; causal language)

- Explain why the new behavior is correct BECAUSE duplicates are prevented and least-squares projections behave predictably with unique atoms.\n- If FAIL, explain what remains broken DUE TO either masking, stop-step indexing, or mismatch between student STOP and teacher STOP semantics.\n+
## 8) Next Step (REQUIRED)

Choose exactly one:
- Fix step indexing to be strictly “number of selected atoms” (avoid 0/1-based mixing), OR\n- Extend B acceptance to use `teacher_steps_mean` directly for monotonicity if it better matches the intended semantics.\n
