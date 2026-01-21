# Plan: E4d — Fix Teacher-Forced Masking and Rerun B (Free vs Teacher-Forced)

E4c passes the core lambda-grid acceptance and the extended validation suite A/C. However, the B check (free vs teacher-forced) is currently not trustworthy because `teacher_forced` outputs show implausibly low capture and the wrong monotonicity sign.

This plan defines a minimal follow-up experiment (E4d): fix the `teacher_forced` evaluation to enforce **unique lag selection** (no duplicates), then rerun only the B suite and re-evaluate B acceptance.

---

## Background

From the executed run:
- `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/C_free/acceptance_check.json` → `overall_pass=true` (E4c remains validated under free rollout).
- `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/B_teacher_forced/lambda_grid.json` shows:
  - capture ≈ 0.40 and inconsistent step monotonicity, which is not physically plausible under the same dictionary physics.

**Hypothesis (root cause)**: the teacher path is selecting duplicate lags BECAUSE `mask_teacher` is not updated in `scripts/h_exploration/run_lambda_override_grid_eval.py` during teacher-forced rollout. Duplicate selections reduce marginal residual reduction, causing premature STOP and depressed capture.

---

## Minimal Change (E4d)

In `scripts/h_exploration/run_lambda_override_grid_eval.py`, in `teacher_forced` mode:
- After selecting `best_teacher`, update `mask_teacher` so the same lag cannot be selected again for that frequency.

No other algorithmic changes are allowed in E4d (keep the fix minimal).

---

## Experiment Design (E4d)

Re-run ONLY the B suite (free + teacher-forced) using:
- the same checkpoint as E4c:
  - `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- the same subset manifest + fingerprint:
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- the same lambda grid and parameters as E4c extval.

Output a new run directory:
- `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_<timestamp>/`

---

## Acceptance Targets (E4d)

E4d PASS if all are true:
- Teacher-forced capture is no longer collapsed:
  - at `lambda_c=1e-4`, `final_capture_mean_teacher_forced >= 0.98`
- Teacher-forced monotonicity matches the expected direction:
  - `spearman(lambda_c, steps_used_mean_teacher_forced) <= -0.6`
- Free vs teacher-forced consistency at low penalty:
  - at `lambda_c=1e-4`, `capture_free >= capture_teacher_forced - 0.02`

If any fail, E4d is FAIL and must not be merged into a “PASS” narrative.

