# Acceptance Report Template: E4e — Teacher-Forced Semantics + Step Indexing Fix

Use this template for E4e (evaluation semantics fix). Fill all placeholders. Keep everything in English.

---

# Acceptance Report: E4e — Teacher-Forced Semantics + Step Indexing Fix

## 1) Executive Summary

- Run: `results/<run_name>/`
- Outcome: `<PASS/FAIL>`
- Key claim: Teacher-forced now measures student STOP under teacher residual evolution with unambiguous step counting.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: `<mps/cpu/etc>`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/<run_name>/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
- Lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`

## 3) Exact Commands (REQUIRED)

Record the exact commands executed (from the E4e spec), including `conda run -n trl-training` and `PYTHONPATH=.`.

## 4) Results (REQUIRED)

### 4.1 Free rollout

From `B_free/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_range = <value>`
- `capture(lambda_c=1e-4) = <value>`

### 4.2 Teacher-forced (new semantics)

From `B_teacher_forced/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = <value>`
- `steps_range = <value>`
- `capture(lambda_c=1e-4) = <value>`
- `steps_used_mean` range: `<min..max>` (must lie within `[1, max_k]`)

### 4.3 Free vs teacher-forced consistency at low penalty

- `capture_free(1e-4) - capture_teacher_forced(1e-4) = <value>` (target ≤ 0.02)

## 5) Acceptance Decision (REQUIRED)

- Teacher-forced high capture at low penalty: `<PASS/FAIL>` (≥ 0.98 at 1e-4)
- Consistency at low penalty: `<PASS/FAIL>` (≤ 0.02 gap)
- Teacher-forced STOP controllability: `<PASS/FAIL>` (rho ≤ -0.6 and steps_range ≥ 0.10)
- Step counting semantics: `<PASS/FAIL>` (steps in [1, max_k])
- Overall: `<PASS/FAIL>`

## 6) Interpretation (REQUIRED; causal language)

- Explain why this semantics is correct BECAUSE student STOP now directly controls which frequencies stop receiving teacher residual updates, and steps are counted as number of selected atoms.\n- If FAIL, explain DUE TO which remaining mismatch (STOP decision point, residual update masking, or off-by-one definitions), and propose a minimal next fix.\n

