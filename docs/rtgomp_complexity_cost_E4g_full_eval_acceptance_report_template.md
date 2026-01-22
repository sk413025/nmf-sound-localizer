# Acceptance Report Template: E4g — Full (Non-Smoke) Evaluation on a Larger Real Subset

Use this template for E4g. Fill all placeholders. Keep everything in English.

Save the filled report to:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/ACCEPTANCE_REPORT.md`

---

# Acceptance Report: E4g — Full (Non-Smoke) Evaluation on a Larger Real Subset

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/`
- Outcome: `PASS` / `FAIL`
- Purpose: Validate E4-series lambda-cost controllability and baselines beyond the smoke subset (`num_pairs=3`).

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device(s): (record what each script used; e.g., `mps` for lambda-grid, `cpu` for OMP-vs-Random)
- Checkpoint: `<path>`
- Subset manifest: `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/subset_manifest.json`
  - `num_pairs = <N>`
  - `fingerprint_md5 = <hash>`
- Data roots (from manifest):
  - `mic_root = <path>`
  - `ldv_root = <path>`
- Fixed params:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`
  - `rtg_dim=2`, `use_stop_action=true`
- Lambda grid:
  - `lambda_c_values = <list>`
- E4e prerequisite:
  - Confirmed `teacher_forced` semantics and `steps_used_mean` indexing match E4e spec: `YES/NO`

## 3) Exact Commands (REQUIRED)

Paste the exact commands executed from:
- `docs/rtgomp_complexity_cost_E4g_full_eval_spec.md`

Include the resolved values for:
- `RUN_DIR`, `NUM_PAIRS`, `LAMBDA_LIST`, `SEED_OMP_RANDOM`

## 4) Results (REQUIRED)

### 4.1 A) OMP vs Random (larger subset)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/A_verify_omp_superiority/run.log`
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/summary/omp_vs_random_k_sweep.json`

Table summary (paste key rows):
- K=1: OMP=`...` Random=`...` Gap=`...`
- K=2: ...
- K=4: ...
- K=8: ...
- K=16: ...

Decision (A):
- PASS if: all gaps > 0 and gap(K=16) >= 0.05
- A = `PASS/FAIL`

### 4.2 B) Free vs Teacher-Forced (E4e semantics)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/B_free/lambda_grid.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/B_teacher_forced/lambda_grid.json`

Report:
- Free: `spearman(lambda_c, steps_used_mean) = ...`
- Teacher-forced: `spearman(lambda_c, steps_used_mean) = ...`
- Low-penalty consistency at `lambda_c=min`:
  - `capture_free = ...`
  - `capture_teacher_forced = ...`
  - `Δcapture = free - teacher_forced = ...`

Decision (B):
- PASS if both spearman values <= -0.6
- B = `PASS/FAIL`

### 4.3 C) Lambda-grid acceptance (free rollout)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/C_free/lambda_grid.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/C_free/acceptance_check.json`

Report the key numeric checks:
- `spearman(lambda_c, steps_used_mean) = ...` (target <= -0.6)
- `steps_range = ...` (target >= 0.10)
- `capture_range = ...` (target >= 0.001)
- `max(action_change_rate_vs_ref) = ...` (target >= 0.05)
- `max(logits_kl_mean_vs_ref) = ...` (target > 0)

Decision (C):
- C = `PASS/FAIL`

### 4.4 Baseline: OMP vs DTmin vs Random (K-sweep; stop-action)

Artifacts:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/baseline_dt_vs_omp_stop_action/eval_stats.pt`
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/summary/dt_vs_omp_k_sweep.json`
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/summary/omp_dtmin_random_k_sweep.json`

Paste key rows and highlight whether `DT - Random` becomes negative at higher K.

This baseline is descriptive (not a PASS gate), but MUST be analyzed causally.

## 5) Acceptance Decision (REQUIRED)

- A: `PASS/FAIL`
- B: `PASS/FAIL`
- C: `PASS/FAIL`
- Overall (E4g): `PASS` if A+B+C pass, else `FAIL`

## 6) Interpretation (REQUIRED; causal language)

- A interpretation:
  - Explain why OMP should beat Random BECAUSE of the least-squares/active-set property.
- B interpretation:
  - If B fails, explain whether it fails DUE TO semantics, STOP conditioning, or residual-evolution mismatch.
  - If B passes, explain what it implies about STOP controllability under teacher residual evolution.
- C interpretation:
  - Explain why negative spearman implies controllability and what the observed ranges imply.
- Baseline interpretation:
  - If `DT - Random` becomes negative at high K, explain it BECAUSE of early STOP truncation or policy incentives.

## 7) Next Steps (REQUIRED)

Concrete follow-ups based on PASS/FAIL and the baseline table.

