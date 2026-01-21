# Acceptance Report: E4c — Extended Validation Suite (A/B/C)

- Run: `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247`
- Outcome: **PARTIAL**
  - A (OMP vs Random): **PASS**
  - B (Free vs Teacher-Forced): **FAIL** (teacher-forced mode appears incorrect)
  - C (Lambda-grid acceptance): **PASS**

## 1) Executive Summary

This run validates the E4c checkpoint on the A/B/C suite using the real dataset subset specified by `subset_manifest.json`.

- A passes: OMP strongly outperforms a weak Random baseline, which is consistent with the geometry of greedy correlation selection + least-squares projection.
- C passes: free-rollout lambda-grid acceptance is non-degenerate and monotone in the expected direction.
- B fails: `teacher_forced` mode produces **implausibly low capture (~0.40)** and the **wrong monotonicity sign**, which is inconsistent with free rollout and strongly suggests a bug in the `teacher_forced` evaluation implementation (not a model failure).

Therefore, the model-level conclusion is: **E4c remains validated (C PASS)**, but the B check cannot be trusted until `teacher_forced` is fixed and rerun.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: `mps` (printed in logs)
- Checkpoint:
  - `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest:
  - `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Environment snapshot:
  - `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/env_info.json`
- Code snapshot:
  - `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/code_state.json`

## 3) Exact Commands (REPRO)

All commands should run from repo root with `PYTHONPATH=.` and `conda run -n trl-training`.

```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247"

CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MANIFEST="$RUN_DIR/subset_manifest.json"

# Read roots from subset manifest (or paste them explicitly)
MIC_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC"
LDV_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV"

LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

# A) OMP vs Random sanity
conda run -n trl-training python -u verify_omp_superiority.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" --all_angles \
  2>&1 | tee -a "$RUN_DIR/A_verify_omp_superiority/run.log"

# B) Free rollout
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$RUN_DIR/B_free" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" \
  2>&1 | tee -a "$RUN_DIR/B_free/run.log"

# B) Teacher-forced rollout
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$RUN_DIR/B_teacher_forced" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" \
  2>&1 | tee -a "$RUN_DIR/B_teacher_forced/run.log"

# C) Acceptance check (free rollout)
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$RUN_DIR/C_free" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" \
  2>&1 | tee -a "$RUN_DIR/C_free/run.log"

conda run -n trl-training python -u scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid "$RUN_DIR/C_free/lambda_grid.json" \
  --out_json "$RUN_DIR/C_free/acceptance_check.json" \
  2>&1 | tee -a "$RUN_DIR/C_free/run.log"
```

## 4) Results

### 4.1 A) OMP vs Random (PASS)

From `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/A_verify_omp_superiority/run.log`:
- K=1 gap: `+0.7404`
- K=16 gap: `+0.1355`

This meets the spec requirement that OMP beats Random for all tested K and remains meaningfully better at K=16.

### 4.2 C) Lambda-grid acceptance (PASS)

From `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/C_free/acceptance_check.json`:
- `overall_pass = true`
- `spearman(lambda_c, steps_used_mean) = -0.9`
- `steps_range = 0.51954`
- `capture_range = 0.00735`
- `max(action_change_rate_vs_ref) = 0.38297`
- `max(logits_kl_mean_vs_ref) = 3.74263`

Interpretation:
- `lambda_c ↑ ⇒ steps_used_mean ↓` and capture drops slightly, which is the expected tradeoff under a higher STOP penalty.
- Action-change and KL shift vs the reference lambda are non-zero, which indicates RTG0 changes the policy distribution measurably.

### 4.3 B) Free vs Teacher-Forced (FAIL: teacher-forced appears wrong)

Artifacts:
- Free: `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/B_free/lambda_grid.json`
- Teacher-forced: `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/B_teacher_forced/lambda_grid.json`

Observed summaries:
- Free: `spearman(lambda, steps) = -0.9`, `steps_range = 0.5195`, `capture ≈ 0.99..0.997`
- Teacher-forced: `spearman(lambda, steps) = +0.7`, `steps ≈ 3.15..3.83`, `capture ≈ 0.398..0.408`

This is implausible:
- Teacher-forced should not collapse capture by ~60% relative to free rollout for the same dictionary physics.
- The monotonicity sign flip indicates the mode is not measuring the intended quantity.

## 5) Interpretation (REQUIRED; causal language)

### 5.1 First principles

- OMP dominates Random BECAUSE greedy selection by correlation is a principled heuristic for reducing residual energy when followed by least-squares projection.
- Under lambda-cost, STOP occurs earlier as `lambda_c` increases BECAUSE the stop rule compares marginal residual reduction to a scaled penalty threshold; therefore higher penalties should reduce the number of selected atoms.

### 5.2 Why B likely fails due to evaluation, not model

`teacher_forced` mode appears to let the teacher repeatedly choose the same lag (duplicate selections) BECAUSE the teacher selection mask is not updated in the current implementation. This would reduce marginal improvements (delta energy) and trigger earlier STOP decisions, producing:
- low capture (~0.40),
- inconsistent step behavior vs lambda,
THEREFORE B cannot be used as a meaningful comparison until the teacher mask is fixed and the suite is rerun.

## 6) Next Steps

1) Fix `teacher_forced` evaluation in `scripts/h_exploration/run_lambda_override_grid_eval.py`:
   - enforce uniqueness by updating `mask_teacher` after selecting `best_teacher`.
2) Rerun only B (free + teacher-forced) into a new run directory and re-evaluate B acceptance.

