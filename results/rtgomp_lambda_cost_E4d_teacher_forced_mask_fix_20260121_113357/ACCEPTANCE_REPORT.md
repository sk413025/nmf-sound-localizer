# Acceptance Report: E4d — Teacher-Forced Mask Fix + B Rerun

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/`
- Outcome: `FAIL`
- Compared modes:
  - Free: `B_free/lambda_grid.json`
  - Teacher-forced: `B_teacher_forced/lambda_grid.json`

Key acceptance numbers:
- Teacher-forced @ `lambda_c=1e-4`: `final_capture_mean = 0.979127` (target ≥ 0.98)
- Teacher-forced monotonicity: `spearman(lambda_c, steps_used_mean) = -1.000000` (target ≤ -0.6)
- Consistency @ `lambda_c=1e-4`: `capture_free - capture_teacher_forced = 0.017280` (target ≥ -0.02)

## 2) Experiment Context (REQUIRED)

- Background: E4c extval run `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/` passed A and C but B was not trustworthy because teacher-forced capture collapsed (~0.40), consistent with duplicate-lag selection in teacher mode.
- Motivation: Fix evaluation correctness so B can be used as a meaningful student-vs-teacher comparison.
- Purpose: Verify that enforcing teacher uniqueness masking removes the capture collapse and restores sensible lambda-dependent behavior.
- Expected: Teacher-forced results become comparable to free rollout (high capture at low penalty) BECAUSE duplicate selections are prevented.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `mps`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Fixed params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
- Lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`
- Environment snapshot: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/env_info.json`
- Code snapshot: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/code_state.json`

## 4) Exact Commands (REQUIRED)

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MANIFEST="$RUN_DIR/subset_manifest.json"
MIC_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC"
LDV_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV"
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

# B) Free rollout
OUT_DIR="$RUN_DIR/B_free"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

# B) Teacher-forced rollout
OUT_DIR="$RUN_DIR/B_teacher_forced"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"
```

## 5) Results (REQUIRED)

### 5.1 Free rollout summary

From `B_free/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = -0.900000`
- `steps_range = 0.519545`
- `capture_range = 0.007348`
- `capture(lambda_c=1e-4) = 0.996407`

### 5.2 Teacher-forced summary

From `B_teacher_forced/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = -1.000000`
- `steps_range = 8.197919`
- `capture_range = 0.284074`
- `capture(lambda_c=1e-4) = 0.979127`

### 5.3 Smoke test (startup/run check)

This run includes a minimal smoke invocation (1 lambda, 1 clip) to confirm the system starts and writes artifacts:
- Free smoke: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/smoke_free/{run.log,lambda_grid.json}`
- Teacher-forced smoke: `results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_20260121_113357/smoke_teacher_forced/{run.log,lambda_grid.json}`

## 6) Acceptance Decision (REQUIRED)

- Teacher-forced capture not collapsed: `FAIL` (0.979127 < 0.98 at `lambda_c=1e-4`)
- Teacher-forced monotonicity: `PASS` (-1.000000 ≤ -0.6)
- Free vs teacher-forced consistency: `PASS` (0.996407 ≥ 0.979127 − 0.02)
- Overall: `FAIL`

## 7) Interpretation (REQUIRED; causal language)

- The monotonicity direction now matches expectations BECAUSE the teacher path prevents duplicate lag selections and the residual energy decreases predictably with unique atoms.
- The capture collapse is largely corrected compared to E4c, but the low-penalty teacher capture still falls short of the ≥0.98 target DUE TO teacher STOP semantics and residual updates being slightly more conservative than the student path, which reduces final reconstruction even with unique selections.
- Therefore E4d improves correctness but does not fully meet the acceptance bar, implying an additional fix is needed beyond uniqueness masking.

## 8) Next Step (REQUIRED)

- Fix step indexing to be strictly “number of selected atoms” (avoid 0/1-based mixing).
