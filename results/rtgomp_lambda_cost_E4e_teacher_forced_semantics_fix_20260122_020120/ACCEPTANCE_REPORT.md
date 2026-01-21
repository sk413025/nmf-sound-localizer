# Acceptance Report: E4e — Teacher-Forced Semantics + Step Indexing Fix

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_020120/`
- Outcome: **FAIL**
- Key claim: Not validated. The evaluation runs did not complete, so teacher-forced semantics and step counting could not be verified.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: **unknown** (run produced no stdout/stderr before termination)
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_020120/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
- Lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`

## 3) Exact Commands (REQUIRED)

Commands were started exactly as specified, but did not complete. The runs produced no stdout/stderr and were terminated after prolonged runtime.

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_020120"
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

# B) Teacher-forced rollout (new semantics)
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

## 4) Results (REQUIRED)

### 4.1 Free rollout

- `B_free/lambda_grid.json`: **missing** (run did not complete)
- `spearman(lambda_c, steps_used_mean) = N/A`
- `steps_range = N/A`
- `capture(lambda_c=1e-4) = N/A`

### 4.2 Teacher-forced (new semantics)

- `B_teacher_forced/lambda_grid.json`: **missing** (run not executed due to free rollout hang)
- `spearman(lambda_c, steps_used_mean) = N/A`
- `steps_range = N/A`
- `capture(lambda_c=1e-4) = N/A`
- `steps_used_mean` range: `N/A`

### 4.3 Free vs teacher-forced consistency at low penalty

- `capture_free(1e-4) - capture_teacher_forced(1e-4) = N/A`

## 5) Acceptance Decision (REQUIRED)

- Teacher-forced high capture at low penalty: **FAIL** (not measured)
- Consistency at low penalty: **FAIL** (not measured)
- Teacher-forced STOP controllability: **FAIL** (not measured)
- Step counting semantics: **FAIL** (not measured)
- Overall: **FAIL**

## 6) Interpretation (REQUIRED; causal language)

- This evaluation could not validate the new teacher-forced semantics BECAUSE the free rollout did not complete and produced no `lambda_grid.json`, THEREFORE the teacher-forced run was not executed and all acceptance metrics are unavailable.
- The most likely cause is the extremely high per-frequency least-squares workload in `run_lambda_override_grid_eval.py`, which can lead to very long runtimes on real data. A minimal next fix is to add a runtime guardrail (e.g., a max-window cap for eval-only runs) or a documented smaller deterministic subset for evaluation that preserves the same subset fingerprinting rules.
