# Acceptance Report: E4e — Teacher-Forced Semantics + Step Indexing Fix

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_030214/`
- Outcome: **PASS**
- Key claim: Teacher-forced now measures student STOP under teacher residual evolution with unambiguous step counting.

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device: `mps`
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_030214/subset_manifest.json`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Params: `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
- Lambda grid: `1e-4,3e-4,1e-3,3e-3,1e-2`

## 3) Exact Commands (REQUIRED)

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_20260122_030214"
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

From `B_free/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = -0.70`
- `steps_range = 0.3952`
- `capture(lambda_c=1e-4) = 0.9964`

### 4.2 Teacher-forced (new semantics)

From `B_teacher_forced/lambda_grid.json`:
- `spearman(lambda_c, steps_used_mean) = -0.90`
- `steps_range = 0.3909`
- `capture(lambda_c=1e-4) = 0.9837`
- `steps_used_mean` range: `15.5566..15.9474` (within `[1, 16]`)

### 4.3 Free vs teacher-forced consistency at low penalty

- `capture_free(1e-4) - capture_teacher_forced(1e-4) = 0.0127` (target ≤ 0.02)

## 5) Acceptance Decision (REQUIRED)

- Teacher-forced high capture at low penalty: **PASS** (0.9837 ≥ 0.98 at 1e-4)
- Consistency at low penalty: **PASS** (0.0127 ≤ 0.02)
- Teacher-forced STOP controllability: **PASS** (rho = -0.90 ≤ -0.6 and steps_range = 0.3909 ≥ 0.10)
- Step counting semantics: **PASS** (steps in [1, max_k])
- Overall: **PASS**

## 6) Interpretation (REQUIRED; causal language)

- This semantics is correct BECAUSE student STOP now directly controls which frequencies stop receiving teacher residual updates, and steps are counted as the number of selected atoms rather than 0-based indices; THEREFORE teacher-forced measurements reflect student STOP under teacher residual evolution and meet the acceptance thresholds.
