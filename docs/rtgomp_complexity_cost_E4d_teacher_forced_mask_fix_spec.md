# Spec: E4d — Teacher-Forced Uniqueness Mask Fix + B Rerun

This spec defines the required behavior and reproduction steps for E4d.

Dependencies:
- `docs/rtgomp_complexity_cost_E4c_extended_validation_ABC_spec.md`
- `docs/rtgomp_complexity_cost_E4d_teacher_forced_mask_fix_plan.md`

---

## 1) Required Code Change (minimal)

File:
- `scripts/h_exploration/run_lambda_override_grid_eval.py`

Requirement:
- In `teacher_forced` mode, after `best_teacher` is selected for step `k`, enforce uniqueness:
  - update `mask_teacher` so `best_teacher[f]` is masked for subsequent teacher steps at frequency `f`.

Prohibited in E4d:
- changing the model checkpoint, dataset, lambda grid, or free-rollout logic
- adding silent fallbacks/coercions
- changing metrics without explicitly versioning them

---

## 2) Fixed Inputs (must match E4c extval)

- Checkpoint:
  - `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest (copied into the run directory):
  - `fingerprint_md5 == 668135f8f6f7baaf99dffeef4cbb1a21`
  - use `mic_root` / `ldv_root` exactly as recorded in the manifest
- Parameters:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`
  - `rtg_dim=2`, `use_stop_action=true`
  - `lambda_c_values=1e-4,3e-4,1e-3,3e-3,1e-2`

---

## 3) Execution (B rerun only)

Use a new run directory:
- `RUN_DIR="results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_<timestamp>"`

Run with per-step locks and append-only logs (same robustness rules as E4c extval spec).

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4d_teacher_forced_mask_fix_<timestamp>"
mkdir -p "$RUN_DIR"

CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MANIFEST_SRC="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/subset_manifest.json"
cp -f "$MANIFEST_SRC" "$RUN_DIR/subset_manifest.json"
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

Required artifacts:
- `results/<run>/B_free/lambda_grid.json`
- `results/<run>/B_teacher_forced/lambda_grid.json`
- both run logs under the same folders
- a filled acceptance report for E4d

---

## 4) Acceptance (E4d)

PASS if all:
- Teacher-forced capture is no longer collapsed:
  - at `lambda_c=1e-4`, `final_capture_mean_teacher_forced >= 0.98`
- Teacher-forced monotonicity direction matches:
  - `spearman(lambda_c, steps_used_mean_teacher_forced) <= -0.6`
- Free vs teacher-forced low-penalty consistency:
  - at `lambda_c=1e-4`, `capture_free >= capture_teacher_forced - 0.02`

FAIL if any fail.

