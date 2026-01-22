# Spec: E4e — Teacher-Forced Semantics + Step Indexing Fix (B Rerun)

This spec defines E4e: an evaluation-only semantic cleanup so the B check (free vs teacher-forced) measures “student STOP under teacher residual evolution” with unambiguous step indexing.

Dependencies:
- `docs/rtgomp_complexity_cost_E4e_teacher_forced_semantics_fix_plan.md`
- `docs/rtgomp_complexity_cost_E4c_extended_validation_ABC_spec.md`
- `docs/rtgomp_complexity_cost_E4d_teacher_forced_mask_fix_spec.md`

---

## 1) Required Code Changes (evaluation only; minimal)

File:
- `scripts/h_exploration/run_lambda_override_grid_eval.py`

### 1.1 Teacher-forced semantics

When `--rollout_mode teacher_forced`:
- Teacher must still select lags greedily with uniqueness masking.\n- Teacher must update residuals ONLY for frequencies where the student has not stopped.\n- Student STOP must be the only stop mechanism used to determine `steps_used_mean`.\n- Optional: compute teacher penalty-stop as a diagnostic field only.\n
### 1.2 Step indexing / reporting

Define `steps_used_mean` as:
- the number of selected atoms, integer in `[1, max_k]`.\n
If STOP occurs at loop index `k` (0-based) immediately after choosing an action at step `k`, then:
- `steps_used = k + 1`.\n
Prohibited:
- mixing 0-based indices and “count of selected atoms” in the same reported field.\n
---

## 2) Fixed Inputs (must match E4c extval)

- Checkpoint:
  - `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`\n- Subset manifest:
  - `fingerprint_md5 == 668135f8f6f7baaf99dffeef4cbb1a21`\n- Parameters:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`\n  - `rtg_dim=2`, `use_stop_action=true`\n  - `lambda_c_values=1e-4,3e-4,1e-3,3e-3,1e-2`\n
---

## 3) Execution (B rerun only)

```bash
set -euo pipefail
export PYTHONPATH=.

RUN_DIR="results/rtgomp_lambda_cost_E4e_teacher_forced_semantics_fix_<timestamp>"
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
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"
```

---

## 4) Acceptance (E4e)

PASS if all:
- At `lambda_c=1e-4`:\n  - `final_capture_mean_teacher_forced >= 0.98`\n  - `capture_free - capture_teacher_forced <= 0.02`\n- Teacher-forced STOP controllability:\n  - `spearman(lambda_c, steps_used_mean_teacher_forced) <= -0.6`\n  - `steps_range_teacher_forced >= 0.10`\n- Sanity: reported `steps_used_mean` is in `[1, max_k]` (count of selected atoms).\n
FAIL otherwise.

