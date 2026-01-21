# Spec: E4c — Extended Validation Suite (A/B/C)

This spec defines the A/B/C validation suite for **E4c lambda-cost RTG-OMP**. It is intended to make the E4c results comparable to earlier DT/RTG experiments by validating:
- A) teacher physics sanity (OMP vs Random),
- B) student vs teacher stop behavior (free vs teacher-forced),
- C) RTG0 controllability under a lambda grid (acceptance).

Dependencies:
- `docs/rtgomp_complexity_cost_spec.md`
- `docs/rtgomp_complexity_cost_E4_spec.md`
- `docs/rtgomp_complexity_cost_E4c_stop_state_fix_spec.md`
- `docs/rtgomp_complexity_cost_E4c_extended_validation_ABC_plan.md`

---

## 0) Execution Robustness (MUST)

This suite is compute-heavy and logs can be clobbered if the same step is started twice (especially under agent automation).

Required safeguards:
- Run steps strictly sequentially (never concurrently).
- Use per-step directory locks.
- Never truncate logs; always append.

### 0.1 Locking rule (per step)

Before running a step that writes into `OUT_DIR`, acquire an atomic lock:

```bash
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
```

### 0.2 Logging rule

Always use `tee -a` (append), never plain `tee`:

```bash
some_command 2>&1 | tee -a "$OUT_DIR/run.log"
```

---

## 1) Inputs (MUST be explicit)

### 1.1 E4c checkpoint
- Default (expected): `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`

### 1.2 Subset manifest (must match checkpoint evaluation subset)
- Default (expected): `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/subset_manifest.json`
- Fingerprint requirement:
  - `fingerprint_md5 == 668135f8f6f7baaf99dffeef4cbb1a21`

### 1.3 Data roots (must exist; no fallbacks)
- Read from `subset_manifest.json`:
  - `mic_root`
  - `ldv_root`
- If these paths do not exist in the current environment, the run MUST fail fast and the missing prerequisite must be documented.

---

## 2) Canonical Metric Definitions

All A/B/C checks use a per-frequency capture definition derived from least-squares projection:

Let `Y` be the complex STFT target vector (per frequency, length `Tw`) and `Ŷ` be the reconstruction from the selected lag atoms.

- Residual energy: `E_res = ||Y - Ŷ||^2`
- Initial energy: `E0 = ||Y||^2`
- Capture ratio (per frequency): `capture = 1 - E_res / max(E0, eps_energy)`

Expected normal ranges:
- `capture ∈ [0, 1]` (minor numerical overshoots should be treated as a bug; do not silently clamp in the evaluation without reporting)
- `steps_used ∈ [1, K_max]` when STOP is enabled (else always `K_max`)

Physical / mathematical invariants (acceptance-relevant):
- Adding atoms cannot increase the least-squares residual norm; therefore capture should be non-decreasing as the active set grows (up to numerical tolerance).
- STOP must be decided from the same state representation used by non-STOP actions (E4c fix premise).
- The action mask must prevent duplicates; selecting the same lag twice is invalid.

---

## 3) A) OMP vs Random (Teacher Sanity)

### 3.1 Command (required)

```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4c_extval_ABC_<timestamp>"
mkdir -p "$RUN_DIR/A_verify_omp_superiority"
OUT_DIR="$RUN_DIR/A_verify_omp_superiority"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

# Use the same mic_root/ldv_root as in the subset manifest.
# Random baseline: with replacement (weak baseline; OMP should dominate).
conda run -n trl-training python -u verify_omp_superiority.py \
  --mic_root "<mic_root_from_manifest>" \
  --ldv_root "<ldv_root_from_manifest>" \
  --all_angles 2>&1 | tee -a "$OUT_DIR/run.log"
```

### 3.2 Acceptance (A)

Parse the printed table from `A_verify_omp_superiority/run.log`.

PASS (A) if:
- For all reported K in `{1,2,4,8,16}`: `Gap = OMP - Random > 0`.
- At `K=16`: `Gap >= 0.05`.

FAIL (A) if:
- Any `Gap <= 0` (OMP does not beat the weak Random baseline), DUE TO likely metric/projection/masking bugs.

---

## 4) B) Free vs Teacher-Forced (Student vs Teacher Stop Rule)

### 4.1 Commands (required)

Use the same lambda list as E4c acceptance.

```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4c_extval_ABC_<timestamp>"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MANIFEST="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/subset_manifest.json"

LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"

mkdir -p "$RUN_DIR/B_free" "$RUN_DIR/B_teacher_forced"
OUT_DIR="$RUN_DIR/B_free"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

# Free rollout: student chooses lags + STOP; student updates residual.
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "<mic_root_from_manifest>" \
  --ldv_root "<ldv_root_from_manifest>" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free \
  --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

# Teacher-forced: teacher updates residual; student influences stopping (STOP timing).
OUT_DIR="$RUN_DIR/B_teacher_forced"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "<mic_root_from_manifest>" \
  --ldv_root "<ldv_root_from_manifest>" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced \
  --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"
```

Required files:
- `B_free/lambda_grid.json`
- `B_teacher_forced/lambda_grid.json`

### 4.2 Acceptance (B)

Compute summaries from the two JSON files:
- For each `lambda_c`, extract:
  - `steps_used_mean`
  - `final_capture_mean`
  - (teacher-forced only) `teacher_steps_mean`, `student_stop_before_teacher_rate`, `student_stop_at_teacher_rate`

PASS (B) if all:
- Tradeoff direction matches in both modes:
  - `spearman(lambda_c, steps_used_mean) <= -0.6` for `B_free` AND `B_teacher_forced`.
- STOP is non-degenerate in teacher-forced mode:
  - `max(student_stop_at_teacher_rate) >= 0.05` AND `max(student_stop_before_teacher_rate) <= 0.95`.
- Free rollout capture is not catastrophically worse than teacher-forced capture:
  - For the smallest `lambda_c` (lowest penalty): `final_capture_mean_free >= final_capture_mean_teacher_forced - 0.02`.

FAIL (B) if:
- Teacher-forced monotonicity fails (STOP rule not learned even under teacher residual evolution), OR
- Free rollout monotonicity fails while teacher-forced passes (exposure/lag-selection mismatch dominates), OR
- Free capture collapses relative to teacher-forced at low penalty (lag selection broken).

---

## 5) C) Lambda-Grid Acceptance (RTG0 Compliance)

### 5.1 Commands (required)

```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4c_extval_ABC_<timestamp>"
mkdir -p "$RUN_DIR/C_free"
OUT_DIR="$RUN_DIR/C_free"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "<mic_root_from_manifest>" \
  --ldv_root "<ldv_root_from_manifest>" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free \
  --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

conda run -n trl-training python -u scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid "$OUT_DIR/lambda_grid.json" \
  --out_json "$OUT_DIR/acceptance_check.json" 2>&1 | tee -a "$OUT_DIR/run.log"
```

### 5.2 Acceptance (C)

PASS (C) if all thresholds are met (same as E4c):
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`
- `capture_range >= 0.001`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

---

## 6) Reporting

Fill:
- `docs/rtgomp_complexity_cost_E4c_extended_validation_ABC_acceptance_report_template.md`

The filled report MUST:
- Use causal language (“BECAUSE”, “DUE TO”, “THEREFORE”).
- Explicitly record data roots and `fingerprint_md5`.
- Include exact commands and paths to artifacts.

---

## 7) Results Commit Hygiene (when committing)

- `results/` is typically ignored; add artifacts with `git add -f results/<run_name>/`.
- If artifacts are large, store them under Git LFS-tracked patterns (do not add untracked binary blobs outside LFS).
