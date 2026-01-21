# Acceptance Report: E4c — Extended Validation Suite (A/B/C)

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247/`
- Validated checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Validated subset manifest: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/subset_manifest.json`
- Subset fingerprint: `fingerprint_md5=668135f8f6f7baaf99dffeef4cbb1a21` (expected `668135f8f6f7baaf99dffeef4cbb1a21`)
- Outcome:
  - A (OMP vs Random): PASS
  - B (Free vs Teacher-Forced): FAIL
  - C (Lambda-grid acceptance): PASS
  - Overall: FAIL

## 2) Experiment Context (REQUIRED)

- Background: E4c passes the core lambda-grid acceptance after fixing STOP-state supervision alignment.
- Motivation: We need A/B/C validation to ensure (i) teacher physics sanity, (ii) student vs teacher gap is not pathological, and (iii) RTG0 control is truly used and non-degenerate, so results are comparable to earlier RTG/DT evaluations.
- Purpose: Validate E4c under the same subset and fixed physics constraints with a broader acceptance suite.
- Expected:
  - A: OMP dominates a weak Random baseline (with replacement).
  - B: Teacher-forced and free rollouts both show `lambda_c ↑ ⇒ steps ↓`; free capture does not collapse vs teacher-forced at low penalty.
  - C: Reproduces E4c acceptance PASS.

## 3) Setup (REQUIRED)

- Env: `trl-training`
- Device: `mps` (from logs: “Using device: mps”)
- Repo state:
  - `git rev-parse HEAD = 7401bc058ebef0422ea3f9350b5363675bd73e62`
  - Dirty state: `dirty` DUE TO untracked files (no tracked-file diffs). Untracked: `.rtgomp_E4c_extval_ABC_env.sh`, `.rtgomp_E4c_extval_ABC_run_dir`, and multiple `docs/*` files listed by `git status --porcelain` at run time.
- Data roots:
  - `mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - `ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset:
  - `selection = first 3 clip pairs in dataset order (all angles)`
  - `num_pairs = 3`
  - `fingerprint_md5 = 668135f8f6f7baaf99dffeef4cbb1a21`
- Fixed parameters:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `K_max=16`, `gain=100.0`, `freq_min=0`, `freq_max=8000`
- Lambda grid:
  - `lambda_c_values = 1e-4,3e-4,1e-3,3e-3,1e-2`

## 4) Exact Commands (REQUIRED)

```bash
export PYTHONPATH=.
RUN_DIR="results/rtgomp_lambda_cost_E4c_extval_ABC_20260121_093247"
CKPT="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth"
MANIFEST="results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/subset_manifest.json"
LAMBDA_LIST="1e-4,3e-4,1e-3,3e-3,1e-2"
MIC_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC"
LDV_ROOT="/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV"

# A) OMP vs Random (teacher sanity)
OUT_DIR="$RUN_DIR/A_verify_omp_superiority"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
conda run -n trl-training python -u verify_omp_superiority.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --all_angles 2>&1 | tee -a "$OUT_DIR/run.log"

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
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
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
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode teacher_forced --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

# C) Acceptance check (free rollout)
OUT_DIR="$RUN_DIR/C_free"
mkdir -p "$OUT_DIR"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then
  echo "ERROR: lock exists ($LOCKDIR). Another run is using OUT_DIR=$OUT_DIR" >&2
  exit 1
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
conda run -n trl-training python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root "$MIC_ROOT" \
  --ldv_root "$LDV_ROOT" \
  --ckpt_path "$CKPT" \
  --subset_manifest "$MANIFEST" \
  --out_dir "$OUT_DIR" \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 --use_stop_action \
  --rollout_mode free --teacher_min_k 1 \
  --lambda_c_values "$LAMBDA_LIST" 2>&1 | tee -a "$OUT_DIR/run.log"

conda run -n trl-training python -u scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid "$OUT_DIR/lambda_grid.json" \
  --out_json "$OUT_DIR/acceptance_check.json" 2>&1 | tee -a "$OUT_DIR/run.log"
```

## 5) Results (REQUIRED)

### 5.1 A) OMP vs Random

From `A_verify_omp_superiority/run.log`:

```
K     | OMP (W%)     | Rnd (W%)     | Gap          | Ratio(O/R)
1     | 0.9140       | 0.1736       | +0.7404       | 5.27x
2     | 0.9565       | 0.2989       | +0.6576       | 3.20x
4     | 0.9767       | 0.5628       | +0.4139       | 1.74x
8     | 0.9914       | 0.7951       | +0.1964       | 1.25x
16    | 0.9989       | 0.8634       | +0.1355       | 1.16x
```

- `gap_min = 0.1355` over K∈{1,2,4,8,16}
- `gap_at_K16 = 0.1355`
- Decision:
  - A = PASS because all gaps are positive AND `gap_at_K16 >= 0.05`.

### 5.2 B) Free vs Teacher-Forced

Artifacts:
- `B_free/lambda_grid.json`
- `B_teacher_forced/lambda_grid.json`

Free rollout:
- `spearman(lambda_c, steps_used_mean) = -0.9000`
- `steps_range = 0.5195447154471555`
- `capture_range = 0.007347841468283822`

Teacher-forced rollout:
- `spearman(lambda_c, steps_used_mean) = +0.7000`
- `steps_range = 0.6848130081300816`
- `capture_range = 0.010484589844215186`
- `max(student_stop_at_teacher_rate) = 0.9946666666666667`
- `max(student_stop_before_teacher_rate) = 0.002991869918699187`

Compare capture at low lambda:
- `capture_free(lambda_min=1e-4) = 0.996407158549239`
- `capture_teacher_forced(lambda_min=1e-4) = 0.40827099972236447`
- `Δcapture = free - teacher_forced = 0.5881361588268745`

Decision:
- B = FAIL because teacher-forced monotonicity fails (`spearman = +0.7`), which violates the required `spearman <= -0.6` criterion, EVEN THOUGH free rollout monotonicity passes.

### 5.3 C) Lambda-grid acceptance

From `C_free/acceptance_check.json`:
- `spearman(lambda_c, steps_used_mean) = -0.9` (target ≤ -0.6)
- `steps_range = 0.5195447154471555` (target ≥ 0.10)
- `capture_range = 0.007347841468283822` (target ≥ 0.001)
- `max(action_change_rate_vs_ref) = 0.38297421097421097` (target ≥ 0.05)
- `max(logits_kl_mean_vs_ref) = 3.7426251125753702` (target > 0)

Decision:
- C = PASS because all thresholds are met.

## 6) Interpretation (REQUIRED; causal language)

### 6.1 Physical / mathematical interpretation

- OMP capture increases with active-set growth BECAUSE least-squares projection onto a superset of atoms cannot increase the residual norm; THEREFORE capture is expected to be monotone for an exact OMP solver.
- STOP timing should shift with lambda cost BECAUSE the policy trades marginal capture gain against penalty; THEREFORE higher lambda should yield fewer steps when the STOP decision is conditioned on the same state features used for non-STOP actions.

### 6.2 Cross-check interpretation

- A: OMP dominates Random BECAUSE the dictionary projection with greedy selection aligns with the least-squares objective, DUE TO OMP adding atoms that minimize residual energy at each step.
- B: Teacher-forced monotonicity fails (positive correlation) DUE TO the STOP policy increasing steps as lambda grows, which implies the RTG0 signal is either inverted or not integrated consistently in teacher-forced mode; THEREFORE student STOP behavior is misaligned with the penalty interpretation even when residual evolution is fixed by the teacher.
- C: RTG0 controllability is demonstrated in the free rollout BECAUSE action_change_rate and logits_kl increase across lambda values; THEREFORE the policy responds to RTG0 in free mode despite the teacher-forced anomaly.

## 7) Conclusion and Next Steps (REQUIRED)

- Overall outcome: FAIL (A PASS, B FAIL, C PASS).
- Next experiment: Run a teacher-forced sweep that logs STOP logits vs lambda and checks the sign of the RTG0 embedding BECAUSE the observed `lambda_c ↑ ⇒ steps ↑` pattern suggests an inverted or mis-scaled RTG0 pathway; THEREFORE add a minimal diagnostic run (same manifest, same ckpt) that prints mean STOP logit per lambda and correlates it with lambda to confirm or refute a sign inversion.
