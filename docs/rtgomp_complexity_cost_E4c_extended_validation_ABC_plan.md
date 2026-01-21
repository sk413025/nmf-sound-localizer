# Plan: E4c — Extended Validation Suite (A/B/C)

E4c passes the core lambda-cost acceptance criteria (monotonic tradeoff under free rollout). This plan adds the **A/B/C validation suite** so E4c can be compared fairly to earlier RTG-OMP/DT experiments and so we can detect “passes acceptance but physics/controls are broken” failure modes.

All validation runs MUST use:
- Real data only (no synthetic).
- The same subset selection and fingerprint as the E4c run being validated.
- Fail-fast behavior (missing roots / fingerprint mismatch => stop).

---

## Background

E4c fixes STOP supervision alignment so STOP is learned from the same correlation state distribution encountered during free rollout. Passing the lambda-grid acceptance proves **RTG0 (lambda-cost) controllability** and **non-degenerate tradeoff**, but it does NOT fully validate:
- the underlying “teacher physics” sanity (OMP vs Random),
- student vs teacher gap under the same dictionary physics,
- whether control signals (RTG inputs) are being used for meaningful behavior changes beyond the minimal acceptance statistics.

---

## Suite Overview (A/B/C)

### A) Teacher Physics Sanity: OMP vs Random
**Motivation**: Ensure the capture metric and OMP dictionary projection behave sensibly; if “random beats OMP” under a weak baseline, something is wrong (metric, masking, projection, or dictionary construction).

**Design**:
- Run `verify_omp_superiority.py` (Random baseline = with replacement).
- Report weighted capture (energy reduction) at K ∈ {1,2,4,8,16}.

**Expected**:
- OMP > Random for all tested K (positive gaps), with the largest gap at small K.

### B) Student Quality vs Teacher Stop Rule: Free vs Teacher-Forced
**Motivation**: Separate “can the model pick lags” from “can the model decide STOP”, and quantify how much free-rollout performance deviates from a teacher-residual evolution.

**Design**:
- Run `scripts/h_exploration/run_lambda_override_grid_eval.py` twice on the same checkpoint/subset:
  - `rollout_mode=free` (student chooses lags + STOP; student updates residual).
  - `rollout_mode=teacher_forced` (teacher updates residual; student only influences stopping).
- Compare capture/steps between the two modes across the same lambda grid.

**Expected**:
- Free rollout shows the same qualitative tradeoff direction as teacher-forced.
- Free capture is close to teacher-forced capture at low lambda (where stopping late is optimal), and lower at high lambda only to the extent that the student stops earlier.

### C) Control Compliance: Lambda-Grid Acceptance (RTG0)
**Motivation**: Ensure RTG0 (lambda-cost) is not ignored and produces a non-degenerate tradeoff under free rollout, and that policies differ meaningfully vs the reference lambda.

**Design**:
- Re-run the standard lambda-grid evaluation and acceptance check:
  - `scripts/h_exploration/run_lambda_override_grid_eval.py` (free rollout)
  - `scripts/h_exploration/check_rtgomp_acceptance.py`

**Expected**:
- Pass the same acceptance thresholds as E4c.

---

## Fixed Factors (Fair Comparison Contract)

These parameters MUST match the validated E4c configuration unless explicitly justified:
- STFT hop length: `hop_length=160`
- Lag range: `max_lag=50` (dictionary size `M=101`)
- Window length: `Tw=32`
- Budget: `K_max=16`
- Gain: `gain=100.0`
- Frequency range: `freq_min=0`, `freq_max=8000`
- Lambda grid: same list as the acceptance run
- Subset: use the same `subset_manifest.json` and require the same `fingerprint_md5`

---

## Required Artifacts (per run)

Use one run directory:
`results/rtgomp_lambda_cost_E4c_extval_ABC_<timestamp>/`

Minimum artifacts:
- `run.log`
- `subset_manifest.json` (copied from the validated E4c run)
- A: `A_verify_omp_superiority/run.log`
- B: `B_free/lambda_grid.json`, `B_teacher_forced/lambda_grid.json`
- C: `C_free/lambda_grid.json`, `C_free/acceptance_check.json`
- `ACCEPTANCE_REPORT.md` (filled from the template in the spec)

