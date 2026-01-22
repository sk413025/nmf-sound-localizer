# Plan + Spec: Fixing Inverted `lambda_c → steps_used` in RTG-OMP (Complexity Cost)

This document is a self-contained engineering plan/spec to diagnose and fix the **inverted monotonicity** observed in the RTG-OMP “complexity cost” prototype:

- Expected (teacher / physics): higher `lambda_c` (higher complexity penalty) ⇒ **fewer steps** (earlier STOP) and typically **lower capture**.
- Observed (student eval rollout): higher `lambda_c` ⇒ **more steps** and **higher capture** (inverted trend).

Primary evidence comes from the executed smoke run:
- Run artifacts: `results/rtgomp_lambda_cost_smoke_20260120_175600/`
- Acceptance summary: `results/rtgomp_lambda_cost_smoke_20260120_175600/ACCEPTANCE_REPORT.md`
- Eval grid JSON: `results/rtgomp_lambda_cost_smoke_20260120_175600/eval/lambda_grid.json`
- Trajectories: `results/rtgomp_lambda_cost_smoke_20260120_175600/data/lag_trajectories.pt`

Canonical spec reference:
- `docs/rtgomp_complexity_cost_spec.md`

---

## 0) System snapshot (minimal context)

### 0.1 Problem and framing
Per frequency bin `f`, for each window of `Tw` STFT frames we reconstruct LDV STFT `Y_f ∈ C^{Tw}` from a **lag dictionary** built from microphone STFT `X`:

```
Y_f ≈ D_f a_f,  where D_f ∈ C^{Tw×M}, M = 2*max_lag+1
```

We choose up to `K_max` atoms (lags) and solve least squares at each step to reduce residual energy.

### 0.2 Teacher: penalty-OMP (RTG-OMP teacher)
Teacher policy is a greedy sparse selection with an explicit **complexity cost**.

At each step `k`, for each frequency `f`:
- Compute the marginal energy improvement `ΔE_f(k) = E_f(k-1) - E_f(k)` after adding the best next atom (greedy).
- Define a per-frequency **absolute penalty threshold**:
  - `lambda_abs_f = lambda_c * E0_f` where `E0_f` is initial energy in that window.
- If `k >= min_k` and `ΔE_f(k) <= lambda_abs_f`, teacher emits STOP at that frequency and ends the sequence.

Implemented in:
- `scripts/h_exploration/generate_lag_omp.py` (`run_penalty_omp_lag_capture`)

Trajectory fields (per window):
- `corrs`: `(F, K_max, M)` (correlation states)
- `actions`: `(F, K_max)` (selected lag id per step; STOP not stored here)
- `valid_len`: `(F,)` number of steps before STOP (teacher’s stopping time)
- `lambda_c`: `(F,)` constant per window, broadcast across freqs
- plus energy diagnostics: `E0`, `E_res`, `deltaE`, `lambda_abs`

### 0.3 Student: DT/GRU imitation with STOP action
Student model is a GRU sequence policy that consumes:
- state: correlation profile `|D^H r|` (size `M`)
- RTG: currently `rtg_dim=2`
  - `rtg0`: normalized `log10(lambda_c)` (higher means higher penalty)
  - `rtg1`: remaining-steps fraction (see mismatch section)
- freq id embedding

Student predicts a categorical action over:
- lag ids `0..M-1`
- optional STOP action id `M` (enabled in this prototype)

Training script:
- `scripts/h_exploration/train_dt_lag_seq_rtg.py`

Eval script:
- `scripts/h_exploration/run_lambda_override_grid_eval.py`

---

## 1) Problem statement (what failed)

### 1.1 Acceptance failure mode
In the smoke run, eval reports:
- `lambda_c ↑  ⇒  steps_used_mean ↑` (Spearman ρ = +1.0)
- `lambda_c ↑  ⇒  final_capture_mean ↑` (Spearman ρ = +1.0)

See:
- `results/rtgomp_lambda_cost_smoke_20260120_175600/eval/acceptance_check.json`
- `results/rtgomp_lambda_cost_smoke_20260120_175600/eval/lambda_grid.json`

Yet the teacher monotonicity from trajectories is correct:
- `lambda_c ↑ ⇒ valid_len_mean ↓` (Spearman ρ = -1.0)

### 1.2 Minimal factual observations (must remain true in follow-ups)

These are the key “ground truths” we should preserve (or re-check) after any modification:

1) **Teacher is monotone** on the same dataset/windowing:
   - We must re-check `valid_len` vs `lambda_c` after code changes.

2) **Student uses RTG at all**:
   - The smoke run already shows non-trivial action/logit shifts across lambda.
   - We must keep (or improve) measurability.

3) **The inversion happens during free rollout**, not necessarily on teacher states:
   - This is the leading hypothesis (see Section 2).

---

## 2) Evidence-backed hypotheses (why we think this happened)

This section is designed to be falsifiable and tied to concrete code/data.

### H1) RTG1 semantics mismatch (train vs eval) breaks STOP behavior for short sequences

**Code fact**
- Training (`rtg_mode=lambda_cost`) constructs RTG1 using *sequence length*:
  - `remaining_steps_fraction = (seq_len - t) / seq_len`
  - where `seq_len = valid_len + 1` if STOP token is appended.
- Eval constructs RTG1 using *fixed* `max_k`:
  - `remaining_steps_fraction = (max_k - k) / max_k`

This mismatch is visible in:
- `scripts/h_exploration/train_dt_lag_seq_rtg.py` (lambda_cost branch)
- `scripts/h_exploration/run_lambda_override_grid_eval.py` (inside `simulate_for_lambda`)

**Why it matters**
- High `lambda_c` yields short teacher sequences (`valid_len` small), so `seq_len` is small.
- Therefore RTG1 values seen in training for high `lambda_c` differ significantly from eval RTG1 values (distribution shift).

**Falsifiable prediction**
- If we evaluate the trained model on teacher states (not free rollout) but feed eval-style RTG1, STOP accuracy and/or STOP rate will degrade most strongly at high `lambda_c`.

**Status**
- This degradation has already been observed in a quick diagnostic (teacher-state STOP accuracy drops for large `lambda_c` under eval-style RTG1).

### H2) Free-rollout distribution shift (exposure bias) causes STOP to fail beyond teacher state manifold

**Code fact**
- Training is pure imitation (cross-entropy on teacher actions / STOP) and does not train on student-visited states.
- Eval is autoregressive / free-rollout:
  - the residual is updated based on student-chosen atoms, which changes the next correlation state.

**Why it matters**
- If early actions differ from teacher, the state distribution drifts.
- STOP is especially sensitive because it is a “rare” action and depends on subtle conditions near stopping time.
- High `lambda_c` sequences are shorter, so the student sees fewer “late” states for these conditions; drift is more likely to push it outside the trained region.

**Falsifiable prediction**
- Under teacher-forced rollout (use teacher actions to update residual), student’s STOP monotonicity should match teacher.
- Under free rollout, monotonicity will degrade or invert.

### H3) RTG0 directionality ambiguity (“bigger rtg0” interpreted as “higher quality target”)

**Code fact**
- RTG0 uses `rtg0 = normalize(log10(lambda_c))`, thus `lambda_c ↑ ⇒ rtg0 ↑`.
- But `lambda_c ↑` means “more penalty” (prefer simpler models / earlier STOP).
- Many DT/RTG conventions use larger RTG to mean “try harder / reach higher return”.

**Falsifiable prediction**
- If we flip the mapping `rtg0 := 1 - normalize(log10(lambda_c))`, we should see step monotonicity direction flip (all else equal).

---

## 3) Engineering spec for follow-up experiments

### 3.1 Goals (acceptance criteria)
We will consider the RTG-OMP complexity-cost mechanism “working” when ALL are true on a real-data subset:

1) **Teacher monotonicity**:
   - `Spearman rho(lambda_c, teacher_valid_len_mean) <= -0.6`

2) **Student monotonicity under eval** (free rollout):
   - `Spearman rho(lambda_c, student_steps_used_mean) <= -0.6`
   - (or tighter threshold if stable)

3) **Trade-off is present**:
   - `steps_range > 0` AND `capture_range > 0`

4) **RTG sensitivity remains measurable**:
   - `max(action_change_rate_vs_ref) >= 0.05`
   - `max(logits_kl_mean_vs_ref) > 0`

All metrics are computed from:
- `results/<run>/eval/lambda_grid.json`
- `scripts/h_exploration/check_rtgomp_acceptance.py`

### 3.2 Required artifacts per experiment commit (atomic)
Each experiment commit MUST include:
- `results/<run_name>/run.log` (full stdout/stderr)
- `results/<run_name>/subset_manifest.json` (paths + MD5 + fingerprint)
- `results/<run_name>/data/lag_trajectories.pt`
- `results/<run_name>/model/dt_freq_aware_best.pth` (or equivalent)
- `results/<run_name>/train/diagnostics.json`
- `results/<run_name>/eval/lambda_grid.json`
- `results/<run_name>/eval/acceptance_check.json`
- `results/<run_name>/ACCEPTANCE_REPORT.md`

### 3.3 Repro commands (template)
Use the same four-step structure as the smoke run, with a new `run_name`.

---

## 4) Planned experiments (minimal, falsifiable sequence)

### Experiment E1: Align RTG1 semantics (training-side change)

**Question**
Does RTG1 mismatch materially cause inverted or unstable STOP behavior (especially at high `lambda_c`)?

**Change**
Modify training `rtg_mode=lambda_cost` so RTG1 matches eval semantics:

Option A (preferred): store `max_k` in the trajectories and compute:
- `rtg1(t) = (max_k - t) / max_k` for `t=0..seq_len-1`

Option B: add `--max_k` CLI to training and compute the same schedule; ensure it matches data generation.

**Expected outcome**
- Teacher-state STOP accuracy under eval-style RTG1 does not degrade significantly with `lambda_c`.
- Free-rollout `rho(lambda_c, steps_used_mean)` moves toward negative.

**Acceptance**
Run checker; target `rho <= -0.6` for student monotonicity (or at least no longer `+1.0`).

### Experiment E2: Teacher-forced eval mode (diagnostic, not a fix)

**Question**
Is the inversion primarily due to free-rollout state distribution shift?

**Change**
Add a new eval mode in `run_lambda_override_grid_eval.py`:
- `--rollout_mode {free, teacher_forced}`

Where `teacher_forced`:
- updates residual using teacher actions (from penalty-OMP recomputed online, or read from saved trajectories keyed by window)
- but still records student logits/actions/STOP predictions

**Expected outcome**
- In `teacher_forced`, student monotonicity matches teacher (negative rho).
- In `free`, monotonicity is worse/inverted.

This experiment proves whether to invest in DAgger-style data augmentation.

### Experiment E3: RTG0 direction flip (semantic alignment)

**Question**
Is the sign inversion because RTG0 direction conflicts with learned RTG conventions?

**Change**
Flip RTG0 mapping consistently in both training and eval:
- `rtg0 := 1 - normalize(log10(lambda_c))`

**Expected outcome**
- If H3 is major, monotonicity should flip direction (toward negative).

### Experiment E4: DAgger-lite (if E2 confirms exposure bias)

**Question**
Can we remove the inversion by training on student-visited states labeled by teacher?

**Change**
Implement one iteration:
1) Roll out student for each `lambda_c` on a small subset.
2) For each visited state, compute teacher label (next action / STOP) using penalty-OMP criterion.
3) Append to training set and retrain.

**Expected outcome**
- Free-rollout step monotonicity becomes negative and stable.

---

## 5) Implementation checklist (engineer-ready)

### 5.1 Code touch points
- Training RTG1 (E1):
  - `scripts/h_exploration/train_dt_lag_seq_rtg.py` (lambda_cost branch)
  - Potentially extend trajectory dict in `scripts/h_exploration/generate_lag_omp.py` to include `max_k` and/or `rtg1_mode`.

- Eval rollout modes (E2):
  - `scripts/h_exploration/run_lambda_override_grid_eval.py`

- RTG0 mapping flip (E3):
  - `scripts/h_exploration/train_dt_lag_seq_rtg.py`
  - `scripts/h_exploration/run_lambda_override_grid_eval.py`

- Acceptance:
  - `scripts/h_exploration/check_rtgomp_acceptance.py` (no change expected)

### 5.2 Logging requirements (must not regress)
Per lambda entry in `lambda_grid.json` must include:
- `steps_used_mean`
- `final_capture_mean`
- `action_change_rate_vs_ref`
- `logits_kl_mean_vs_ref`
- `rtg0` (and `rtg1_mode` in config)

Per run:
- record `git_head` and ensure report fingerprint matches manifest fingerprint.

---

## 6) Commit plan (strict “one experiment per commit”)

For each experiment Ei:
1) Make code changes required for Ei only.
2) Execute the full pipeline (generate → train → eval → check).
3) Write/update `results/<run>/ACCEPTANCE_REPORT.md` with:
   - Background/Motivation/Purpose/Expected
   - Actual results (numbers + pass/fail)
   - Interpretation (BECAUSE/THEREFORE)
   - Exact commands (copy/paste)
   - Artifact index
4) `git add` code + `git add -f results/<run>/`
5) `git commit -m "Results: RTG-OMP complexity cost — Ei <short desc> (PASS/FAIL)"` with full body.

No code-only commits and no results-only commits.

