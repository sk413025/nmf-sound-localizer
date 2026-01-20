# Plan: E4 DAgger-lite for RTG-OMP (Exposure-Bias Fix)

This plan defines the **E4 experiment** to mitigate exposure bias by augmenting training with **student-visited states** labeled by the **penalty-OMP teacher** (“DAgger-lite”).

Goal: make **free-rollout** monotonicity correct:

```
lambda_c ↑  ⇒  steps_used_mean ↓   (target Spearman ρ <= -0.6)
```

Primary evidence motivating E4:
- Free rollout fails (E1): `rho = +1.0`, steps saturate near K_max.
- Teacher-forced eval passes (E2): `rho = -1.0`.
Therefore, the dominant failure mode is **exposure bias / state drift**.

---

## 1) Hypothesis & Rationale

**Hypothesis**: The student learns STOP on teacher states but fails when the state distribution shifts during free rollout.  
**Rationale**: E2 shows monotonicity is correct when residuals are updated by the teacher.

**DAgger-lite** will reduce this mismatch by:
1) running the student policy to collect its own states,
2) labeling those states with teacher actions (including STOP),
3) retraining with the augmented dataset.

This is a minimal intervention (1 iteration) to test exposure-bias mitigation without full RL.

---

## 2) Experiment Design

### 2.1 Baseline for comparison
Use the latest E1 checkpoint and dataset configuration:
- teacher trajectories: penalty-OMP
- RTG1: max_k aligned
- K_max=16, Tw=32, max_lag=50, gain=100

### 2.2 DAgger-lite iteration (single round)

**Phase A: Student rollout (data collection)**
- Run the student on a small real subset, for each `lambda_c` in the sweep.
- Collect per-step states:
  - correlation state `|D^H r|`
  - step index `k`
  - frequency id
  - current residual `r` (optional)

**Phase B: Teacher labeling**
- For each collected state, compute the teacher’s greedy action and STOP decision using the penalty-OMP rule:
  - choose best lag on the *current* residual
  - compute `ΔE` and apply `lambda_abs` threshold
  - if STOP: label STOP token

**Phase C: Dataset merge**
- Merge the DAgger samples with the original teacher trajectories:
  - `train_data = teacher_data + dagger_data`
  - track and log the ratio of DAgger samples

**Phase D: Retrain**
- Retrain the student using the merged dataset.
- Evaluate with free rollout on the same lambda sweep.

---

## 3) Acceptance Criteria

E4 is **PASS** if ALL are true on the evaluation subset:

1) **Free-rollout monotonicity (primary)**
   - `spearman(lambda_c, steps_used_mean) <= -0.6`

2) **Trade-off exists**
   - `steps_range > 0`
   - `capture_range > 0`

3) **RTG sensitivity**
   - `max(action_change_rate_vs_ref) >= 0.05`
   - `max(logits_kl_mean_vs_ref) > 0`

4) **No catastrophic regression**
   - Free-rollout capture must not collapse to ~0 (set a floor based on baseline).

---

## 4) Required Artifacts

Every E4 run MUST include:

- `results/<run>/run.log`
- `results/<run>/subset_manifest.json`
- `results/<run>/code_state.json`
- `results/<run>/data/lag_trajectories.pt` (teacher data)
- `results/<run>/data/dagger_trajectories.pt` (new DAgger data)
- `results/<run>/data/merged_trajectories.pt`
- `results/<run>/model/dt_freq_aware_best.pth`
- `results/<run>/train/diagnostics.json`
- `results/<run>/eval/lambda_grid.json`
- `results/<run>/eval/acceptance_check.json`
- `results/<run>/ACCEPTANCE_REPORT.md`

---

## 5) Minimal Implementation Notes

Suggested minimal new script (or new mode in eval script):
- `scripts/h_exploration/run_dagger_collect.py`
  - input: model ckpt, dataset, lambda sweep
  - output: `dagger_trajectories.pt`

Label format should match the existing training dataset:
- `corrs`: `(F, K, M)`
- `actions`: `(F, K)`
- `lambda_c`, `valid_len`, `E0`, `E_res`, `deltaE` (if available)

If adding a new script is too heavy, extend `run_lambda_override_grid_eval.py` with a `--dump_dagger` mode.

---

## 6) Risks / Failure Modes

- **Teacher label mismatch**: if teacher uses different residual than student state, labels are invalid.
  - Fix: recompute residual from student’s selected atoms before labeling.
- **Dataset imbalance**: DAgger samples dominate and wash out teacher data.
  - Fix: cap DAgger samples per clip, or downsample DAgger to a fixed ratio.
- **Compute cost**: labeling requires LS; keep subset small for smoke validation.

---

## 7) Next Step If E4 Fails

If free-rollout monotonicity is still inverted:
- test RTG0 direction flip (E3)
- consider teacher-forced training (not just eval)
- consider STOP-loss reweighting

