# Spec: E4 DAgger-lite (Exposure-Bias Mitigation)

This spec defines the required behavior and data schema for **E4 DAgger-lite**.
It is complementary to:
- `docs/rtgomp_complexity_cost_spec.md`
- `docs/rtgomp_complexity_cost_E4_plan.md`

---

## 1) Overview

### 1.1 Objective
Reduce exposure bias by training the student on **student-visited states** labeled by the **penalty-OMP teacher**.

### 1.2 Non-goals
- This is not full RL; no policy gradients.
- This is not a complete DAgger loop; only **one iteration** is required.

---

## 2) Data Collection (DAgger-lite)

### 2.1 Inputs
- Student checkpoint: `--ckpt_path`
- Real data roots: `--mic_root`, `--ldv_root`
- Lambda sweep: `--lambda_c_values`
- Hyperparams: `Tw`, `max_lag`, `max_k`, `gain`, `min_k`

### 2.2 Outputs
Write DAgger data to:

```
results/<run_name>/data/dagger_trajectories.pt
```

### 2.3 Required fields in DAgger dataset
Each block (window) MUST include:
- `corrs`: `float32`, shape `(F, K, M)` (student-visited correlation states)
- `actions`: `int`, shape `(F, K)` (teacher labels; includes STOP where applicable)
- `lambda_c`: `float32`, shape `(F,)`
- `valid_len`: `int`, shape `(F,)` (teacher stop step)

Optional but recommended:
- `E0`, `E_res`, `deltaE`, `lambda_abs` for debug.

### 2.4 Teacher labeling rules
For each student-visited state:
1) Compute greedy atom based on teacher OMP on the **current residual**.
2) Compute `ΔE` after adding that atom.
3) Apply STOP if `k >= min_k` AND `ΔE <= lambda_abs`.
4) If STOP triggers: label STOP token id `M`.

**STOP token**:
- `STOP_ID = M`
- action dimension = `M + 1`

---

## 3) Dataset Merge

### 3.1 Merge rule
Merged training dataset should be written to:

```
results/<run_name>/data/merged_trajectories.pt
```

Merge strategy:
- `merged = teacher_data + dagger_data`
- Optionally cap DAgger samples to a fixed ratio (e.g., 1:1 or 1:2).

### 3.2 Logging
Log:
- `num_teacher_blocks`
- `num_dagger_blocks`
- `dagger_ratio`

---

## 4) Training (E4)

Train on `merged_trajectories.pt` with:
- `rtg_mode = lambda_cost`
- `rtg1_mode = max_k`
- `use_stop_action = true`

Diagnostics must include:
- grad norms (`rtg_embed`, `state_embed`, `freq_embed`)
- RTG ablation losses
- `rtg1_mode` and `rtg1_max_k`

---

## 5) Evaluation

Run **free rollout** eval (not teacher-forced):
- `run_lambda_override_grid_eval.py` (rollout_mode=free)

Compute acceptance metrics with:
- `scripts/h_exploration/check_rtgomp_acceptance.py`

---

## 6) Acceptance Criteria

PASS if all:
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range > 0`
- `capture_range > 0`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

FAIL if monotonicity remains positive or steps saturate at `K_max`.

