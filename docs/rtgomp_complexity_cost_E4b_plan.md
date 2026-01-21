# Plan: E4b DAgger-lite (Increase DAgger Ratio + Coverage)

This plan extends E4 to **increase the DAgger signal** and **expand coverage** so the
free-rollout step range grows beyond near-saturation at `K_max`.

E4 result summary (from `results/rtgomp_lambda_cost_E4_daggerlite_20260120_204838/`):
- Monotonicity fixed (Spearman ρ = -0.9), but
- `steps_range` is very small (~0.038), still near `K_max`.

**Hypothesis**: The DAgger signal is too weak (too few DAgger samples, small ratio).
Increasing the DAgger ratio and collecting more clips should widen the step range.

---

## 1) Change from E4

E4b modifies two knobs only:

1) **DAgger ratio**: increase from `1.0` to `3.0` (3 DAgger blocks per teacher block).
2) **DAgger coverage**: collect from `num_clips=3` instead of `1`.
3) **DAgger stride**: use a larger stride (e.g., `stride=128`) to keep runtime tractable while increasing clip coverage.

Everything else remains the same (RTG1 max_k, K_max=16, Tw=32, etc.).

---

## 2) Acceptance Targets

E4b PASS conditions (stricter than E4):

- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`  (target: clearly > E4)
- `capture_range >= 0.001`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

If monotonicity is correct but `steps_range` is still < 0.10, treat as **partial**.

---

## 3) Execution Outline

1) Generate penalty-OMP teacher data.
2) Collect DAgger data:
   - model checkpoint: latest E4/E1 model
   - `num_clips=3`
   - Expect CPU-heavy runtime (least-squares); use `--log_every_windows` for progress visibility.
3) Merge datasets with `dagger_ratio=3.0`.
4) Train on merged data.
5) Free-rollout eval (rollout_mode=free).
6) Acceptance check.

---

## 4) Required Artifacts

Same as E4, plus the merge meta JSON:
- `results/<run>/data/merged_trajectories.meta.json`

---

## 5) If E4b Fails

If monotonicity holds but range is still too small:
- increase `dagger_ratio` to `5.0`, or
- perform a second DAgger iteration using the E4b model as the rollout policy.
