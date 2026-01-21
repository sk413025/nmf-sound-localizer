# Spec: E4c — Fix STOP-State Alignment (Lambda-Cost RTG-OMP)

This spec defines the required behavior for the E4c fix: STOP supervision must be aligned with the **real correlation state** at which STOP is decided.

Dependencies:
- `docs/rtgomp_complexity_cost_spec.md`
- `docs/rtgomp_complexity_cost_E4_spec.md`
- `docs/rtgomp_complexity_cost_E4b_spec.md`
- `docs/rtgomp_complexity_cost_E4c_stop_state_fix_plan.md`

---

## 1) Problem Statement

Given trajectories that include `valid_len` (stop step), training currently appends STOP using a dummy all-zero state. This is not representative of free-rollout inference states.

---

## 2) Required Dataset Semantics (lambda_cost mode)

Let:
- `corrs` have shape `(F, K, M)` (student-visited states per step),
- `actions` have shape `(F, K)` (teacher lag labels for the selected steps),
- `valid_len` have shape `(F,)` where `valid_len[f] = L` is the number of selected atoms before stop for frequency `f`,
- `STOP_ID = M` (requires `use_stop_action=true` so action dim is `M+1`).

For each frequency `f`:

### 2.1 If `L < K` (STOP occurs within budget)
- `corr_seq = corrs[f, :L+1]`  (includes the real stop-decision state at index `L`)
- `act_seq  = actions[f, :L] + [STOP_ID]`
- Both sequences have length `L+1`, and the STOP label is aligned to `corrs[f, L]`.

### 2.2 If `L == K` (no STOP within budget)
- Do not append STOP.
- `corr_seq = corrs[f, :K]`
- `act_seq  = actions[f, :K]`

---

## 3) Acceptance Criteria (same as E4b)

PASS if all:
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`
- `capture_range >= 0.001`
- `max(action_change_rate_vs_ref) >= 0.05`
- `max(logits_kl_mean_vs_ref) > 0`

