# Implementation Plan: RTG-OMP (Complexity Cost / Model Selection Penalty)

This document is a self-contained, end-to-end plan: start a new branch, implement RTG as a complexity cost (`+ lambda * |S|`), prove RTG changes teacher behavior, and compare against the existing “RTG is ineffective” baseline with reproducible artifacts.

Key idea:
- Do not just feed RTG into the model. Make RTG enter a physically consistent teacher/solver objective so the data distribution shows `I(action; RTG | state) > 0`.
- Avoid known pitfalls: fixed K (always run to `K_max`) erases RTG effects that only impact stopping; fixed epsilon floors distort low-energy bands.

---

## A. Executive Summary (<= 15 lines)

Goal: Introduce a complexity penalty into the OMP teacher so RTG is redefined as `lambda` (or normalized lambda), and the optimal teacher behavior depends on RTG in the same or similar states. The DT should learn `pi(a | s, RTG)`.

Core method: Replace “fixed-K residual minimization” with a model selection objective:

```
min_{S,w} ||y - D_S w||_2^2 + lambda * |S|
```

Greedy interpretation: At each step, check marginal energy drop `DeltaE`. If `DeltaE < lambda`, STOP. Therefore `lambda` directly controls “Is adding another path physically worth it?”

Success criteria (same data, same major hyperparams):
1) Teacher behavior shows monotonic dependence: `lambda` up -> `steps_used` down.
2) Student behavior changes under RTG sweep: `action_change_rate` and `logits_KL` move away from baseline ~0.
3) A clear Pareto trade-off appears (capture vs steps), not “always run to Kmax.”

---

## B. Background (Self-contained)

### B1) Physical model (minimum)
In a short window, assume mic->LDV is LTI:
```
y(t) = (h * x)(t) + e(t)
```
Approximate with a small number of delay paths:
```
h(tau) ~= sum_{k=1..K} w_k * delta(tau - tau_k)
=> y(t) ~= sum_{k=1..K} w_k * x(t - tau_k) + e(t)
```

### B2) Per-frequency lag dictionary / LS projection
For a fixed frequency bin `f` in a window length `Tw`:
- Target: `y_f in C^{Tw}`
- Dictionary: `D_f in C^{Tw x M}` (M lag atoms, `M = 2*max_lag + 1`)
- After selecting set `S`, solve complex LS:
```
what_w = argmin_w ||y_f - D_{f,S} w||_2^2
r = y_f - D_{f,S} what_w
```

### B3) OMP as an MDP (but RTG can be ineffective)
OMP maps to MDP naturally:
- state: residual or correlations (e.g., `abs_corrs = |D^H r|`)
- action: select a lag atom
- transition: add atom, update LS residual
- reward: energy drop `DeltaE`

If teacher is deterministic `a = argmax(|corr|)` and always runs to `K_max`, then
`P(a | s, RTG) = P(a | s)` and `I(a; RTG | s) = 0`. DT ignoring RTG is optimal.

To fix RTG ineffectiveness, RTG must change the teacher objective itself.

---

## C. Baseline Design (must be reproducible)

### C1) Baseline commits
Use results commits that already include artifacts:
- Baseline start commit: `205a6ae`
- RTG-ineffective evidence commit: `d843ed3` (includes RTG grid and report)

### C2) Baseline definition
RTG override grid shows minimal variance across RTG values. This is the “RTG ineffective” baseline we must beat.

### C3) Fixed conditions for fair comparison
- Use the same data subset (same `subset_manifest.json`).
- Keep key params fixed (first round):
  - `Tw=32`, `max_lag=50`, `K_max=16`
  - Same `gain` (avoid epsilon-floor artifacts)
  - Same seed for data selection and evaluation

### C4) Comparison matrix

| Category | Baseline (RTG ineffective) | New (RTG= lambda) | Suggested acceptance (round 1) |
|---|---|---|---|
| Teacher behavior | No RTG dependence; always Kmax | lambda up => steps down | Spearman rho(lambda, steps) <= -0.6 |
| Student behavior | RTG sweep barely changes outputs | RTG sweep changes output | action_change_rate > baseline (e.g., > 5%) |
| Distribution shift | logits/softmax unchanged | logits shift | logits_KL_mean > baseline |
| Trade-off | No Pareto | capture vs steps Pareto | clear monotonic trade-off |

---

## D. Branch / Commit Plan (one step = one acceptance)

### D0) Create branch
Start from a results commit to keep comparisons valid.

```
git switch -c exp-rtgomp-complexity-cost 205a6ae
```

### Step 1: Reproduce baseline
- Run baseline pipeline with a new output dir.
- Generate a baseline RTG grid.
- Artifacts:
  - `results/<baseline_run>/rtg_grid/*.json`
  - `results/<baseline_run>/run.log`
  - `results/<baseline_run>/subset_manifest.json`

Acceptance:
- RTG grid variance is near zero, matching the baseline.

### Step 2: Define lambda scale
`lambda` has energy units and must match `DeltaE` scale. Recommended first pass:
- Relative scale: `lambda = c * E0`, `E0 = ||y||^2` per bin/window
- Sweep: `c in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`

Acceptance:
- `c` up => fewer steps. If steps all zero: `c` too large. If steps always Kmax: `c` too small.

### Step 3: Implement RTG= lambda in teacher
- Integrate `lambda` into the OMP stopping logic: STOP if `DeltaE < lambda`.
- Ensure no silent fallbacks; explicit error when shapes/grids mismatch.

Acceptance:
- Teacher logs show `steps_used` decreases with `lambda`.
- RTG sweep changes action sequences or distributions.

### Step 4: Evaluate student sensitivity
- Run RTG override grid and compare to baseline.
- Metrics: `action_change_rate`, `logits_KL_mean`, steps/capture Pareto.

Acceptance:
- Clear shift away from baseline ~0 metrics.

---

## E. Acceptance Criteria (Round 1)

- Teacher: Spearman rho(lambda, steps_used) <= -0.6.
- Student: action_change_rate > 0 (target >= 5% vs baseline ~0).
- Logits: logits_KL_mean clearly > baseline.
- Trade-off: visible Pareto between steps and capture.

---

## F. Reproducibility Requirements

- Use real datasets only.
- Record data fingerprint and subset manifest.
- Log exact commands, env (conda env `trl-training`), device, and seeds.
- Store artifacts under `results/<run_name>/` with `run.log`, `acceptance_check.json`, and `code_state.json`.

---

## G. Notes / Risks

- Fixed-K behavior will erase RTG effects; avoid forcing `K_max`.
- Epsilon floors can dominate low-energy bins; keep consistent with baseline for fair comparison.
- If MPS stalls during evaluation, rerun on CPU and document the device choice.

