# Fig5 B3 Frequency-Band Decomposition (Correct-by-Construction Spec)

This document is a **decision-complete** specification for producing frequency-band (“banded”) views of **Figure 5 panel B3** while preserving **mathematical equivalence** to the original full-band quantities.

It is written so that an agent with **no prior context** can:
1) understand **why v9–v10 diverged** from v1/v3,
2) avoid common traps, and
3) implement a band-decomposition that is **provably consistent** with the full-band definitions.

---

## 1) Background (what B3 originally meant)

### 1.1 Original B3 (v1/v3) semantics
The original Fig5 B3 “polar / final estimation” plot compares two **angle-level** score vectors (length **E=37**):

- **Physics baseline (OMP)**: `g_energy_expert` (orange)
- **Model-native routing signal**: `scores_expert` (green)

These are loaded from `modal_routing_val.npz` and reflect **step-0 (initial)** routing signals in the trained model’s forward pass (the scripts store step-0 scores in `model.last_outputs`).

Relevant code references:
- B3 uses `routing_data['g_energy_expert']` vs `routing_data['scores_expert']`: `generate_figure5_atomic.py:198` and `generate_figure5_atomic.py:204`
- `scores_expert` is model-native, computed from Transformer QK attention pooled over atoms: `scripts/omp-transformer-ldv.py:683`–`scripts/omp-transformer-ldv.py:691`
- `modal_routing_val.npz` is produced from step-0 model outputs (`model.last_outputs`): `scripts/visualize_omp_transformer_routing.py:120`–`scripts/visualize_omp_transformer_routing.py:167`

### 1.2 What went wrong in v9–v10
v9/v10 introduced banded plots by **changing the green curve definition**:
- Instead of model-native `scores_expert`, v9/v10 use **projection-based proxies** derived from `D^T Y` and `scores_atoms`.
- This is **not frequency-separable model-native attention**; it is a derived measure, so it will not match v1/v3 B3.

The repo already documents this mismatch:
- `docs/working-notes/figure5_b3_v3_vs_v10_report.md` shows that OMP agrees but QK does not, because the definitions differ.
- `docs/working-notes/polar_plot_architecture_analysis.md` explains why `scores_expert` is not frequency-separable and why any banded QK score is necessarily indirect/proxy unless you use attribution.

---

## 2) Definitions (symbols and data contracts)

### 2.1 Core dimensions
- `F = 346` frequency bins
- `E = 37` experts / angles
- `M = 8` atoms per expert
- `P = E*M = 296` dictionary atoms

### 2.2 Files (source of truth)
Use this run directory as the default, because it contains everything needed:

`results/omp_transformer_speech260_trainval_split_full_20251115_082341/`

It contains:
- `dictionary.npz` with:
  - `D` shape `(F, P)`
  - `angles` shape `(E,)`
  - `H` shape `(F, E)`
  - `W_reduced` shape `(F, M)`
- `modal_routing_val.npz` with:
  - `Y_val` shape `(N, F)`
  - `labels` shape `(N,)`
  - `scores_atoms` shape `(N, E, M)` (step-0)
  - `scores_expert` shape `(N, E)` (step-0, model-native)
  - `g_energy_expert` shape `(N, E)` (baseline)
- `model_best.pth` and `code_state.json` for reconstructing the trained model.

### 2.3 Frequency axis mapping (band masks)
The figure scripts define the frequency axis as:

`freqs = np.linspace(300, 3000, F)`

Band masks MUST be built on this same `freqs`.

### 2.4 The two “full-band” target quantities

#### A) Physics baseline (per expert)
Let:
- `y ∈ R^F` be one sample spectrum (`Y_val[i]`)
- `D ∈ R^{F×P}` be the dictionary
- `g = D^T y ∈ R^P` be the atom projections

`modal_routing_val.npz` stores the baseline expert score as:

`g_energy_expert[e] = Σ_{m=1..M} |g[e,m]|`

where `g[e,m]` is the `(e,m)` entry of `g` reshaped to `(E, M)`.

Reference: `scripts/visualize_omp_transformer_routing.py:184`–`scripts/visualize_omp_transformer_routing.py:200`.

#### B) Model-native expert routing score (per expert)
`scores_expert[e]` is computed by the model (Transformer QK attention pooled over atoms). It is **not** a simple per-frequency additive quantity.

Reference: `scripts/omp-transformer.py:192`–`scripts/omp-transformer.py:196` and the architectural notes in `docs/working-notes/polar_plot_architecture_analysis.md`.

---

## 3) The central requirement: “effective equivalence”

### 3.1 What “equivalent decomposition” means in this spec
For a chosen full-band quantity `S_full[e]`, we want a band decomposition `S_band[k,e]` such that:

- **Additivity / reconstruction**
  - Physics: `S_full[e] == Σ_k S_band[k,e]` (up to floating error)
  - Model-native: `S_full[e] == S_base[e] + Σ_k S_band[k,e]` (IG completeness, see below)

- **No double counting**
  - Bands must be a partition of `[300, 3000]` (disjoint masks whose union is all bins).

### 3.2 Why “compute a new score on each band and sum” usually fails
If your full-band uses non-linear operations like `abs()` or `sqrt(sum(.^2))`, then:
- `abs(a+b) != abs(a)+abs(b)` in general
- `sqrt(x+y) != sqrt(x)+sqrt(y)` in general

Therefore, naive “band score = abs(D_band^T y_band)” is **not** equivalent to the full-band definition unless you apply a correct alignment / attribution rule.

---

## 4) Band definitions (MUST use these)

We define a partition of the full band `[300, 3000]` as 4 disjoint bands:

1) `B1 = [300, 500)`
2) `B2 = [500, 1000)`
3) `B3 = [1000, 2000)`
4) `B4 = [2000, 3000]` (inclusive upper to capture 3000)

Implementation rule:
- For all but the last band: `min_hz <= f < max_hz`
- For the last band: `min_hz <= f <= max_hz`

This choice avoids a known pitfall in v10: bands that stop at 2000 Hz do **not** sum to the full `[300,3000]` band.

---

## 5) Correct decomposition methods (what to implement)

This spec defines **two** decompositions: one for the physics baseline and one for the model-native score. They are different because the underlying math is different.

### 5.1 Physics baseline: exact band decomposition of `g_energy_expert`

#### Step 1 — Decompose the linear projection per band
For each band `Bk`, compute:

`g_k = D[Bk,:]^T · y[Bk]   ∈ R^P`

Because this is linear:

`g = Σ_k g_k` exactly.

#### Step 2 — Convert to the stored full-band definition with sign-aligned contributions
The stored full-band uses absolute values per atom then sums over atoms per expert:

`g_energy_expert[e] = Σ_m |g[e,m]|`

To get a band contribution that sums back to `|g|`, define:

`c_k[e,m] = sign(g[e,m]) · g_k[e,m]`

Then:

`Σ_k c_k[e,m] = |g[e,m]|` (for real-valued g)

and:

`G_k[e] = Σ_m c_k[e,m]`

Finally:

`Σ_k G_k[e] = g_energy_expert[e]`

#### Why this is the only “equivalent” choice
Any method that applies `abs()` inside each band will generally fail reconstruction due to cross-band cancellation.

#### Required validation
For any tested sample:
- `max_abs_error = max_e |g_energy_expert[e] - Σ_k G_k[e]|`
- Must be `< 1e-6` when computed in float64.

### 5.2 Model-native `scores_expert`: frequency attribution via Integrated Gradients (IG)

Because `scores_expert` is produced by a non-linear Transformer, it is not frequency-separable.
The correct way to obtain an additive “band decomposition” is to compute **attributions** of the output w.r.t the input frequency bins.

#### Target function (must match the original B3 quantity)
We target the **step-0** model-native output:

`S_full[e] = scores_expert_step0(y, D)[e]`

Implementation must extract `scores_expert` exactly as the run’s `modal_routing_val.npz` does:
- `model.eval()`
- call `model(y, D, train_mode=True)`
- read `model.last_outputs['scores_expert']` (step-0) as the output vector

Reference: `scripts/visualize_omp_transformer_routing.py:137`–`scripts/visualize_omp_transformer_routing.py:167`.

#### IG baseline (fixed decision)
Use:
- `y0 = mean(Y_val)` (mean over validation spectra from `modal_routing_val.npz`), then unit-normalize:
  - `y0 = y0 / (||y0|| + 1e-12)`

Rationale (causal):
- This model applies **per-sample score standardization/centering** (see `code_state.json` flags like `score_norm="std"` and `score_center_expert=true`).
- A zero-vector baseline can push intermediate `scores_expert` distributions into near-degenerate regimes (very small std), which makes the IG integral converge extremely slowly.
- Using a baseline on the data manifold makes gradients numerically stable while preserving IG reconstruction.

Note: `scores_expert(y0, D)` is generally **non-zero**. Treat it as the baseline offset term.

#### IG definition (per expert)
For each expert `e`, define:

`f_e(y) = scores_expert_step0(y, D)[e]`

Integrated Gradients per frequency bin `f`:

`IG_e[f] = (y[f]-y0[f]) · ∫_{α=0..1} ∂f_e(y0 + α(y-y0)) / ∂y[f] dα`

Numerical approximation:
- use uniform α grid: `α_t = t/n_steps` for `t=0..n_steps`
- integrate over α with **trapezoidal rule** (default) and `n_steps = 2048` (CPU-safe)
- optional faster smoke: `n_steps = 1024` with a looser tolerance (see Acceptance)

#### Band aggregation and reconstruction
Band contribution:

`IG_e[Bk] = Σ_{f∈Bk} IG_e[f]`

Completeness (required check):

`f_e(y) - f_e(y0) ≈ Σ_k IG_e[Bk]`

Therefore:

`f_e(y) ≈ f_e(y0) + Σ_k IG_e[Bk]`

#### Required validation
Compute, for the chosen sample:
- `max_abs_error = max_e | f_e(y) - (f_e(y0) + Σ_k IG_e[Bk]) |`
- Recommended integration:
  - Use **trapezoidal rule** over α∈[0,1] with `n_steps=2048` (CPU-safe default).
- Must be:
  - `< 1e-3` with `n_steps>=2048` (trapezoid, `y0=mean_val`)
  - `< 2e-3` with `n_steps>=1024` (trapezoid, `y0=mean_val`) for a faster smoke

If the tolerance is not met, increase `n_steps` (do not silently accept a poor reconstruction).

#### Why IG (and not v9/v10 proxy) is “correct”
IG is additive by construction and is the standard method to attribute non-linear model outputs to input features, yielding a band-decomposition that reconstructs the original model-native score (up to numerical error).

---

## 6) Scope, outputs, and naming (what “done” looks like)

### 6.1 Scope (explicit)
This spec is for B3 frequency-band decomposition on:
- **one representative sample** (the same sample selection logic used in Figure 5 scripts)
- producing band contributions that are **reconstructible**

Dataset-wide averaging is optional and out-of-scope for a first correct implementation.

### 6.2 Required outputs (all under `results/<run_name>/`)
The implementation must write:

1) `results/<run_name>/run.log`
   - command line, environment, device, git hash (if available), run_dir used

2) `results/<run_name>/band_defs.json`
   - band boundaries and the exact bin indices per band (to make reconstruction audit-able)

3) `results/<run_name>/decomposition.npz` (or `.json` if preferred)
   Must contain:
   - `angles_deg` (E,)
   - `freqs_hz` (F,)
   - `sample_idx` (int) and `label_idx`/`label_deg`
   - physics:
     - `g_energy_full` (E,)
     - `g_energy_band` (K,E) (the sign-aligned decomposition)
     - `g_energy_recon` (E,)
   - model-native IG:
     - `scores_expert_full` (E,)
     - `scores_expert_base` (E,) = f(y0)
     - `scores_expert_band` (K,E) (IG band contributions)
     - `scores_expert_recon` (E,) = base + Σ bands

4) `results/<run_name>/checks.json`
   - reconstruction errors and tolerances used

5) `results/<run_name>/code_state.json`
   - binds artifacts to the exact executed code + inputs (git head/dirty + SHA256 for key files)

6) Plots (PDF preferred; PNG optional):
   - Five comparison line plots (each includes QK + PHYSICS + True DoA marker):
     - `results/<run_name>/Fig5_B3_LINE_300_3000.pdf` (full band)
     - `results/<run_name>/Fig5_B3_LINE_300_500.pdf`
     - `results/<run_name>/Fig5_B3_LINE_500_1000.pdf`
     - `results/<run_name>/Fig5_B3_LINE_1000_2000.pdf`
     - `results/<run_name>/Fig5_B3_LINE_2000_3000.pdf`
   - Labeling requirements:
     - Physics band curves are **sign-aligned contributions** (equivalent to `g_energy_expert` when summed).
     - QK band curves are **IG attributions** (equivalent to `scores_expert` when added to the baseline and summed).
   - Normalization (recommended):
     - Use per-curve min-max normalization in each plot so QK and PHYSICS are comparable on a shared y-axis without changing their argmax (peak angle).

Optional (visualization only):
- It is allowed to apply light smoothing across the **angle axis** when rendering plots.
- Smoothing MUST NOT change any stored arrays or equivalence checks; apply it only to the plotted curves.
- If peak/argmax markers are shown, they MUST be computed from the underlying raw (unsmoothed) curves, because smoothing can shift peaks.

### 6.3 Output policy (hard constraint)
Never write artifacts to repo root. All outputs must live under `results/<run_name>/`.

---

## 7) Guardrails and pitfalls (read this before implementing)

### 7.1 Non-negotiable constraints
- **No “best-effort” fallbacks**:
  - If shapes mismatch (`Y.F != D.F`), raise.
  - If angles mismatch or duplicates exist, raise.
  - If bands do not partition `[300,3000]`, raise.
- **No per-band renormalization of dictionary atoms**.
  - Mask rows only; do not re-normalize D_band columns.
- **No “abs per band then sum” if the goal is equivalence**.
  - Use sign-aligned decomposition for physics.
  - Use IG for model-native.
- **All documentation and new files must be in English** (project policy).

### 7.2 Common failure modes (what will make your output wrong)
- Using v10’s `abs(sum(atom_proj * scores_atoms))` and calling it “model-native QK”.
  - That quantity is a proxy; it will not match `scores_expert`.
- Using bands that don’t cover `[300,3000]` (e.g., stopping at 2000 Hz).
- Double counting bins at band boundaries (inclusive on both sides).
- Running model in train mode (dropout on) when computing IG.
  - Must use `model.eval()`.
- Computing IG with too few steps and claiming exact reconstruction.

---

## 8) Acceptance checklist (must pass)

### A) Input validation
- [ ] `dictionary.npz` provides `D` with shape `(346, 296)` and `angles` with shape `(37,)`.
- [ ] `modal_routing_val.npz` provides `Y_val` with shape `(N, 346)` and `scores_expert` with shape `(N, 37)`.
- [ ] Bands partition `[300,3000]` with no overlaps and no gaps.

### B) Physics reconstruction
- [ ] `max_abs_error(g_energy_full, sum_bands(g_energy_band)) < 1e-6` (float64).

### C) Model-native reconstruction (IG)
- [ ] `max_abs_error(scores_full, base + sum_bands(scores_band)) < 1e-3` with
      `ig_baseline=mean_val`, `ig_method=trapezoid`, and `n_steps>=2048`.
- [ ] Optional faster smoke: `< 2e-3` with the same settings and `n_steps>=1024`.

### D) Labeling correctness (avoiding interpretability traps)
- [ ] All plots and filenames clearly indicate:
  - physics band decomposition is sign-aligned (equivalent to `g_energy_expert`)
  - QK band decomposition is IG attribution (equivalent to `scores_expert`)

---

## 9) Implementation notes (recommended code reuse)

Model reconstruction:
- Reuse `scripts/eval_omp_transformer_split.py` logic to rebuild the model from:
  - `results/omp_transformer_speech260_trainval_split_full_20251115_082341/code_state.json`
  - and load `model_best.pth`

Representative sample selection:
- Reuse the selection logic from `generate_figure5_atomic.py:28` (true=145°, physics=60°, qk=145° fallback).
