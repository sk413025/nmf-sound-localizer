# Spec: E4p-Speech — Physics-Derived Frequency Conditioning via Dispersion Prior (Phase-Slope Subbands → Lag Prior)

This spec defines **E4p-Speech**.

E4p-Speech introduces a **new, physics-derived “frequency conditioning” mechanism** for DTmin, designed to be meaningful (non-degenerate) and to address **dispersion-like frequency dependence** without training “one model per band”.

All content and artifacts MUST be in English.

Implementation target:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

Non-goals:
- E4p does **not** retrain DTmin.
- E4p does **not** change OMP or Random baselines.
- E4p does **not** apply any implicit fallbacks (fail fast on undefined prior fits).

---

## 0) Background + Motivation (Physics → DTmin)

### 0.1 Why E4o showed “freq conditioning ignored”

E4o-Speech measured inference-time ablations on the existing model’s `freq_idx` embedding and found negligible impact on capture metrics. This is physically and mathematically plausible because the DTmin state is an (approximate) sufficient statistic for the OMP greedy decision:

- DTmin receives `s_f^(k) ≈ |D_f^H r_f^(k)|`, which almost fully determines the argmax lag choice.
- Therefore, conditioning on the *frequency index* `f` provides little additional information once `s` is known.

Conclusion: **“freq_idx embedding” is not the right physics object to condition on.**

### 0.2 What physical object actually couples frequency bins

Dispersion-like behavior appears as frequency-dependent phase structure in the MIC→LDV transfer relationship.

Let `C(f) = X*(f) Y(f)` be the cross-spectrum / cross-power. The phase of `C(f)` contains delay structure:

- For a pure delay `τ`, `arg C(f) ≈ -2π f τ + const`
- For dispersive / multi-path / system phase effects, the effective delay can vary with frequency, but it is typically *smooth/structured* across frequency.

Therefore, the natural physics-derived “frequency conditioning” is **a prior on lag (delay) as a function of frequency**, not an embedding ID.

---

## 1) Proposed Solution (Physics → MAP-style Lag Selection)

E4p introduces a **dispersion prior** that biases DTmin lag logits using a per-frequency expected delay derived from phase slope fits in subbands.

### 1.1 Phase-slope subband fits (physics estimate)

From STFT cross-power estimated on real speech WAV:

`C_f = Σ_t conj(X_f[t]) * Y_f[t]`

For each subband `[f_lo, f_hi]`, fit a weighted phase slope:

`unwrap(arg(C)) ≈ slope * f + intercept`

Then:

`τ_sec = -slope / (2π)` and `τ_samples = τ_sec * fs`

This provides a **physically interpretable group-delay estimate** per subband.

### 1.2 Convert delay to DT lag prior (frames)

DT lag actions are in STFT frame units:

`lag_frames ≈ τ_samples / hop_length`

For each frequency bin, assign it the τ of its subband to obtain `τ(f)` in frames.

Define a Gaussian log-prior over candidate lags `ℓ`:

`log p(ℓ | f) = -0.5 * ((ℓ - τ(f)) / σ)^2`

Bias DT logits:

`logits_f[lag] += β * log p(lag | f)`

This is a MAP-style fusion of:
- data-driven DT logits (from `|D^H r|` and RTG)
- a physics-derived frequency-dependent lag prior

### 1.3 Ablations (must be measurable)

E4p includes explicit ablations for the prior assignment:
- `normal`: use τ(f) from phase-slope fits (physical mapping)
- `shuffle`: permute τ(f) across frequency bins (wrong conditioning)
- `constant`: replace τ(f) with its median (remove frequency dependence)

---

## 2) Hard Guardrails (Required)

Data:
- Real data only
- WAV-only roots (boy1 speech):
  - mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
  - ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- `require_wav_only=1`
- `fs=16000` only; any mismatch MUST fail (no resampling)

Model checkpoint (DTmin):
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

---

## 3) Fixed Parameters (Must Match Paper Defaults)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- device = cpu (recommended for stability)

Dispersion prior defaults:
- disp_prior_mode = `phase_slope_subbands`
- disp_prior_num_subbands = 3
- disp_prior_min_bins = 64 (fail fast if fewer)
- disp_prior_sigma_frames = 2.0
- disp_prior_beta = 2.0

---

## 4) Required Implementation Details (Evaluator)

Add optional physics prior that affects **DT only**:
- new CLI:
  - `--disp_prior_mode {none,phase_slope_subbands}`
  - `--disp_prior_cond_mode {normal,shuffle,constant}`
  - `--disp_prior_seed`
  - `--disp_prior_beta`
  - `--disp_prior_sigma_frames`
  - `--disp_prior_num_subbands`
  - `--disp_prior_min_bins`
- apply the prior as an additive bias to DT lag logits (`0..M_lags-1`) without changing:
  - OMP baseline
  - Random baseline
  - dictionary construction / residual update

Artifacts (required when prior enabled):
- `summary/dispersion_prior_summary.json`
  - config
  - per-clip phase-slope fits (tau_ms/rmse/r2/bin counts)
  - DT-first-lag abs error vs **physical τ(f)** by lambda:
    - `dt_first_lag_abs_err_vs_tau_physical.mean/std/count_defined/count_undefined`

---

## 5) Acceptance Criteria (PASS / PASS_WITH_WARNINGS / FAIL)

E4p is a measurement+mechanism experiment: it is acceptable if the prior does not improve capture, as long as it is **measurably used** and consistent with physics.

### 5.1 Run validity (all runs)
PASS requires:
- guardrails satisfied (wav-only, fs=16000)
- no NaN/Inf in summaries
- required artifacts exist:
  - `summary/compute_matched_summary.json`
  - `summary/forced_k_summary.json`
  - `summary/rtg_controllability_summary.json`
  - `summary/freq_cond_audit_summary.json`
  - `summary/dispersion_prior_summary.json` (if prior enabled)
  - `subset_manifest.json`, `run.log`, `code_state.json`, `ACCEPTANCE_REPORT.md`

### 5.2 Conditioning effectiveness (scale_check_subset, paired)
Primary metric (physics-consistent):
- `E_tau(lambda) = dt_first_lag_abs_err_vs_tau_physical.mean` (frames)

Define `E_tau` at `lambda_c = 3e-4`:
- `E_tau_normal` (prior enabled, cond_mode=normal)
- `E_tau_shuffle` (prior enabled, cond_mode=shuffle)

PASS criteria:
- `E_tau_shuffle - E_tau_normal >= 0.5` frames (conditioning is used and ablation breaks it)

Warnings:
- If capture drops by > 0.02 absolute vs prior disabled at `lambda_c=3e-4`, mark `PASS_WITH_WARNINGS` and analyze why.

FAIL:
- Any guardrail violation or missing required artifacts.
- Prior enabled but `dispersion_prior_summary.json` missing.
- `E_tau_shuffle - E_tau_normal < 0.1` frames (inconclusive / prior too weak / dt ignores bias).

---

## 6) Outputs to Report

Required table (scale_check_subset):
- For each mode (`none`, `normal`, `shuffle`, `constant`):
  - `DT_cm` = compute-matched DT capture mean at `lambda_c=3e-4`
  - `k_selected_mean` at `lambda_c=3e-4`
  - `E_tau` at `lambda_c=3e-4` (prior modes)

Required interpretation:
- Explain in causal language why the prior should matter (or not) given the physics and the DT state definition.

