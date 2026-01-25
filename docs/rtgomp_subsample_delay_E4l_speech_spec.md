# Spec: E4l-Speech -- Sub-sample Delay Refinement Diagnostics (GCC-PHAT / Phase-Slope) Under Compute Control

This spec defines **E4l-Speech**, a Phase-2b follow-up to E4k-Speech.

E4k-Speech established that:
- compute is controllable via lambda_c (RTG/STOP),
- DT remains non-degenerate at the target compute regime (~12 selected lags at lambda_c=3e-4), and
- a coarse delay proxy (first-lag dispersion across frequency) is stable on paired speech and explodes under a
  mispair guardrail.

E4l-Speech extends this to **sub-sample delay refinement**:

> Given a coarse, window-level lag estimate (in STFT frames), can we estimate a **sample-level delay** tau_hat
> (in samples or milliseconds) using waveform-domain estimators (GCC-PHAT and/or phase-slope), and does the estimate
> remain stable as compute is reduced (higher lambda_c), and does it fail under mispairing?

This spec is written for an engineer with **no prior signal processing background**.

Implementation to be executed:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py (extend with E4l diagnostics flags)

---

## 0) Why This Exists (Phase Context)

Our long-term goal is MIC+LDV time alignment for downstream TDoA-like tasks. LDV signals may have frequency-dependent
phase distortion (material resonance / dispersion), which can make time-delay estimation unstable if treated as a pure
delay.

E4l-Speech does not claim "TDoA solved". Instead it provides a necessary bridge:
- It produces a **sample-level delay estimate** and **confidence diagnostics** on real speech WAV,
- under a **compute knob** (lambda_c) inherited from the DT policy,
- and validates sensitivity with a **mispair guardrail**.

---

## 1) Minimal Primer

### 1.1 What is "sub-sample delay"?

A delay tau between two signals is typically measured in **samples**. For fs=16000 Hz:

  1 sample = 1/16000 sec = 0.0625 ms

Sub-sample delay estimation means estimating tau more precisely than coarse STFT-frame lags.

E4k used hop_length=160, so:

  1 STFT frame lag = 160 samples = 10 ms

E4l aims to refine within that 10ms grid.

### 1.2 GCC-PHAT (core required estimator)

Given two real signals x(t) and y(t), GCC-PHAT estimates time delay by:
- computing the cross-power spectrum X(f) * conj(Y(f)),
- applying PHAT weighting (normalize magnitude),
- inverse FFT to obtain a correlation function whose peak indicates delay.

PHAT weighting emphasizes phase agreement and suppresses magnitude coloration.

### 1.3 Phase-slope / group delay (optional estimator)

If y(t) is approximately a delayed version of x(t) through an LTI path, the cross-spectrum phase is approximately
linear in frequency:

  phi(f) ≈ -2*pi*f*tau + b

Fitting a line to unwrapped phase vs frequency yields tau.

If LDV has strong dispersion (frequency-dependent delay), this fit will have large residual error.

---

## 2) Dataset, Pairing, and Guardrails

### 2.1 Speech WAV-only roots (required)

Allowed roots (do not change):
- mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Guardrail:
- subset_manifest.json must contain only `.wav` paths (no `.npy`), otherwise FAIL.

### 2.2 Pairing modes (two runs required)

1) paired
   - Correct MIC↔LDV pairing by filename rule.

2) mispair_shift1 (guardrail diagnostic; still real WAV)
   - MIC list unchanged.
   - LDV list shifted by +1 index (cyclic wrap).
   - This should break alignment and confidence metrics.

### 2.3 No-fallback policy (hard failures)

E4l must be fail-fast and avoid silent coercions:
- If WAV sample rate is not exactly fs=16000: FAIL (no resampling fallback).
- If a window waveform segment cannot be extracted (too short): skip the window but log counters; do not fabricate data.

---

## 3) Fixed Parameters (must match E4k unless explicitly justified)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2

Compute knob (required 5-point grid):
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

Sub-sample refinement defaults (required unless justified):
- method_gcc_phat = enabled
- method_phase_slope = optional (if enabled, must log fit quality)
- search_radius_frames = 2 (=> +/- 20ms around coarse estimate)
- bandpass filter = [300, 3000] Hz (zero-phase filtering recommended)

---

## 4) How to Define the Window in Waveform Samples (must be explicit)

E4l operates per (clip_idx, window_idx) aligned to the STFT windows used by E4h/E4k.

Given:
- start_t: STFT frame index for window start
- tw: number of frames in the window
- hop_length
- n_fft

Important alignment note:
- The dataset STFT is produced by `scipy.signal.stft(...)` with default `boundary="zeros"` and `padded=True`.
- Therefore, STFT frame indices correspond to a **zero-padded waveform** where the first analysis frame is centered at
  time 0 (and the waveform is pre-padded by `n_fft//2` zeros).

Define a waveform segment that covers exactly the support of those STFT frames in the **padded waveform domain**:

  segment_start_sample = start_t * hop_length
  segment_len_samples  = (tw - 1) * hop_length + n_fft
  segment_end_sample   = segment_start_sample + segment_len_samples

Extract:
- x_seg = mic_wav_padded[segment_start_sample : segment_end_sample]
- y_seg = ldv_wav_padded[segment_start_sample : segment_end_sample]

If segment_end_sample exceeds either padded waveform length, skip this window and increment a counter.

---

## 5) Coarse-to-fine Strategy (how lambda_c affects sub-sample delay)

We use DT/OMP to define a coarse lag estimate in frames, then refine within a small sample search window.

### 5.1 Coarse lag (frame units)

For each window and each lambda_c:
- DT provides dt_first_lag_median_frames (median of first selected lags across frequency bins).
- OMP provides omp_first_lag_median_frames (same definition; independent of lambda).

Convert coarse lag to samples:

  coarse_delay_samples = round(coarse_lag_frames * hop_length)

### 5.2 Search window around the coarse estimate

Define:

  search_radius_samples = search_radius_frames * hop_length

The refined tau_hat must be searched only in:

  tau_hat_samples ∈ [coarse_delay_samples - search_radius_samples,
                     coarse_delay_samples + search_radius_samples]

This makes the estimator compute-bounded and tests whether coarse selection is good enough to guide sub-sample delay.

---

## 6) Required Sub-sample Estimator: GCC-PHAT

### 6.1 Preprocessing (required)

Before GCC-PHAT:
- convert to float32
- remove mean (DC)
- apply bandpass [300, 3000] Hz to both x_seg and y_seg using zero-phase filtering (e.g., filtfilt), to match the STFT
  band used by lag-selection.

### 6.2 GCC-PHAT computation (required)

Compute GCC-PHAT cross-correlation cc[lag] between x_seg and y_seg.

Sign convention (must be consistent and documented):
- Define tau_hat_samples such that shifting MIC forward by tau aligns to LDV:
  - y[t] ≈ x[t - tau_hat_samples]
  - tau_hat_samples > 0 means LDV lags MIC (arrives later).

Compute:
- peak lag (within the search window) => tau_hat_samples
- peak amplitude => peak_abs

### 6.3 Confidence metrics (required)

Compute at minimum:
- psr (peak-to-sidelobe ratio):
  - psr = peak_abs / (median(|cc| outside a small exclusion zone around the peak) + eps)
- boundary_hit: peak is at the edge of the search window (indicates failure)

Optional:
- peak_width_samples (e.g., number of samples above 0.5*peak)
- peak_to_second_ratio

---

## 7) Optional Estimator: Phase-slope group delay (if enabled)

If enabled, compute:
- cross-spectrum phase in [300, 3000] Hz
- unwrap phase
- weighted linear fit phi(f) ≈ a f + b
- tau_hat = -a / (2*pi)

Required fit diagnostics:
- fit_rmse_rad
- r2 (or equivalent)

If fit is ill-posed (insufficient valid bins), output null values and increment counters (no fallback values).

---

## 8) Outputs (Artifacts and Schemas)

E4l produces all E4h/E4k summaries PLUS new sub-sample diagnostics artifacts.

### 8.1 Per-window JSONL (required)

Write:
- results/<run>/subsample_delay_diagnostics.jsonl

One row per:
- (clip_idx, window_idx, lambda_c, method, coarse_source)

Minimum required fields:
- pairing_mode, clip_idx, window_idx, start_t, lambda_c
- coarse_source: "dt" or "omp"
- coarse_lag_frames, coarse_delay_samples
- search_radius_samples
- gcc_phat_tau_hat_samples, gcc_phat_tau_hat_ms
- gcc_phat_psr, gcc_phat_boundary_hit

If phase-slope is enabled:
- phase_slope_tau_hat_ms, phase_slope_fit_rmse_rad, phase_slope_r2

### 8.2 Summary JSON (required)

Write:
- results/<run>/summary/subsample_delay_diagnostics_summary.json

One row per lambda_c, reporting for each coarse_source ("dt", "omp"):
- tau_hat_ms_p50/p90
- psr_p50/p90
- boundary_hit_rate
- fraction_defined (windows with valid estimate)
- within_clip_tau_mad_ms_p50 (see below)

### 8.3 Within-clip stability (required)

For each clip, compute the dispersion of tau_hat_ms across its windows:
- clip_tau_mad_ms = median(|tau - median(tau)|)

Then report:
- within_clip_tau_mad_ms_p50/p90 across clips.

This is the primary “stability” metric when ground truth geometry is unavailable.

---

## 9) Hard Guardrails (FAIL conditions)

Inherit E4h/E4k hard guardrails:
- missing files: FAIL
- md5 mismatches vs manifest: FAIL
- NaN/Inf in capture: FAIL
- capture out-of-range (paper baseline): FAIL
- OMP monotonicity violations: FAIL
- DT duplicates under forced-K: FAIL

E4l additional hard guardrails:
- require_wav_only=1: any non-wav path in subset => FAIL
- fs mismatch (wav header fs != 16000): FAIL (no resampling fallback)
- subsample_delay_diagnostics.jsonl missing when enabled: FAIL
- summary/subsample_delay_diagnostics_summary.json missing when enabled: FAIL

---

## 10) Acceptance Criteria (PASS vs PASS_WITH_WARNINGS vs FAIL)

Because this is a diagnostic stage, acceptance focuses on:
- pipeline validity,
- non-degeneracy of DT under compute control,
- and strong guardrail separation.

### 10.1 Paired run (required for PASS)

- All hard guardrails pass.
- DT remains non-degenerate at lambda_c=3e-4:
  - DT - Random (compute-matched capture mean) > 0
- GCC-PHAT produces valid estimates for most windows:
  - fraction_defined >= 0.90 at lambda_c=3e-4 for coarse_source="dt"
  - boundary_hit_rate <= 0.20 at lambda_c=3e-4 for coarse_source="dt"

### 10.2 Guardrail separation (paired vs mispair; required for PASS)

On scale_check_subset (48 pairs), comparing paired vs mispair_shift1:
- For at least 4/5 lambdas:
  - psr_p50(paired, dt) > psr_p50(mispair, dt)
  - within_clip_tau_mad_ms_p50(paired, dt) < within_clip_tau_mad_ms_p50(mispair, dt)

If paired passes but separation fails:
- PASS_WITH_WARNINGS (must explain why, because the metric may be insensitive or mispair is not adversarial enough).

### 10.3 Compute-vs-stability trend (informational; not a hard FAIL)

Report whether stability degrades as lambda increases:
- If within_clip_tau_mad_ms_p50 increases sharply with lambda, interpret as compute reduction harming alignment.
- Mild monotonic drift is acceptable at this stage.

---

## 11) Required Runs (Sequence)

E4l must run the following (each with lockdir + tee):

1) Preflight: dataset exists, len(dataset)==416, wav-only.
2) Smoke (paired): 1 pair.
3) Functional (scale_check_subset):
   - paired (positive path)
   - mispair_shift1 (guardrail)
4) Full dataset (paired): 416 pairs.

---
