# Spec: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness (GCC-PHAT + Phase-Slope) Under Compute Control

This spec defines **E4m-Speech** (Phase-2c).

E4m-Speech is a follow-up to:
- **E4h-Speech**: paper eval harness on speech WAV (DT vs OMP vs Random).
- **E4j-Speech**: STOP/cost calibration to hit mean compute ~12 at lambda_c=3e-4 while keeping DT>Random.
- **E4k-Speech**: coarse (frame-level) delay stability on paired speech and strong failure under mispair guardrail.
- **E4l-Speech**: waveform-domain **GCC-PHAT** sub-sample delay refinement stable under compute control and sensitive to mispair.

E4m-Speech adds **dispersion diagnostics** to answer:

> Is the MIC->LDV relation consistent with a **pure time delay** (single tau), or does it show evidence of
> **frequency-dependent group delay / dispersion** (tau varies with frequency), which would require
> calibration (e.g., phase equalization) before any TDoA claim?

This spec is written for an engineer with **no assumed signal processing background**.

Implementation target:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py` (extend with E4m diagnostics)

---

## 0) Why This Exists (Phase Context)

Long-term goal: MIC+LDV time alignment for downstream TDoA-like tasks.

Problem: LDV can introduce frequency-dependent phase distortion (material resonance / dispersion / phase wrapping).
If we treat LDV as a pure delayed MIC signal, but the true phase is not linear in frequency, then:
- a single delay estimate can be **stable but biased** (wrong),
- or unstable depending on frequency content and windowing.

E4m-Speech is the decision point:
- If the path is close to pure delay, we are **ready for Phase-3 correctness evaluation** (geometry/known delay).
- If dispersion is present, we are **ready for calibration design** (phase equalization) and should not jump to TDoA.

---

## 1) Minimal Primer

### 1.1 Time delay and frequency-domain phase

If `y(t)` is a delayed version of `x(t)`:

  y(t) ≈ x(t - tau)

Then in frequency domain:

  Y(f) ≈ X(f) * exp(-j 2 pi f tau)

So the cross-spectrum phase is approximately linear in `f`:

  phi(f) = angle(conj(X(f)) * Y(f)) ≈ -2 pi f tau + b

### 1.2 Dispersion / frequency-dependent group delay

If the LDV path has frequency-dependent delay:

  Y(f) ≈ X(f) * exp(-j 2 pi f tau(f))

Then `phi(f)` is no longer well-approximated by a single straight line.
Signatures:
- phase-slope linear fit has poor quality (low R^2, high RMSE),
- per-band estimated delays differ (tau varies across sub-bands),
- disagreement between GCC-PHAT tau and phase-slope tau increases.

### 1.3 GCC-PHAT vs phase-slope

GCC-PHAT:
- Finds delay as the peak of a PHAT-weighted correlation function.
- Robust when a single dominant delay exists in-band.

Phase-slope (group delay):
- Fits a straight line to unwrapped cross-spectrum phase vs frequency.
- Produces fit diagnostics (R^2, RMSE) that directly reveal non-linear phase (dispersion).

E4m uses BOTH:
- GCC-PHAT (required) for continuity with E4l.
- Phase-slope (required) to quantify dispersion and calibration readiness.

---

## 2) Dataset, Pairing, and Guardrails

### 2.1 Speech WAV-only roots (required)

Allowed roots (do not change):
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

Guardrail:
- `subset_manifest.json` must contain only `.wav` paths (no `.npy`), otherwise FAIL.

### 2.2 Pairing modes (two runs required)

1) `paired`
   - Correct MIC<->LDV pairing by filename rule.

2) `mispair_shift1` (guardrail diagnostic; still real WAV)
   - MIC list unchanged.
   - LDV list shifted by +1 index (cyclic wrap).
   - Expected to break alignment and degrade phase-slope fit quality and GCC-PHAT confidence.

### 2.3 No-fallback policy (hard failures)

E4m must be fail-fast and avoid silent coercions:
- If WAV fs is not exactly `fs=16000`: FAIL (no resampling).
- If any non-wav path appears when `require_wav_only=1`: FAIL.
- If estimators cannot be computed (insufficient bins, etc.), output nulls + counters (no fabricated defaults).

---

## 3) Fixed Parameters (must match E4l unless explicitly justified)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2

Compute knob (required 5-point grid; exact values):
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

Sub-sample refinement:
- search_radius_frames = 2 (=> +/- 20ms around coarse estimate)
- bandpass filter: [300, 3000] Hz (zero-phase filtering)

Dispersion diagnostics (E4m-specific):
- phase_slope enabled (required)
- phase_slope_min_bins = 64 (minimum bins required for a valid fit)
- phase_slope_weights = |cross_power| (weighted least squares)
- phase_slope_subbands_hz (fixed for E4m):
  - [300, 900], [900, 1800], [1800, 3000]

---

## 4) Waveform Segment Definition (must match STFT indexing)

E4m operates per (clip_idx, window_idx) aligned to the STFT windows used by E4h/E4k/E4l.

Important alignment note:
- The dataset STFT is produced by `scipy.signal.stft(...)` with default `boundary="zeros"` and `padded=True`.
- Therefore, STFT frame indices correspond to a **zero-padded waveform** with pre-padding `n_fft//2` zeros.

Define a waveform segment in the padded-waveform domain:

  segment_start_sample = start_t * hop_length
  segment_len_samples  = (tw - 1) * hop_length + n_fft
  segment_end_sample   = segment_start_sample + segment_len_samples

Extract:
- x_seg = mic_wav_padded[segment_start_sample : segment_end_sample]
- y_seg = ldv_wav_padded[segment_start_sample : segment_end_sample]

If the segment is out-of-bounds for either waveform, skip the window and increment counters.

---

## 5) Coarse-to-fine Strategy (coarse lag -> waveform-domain search)

### 5.1 Coarse lag (frame units)

For each window and each lambda_c:
- DT provides `dt_first_lag_median_frames` (median first selected lag across frequency bins).
- OMP provides `omp_first_lag_median_frames` (same definition; independent of lambda).

Convert to samples:

  coarse_delay_samples = round(coarse_lag_frames * hop_length)

### 5.2 Search window

Define:

  search_radius_samples = search_radius_frames * hop_length

All waveform-domain delay estimates are constrained to:

  tau_hat_samples in [coarse_delay_samples - search_radius_samples,
                      coarse_delay_samples + search_radius_samples]

This tests whether reduced compute still yields sufficiently accurate coarse guidance.

---

## 6) Required Estimator A: GCC-PHAT (same as E4l)

Preprocessing (required):
- float32
- mean removal (DC)
- bandpass [300,3000] Hz (zero-phase filtfilt)

GCC-PHAT outputs (required):
- gcc_phat_tau_hat_samples, gcc_phat_tau_hat_ms
- gcc_phat_psr (peak-to-sidelobe ratio)
- gcc_phat_boundary_hit (peak at search boundary)

Sign convention (required, must be consistent):
- tau_hat > 0 means LDV lags MIC:
  - y[t] ~= x[t - tau_hat]

---

## 7) Required Estimator B: Phase-slope group delay (E4m core)

Goal: estimate tau from the slope of unwrapped cross-spectrum phase and quantify fit quality.

### 7.1 Cross-spectrum definition (required)

Using FFT of the filtered segments:

  X(f) = FFT(x_filt)
  Y(f) = FFT(y_filt)
  C(f) = conj(X(f)) * Y(f)

Use the same sign convention as GCC-PHAT: if y lags x by +tau, then phase slope is negative and:

  phi(f) ~= -2 pi f tau + b
  tau_sec = -slope / (2 pi)

### 7.2 Fit procedure (required)

For a given frequency set {f_i} in a band:
1) Compute phase:
   - phi_i = unwrap(angle(C(f_i)))
2) Weighted linear regression:
   - phi_i ~= a * f_i + b
   - weights w_i = |C(f_i)| (magnitude of cross-power)
3) Convert to delay:
   - tau_sec = -a / (2*pi)
   - tau_ms = 1000 * tau_sec
4) Fit diagnostics (required):
   - phase_slope_fit_rmse_rad: sqrt(mean((phi - (a f + b))^2)) (use weights consistently)
   - phase_slope_r2: 1 - SSE/SST (use weights consistently)
   - phase_slope_num_bins_used

Undefined behavior (no fallback):
- If fewer than `phase_slope_min_bins` valid bins exist in a band, output nulls for that fit and increment counters.

### 7.3 Dispersion via sub-bands (required)

Compute phase-slope tau_ms separately for each sub-band:
- band_0: 300-900 Hz
- band_1: 900-1800 Hz
- band_2: 1800-3000 Hz

Compute dispersion summary per window:
- tau_band_spread_ms = max(tau_ms_by_band) - min(tau_ms_by_band) (if >=2 bands defined; else null)

Interpretation:
- Small spread implies delay is consistent across frequency (near pure delay).
- Large spread implies frequency-dependent delay (dispersion), motivating calibration.

---

## 8) Method Agreement (required)

For each window:
- tau_agreement_ms = phase_slope_tau_hat_ms - gcc_phat_tau_hat_ms (null if either is null)

Interpretation:
- Small |agreement| implies both estimators support a single delay model.
- Large |agreement| can indicate dispersion, multi-path, or low SNR/unreliable estimates.

---

## 9) Outputs (Artifacts and Schemas)

E4m produces all E4h/E4k/E4l summaries PLUS extended sub-sample diagnostics.

### 9.1 Per-window JSONL (required)

Write:
- `results/<run>/subsample_delay_diagnostics.jsonl`

One row per (clip_idx, window_idx, lambda_c, coarse_source), with fields:
- pairing_mode, clip_idx, window_idx, start_t, lambda_c
- coarse_source: "dt" or "omp"
- coarse_lag_frames, coarse_delay_samples
- search_radius_samples

GCC-PHAT fields (required):
- gcc_phat_tau_hat_samples, gcc_phat_tau_hat_ms
- gcc_phat_psr, gcc_phat_boundary_hit

Phase-slope global fields (required in E4m):
- phase_slope_tau_hat_ms
- phase_slope_fit_rmse_rad
- phase_slope_r2
- phase_slope_num_bins_used

Dispersion fields (required in E4m):
- phase_slope_tau_hat_ms_by_band (length 3; nulls allowed)
- phase_slope_r2_by_band (length 3; nulls allowed)
- phase_slope_fit_rmse_rad_by_band (length 3; nulls allowed)
- tau_band_spread_ms

Agreement fields (required in E4m):
- tau_agreement_ms

If undefined, write nulls and include:
- undefined_reason (string; e.g., segment_oob, coarse_undefined, phase_slope_too_few_bins, gcc_phat_undefined)

### 9.2 Summary JSON (required)

Write:
- `results/<run>/summary/subsample_delay_diagnostics_summary.json`

One row per lambda_c, with per coarse_source ("dt","omp") summary statistics:

GCC-PHAT summary:
- fraction_defined
- boundary_hit_rate
- psr_p50/p90
- within_clip_tau_mad_ms_p50/p90

Phase-slope summary:
- phase_slope_fraction_defined
- phase_slope_r2_p50/p90
- phase_slope_fit_rmse_rad_p50/p90
- phase_slope_tau_hat_ms_p50/p90

Agreement summary:
- abs_tau_agreement_ms_p50/p90

Dispersion summary:
- tau_band_spread_ms_p50/p90
- within_clip_tau_band_spread_ms_p50/p90

---

## 10) Hard Guardrails (FAIL conditions)

Inherit E4h/E4k/E4l hard guardrails:
- missing files: FAIL
- md5 mismatches vs manifest: FAIL
- NaN/Inf in capture: FAIL
- capture out-of-range: FAIL
- OMP monotonicity violations: FAIL
- DT duplicates under forced-K: FAIL

E4m additional hard guardrails:
- require_wav_only=1: any non-wav path in subset => FAIL
- fs mismatch (wav header fs != 16000) => FAIL
- subsample_delay_diagnostics.jsonl missing => FAIL
- summary/subsample_delay_diagnostics_summary.json missing => FAIL

---

## 11) Acceptance Criteria (PASS / PASS_WITH_WARNINGS / FAIL)

E4m is a diagnostic stage. Acceptance focuses on:
- pipeline validity,
- compute control + DT non-degeneracy at target compute,
- strong guardrail separation,
- and interpretable dispersion diagnostics.

### 11.1 Paired full_dataset PASS requirements

- All hard guardrails pass.
- RTG controllability holds:
  - spearman(lambda_c, k_selected_mean) <= -0.9
- DT remains non-degenerate at lambda_c=3e-4:
  - DT - Random (compute-matched capture mean) > 0

GCC-PHAT stability at lambda_c=3e-4 (dt coarse):
- fraction_defined >= 0.90
- boundary_hit_rate <= 0.20

Phase-slope fit quality at lambda_c=3e-4 (dt coarse):
- phase_slope_fraction_defined >= 0.80
- phase_slope_r2_p50 >= 0.80
- phase_slope_fit_rmse_rad_p50 <= 1.0

### 11.2 Guardrail separation PASS requirements (scale_check_subset)

Comparing paired vs mispair_shift1 on 48 pairs, for at least 4/5 lambdas (dt coarse):
- psr_p50(paired) > psr_p50(mispair)
- within_clip_tau_mad_ms_p50(paired) < within_clip_tau_mad_ms_p50(mispair)
- phase_slope_r2_p50(paired) > phase_slope_r2_p50(mispair)
- tau_band_spread_ms_p50(paired) < tau_band_spread_ms_p50(mispair)  (or mispair has many undefined)

### 11.3 Readiness decision (not a hard FAIL; required to report)

Based on paired full_dataset (dt coarse) at lambda_c=3e-4:
- If abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms:
  - readiness = READY_FOR_PHASE3 (correctness / geometry-grounded TDoA-like evaluation)
- Else:
  - readiness = READY_FOR_CALIBRATION (phase equalization / dispersion compensation needed)

This is a decision rule, not a pass/fail rule; the report must explain causally (BECAUSE/THEREFORE).

---

## 12) Required Runs (Sequence)

Each run must use lockdir + tee and produce complete artifacts.

1) Preflight: dataset exists, len(dataset)==416, wav-only.
2) Smoke (paired): 1 pair.
3) Functional (scale_check_subset, 48 pairs):
   - paired (positive path)
   - mispair_shift1 (guardrail)
4) Full dataset (paired): 416 pairs.

