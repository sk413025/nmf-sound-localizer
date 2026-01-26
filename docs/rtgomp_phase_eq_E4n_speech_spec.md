# Spec: E4n-Speech -- Phase Equalization Calibration (Phase-2d) + E4m Re-validation Under Compute Control

This spec defines **E4n-Speech (Phase-2d)**.

E4n-Speech follows:
- **E4l-Speech**: GCC-PHAT sub-sample delay refinement stable on paired speech; strong failure under mispair guardrail.
- **E4m-Speech**: phase-slope (group delay) + dispersion diagnostics showed **stable GCC-PHAT** but **poor phase-slope fit** and **large band-spread**, implying dispersion / frequency-dependent phase.

E4n-Speech adds **calibration**:

> Estimate a reproducible **phase equalization** (all-pass correction) that removes the **frequency-dependent phase residual** of the LDV path, then re-run the **E4m evaluator** with the same 5-point lambda grid to verify (1) compute control still works and (2) dispersion metrics improve and remain improved as compute decreases.

All content and artifacts MUST be in English.

Implementation targets:
- New calibration fitter (E4n): `scripts/h_exploration/fit_phase_eq_e4n_speech.py`
- Extend evaluator to apply calibration (E4n->E4m revalidation):
  - `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

---

## 0) Why This Exists (Phase Context)

Long-term goal: **MIC + LDV time alignment** for downstream TDoA-like tasks.

E4m-Speech indicated:
- GCC-PHAT can be stable under compute control,
- BUT phase-slope fit is poor and sub-band delays disagree,
BECAUSE the LDV path likely introduces **frequency-dependent phase** (dispersion / resonance / phase wrapping).

Therefore we must **calibrate** before claiming a single physical time delay tau is meaningful.

E4n-Speech answers:
- Can a simple, reproducible **phase equalizer** reduce dispersion diagnostics?
- Does this improvement persist when compute is reduced (lambda grid)?
- Does the mispair guardrail still fail strongly after calibration (to avoid \"false good\")?

---

## 1) Minimal Primer (No Background Assumed)

### 1.1 Pure delay => linear phase

If LDV is (approximately) a delayed MIC:

  y(t) ≈ x(t - tau)

Then in frequency domain:

  Y(f) ≈ X(f) * exp(-j 2 pi f tau)

Define the cross-spectrum:

  C(f) = conj(X(f)) * Y(f)

Then the phase:

  phi(f) = angle(C(f)) ≈ -2 pi f tau + b

This is a straight line vs frequency.

### 1.2 Dispersion => non-linear phase

If the LDV path has frequency-dependent phase:

  Y(f) ≈ X(f) * exp(-j 2 pi f tau) * exp(+j phi_residual(f))

Then:
- phi(f) is not well explained by one straight line,
- different sub-bands produce different implied delays,
BECAUSE phi_residual(f) bends the phase curve.

### 1.3 Phase equalization (what we do in E4n)

We estimate a per-frequency complex unit-magnitude correction:

  G(f) = exp(-j phi_hat_residual(f))

and apply it to LDV:

  Y_eq(f) = Y(f) * G(f)

If G(f) matches the residual, THEN the corrected cross-spectrum phase becomes closer to linear, and the phase-slope diagnostics should improve.

This is an **all-pass** correction (phase-only). It should not change magnitudes.

---

## 2) Dataset and Hard Guardrails

### 2.1 Speech WAV-only roots (required)

Allowed roots (do not change):
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

Hard guardrails:
- Real data only.
- `require_wav_only=1`: any non-`.wav` path => FAIL.
- WAV sample rate must be exactly `fs=16000`: any mismatch => FAIL (no resampling).

### 2.2 Pairing modes

For E4n calibration FIT:
- Use `pairing_mode=paired` only (calibration requires correct pairing).

For E4n re-validation (E4m-style eval with apply):
- Run `paired` (positive path).
- Run `mispair_shift1` (guardrail diagnostic).

---

## 3) Fixed Parameters (Must Match E4m)

Common parameters:
- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2

Compute knob (same 5-point grid; exact values):
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

Sub-sample refinement / diagnostics (same as E4m):
- search_radius_frames = 2 (=> +/- 20ms around coarse)
- subsample_method = gcc_phat,phase_slope
- phase_slope_min_bins = 64
- phase_slope_subbands_hz = [[300,900],[900,1800],[1800,3000]]

Checkpoint (DT; required for eval runs):
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

---

## 4) E4n Component A: Phase Equalization Fitting (Calibration)

### 4.1 Inputs

Calibration fitter MUST:
- operate on real speech WAV only,
- use the same STFT alignment convention as E4m (window indexing in frames),
- compute a per-frequency correction vector `phase_eq[f]` (unit-magnitude complex).

### 4.2 Window definition (must match E4m)

Let `valid_starts` match evaluator:
- `start_limit = max_lag`
- `end_limit = T_total - tw + (-max_lag)` (Lag_Min = -max_lag)
- iterate `start_t` in `range(start_limit, end_limit, tw)`

### 4.3 Estimating the residual phase

For each window i:
1) Extract the waveform segment for the window using the same padded indexing as E4m.
2) Bandpass filter x and y to [300,3000] Hz with zero-phase filtfilt.
3) Estimate a coarse delay tau_i using GCC-PHAT peak picking within a fixed window:
   - tau_center_samples = 0 (design choice for fit stage; actual delays are near 0 in E4m)
   - search_radius_samples = search_radius_frames * hop_length
   - sign convention: tau > 0 means LDV lags MIC (y[t] ~= x[t - tau])
4) Compute FFT cross-power on a fixed FFT grid:
   - Choose nfft_fit = next_pow2(4 * segment_len_samples) (design choice: match evaluator worst-case FFT length)
   - For tw=32, hop=160, n_fft=2048: segment_len_samples=(tw-1)*hop + n_fft = 7008, so nfft_fit = 32768.
   - Hard guardrail (required): `nfft_fit % n_fft == 0`. Otherwise FAIL (we require exact bin alignment).
   - Compute X, Y via rFFT with nfft_fit.
   - cross_power C(f) = conj(X(f)) * Y(f)
5) Remove the delay term:
   - Define u_i(f) = (C(f) / |C(f)|) * exp(+j 2 pi f tau_i_sec)
   - This \"rotates\" C(f) by the estimated pure-delay phase so only residual phase remains.
6) Aggregate across windows with weights:
   - weights w_i(f) = |C(f)|
   - H_hat(f) = sum_i w_i(f) * u_i(f) / sum_i w_i(f)
7) Convert to phase equalizer:
   - phase_eq(f) = conj(H_hat(f) / |H_hat(f)|)

Notes:
- This is a deterministic weighted circular mean in the complex plane.
- It is all-pass (unit magnitude) by construction.
- Out-of-band bins (outside [300,3000] Hz) MUST use phase_eq=1+0j.
- Mapping to STFT bins must be exact: let `stride = nfft_fit // n_fft` and take `phase_eq_stft[k] = phase_eq_rfft[stride * k]`.

### 4.4 Fit-stage outputs (required artifacts)

The fitter MUST write under `results/<run>/phase_eq/`:
- `phase_eq.npz` (required):
  - freqs_hz: float64, length n_fft//2+1 (STFT bin grid, 1025 bins)
  - phase_eq_stft: complex64, length 1025, unit magnitude
  - defined_mask_stft: bool, length 1025
- nfft_fit: int (should be 32768)
- phase_eq_rfft: complex64, length nfft_fit//2+1 (16385), unit magnitude
  - defined_mask_rfft: bool, length nfft_fit//2+1
- `phase_eq_fit_summary.json` (required): key stats including:
  - num_pairs, num_windows_total, num_windows_used (tau defined)
  - gcc_psr_p50/p90, boundary_hit_rate
  - inband_defined_fraction_stft (freq in [300,3000])
  - phase_eq_unit_mag_max_err (max | |phase_eq|-1 |)
  - any_nan_or_inf: bool
- `subset_manifest.json`, `run.log`, and `code_state.json` (same standard as other runs).

Hard failures:
- Any NaN/Inf in phase_eq arrays => FAIL.
- Any in-band stft bin undefined beyond a threshold (see acceptance) => FAIL.

---

## 5) E4n Component B: Apply Calibration + Re-run E4m (Re-validation)

### 5.1 Applying phase_eq consistently

Re-validation MUST use the existing evaluator (E4m diagnostics) but with calibration applied:

- Apply to STFT inputs:
  - ldv_stft[t,f] := ldv_stft[t,f] * phase_eq_stft[f]

- Apply to waveform-domain subsample diagnostics:
  - When computing rFFT(Y) for GCC-PHAT / phase-slope, multiply:
    - Y(f) := Y(f) * phase_eq_rfft[f]
  - This ensures `subsample_delay_diagnostics.jsonl` reflects the calibrated LDV signal.

Hard guardrails:
- The loaded phase_eq must match expected shapes (1025 STFT bins; 8193 rFFT bins for nfft_fit=16384). Any mismatch => FAIL.
- phase_eq must be unit magnitude within tolerance (e.g., max | |eq|-1 | <= 1e-3). Otherwise => FAIL.

### 5.2 Evaluation outputs

Re-validation runs MUST produce the full E4m artifact set:
- `summary/compute_matched_summary.json`
- `summary/rtg_controllability_summary.json`
- `subsample_delay_diagnostics.jsonl`
- `summary/subsample_delay_diagnostics_summary.json`
- plus the standard manifest/log/integrity/code_state/ACCEPTANCE_REPORT.md.

---

## 6) Acceptance Criteria (PASS / PASS_WITH_WARNINGS / FAIL)

E4n has two stages: FIT and RE-VALIDATION.

### 6.1 Fit-stage acceptance (paired)

PASS requirements:
- All hard guardrails pass (wav-only; fs=16000; no NaN/Inf).
- GCC-PHAT fit stability (sanity check):
  - fraction_defined >= 0.90
  - boundary_hit_rate <= 0.20
- Phase-eq validity:
  - inband_defined_fraction_stft >= 0.98
  - phase_eq_unit_mag_max_err <= 1e-3

### 6.2 Re-validation (paired full_dataset) PASS requirements

The calibrated evaluator run MUST satisfy E4m baseline validity:
- All hard guardrails pass.
- RTG controllability holds:
  - spearman(lambda_c, k_selected_mean) <= -0.9
- DT remains non-degenerate at lambda_c=3e-4:
  - DT - Random (compute-matched capture mean) > 0
- GCC-PHAT stability at lambda_c=3e-4 (dt coarse):
  - fraction_defined >= 0.90
  - boundary_hit_rate <= 0.20

Dispersion improvement targets (dt coarse):
- At lambda_c=3e-4:
  - phase_slope_r2_p50 >= 0.75
  - phase_slope_fit_rmse_rad_p50 <= 2.0
  - tau_band_spread_ms_p50 <= 2.0 ms
  - abs_tau_agreement_ms_p50 <= 1.0 ms

Compute-robust improvement (dt coarse):
- For at least 4/5 lambdas:
  - phase_slope_r2_p50 is higher than the E4m baseline (commit 18a82aa) AND
  - tau_band_spread_ms_p50 is lower than the E4m baseline.

### 6.3 Guardrail separation (paired vs mispair_shift1) PASS requirements

On scale_check_subset (48 pairs), for at least 4/5 lambdas (dt coarse):
- psr_p50(paired) > psr_p50(mispair)
- within_clip_tau_mad_ms_p50(paired) < within_clip_tau_mad_ms_p50(mispair)
- phase_slope_fraction_defined(paired) > phase_slope_fraction_defined(mispair)

Rationale:
- Calibration must not make mispair appear aligned; otherwise it is overfitting / masking.

### 6.4 Readiness decision (reported; not a hard FAIL)

Use the same readiness rule as E4m at lambda_c=3e-4 (paired full_dataset, dt coarse):
- READY_FOR_PHASE3 if:
  - abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms
- READY_FOR_CALIBRATION otherwise.

E4n is successful calibration progress if it moves metrics materially toward READY_FOR_PHASE3 while keeping compute control and DT>Random.

---

## 7) Required Runs (Sequence)

All runs must use lockdir + tee and write artifacts under `results/<run>/`.

1) Preflight: roots exist; len(dataset)==416; wav-only.

2) FIT runs (paired only):
   - smoke (num_pairs=1)
   - scale_check_subset (num_pairs=48)
   - full_dataset (num_pairs=416) => produces the phase_eq used for re-validation

3) RE-VALIDATION runs (apply phase_eq from fit full_dataset):
   - smoke (paired; num_pairs=1)
   - functional scale_check_subset (paired; 48)
   - functional scale_check_subset (mispair_shift1; 48)
   - full_dataset (paired; 416)

Each run directory must contain a filled `ACCEPTANCE_REPORT.md` in English with BECAUSE/THEREFORE causal interpretation.
