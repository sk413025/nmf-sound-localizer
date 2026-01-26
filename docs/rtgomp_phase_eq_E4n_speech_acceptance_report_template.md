# Acceptance Report Template: E4n-Speech -- Phase Equalization Calibration + E4m Re-validation

Use this template for E4n-Speech. Fill all placeholders. Keep everything in English.

E4n has two run types:
- FIT runs (produce phase_eq)
- EVAL runs (apply phase_eq and re-run E4m-style dispersion diagnostics)

Save the filled report to:
- `results/<run>/ACCEPTANCE_REPORT.md`

---

# Acceptance Report: E4n-Speech -- Phase Equalization Calibration + E4m Re-validation

## 1) Executive Summary

- Run: results/<run>/
- Run type: FIT / EVAL
- Mode: smoke / scale_check_subset / full_dataset
- Pairing mode: paired / mispair_shift1
- Outcome: PASS / PASS_WITH_WARNINGS / FAIL
- Readiness decision (EVAL full_dataset only): READY_FOR_PHASE3 / READY_FOR_CALIBRATION / NOT_READY

Primary questions answered:
- Q1 (FIT): Did we estimate a valid, reproducible phase equalizer (unit magnitude, wav-only, stable GCC)?
- Q2 (EVAL): Does compute controllability still hold after applying phase_eq?
- Q3 (EVAL): Does DT remain non-degenerate at target compute (~12) after applying phase_eq?
- Q4 (EVAL): Do dispersion diagnostics improve vs E4m baseline AND remain improved as compute decreases?
- Q5 (EVAL): Does mispair_shift1 remain a strong guardrail failure after applying phase_eq?

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES/NO

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (paste `python --version`)
- Device(s): cpu / mps (and rationale)
- MPLCONFIGDIR: (value used; recommended /tmp/mpl)

### 2.2 Code provenance (Required)

- code_state.json: results/<run>/code_state.json
  - git_head: <hash>
  - dirty: true/false
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: <sha256 or N/A for FIT-only>
    - scripts/h_exploration/fit_phase_eq_e4n_speech.py: <sha256 or N/A for EVAL-only>
    - scripts/h_exploration/dataset_lag.py: <sha256>

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired / mispair_shift1 (describe)
- Dataset length check (preflight output):
  - len(dataset) = <value> (expected 416)
  - first_pair = <path1>, <path2>
  - last_pair  = <path1>, <path2>
- Subset manifest: results/<run>/subset_manifest.json
  - num_pairs = <N>
  - fingerprint_md5 = <hash>
  - Domain validation:
    - all paths end with .wav: YES/NO (must be YES)
    - any .npy paths: YES/NO (must be NO)

### 2.4 Fixed parameters (Must be explicit)

Common:
- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101 (EVAL)
- tw = 32
- search_radius_frames = 2

EVAL-only:
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- subsample_method = gcc_phat,phase_slope
- phase_slope_min_bins = 64
- phase_slope_subbands_hz = [[300,900],[900,1800],[1800,3000]]
- checkpoint: (paste path)

FIT-only:
- nfft_fit = 32768 (expected; must match summary)
- tau_center_samples = 0 (design choice)

### 2.5 Definitions (Must be explicit)

- phase_eq: a per-frequency unit-magnitude complex correction applied to LDV
- phase_eq_stft: length 1025, applied to ldv_stft
- phase_eq_rfft: length 16385 (for nfft_fit=32768), applied inside subsample GCC/phase-slope FFT path
- Pure delay: phi(f) ~= -2 pi f tau + b
- Dispersion: phi(f) non-linear; sub-band taus disagree

## 3) Exact Commands (Required)

Paste the exact command(s) executed, including tee, lockdir pattern, and env exports.

## 4) Results (Required)

### 4.0 Hard guardrails (Must pass)

For all runs:
- wav-only manifest: PASS/FAIL
- fs mismatch encountered: YES/NO (must be NO)
- any NaN/Inf in outputs: YES/NO (must be NO)

FIT-only (phase_eq artifacts):
- phase_eq/phase_eq.npz exists: YES/NO
- phase_eq/phase_eq_fit_summary.json exists: YES/NO

EVAL-only:
- subsample_delay_diagnostics.jsonl exists: YES/NO
- summary/subsample_delay_diagnostics_summary.json exists: YES/NO

Decision:
- PASS/FAIL with causal language.

### 4.1 FIT stage: phase_eq validity (FIT runs only)

Copy from phase_eq/phase_eq_fit_summary.json:
- num_windows_total = <int>
- num_windows_used (tau defined) = <int>
- fraction_defined (tau) = <float>
- boundary_hit_rate = <float>
- gcc_psr_p50/p90 = <values>
- inband_defined_fraction_stft = <float>
- phase_eq_unit_mag_max_err = <float>

Decision:
- PASS/FAIL with causal language.

### 4.2 EVAL stage: RTG controllability (EVAL runs only)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = <value>
- k_selected_mean list = <list>

Decision:
- PASS/FAIL with causal language.

### 4.3 EVAL stage: non-degeneracy at target compute (EVAL runs only)

At lambda_c = 3e-4:
- k_selected_mean = <value>
- DT - Random (compute-matched capture mean) = <value> (must be > 0 for PASS)

Decision:
- PASS/FAIL with causal language.

### 4.4 EVAL stage: dispersion improvement vs baseline (EVAL full_dataset paired only)

Baseline reference (required):
- E4m baseline commit hash: 18a82aa (or a later explicitly stated baseline)
- Baseline run directory (must exist in git history): results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/

Report baseline (lambda_c=3e-4, dt coarse):
- phase_slope_r2_p50 = <value>
- phase_slope_fit_rmse_rad_p50 = <value>
- tau_band_spread_ms_p50 = <value>
- abs_tau_agreement_ms_p50 = <value>

Report calibrated (this run, lambda_c=3e-4, dt coarse):
- phase_slope_r2_p50 = <value>
- phase_slope_fit_rmse_rad_p50 = <value>
- tau_band_spread_ms_p50 = <value>
- abs_tau_agreement_ms_p50 = <value>

Compute-robust improvement across lambdas (dt coarse):
- Provide a 5-row table for the calibrated run with columns:
  - lambda_c
  - phase_slope_r2_p50
  - phase_slope_fit_rmse_rad_p50
  - tau_band_spread_ms_p50
  - abs_tau_agreement_ms_p50

Decision:
- PASS / PASS_WITH_WARNINGS / FAIL with causal language (BECAUSE/THEREFORE).

### 4.5 Guardrail separation after calibration (EVAL functional runs, 48 pairs)

Compare paired vs mispair_shift1 after applying phase_eq (dt coarse), for >=4/5 lambdas:
- psr_p50(paired) > psr_p50(mispair)
- within_clip_tau_mad_ms_p50(paired) < within_clip_tau_mad_ms_p50(mispair)
- phase_slope_fraction_defined(paired) > phase_slope_fraction_defined(mispair)

Decision:
- PASS/FAIL with causal language.

### 4.6 Readiness decision (EVAL full_dataset paired only)

Apply readiness rule at lambda_c=3e-4 (dt coarse):
- READY_FOR_PHASE3 if:
  - abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms
- READY_FOR_CALIBRATION otherwise.

Explain causally (BECAUSE/THEREFORE) which condition failed and why it implies calibration is still needed.

## 5) Physical / Mathematical Analysis (Required)

Explain from first principles:
- Why pure delay implies linear phase and why phase equalization can remove residual phase distortion.
- Why dispersion causes sub-band delay disagreement and poor phase-slope fit.
- Why calibration must preserve the mispair guardrail separation (avoid “masking” incorrect pairing).

Use causal phrases:
- BECAUSE / DUE TO / THEREFORE / THIS IMPLIES

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference at least 3 Results commits (hashes), e.g.:
- E4h-Speech paper eval (6af8a63)
- E4j-Speech compute calibration (db556db)
- E4l-Speech GCC-PHAT refinement (9e6e402)
- E4m-Speech dispersion diagnostics baseline (18a82aa)

Connect patterns causally.

## 7) Extracted Principles for Next Steps (Required)

Convert observations into rules:
- If calibration reduces dispersion robustly across compute, THEN proceed to Phase-3 correctness evaluation.
- If calibration helps but not enough, THEN refine equalizer design (smoothing/regularization/band-limited model).

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- Paste exact command(s).

3) Verification:
- FIT runs must contain phase_eq artifacts.
- EVAL runs must contain the full E4m artifact set plus references to the phase_eq used.

## 9) Failures / Limitations (Required even if PASS)

- Explain what calibration cannot fix (multi-path, non-minimum-phase behavior, time-varying transfer).
- Any trade-offs observed between compute reduction and dispersion stability.
