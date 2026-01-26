# Acceptance Report Template: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

Use this template for E4m-Speech. Fill all placeholders. Keep everything in English.

You will typically write **three reports**:
- smoke paired (sanity)
- functional paired (positive path) + functional mispair_shift1 (guardrail)
- full_dataset paired (main)

Save the filled report to:
- `results/<run>/ACCEPTANCE_REPORT.md`

---

# Acceptance Report: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

## 1) Executive Summary

- Run: results/<run>/
- Mode: smoke / scale_check_subset / full_dataset
- Pairing mode: paired / mispair_shift1
- Outcome: PASS / PASS_WITH_WARNINGS / FAIL
- Readiness decision: READY_FOR_PHASE3 / READY_FOR_CALIBRATION / NOT_READY

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability)
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda)
- Q3: Does GCC-PHAT produce stable tau estimates on paired data under compute control?
- Q4: Does phase-slope fit indicate near-pure delay or dispersion (non-linear phase)?
- Q5: Do paired vs mispair_shift1 runs separate strongly in confidence and dispersion metrics?

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
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: <sha256>
    - scripts/h_exploration/dataset_lag.py: <sha256>

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired / mispair_shift1 (describe)
- Dataset length check (preflight output):
  - len(dataset) = <value>
  - first_pair = <path1>, <path2>
  - last_pair  = <path1>, <path2>
- Subset manifest: results/<run>/subset_manifest.json
  - num_pairs = <N>
  - fingerprint_md5 = <hash>
  - Domain validation:
    - all paths end with .wav: YES/NO (must be YES)
    - any .npy paths: YES/NO (must be NO)

### 2.4 Fixed parameters (Must be explicit)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- search_radius_frames = 2
- subsample_method = gcc_phat,phase_slope
- phase_slope_min_bins = 64
- phase_slope_subbands_hz = [[300,900],[900,1800],[1800,3000]]

### 2.5 Definitions (Must be explicit)

- k_selected: number of selected lag atoms (compute-aligned)
- steps_decision: number of decision steps including STOP (policy-step-aligned)
- coarse_lag_frames: median first-lag across frequency (DT or OMP)
- tau_hat_samples: delay estimate sign convention (define): tau_hat > 0 means LDV lags MIC (y[t] ~= x[t - tau_hat])
- psr: peak-to-sidelobe ratio (confidence)
- boundary_hit: peak at search boundary (confidence failure)
- phase_slope_tau_hat_ms: group delay estimate from linear fit of unwrapped cross-spectrum phase
- phase_slope_fit_rmse_rad, phase_slope_r2: fit quality (dispersion signature)
- tau_agreement_ms: phase_slope_tau_hat_ms - gcc_phat_tau_hat_ms
- tau_band_spread_ms: max(tau_ms_by_band) - min(tau_ms_by_band)

## 3) Exact Commands (Required)

Paste the exact command(s) executed, including tee, lockdir pattern, and env exports.

## 4) Results (Required)

All numeric fields below must be copied from:
- summary/compute_matched_summary.json
- summary/rtg_controllability_summary.json
- summary/subsample_delay_diagnostics_summary.json
- run.log

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_missing_files = <int>
- num_md5_mismatches = <int>
- num_nan_or_inf = <int>
- num_capture_out_of_range_total = <int>
- num_omp_monotonicity_violations = <int>
- num_dt_duplicate_actions_forced_k = <int>

E4m-specific hard checks:
- any fs mismatch encountered: YES/NO (must be NO; otherwise FAIL)
- subsample_delay_diagnostics.jsonl exists: YES/NO
- summary/subsample_delay_diagnostics_summary.json exists: YES/NO

Decision:
- PASS/FAIL with causal language.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = <value>
- k_selected_mean list = <list>

Decision:
- PASS/FAIL with causal language.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = <value>
- DT - Random (compute-matched capture mean) = <value> (must be > 0 for PASS)

Decision:
- PASS/FAIL with causal language.

### 4.3 GCC-PHAT stability (Required)

From summary/subsample_delay_diagnostics_summary.json, report at minimum for coarse_source="dt":

At lambda_c=3e-4:
- fraction_defined = <value>
- boundary_hit_rate = <value>
- psr_p50/p90 = <values>
- within_clip_tau_mad_ms_p50/p90 = <values>

Decision:
- PASS/FAIL with causal language.

### 4.4 Phase-slope fit quality and dispersion (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt"):

At lambda_c=3e-4:
- phase_slope_fraction_defined = <value>
- phase_slope_r2_p50/p90 = <values>
- phase_slope_fit_rmse_rad_p50/p90 = <values>
- abs_tau_agreement_ms_p50/p90 = <values>
- tau_band_spread_ms_p50/p90 = <values>

Interpretation hints (use causal language):
- If r2 is high and rmse is low, THEN phase is approximately linear in f (near pure delay).
- If r2 is low and tau_band_spread is high, THEN dispersion / non-linear phase is present.
- If GCC-PHAT is stable but phase-slope is poor, THEN delay may be stable-but-biased, motivating calibration.

### 4.5 Guardrail separation (paired vs mispair_shift1; Required)

Compare paired vs mispair on scale_check_subset (48 pairs), for dt coarse, for >=4/5 lambdas:
- psr_p50(paired) > psr_p50(mispair)
- within_clip_tau_mad_ms_p50(paired) < within_clip_tau_mad_ms_p50(mispair)
- phase_slope_r2_p50(paired) > phase_slope_r2_p50(mispair) (or mispair mostly undefined)
- tau_band_spread_ms_p50(paired) < tau_band_spread_ms_p50(mispair)

Decision:
- PASS if separation holds.
- PASS_WITH_WARNINGS if paired passes but separation is weak (must analyze why).

### 4.6 Readiness decision (Required)

Apply the decision rule at lambda_c=3e-4 (paired full_dataset, dt coarse):
- READY_FOR_PHASE3 if:
  - abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms
- READY_FOR_CALIBRATION otherwise.

Explain causally (BECAUSE/THEREFORE) which condition failed and why it implies calibration.

## 5) Physical / Mathematical Analysis (Required)

Explain from first principles:
- Why a pure delay induces linear cross-spectrum phase: phi(f) ~= -2 pi f tau.
- Why dispersion makes phi(f) non-linear and reduces linear-fit quality (low R^2, high RMSE).
- Why GCC-PHAT can remain stable even when phase-slope indicates dispersion (stable-but-biased risk).

Use causal phrases:
- BECAUSE / DUE TO / THEREFORE / THIS IMPLIES

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference at least 3 Results commits (hashes) and connect patterns causally.

Suggested minimum set:
- E4h-Speech paper eval (DT vs OMP vs Random)
- E4j-Speech stop/cost calibration (compute ~12 at lambda=3e-4)
- E4l-Speech sub-sample (GCC-PHAT) stability

## 7) Extracted Principles for Next Steps (Required)

Convert observations into rules:
- If dispersion is low and agreement is strong, THEN proceed to Phase-3 correctness evaluation.
- If dispersion is high or agreement is poor, THEN prioritize calibration design (phase equalization) before TDoA.

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
- Expected outputs:
  - subset_manifest.json (wav-only)
  - subsample_delay_diagnostics.jsonl
  - summary/subsample_delay_diagnostics_summary.json
  - summary/compute_matched_summary.json
  - summary/rtg_controllability_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json
  - ACCEPTANCE_REPORT.md

## 9) Failures / Limitations (Required even if PASS)

- Estimator limitations (phase unwrap sensitivity, multi-path peaks).
- Lack of geometry ground truth (stability does not imply correctness).
- Any compute-vs-stability trade-offs observed.

