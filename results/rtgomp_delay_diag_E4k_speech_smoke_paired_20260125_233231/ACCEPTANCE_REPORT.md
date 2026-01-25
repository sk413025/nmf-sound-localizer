# Acceptance Report: E4k-Speech -- Delay & Phase-Consistency Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_delay_diag_E4k_speech_smoke_paired_20260125_233231/
- Mode: smoke
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? YES (Spearman=-1.0 on this smoke sample)
- Q2: Does DT remain non-degenerate at the target compute regime? YES (DT-Random>0 at lambda=3e-4)
- Q3: Do we successfully emit delay diagnostics artifacts? YES (delay_diagnostics.jsonl + summary JSON present)
- Q4: Guardrail separation (paired vs mispair) is not evaluated in this smoke run (see scale_check_subset runs).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (not recorded in this smoke report)
- Device(s): cpu (stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_delay_diag_E4k_speech_smoke_paired_20260125_233231/code_state.json
  - git_head: db556db5f82237298697eea85a099326cf4daad1
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: eb2a09bfc5db9c844ab3c1d1af912ec0303b0cde451b2402abaa9749de0e4153
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired (MIC->LDV filename replacement)
- Dataset length check (preflight output; same machine):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_delay_diag_E4k_speech_smoke_paired_20260125_233231/subset_manifest.json
  - num_pairs = 1
  - fingerprint_md5 = 9b169d8cb234f30431fc7178d5aafb33
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (Must be explicit)

- fs = 16000
- hop_length = 160 (10ms per frame)
- n_fft = 2048
- freq band = [freq_min, freq_max] = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- Random baseline: without_replacement

## 3) Exact Commands (Required)

Recorded in:
- results/rtgomp_delay_diag_E4k_speech_smoke_paired_20260125_233231/run.log

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_samples_total = 1730
- num_samples_used = 1730
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0

Decision:
- PASS because all hard guardrails are satisfied (no missing files, no md5 mismatches, no NaN/Inf, and capture stayed
  within [0,1]).

### 4.1 RTG controllability (Compute vs lambda_c; Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean = [15.9, 15.43, 14.00, 12.75, 11.98]

Decision:
- PASS because compute decreases monotonically with increasing lambda_c.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4 (summary/compute_matched_summary.json):
- k_selected_mean = 11.976
- DT - Random (compute-matched capture mean) = +0.0348
- DT/OMP = 0.9954

Decision:
- PASS because DT remains better than Random at the target compute regime.

### 4.3 Delay / phase-consistency diagnostics (Required)

At lambda_c = 3e-4 (summary/delay_diagnostics_summary.json):
- dt_first_lag_mad_frames_p50 = 2.0
- dt_stop0_frac_mean = 0.0
- dt_vs_omp_first_lag_match_frac_mean = 0.601

Interpretation:
- DT first-lag dispersion is low (few frames) on this smoke clip, which is consistent with a stable coarse alignment.

## 5) Physical / Mathematical Analysis (Required)

- A pure delay corresponds to approximately linear phase vs frequency, which implies a roughly constant group delay.
- If the LDV channel exhibits frequency-dependent phase distortions (dispersion), different frequency bins will prefer
  different effective lags; THEREFORE first-lag dispersion (MAD across frequency) should increase.
- In this smoke run, dispersion is low, which is consistent with a coarse-delay-like behavior at least for this clip.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Not performed in this smoke report. Cross-experiment analysis is performed in the full_dataset acceptance report.

## 7) Extracted Principles for Next Steps (Required)

- If paired full_dataset shows low dispersion across lambda, THEN Phase-2b should implement sub-sample delay refinement.
- If dispersion increases sharply with lambda, THEN STOP/cost calibration likely reduces compute by sacrificing
  frequency-consistent alignment, therefore Phase-2b must re-balance the cost or add explicit delay-consistency signals.

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- See results/.../run.log for the exact command.

3) Verification:
- Required outputs exist:
  - subset_manifest.json (wav-only)
  - integrity_diagnostics.jsonl
  - delay_diagnostics.jsonl
  - summary/delay_diagnostics_summary.json
  - summary/compute_matched_summary.json
  - summary/forced_k_summary.json
  - summary/rtg_controllability_summary.json
  - code_state.json

## 9) Failures / Limitations (Required even if PASS)

- This is a smoke run (1 pair). It only verifies the pipeline works and produces diagnostics.
- Frame-level lag resolution is 10ms; it is not a final TDoA.

