# Acceptance Report: E4k-Speech -- Delay & Phase-Consistency Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/
- Mode: scale_check_subset (48 pairs)
- Pairing mode: mispair_shift1
- Outcome: PASS (guardrail diagnostic)

Primary questions answered:
- Q1: Is the pipeline valid under deliberate mispairing (still real WAV)? YES (hard guardrails pass).
- Q2: Does mispairing degrade delay stability as expected? YES (first-lag dispersion explodes vs paired).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (not recorded in this report)
- Device(s): cpu (stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/code_state.json
  - git_head: db556db5f82237298697eea85a099326cf4daad1
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: eb2a09bfc5db9c844ab3c1d1af912ec0303b0cde451b2402abaa9749de0e4153
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: mispair_shift1
  - MIC order is unchanged
  - LDV list is shifted by +1 (cyclic wrap), deliberately breaking alignment
- Subset manifest: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 5997f1271e9a08d833083897c5c41fc1
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO
  - Example of deliberate mispair (first MIC file paired with second LDV file):
    - boy1_papercup_MIC_001.wav -> boy1_papercup_LDV_002.wav

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

## 3) Exact Commands (Required)

Recorded in:
- results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/run.log

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0

Decision:
- PASS because even under mispairing, the least-squares projection remains valid and all guardrails pass.

### 4.1 RTG controllability (Compute vs lambda_c; Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0

Decision:
- PASS because compute remains controllable even when pairing is wrong.

### 4.2 Delay / phase-consistency diagnostics (Guardrail; Required)

From summary/delay_diagnostics_summary.json (mispair_shift1):
- dt_first_lag_mad_frames_p50 is very large (~23 frames) across the grid:
  - 1e-5: 22.919
  - 3e-5: 22.917
  - 1e-4: 23.083
  - 2e-4: 23.086
  - 3e-4: 23.409
- dt_stop0_frac_mean = 0.0 (no immediate STOP collapse; the instability is in lag dispersion, not in STOP0).

Comparison to the paired subset run:
- results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/
- At lambda_c = 3e-4:
  - paired dt_first_lag_mad_p50 = 1.006
  - mispair dt_first_lag_mad_p50 = 23.409

Decision:
- PASS because the mispair guardrail strongly degrades delay stability, which validates that the diagnostics are
  sensitive to correct pairing.

## 5) Physical / Mathematical Analysis (Required)

- When MIC and LDV are correctly paired, their STFT windows correspond to the same underlying speech content; therefore
  a small set of lags should align them consistently across frequency.
- Under mispair_shift1, MIC and LDV windows come from different utterances/segments, so their phase/alignment is not
  physically meaningful; THEREFORE different frequency bins will select unrelated lags, producing very large lag
  dispersion (high MAD).

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Not included in this guardrail-only report. Cross-experiment analysis is performed in the full_dataset acceptance report.

## 7) Extracted Principles for Next Steps (Required)

- Always run this mispair guardrail before expensive full_dataset evaluations: if mispair does NOT get much worse than
  paired, then either the metric is broken or the dataset pairing is wrong.

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
  - subset_manifest.json (wav-only; shows mispair)
  - delay_diagnostics.jsonl
  - summary/delay_diagnostics_summary.json
  - summary/compute_matched_summary.json
  - summary/rtg_controllability_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json

## 9) Failures / Limitations (Required even if PASS)

- This run is not intended to produce good capture; it is a guardrail diagnostic.
- Frame-level lag resolution is 10ms; it is not a final TDoA.

