# Acceptance Report: E4j-Speech -- Scale-Check Subset (DT vs OMP vs Random)

## 1) Executive Summary

- Run (scale_check_subset): results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stepwise_freezebn_stopw0p025_20260124_021119/
- Outcome: PASS
- Dataset domain statement:
  - This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup

### 2.1 Environment

- Conda env: trl-training
- Python: 3.11.13
- Device(s): cpu
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stepwise_freezebn_stopw0p025_20260124_021119/code_state.json
  - git_head: 6af8a630c82d93dbbf66fd61a7a52706dc8c8787
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 0442f1b160c46253e6b5e96a6109be235fff9dd3ab3a3a8baf7dc61ee1d8b058
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Subset manifest: results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stepwise_freezebn_stopw0p025_20260124_021119/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters

- hop_length = 160
- fs = 16000
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- Tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- use_stop_action = true
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

## 3) Exact Commands

See:
- results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stepwise_freezebn_stopw0p025_20260124_021119/run.log

## 4) Results

### 4.0 Integrity

From summary/compute_matched_summary.json integrity:
- num_samples_total = 80964
- num_samples_used = 80964
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0
- random_duplicate_rate (paper baseline) = 0.0

Decision:
- PASS because all hard guardrails are satisfied.

### 4.1 RTG controllability (Free rollout)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_range = 4.6039
- k_selected_mean @ 1e-5 = 15.8545
- k_selected_mean @ 3e-4 = 11.2506

Decision:
- PASS because compute decreases monotonically with lambda_c and the mean shift is material.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP)

Key endpoints:
- lambda=1e-5: DT-Random = 0.001033, k_selected_mean = 15.8545
- lambda=3e-4: DT-Random = 0.037618, k_selected_mean = 11.2506

Decision:
- PASS because DT-Random > 0 at both low and high penalty.

## 5) Interpretation (Causal)

- The scale-check subset reproduces the full-dataset trend BECAUSE STOP/cost behavior is calibrated in evaluation-style stepwise mode; THEREFORE the compute curve is stable when moving from smoke to 48 pairs.

## 6) Reproduction

1) Environment:
- source ~/.zshrc
- conda activate trl-training
- export PYTHONPATH=.
- export MPLCONFIGDIR=/tmp/mpl

2) Execution:
- run the exact command in run.log

3) Verification:
- Confirm num_capture_out_of_range_total == 0 and k_selected_mean @ 3e-4 ~ 11.25.

## 7) Next Steps

- Proceed to (or compare against) full_dataset run BECAUSE scale_check_subset passed all guardrails and demonstrates stable RTG controllability.
