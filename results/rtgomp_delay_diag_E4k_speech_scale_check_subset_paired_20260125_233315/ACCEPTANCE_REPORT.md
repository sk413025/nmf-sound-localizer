# Acceptance Report: E4k-Speech -- Delay & Phase-Consistency Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/
- Mode: scale_check_subset (48 pairs)
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? YES (Spearman=-1.0 on this subset)
- Q2: Does DT remain non-degenerate at the target compute regime? YES (DT-Random>0 at lambda=3e-4)
- Q3: Is first-lag dispersion small (frequency-consistent) and does it remain small as compute decreases? YES on this
  subset (MAD stays ~1 frame at p50 across the lambda grid).
- Q4: Does the mispair guardrail break delay stability? YES (see comparison to mispair run below).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (not recorded in this report)
- Device(s): cpu (stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/code_state.json
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
- Subset manifest: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
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
- results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/run.log

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
- PASS because all hard guardrails are satisfied (stable least-squares behavior, capture within [0,1]).

### 4.1 RTG controllability (Compute vs lambda_c; Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0

Decision:
- PASS because compute decreases monotonically with increasing lambda_c.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4 (summary/compute_matched_summary.json):
- k_selected_mean = 11.948
- DT - Random (compute-matched capture mean) = +0.0293
- DT/OMP = 0.9957

Decision:
- PASS because DT remains better than Random at the target compute regime.

### 4.3 Delay / phase-consistency diagnostics (Required)

From summary/delay_diagnostics_summary.json (paired):
- dt_first_lag_mad_frames_p50 stays ~1 frame across the grid:
  - 1e-5: 0.999
  - 3e-5: 1.147
  - 1e-4: 1.194
  - 2e-4: 1.009
  - 3e-4: 1.006
- dt_stop0_frac_mean = 0.0 across the grid (no immediate STOP collapse on this subset).

Interpretation:
- First-lag dispersion remains low even as mean compute decreases, which suggests the coarse alignment signal is
  robust to compute reduction on this subset.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Compare to mispair run:
- results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/

At lambda_c = 3e-4:
- paired dt_first_lag_mad_p50 = 1.006
- mispair dt_first_lag_mad_p50 = 23.409

Decision:
- PASS because mispair has dramatically worse delay stability than paired across all 5 lambdas, which indicates the
  diagnostics are sensitive to correct pairing.

## 5) Physical / Mathematical Analysis (Required)

- A near-pure delay implies cross-phase is approximately linear with frequency (constant group delay), therefore a
  lag-based proxy should be consistent across frequency bins.
- Material resonances / dispersion in the LDV path introduce frequency-dependent phase, which should manifest as
  frequency-dependent preferred lags (large dispersion).
- The paired subset shows low dispersion (~1 frame MAD), which is consistent with the existence of a stable coarse
  alignment component, despite any finer-scale phase wrapping that may exist.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Not included in this subset report. Cross-experiment analysis is performed in the full_dataset acceptance report.

## 7) Extracted Principles for Next Steps (Required)

- Since paired delay dispersion stays low while compute decreases on this subset, Phase-2b can attempt sub-sample delay
  refinement (e.g., GCC-PHAT / phase-slope) without immediately re-tuning the cost scale.
- The guardrail separation is strong, THEREFORE these diagnostics are suitable for catching pairing/domain mistakes
  before investing in full TDoA experiments.

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
  - delay_diagnostics.jsonl
  - summary/delay_diagnostics_summary.json
  - summary/compute_matched_summary.json
  - summary/rtg_controllability_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json

## 9) Failures / Limitations (Required even if PASS)

- This is a 48-pair subset; results may differ on full_dataset.
- Frame-level lag resolution is 10ms; it is not a final TDoA.

