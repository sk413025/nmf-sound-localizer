# Acceptance Report: E4o-Speech — DTmin Frequency Conditioning Audit (Band Sensitivity)

## 1) Executive Summary

- Run: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_hi_normal_20260127_135638/
- Mode: scale_check_subset
- Pairing mode: paired
- freq_cond_mode: normal
- Band: [1800.0, 3000.0] Hz
- Outcome: PASS
- Band delta @ lambda_c=3e-4 (DT_cm(normal) - DT_cm(shuffle)): 0.0000
- Band-only classification (shuffle only): FREQ_COND_IGNORED

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_hi_normal_20260127_135638/code_state.json
  - git_head: 7a5f017cb45f905016a7ef6d919695e924564f93
  - dirty: True
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 9e276bc615debccb299408a9468243b6a2b552c684184125b844eadbb03df0f8
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired
- Subset manifest: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_hi_normal_20260127_135638/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

## 3) Exact Commands (Required)

See run.log for stdout/stderr; command reconstructed from docs/rtgomp_dtmin_freq_cond_E4o_speech_plan.md with this out_dir.

## 4) Results (Required)

### 4.0 Hard guardrails (Must pass)

- wav-only manifest: PASS
- fs mismatch encountered: NO (would have failed fast)
- any NaN/Inf in summary outputs: NO
- required artifacts exist:
  - summary/compute_matched_summary.json: YES
  - summary/rtg_controllability_summary.json: YES
  - summary/freq_cond_audit_summary.json: YES

### 4.1 Key metrics

- spearman(lambda_c, k_selected_mean) = -0.9999999999999999
- k_selected_mean list = [15.993256743256744, 15.877039627039627, 14.96972471972472, 14.02891552891553, 13.371600621600622]

Compute-matched capture summary (DT/OMP/Random):
- lambda_c=1e-05: k_mean=15.993, DT=0.9992, OMP=0.9999, Random=0.9991, DT-Random=0.0001
- lambda_c=3e-05: k_mean=15.877, DT=0.9991, OMP=0.9999, Random=0.9990, DT-Random=0.0001
- lambda_c=1e-04: k_mean=14.970, DT=0.9983, OMP=0.9997, Random=0.9978, DT-Random=0.0005
- lambda_c=2e-04: k_mean=14.029, DT=0.9964, OMP=0.9994, Random=0.9949, DT-Random=0.0015
- lambda_c=3e-04: k_mean=13.372, DT=0.9937, OMP=0.9989, Random=0.9907, DT-Random=0.0030

## 5) Interpretation (Band sensitivity)

At this band, shuffle changes DT_cm by 0.0000 at lambda_c=3e-4. If this delta is small, THEN frequency conditioning is not materially used even in this band; if it is large, THEN high-frequency regimes may depend on correct freq_idx.
