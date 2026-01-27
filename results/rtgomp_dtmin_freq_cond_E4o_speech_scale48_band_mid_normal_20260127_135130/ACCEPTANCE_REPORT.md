# Acceptance Report: E4o-Speech — DTmin Frequency Conditioning Audit (Band Sensitivity)

## 1) Executive Summary

- Run: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_mid_normal_20260127_135130/
- Mode: scale_check_subset
- Pairing mode: paired
- freq_cond_mode: normal
- Band: [900.0, 1800.0] Hz
- Outcome: PASS
- Band delta @ lambda_c=3e-4 (DT_cm(normal) - DT_cm(shuffle)): 0.0015
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

- code_state.json: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_mid_normal_20260127_135130/code_state.json
  - git_head: 7a5f017cb45f905016a7ef6d919695e924564f93
  - dirty: True
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 9e276bc615debccb299408a9468243b6a2b552c684184125b844eadbb03df0f8
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired
- Subset manifest: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_band_mid_normal_20260127_135130/subset_manifest.json
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
- k_selected_mean list = [15.969639539204756, 15.581196581196581, 14.045484949832776, 12.686845039018952, 11.803976217019695]

Compute-matched capture summary (DT/OMP/Random):
- lambda_c=1e-05: k_mean=15.970, DT=0.9993, OMP=0.9998, Random=0.9984, DT-Random=0.0009
- lambda_c=3e-05: k_mean=15.581, DT=0.9992, OMP=0.9998, Random=0.9979, DT-Random=0.0013
- lambda_c=1e-04: k_mean=14.045, DT=0.9984, OMP=0.9996, Random=0.9932, DT-Random=0.0051
- lambda_c=2e-04: k_mean=12.687, DT=0.9966, OMP=0.9992, Random=0.9830, DT-Random=0.0136
- lambda_c=3e-04: k_mean=11.804, DT=0.9945, OMP=0.9987, Random=0.9710, DT-Random=0.0234

## 5) Interpretation (Band sensitivity)

At this band, shuffle changes DT_cm by 0.0015 at lambda_c=3e-4. If this delta is small, THEN frequency conditioning is not materially used even in this band; if it is large, THEN high-frequency regimes may depend on correct freq_idx.
