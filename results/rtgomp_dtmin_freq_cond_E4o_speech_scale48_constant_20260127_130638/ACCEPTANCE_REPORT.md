# Acceptance Report: E4o-Speech — DTmin Frequency Conditioning Audit

## 1) Executive Summary

- Run: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/
- Mode: scale_check_subset
- Pairing mode: paired
- freq_cond_mode: constant
- freq_cond_seed: 0
- freq_cond_constant_idx: 200
- Band: [300.0, 3000.0] Hz
- Outcome: PASS
- Classification (suite-level): FREQ_COND_IGNORED

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/code_state.json
  - git_head: 5ff97bfc49e02d246d09e562fcf5986dc21aab2f
  - dirty: True
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 9e276bc615debccb299408a9468243b6a2b552c684184125b844eadbb03df0f8
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired
- Subset manifest: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/subset_manifest.json
  - num_pairs = 48 (expected 416 for full dataset)
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

## 3) Exact Commands (Required)

See run.log for stdout/stderr; command reconstructed from docs/rtgomp_dtmin_freq_cond_E4o_speech_plan.md with this out_dir.

Reproduction command (copy/paste; fill OUT_DIR):
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir <OUT_DIR> \
  --mode scale_check_subset --num_pairs 48 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 0 --device cpu \
  --pairing_mode paired --require_wav_only 1 \
  --write_delay_diagnostics 1 \
  --freq_cond_mode constant --freq_cond_seed 0 --freq_cond_constant_idx 200
```

## 4) Results (Required)

### 4.0 Hard guardrails (Must pass)

- wav-only manifest: PASS
- fs mismatch encountered: NO (would have failed fast)
- any NaN/Inf in summary outputs: NO
- required artifacts exist:
  - summary/compute_matched_summary.json: YES
  - summary/rtg_controllability_summary.json: YES
  - summary/freq_cond_audit_summary.json: YES

### 4.1 Key metrics (copy/paste values)

- spearman(lambda_c, k_selected_mean) = -0.9999999999999999
- k_selected_mean list = [15.97544587717998, 15.831295390543945, 14.594239415048664, 12.990205523442517, 12.045155871745466]

Compute-matched capture summary (DT/OMP/Random):
- lambda_c=1e-05: k_mean=15.975, DT=0.9993, OMP=0.9998, Random=0.9984, DT-Random=0.0008
- lambda_c=3e-05: k_mean=15.831, DT=0.9992, OMP=0.9998, Random=0.9983, DT-Random=0.0009
- lambda_c=1e-04: k_mean=14.594, DT=0.9984, OMP=0.9997, Random=0.9962, DT-Random=0.0022
- lambda_c=2e-04: k_mean=12.990, DT=0.9962, OMP=0.9992, Random=0.9900, DT-Random=0.0062
- lambda_c=3e-04: k_mean=12.045, DT=0.9934, OMP=0.9987, Random=0.9813, DT-Random=0.0121

### 4.2 Suite-level classification (required)

- Reference (normal) run: results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_normal_20260127_125436/
- Target lambda_c=3e-4: DT_cm(normal)=0.9945, DT_cm(this)=0.9934, Δ=0.0011
- Classification (suite-level): FREQ_COND_IGNORED

Causal interpretation:
- DT capture changes are <= 0.01 under shuffle/constant/zero_embed BECAUSE the policy decisions are driven primarily by state correlations + RTG, not by the frequency embedding; THEREFORE frequency conditioning is effectively ignored in this regime.

## 5) Physical / Mathematical Analysis (Required)

Frequency-dependent transfer functions can produce frequency-dependent optimal lag selections BECAUSE phase response varies with frequency. A frequency embedding provides a conditional context that could, in principle, shift the policy toward different lag/STOP behavior across bins. However, shuffling/collapsing/zeroing `freq_idx` produced negligible DT capture changes, which implies the model is not relying on that context in this evaluation regime.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech) established DT vs OMP vs Random baselines on WAV-only speech under compute control.
- Commit db556db (E4j-Speech) calibrated STOP/compute behavior and produced the checkpoint used here.
- Commit 18a82aa (E4m-Speech) showed dispersion diagnostics issues despite stable GCC-PHAT, motivating careful interpretation of ‘physics-aware’ claims.
Pattern: DT can track OMP-like capture under compute control even when dispersion remains, suggesting capture can be achieved without explicitly modeling frequency-dependent phase correction. E4o adds that the current checkpoint’s frequency embedding is not a key driver of performance in this capture objective BECAUSE ablations barely change capture.

## 7) Extracted Principles for Next Steps (Required)

- If frequency conditioning is ignored, THEN consider simplifying the model by removing `freq_embed` to reduce complexity.
- If we actually want frequency-conditioned behavior, THEN re-train with `freq_embed` unfrozen and evaluate on a harder regime (e.g., high band only or held-out bands).
- Always compare at fixed compute (forced-k) in addition to fixed lambda, BECAUSE ablations can shift the lambda→compute mapping.

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```
2) Execution: use the command in section 3 with OUT_DIR set to the desired results folder.
3) Verification:
- Confirm files exist:
  - results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/summary/compute_matched_summary.json
  - results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/summary/rtg_controllability_summary.json
  - results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/summary/freq_cond_audit_summary.json
- Confirm no wav/fs guardrail failures in results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_constant_20260127_130638/run.log
