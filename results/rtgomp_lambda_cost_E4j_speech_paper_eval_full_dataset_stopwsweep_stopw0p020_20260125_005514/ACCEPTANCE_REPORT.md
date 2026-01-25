# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514/
- Mode: full_dataset
- Outcome: PASS
- Primary paper outputs:
  - Compute-matched DT vs OMP vs Random (WITH STOP; matched by k_selected)
  - Forced-K DT vs OMP vs Random (NO STOP)
- Dataset domain statement (required):
  - This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: 3.11.13
- Device(s): cpu
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514/code_state.json
  - git_head: 03348020e53873dc89ea9f5db183be5801d2caaf
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 0442f1b160c46253e6b5e96a6109be235fff9dd3ab3a3a8baf7dc61ee1d8b058
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Dataset pairing rule:
  - wav pair is determined by replacing MIC -> LDV in the filename
- Dataset length check:
  - len(dataset) = 416
  - first_pair = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_001.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_001.wav
  - last_pair  = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_xnonoise_320.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514/subset_manifest.json
  - num_pairs = 416
  - fingerprint_md5 = 13356fdb74d2acb7e85361a4dfe5c3d2
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (Must be explicit)

- hop_length = 160
- fs = 16000
- n_fft = 2048
- freq band = [freq_min, freq_max] = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- Tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- use_stop_action = true
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

### 2.5 Baselines (Must be explicit)

- Random baseline:
  - random_trials = 1
  - random_sampling = without_replacement

### 2.6 Step/compute definitions (Must be explicit)

- k_selected:
  - number of selected lag atoms (number of LS solves; compute-aligned)
- steps_decision:
  - number of decision steps including STOP (policy-step-aligned)

## 3) Exact Commands (Required)

This run used a lockdir + tee pattern. The full command line is recorded in:
- results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514/run.log

Evaluator invocation:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514 \
  --mode full_dataset \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --random_trials 1 \
  --random_sampling without_replacement \
  --seed 0 \
  --device cpu \
  --write_per_sample 0 \
  |& tee -a results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514/run.log
```

## 4) Results (Required)

All numeric fields below are copied from:
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_samples_total = 683004
- num_samples_used = 683004
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
  - num_capture_out_of_range_dt = 0
  - num_capture_out_of_range_omp = 0
  - num_capture_out_of_range_random = 0
- capture_min/max per method:
  - DT: min 5.424e-06, max 1.0
  - OMP: min 0.023457, max 0.99999994
  - Random: min 0.0, max 0.99999994
- Worst violation summary:
  - none
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0
- Random duplicate stats:
  - random_duplicate_rate = 0.0

Decision:
- PASS because all hard guardrails are satisfied (no missing files, no md5 mismatches, no NaN/Inf, no out-of-range capture, no OMP monotonicity violations, and no DT duplicate actions under forced-K).

### 4.1 RTG controllability (Free rollout; Required)

Primary metric (compute-aligned):
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_range = 3.8553932334217667

Distributional metrics (required):
- P(k_selected < max_k) at min lambda (1e-5) = 0.048124169111747515
- P(k_selected < max_k) at max lambda (3e-4) = 0.7391860076954161
- k_selected quantiles by lambda:
  - p50 = [16.0, 16.0, 15.0002, 13.0004, 12.0004]
  - p90 = [16.0, 16.0, 16.0, 16.0, 16.0]
  - p99 = [16.0, 16.0, 16.0, 16.0, 16.0]

Also (policy-step-aligned):
- spearman(lambda_c, steps_decision_mean) = -1.0
- steps_decision_range = 3.164331394838097

Decision:
- PASS because lambda_c monotonically controls compute (k_selected and steps_decision) on the full speech dataset.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP; Required)

Across lambdas (summary/compute_matched_summary.json: rows):

- lambda_c=1e-5: DT/OMP=0.999381, DT-Random=0.000756, k_selected_mean=15.904, steps_decision_mean=15.952
- lambda_c=3e-5: DT/OMP=0.999292, DT-Random=0.001581, k_selected_mean=15.462, steps_decision_mean=15.682
- lambda_c=1e-4: DT/OMP=0.998598, DT-Random=0.007009, k_selected_mean=14.029, steps_decision_mean=14.589
- lambda_c=2e-4: DT/OMP=0.997149, DT-Random=0.016146, k_selected_mean=12.817, steps_decision_mean=13.509
- lambda_c=3e-4: DT/OMP=0.995399, DT-Random=0.024739, k_selected_mean=12.049, steps_decision_mean=12.788

Decision:
- DT - Random > 0 at low penalty (1e-5) AND at max penalty in this grid (3e-4), therefore DT > Random under compute-matched evaluation holds on the full dataset.

### 4.3 Forced-K DT vs OMP vs Random (NO STOP; Required)

From summary/forced_k_summary.json: rows:
- K=1:  DT=0.658804, OMP=0.694439, Random=0.176231, DT/OMP=0.948686, DT-Random=0.482574
- K=2:  DT=0.752309, OMP=0.846325, Random=0.334920, DT/OMP=0.888912, DT-Random=0.417389
- K=4:  DT=0.850330, OMP=0.952876, Random=0.601857, DT/OMP=0.892383, DT-Random=0.248473
- K=8:  DT=0.969722, OMP=0.993998, Random=0.923221, DT/OMP=0.975578, DT-Random=0.046502
- K=16: DT=0.999251, OMP=0.999853, Random=0.998636, DT/OMP=0.999399, DT-Random=0.000615

## 5) Interpretation (Required; causal language)

- Compute is controllable because increasing lambda_c increases the penalty for additional selections; THEREFORE a calibrated STOP head should trigger earlier, reducing k_selected_mean. The measured Spearman correlation of -1.0 confirms monotonic control on the full dataset.
- DT beats Random (compute-matched) because DT learns structured lag selection that concentrates capture early, whereas Random wastes selections on low-utility lags; THEREFORE DT-Random stays positive even when compute is reduced at higher lambda.
- DT stays close to OMP because it is trained to imitate a pursuit-like teacher; THEREFORE DT/OMP remains ~0.995 at the highest penalty in this grid.
- Tail saturation remains (p90/p99 ~ max_k) because a subset of frequency-time samples still require near-max selections to achieve high capture; THEREFORE shifting the mean compute further will likely require additional calibration and/or different cost scaling rather than expecting a uniform shift in all quantiles.
## 6) Failures (Required even if PASS)

- No hard failures in this run.
- Remaining risks / known limitations:
  - Quantile tail saturation at high lambda (p90/p99 near max_k) indicates that some samples still rarely STOP early; therefore further reducing mean compute without hurting DT>Random may require more targeted cost/STOP calibration or a wider RTG/lambda training distribution.

## 7) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
```bash
OUT_DIR=results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stopwsweep_stopw0p020_20260125_005514
mkdir -p \"$OUT_DIR/summary\"
mkdir \"$OUT_DIR/.lock\"
conda run -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir \"$OUT_DIR\" \
  --mode full_dataset \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --random_trials 1 \
  --random_sampling without_replacement \
  --seed 0 \
  --device cpu \
  --write_per_sample 0 \
  |& tee -a \"$OUT_DIR/run.log\"
```

3) Verification:
- Expected outputs:
  - summary/compute_matched_summary.json
  - summary/forced_k_summary.json
  - summary/rtg_controllability_summary.json
  - subset_manifest.json (must contain only .wav paths)
  - integrity_diagnostics.jsonl (expected empty for paper baseline)
  - run.log
  - code_state.json
- Key numeric checks (no strict tolerance; must match trend):
  - spearman(lambda_c, k_selected_mean) <= -0.6 (observed: -1.0)
  - k_selected_mean at lambda_c=3e-4 ~ 12 (observed: 12.049)
  - DT - Random > 0 at lambda_c=1e-5 and 3e-4 (observed: +0.000756 and +0.024739)

## 8) Next Steps (Required)

- If we want mean compute even closer to an exact target (e.g., 12.0 +/- 0.1 across seeds), increase random_trials and/or run a narrower stop_weight sweep around 0.020 with full_dataset evaluation.
- If we want a larger compute range (not just mean shift): widen the lambda grid and fix RTG normalization semantics before retraining, so the meaning of RTG does not drift when the grid changes.
