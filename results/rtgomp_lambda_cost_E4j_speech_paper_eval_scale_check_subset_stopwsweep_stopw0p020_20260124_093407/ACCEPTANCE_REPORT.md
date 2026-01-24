# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stopwsweep_stopw0p020_20260124_093407/
- Mode: scale_check_subset
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
- Device(s): cpu (MPS was not used for this run; evaluator executed on CPU for stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stopwsweep_stopw0p020_20260124_093407/code_state.json
  - git_head: 6af8a630c82d93dbbf66fd61a7a52706dc8c8787
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 0442f1b160c46253e6b5e96a6109be235fff9dd3ab3a3a8baf7dc61ee1d8b058
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Dataset pairing rule:
  - wav pair is determined by replacing MIC -> LDV in the filename
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_001.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_001.wav
  - last_pair  = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_xnonoise_320.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stopwsweep_stopw0p020_20260124_093407/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
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
  - random_trials = 3
  - random_sampling = without_replacement

### 2.6 Step/compute definitions (Must be explicit)

- k_selected:
  - number of selected lag atoms (number of LS solves; compute-aligned)
- steps_decision:
  - number of decision steps including STOP (policy-step-aligned)

## 3) Exact Commands (Required)

The run used a lockdir + tee pattern. Exact evaluator invocation (from run.log):

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stopwsweep_stopw0p020_20260124_093407 \
  --mode scale_check_subset \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --random_trials 3 \
  --random_sampling without_replacement \
  --seed 0 \
  --device cpu \
  --write_per_sample 0
```

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary integrity (summary/compute_matched_summary.json: integrity):
- num_samples_total = 80964
- num_samples_used = 80964
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
  - num_capture_out_of_range_dt = 0
  - num_capture_out_of_range_omp = 0
  - num_capture_out_of_range_random = 0
- capture_min/max per method:
  - DT: min 0.0003377795, max 1.0
  - OMP: min 0.0234571695, max 0.9999999404
  - Random: min 0.0, max 0.9999997616
- Worst violation summary:
  - none
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0
- Random duplicate stats:
  - random_duplicate_rate = 0.0 (expected near 0 for without_replacement)

Decision:
- PASS because all hard guardrails are satisfied (no missing files, no md5 mismatches, no NaN/Inf, no out-of-range capture, no OMP monotonicity violations, and no DT duplicate actions under forced-K).

### 4.1 RTG controllability (Free rollout; Required)

Primary metric (compute-aligned):
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_range = 3.953633713749319

Distributional metrics:
- P(k_selected < max_k) at min lambda (1e-5) = 0.050392767155772934
- P(k_selected < max_k) at max lambda (3e-4) = 0.7454794723580851
- k_selected quantiles by lambda:
  - p50 = [15.999999998645448, 15.999999998185128, 14.999955026017311, 13.000751085655587, 12.000400711564978]
  - p90 = [16.0, 15.999999999992784, 15.999999808184167, 15.999999462478566, 15.999994457257323]
  - p99 = [16.0, 16.0, 16.0, 16.0, 15.999999999999682]

Also (policy-step-aligned):
- spearman(lambda_c, steps_decision_mean) = -1.0
- steps_decision_range = 3.2585470085470085

Decision:
- PASS because RTG (lambda_c) monotonically controls compute: higher lambda produces lower k_selected and fewer decision steps, with strong rank correlation (Spearman = -1.0).

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP; Required)

Across lambdas (from summary/compute_matched_summary.json: rows):

- lambda_c=1e-5: DT/OMP=0.999390, DT-Random=0.000922, k_selected_mean=15.901, steps_decision_mean=15.952
- lambda_c=3e-5: DT/OMP=0.999319, DT-Random=0.001937, k_selected_mean=15.437, steps_decision_mean=15.665
- lambda_c=1e-4: DT/OMP=0.998714, DT-Random=0.008566, k_selected_mean=13.962, steps_decision_mean=14.531
- lambda_c=2e-4: DT/OMP=0.997425, DT-Random=0.019347, k_selected_mean=12.734, steps_decision_mean=13.432
- lambda_c=3e-4: DT/OMP=0.995744, DT-Random=0.029256, k_selected_mean=11.948, steps_decision_mean=12.693

Decision:
- DT - Random > 0 at low penalty (lambda_c=1e-5) AND at high penalty (lambda_c=3e-4), therefore the core paper claim (DT beats Random under compute-matched evaluation) is satisfied on this subset.

### 4.3 Forced-K DT vs OMP vs Random (NO STOP; Required)

From summary/forced_k_summary.json: rows:
- K=1:  DT=0.678113, OMP=0.710736, Random=0.174302, DT/OMP=0.954100, DT-Random=0.503811
- K=2:  DT=0.761990, OMP=0.854290, Random=0.332187, DT/OMP=0.891957, DT-Random=0.429804
- K=4:  DT=0.853733, OMP=0.955119, Random=0.598261, DT/OMP=0.893850, DT-Random=0.255473
- K=8:  DT=0.970233, OMP=0.994233, Random=0.920198, DT/OMP=0.975861, DT-Random=0.050035
- K=16: DT=0.999251, OMP=0.999852, Random=0.998476, DT/OMP=0.999399, DT-Random=0.000775

## 5) Interpretation (Required; causal language)

- RTG controls compute because lambda_c directly increases the penalty for additional selections, therefore a well-trained STOP head should become more likely at higher lambda_c. The measured Spearman correlation of -1.0 confirms monotonic control over k_selected_mean on this speech subset.
- DT beats Random (compute-matched) because DT learns structured lag selection that concentrates capture early, whereas Random wastes selections on low-utility lags; the gap grows with lambda_c because higher penalty forces earlier stopping where selection quality matters more.
- DT remains close to OMP (DT/OMP ~ 0.996-0.999) because the policy is trained to imitate OMP-like decisions; the remaining gap is expected due to limited model capacity and distribution shift between training rollouts and evaluation clips.
- The remaining saturation is visible because p90/p99 are ~max_k even at high lambda_c; this implies some frequency bins still require near-max selections to achieve good capture, therefore mean compute can move while the tail remains near max_k.

## 6) Failures (Required even if PASS)

- No hard failures in this run.
- Remaining risks / known limitations:
  - Tail saturation: even at lambda_c=3e-4, p90 ~ max_k, therefore reducing mean compute further will likely require either a wider lambda grid, stronger STOP/cost calibration, or a change in training data distribution to emphasize early stopping.

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
OUT_DIR=results/rtgomp_lambda_cost_E4j_speech_paper_eval_scale_check_subset_stopwsweep_stopw0p020_20260124_093407
mkdir -p "$OUT_DIR/summary"
mkdir "$OUT_DIR/.lock"
conda run -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode scale_check_subset \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --random_trials 3 \
  --random_sampling without_replacement \
  --seed 0 \
  --device cpu \
  --write_per_sample 0 \
  |& tee -a "$OUT_DIR/run.log"
```

3) Verification:
- Expected outputs:
  - summary/compute_matched_summary.json
  - summary/forced_k_summary.json
  - summary/rtg_controllability_summary.json
  - subset_manifest.json (must contain only .wav paths)
  - integrity_diagnostics.jsonl (expected empty for this paper baseline)
  - run.log
  - code_state.json
- Key numeric checks (no strict tolerance; must match trend):
  - spearman(lambda_c, k_selected_mean) <= -0.6 (observed: -1.0)
  - k_selected_mean at lambda_c=3e-4 ~ 12 (observed: 11.948)
  - DT - Random > 0 at lambda_c=3e-4 (observed: 0.029256)

## 8) Next Steps (Required)

- Run full_dataset evaluation for this checkpoint to confirm the ~12 mean compute target holds on all 416 speech pairs.
- If broader compute range is desired (not just moving the mean): widen lambda grid (include >3e-4) and/or recalibrate STOP/cost (e.g., STOP label weighting and RTG scaling) so that more bins stop earlier (reduce p90 tail saturation) while keeping DT > Random.

