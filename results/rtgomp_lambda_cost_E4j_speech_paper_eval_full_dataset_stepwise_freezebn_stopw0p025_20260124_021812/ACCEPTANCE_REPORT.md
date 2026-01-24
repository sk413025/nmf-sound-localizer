# Acceptance Report: E4j-Speech -- RTG/STOP Compute Controllability (DT vs OMP vs Random)

## 1) Executive Summary

- Run (full_dataset): results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stepwise_freezebn_stopw0p025_20260124_021812/
- Outcome: PASS
- Primary outputs:
  - Compute-matched DT vs OMP vs Random (WITH STOP; matched by DT k_selected)
  - Forced-K DT vs OMP vs Random (NO STOP)
  - RTG controllability (free rollout k_selected vs lambda_c)
- Dataset domain statement:
  - This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: 3.11.13
- Device(s): cpu (torch.backends.mps.is_available() == False in this env)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stepwise_freezebn_stopw0p025_20260124_021812/code_state.json
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
- Dataset length check:
  - len(dataset) = 416
  - first_pair = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_001.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_001.wav
  - last_pair  = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_xnonoise_320.wav, /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stepwise_freezebn_stopw0p025_20260124_021812/subset_manifest.json
  - num_pairs = 416
  - fingerprint_md5 = 13356fdb74d2acb7e85361a4dfe5c3d2
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (explicit)

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

### 2.5 Baselines

- Random baseline:
  - random_trials = 1
  - random_sampling = without_replacement (paper baseline)

### 2.6 Step/compute definitions

- k_selected:
  - number of selected lag atoms (number of LS solves; compute-aligned)
- steps_decision:
  - number of decision steps including STOP (policy-step-aligned)

### 2.7 Model checkpoint provenance (Warm-start + STOP calibration)

This evaluation uses a checkpoint trained in:
- Training run: results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p025_20260124_020736/
  - ckpt: results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p025_20260124_020736/model/dt_freq_aware_best.pth
  - init_ckpt (warm-start): results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth
  - training code_state.json: results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p025_20260124_020736/code_state.json
  - key training setting: --train_stepwise_eval
    - This is required BECAUSE the evaluator calls the model with K=1 per decision (GRU hidden reset each step), and sequence-style training caused STOP logits to depend on hidden state that is not preserved at eval.

Training data (penalty-OMP trajectories):
- Data run: results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/
  - data_path: results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/data/lag_trajectories.pt
  - subset_manifest fingerprint_md5: 72499de7e7c0b0c8d650954ae52fcb48
  - data code_state.json: results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/code_state.json

## 3) Exact Commands (Executed)

See run logs for exact command lines:

- Full-dataset eval log:
  - results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stepwise_freezebn_stopw0p025_20260124_021812/run.log

- Training log:
  - results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p025_20260124_020736/run.log

- Data generation log:
  - results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/run.log

## 4) Results (Full Dataset)

All numbers below are copied from:
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary integrity:
- num_samples_total = 683004
- num_samples_used = 683004
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0 (paper baseline requirement)
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0
- Random duplicate stats:
  - random_duplicate_rate = 0.0

Decision:
- PASS because all hard guardrails are satisfied (no missing files, no NaN/inf, no out-of-range capture for paper Random, no OMP monotonicity violations, and no DT duplicate actions in forced-K).

### 4.1 RTG controllability (Free rollout)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_range = 4.5154
- P(k_selected < max_k) at min lambda (1e-5) = 0.06685
- P(k_selected < max_k) at max lambda (3e-4) = 0.80727

Key means:
- k_selected_mean @ 1e-5 = 15.8582
- k_selected_mean @ 3e-4 = 11.3428

Decision:
- PASS because k_selected_mean decreases monotonically with lambda_c (spearman=-1.0), and the mean compute shift is material (range ~4.5 steps) BECAUSE STOP is now learned in the same “K=1 per decision” regime as evaluation.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP)

Summary table (lambda_c, DT/OMP, DT-Random, k_selected_mean):
- 1e-5: 0.999365, 0.000846, 15.858
- 3e-5: 0.999238, 0.002042, 15.279
- 1e-4: 0.998280, 0.009448, 13.553
- 2e-4: 0.996299, 0.020958, 12.195
- 3e-4: 0.993882, 0.032248, 11.343

Decision:
- PASS because DT - Random > 0 at low penalty (1e-5) and remains > 0 at high penalty (3e-4).

### 4.3 Forced-K DT vs OMP vs Random (NO STOP)

From summary/forced_k_summary.json (k, DT/OMP, DT-Random):
- K=1: 0.948701, 0.482585
- K=2: 0.888932, 0.417405
- K=4: 0.892405, 0.248495
- K=8: 0.975581, 0.046505
- K=16: 0.999399, 0.000615

Interpretation:
- Forced-K remains strong (DT > Random across K) BECAUSE the lag ranking is already informative; compute-matched improvements at high lambda come from STOP/cost calibration rather than changing the core lag-selection quality.

## 5) Interpretation (Causal)

- The original “RTG controllability” failure mode (mean steps barely moving) happens BECAUSE the evaluator calls the GRU policy with K=1 each decision and therefore resets the hidden state every step; sequence-style training can make STOP logits depend on hidden state that is not present at evaluation time.
- Switching to stepwise-eval training (--train_stepwise_eval) fixes this BECAUSE it forces the model to predict STOP from per-step inputs (abs_corrs, rtg, freq) under the same hidden-reset regime.
- STOP calibration via stop_weight=0.025 matters BECAUSE it controls how strongly the supervised objective prioritizes STOP vs selecting another lag; therefore it lets us target a desired mean compute (k_selected_mean ~ 11-12 at max lambda) while still keeping DT > Random.

## 6) Failures / Diagnostics (Required)

- Guardrail diagnostic run (expected to show duplicates and out-of-range capture):
  - results/rtgomp_lambda_cost_E4j_speech_paper_eval_guardrail_random_with_replacement_20260124_021728/
  - random_duplicate_rate = 0.07198
  - num_capture_out_of_range_total = 651
  - This is expected BECAUSE sampling with replacement introduces duplicate atoms, which can make the LS problem ill-conditioned and produce capture outside [0,1].

## 7) Reproduction Instructions (Step-by-step)

1) Environment:

- source ~/.zshrc
- conda activate trl-training
- export PYTHONPATH=.
- export MPLCONFIGDIR=/tmp/mpl

2) (Optional) Generate penalty-OMP trajectories (small speech subset used for STOP calibration):

- See results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/run.log

3) Train STOP-calibrated DT checkpoint (warm-start):

- See results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p025_20260124_020736/run.log

4) Evaluate (full dataset):

- See results/rtgomp_lambda_cost_E4j_speech_paper_eval_full_dataset_stepwise_freezebn_stopw0p025_20260124_021812/run.log

5) Verification:

Expected key outputs:
- summary/rtg_controllability_summary.json exists and has:
  - spearman_lambda_k_selected == -1.0
  - k_selected_mean at max lambda (3e-4) ~ 11.34
- summary/compute_matched_summary.json exists and has:
  - dt_minus_random_mean at min lambda (1e-5) > 0
  - dt_minus_random_mean at max lambda (3e-4) > 0
  - num_capture_out_of_range_total == 0

## 8) Next Steps

- If you want k_selected_mean even closer to 12.0 at max lambda, sweep stop_weight slightly (e.g., 0.02, 0.03) BECAUSE it directly shifts STOP aggressiveness while preserving warm-started lag ranking.
- If you want the compute curve to be less “linear”, expand lambda grid density (more points) BECAUSE spearman is already saturated (-1.0) and we can now characterize the full response curve rather than just endpoints.
