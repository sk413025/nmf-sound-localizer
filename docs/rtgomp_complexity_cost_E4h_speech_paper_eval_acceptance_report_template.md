# Acceptance Report Template: E4h-Speech -- Paper-Grade DT vs OMP vs Random

Use this template for E4h-Speech. Fill all placeholders. Keep everything in English.

Save the filled report to:
- results/rtgomp_lambda_cost_E4h_speech_paper_eval_<mode>_<timestamp>/ACCEPTANCE_REPORT.md

This report is required for both:
- scale_check_subset (48 pairs)
- full_dataset (all pairs in dataset order)

Also include a short report for the guardrail diagnostic (Random with replacement), even if it is expected to FAIL.

---

# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: results/rtgomp_lambda_cost_E4h_speech_paper_eval_<mode>_<timestamp>/
- Mode: smoke / scale_check_subset / full_dataset
- Outcome: PASS / FAIL / PASS_WITH_WARNINGS
- Primary paper outputs:
  - Compute-matched DT vs OMP vs Random (WITH STOP; matched by k_selected)
  - Forced-K DT vs OMP vs Random (NO STOP)
- Dataset domain statement (required):
  - This run uses the speech WAV dataset only (no .npy files in the manifest): YES/NO

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (paste `python --version`)
- Device(s): cpu / mps (and rationale)
- MPLCONFIGDIR: (value used; recommended /tmp/mpl)

### 2.2 Code provenance (Required)

- code_state.json: results/.../code_state.json
  - git_head: <hash>
  - dirty: true/false
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: <sha256>
    - scripts/h_exploration/dataset_lag.py: <sha256>

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Dataset pairing rule:
  - wav pair is determined by replacing MIC -> LDV in the filename
- Dataset length check (preflight output):
  - len(dataset) = <value>
  - first_pair = <path1>, <path2>
  - last_pair  = <path1>, <path2>
- Subset manifest: results/.../subset_manifest.json
  - num_pairs = <N>
  - fingerprint_md5 = <hash>
  - Domain validation:
    - all paths end with .wav: YES/NO (must be YES for E4h-Speech)
    - any .npy paths: YES/NO (must be NO for E4h-Speech)

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
- lambda_c_values = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

### 2.5 Baselines (Must be explicit)

- Random baseline:
  - random_trials = <N>
  - random_sampling = without_replacement (paper) OR with_replacement (diagnostic)

### 2.6 Step/compute definitions (Must be explicit)

- k_selected:
  - number of selected lag atoms (number of LS solves; compute-aligned)
- steps_decision:
  - number of decision steps including STOP (policy-step-aligned)

## 3) Exact Commands (Required)

Paste the exact command(s) executed, including tee, lockdir pattern, and env exports (PYTHONPATH, MPLCONFIGDIR).

## 4) Results (Required)

All numeric fields below must be copied from:
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- run.log (for timing and warnings)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary integrity:
- num_samples_total = <int>
- num_samples_used = <int>
- num_missing_files = <int>
- num_md5_mismatches = <int>
- num_nan_or_inf = <int>
- num_capture_out_of_range_total = <int> (must be 0 for paper baseline)
- num_capture_out_of_range_dt = <int>
- num_capture_out_of_range_omp = <int>
- num_capture_out_of_range_random = <int>
- capture_min/max per method:
  - DT: min <...>, max <...>
  - OMP: min <...>, max <...>
  - Random: min <...>, max <...>
- Worst violation summary:
  - <paste worst_violation dict or "none">
- num_omp_monotonicity_violations = <int> (must be 0 for PASS)
- num_dt_duplicate_actions_forced_k = <int> (must be 0 for PASS)
- Random duplicate stats:
  - random_duplicate_rate = <float> (expected near 0 for without_replacement; >0 for with_replacement)

Decision:
- PASS/FAIL with causal language:
  - PASS because ...
  - FAIL because ...

### 4.1 RTG controllability (Free rollout; Required)

Primary metric (compute-aligned):
- spearman(lambda_c, k_selected_mean) = <value> (recommended <= -0.6)
- k_selected_range = <value> (recommended >= 0.10)

Distributional metrics (required):
- P(k_selected < max_k) at min lambda = <value>
- P(k_selected < max_k) at max lambda = <value>
- k_selected quantiles by lambda:
  - p50: <list>
  - p90: <list>
  - p99: <list>

Also report (policy-step-aligned, for completeness):
- spearman(lambda_c, steps_decision_mean) = <value>
- steps_decision_range = <value>

Decision:
- PASS/FAIL with causal language.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP; Required)

Report at minimum for lambda_c = min (1e-4):
- DT_capture_mean = <value>
- OMP_capture_mean = <value>
- Random_capture_mean = <value>
- DT/OMP = <value>
- DT - Random = <value>
- k_selected_mean = <value> (compute-aligned)
- steps_decision_mean = <value> (policy-step-aligned)

Optionally (recommended) add a small table across all lambdas:
- lambda_c, DT/OMP, DT-Random, k_selected_mean

Decision:
- For paper claim, DT - Random must be > 0 at low penalty (state if satisfied).

### 4.3 Forced-K DT vs OMP vs Random (NO STOP; Required)

Report for K in {1, 2, 4, 8, 16} (or those <= max_k):
- K=1: DT <...>, OMP <...>, Random <...>, DT/OMP <...>, DT-Random <...>
- K=2: ...
- K=4: ...
- K=8: ...
- K=16: ...

Interpretation hint:
- If forced-K looks good but compute-matched looks weak at high K, STOP/early stopping is likely the cause.

## 5) Interpretation (Required; causal language)

Use BECAUSE / DUE TO / THEREFORE to connect:
- why RTG affects or does not affect compute on speech
- why DT beats or loses to Random (compute-matched)
- why DT is close to or far from OMP (compute-matched and forced-K)
- whether saturation at max_k explains the "mean steps barely moves" observation

## 6) Failures (Required even if PASS)

If any check failed:
- Root cause hypothesis (BECAUSE ...)
- Evidence: cite exact files/lines (run.log) or fields (summary JSON paths)
- Minimal fix (smallest change likely to pass)
- Fundamental fix (if minimal fails)

If PASS:
- Remaining risks / known limitations (e.g., saturation at max_k)

## 7) Reproduction Instructions (Required)

Provide exact step-by-step commands to reproduce:

1) Environment:
   - source ~/.zshrc
   - conda activate trl-training
   - export PYTHONPATH=.
   - export MPLCONFIGDIR=/tmp/mpl

2) Execution:
   - paste the exact evaluator command

3) Verification:
   - list the expected output files and key numbers that should match (within stated tolerance)

## 8) Next Steps (Required)

Provide concrete follow-ups based on this run:
- If controllability is saturated: propose a targeted change (e.g., STOP calibration, RTG scaling, lambda grid widening,
  or training on a broader speech subset).
- If DT is far from OMP under forced-K: propose a lag-selection improvement (inputs/architecture/training data).

