# Acceptance Report Template: E4k-Speech -- Delay & Phase-Consistency Diagnostics

Use this template for E4k-Speech. Fill all placeholders. Keep everything in English.

You will typically write **two reports**:
- Paired (paper-like, correct pairing)
- Mispair (guardrail diagnostic; still real WAV)

Save the filled report to:
- results/<run>/ACCEPTANCE_REPORT.md

---

# Acceptance Report: E4k-Speech -- Delay & Phase-Consistency Diagnostics

## 1) Executive Summary

- Run: results/<run>/
- Mode: smoke / scale_check_subset / full_dataset
- Pairing mode: paired / mispair_shift1
- Outcome: PASS / FAIL / PASS_WITH_WARNINGS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability)
- Q2: Does DT remain non-degenerate at the target compute regime? (DT>Random at high lambda)
- Q3: Does a coarse delay proxy (first selected lag) remain stable across frequency and across windows?
- Q4: Does the mispair guardrail break delay stability (as expected)?

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES/NO

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (paste `python --version`)
- Device(s): cpu / mps (and rationale)
- MPLCONFIGDIR: (value used; recommended /tmp/mpl)

### 2.2 Code provenance (Required)

- code_state.json: results/<run>/code_state.json
  - git_head: <hash>
  - dirty: true/false
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: <sha256>
    - scripts/h_exploration/dataset_lag.py: <sha256>

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode:
  - paired: MIC->LDV filename replacement
  - mispair_shift1: LDV list shifted by +1 (cyclic wrap)
- Dataset length check (preflight output):
  - len(dataset) = <value>
  - first_pair = <path1>, <path2>
  - last_pair  = <path1>, <path2>
- Subset manifest: results/<run>/subset_manifest.json
  - num_pairs = <N>
  - fingerprint_md5 = <hash>
  - Domain validation:
    - all paths end with .wav: YES/NO (must be YES)
    - any .npy paths: YES/NO (must be NO)

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

### 2.5 Step/compute definitions (Must be explicit)

- k_selected:
  - number of selected lag atoms (number of LS solves; compute-aligned)
- steps_decision:
  - number of decision steps including STOP (policy-step-aligned)
- lag_frames:
  - lag_frames = action_id + (-max_lag)
  - coarse delay proxy in STFT-frame units (not a final TDoA)

## 3) Exact Commands (Required)

Paste the exact command(s) executed, including:
- env exports (PYTHONPATH, MPLCONFIGDIR)
- lockdir pattern
- tee -a to run.log

## 4) Results (Required)

All numeric fields below must be copied from:
- summary/compute_matched_summary.json
- summary/rtg_controllability_summary.json
- summary/delay_diagnostics_summary.json
- run.log (for timing and warnings)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_samples_total = <int>
- num_samples_used = <int>
- num_missing_files = <int>
- num_md5_mismatches = <int>
- num_nan_or_inf = <int>
- num_capture_out_of_range_total = <int> (must be 0 for paper baseline)
- num_omp_monotonicity_violations = <int> (must be 0 for PASS)
- num_dt_duplicate_actions_forced_k = <int> (must be 0 for PASS)

Decision:
- PASS/FAIL with causal language (BECAUSE / THEREFORE).

### 4.1 RTG controllability (Compute vs lambda_c; Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = <value> (recommended <= -0.6)
- k_selected_mean list = <list aligned with lambda_c_values>
- steps_decision_mean list = <list aligned with lambda_c_values>

Interpretation:
- Compute decreases with higher lambda_c BECAUSE the learned STOP head trades off capture vs cost.

Decision:
- PASS/FAIL with causal language.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = <value> (expected ~12 region for this checkpoint)
- DT - Random (compute-matched capture mean) = <value> (must be > 0 for PASS)
- DT/OMP = <value> (for context)

Decision:
- PASS/FAIL with causal language.

### 4.3 Delay / phase-consistency diagnostics (Required)

From summary/delay_diagnostics_summary.json (one row per lambda_c), report at minimum:
- dt_stop0_frac_mean
- dt_first_lag_mad_frames_p50 and p90
- dt_vs_omp_first_lag_match_frac_mean
- Optional: dt_first_lag_median_frames_mean (for sign / direction sanity check)

Add a small table across all lambdas (recommended):
- lambda_c
- k_selected_mean (from compute_matched_summary)
- dt_first_lag_mad_p50 (from delay summary)
- dt_stop0_frac_mean

Interpretation hints (use causal language):
- If dt_first_lag_mad increases when lambda_c increases, it suggests delay estimates become less frequency-consistent
  as compute is reduced, DUE TO earlier stopping leaving residual structure unexplained.
- If dt_stop0_frac increases sharply, compute reduction may be coming from immediate STOP on many bins, THEREFORE delay
  diagnostics may be computed on fewer bins and should be interpreted with dt_first_lag_defined_frac.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Compare paired vs mispair (on the same mode, ideally scale_check_subset):
- For each lambda_c, compare dt_first_lag_mad_p50.
- State whether mispair > paired for at least 4/5 lambdas (spec acceptance).

Decision:
- PASS if separation holds.
- PASS_WITH_WARNINGS if separation does not hold but pipeline is valid (explain why).

## 5) Physical / Mathematical Analysis (Required)

Write a first-principles explanation connecting:
- Pure delay model: phase approximately linear with frequency (group delay approximately constant).
- Dispersion / material resonances: frequency-dependent phase, causing different frequencies to prefer different lags.

Use causal phrases:
- BECAUSE of ...
- DUE TO ...
- THEREFORE ...
- THIS IMPLIES ...

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference at least 3 prior Results commits (hashes) and connect patterns causally.

Example structure:
- Pattern recognition: ...
- Success factors: ...
- Failure modes: ...
- Parameter sensitivity: ...

## 7) Extracted Principles for Next Steps (Required)

Convert observations into actionable rules:
- If delay stability is good at high lambda, THEN Phase-2b can proceed to sub-sample refinement (GCC-PHAT / phase-slope).
- If delay stability collapses with lambda, THEN re-train with a wider lambda grid and/or add explicit delay-consistency
  regularization or a different STOP/cost calibration.

## 8) Reproduction Instructions (Required)

Provide step-by-step commands:

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- Paste the exact evaluator command you ran (include pairing_mode, require_wav_only, write_delay_diagnostics).

3) Verification:
- Expected outputs:
  - subset_manifest.json (wav-only)
  - summary/compute_matched_summary.json
  - summary/rtg_controllability_summary.json
  - delay_diagnostics.jsonl
  - summary/delay_diagnostics_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json
  - ACCEPTANCE_REPORT.md

## 9) Failures / Limitations (Required even if PASS)

State remaining risks and limitations:
- Frame-level lag resolution is 10ms; not a final TDoA.
- Need sub-sample refinement + geometry-grounded evaluation for Phase-2b/3.
- Any surprising guardrail behavior (mispair not worse) must be analyzed.

---

