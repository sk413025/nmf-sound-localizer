# Acceptance Report Template: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

Use this template for E4l-Speech. Fill all placeholders. Keep everything in English.

You will typically write **two reports**:
- paired (positive path)
- mispair_shift1 (guardrail diagnostic)

Save the filled report to:
- results/<run>/ACCEPTANCE_REPORT.md

---

# Acceptance Report: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

## 1) Executive Summary

- Run: results/<run>/
- Mode: smoke / scale_check_subset / full_dataset
- Pairing mode: paired / mispair_shift1
- Outcome: PASS / FAIL / PASS_WITH_WARNINGS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability)
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda)
- Q3: Do sub-sample delay estimators (GCC-PHAT / phase-slope) produce stable tau estimates on paired data?
- Q4: Does mispair_shift1 degrade tau stability and confidence (guardrail separation)?

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
- Pairing mode: paired / mispair_shift1 (describe)
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
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- search_radius_frames = <int> (default 2)
- subsample_method = gcc_phat (+ phase_slope if enabled)

### 2.5 Definitions (Must be explicit)

- k_selected: number of selected lag atoms (compute-aligned)
- steps_decision: number of decision steps including STOP (policy-step-aligned)
- coarse_lag_frames: median first-lag across frequency (DT or OMP)
- tau_hat_samples: sub-sample delay estimate (define sign)
- psr: peak-to-sidelobe ratio (confidence)
- boundary_hit: peak at search boundary (confidence failure)
- within_clip_tau_mad_ms: per-clip stability of tau across windows

## 3) Exact Commands (Required)

Paste the exact command(s) executed, including tee, lockdir pattern, and env exports.

## 4) Results (Required)

All numeric fields below must be copied from:
- summary/compute_matched_summary.json
- summary/rtg_controllability_summary.json
- summary/subsample_delay_diagnostics_summary.json
- run.log

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_missing_files = <int>
- num_md5_mismatches = <int>
- num_nan_or_inf = <int>
- num_capture_out_of_range_total = <int>
- num_omp_monotonicity_violations = <int>
- num_dt_duplicate_actions_forced_k = <int>

E4l-specific hard checks:
- any fs mismatch encountered: YES/NO (must be NO; otherwise FAIL)
- subsample_delay_diagnostics.jsonl exists: YES/NO
- summary/subsample_delay_diagnostics_summary.json exists: YES/NO

Decision:
- PASS/FAIL with causal language.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = <value>
- k_selected_mean list = <list>

Decision:
- PASS/FAIL with causal language.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = <value>
- DT - Random (compute-matched capture mean) = <value> (must be > 0 for PASS)

Decision:
- PASS/FAIL with causal language.

### 4.3 Sub-sample delay diagnostics (Required)

From summary/subsample_delay_diagnostics_summary.json, report at minimum (for coarse_source="dt"):
- fraction_defined at lambda_c=3e-4
- boundary_hit_rate at lambda_c=3e-4
- psr_p50/p90 across lambdas
- within_clip_tau_mad_ms_p50/p90 across lambdas

Add a small table across lambdas (recommended):
- lambda_c
- k_selected_mean
- psr_p50 (dt)
- within_clip_tau_mad_ms_p50 (dt)
- boundary_hit_rate (dt)

Interpretation hints (use causal language):
- If psr decreases and boundary_hit rises with lambda, compute reduction harms coarse guidance quality.
- If within-clip tau MAD stays low while compute decreases, sub-sample refinement is robust under compute control.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Compare paired vs mispair on scale_check_subset:
- psr_p50(paired, dt) > psr_p50(mispair, dt) for >=4/5 lambdas
- within_clip_tau_mad_ms_p50(paired, dt) < within_clip_tau_mad_ms_p50(mispair, dt) for >=4/5 lambdas

Decision:
- PASS if separation holds.
- PASS_WITH_WARNINGS if paired passes but separation fails (must analyze why).

## 5) Physical / Mathematical Analysis (Required)

Explain from first principles:
- Why GCC-PHAT peak indicates delay when phase is coherent.
- How dispersion/phase wrapping reduces coherence and flattens the correlation peak.
- Why coarse lag consistency (E4k) is necessary but not sufficient for sub-sample stability.

Use causal phrases:
- BECAUSE / DUE TO / THEREFORE / THIS IMPLIES

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference at least 3 Results commits (hashes) and connect patterns causally.

## 7) Extracted Principles for Next Steps (Required)

Convert observations into rules:
- If paired stability is good and guardrail separation is strong, THEN Phase-3 can proceed to geometry-grounded TDoA.
- If stability collapses at high lambda, THEN adjust coarse guidance or search strategy or cost calibration.

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- Paste exact command(s).

3) Verification:
- Expected outputs:
  - subset_manifest.json (wav-only)
  - subsample_delay_diagnostics.jsonl
  - summary/subsample_delay_diagnostics_summary.json
  - summary/compute_matched_summary.json
  - summary/rtg_controllability_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json
  - ACCEPTANCE_REPORT.md

## 9) Failures / Limitations (Required even if PASS)

- Any estimator limitations (peak ambiguity, low SNR).
- Lack of geometry-ground truth if applicable.
- Remaining gap to a full TDoA claim.

---

