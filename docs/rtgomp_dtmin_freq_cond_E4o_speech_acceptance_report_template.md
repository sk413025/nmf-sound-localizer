# Acceptance Report Template: E4o-Speech — DTmin Frequency Conditioning Audit

Use this template for E4o-Speech. Fill all placeholders. Keep everything in English.

Save the filled report to:
- `results/<run>/ACCEPTANCE_REPORT.md`

---

# Acceptance Report: E4o-Speech — DTmin Frequency Conditioning Audit

## 1) Executive Summary

- Run: results/<run>/
- Mode: smoke / scale_check_subset / full_dataset
- Pairing mode: paired / mispair_shift1
- freq_cond_mode: normal / shuffle / constant / zero_embed
- freq_cond_seed: <int or N/A>
- freq_cond_constant_idx: <int or N/A>
- Band: [freq_min, freq_max] Hz
- Outcome: PASS / PASS_WITH_WARNINGS / FAIL
- Classification (required): FREQ_COND_USED / FREQ_COND_IGNORED / INCONCLUSIVE

Primary questions answered:
- Q1: Does breaking `freq_idx` (shuffle/constant/zero) degrade DT performance?
- Q2: Are effects consistent across the 5-point lambda grid (compute knob)?
- Q3 (optional): Is the effect stronger in high frequency bands?

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
  - len(dataset) = <value> (expected 416)
  - first_pair = <path1>, <path2>
  - last_pair  = <path1>, <path2>
- Subset manifest: results/<run>/subset_manifest.json
  - num_pairs = <N>
  - fingerprint_md5 = <hash>
  - Domain validation:
    - all paths end with .wav: YES/NO (must be YES if require_wav_only=1)
    - any .npy paths: YES/NO (must be NO)

### 2.4 Fixed parameters (Must be explicit)

Common:
- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [freq_min, freq_max] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- checkpoint: (paste path)

freq_cond settings:
- freq_cond_mode: <value>
- freq_cond_seed: <value or N/A>
- freq_cond_constant_idx: <value or N/A>

## 3) Exact Commands (Required)

Paste the exact command executed, including tee, lockdir pattern, and env exports.

## 4) Results (Required)

### 4.0 Hard guardrails (Must pass)

- wav-only manifest: PASS/FAIL
- fs mismatch encountered: YES/NO (must be NO)
- any NaN/Inf in summary outputs: YES/NO (must be NO)
- required artifacts exist:
  - summary/compute_matched_summary.json: YES/NO
  - summary/rtg_controllability_summary.json: YES/NO
  - summary/freq_cond_audit_summary.json: YES/NO

Decision:
- PASS/FAIL with causal language.

### 4.1 Key metrics (copy/paste values)

From `summary/rtg_controllability_summary.json`:
- spearman(lambda_c, k_selected_mean) = <value>
- k_selected_mean list = <list>

From `summary/compute_matched_summary.json` (at each lambda_c; provide 5 rows):
- DT compute-matched capture mean
- OMP compute-matched capture mean
- Random compute-matched capture mean
- DT - Random (compute-matched)

From `summary/freq_cond_audit_summary.json`:
- confirm freq_cond_mode/seed/constant_idx, start_bin/end_bin

### 4.2 Classification for this run (required)

If this run is NOT `normal`:
- Define the reference run: results/<normal_run>/ (must exist)
- Extract the target metric at lambda_c=3e-4:
  - DT_cm(normal) = <value>
  - DT_cm(this run) = <value>
  - Δ = DT_cm(normal) - DT_cm(this run) = <value>

If this run IS `normal`:
- Leave placeholders for deltas (computed in the aggregated report) or note “reference run”.

Classification decision (must use the spec’s rules):
- FREQ_COND_USED / FREQ_COND_IGNORED / INCONCLUSIVE

Explain causally:
- BECAUSE / DUE TO / THEREFORE / THIS IMPLIES

## 5) Physical / Mathematical Analysis (Required)

Explain from first principles:
- Why frequency-dependent MIC→LDV transfer can require different optimal lags per frequency.
- Why a frequency embedding is a plausible conditioning mechanism (contextual policy) but can also degenerate into a lookup table.
- Why shuffling or collapsing `freq_idx` should reduce performance if the model uses frequency conditioning.

Use causal phrases:
- BECAUSE / DUE TO / THEREFORE / THIS IMPLIES

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference at least 3 Results commits (hashes) relevant to DTmin + speech eval, e.g.:
- E4h-Speech paper eval
- E4j-Speech compute calibration (checkpoint source)
- E4m-Speech dispersion diagnostics baseline

Connect patterns causally (BECAUSE/DUE TO/THEREFORE).

## 7) Extracted Principles for Next Steps (Required)

Convert observations into actionable rules, e.g.:
- If FREQ_COND_USED, THEN preserve `freq_idx` integrity in all future pipelines and consider band-specific evaluation.
- If FREQ_COND_IGNORED, THEN simplify the model (remove freq_embed) or redesign conditioning (band tokens / physics features).
- If INCONCLUSIVE, THEN increase sample size (full_dataset) and/or reduce noise sources (device, seeds).

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Preflight:
```bash
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav | head
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav | head
```

3) Execution:
- Paste the exact command from section 3.

4) Verification:
- Confirm files exist:
  - results/<run>/summary/compute_matched_summary.json
  - results/<run>/summary/rtg_controllability_summary.json
  - results/<run>/summary/freq_cond_audit_summary.json
- Confirm no wav/fs guardrail failures in `run.log`.

