# Acceptance Report: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_subsample_delay_E4l_speech_smoke_paired_20260126_021616/
- Mode: smoke
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability) -> YES (smoke-level evidence)
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda) -> YES
- Q3: Do sub-sample delay estimators (GCC-PHAT / phase-slope) produce stable tau estimates on paired data? -> YES (GCC-PHAT)
- Q4: Does mispair_shift1 degrade tau stability and confidence (guardrail separation)? -> NOT EVALUATED in smoke; see functional runs.

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu (chosen for stability and determinism)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_subsample_delay_E4l_speech_smoke_paired_20260126_021616/code_state.json
  - git_head: 16e66d0bf00b38a3ef7d4fa1d0bcab5c86f92032
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: b22740e9ee2d4f9b7009ffe58d7dd7cc2fc7955cd0fea67d53b114198dd6d42b
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired (MIC_i paired with LDV_i by filename rule)
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_001.wav , /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_001.wav
  - last_pair  = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/boy1_papercup_MIC_xnonoise_320.wav , /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_subsample_delay_E4l_speech_smoke_paired_20260126_021616/subset_manifest.json
  - num_pairs = 1
  - fingerprint_md5 = 9b169d8cb234f30431fc7178d5aafb33
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

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
- search_radius_frames = 2
- subsample_method = gcc_phat

### 2.5 Definitions (Must be explicit)

- k_selected: number of selected lag atoms (compute-aligned)
- steps_decision: number of decision steps including STOP (policy-step-aligned)
- coarse_lag_frames: median first-lag across frequency (DT or OMP)
- tau_hat_samples: GCC-PHAT delay estimate where tau_hat > 0 means LDV lags MIC (y[t] ~= x[t - tau_hat])
- psr: peak-to-sidelobe ratio (higher is more confident)
- boundary_hit: peak at search boundary (failure indicator)
- within_clip_tau_mad_ms: per-clip stability of tau across windows (median absolute deviation)

## 3) Exact Commands (Required)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

OUT_DIR="results/rtgomp_subsample_delay_E4l_speech_smoke_paired_20260126_021616"
mkdir -p "$OUT_DIR/summary"

# lockdir pattern
LOCKDIR="$OUT_DIR/.lock"
mkdir "$LOCKDIR"
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode smoke --num_pairs 1 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 1 --device cpu \
  --pairing_mode paired --require_wav_only 1 \
  --write_delay_diagnostics 1 \
  --write_subsample_delay_diagnostics 1 --subsample_method gcc_phat --search_radius_frames 2 \
  |& tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

All numeric fields below are copied from:
- summary/compute_matched_summary.json
- summary/rtg_controllability_summary.json
- summary/subsample_delay_diagnostics_summary.json
- run.log

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0

E4l-specific hard checks:
- any fs mismatch encountered: NO (would have failed fast)
- subsample_delay_diagnostics.jsonl exists: YES
- summary/subsample_delay_diagnostics_summary.json exists: YES

Decision:
- PASS BECAUSE all integrity counters are zero and required artifacts were written; THEREFORE the evaluator executed end-to-end on real WAV without silent fallbacks.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean list = [15.900, 15.429, 13.999, 12.747, 11.976]

Decision:
- PASS BECAUSE k_selected decreases monotonically as lambda_c increases; THEREFORE compute is controllable on speech in this configuration.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 11.976
- DT - Random (compute-matched capture mean) = 0.034815 (> 0)

Decision:
- PASS BECAUSE DT remains better than compute-matched Random even at the high-cost regime; THEREFORE the policy is not collapsing when compute is reduced.

### 4.3 Sub-sample delay diagnostics (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt") at lambda_c=3e-4:
- fraction_defined = 1.0
- boundary_hit_rate = 0.0

Small table across lambdas (dt coarse):

| lambda_c | k_selected_mean | psr_p50 | within_clip_tau_mad_ms_p50 | boundary_hit_rate |
|---:|---:|---:|---:|---:|
| 1e-5 | 15.900 | 59.824 | 0.010094 | 0.000 |
| 3e-5 | 15.429 | 59.824 | 0.010094 | 0.000 |
| 1e-4 | 13.999 | 59.824 | 0.010094 | 0.000 |
| 2e-4 | 12.747 | 59.824 | 0.010094 | 0.000 |
| 3e-4 | 11.976 | 59.824 | 0.010094 | 0.000 |

Interpretation:
- PSR stays high while k_selected decreases BECAUSE the MIC/LDV pairing is correct and band-limited phase remains coherent within the search window; THEREFORE the GCC-PHAT peak remains sharp under this compute knob (at least on this 1-pair smoke).

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

- Not evaluated in smoke.
- Guardrail separation is evaluated on scale_check_subset runs:
  - paired: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/
  - mispair: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/

## 5) Physical / Mathematical Analysis (Required)

- GCC-PHAT identifies delay from the cross-spectrum phase agreement BECAUSE PHAT normalizes away magnitude coloration and emphasizes phase consistency across frequency.
- A correct MIC↔LDV pair tends to preserve a dominant relative delay within a short window, so the correlation peak is sharp and PSR is high.
- If the LDV path introduces frequency-dependent phase distortion (dispersion / phase wrapping), then the phase is not globally consistent with a single delay; THIS IMPLIES the correlation peak can broaden or become ambiguous, increasing within-clip tau dispersion.
- Coarse lag stability (E4k) is necessary but not sufficient BECAUSE selecting consistent STFT-frame lags does not guarantee sample-level phase linearity across the full band.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech paper eval) established DT vs OMP vs Random baselines on speech WAV.
- Commit db556db (E4j-Speech full_dataset) tuned STOP/cost so mean compute approaches ~12 at lambda_c=3e-4 while keeping DT>Random.
- Commit 16e66d0 (E4k-Speech) showed coarse delay stability on paired speech and failure under mispair guardrail.

This smoke run extends the chain by showing that, GIVEN coarse guidance exists (E4k), GCC-PHAT can produce defined sub-sample tau estimates on a real paired window, WHICH IS CONSISTENT with the hypothesis that the MIC/LDV pair still contains time-alignment information in-band.

## 7) Extracted Principles for Next Steps (Required)

- If compute control preserves DT>Random AND GCC-PHAT remains high-PSR with low within-clip tau MAD, THEN we can proceed to a Phase-3 geometry-grounded evaluation (TDoA-like) without immediately changing the compute knob.
- If future runs show tau MAD inflating at high lambda, THEN coarse guidance is becoming too weak and we should recalibrate STOP/cost or widen search radius (with explicit acceptance thresholds).

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- Use the exact command in section 3.

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

- This is a smoke run (1 pair), so it cannot validate guardrail separation or dataset-wide stability.
- GCC-PHAT assumes a dominant time delay; strong dispersion or multi-path can violate this and reduce peak sharpness.
- No geometry-ground truth is used here; stable tau does not yet imply correct physical TDoA without a controlled setup.
