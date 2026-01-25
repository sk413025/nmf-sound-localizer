# Acceptance Report: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_subsample_delay_E4l_speech_full_dataset_paired_20260126_023025/
- Mode: full_dataset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability) -> YES
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda) -> YES
- Q3: Do sub-sample delay estimators (GCC-PHAT / phase-slope) produce stable tau estimates on paired data? -> YES (GCC-PHAT)
- Q4: Does mispair_shift1 degrade tau stability and confidence (guardrail separation)? -> YES (validated on functional runs)

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu (chosen for stability and determinism)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_subsample_delay_E4l_speech_full_dataset_paired_20260126_023025/code_state.json
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
- Subset manifest: results/rtgomp_subsample_delay_E4l_speech_full_dataset_paired_20260126_023025/subset_manifest.json
  - num_pairs = 416
  - num_files = 832
  - fingerprint_md5 = 13356fdb74d2acb7e85361a4dfe5c3d2
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

OUT_DIR="results/rtgomp_subsample_delay_E4l_speech_full_dataset_paired_20260126_023025"
mkdir -p "$OUT_DIR/summary"

LOCKDIR="$OUT_DIR/.lock"
mkdir "$LOCKDIR"
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode full_dataset \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 0 --device cpu \
  --pairing_mode paired --require_wav_only 1 \
  --write_delay_diagnostics 1 \
  --write_subsample_delay_diagnostics 1 --subsample_method gcc_phat --search_radius_frames 2 \
  |& tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

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
- PASS BECAUSE all integrity counters are zero and required artifacts were written; THEREFORE the full-dataset results are trustworthy under the spec.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean list = [15.904, 15.462, 14.029, 12.817, 12.049]

Decision:
- PASS BECAUSE k_selected decreases monotonically as lambda_c increases; THEREFORE compute is controllable at dataset scale.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 12.049
- DT - Random (compute-matched capture mean) = 0.024830 (> 0)

Decision:
- PASS BECAUSE DT remains better than compute-matched Random at the target mean compute (~12); THEREFORE the policy stays non-degenerate at the intended compute regime.

### 4.3 Sub-sample delay diagnostics (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt") at lambda_c=3e-4:
- fraction_defined = 1.0
- boundary_hit_rate = 0.0

Small table across lambdas (dt coarse):

| lambda_c | k_selected_mean | DT-Random (mean) | psr_p50 | within_clip_tau_mad_ms_p50 | boundary_hit_rate |
|---:|---:|---:|---:|---:|---:|
| 1e-5 | 15.904 | 0.000754 | 53.264 | 0.010490 | 0.000 |
| 3e-5 | 15.462 | 0.001588 | 53.275 | 0.010490 | 0.000 |
| 1e-4 | 14.029 | 0.007014 | 53.306 | 0.010490 | 0.000 |
| 2e-4 | 12.817 | 0.016185 | 53.210 | 0.010490 | 0.000 |
| 3e-4 | 12.049 | 0.024830 | 53.259 | 0.010490 | 0.000 |

Interpretation:
- PSR and within-clip tau MAD are essentially constant across the lambda grid BECAUSE coarse lag guidance remains sufficient and the MIC/LDV pairing preserves a consistent in-band phase relationship; THEREFORE compute reduction (via higher lambda_c) does not degrade the GCC-PHAT delay stability in this Phase-2b diagnostic.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Guardrail runs (scale_check_subset, 48 pairs):
- paired: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/
- mispair_shift1: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/

Separation checks:
- For 5/5 lambdas:
  - psr_p50(paired, dt) > psr_p50(mispair, dt)
  - within_clip_tau_mad_ms_p50(paired, dt) < within_clip_tau_mad_ms_p50(mispair, dt)

Decision:
- PASS BECAUSE mispairing collapses PSR and inflates tau dispersion; THEREFORE the delay refinement diagnostics are sensitive and have a strong negative control.

## 5) Physical / Mathematical Analysis (Required)

- GCC-PHAT works BECAUSE a pure time delay produces a linear phase term across frequency; PHAT weighting emphasizes phase coherence rather than magnitude.
- The near-constant PSR and tau MAD across lambda implies the phase relationship is robust to changes in the selected lag set size, WHICH IMPLIES that the coarse lag estimate still lands close enough to the true delay for a narrow search.
- This is a necessary Phase-2b milestone toward MIC+LDV time alignment because it demonstrates sample-level delay estimability under an explicit compute knob.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech paper eval) validated the DT vs OMP vs Random evaluation harness on speech WAV.
- Commit db556db (E4j-Speech full_dataset) calibrated STOP/cost to reach mean compute ~12 at lambda_c=3e-4 while preserving DT>Random.
- Commit 16e66d0 (E4k-Speech) established coarse delay stability on paired speech and strong failure under mispair.

This run confirms the intended chain: E4j provides the compute operating point (~12), E4k shows coarse delay stability, and E4l shows sub-sample delay stability under that same compute control; THEREFORE we have evidence that Phase-1/2 work is compatible with downstream delay refinement rather than being a metric-only artifact.

## 7) Extracted Principles for Next Steps (Required)

- Keep using the 5-point lambda grid to jointly track compute (k_selected) and delay stability, BECAUSE we need to ensure any compute improvements do not silently break alignment.
- Proceed to a Phase-3 experiment that validates delay correctness with stronger ground truth (e.g., controlled known delays or geometry-based TDoA evaluation), BECAUSE Phase-2b only proves stability and guardrail sensitivity, not absolute physical correctness.
- If Phase-3 exposes bias (stable but wrong tau), add the optional phase-slope diagnostics to detect dispersion residuals and/or refine the estimator (e.g., band-dependent tau).

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
- Confirm outputs exist under results/rtgomp_subsample_delay_E4l_speech_full_dataset_paired_20260126_023025/:
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

- Phase-2b has no geometry ground truth, so we only claim stability and guardrail sensitivity, not physical correctness of TDoA.
- GCC-PHAT assumes a dominant delay; strong dispersion or multi-path can require frequency-dependent models.
- The constant low tau MAD does not guarantee the estimated tau is correct in absolute value; Phase-3 must validate against known delays or geometry.
