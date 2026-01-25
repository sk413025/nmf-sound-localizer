# Acceptance Report: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/
- Mode: scale_check_subset
- Pairing mode: mispair_shift1
- Outcome: PASS (guardrail diagnostic triggers as expected)

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability) -> YES (even under mispair)
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda) -> YES (but weaker)
- Q3: Do sub-sample delay estimators (GCC-PHAT / phase-slope) produce stable tau estimates on paired data? -> NOT APPLICABLE here (this run is intentionally mispaired)
- Q4: Does mispair_shift1 degrade tau stability and confidence (guardrail separation)? -> YES (strongly)

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu (chosen for stability and determinism)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/code_state.json
  - git_head: 16e66d0bf00b38a3ef7d4fa1d0bcab5c86f92032
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: b22740e9ee2d4f9b7009ffe58d7dd7cc2fc7955cd0fea67d53b114198dd6d42b
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: mispair_shift1
  - MIC list unchanged.
  - LDV list shifted by +1 index (cyclic wrap).
  - Example: MIC_001.wav paired with LDV_002.wav.
- Subset manifest: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 5997f1271e9a08d833083897c5c41fc1
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

OUT_DIR="results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335"
mkdir -p "$OUT_DIR/summary"

LOCKDIR="$OUT_DIR/.lock"
mkdir "$LOCKDIR"
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode scale_check_subset --num_pairs 48 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 1 --device cpu \
  --pairing_mode mispair_shift1 --require_wav_only 1 \
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
- PASS BECAUSE the evaluator ran correctly end-to-end with all hard guardrails satisfied; THEREFORE the guardrail diagnostic is valid.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean list = [15.928, 15.613, 14.534, 13.590, 12.974]

Decision:
- PASS BECAUSE compute remains controllable even under mispairing; THEREFORE the compute knob is not an artifact of correct pairing.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 12.974
- DT - Random (compute-matched capture mean) = 0.011170 (> 0)

Decision:
- PASS (weak) BECAUSE DT remains > Random in capture metric, but the margin shrinks DUE TO loss of true MIC↔LDV alignment; THEREFORE capture-based metrics alone are not sufficient to detect mispairing.

### 4.3 Sub-sample delay diagnostics (Required)

This run is intentionally mispaired, so the expected behavior is:
- PSR collapses,
- within-clip tau MAD inflates,
- and boundary_hit may increase.

At lambda_c = 3e-4 (coarse_source="dt"):
- fraction_defined = 1.0
- boundary_hit_rate = 0.065
- psr_p50 = 7.381
- within_clip_tau_mad_ms_p50 = 15.567 ms

Small table across lambdas (dt coarse):

| lambda_c | k_selected_mean | psr_p50 | within_clip_tau_mad_ms_p50 | boundary_hit_rate |
|---:|---:|---:|---:|---:|
| 1e-5 | 15.928 | 8.547 | 17.392 | 0.084 |
| 3e-5 | 15.613 | 8.250 | 17.832 | 0.074 |
| 1e-4 | 14.534 | 7.057 | 12.665 | 0.070 |
| 2e-4 | 13.590 | 7.444 | 13.748 | 0.070 |
| 3e-4 | 12.974 | 7.381 | 15.567 | 0.065 |

Interpretation:
- PSR is low and tau MAD is large BECAUSE the MIC and LDV windows do not share a common underlying waveform; THEREFORE the cross-spectrum phase is incoherent and the correlation peak is ambiguous.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Paired reference run:
- results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/

Separation checks (48 pairs):
- For 5/5 lambdas:
  - psr_p50(paired, dt) > psr_p50(mispair, dt)
  - within_clip_tau_mad_ms_p50(paired, dt) < within_clip_tau_mad_ms_p50(mispair, dt)

Decision:
- PASS BECAUSE mispairing reliably degrades sub-sample delay confidence and stability; THEREFORE the diagnostics have a strong negative control.

## 5) Physical / Mathematical Analysis (Required)

- GCC-PHAT relies on coherent phase differences across frequency BECAUSE a time delay induces a linear phase term.
- In mispair_shift1, the MIC and LDV clips come from different utterances; THEREFORE the cross-spectrum phase behaves like noise and PHAT weighting cannot create a consistent peak.
- The large within-clip tau MAD is expected DUE TO the estimator selecting different spurious peaks across windows when no true alignment exists.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech) established baseline DT vs OMP vs Random plumbing on speech WAV.
- Commit db556db (E4j-Speech full_dataset) calibrated STOP/cost to reach compute ~12 at lambda_c=3e-4 with DT>Random.
- Commit 16e66d0 (E4k-Speech) showed coarse delay stability on paired speech and failure under mispair guardrail.

This run strengthens the guardrail story by showing that sub-sample (waveform-domain) delay diagnostics fail strongly under mispairing, WHICH IMPLIES the estimator is not trivially “always confident” and can detect loss of shared content.

## 7) Extracted Principles for Next Steps (Required)

- Use capture-based metrics for policy quality, but rely on waveform-domain diagnostics (PSR, within-clip tau MAD) to validate alignment, BECAUSE capture can remain superficially high even under mispair.
- Treat mispair_shift1 as a required negative control for any future delay refinement method.

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
- Confirm outputs exist under results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/ (subset_manifest.json, summaries, jsonl, run.log, code_state.json, ACCEPTANCE_REPORT.md).

## 9) Failures / Limitations (Required even if PASS)

- Mispair_shift1 is a strong negative control but not physically realistic; real failures may be subtler (e.g., mild drift, dispersion).
- This diagnostic cannot distinguish between “true dispersion” vs “mispair” without additional context; it only indicates lack of a stable single delay.
