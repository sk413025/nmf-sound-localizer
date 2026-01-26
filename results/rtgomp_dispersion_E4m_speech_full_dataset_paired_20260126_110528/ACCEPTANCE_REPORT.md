# Acceptance Report: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

## 1) Executive Summary

- Run: results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/
- Mode: full_dataset
- Pairing mode: paired
- Outcome: PASS_WITH_WARNINGS
- Readiness decision: READY_FOR_CALIBRATION

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability)
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda)
- Q3: Does GCC-PHAT produce stable tau estimates on paired data under compute control?
- Q4: Does phase-slope fit indicate near-pure delay or dispersion (non-linear phase)?
- Q5: Do paired vs mispair_shift1 runs separate strongly in confidence and dispersion metrics?

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu (chosen for stability; avoids potential MPS stalls)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/code_state.json
  - git_head: 9e6e402f1c0f75b52857fb6caae3908fa739d592
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 99bc4a99a19ad775f025db2727f24655031e7158e73fe1f17e832cb21cfdfd0f
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired (MIC<->LDV paired by filename rule)
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/subset_manifest.json
  - num_pairs = 416
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
- subsample_method = gcc_phat,phase_slope
- phase_slope_min_bins = 64
- phase_slope_subbands_hz = [[300,900],[900,1800],[1800,3000]]

### 2.5 Definitions (Must be explicit)

- k_selected: number of selected lag atoms (compute-aligned)
- steps_decision: number of decision steps including STOP (policy-step-aligned)
- coarse_lag_frames: median first-lag across frequency (DT or OMP)
- tau_hat_samples: delay estimate sign convention: tau_hat > 0 means LDV lags MIC (y[t] ~= x[t - tau_hat])
- psr: peak-to-sidelobe ratio (confidence)
- boundary_hit: peak at search boundary (confidence failure)
- phase_slope_tau_hat_ms: group delay estimate from linear fit of unwrapped cross-spectrum phase
- phase_slope_fit_rmse_rad, phase_slope_r2: fit quality (dispersion signature)
- tau_agreement_ms: phase_slope_tau_hat_ms - gcc_phat_tau_hat_ms
- tau_band_spread_ms: max(tau_ms_by_band) - min(tau_ms_by_band)

## 3) Exact Commands (Required)

```bash
OUT_DIR="results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528"
LOCKDIR="$OUT_DIR/.lock"

mkdir -p "$OUT_DIR/summary"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
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
  --write_subsample_delay_diagnostics 1 --subsample_method gcc_phat,phase_slope --search_radius_frames 2 \
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

E4m-specific hard checks:
- any fs mismatch encountered: NO (would hard-fail)
- subsample_delay_diagnostics.jsonl exists: YES
- summary/subsample_delay_diagnostics_summary.json exists: YES

Decision:
- PASS BECAUSE all hard guardrails pass, THEREFORE the outputs reflect real-data behavior rather than pipeline errors.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.000
- k_selected_mean list = [15.904, 15.462, 14.029, 12.817, 12.049]

Decision:
- PASS BECAUSE k_selected_mean decreases monotonically with lambda_c and spearman is ~-1, THEREFORE lambda_c provides reliable compute control on full_dataset.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 12.049
- DT - Random (compute-matched capture mean) = 0.0248 (> 0)

Decision:
- PASS BECAUSE DT remains better than Random at the target compute regime, THEREFORE the policy is not collapsing when compute is reduced.

### 4.3 GCC-PHAT stability (Required)

From summary/subsample_delay_diagnostics_summary.json, coarse_source="dt":

At lambda_c=3e-4:
- fraction_defined = 1.000
- boundary_hit_rate = 0.000
- psr_p50/p90 = 53.259 / 83.593
- within_clip_tau_mad_ms_p50/p90 = 0.0105 / 0.0322

Decision:
- PASS BECAUSE refined waveform-domain tau remains stable under compute control on the full dataset, THEREFORE coarse-to-fine delay refinement is robust in the paired setting.

### 4.4 Phase-slope fit quality and dispersion (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt"):

At lambda_c=3e-4:
- phase_slope_fraction_defined = 0.904
- phase_slope_tau_hat_ms_p50/p90 = -3.321 / 6.802
- phase_slope_r2_p50/p90 = 0.501 / 0.834
- phase_slope_fit_rmse_rad_p50/p90 = 4.778 / 14.043
- abs_tau_agreement_ms_p50/p90 = 4.661 / 12.561
- tau_band_spread_ms_p50/p90 = 9.697 / 22.281

Interpretation:
- The phase-slope fit is not "pure-delay-like" BECAUSE R^2 is low at the median and RMSE is several radians, THEREFORE the cross-spectrum phase is substantially non-linear across 300-3000 Hz.
- The tau_band_spread_ms is large (median ~9.7 ms), WHICH IMPLIES group delay varies by frequency; THIS IS consistent with LDV path dispersion / phase distortion.
- Large abs_tau_agreement_ms indicates disagreement between GCC and phase-slope; THEREFORE a single tau estimate may be stable-but-biased and calibration is required before moving to Phase-3 correctness evaluation.

### 4.5 Guardrail separation (paired vs mispair_shift1; Required)

Using the functional runs on 48 pairs:
- paired: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/
- mispair_shift1: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910/

For dt coarse, for all 5 lambdas:
- psr_p50(paired) >> psr_p50(mispair) (paired ~56.9 vs mispair ~7-8)
- within_clip_tau_mad_ms_p50(paired) << within_clip_tau_mad_ms_p50(mispair) (paired ~0.008 ms vs mispair ~12-18 ms)
- phase_slope_fraction_defined(paired) >> phase_slope_fraction_defined(mispair) (paired ~0.970 vs mispair ~0.14-0.18)
- tau_band_spread_ms_p50(paired) < tau_band_spread_ms_p50(mispair) (paired ~10 ms vs mispair ~14-15 ms)

Decision:
- PASS BECAUSE the paired and mispair runs separate strongly across multiple independent metrics; THEREFORE the diagnostics are sensitive to true MIC-LDV pairing.

### 4.6 Readiness decision (Required)

Decision rule at lambda_c=3e-4 (paired full_dataset, dt coarse):
- READY_FOR_PHASE3 if abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms
- Observed: abs_tau_agreement_ms_p50 = 4.661 ms, tau_band_spread_ms_p50 = 9.697 ms

Readiness:
- READY_FOR_CALIBRATION BECAUSE both agreement and band-spread thresholds fail by large margins; THEREFORE we should design phase equalization / dispersion compensation before any geometry-grounded TDoA evaluation.

## 5) Physical / Mathematical Analysis (Required)

- Pure delay implies linear phase: If y(t) ~= x(t - tau), then Y(f) ~= X(f) * exp(-j 2 pi f tau), so phi(f)=angle(conj(X)*Y) ~= -2 pi f tau + b. Therefore a weighted linear fit of phi vs f should have high R^2 and low RMSE.
- Dispersion implies non-linear phase: If the LDV path contributes frequency-dependent phase (e.g., material resonances, all-pass distortion), then phi(f) is not linear. Therefore the phase-slope linear model fails (lower R^2, higher RMSE) and different sub-bands yield different tau estimates.
- Why GCC can stay stable: GCC-PHAT produces a single in-band alignment peak and can remain stable even when the true phase is not linear. Therefore GCC stability is necessary but not sufficient for correctness; THIS IMPLIES we need calibration if phase-slope indicates dispersion.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech paper eval) showed DT outperforms Random on speech WAV BECAUSE DT exploits structured spectral redundancy.
- Commit 76d63fe / db556db (E4j-Speech STOP/cost calibration) achieved compute ~12 at lambda_c=3e-4 while keeping DT>Random; THEREFORE the compute regime evaluated here is the intended target.
- Commit 9e6e402 (E4l-Speech) established GCC-PHAT sub-sample refinement is stable for paired data and unstable for mispair; THIS RUN confirms stability on full_dataset.
- This E4m run adds phase-slope dispersion diagnostics and shows that despite GCC stability, phase is not well explained by a pure delay; THEREFORE the next step is calibration rather than Phase-3 correctness.

## 7) Extracted Principles for Next Steps (Required)

- Do not claim TDoA correctness from stability alone BECAUSE stable-but-biased delay estimates can occur under dispersion.
- Use phase-slope fit (R^2/RMSE) and band-spread as the readiness gate: if low dispersion, proceed to Phase-3; if high dispersion, prioritize calibration.
- Keep mispair_shift1 guardrail in every phase BECAUSE it proves diagnostics are sensitive to true pairing.

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- Run the exact command in Section 3.

3) Verification:
- Expected outputs exist under results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/:
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

- Phase-slope limitations: unwrap/fit can be sensitive to low-magnitude bins; however the consistent large band-spread across the dataset is strong evidence of dispersion.
- No geometry ground truth: we still need a Phase-3 experiment with known physical delay to evaluate correctness after calibration.
- Compute vs correctness: compute control and DT>Random do not guarantee correct delay; THIS IMPLIES calibration is the next gating step.
