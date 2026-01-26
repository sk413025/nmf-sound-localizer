# Acceptance Report: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

## 1) Executive Summary

- Run: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910/
- Mode: scale_check_subset
- Pairing mode: mispair_shift1
- Outcome: PASS (guardrail diagnostic behaved as expected)
- Readiness decision: NOT_READY (guardrail run; readiness must be decided on paired full_dataset)

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
- Device(s): cpu (chosen for stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910/code_state.json
  - git_head: 9e6e402f1c0f75b52857fb6caae3908fa739d592
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 99bc4a99a19ad775f025db2727f24655031e7158e73fe1f17e832cb21cfdfd0f
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: mispair_shift1
  - MIC list unchanged.
  - LDV list shifted by +1 index with cyclic wrap.
  - Purpose: guardrail diagnostic that intentionally breaks true pairing.
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910/subset_manifest.json
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
OUT_DIR="results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910"
LOCKDIR="$OUT_DIR/.lock"

mkdir -p "$OUT_DIR/summary"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
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
  --write_per_sample 0 --device cpu \
  --pairing_mode mispair_shift1 --require_wav_only 1 \
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
- PASS BECAUSE the pipeline executes cleanly on real speech WAV with no integrity violations; THEREFORE this guardrail run is valid for separation diagnostics.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.000
- k_selected_mean list = [15.928, 15.613, 14.534, 13.590, 12.974]

Decision:
- PASS BECAUSE compute remains controllable even under mispairing; THIS IMPLIES compute control is a property of the policy/cost mechanism rather than pairing correctness.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 12.974
- DT - Random (compute-matched capture mean) = 0.0112 (> 0)

Decision:
- PASS BECAUSE DT still beats compute-matched Random in capture even when pairing is wrong; THIS IMPLIES the capture objective is not a direct proof of correct MIC-LDV alignment and must be combined with the delay diagnostics.

### 4.3 GCC-PHAT stability (Required)

From summary/subsample_delay_diagnostics_summary.json, coarse_source="dt":

At lambda_c=3e-4:
- fraction_defined = 1.000
- boundary_hit_rate = 0.065
- psr_p50/p90 = 7.381 / 87.771
- within_clip_tau_mad_ms_p50/p90 = 15.567 / 39.921

Decision:
- EXPECTED_GUARDRAIL_FAILURE BECAUSE mispairing breaks true alignment, THEREFORE the refined GCC-PHAT tau becomes highly unstable within each clip (large within-clip MAD) and the median PSR collapses.

### 4.4 Phase-slope fit quality and dispersion (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt"):

At lambda_c=3e-4:
- phase_slope_fraction_defined = 0.177
- phase_slope_tau_hat_ms_p50/p90 = -13.749 / 41.447
- phase_slope_r2_p50/p90 = 0.663 / 0.942
- phase_slope_fit_rmse_rad_p50/p90 = 20.510 / 44.828
- abs_tau_agreement_ms_p50/p90 = 11.364 / 20.671
- tau_band_spread_ms_p50/p90 = 13.577 / 25.667

Interpretation:
- phase_slope_fraction_defined collapses BECAUSE many windows cannot produce a constrained delay estimate within the coarse search window under wrong pairing; THEREFORE phase-slope is sensitive to pairing correctness.
- Even when defined, RMSE is extremely large BECAUSE the cross-spectrum phase is effectively unrelated under mispairing; THIS IMPLIES any apparent R^2 can be misleading due to selection bias (only a subset of windows are defined).

### 4.5 Guardrail separation (paired vs mispair_shift1; Required)

Comparing against the paired functional run (results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/), dt coarse:
- psr_p50: paired ~56.9 vs mispair ~7-8 across all lambdas
- within_clip_tau_mad_ms_p50: paired ~0.008 ms vs mispair ~12-18 ms across all lambdas
- phase_slope_fraction_defined: paired ~0.970 vs mispair ~0.14-0.18 across all lambdas
- tau_band_spread_ms_p50: paired ~10 ms vs mispair ~14-15 ms across all lambdas

Decision:
- PASS BECAUSE paired vs mispair separates strongly in confidence, stability, and dispersion metrics; THEREFORE the diagnostics are not trivially constant and respond to true MIC-LDV pairing.

### 4.6 Readiness decision (Required)

- NOT_READY BECAUSE this run intentionally violates MIC-LDV pairing, THEREFORE it cannot be used to decide readiness for Phase-3 or calibration.

## 5) Physical / Mathematical Analysis (Required)

- Wrong pairing destroys a single-delay model BECAUSE the MIC and LDV signals come from different utterances; THEREFORE no consistent tau should exist.
- As a result, correlation peaks become unstable and phase-slope becomes undefined or high-error BECAUSE the cross-spectrum is dominated by unrelated content.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 16e66d0 (E4k-Speech) established that mispair_shift1 is a strong guardrail that breaks delay consistency; THIS RUN reproduces the same pattern with additional dispersion diagnostics.
- Commit 9e6e402 (E4l-Speech) showed GCC-PHAT refinement is stable for paired data and unstable for mispair; THIS RUN confirms the instability at the waveform level.
- Commit 6af8a63 (E4h-Speech) and 76d63fe/db556db (E4j-Speech) show compute control and DT>Random can hold even without alignment guarantees; THEREFORE we must use explicit delay/dispersion diagnostics to judge readiness.

## 7) Extracted Principles for Next Steps (Required)

- Always include a mispair guardrail run BECAUSE it validates that metrics respond to true pairing.
- Do not interpret capture (DT>Random) as a proof of alignment BECAUSE it can remain positive even under wrong pairing; THEREFORE use delay/dispersion diagnostics as the correctness gate.

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
- Expected outputs exist under results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_mispair_shift1_20260126_105910/:
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

- This run is intentionally incorrect pairing; its purpose is separation, not performance.
- R^2 under mispair can appear high on the small subset of defined windows; THIS IMPLIES RMSE and defined fraction must be considered together.
