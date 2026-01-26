# Acceptance Report: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

## 1) Executive Summary

- Run: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/
- Mode: scale_check_subset
- Pairing mode: paired
- Outcome: PASS_WITH_WARNINGS
- Readiness decision: READY_FOR_CALIBRATION (preliminary; final decision is based on full_dataset)

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

- code_state.json: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/code_state.json
  - git_head: 9e6e402f1c0f75b52857fb6caae3908fa739d592
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 99bc4a99a19ad775f025db2727f24655031e7158e73fe1f17e832cb21cfdfd0f
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired (MIC/LVD paired by filename rule)
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/subset_manifest.json
  - num_pairs = 48
  - fingerprint_md5 = 739a181c331f347614090fffe6f4b491
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
OUT_DIR="results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338"
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
- PASS BECAUSE all file and numeric integrity checks are clean (no missing files, no hash mismatches, no NaN/Inf), THEREFORE the diagnostic metrics can be interpreted as valid signals rather than pipeline artifacts.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.000
- k_selected_mean list = [15.901, 15.437, 13.962, 12.734, 11.948]

Decision:
- PASS BECAUSE k_selected_mean decreases monotonically with lambda_c and spearman is ~-1, THEREFORE lambda_c provides reliable compute control.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 11.948
- DT - Random (compute-matched capture mean) = 0.0293 (> 0)

Decision:
- PASS BECAUSE DT still outperforms compute-matched Random at the high-lambda regime, THEREFORE compute control does not collapse DT into random behavior.

### 4.3 GCC-PHAT stability (Required)

From summary/subsample_delay_diagnostics_summary.json, coarse_source="dt":

At lambda_c=3e-4:
- fraction_defined = 1.000
- boundary_hit_rate = 0.000
- psr_p50/p90 = 56.818 / 79.971
- within_clip_tau_mad_ms_p50/p90 = 0.00834 / 0.0300

Decision:
- PASS BECAUSE the refined GCC-PHAT tau is defined for essentially all windows and is highly stable within each clip, THEREFORE the coarse-to-fine (DT coarse -> waveform GCC) pipeline remains stable under compute control.

### 4.4 Phase-slope fit quality and dispersion (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt"):

At lambda_c=3e-4:
- phase_slope_fraction_defined = 0.970
- phase_slope_tau_hat_ms_p50/p90 = -3.478 / 1.975
- phase_slope_r2_p50/p90 = 0.483 / 0.801
- phase_slope_fit_rmse_rad_p50/p90 = 4.365 / 9.364
- abs_tau_agreement_ms_p50/p90 = 4.031 / 9.114
- tau_band_spread_ms_p50/p90 = 10.429 / 22.492

Interpretation:
- Phase-slope fit quality is poor (low R^2 and high RMSE) BECAUSE the cross-spectrum phase is not well-approximated by a single straight line over 300-3000 Hz; THEREFORE the MIC->LDV relation is not consistent with a pure time delay in most windows.
- Large tau_band_spread_ms indicates frequency-dependent group delay (dispersion) BECAUSE the estimated tau differs substantially across sub-bands; THEREFORE phase equalization / calibration is likely required before any geometry-grounded TDoA claim.
- GCC-PHAT can still look stable even when phase-slope is poor BECAUSE GCC-PHAT effectively produces an in-band "best peak" alignment, which can be stable-but-biased when the true phase is non-linear; THIS IMPLIES stability alone is insufficient for correctness.

### 4.5 Guardrail separation (paired vs mispair_shift1; Required)

Compare scale_check_subset paired vs mispair_shift1 (48 pairs), dt coarse:
- psr_p50: paired ~56.9 vs mispair ~7-8 across all lambdas
- within_clip_tau_mad_ms_p50: paired ~0.008 ms vs mispair ~12-18 ms across all lambdas
- phase_slope_fraction_defined: paired ~0.970 vs mispair ~0.14-0.18 across all lambdas
- tau_band_spread_ms_p50: paired ~10 ms vs mispair ~14-15 ms across all lambdas

Decision:
- PASS BECAUSE confidence/stability metrics separate strongly between correct pairing and mispair guardrail, THEREFORE the diagnostics are sensitive to true alignment and not trivially constant.

### 4.6 Readiness decision (Required)

Decision rule at lambda_c=3e-4 (paired, dt coarse):
- READY_FOR_PHASE3 if abs_tau_agreement_ms_p50 <= 0.1 ms AND tau_band_spread_ms_p50 <= 0.2 ms
- Observed: abs_tau_agreement_ms_p50 = 4.031 ms, tau_band_spread_ms_p50 = 10.429 ms

Readiness:
- READY_FOR_CALIBRATION BECAUSE both agreement and band-spread thresholds fail by large margins; THEREFORE the system shows dispersion/non-linear phase that must be compensated before Phase-3 correctness evaluation.

## 5) Physical / Mathematical Analysis (Required)

- Pure delay model: If y(t) ~= x(t - tau), then in frequency domain Y(f) ~= X(f) * exp(-j 2 pi f tau). Therefore the cross-spectrum phase phi(f) = angle(conj(X(f)) * Y(f)) is approximately linear in f with slope -2 pi tau.
- Dispersion signature: If the LDV path contributes frequency-dependent phase (material resonances / all-pass-like distortion), then phi(f) is not linear. This reduces linear-fit quality (lower R^2, higher RMSE) BECAUSE no single slope can explain phi(f) across the band.
- Why GCC-PHAT may remain stable: GCC-PHAT searches for a correlation peak in-band and can yield a consistent peak even when phi(f) is non-linear. Therefore GCC stability does NOT guarantee the underlying system is a pure delay; it can be stable-but-biased.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech paper eval) established DT>Random on speech WAV BECAUSE the DT policy can exploit structured spectral redundancy better than random lag selection.
- Commit 76d63fe / db556db (E4j-Speech STOP/cost calibration) achieved mean compute ~12 at lambda_c=3e-4 WHILE keeping DT>Random; THEREFORE Phase-2 diagnostics at compute~12 are meaningful.
- Commit 9e6e402 (E4l-Speech) showed GCC-PHAT sub-sample refinement is stable on paired speech and fails strongly under mispair; THEREFORE the waveform-domain refinement is a valid alignment diagnostic.
- This E4m functional run extends E4l by adding phase-slope dispersion metrics; results show high GCC stability but poor phase-slope fit quality, WHICH IMPLIES we likely have frequency-dependent phase distortion (dispersion) that requires calibration.

## 7) Extracted Principles for Next Steps (Required)

- If phase_slope_r2 is high and tau_band_spread is small, THEN proceed to Phase-3 correctness evaluation BECAUSE the system behaves like a pure delay.
- If phase_slope_r2 is low OR tau_band_spread is large, THEN prioritize calibration design BECAUSE the LDV transfer path is not a pure delay and a single tau estimate can be biased.
- Always include a mispair guardrail run BECAUSE it verifies diagnostics respond to true pairing rather than incidental signal structure.

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
- Expected outputs exist under results/rtgomp_dispersion_E4m_speech_functional_scale_check_subset_paired_20260126_105338/:
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

- Phase unwrap sensitivity: phase-slope relies on unwrap(angle(C(f))) and can be affected by low-magnitude bins; however the large band-spread and poor global fit are consistent with dispersion signatures.
- No geometry ground truth: these diagnostics show stability/dispersion, but they do not by themselves prove TDoA correctness.
- Compute vs stability: while compute is controllable, correctness may still fail if dispersion dominates; THIS IMPLIES calibration readiness is the right next phase.
