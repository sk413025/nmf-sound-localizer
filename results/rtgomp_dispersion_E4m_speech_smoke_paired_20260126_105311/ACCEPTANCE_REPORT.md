# Acceptance Report: E4m-Speech -- Dispersion Diagnostics + Calibration Readiness

## 1) Executive Summary

- Run: results/rtgomp_dispersion_E4m_speech_smoke_paired_20260126_105311/
- Mode: smoke
- Pairing mode: paired
- Outcome: PASS
- Readiness decision: NOT_READY (smoke run)

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
- Device(s): cpu
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_dispersion_E4m_speech_smoke_paired_20260126_105311/code_state.json
  - git_head: 9e6e402f1c0f75b52857fb6caae3908fa739d592
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: 99bc4a99a19ad775f025db2727f24655031e7158e73fe1f17e832cb21cfdfd0f
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired
- Dataset length check (preflight output):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_dispersion_E4m_speech_smoke_paired_20260126_105311/subset_manifest.json
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
OUT_DIR="results/rtgomp_dispersion_E4m_speech_smoke_paired_20260126_105311"
LOCKDIR="$OUT_DIR/.lock"

mkdir -p "$OUT_DIR/summary"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
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
- PASS BECAUSE all hard guardrails pass, THEREFORE the smoke run validates basic end-to-end execution.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.000
- k_selected_mean list = [15.900, 15.429, 13.999, 12.747, 11.976]

Decision:
- PASS BECAUSE compute decreases monotonically with lambda_c even on smoke, THEREFORE the compute knob is wired correctly.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 11.976
- DT - Random (compute-matched capture mean) = 0.0348 (> 0)

Decision:
- PASS BECAUSE DT>Random at high lambda even on smoke, THEREFORE the evaluation pipeline is not obviously degenerate.

### 4.3 GCC-PHAT stability (Required)

At lambda_c=3e-4 (dt coarse):
- fraction_defined = 1.000
- psr_p50 = 59.824
- within_clip_tau_mad_ms_p50 = 0.0101

Decision:
- PASS BECAUSE the refinement step executes and yields defined outputs; HOWEVER this is not statistically meaningful on 1 pair.

### 4.4 Phase-slope fit quality and dispersion (Required)

At lambda_c=3e-4 (dt coarse):
- phase_slope_fraction_defined = 0.800
- phase_slope_r2_p50 = 0.512
- phase_slope_fit_rmse_rad_p50 = 2.629

Interpretation:
- The phase-slope metrics are recorded correctly; HOWEVER the smoke run is too small to decide readiness.

### 4.5 Guardrail separation (paired vs mispair_shift1; Required)

- Not evaluated in smoke.

### 4.6 Readiness decision (Required)

- NOT_READY BECAUSE smoke is a 1-pair sanity run; readiness must be decided on paired full_dataset.

## 5) Physical / Mathematical Analysis (Required)

- Pure delay implies linear cross-spectrum phase phi(f) ~= -2 pi f tau; dispersion implies non-linear phi(f).
- This smoke run only checks that the estimators execute and write outputs; it does not support strong physical conclusions.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech) established the baseline paper eval harness on speech WAV.
- Commit 76d63fe/db556db (E4j-Speech) calibrated compute to ~12 at lambda_c=3e-4.
- Commit 9e6e402 (E4l-Speech) validated GCC-PHAT refinement stability and guardrail failure.
- This smoke run confirms the E4m extensions (phase-slope + dispersion fields) execute end-to-end.

## 7) Extracted Principles for Next Steps (Required)

- Always run smoke first BECAUSE it catches wiring errors cheaply.
- Use full_dataset + guardrail comparison for conclusions BECAUSE dispersion diagnostics require enough windows/pairs.

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
- Expected outputs exist under results/rtgomp_dispersion_E4m_speech_smoke_paired_20260126_105311/:
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

- Smoke (N=1) is not representative; interpret metrics only as execution sanity.
