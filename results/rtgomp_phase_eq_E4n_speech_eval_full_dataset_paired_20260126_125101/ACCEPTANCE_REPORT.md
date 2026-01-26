# Acceptance Report: E4n-Speech -- Phase Equalization Calibration + E4m Re-validation

## 1) Executive Summary

- Run: `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/`
- Run type: EVAL
- Mode: full_dataset
- Pairing mode: paired
- Outcome: FAIL
- Readiness decision (EVAL full_dataset only): READY_FOR_CALIBRATION

Primary questions answered:
- Q1 (FIT): Did we estimate a valid, reproducible phase equalizer? PASS (see `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/`).
- Q2 (EVAL): Does compute controllability still hold after applying phase_eq? PASS (Spearman = -1.0).
- Q3 (EVAL): Does DT remain non-degenerate at target compute (~12) after applying phase_eq? PASS (DT - Random > 0 at lambda_c=3e-4).
- Q4 (EVAL): Do dispersion diagnostics improve vs E4m baseline AND remain improved as compute decreases? FAIL.
- Q5 (EVAL): Does mispair_shift1 remain a strong guardrail failure after applying phase_eq? PASS (see functional paired vs mispair runs).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- Device(s): `cpu` (chosen to avoid potential GPU/CPU mixed-op stalls during long evaluation)
- MPLCONFIGDIR: `/tmp/mpl`

### 2.2 Code provenance (Required)

- code_state.json: `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/code_state.json`
  - git_head: `18a82aac2d8a5c391bd31b830d9754cffecec6fa`
  - dirty: `true`
  - sha256 files:
    - `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`: `1a1b0c4ccb4eed0cf0d3140b3beb62cff633d9e02400f7aac49245964685d911`
    - `scripts/h_exploration/fit_phase_eq_e4n_speech.py`: `5125bed0dbe84bf1a88649d7550423f00c7638207d0dd0a599e452a665d1e85b`
    - `scripts/h_exploration/dataset_lag.py`: `6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb`

### 2.3 Data lineage (Speech-only; Required)

- mic_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Pairing mode: `paired` (MIC_i aligned with LDV_i in dataset order)
- Dataset length check (preflight output):
  - len(dataset) = 416 (expected 416)
  - first_pair = `.../boy1_papercup_MIC_001.wav`, `.../boy1_papercup_LDV_001.wav`
  - last_pair  = `.../boy1_papercup_MIC_xnonoise_320.wav`, `.../boy1_papercup_LDV_xnonoise_320.wav`
- Subset manifest: `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/subset_manifest.json`
  - num_pairs = 416
  - fingerprint_md5 = `13356fdb74d2acb7e85361a4dfe5c3d2`
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (Must be explicit)

Common:
- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- search_radius_frames = 2

EVAL-only:
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- subsample_method = gcc_phat,phase_slope
- phase_slope_min_bins = 64
- phase_slope_subbands_hz = [[300,900],[900,1800],[1800,3000]]
- checkpoint: `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Calibration application:
- apply_phase_eq = 1
- phase_eq_path = `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/phase_eq/phase_eq.npz`
- phase_eq_nfft_fit = 32768

### 2.5 Definitions (Must be explicit)

- phase_eq: a per-frequency unit-magnitude complex correction applied to LDV
- phase_eq_stft: length 1025, applied to ldv_stft
- phase_eq_rfft: length 16385 (for nfft_fit=32768), applied inside subsample GCC/phase-slope FFT path
- Pure delay: phi(f) ~= -2 pi f tau + b
- Dispersion: phi(f) non-linear; sub-band taus disagree

## 3) Exact Commands (Required)

EVAL full_dataset (this run):

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
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
  --apply_phase_eq 1 --phase_eq_path results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/phase_eq/phase_eq.npz \
  |& tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

### 4.0 Hard guardrails (Must pass)

For this run:
- wav-only manifest: PASS (no `.npy` in `subset_manifest.json`) BECAUSE `--require_wav_only 1` hard-fails on non-wav.
- fs mismatch encountered: NO (fs hard-check in WAV loader).
- any NaN/Inf in outputs: NO (integrity `num_nan_or_inf=0`).

EVAL artifacts present:
- `integrity_diagnostics.jsonl`: YES
- `delay_diagnostics.jsonl`: YES
- `subsample_delay_diagnostics.jsonl`: YES
- `summary/*.json`: YES

Decision:
- PASS for guardrails BECAUSE all required artifacts exist and integrity counters show no missing files, md5 mismatches, or NaN/Inf.

### 4.2 EVAL stage: RTG controllability

From `summary/rtg_controllability_summary.json`:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean = [15.9040, 15.4617, 14.0290, 12.8170, 12.0486]

Decision:
- PASS BECAUSE k_selected_mean decreases monotonically as lambda_c increases (Spearman=-1.0), THEREFORE lambda_c remains a reliable compute knob after applying phase_eq.

### 4.3 EVAL stage: non-degeneracy at target compute

At lambda_c = 3e-4 (from `summary/compute_matched_summary.json`):
- k_selected_mean = 12.0486
- DT - Random (compute-matched capture mean) = +0.02483 (must be > 0 for PASS)

Decision:
- PASS BECAUSE DT remains strictly better than compute-matched Random at the target compute point, THEREFORE compute control does not collapse DT into a degenerate policy.

### 4.4 EVAL stage: dispersion improvement vs baseline (EVAL full_dataset paired only)

Baseline reference (required):
- E4m baseline commit hash: `18a82aa`
- Baseline run directory: `results/rtgomp_dispersion_E4m_speech_full_dataset_paired_20260126_110528/`

Report baseline (lambda_c=3e-4, dt coarse; from baseline `summary/subsample_delay_diagnostics_summary.json`):
- phase_slope_r2_p50 = 0.50062
- phase_slope_fit_rmse_rad_p50 = 4.77768
- tau_band_spread_ms_p50 = 9.69709
- abs_tau_agreement_ms_p50 = 4.66087

Report calibrated (this run; lambda_c=3e-4, dt coarse; from this run `summary/subsample_delay_diagnostics_summary.json`):
- phase_slope_r2_p50 = 0.29606
- phase_slope_fit_rmse_rad_p50 = 6.11119
- tau_band_spread_ms_p50 = 9.68219
- abs_tau_agreement_ms_p50 = 3.73562

Calibrated run table (dt coarse):

| lambda_c | phase_slope_r2_p50 | rmse_rad_p50 | tau_band_spread_ms_p50 | abs_tau_agreement_ms_p50 |
| ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 0.29439 | 6.08438 | 9.78966 | 3.71458 |
| 3e-5 | 0.29395 | 6.11146 | 9.75617 | 3.71447 |
| 1e-4 | 0.29596 | 6.07927 | 9.78598 | 3.71043 |
| 2e-4 | 0.29521 | 6.08559 | 9.73643 | 3.70374 |
| 3e-4 | 0.29606 | 6.11119 | 9.68219 | 3.73562 |

Decision:
- FAIL BECAUSE phase_slope_r2_p50 decreases materially vs baseline (0.50 -> 0.30) AND phase_slope_fit_rmse_rad_p50 increases (4.78 -> 6.11), THEREFORE the phase_eq calibration did not improve linear-phase consistency.
- Additionally, tau_band_spread_ms_p50 remains ~9.7ms (>> 2.0ms target), THEREFORE the core dispersion symptom persists.
- While abs_tau_agreement_ms_p50 improves (~4.66ms -> ~3.74ms), it remains > 1.0ms and does not satisfy the improvement targets.

### 4.5 Guardrail separation after calibration (EVAL functional runs, 48 pairs)

Functional runs (after applying phase_eq):
- paired: `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_paired_20260126_123812/`
- mispair_shift1: `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_mispair_shift1_20260126_124541/`

Result:
- 5 / 5 lambdas satisfy all three guardrail inequalities:
  - psr_p50(paired) > psr_p50(mispair)
  - within_clip_tau_mad_ms_p50(paired) < within_clip_tau_mad_ms_p50(mispair)
  - phase_slope_fraction_defined(paired) > phase_slope_fraction_defined(mispair)

Decision:
- PASS BECAUSE calibration preserves strong paired-vs-mispair separation, THEREFORE phase_eq is not masking incorrect pairing (no “false good”).

### 4.6 Readiness decision (EVAL full_dataset paired only)

At lambda_c=3e-4 (dt coarse):
- abs_tau_agreement_ms_p50 = 3.73562 ms
- tau_band_spread_ms_p50 = 9.68219 ms

Decision:
- READY_FOR_CALIBRATION (not Phase-3) BECAUSE both metrics are far above the Phase-3 thresholds (0.1ms and 0.2ms), THEREFORE a single physical delay tau is not yet reliable after this calibration attempt.

## 5) Physical / Mathematical Analysis (Required)

First principles:
- A pure time delay y(t)=x(t-tau) implies Y(f)=X(f)·exp(-j2pi f tau), so the cross-spectrum C(f)=conj(X(f))·Y(f) has phase phi(f) that is linear in f. Therefore phase-slope regression should fit well (high r2, low RMSE) and sub-band delays should agree (small band spread).
- Dispersion / frequency-dependent phase means Y(f)=X(f)·exp(-j2pi f tau)·exp(+j phi_residual(f)), where phi_residual(f) bends phi(f) away from a straight line. Therefore phase-slope fit quality drops and different sub-bands imply different taus.

Why phase equalization should help (in theory):
- If we can estimate G(f)=exp(-j phi_hat_residual(f)) and apply it to LDV (Y_eq(f)=Y(f)·G(f)), THEN the residual phase is removed and phi_eq(f) becomes closer to a straight line. Therefore phase_slope_r2 should increase and tau_band_spread should decrease.

What we observed and what it implies:
- We observed strong GCC-PHAT peaks (psr_p50 increases substantially) but phase-slope r2 decreases and band-spread is essentially unchanged. This implies the estimated phase_eq does not match the per-window residual phase that drives dispersion, BECAUSE the residual is likely not a single fixed all-pass curve shared across windows/clips (or the estimator is not aligned with the diagnostic definition).
- The guardrail remains strong, which implies the calibration is not “cheating” by making unrelated pairs appear aligned; therefore the failure is not due to overfitting the guardrail but due to insufficient physical correction.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Referenced Results commits:
- `6af8a63` (E4h-Speech paper eval): established DT vs OMP vs Random evaluation pipeline on speech WAV.
- `db556db` (E4j-Speech full_dataset eval): achieved compute control with STOP calibration at mean compute ~12 while keeping DT > Random.
- `9e6e402` (E4l-Speech): validated GCC-PHAT sub-sample refinement stability and strong mispair guardrail failure.
- `18a82aa` (E4m-Speech): showed dispersion symptoms (sub-band disagreement and modest phase-slope fit) despite stable GCC.

Pattern recognition:
- Across E4j/E4l/E4m, compute controllability and GCC stability are consistently achievable, BECAUSE they mainly depend on coarse lag guidance and robust correlation structure rather than detailed phase linearity.
- Dispersion metrics remain poor in E4m and do not improve in E4n, WHICH IMPLIES the dominant error is not just a fixed global phase residual removable by a single averaged equalizer; it is likely time-varying, clip-dependent, or multi-path.

## 7) Extracted Principles for Next Steps (Required)

- Calibration principle: If dispersion is driven by clip-dependent residual phase, THEN a single global phase_eq is insufficient; therefore next experiments should estimate phase correction per clip (or per condition) and measure whether dispersion improves within-clip before attempting a global model.
- Diagnostics principle: If phase-slope is unstable due to unwrap errors, THEN derotating by a coarse delay (GCC/DT) before unwrapping should improve fit robustness; therefore consider a “derotated phase-slope” diagnostic to separate unwrap artifacts from true dispersion.
- Progress gating: Only proceed to Phase-3 (TDoA-like correctness claims) if abs_tau_agreement and band_spread reach sub-ms levels; otherwise Phase-3 would be invalid BECAUSE the underlying single-tau assumption is violated.

## 8) Reproduction Instructions (Required)

1) Environment:

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Fit phase_eq (full_dataset; paired):

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/fit_phase_eq_e4n_speech.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --out_dir "$OUT_DIR" \
  --mode full_dataset \
  --pairing_mode paired --require_wav_only 1 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --search_radius_frames 2 --seed 0 \
  |& tee -a "$OUT_DIR/run.log"
```

Expected fit outputs:
- `phase_eq/phase_eq.npz`
- `phase_eq/phase_eq_fit_summary.json`

3) Re-validation (this run; EVAL full_dataset paired, apply phase_eq):
- Run the command in Section 3.

4) Verification:
- `summary/rtg_controllability_summary.json` exists and Spearman is <= -0.9.
- `summary/compute_matched_summary.json` has `dt_minus_random_mean > 0` at lambda_c=3e-4.
- Dispersion improvement targets are currently NOT met (this is a negative result to reproduce).

## 9) Failures / Limitations (Required even if PASS)

- This calibration approach estimates a single global phase_eq across all windows/clips. It cannot correct clip-dependent phase behavior, multi-path interference, or time-varying transfer functions; therefore it may fail even when GCC remains stable.
- The phase-slope diagnostic can be sensitive to phase unwrapping; therefore poor r2/RMSE may reflect unwrap artifacts as well as true dispersion.
- Compute reduction does not materially change dispersion metrics in this setup (values are nearly constant across lambdas), WHICH IMPLIES dispersion is not currently bottlenecked by compute in the selection policy but by underlying phase physics / estimation.

