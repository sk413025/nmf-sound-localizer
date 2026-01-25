# Acceptance Report: E4k-Speech -- Delay & Phase-Consistency Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_delay_diag_E4k_speech_full_dataset_paired_20260125_234744/
- Mode: full_dataset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? YES (Spearman=-1.0 on full_dataset).
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? YES (DT-Random>0 at lambda=3e-4).
- Q3: Does a coarse delay proxy (first selected lag) remain frequency-consistent as compute decreases? YES, with mild
  degradation: dt_first_lag_mad_p50 increases from ~1.01 to ~1.18 frames across the lambda grid while k_selected_mean
  drops from ~15.9 to ~12.0.
- Q4: Does the mispair guardrail break delay stability? YES (validated on scale_check_subset; mispair MAD p50 ~23 vs
  paired ~1).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: (not recorded in this report)
- Device(s): cpu (stability)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_delay_diag_E4k_speech_full_dataset_paired_20260125_234744/code_state.json
  - git_head: db556db5f82237298697eea85a099326cf4daad1
  - dirty: true
  - sha256 files:
    - scripts/h_exploration/run_rtgomp_e4h_paper_eval.py: eb2a09bfc5db9c844ab3c1d1af912ec0303b0cde451b2402abaa9749de0e4153
    - scripts/h_exploration/dataset_lag.py: 6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb

### 2.3 Data lineage (Speech-only; Required)

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing mode: paired (MIC->LDV filename replacement)
- Dataset length check (preflight output; same machine):
  - len(dataset) = 416
  - first_pair = boy1_papercup_MIC_001.wav, boy1_papercup_LDV_001.wav
  - last_pair  = boy1_papercup_MIC_xnonoise_320.wav, boy1_papercup_LDV_xnonoise_320.wav
- Subset manifest: results/rtgomp_delay_diag_E4k_speech_full_dataset_paired_20260125_234744/subset_manifest.json
  - num_pairs = 416
  - fingerprint_md5 = 13356fdb74d2acb7e85361a4dfe5c3d2
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (Must be explicit)

- fs = 16000
- hop_length = 160 (10ms per frame)
- n_fft = 2048
- freq band = [freq_min, freq_max] = [300, 3000] Hz
- max_lag = 50 -> M_lags = 101
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]
- Random baseline: without_replacement

## 3) Exact Commands (Required)

Recorded in:
- results/rtgomp_delay_diag_E4k_speech_full_dataset_paired_20260125_234744/run.log

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From summary/compute_matched_summary.json: integrity
- num_missing_files = 0
- num_md5_mismatches = 0
- num_nan_or_inf = 0
- num_capture_out_of_range_total = 0
- num_omp_monotonicity_violations = 0
- num_dt_duplicate_actions_forced_k = 0

Decision:
- PASS because all hard guardrails are satisfied on full_dataset (stable least-squares behavior, capture within [0,1]).

### 4.1 RTG controllability (Compute vs lambda_c; Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean = [15.904, 15.462, 14.029, 12.817, 12.049]

Decision:
- PASS because compute decreases monotonically with increasing lambda_c on full_dataset.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4 (summary/compute_matched_summary.json):
- k_selected_mean = 12.049
- DT - Random (compute-matched capture mean) = +0.024739
- DT/OMP = 0.995399

Decision:
- PASS because DT remains better than Random at the target compute regime.

### 4.3 Delay / phase-consistency diagnostics (Required)

From summary/delay_diagnostics_summary.json:

First-lag dispersion vs compute (DT):
- 1e-5:  dt_first_lag_mad_frames_p50 = 1.006, dt_first_lag_mad_frames_mean = 2.269, k_selected_mean = 15.904
- 3e-5:  dt_first_lag_mad_frames_p50 = 1.057, dt_first_lag_mad_frames_mean = 2.359, k_selected_mean = 15.462
- 1e-4:  dt_first_lag_mad_frames_p50 = 1.116, dt_first_lag_mad_frames_mean = 2.445, k_selected_mean = 14.029
- 2e-4:  dt_first_lag_mad_frames_p50 = 1.161, dt_first_lag_mad_frames_mean = 2.502, k_selected_mean = 12.817
- 3e-4:  dt_first_lag_mad_frames_p50 = 1.176, dt_first_lag_mad_frames_mean = 2.531, k_selected_mean = 12.049

Additional diagnostics:
- dt_stop0_frac_mean = 0.0 across the grid (no immediate STOP collapse).
- dt_vs_omp_first_lag_match_frac_mean ~ 0.477–0.479 across the grid.

Interpretation:
- Delay stability degrades mildly as compute decreases (MAD p50 increases by ~0.17 frames) BECAUSE the policy is
  explicitly conditioned on cost (lambda_c), so it may trade off “best greedy lag” vs “lags that work well under fewer
  future steps”; HOWEVER the dispersion remains low overall, THEREFORE a stable coarse alignment exists on full_dataset.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Guardrail evaluated on the deterministic subset:
- Paired:  results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_20260125_233315/
- Mispair: results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_20260125_233943/

At lambda_c = 3e-4:
- paired dt_first_lag_mad_p50 = 1.006
- mispair dt_first_lag_mad_p50 = 23.409

Decision:
- PASS because mispair has dramatically worse delay stability than paired, validating that the metric detects pairing
  errors.

## 5) Physical / Mathematical Analysis (Required)

- For a near-pure delay between two channels, the cross-spectrum phase is approximately linear with frequency, which
  implies an approximately constant group delay. In the STFT-lag view, this implies that “best lag” should be similar
  across frequency bins, therefore lag dispersion (MAD across frequency) should be small.
- LDV material resonances and dispersion create frequency-dependent phase distortions, which can make the effective
  delay vary with frequency; THEREFORE lag dispersion should increase if the system is strongly dispersive or if the
  channels are mispaired.
- On full_dataset paired speech, DT first-lag dispersion remains low (p50 ~1 frame) while compute decreases, which
  implies there is a strong coarse alignment component that is not destroyed by cost-conditioning at the ~12-step
  regime. This is a necessary (but not sufficient) condition to proceed toward sample-level TDoA.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

Reference Results commits:
- 76d63fe (E4j-Speech STOP calibration, warm-start stepwise-eval)
- 0334802 (E4j-Speech STOP micro-sweep, stop_weight tuning)
- db556db (E4j-Speech full_dataset eval, mean compute ~12 with DT>Random)

Patterns and causal interpretation:
- Pattern recognition: Across 76d63fe -> 0334802 -> db556db, RTG controllability became monotonic (Spearman ~ -1)
  BECAUSE training was aligned to evaluator semantics (stepwise K=1) and STOP was explicitly calibrated via stop_weight.
- Success factors: DT maintained DT>Random at high lambda (db556db) BECAUSE the policy learned structured lag choices
  that concentrate capture early, therefore reducing compute did not fully randomize the selections.
- Failure modes avoided: Earlier “RTG controllability collapse” (historical issue) occurs when training/eval differ in
  how history is presented (hidden-state handling), DUE TO the STOP head learning the wrong compute semantics.
- Phase-2a extension (this run): Even under reduced compute, first-lag dispersion remains low BECAUSE the dominant
  alignment component between correctly paired MIC/LDV is shared across frequency; THEREFORE the system is a plausible
  candidate for downstream delay refinement rather than being purely frequency-chaotic.

## 7) Extracted Principles for Next Steps (Required)

- Design principle: Always validate pairing and alignment with a mispair guardrail BEFORE claiming delay/phase progress,
  BECAUSE mispair should explode lag dispersion if the metric is sensitive and the data is not accidentally leaking.
- Hypothesis formation: GIVEN that lag dispersion remains low at k_selected_mean ~12, predict that sub-sample delay
  refinement (GCC-PHAT / phase-slope) will be feasible on paired data.
- Resource allocation: Invest next in sub-sample delay estimation and a TDoA objective, rather than re-tuning STOP,
  BECAUSE the coarse alignment signal is already stable in the intended compute regime.
- Risk mitigation: Keep the 5-point lambda grid in future runs, because it exposes whether compute control trades off
  against delay stability (dispersion trend).

## 8) Reproduction Instructions (Required)

1) Environment:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

2) Execution:
- See results/.../run.log for the exact command.

3) Verification:
- Required outputs exist:
  - subset_manifest.json (wav-only)
  - summary/compute_matched_summary.json
  - summary/forced_k_summary.json
  - summary/rtg_controllability_summary.json
  - delay_diagnostics.jsonl
  - summary/delay_diagnostics_summary.json
  - integrity_diagnostics.jsonl
  - run.log
  - code_state.json
  - ACCEPTANCE_REPORT.md
- Key numeric checks:
  - spearman(lambda_c, k_selected_mean) <= -0.6 (observed: -1.0)
  - DT - Random > 0 at lambda_c=3e-4 (observed: +0.024739)
  - dt_first_lag_mad_p50 remains low and changes mildly across lambda (observed: 1.006 -> 1.176)

## 9) Failures / Limitations (Required even if PASS)

- Lag is measured in STFT frames (10ms), which is far too coarse to be a final TDoA. This experiment only validates a
  coarse alignment proxy.
- We do not yet compute a sub-sample delay estimator (GCC-PHAT / phase-slope) nor compare against geometry-ground truth.
  Therefore we cannot claim “TDoA solved” yet.

