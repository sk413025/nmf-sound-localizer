# Acceptance Report: E4l-Speech -- Sub-sample Delay Refinement Diagnostics

## 1) Executive Summary

- Run: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/
- Mode: scale_check_subset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1: Is compute controllable via lambda_c on speech WAV? (RTG controllability) -> YES
- Q2: Does DT remain non-degenerate at the target compute regime (~12)? (DT>Random at high lambda) -> YES
- Q3: Do sub-sample delay estimators (GCC-PHAT / phase-slope) produce stable tau estimates on paired data? -> YES (GCC-PHAT)
- Q4: Does mispair_shift1 degrade tau stability and confidence (guardrail separation)? -> YES (see section 4.4)

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: trl-training
- Python: Python 3.11.13
- Device(s): cpu (chosen for stability and determinism)
- MPLCONFIGDIR: /tmp/mpl

### 2.2 Code provenance (Required)

- code_state.json: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/code_state.json
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
- Subset manifest: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/subset_manifest.json
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

OUT_DIR="results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720"
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
- PASS BECAUSE all integrity counters are zero and required artifacts were written; THEREFORE the evaluator is correct enough to trust downstream diagnostics.

### 4.1 RTG controllability (Required)

From summary/rtg_controllability_summary.json:
- spearman(lambda_c, k_selected_mean) = -1.0
- k_selected_mean list = [15.901, 15.437, 13.962, 12.734, 11.948]

Decision:
- PASS BECAUSE k_selected decreases monotonically with lambda_c; THEREFORE compute is controllable on speech.

### 4.2 Non-degeneracy at target compute (Required)

At lambda_c = 3e-4:
- k_selected_mean = 11.948
- DT - Random (compute-matched capture mean) = 0.029256 (> 0)

Decision:
- PASS BECAUSE DT remains better than compute-matched Random at the target compute regime; THEREFORE the policy stays non-degenerate.

### 4.3 Sub-sample delay diagnostics (Required)

From summary/subsample_delay_diagnostics_summary.json (coarse_source="dt") at lambda_c=3e-4:
- fraction_defined = 1.0
- boundary_hit_rate = 0.0

Small table across lambdas (dt coarse):

| lambda_c | k_selected_mean | psr_p50 | within_clip_tau_mad_ms_p50 | boundary_hit_rate |
|---:|---:|---:|---:|---:|
| 1e-5 | 15.901 | 56.896 | 0.008344 | 0.000 |
| 3e-5 | 15.437 | 56.896 | 0.008344 | 0.000 |
| 1e-4 | 13.962 | 56.896 | 0.008344 | 0.000 |
| 2e-4 | 12.734 | 56.896 | 0.008344 | 0.000 |
| 3e-4 | 11.948 | 56.818 | 0.008344 | 0.000 |

Interpretation:
- Tau stability stays extremely tight (sub-millisecond MAD) while compute decreases BECAUSE coarse lag guidance stays consistent and the in-band phase relationship remains coherent enough for GCC-PHAT.

### 4.4 Guardrail separation (paired vs mispair_shift1; Required)

Guardrail run:
- mispair_shift1: results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_mispair_shift1_20260126_022335/

Separation checks on scale_check_subset (48 pairs):
- For 5/5 lambdas:
  - psr_p50(paired, dt) > psr_p50(mispair, dt)
  - within_clip_tau_mad_ms_p50(paired, dt) < within_clip_tau_mad_ms_p50(mispair, dt)

Example at lambda_c = 3e-4 (dt coarse):
- psr_p50 paired = 56.818 vs mispair = 7.381 (paired > mispair)
- within_clip_tau_mad_ms_p50 paired = 0.008 ms vs mispair = 15.567 ms (paired < mispair)

Decision:
- PASS BECAUSE mispair_shift1 destroys phase/delay coherence, collapsing PSR and inflating within-clip tau dispersion; THEREFORE these diagnostics are sensitive to correct pairing.

## 5) Physical / Mathematical Analysis (Required)

- GCC-PHAT peak indicates delay BECAUSE a true relative delay induces consistent phase differences across frequency, which PHAT emphasizes by normalizing magnitude.
- In paired MIC↔LDV, the signals share the same speech content, so even with spectral coloration the phase remains sufficiently coherent within [300,3000] Hz, yielding high PSR and low tau MAD.
- In mispair_shift1, MIC and LDV come from different utterances, so the cross-spectrum phase is effectively incoherent; THEREFORE the correlation function has no dominant peak (low PSR) and the estimated tau fluctuates across windows (high within-clip MAD).
- This implies Phase-2b can validate “delay-estimability under compute control” without requiring geometry ground truth yet.

## 6) Cross-Experiment Analysis (Required; reference >= 3 commits)

- Commit 6af8a63 (E4h-Speech paper eval) validated DT vs OMP vs Random evaluation plumbing on speech WAV.
- Commit db556db (E4j-Speech full_dataset) calibrated STOP/cost to hit mean compute ~12 at lambda_c=3e-4 while keeping DT>Random.
- Commit 16e66d0 (E4k-Speech) showed coarse delay stability on paired speech and strong failure under mispair guardrail.

This run adds that the “coarse delay stability” from E4k translates into sample-level GCC-PHAT stability BECAUSE the coarse lag can constrain the search window; THEREFORE compute control does not only change tail behavior but still yields usable fine delay estimates.

## 7) Extracted Principles for Next Steps (Required)

- If guardrail separation remains strong and within-clip tau MAD stays low at lambda_c=3e-4, THEN a Phase-3 TDoA-like experiment is justified (e.g., geometry-grounded evaluation or controlled known delays).
- If future full_dataset results show MAD increasing with lambda, THEN we should recalibrate STOP/cost or refine the coarse-to-fine strategy (e.g., widen search radius with explicit acceptance thresholds).

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
- Confirm these outputs exist under results/rtgomp_subsample_delay_E4l_speech_functional_scalecheck_paired_20260126_021720/:
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

- Guardrail uses mispair_shift1 (cyclic shift); it is a strong negative control but not a physically realistic failure mode.
- GCC-PHAT assumes a dominant delay; multi-path or strong dispersion can violate this and require richer models (e.g., phase-slope residual diagnostics).
- We still lack geometry-ground truth; stable tau alone does not prove correct physical TDoA without controlled distance/position calibration.
