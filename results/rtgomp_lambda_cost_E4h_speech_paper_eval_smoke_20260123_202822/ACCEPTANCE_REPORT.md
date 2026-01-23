# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/`
- Mode: `smoke`
- Outcome: `PASS`
- Primary paper outputs:
  - Compute-matched DT vs OMP vs Random (WITH STOP; matched by `k_selected`)
  - Forced-K DT vs OMP vs Random (NO STOP)
- Dataset domain statement (required):
  - Speech WAV only (no `.npy` in manifest): `YES`

## 2) Setup (Required)

### 2.1 Environment

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- Device(s): `cpu` (chosen for stability)
- `MPLCONFIGDIR`: `/tmp/mpl`

### 2.2 Code provenance (Required)

- `code_state.json`: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/code_state.json`
  - `git_head`: `430d0a51da85675b67bb432b8406145f3ef24535`
  - `dirty`: `true`
  - SHA256:
    - `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`: `0442f1b160c46253e6b5e96a6109be235fff9dd3ab3a3a8baf7dc61ee1d8b058`
    - `scripts/h_exploration/dataset_lag.py`: `6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb`

### 2.3 Data lineage (Speech-only; Required)

- mic_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Dataset pairing rule:
  - Pair by filename replacement: `MIC` -> `LDV`
- Dataset length check (preflight):
  - `len(dataset) = 416`
  - first_pair: `.../boy1_papercup_MIC_001.wav`, `.../boy1_papercup_LDV_001.wav`
  - last_pair: `.../boy1_papercup_MIC_xnonoise_320.wav`, `.../boy1_papercup_LDV_xnonoise_320.wav`
- Subset manifest: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/subset_manifest.json`
  - `num_pairs = 1`
  - `fingerprint_md5 = 9b169d8cb234f30431fc7178d5aafb33`
  - Domain validation:
    - all paths end with `.wav`: `YES`
    - any `.npy` paths: `NO`

### 2.4 Fixed parameters (Must be explicit)

- hop_length = `160`
- fs = `16000`
- n_fft = `2048`
- freq band = `[300, 3000]` Hz
- max_lag = `50` -> `M_lags = 101`
- Tw = `32`
- max_k = `16`
- gain = `100.0`
- rtg_dim = `2`
- use_stop_action = `true`
- lambda_c_values = `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`

### 2.5 Baselines (Must be explicit)

- Random baseline:
  - random_trials = `3`
  - random_sampling = `without_replacement` (paper baseline)

### 2.6 Step/compute definitions (Must be explicit)

- `k_selected`: number of selected lag atoms (number of LS solves; compute-aligned)
- `steps_decision`: number of decision steps including STOP (policy-step-aligned)

## 3) Exact Commands (Required)

```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth   --out_dir "$OUT_DIR"   --mode smoke --num_pairs 1   --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000   --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --random_trials 3 --random_sampling without_replacement --seed 0   --write_per_sample 1 --device cpu   2>&1 | tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/summary/compute_matched_summary.json`:

- num_samples_total = `1730`
- num_samples_used = `1730`
- num_missing_files = `0`
- num_md5_mismatches = `0`
- num_nan_or_inf = `0`
- num_capture_out_of_range_total = `0` (PASS)
- num_capture_out_of_range_dt = `0`
- num_capture_out_of_range_omp = `0`
- num_capture_out_of_range_random = `0`
- capture_min/max per method:
  - DT: min `0.1306744814`, max `0.9999990463`
  - OMP: min `0.1502072811`, max `0.9999997616`
  - Random: min `0.0`, max `0.9999989867`
- Worst violation summary: none
- num_omp_monotonicity_violations = `0`
- num_dt_duplicate_actions_forced_k = `0`
- Random duplicate rate (paper baseline): `0.0`

Decision:
- PASS BECAUSE all hard guardrails are satisfied and capture remains within [0,1] for DT/OMP/Random.

### 4.1 RTG controllability (Free rollout; Required)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/summary/rtg_controllability_summary.json`:

- spearman(lambda_c, k_selected_mean) = `-1.0` (PASS)
- k_selected_range = `0.7150289` (non-degenerate)
- Distributional metrics:
  - P(k_selected < max_k) at min/max lambda: `0.0057803` -> `0.1710983`
  - Quantiles (k_selected): p50 ~ `16`, p90 `16`, p99 `16`

Decision:
- PASS BECAUSE increasing lambda_c increases early stopping probability (tail mass below max_k), even though the median is saturated at max_k.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP; Required)

At lambda_c = `1e-4` (min penalty):
- DT_capture_mean = `0.9987670`
- OMP_capture_mean = `0.9996049`
- Random_capture_mean = `0.9974772`
- DT/OMP = `0.9991618`
- DT - Random = `+0.0012898`
- k_selected_mean = `15.9532`
- steps_decision_mean = `15.9590`

Decision:
- PASS BECAUSE DT beats Random at matched compute (DT - Random > 0) at low penalty.

### 4.3 Forced-K DT vs OMP vs Random (NO STOP; Required)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822/summary/forced_k_summary.json`:

- K=1: DT `0.6878491`, OMP `0.6969550`, Random `0.1703213`, DT/OMP `0.9869346`, DT-Random `0.5175278`
- K=2: DT `0.7747790`, OMP `0.8479912`, Random `0.3242775`, DT/OMP `0.9136639`, DT-Random `0.4505015`
- K=4: DT `0.8662367`, OMP `0.9531893`, Random `0.5847199`, DT/OMP `0.9087772`, DT-Random `0.2815168`
- K=8: DT `0.9719518`, OMP `0.9940383`, Random `0.9133876`, DT/OMP `0.9777810`, DT-Random `0.0585641`
- K=16: DT `0.9994325`, OMP `0.9998508`, Random `0.9985687`, DT/OMP `0.9995816`, DT-Random `0.0008639`

## 5) Interpretation (Required; causal language)

- RTG controllability is visible primarily in the stop-rate tail BECAUSE most samples still saturate at max_k=16, THEREFORE mean k_selected moves modestly while P(k_selected < max_k) increases strongly with lambda.
- DT beats Random under compute-matched evaluation at low penalty BECAUSE DT selects informative lags aligned with residual structure rather than uniform random picks, THEREFORE DT-Random is positive.
- DT is close to OMP across forced-K and compute-matched tables BECAUSE the learned policy approximates greedy pursuit behavior on this in-domain speech sample.

## 6) Failures (Required even if PASS)

- No hard failures observed in this smoke run.
- Limitation: quantiles are saturated at 16 BECAUSE max_k=16 caps the distribution, THEREFORE distributional metrics (stop-rate/histograms) are required beyond mean steps.

## 7) Reproduction Instructions (Required)

```bash
# 1) Environment
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

# 2) Execution
OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_20260123_202822"
conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth   --out_dir "$OUT_DIR"   --mode smoke --num_pairs 1   --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000   --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --random_trials 3 --random_sampling without_replacement --seed 0   --write_per_sample 1 --device cpu

# 3) Verification
# Expected:
# - summary/compute_matched_summary.json exists
# - integrity.num_capture_out_of_range_total == 0
# - summary/rtg_controllability_summary.json spearman_lambda_k_selected ~= -1.0
```

## 8) Next Steps (Required)

- Run the functional positive-path scale_check_subset (48 pairs) on speech to confirm stability beyond smoke.
- Run the guardrail diagnostic with random_sampling=with_replacement to validate duplicate-rate reporting and expected integrity failure behavior.
- Only then run full_dataset on speech for paper numbers.
