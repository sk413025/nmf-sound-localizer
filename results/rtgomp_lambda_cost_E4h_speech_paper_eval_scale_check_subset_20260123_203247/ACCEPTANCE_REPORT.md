# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/`
- Mode: `scale_check_subset`
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

- `code_state.json`: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/code_state.json`
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
- Subset manifest: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/subset_manifest.json`
  - `num_pairs = 48`
  - `fingerprint_md5 = 739a181c331f347614090fffe6f4b491`
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
OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth   --out_dir "$OUT_DIR"   --mode scale_check_subset --num_pairs 48   --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000   --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --random_trials 3 --random_sampling without_replacement --seed 0   --write_per_sample 1 --device cpu   2>&1 | tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/summary/compute_matched_summary.json`:

- num_samples_total = `80964`
- num_samples_used = `80964`
- num_missing_files = `0`
- num_md5_mismatches = `0`
- num_nan_or_inf = `0`
- num_capture_out_of_range_total = `0` (PASS)
- num_capture_out_of_range_dt = `0`
- num_capture_out_of_range_omp = `0`
- num_capture_out_of_range_random = `0`
- capture_min/max per method:
  - DT: min `0.0196377635`, max `0.9999999404`
  - OMP: min `0.0234571695`, max `0.9999999404`
  - Random: min `0.0`, max `0.9999997616`
- Worst violation summary: none
- num_omp_monotonicity_violations = `0`
- num_dt_duplicate_actions_forced_k = `0`
- Random duplicate rate (paper baseline): `0.0`

Decision:
- PASS BECAUSE all hard guardrails are satisfied and capture remains within [0,1] for DT/OMP/Random.

### 4.1 RTG controllability (Free rollout; Required)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/summary/rtg_controllability_summary.json`:

- spearman(lambda_c, k_selected_mean) = `-1.0` (PASS)
- k_selected_range = `0.7816066` (non-degenerate)
- Distributional metrics:
  - P(k_selected < max_k) at min/max lambda: `0.0047799` -> `0.1803147`
  - Quantiles (k_selected): p50 ~ `16`, p90 `16`, p99 `16`

Decision:
- PASS BECAUSE increasing lambda_c increases early stopping probability (tail mass below max_k), even though the median is saturated at max_k.

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP; Required)

At lambda_c = `1e-4` (min penalty):
- DT_capture_mean = `0.9985810`
- OMP_capture_mean = `0.9994663`
- Random_capture_mean = `0.9970052`
- DT/OMP = `0.9991143`
- DT - Random = `+0.0015758`
- k_selected_mean = `15.9567`
- steps_decision_mean = `15.9615`

(For context) At lambda_c = `1e-2` (max penalty):
- DT_capture_mean = `0.9964365`
- OMP_capture_mean = `0.9986465`
- Random_capture_mean = `0.9849140`
- DT/OMP = `0.9977870`
- DT - Random = `+0.0115225`
- k_selected_mean = `15.1751`

Decision:
- PASS BECAUSE DT beats Random at matched compute (DT - Random > 0) at low penalty.

### 4.3 Forced-K DT vs OMP vs Random (NO STOP; Required)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/summary/forced_k_summary.json`:

- K=1: DT `0.6898921`, OMP `0.7107355`, Random `0.1743018`, DT/OMP `0.9706734`, DT-Random `0.5155902`
- K=2: DT `0.7730404`, OMP `0.8542900`, Random `0.3321865`, DT/OMP `0.9048922`, DT-Random `0.4408539`
- K=4: DT `0.8624840`, OMP `0.9551194`, Random `0.5982607`, DT/OMP `0.9030118`, DT-Random `0.2642234`
- K=8: DT `0.9712750`, OMP `0.9942327`, Random `0.9201977`, DT/OMP `0.9769091`, DT-Random `0.0510773`
- K=16: DT `0.9993267`, OMP `0.9998521`, Random `0.9984764`, DT/OMP `0.9994745`, DT-Random `0.0008503`

## 5) Interpretation (Required; causal language)

- RTG controllability is visible primarily in the stop-rate tail BECAUSE most samples still saturate at max_k=16, THEREFORE mean k_selected moves modestly while P(k_selected < max_k) increases strongly with lambda.
- DT beats Random under compute-matched evaluation at low penalty BECAUSE DT selects informative lags aligned with residual structure rather than uniform random picks, THEREFORE DT-Random is positive.
- DT stays close to OMP under both compute-matched and forced-K tables BECAUSE the learned lag-selection policy approximates greedy pursuit behavior on this speech subset.

## 6) Failures (Required even if PASS)

- No hard failures observed in this scale-check subset run.
- Note: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247/integrity_diagnostics.jsonl` is empty BECAUSE no out-of-range capture events occurred.

## 7) Reproduction Instructions (Required)

```bash
# 1) Environment
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

# 2) Execution
OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_20260123_203247"
conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth   --out_dir "$OUT_DIR"   --mode scale_check_subset --num_pairs 48   --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000   --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --random_trials 3 --random_sampling without_replacement --seed 0   --write_per_sample 1 --device cpu

# 3) Verification
# Expected:
# - integrity.num_capture_out_of_range_total == 0
# - spearman_lambda_k_selected ~= -1.0
# - per_sample.jsonl exists (enabled for scale-check)
```

## 8) Next Steps (Required)

- Run the guardrail diagnostic with `--random_sampling with_replacement` on the same 48-pair subset to validate duplicate-rate reporting and expected integrity failure behavior.
- After that, run full_dataset on speech (all pairs) for paper numbers.
