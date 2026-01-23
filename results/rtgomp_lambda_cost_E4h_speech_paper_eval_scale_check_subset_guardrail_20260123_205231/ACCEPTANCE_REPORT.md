# Acceptance Report: E4h-Speech -- Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231/`
- Mode: `scale_check_subset` (guardrail diagnostic)
- Outcome: `FAIL` (expected for diagnostic baseline)
- Primary paper outputs:
  - Compute-matched DT vs OMP vs Random (WITH STOP; matched by `k_selected`)
  - Forced-K DT vs OMP vs Random (NO STOP)
- Dataset domain statement (required):
  - Speech WAV only (no `.npy` in manifest): `YES`
- Diagnostic statement (required):
  - This run uses Random sampling **with replacement** (non-paper baseline) to stress numeric integrity.

## 2) Setup (Required)

### 2.1 Environment

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- Device(s): `cpu`
- `MPLCONFIGDIR`: `/tmp/mpl`

### 2.2 Code provenance (Required)

- `code_state.json`: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231/code_state.json`
  - `git_head`: `430d0a51da85675b67bb432b8406145f3ef24535`
  - `dirty`: `true`

### 2.3 Data lineage (Speech-only; Required)

- mic_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset manifest: `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231/subset_manifest.json`
  - `num_pairs = 48`
  - `fingerprint_md5 = 739a181c331f347614090fffe6f4b491`
  - Domain validation: all `.wav` = `YES`, any `.npy` = `NO`

### 2.4 Fixed parameters (Must be explicit)

- hop_length = `160`, fs = `16000`, n_fft = `2048`, band = `[300, 3000]` Hz
- max_lag = `50` (M_lags=101), Tw = `32`, max_k = `16`, gain = `100.0`
- rtg_dim = `2`, use_stop_action = `true`
- lambda_c_values = `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`

### 2.5 Baselines (Must be explicit)

- Random baseline (DIAGNOSTIC):
  - random_trials = `1`
  - random_sampling = `with_replacement`

## 3) Exact Commands (Required)

```bash
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py   --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC   --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV   --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth   --out_dir "$OUT_DIR"   --mode scale_check_subset --num_pairs 48   --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000   --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2   --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"   --random_trials 1 --random_sampling with_replacement --seed 0   --write_per_sample 0 --device cpu   2>&1 | tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

### 4.0 Evaluator integrity / correctness checks (Hard guardrails)

From `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231/summary/compute_matched_summary.json`:

- num_samples_total = `80964`
- num_missing_files = `0`
- num_md5_mismatches = `0`
- num_nan_or_inf = `0`
- num_capture_out_of_range_total = `7242` (FAIL, expected for diagnostic baseline)
  - dt = `0`, omp = `0`, random = `7242`
- Random duplicate rate: `0.0710531841` (expected > 0 for with-replacement)
- Random capture min/max:
  - min `-46.0457115`, max `0.9999998212`
- Worst violation summary:
  - method `random`, eval_context `forced_k`, capture `-46.0457115`
  - k=8, clip_idx=1, window_idx=4, freq_idx=100, duplicates=1 (unique_count=7)
  - E0=6.2089725, E_res=292.1055269
- Diagnostics file:
  - `results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_guardrail_20260123_205231/integrity_diagnostics.jsonl` (7242 rows)

Decision:
- FAIL (expected) BECAUSE Random with replacement produces duplicate atoms, which can make the LS system ill-conditioned; finite-precision lstsq can yield E_res > E0, THEREFORE capture becomes negative and violates the [0,1] integrity bound.

### 4.1 RTG controllability (Free rollout)

- spearman(lambda_c, k_selected_mean) = `-1.0`

Decision:
- Controllability trend remains visible; however, this run is not paper-valid due to integrity failures in the diagnostic Random baseline.

## 5) Interpretation (Required; causal language)

- This diagnostic run validates the guardrail/diagnostics machinery BECAUSE it triggers out-of-range capture events only in Random (with duplicates), while DT and OMP remain within bounds.
- The magnitude of negative capture demonstrates why the paper baseline must be without replacement: duplicates can catastrophically destabilize least-squares projection in finite precision.

## 6) Next Steps (Required)

- Use the PASS scale_check_subset run with Random without replacement for paper shaping.
- Proceed to full_dataset on speech with Random without replacement for paper numbers.
