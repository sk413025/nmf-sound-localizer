# Acceptance Report: E4h — Paper-Grade DT vs OMP vs Random

## 1) Executive Summary

- Run: `results/rtgomp_lambda_cost_E4h_paper_eval_scale_check_subset_20260122_120437/`
- Mode: `scale_check_subset`
- Outcome: `FAIL`
- Primary paper outputs:
  - Compute-matched DT vs OMP vs Random (with STOP)
  - Forced-K DT vs OMP vs Random (no STOP)

## 2) Setup (REQUIRED)

- Env: `trl-training`
- Device(s): `mps` (primary; model inference on MPS with LS projection on CPU to avoid complex gather limitations on MPS)
- Checkpoint: `results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth`
- Subset manifest: `results/rtgomp_lambda_cost_E4h_paper_eval_scale_check_subset_20260122_120437/subset_manifest.json`
  - `num_pairs = 48`
  - `fingerprint_md5 = 4b2abe06b83f9142c14527fe0fd2d494`
- Fixed params:
  - `hop_length=160`, `max_lag=50`, `Tw=32`, `max_k=16`, `gain=100.0`, `rtg_dim=2`, `use_stop_action=true`
  - `fs=16000`, `n_fft=2048`, band `[freq_min, freq_max]=[300, 3000]`
- Random baseline:
  - `random_trials = 3`
  - sampling: with replacement

## 3) Exact Commands (REQUIRED)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.

conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad_normalized" \
  --ldv_root "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized" \
  --ckpt_path "results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth" \
  --out_dir "results/rtgomp_lambda_cost_E4h_paper_eval_scale_check_subset_20260122_120437" \
  --mode scale_check_subset \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --rtg_dim 2 \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --random_trials 3 \
  --seed 0 \
  --write_per_sample 0 \
  --device mps 2>&1 | tee -a "results/rtgomp_lambda_cost_E4h_paper_eval_scale_check_subset_20260122_120437/run.log"
```

## 4) Results (REQUIRED)

### 4.0 Evaluator integrity / correctness checks (MUST)

- `num_samples_total = 415200`
- `num_samples_used = 415200`
- `num_missing_files = 0`
- `num_md5_mismatches = 0`
- `num_nan_or_inf = 0`
- `num_capture_out_of_range = 64`
- `num_omp_monotonicity_violations = 0`
- `num_dt_duplicate_actions_forced_k = 0`

Decision:
- `FAIL` BECAUSE `num_capture_out_of_range > 0`, which violates the guardrail and makes the evaluation not fully trustworthy until the cause is diagnosed.

### 4.1 RTG controllability (free rollout)

- `spearman(lambda_c, steps_used_mean) = -0.9` (target <= -0.6)
- steps range = `0.4150`

Decision:
- `PASS`

### 4.2 Compute-matched DT vs OMP vs Random (WITH STOP)

At `lambda_c = 1e-4`:
- `DT_capture = 0.99768`
- `OMP_capture = 0.99988`
- `Random_capture = 0.93081`
- `DT/OMP = 0.99780`
- `DT - Random = 0.06687`

Decision:
- `PASS` for the paper claim (DT - Random > 0 at low penalty).

### 4.3 Forced-K DT vs OMP vs Random (NO STOP)

- K=1: DT=0.24532, OMP=0.93014, Random=0.20592, DT/OMP=0.26375, DT-Random=0.03940
- K=2: DT=0.41030, OMP=0.97071, Random=0.37807, DT/OMP=0.42268, DT-Random=0.03223
- K=4: DT=0.67320, OMP=0.99083, Random=0.63916, DT/OMP=0.67944, DT-Random=0.03404
- K=8: DT=0.95377, OMP=0.99839, Random=0.88613, DT/OMP=0.95531, DT-Random=0.06764
- K=16: DT=0.99946, OMP=0.99994, Random=0.93184, DT/OMP=0.99952, DT-Random=0.06762

Interpretation:
- Forced-K consistently beats Random; if compute-matched ever underperforms, it would be due to early STOP. Here compute-matched remains strong at low penalty.

## 5) Interpretation (REQUIRED; causal language)

- RTG affects compute BECAUSE higher `lambda_c` increases the cost of selecting more lags, THEREFORE the model stops earlier and the mean steps decline.
- DT beats Random under compute-matched evaluation BECAUSE the policy selects lags informed by correlations and RTG conditioning, THEREFORE capture increases above the random baseline.
- DT is close to OMP at low penalty BECAUSE the learned policy approximates greedy correlation-based selection when the penalty is small, THEREFORE DT/OMP remains near 1.0.
- The run still FAILS overall BECAUSE some capture values fall outside the expected range, indicating a numerical or projection accounting issue that must be diagnosed.

## 6) Next Steps (REQUIRED)

- Diagnose out-of-range capture values BECAUSE guardrails prohibit reporting until they are explained; add logging to record min/max capture and the magnitude of violations.
- If violations are numeric noise, document the tolerance explicitly and re-run; otherwise fix projection or residual computation and re-run the scale-check.
