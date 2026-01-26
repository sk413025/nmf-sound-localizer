# Acceptance Report: E4n-Speech -- Phase Equalization EVAL (scale_check_subset, paired)

## 1) Executive Summary

- Run: `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_paired_20260126_123812/`
- Run type: EVAL
- Mode: scale_check_subset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q2 (EVAL): Does compute controllability still hold after applying phase_eq on a functional subset? PASS.
- Q3 (EVAL): Does DT remain non-degenerate at target compute after applying phase_eq on a functional subset? PASS.
- Q5 (EVAL): Provide the paired-side reference for the mispair guardrail comparison after calibration. PASS (paired metrics remain strong).

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- Device(s): `cpu`
- MPLCONFIGDIR: `/tmp/mpl`

- code_state.json: `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_paired_20260126_123812/code_state.json`

- Subset manifest: `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_paired_20260126_123812/subset_manifest.json`
  - num_pairs = 48
  - wav-only: YES
  - any .npy: NO

## 3) Exact Commands (Required)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_paired_20260126_123812"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
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
  --apply_phase_eq 1 --phase_eq_path results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/phase_eq/phase_eq.npz \
  |& tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

Hard guardrails:
- wav-only manifest: PASS
- fs mismatch encountered: NO
- any NaN/Inf in outputs: NO

RTG controllability:
- spearman(lambda_c, k_selected_mean) = -1.0 (from `summary/rtg_controllability_summary.json`)

Non-degeneracy at target compute (lambda_c=3e-4):
- k_selected_mean = 11.9476
- DT - Random (compute-matched capture mean) = +0.0293

Guardrail reference (paired-side, lambda_c=3e-4, dt coarse; from `summary/subsample_delay_diagnostics_summary.json`):
- psr_p50 = 87.71
- within_clip_tau_mad_ms_p50 = 0.0053 ms
- phase_slope_fraction_defined = 0.876

Decision:
- PASS BECAUSE compute remains controllable and DT remains better than Random, and paired subsample diagnostics remain sharply defined; THEREFORE this run is a valid functional “positive path” reference for guardrail comparisons.

Notes:
- Paired-vs-mispair separation is evaluated jointly with:
  - `results/rtgomp_phase_eq_E4n_speech_eval_scale_check_subset_mispair_shift1_20260126_124541/ACCEPTANCE_REPORT.md`
- Calibration effectiveness vs baseline is evaluated in:
  - `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/ACCEPTANCE_REPORT.md`

