# Plan: E4p-Speech — Dispersion-Prior Frequency Conditioning (Execution + Analysis)

This plan is executable. All commands are written for macOS and the project conventions.

---

## 0) Environment

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2:$PYTHONPATH
export MPLCONFIGDIR=/tmp/mpl
```

---

## 1) Fixed Inputs (Do Not Change)

Speech WAV roots:
- MIC: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- LDV: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

DT checkpoint:
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Lambda grid:
- `1e-5,3e-5,1e-4,2e-4,3e-4`

---

## 2) Smoke Test (Required)

Goal: confirm the evaluator runs end-to-end with prior enabled.

```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_smoke_normal_<TIMESTAMP> \
  --mode smoke --num_pairs 1 \
  --device cpu --require_wav_only 1 \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \
  --disp_prior_mode phase_slope_subbands --disp_prior_cond_mode normal \
  --disp_prior_num_subbands 3 --disp_prior_min_bins 64 --disp_prior_sigma_frames 2.0 --disp_prior_beta 2.0 \
  --write_delay_diagnostics 1
```

Artifacts to verify:
- `results/.../summary/dispersion_prior_summary.json` exists
- `run.log` shows `disp_prior_mode=phase_slope_subbands`

---

## 3) Functional Suite (Required; scale_check_subset=48)

Run 4 modes:
1) Baseline (no prior)
2) Prior normal (physical)
3) Prior shuffle (ablation)
4) Prior constant (ablation)

### 3.1 Baseline (no prior)
```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_none_<TIMESTAMP> \
  --mode scale_check_subset \
  --device cpu --require_wav_only 1 \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \
  --disp_prior_mode none \
  --write_delay_diagnostics 1
```

### 3.2 Prior normal
```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_normal_<TIMESTAMP> \
  --mode scale_check_subset \
  --device cpu --require_wav_only 1 \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \
  --disp_prior_mode phase_slope_subbands --disp_prior_cond_mode normal \
  --disp_prior_num_subbands 3 --disp_prior_min_bins 64 --disp_prior_sigma_frames 2.0 --disp_prior_beta 2.0 \
  --write_delay_diagnostics 1
```

### 3.3 Prior shuffle (ablation)
```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_shuffle_<TIMESTAMP> \
  --mode scale_check_subset \
  --device cpu --require_wav_only 1 \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \
  --disp_prior_mode phase_slope_subbands --disp_prior_cond_mode shuffle --disp_prior_seed 0 \
  --disp_prior_num_subbands 3 --disp_prior_min_bins 64 --disp_prior_sigma_frames 2.0 --disp_prior_beta 2.0 \
  --write_delay_diagnostics 1
```

### 3.4 Prior constant (ablation)
```bash
conda run -n trl-training PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/rtgomp_dtmin_disp_prior_E4p_speech_scale48_constant_<TIMESTAMP> \
  --mode scale_check_subset \
  --device cpu --require_wav_only 1 \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --max_k 16 --gain 100 --rtg_dim 2 \
  --disp_prior_mode phase_slope_subbands --disp_prior_cond_mode constant \
  --disp_prior_num_subbands 3 --disp_prior_min_bins 64 --disp_prior_sigma_frames 2.0 --disp_prior_beta 2.0 \
  --write_delay_diagnostics 1
```

---

## 4) Analysis Checklist (What to Read)

From each prior-enabled run:
- `summary/dispersion_prior_summary.json`
  - `dt_first_lag_abs_err_vs_tau_physical` at `lambda_c=3e-4`
- `summary/compute_matched_summary.json`
  - DT/OMP/Random capture means at `lambda_c=3e-4`
- `summary/rtg_controllability_summary.json`
  - Spearman(lambda, k_selected) should remain strongly negative (compute control intact)

Decision rule:
- Confirm `E_tau_shuffle > E_tau_normal` by a meaningful margin (see spec).

