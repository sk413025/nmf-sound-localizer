# Acceptance Report: E4j-Speech -- Guardrail (Bad Warm-Start Checkpoint Must Fail-Fast)

## 1) Executive Summary

- Run: results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_guardrail_bad_initckpt_20260124_011702/
- Outcome: PASS (expected failure occurred)
- Purpose: Validate fail-fast behavior for `--init_ckpt` mismatch in `train_dt_lag_seq_rtg.py`.
- Dataset domain statement:
  - This run uses the speech WAV dataset only: YES (training data is penalty-OMP trajectories generated from speech WAV pairs)

## 2) Setup

### 2.1 Environment

- Conda env: trl-training
- Device: cpu
- Data (teacher trajectories):
  - data_path: results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/data/lag_trajectories.pt

### 2.2 Code provenance

- code_state.json: results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_guardrail_bad_initckpt_20260124_011702/code_state.json

## 3) Exact Command

See:
- results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_guardrail_bad_initckpt_20260124_011702/run.log

## 4) Expected Behavior (Guardrail Spec)

- The training script must fail-fast when `--init_ckpt` does not match the current model definition
  (e.g., wrong `rtg_dim`, wrong `M_lags`, wrong head output dim).
- No silent partial loading, no fallback, no best-effort coercion.

## 5) Observed Result

- The run terminated with a `RuntimeError` from `model.load_state_dict(..., strict=True)` showing multiple size mismatches:
  - `rtg_embed.weight` mismatch (checkpoint rtg_dim=1 vs current rtg_dim=2)
  - `state_embed.weight` / `corr_norm.*` mismatch (checkpoint M_lags=65 vs current M_lags=101)
  - `head.2.*` mismatch (checkpoint action_dim=65 vs current action_dim=102)

Decision:
- PASS because the script failed immediately with an explicit error BECAUSE strict loading rejects mismatched shapes;
  THEREFORE incorrect warm-start checkpoints cannot silently corrupt training.

## 6) Reproduction Instructions

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl

conda run -n trl-training python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/rtgomp_lambda_cost_E4j_speech_penaltyomp_data_stride128_clips12_20260124_011128/data/lag_trajectories.pt \
  --out_dir results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_guardrail_bad_initckpt_20260124_011702/model \
  --init_ckpt results/exp_m32_k16_full_gain100_ep50/model/dt_freq_aware_best.pth \
  --epochs 1 \
  --batch_size 64 \
  --lr 1e-4 \
  --freq_range 39,385 \
  --rtg_dim 2 \
  --rtg_mode lambda_cost \
  --lambda_c_values 1e-5,3e-5,1e-4,2e-4,3e-4 \
  --rtg1_mode max_k \
  --rtg1_max_k 16 \
  --use_stop_action \
  --seed 0 \
  |& tee -a results/rtgomp_lambda_cost_E4j_speech_stopcal_warmstart_guardrail_bad_initckpt_20260124_011702/run.log
```

Expected:
- The command fails with `RuntimeError: Error(s) in loading state_dict ... size mismatch ...`.

