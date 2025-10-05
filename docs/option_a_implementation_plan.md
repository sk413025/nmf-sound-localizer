# Option A Implementation Plan — K‑Step Mixture Construction (ΔIS Rewards)

This document records the concrete, staged plan to implement a physically grounded, multi‑step RL formulation for DoA using PPO/GRPO on top of the current Hugging Face + TRL stack. Each stage is a self‑contained experiment/test with exact reproduction steps and expected outcomes.

## Overview

- Objective: Move from a 1‑step smoke test (single direction token) to a K‑step episode where each step selects a new direction and incrementally reduces the acoustic mismatch `D_IS(Y||Ŷ)`. This aligns with the generative model `Ŷ = (H ⊙ ŝ) · π` and is naturally extensible to multi‑source and ICL.
- Action space: Dataset‑derived angles (e.g., 17 angles in the two box datasets), not a fixed 24‑grid.
- Policy: Small GPT‑2 LM (d_model=256, n_head=8, n_layer=2, n_positions=512) with TRL value head.

## Stage 0 — Hygiene: Left Padding

- Goal: Remove decoder‑only right‑padding warning; keep behavior unchanged.
- Change:
  - In `doa_rl/hf/tokenizer.py:build_patch_tokenizer(...)`, set `hf_tokenizer.padding_side = "left"` before returning.
- Test (smoke): Existing command completes, and the right‑padding warning disappears.
- Commit message template:
  - "Fix: left padding for decoder‑only generation — remove warning; behavior unchanged"

## Stage 1 — K‑Step Generation + No‑Repeat Masking (Scaffold)

- Goal: Turn each episode into K steps (default K=3) with distinct direction tokens.
- Changes:
  - CLI: Add `--K` and set `response_length = K` in `scripts/train_trl_policy.py`.
  - Logits processor: Extend `doa_rl/hf/logits_mask.py` with a history‑aware processor:
    - `NoRepeatDirectionLogitsProcessor(allowed_ids)` that masks previously generated direction IDs per sample by inspecting `input_ids[:, context_len:]`.
  - Wiring: After creating `PPOTrainer`, attach the processor to `trainer.policy_model.pretrained_model.generation_config`.
- Reward (temporary): Keep TRL’s default scalar (value + KL) to validate multi‑token output.
- Test (smoke): With `--K 3`, generated sequences contain 3 distinct direction tokens.
- Commit message:
  - "Experiment: K‑step (K=3) generation with no‑repeat direction masking — multi‑token output validated"

## Stage 2 — Physics Reward V1: Sum of Standardized A[d]

- Goal: Introduce a per‑step, physically meaningful proxy reward without heavy compute.
- Changes:
  - ŝ precompute: During sample prep, compute `ŝ` once per Y using a light NMF (reuse `NMFSoundLocalizer` or `estimate_s_hat` with small `max_iter`).
  - New helper `doa_rl/hf/reward_utils.py`:
    - `compute_stepwise_rewards_A(Y, s_hat, H, selected_dirs)`:
      - Initialize `π_0 = 0`; for step t, call `AdvantageComputer(Y, pi=π_{t-1}, s_hat)`; `r_t = zscore(A[d_t])`; update `π_t[d_t]`.
      - Return scalar `R = sum_t r_t` (keep per‑step `r_t` in logs).
  - Training: Parse the K generated direction tokens; compute `R` and pass to PPO (TRL expects a scalar).
- Test (smoke): With `--K 3`, confirm finite rewards; inspect logs.
- Commit message:
  - "Experiment: physics V1 rewards (sum of standardized A[d_t]) — K‑step set selection"

## Stage 3 — Physics Reward V2: Sum of ΔIS (Full Mixture)

- Goal: Optimize the true acoustic objective: per‑step reduction in IS divergence as we add directions.
- Changes:
  - In `reward_utils.py` add `compute_stepwise_rewards_delta_IS(Y, s_hat, H, selected_dirs, weighting)`:
    - Precompute `Hs = H ⊙ ŝ`.
    - Maintain `Ŷ_0 = eps`, `Ŷ_t = Ŷ_{t-1} + Hs[:, d_t]` (or averaged if `weighting='avg'`).
    - IS divergence: `IS(Y, Ŷ) = Σ (Y/Ŷ − log(Y/Ŷ) − 1)`, with eps clamps.
    - `r_t = −(IS_t − IS_{t-1})`, `R = Σ r_t` (log per‑step contributions).
  - CLI: `--reward-mode {proxyA,deltaIS}`; default to `deltaIS` after validation.
- Test (smoke): With `--K 3`, confirm `r_t ≤ 0`, `R ≤ 0` (normalize for PPO stability).
- Commit message:
  - "Experiment: physics V2 rewards (sum of ΔIS per selected direction) — K‑step mixture construction"

## Stage 4 — GRPO for Set Selection (Optional)

- Goal: Improve relative selection by standardizing rewards within groups.
- Changes:
  - For each query, sample G sets (size K) and compute `R_g` via Stage 3.
  - Standardize `R_g` (z‑score) within the G group and feed to GRPO trainer.
  - CLI: `--algo grpo --G 4`.
- Test (smoke): With `--algo grpo --G 4 --K 3`, confirm stable updates.
- Commit message:
  - "Experiment: GRPO for K‑step set selection — group‑standardized ΔIS rewards"

## Stage 5 — Early Stop + Evaluation

- Goals: Avoid over‑selection; evaluate quality.
- Changes:
  - Early stop: Add a stop token or `--epsilon-stop` threshold on marginal `|ΔIS|`.
  - Evaluation script: Compare selected set vs. ground truth angles; plot ΔIS curves.
- Commit message:
  - "Evaluation: set‑wise metrics and ΔIS curves; add stop token/threshold ε"

## File/Interface Summary

- `doa_rl/hf/tokenizer.py`
  - `padding_side = "left"` (Stage 0)
- `doa_rl/hf/logits_mask.py`
  - `NoRepeatDirectionLogitsProcessor` (Stage 1)
- `doa_rl/hf/reward_utils.py` (new)
  - `compute_stepwise_rewards_A` (Stage 2)
  - `compute_stepwise_rewards_delta_IS` (Stage 3)
- `scripts/train_trl_policy.py`
  - CLI: `--K`, `--reward-mode {proxyA,deltaIS}`, `--mask-no-repeat`, `--epsilon-stop`, `--algo {ppo,grpo}`, `--G`.
  - Data prep: ŝ precompute.
  - Training: set `response_length=K`, attach logits processor, compute scalar `R` from selected dirs.
- (Optional) switch to in‑repo PPO/GRPO to accept per‑step rewards explicitly; otherwise continue with summed scalar `R` for TRL.

## Risks & Mitigations

- Order invariance of sets → tag step index (e.g., `<D1_…>, <D2_…>`) or sort by marginal gain.
- Over‑selection → stop token or ΔIS threshold ε.
- Reward scaling → batch/group standardization (GRPO).

## Reproduction Template (per stage)

```bash
# 0) Env
conda activate trl-training
export PYTHONPATH=/path/to/angle-based-byol:/path/to/development-workspace:$PYTHONPATH

# 1) Run smoke
python scripts/train_trl_policy.py \
  --data-root <dataset_root> \
  --epochs 1 --batch-size 2 --ppo-epochs 1 --max-samples 3 \
  --K 3 --reward-mode deltaIS \
  --patch-fp 16 --patch-np 10 --n-fft 2048 --sample-rate 48000 \
  --freq-min 300 --freq-max 3000

# 2) Inspect
# - Confirm K distinct direction tokens
# - Check ΔIS per step and total reward R
# - Monitor KL and losses
```

## Commit Discipline

- Each stage is a single “Experiment:” commit with background, motivation, purpose, expected outcome, and reproduction steps.
- Follow with a “Results:” commit capturing metrics, ΔIS traces, and findings.

