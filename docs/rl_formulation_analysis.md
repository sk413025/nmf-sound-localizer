# RL Formulation Analysis — PPO/GRPO With Dataset‑Derived Directions

This note documents, from the current code, what observation/state/action/step/trajectory/epoch/agent mean, why the earlier 1‑step episode was intentional for a smoke test, and why a multi‑step, physically grounded formulation (Option A) is recommended over a per‑timeframe formulation (Option B). It concludes with a concrete implementation plan that preserves simplicity and extends to multi‑source DoA and in‑context learning (ICL).

## Current Setup (Code‑verified)

- Observation source
  - `doa_rl/data.py` yields `Y (F,N)` magnitude spectrogram per clip after STFT band pass (300–3000 Hz).
  - `doa_rl/features/tokenizers.py` `PatchTokenizer(Fp=16, Np=10)` emits ~126 patch tokens (`<P_i_j_level>`).

- Direction set and tokenizer
  - `scripts/train_trl_policy.py` discovers angles by scanning `data_root/angle_*` (17 in the two box datasets).
  - `doa_rl/hf/tokenizer.py` `build_patch_tokenizer(direction_angles)` builds vocab = specials + patch tokens + only those direction tokens.

- Policy/value (agent)
  - `doa_rl/hf/model.py` builds a small GPT‑2 LM (d_model=256, n_head=8, n_layer=2) wrapped by TRL’s value head. Returns a `ModelOutput` (logits, hidden_states, value). Exposes `score = v_head` for TRL.

- PPO training (TRL 0.23.1)
  - `scripts/train_trl_policy.py` constructs `PPOTrainer(policy, ref_model, reward_model=value_head)` and runs a 1‑epoch smoke test with `response_length=1` (one direction token).

## Mapping RL Terms to This Project

- Observation
  - What: Patch token sequence (~126 tokens) derived from `Y(F,N)` for one clip.
  - Physical meaning: Local time–frequency energy tiling of the acoustic scene inside the test box.

- State
  - What: Identical to the observation in the current smoke test (no transitions).
  - Physical meaning: One‑shot context — the clip’s TF content; no evolving Markov state.

- Action
  - What: Generate exactly one direction token from the discovered set (17 angles in the two datasets).
  - Physical meaning: Predicted DoA for the clip.

- Step
  - What: One token generation per sample (degenerate single‑step episode for sanity checks).
  - Physical meaning: A final DoA decision; no sequential refinement yet.

- Trajectory
  - What: `[prompt patch tokens] + [one direction token]`.
  - Physical meaning: A one‑decision “episode”.

- Epoch
  - What: One PPO update cycle over the batch (as configured for smoke tests).
  - Physical meaning: One conservative trust‑region update of the policy over the current prompts.

- Agent
  - What: GPT‑2‑sized policy + value head; frozen reference for KL.
  - Physical meaning: Categorical classifier over the current direction simplex (17 angles) with a learned value baseline.

- Reward (current smoke)
  - What: TRL’s reward path via value head + KL penalty (proxy; not physical yet).
  - Limitation: Good for validating the loop; not grounded in acoustic reconstruction.

## Why 1‑Step Was Intentional (and Insufficient)

The 1‑step episode was a deliberate degenerate RL setup to validate the TRL stack and the dataset‑derived action space. It proves logits live on the correct 17‑simplex, KL is small, and the trainer is wired correctly. However, it under‑uses physics: there is no sequential improvement signal nor source composition. We should move to a physically meaningful, multi‑step episode.

## Two Physically Meaningful Options

### Option A — Multi‑Token Mixture Construction (Recommended)

- Principle (generative): In the box, `Y(f, t)` is explained by a mixture of directional transfer functions `H(f, d)` gated by content `ŝ(f)` and direction weights `π(d)`. Constructing `Ŷ_t = (H ⊙ ŝ) · π_t` step‑by‑step reduces the IS divergence `D_IS(Y||Ŷ_t)`.

- Episode of K steps:
  1. At step `t`, select a direction token `d_t` (forbid repeats) and update `π_t` and `Ŷ_t`.
  2. Reward: `r_t = −(D_IS(Y||Ŷ_t) − D_IS(Y||Ŷ_{t−1}))` (marginal IS reduction; ≤ 0). Normalize per batch.
  3. Early stop if marginal gain < ε, or emit a stop token.

- Physical meaning: Greedy mirror‑descent on the direction simplex; each step adds a physically plausible direction that reduces acoustic mismatch.

- Multi‑source readiness: If there are `M` active sources, choose `K ≳ M` (e.g., 2–4) — the set of selected directions approximates the source set.

- ICL friendliness: Few‑shot prompts can show `[patch tokens] → [direction set]`; the LM learns to output K tokens (a set) for new clips.

### Option B — Per‑Timeframe DoA (Temporal)

- Principle: Partition the clip into `T` windows; at each step `t`, predict the DoA for that window.
- Reward: Per‑frame angle error or per‑frame ΔIS; aggregate across `T`.
- When useful: Moving sources or non‑stationary scenes; needs per‑frame labels or robust frame‑wise physical proxies.
- Downsides here: Heavier, noisier rewards; harder credit assignment; duplicated selection work if sources are stationary.

## Why Choose Option A

- Physics alignment: Directly optimizes acoustic reconstruction (IS divergence) via direction set composition.
- Sequential signal: Dense per‑step rewards (ΔIS) give clear guidance and stability with PPO/GRPO.
- Multi‑source and ICL: Natural extension to multiple simultaneous angles and few‑shot in‑context prompting.
- Simplicity: Minimal changes — keep the prompt; output K tokens; compute `Ŷ_t` incrementally; reuse existing `AdvantageComputer` pieces to compute ΔIS or start with standardized `A[d_t]` as a proxy.

## Minimal Implementation Plan (Option A, K=3)

1) Episode length
   - Set `response_length = 3` (configurable `K`).
   - Extend `DirectionLogitsProcessor` to forbid repeats across steps (track history during generation).

2) Reward
   - Phase 1 (fast): `r_t = zscore(A[d_t])` using `AdvantageComputer` (already computes `A(d)`).
   - Phase 2 (full): maintain `Ŷ_t = (H ⊙ ŝ) · π_t` and compute `r_t = −ΔD_IS(Y||Ŷ_t)`; early stop if `|Δ| < ε`.
   - GRPO: sample G sets and group‑standardize rewards per query; fits set selection naturally.

3) Trainer path
   - Easiest: use our in‑repo PPO/GRPO trainers (categorical) and pass per‑step rewards explicitly.
   - Alternatively: fork a thin layer on TRL’s PPO to accept reward vectors `(r_1..r_K)` while retaining TRL’s KL/masking.

4) Logging/metrics
   - Track per‑step ΔIS, cumulative IS, number of steps until stop, selected set vs. known angles.

5) Guardrails
   - Add a stop token and/or threshold ε; canonicalize order (by marginal gain) or tag steps as `<D1_…>, <D2_…>, …`.

## Risks & Mitigations

- Order invariance: The physical set is orderless. Mitigate by sorting by marginal gain or tagging step index in the token.
- Over‑selection: Stop token or ΔIS threshold ε.
- Reward scale: Standardize per batch/group (especially with GRPO) for stability.

## Next Steps (Actionable)

- [ ] Set `response_length=3`; extend logits processor to forbid repeats; add optional stop token.
- [ ] Wire `AdvantageComputer` for `A[d]` → `r_t` (Phase 1), then ΔIS (Phase 2).
- [ ] Use existing PPO/GRPO trainers to accept per‑step rewards; or add a thin TRL reward‑vector adapter.
- [ ] Add left padding `tokenizer.padding_side = "left"` to remove decoder‑only warnings.
- [ ] Add evaluation script: per‑clip selected set vs. ground‑truth angles; ΔIS curves.

## Reproduction (Current Smoke Test)

- Environment: conda `trl-training` (Python 3.11); torch 2.8.0; transformers 4.57.0; trl 0.23.1
- Command:
  ```bash
  export PYTHONPATH=/path/to/angle-based-byol:/path/to/development-workspace:$PYTHONPATH
  python scripts/train_trl_policy.py \
    --data-root <dataset_root> \
    --epochs 1 --batch-size 2 --ppo-epochs 1 --max-samples 3 \
    --response-length 1 --temperature 1.0 \
    --patch-fp 16 --patch-np 10 --n-fft 2048 --sample-rate 48000 \
    --freq-min 300 --freq-max 3000
  ```
- Outcome: Confirms that logits live on the correct 17‑simplex and the trainer runs; move next to Option A multi‑step rewards.

