# RL Implementation Plan (PPO & GRPO) for nmf_localizer

## Objectives
- Implement PPO and GRPO training on the same DoA task while strictly matching the Config C setup to compare fairly with commit c96860b.
- Reuse nmf_localizer assets and processing: datasets, STFT/band limits, transfer functions H, and speech dictionary W.
- Keep the model architecture minimal: add a policy head and trainers without modifying nmf_localizer’s core algorithm.

## Ground Truth Assets and Shapes (Config C)
- USM training root (Original normalized): `/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad_normalized`
- Test root (Box normalized): `/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized`
- H path: `h_matrix_normalized_original_to_box.pth` → `H: (F=346, D=17)`, `angles: (17,)` = `[30..150]`
- W path: `doa_normalized_config_c_corrected_reproduction/models/usm.pth` → `W: (F=346, K=50)`
- STFT: `fs=16000, n_fft=2048, hop=512 (noverlap=1536), window=hann`, band `300–3000 Hz` → `F=346`
- Typical per-clip frames: `N≈286`
- Derived shapes:
  - `Y: (F=346, N≈286)`
  - `H: (346, 17)`
  - `W: (346, 50)`
  - `A=[diag(H_d)W]_d: (346, 850)`
  - `X: (850, N)`
  - `group_norms: (17,)`

Fairness constraints:
- Use the exact H and W files; do not re-estimate.
- Apply identical STFT and frequency band masking.
- Preserve magnitudes (datasets already normalized); no extra normalization.

## Package Layout (rl/)
- `rl/__init__.py` — empty.
- `rl/config.py` — RLConfig with defaults pointing to Config C assets.
  - Fields: `tf_path, w_path, usm_root, test_root, fs, n_fft, hop, window, freq_min, freq_max, device, seed`.
  - PPO: `clip_eps, target_kl, entropy_coef, lr, epochs, minibatch_size`.
  - GRPO: `group_size, beta_kl, clip_eps, lr, epochs`.
- `rl/data.py` — Dataset and DataLoader wrappers.
  - Reads test clips from `test_root` and yields: `Y(F,N), angle_deg, angle_index, path`.
  - STFT path matches `nmf_localizer/utils/audio_utils.AudioProcessor.compute_stft_spectrogram`.
  - Applies band mask 300–3000 Hz to keep `F=346`.
- `rl/assets.py` — Load H and W with assertions on shapes.
  - `load_H(tf_path) -> (H(F,D), angles(D,))` using `torch.load(weights_only=False)`.
  - `load_W(w_path) -> W(F,K)` from USM checkpoint.
- `rl/policy.py` — Policy head over directions.
  - Inputs: features `φ ∈ ℝ^D` (baseline: `group_norms`; optional: physics features like `⟨H_d, s_hat⟩`).
  - Outputs: logits `(D,)`, probs `(D,)`, sampling utilities, KL utility.
- `rl/advantage.py` — Advantage and reward computation in IS geometry.
  - Wraps nmf_localizer localizer to compute `Ŷ=A@X`, caches equivalent base terms.
  - Computes (★) `A_d = Σ_f (H_d ⊙ s)_f (Y_f/Ŷ_f² − 1/Ŷ_f)`; returns `A ∈ ℝ^D` and scalar outcome reward `r=-D_IS(Y||Ŷ)`.
  - `s_hat` options: `s=W z_hat` via 1D IS-MU on `mean_t(|Y|)` (default S1) or decolored by mean H (S2).
- `rl/buffer.py` — Rollout buffer with `(features, logits_old, actions, rewards/advantages)`.
- `rl/ppo_trainer.py` — PPO clipped surrogate + target-KL regulation.
- `rl/grpo_trainer.py` — GRPO group sampling, group-standardized advantages, explicit KL to `π_ref`.
- `scripts/train_ppo.py`, `scripts/train_grpo.py` — CLI entrypoints using RLConfig; defaults to Config C paths.

## Data Flow per Sample (Shapes)
1) Loader: read waveform `x: (T,)` → STFT magnitude → `Y: (F=346, N≈286)`; attach `angle_deg`, `angle_index`.
2) Localizer setup: load `W(346,50)`, `H(346,17)`; build `A(346,850)` as in `localizer._construct_mixing_matrix`.
3) Factorize once (IS): get `X(850,N)`, `Ŷ(346,N)`.
4) Content vector `s`:
   - Global: `s=W z_hat` via IS-MU on `mean_t(|Y|)`; `s: (346,)`.
   - Per-direction (optional): `s_d = mean_t(W X_d)`.
5) Compute advantage (★): `A ∈ ℝ^{17}` from `H_d` and `s` using base terms from `Y, Ŷ`.
6) Features `φ ∈ ℝ^{17}`: start with `group_norms` from factorization; optionally add `⟨H_d, s⟩`.
7) Policy head: logits `(17,)` → `π`; sample action(s), compute ratios and KL.

## PPO Training Loop (Outline)
- For each minibatch of clips:
  - Compute `A` and/or scalar rewards per sample.
  - Build features `φ` and get logits/probs `π_old` (store in buffer).
  - Sample actions `a` (discrete `0..16`), compute `A_t = A[a]`.
  - Epochs over buffer:
    - Recompute logits `π_new` on current features; `r_t = π_new(a)/π_old(a)`.
    - `L_clip = mean(min(r_t A_t, clip(r_t) A_t))` − `c_ent H(π_new)`; early-stop if `D_KL > target_kl`.

## GRPO Training Loop (Outline)
- For each clip `Y`, draw `G` samples and compute scores:
  - `r^{(i)} = A_{a^{(i)}}` (or outcome `-D_IS`); standardize within group.
  - Apply clipped surrogate averaged over the group plus explicit `β KL(π||π_ref)`.

## Validation and Fairness Checks
- Assert shapes: `H.shape == (346,17)`, `W.shape == (346,50)`, `Y.shape[0] == 346`, `A.shape == (17,)`.
- Angle order/values must match `angles` from H file.
- Baseline parity: nmf_localizer deterministic localization should stay ≥ baseline; RL should not degrade.
- Metrics to log: accuracy, mean/median error, KL, entropy, mean advantage, reward hist, per-angle stats.

## CLI Defaults (Config C)
- PPO example:
  - `python scripts/train_ppo.py --tf-path h_matrix_normalized_original_to_box.pth --w-path doa_normalized_config_c_corrected_reproduction/models/usm.pth --test-root /Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized --epochs 50 --clip-eps 0.2 --target-kl 0.02`
- GRPO example:
  - `python scripts/train_grpo.py --group-size 4 --beta-kl 0.01 --tf-path h_matrix_normalized_original_to_box.pth --w-path doa_normalized_config_c_corrected_reproduction/models/usm.pth --test-root /Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized`

## Milestones
1) Scaffolding: `rl/config.py, rl/data.py, rl/assets.py, rl/policy.py, rl/advantage.py`, CLI scripts; no code changes to nmf_localizer.
2) PPO baseline: features=`group_norms`; rewards from (★); confirm training stability and parity.
3) GRPO: implement group sampling and group-standardized advantages; reproduce PPO-level performance.
4) Physics-aware features: add `⟨H_d, s_hat⟩` to features; measure sample efficiency gains.
5) Optional speedups: expose caches from `localizer.factorize` to avoid recompute; add multiprocessing for STFT/IO.

