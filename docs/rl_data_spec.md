# RL Data/Asset Spec (PPO, GRPO) — Fair to commit c96860b

## Data Roots (Config C)
- USM training root (Original, normalized):
  - `/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad_normalized`
  - Structure: 17 angle folders (`angle_30` … `angle_150`), 3 files each
  - Example file: `(145920,) float32` waveform
- Test root (Box, normalized):
  - `/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized`
  - Structure: same as above; total 51 test clips

Environment overrides (optional):
- `RL_USM_ROOT`, `RL_TEST_ROOT` — if set, override the defaults above.

## Transfer Function H
- Path: `h_matrix_normalized_original_to_box.pth`
- Location: same workspace root as `nmf_localizer` train/eval
- Contents: dict with `H` and `H_linear` tensors
  - `H: torch.float32, shape (F=346, D=17)`
  - `angles: torch.float32, shape (17,)`, values `[30, 45, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]`
  - Load via `torch.load(path, weights_only=False)`

## Speech Dictionary W (USM)
- Path: `doa_normalized_config_c_corrected_reproduction/models/usm.pth`
- Contents: dict with `W`
  - `W: torch.float32, shape (F=346, K=50)`
  - `n_freq=346`, `n_atoms_per_speaker=50`
  - Load via `torch.load(path, weights_only=False)`

## STFT and Band Limits (must match)
- Sampling rate: `fs=16000`
- STFT: `nperseg=2048`, `noverlap=1536` (75% overlap), `window='hann'`
- Frequency band: `freq_min=300.0`, `freq_max=3000.0`
- Shapes derived:
  - Frequency bins after band limit: `F=346`
  - Typical frames per clip: `N≈286` (first clip measured in pipeline)
- Reference implementation: `nmf_localizer/utils/audio_utils.py:15`, `nmf_localizer/core/data_processor.py:145`

## Matrix/Tensor Shapes (single-clip)
- Observation spectrogram `Y: (F=346, N≈286)`
- Directional filters `H: (F=346, D=17)`
- Dictionary `W: (F=346, K=50)`
- Mixing matrix `A = [diag(H_d) W]_d: (F=346, KD=850)`
- Coefficients `X: (KD=850, N≈286)`
- Group norms `||X_d||_1: (D=17)`
- Policy logits/probs (to add in RL): `(D=17)`

## RL Loader Contract
- Input unit: one test clip from `TEST_ROOT/angle_XX/clip_YYY.npy`
- Output fields per sample:
  - `Y (float32)`: STFT magnitude spectrogram `(F, N)` with band mask 300–3000 Hz
  - `angle_deg (float)`: ground-truth degree in `[30,150]`
  - `angle_index (int)`: index into `angles` array (0..16)
  - `path (str)`: filesystem path for traceability
- Optional caches (if using nmf_localizer for advantages): `Y_hat (F,N)`, `y_over_yhat2 (F,N)`, `inv_yhat (F,N)`

### Required content-root for s_hat (Configuration C exactness)
- You must supply `--content-root` to RL scripts (e.g., `train_single`) to compute `s_hat` from a parallel Original dataset; using the test-root (Box) for `s_hat` is disallowed.
- Assumptions:
  - The directory structure under `content-root` mirrors `test-root` (`angle_XX/clip_YYY.npy`).
  - `s_hat` is estimated by IS‑NMF on the content clip (Original), then mapped to Box via `H(Original→Box)` during advantage.
- Behavior:
  - Per-clip `s_hat` is computed from `content-root/angle_XX/clip_YYY.npy`.
  - If a matching content file is missing, the run errors out instead of falling back to the test clip.

Batching and groups:
- PPO: any batch size; samples are iid across angles
- GRPO: for each `Y`, draw `G` actions/samples to compute group-standardized advantages

## Advantage and Rewards (with shapes)
- Content vector `s` (per-direction or global): `(F,)` or `(F,N)`
  - Options:
    - `s = W z_hat` by IS-MU on `mean_t(|Y|)` with fixed `W`
    - `s_d = mean_t(W X_d)` recovered from localizer factorization
- Advantage per direction (★): `A ∈ ℝ^{D}`
  - `A_d = Σ_f (H_d ⊙ s)_f (Y_f/Ŷ_f^2 − 1/Ŷ_f)`
  - Use cached tensors `y_over_yhat2` and `inv_yhat` to avoid recomputation
- Scalar reward per rollout:
  - Outcome: `r = -D_IS(Y || Ŷ)` (scalar)
  - Or selected-action: `r = A_{d_selected}` (scalar)

## Strict Fairness Constraints (match c96860b)
- Use EXACT assets: H and W paths above; do not re-estimate
- STFT + band mask identical to pipeline settings
- No extra amplitude normalization (datasets are pre-normalized); keep raw magnitudes
- Frequency weights: default ones `(all ones)` to mirror pipeline
- IS geometry parameters: `beta=0.0`, `lambda_group=5.0`, `gamma_sparse=0.1`, `max_iter=100`
- Angles and count: `D=17`, degrees as listed
- Device: `cpu` unless otherwise validated

## Sanity Checks (before training)
- Load H: assert shape `(346,17)` and angles match
- Load W: assert shape `(346,50)`
- Probe one test clip: assert `Y.shape[0]==346` and `N≈286`
- Build `A`: assert `A.shape == (346, 850)`
- Factorize once: log `final loss` and `Localization result` for the clip to ensure parity

## Minimal RLConfig (defaults to Config C)
- `tf_path = "h_matrix_normalized_original_to_box.pth"`
- `w_path = "doa_normalized_config_c_corrected_reproduction/models/usm.pth"`
- `usm_root = "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_original_data_no_edge_sync_vad_normalized"`
- `test_root = "/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized"`
- `fs=16000, n_fft=2048, hop=512, window='hann', freq_min=300.0, freq_max=3000.0`
- `ppo: clip_eps, target_kl, entropy_coef`; `grpo: group_size, beta_kl`

## Code References (for alignment)
- STFT/Mask: `nmf_localizer/utils/audio_utils.py:15`, `nmf_localizer/core/data_processor.py:145`
- Load W/H: `nmf_localizer/core/localizer.py:63`, `nmf_localizer/core/localizer.py:81`
- Build A: `nmf_localizer/core/localizer.py:131`
- IS updates/caches: `nmf_localizer/core/localizer.py:296`, `nmf_localizer/core/localizer.py:326`
- Group norms (features): `nmf_localizer/core/localizer.py:450`
