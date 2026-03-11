# OMP Transformer — Speech260 16 kHz Full Training (QK + Hard Gumbel)

## Background

- Earlier speech260 experiments reused the white-noise H (`h_matrix_box_ldv_correct.pth`) and a speech-trained USM, but the speech260 pipeline was still at 48 kHz while H/DoADataset assumed `fs=16000`, `n_fft=2048`, band `[300, 3000]` Hz (F=346).  
  - Consequences: slow-sounding reconstructions, hidden STFT grid mismatch, and very low DOA accuracy (~2.7%).
- We resampled the entire speech260 original/box datasets to 16 kHz and verified:
  - `speech260_original_16k_no_edge_sync_vad_normalized` and `speech260_box_16k_no_edge_sync_vad_normalized` each contain 9620 `.npy` files.
  - `DoADataset` on the 16 kHz Box root yields `Y.shape = (346, N)` and angles `[0, 5, ..., 180]` — i.e., Y.F matches H.F.
- A speech260 Config C DOA smoke at 16 kHz (run_localization, speech-trained USM, 200 test examples) showed:
  - CPU: accuracy ≈ 9.7%, mean error ≈ 5.8° (still low but non-degenerate).
  - MPS: accuracy ≈ 4.3%, mean error ≈ 2.5° (qualitatively similar behavior).
- Separate USM reconstruction checks on 16 kHz speech260 showed:
  - Speech-trained W can reconstruct speech spectrograms with rel_Fro_error ≈ 0.23 and intelligible waveforms.
  - This suggests W is not catastrophically wrong; remaining issues likely lie in angle separability and localizer scoring.

## Motivation

- Train a full OMP Transformer (QK + hard Gumbel, 20 epochs) on **fully aligned** 16 kHz speech260:
  - `H`: 16 kHz, F=346 (Original→Box).
  - `W`: speech-trained USM, F=346, 50 atoms.
  - `Y`: 16 kHz speech260 Box, F=346 via `DoADataset`.
- Evaluate whether a token-based OMP Transformer can learn a more discriminative mapping for speech angles than the static H+W localizer alone.

## Experiment Purpose

- Full-training experiment: `omp-transformer-ldv.py` on speech260 Box 16 kHz, using:
  - QK routing with Phase 2-style hard Gumbel + masking.
  - Speech260 16 kHz Box as the dataset, speech260 16 kHz Original as the USM source (indirectly via W).
- Questions:
  1. Does training improve per-angle classification accuracy vs the static DOA localizer?
  2. How do training dynamics (loss curves, teacher metrics, QK–g alignment) behave on real speech vs white-noise?

## Setup

- Environment:
  - `source ~/.zshrc`
  - `conda activate trl-training`
  - `cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace`
  - `export PYTHONPATH=$(pwd):$PYTHONPATH`
- Assets:
  - H (Original→Box, 16 kHz):  
    `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth`
  - Speech260 original (16 kHz, normalized, VAD):  
    `/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized`
  - Speech260 box (16 kHz, normalized, VAD):  
    `/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized`
  - Speech USM (Config C 16 kHz smoke, MPS):  
    `doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth`
- Device: Apple Silicon MPS.

## Training Configuration

High-level OMP Transformer config (QK + hard Gumbel, Phase 2 style):

- Architecture:
  - `F = 346` (frequency bins, 300–3000 Hz at 16 kHz).
  - `E = 37` angles (experts).
  - `M = 8` atoms per expert (after atom reduction).
  - `P = E × M = 296` total dictionary atoms.
  - `d_model = 128`, `nhead = 2`, `nlayers = 1`.
  - `steps = 2` OMP steps.
- Routing / gating:
  - `routing_mode = 'qk'`.
  - `use_hard_gumbel = True`.
  - `score_norm = 'std'`.
  - `score_center_atoms = True`.
  - `score_center_expert = True`.
  - `expert_agg = 'l2'`.
  - `no_type_bias = True` (signal-preserving tokens).
- Atom reduction:
  - `atom_reduce_mode = 'kmeans'`.
  - `n_atoms = 8` (from USM W=346×50 → reduced dictionary).
- Optimization:
  - `epochs = 20`.
  - `batch_size = 32`.
  - `lr = 1e-3`.
  - Device `mps`.

## Exact Training Command (tmux)

Run in a new tmux session to avoid interruption:

```bash
tmux new -s speech260_full

source ~/.zshrc
conda activate trl-training
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace
export PYTHONUNBUFFERED=1
export PYTHONPATH=$(pwd):$PYTHONPATH

RUN_DIR="results/omp_transformer_speech260_train_20251114_185318"
mkdir -p "$RUN_DIR"

python -u scripts/omp-transformer-ldv.py \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth \
  --dataset_root /Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized \
  --routing_mode qk \
  --device mps \
  --epochs 20 \
  --batch_size 32 \
  --d_model 128 \
  --nhead 2 \
  --nlayers 1 \
  --steps 2 \
  --no_type_bias \
  --score_center_atoms \
  --score_center_expert \
  --score_norm std \
  --expert_agg l2 \
  --atom_reduce_mode kmeans \
  --n_atoms 8 \
  --lr 1e-3 \
  --use_hard_gumbel \
  --out_dir "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/run.log"
```

- Current run directory for this experiment:  
  `results/omp_transformer_speech260_train_20251114_185318`

## Execution Notes (Smoke Snapshot)

From `results/omp_transformer_speech260_train_20251114_185318/run.log`:

- STEP 1 (Data loading):
  - H/W:
    - `H shape: torch.Size([346, 37])`
    - `W shape: torch.Size([346, 50])` → reduced to 8 atoms via K-means.
  - Dictionary:
    - `Shape: torch.Size([346, 296]) (F=346, P=296)`
    - `Mutual coherence μ_max: ~1.0000`, `μ_mean: ~0.1534`.
  - Dataset (speech260 Box 16 kHz via DoADataset):
    - `Total samples: 9620`
    - `Y_samples: torch.Size([9620, 346])`
    - `Labels: torch.Size([9620]), unique angles: 37`
    - Dataset fingerprint MD5 (Y *.npy): `f563984848ae49b4443378c4ef720a51`.

- STEP 2 (Model initialization):
  - d_model=128, nhead=2, nlayers=1, routing_mode=qk, P=296.
  - Trainable parameters: 319,875.

- STEP 3 (Training, epoch 1 excerpt):
  - Epoch 1/20:
    - `loss ≈ 2.12`, `rec≈0.0023`, `class≈2.45`, `acc≈0.36`
    - Teacher metrics (subset):
      - `teacher_acc_subset ≈ 0.016` (static localizer baseline).
      - `qk_g_corr_pearson_mean ≈ 0.29` (initial alignment).
    - Temperatures:
      - `tau_e ≈ 0.748`, `tau_a ≈ 0.754`, `eta ≈ 0.654`.

> Note: At the time of this snapshot, the run has completed data loading, model initialization, and the first training epoch. Full 20-epoch metrics (final accuracy / per-angle behavior) will be summarized in a dedicated Results commit after training finishes and evaluation scripts are run.

## Reproduction Checklist

1. **Environment**: `trl-training` env active, `PYTHONPATH` set to repo root.
2. **Data**: Confirm 16 kHz speech260 roots exist:
   - `speech260_original_16k_no_edge_sync_vad_normalized`
   - `speech260_box_16k_no_edge_sync_vad_normalized`
3. **Assets**:
   - `h_matrix_box_ldv_correct.pth` present.
   - `doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth` present (F=346).
4. **Command**: Use the tmux-based training command above with a fresh `RUN_DIR`.
5. **Verification**:
   - Check `run.log` for 20 epochs worth of logs and no runtime errors.
   - Inspect `diagnostics.jsonl` for epoch-wise loss and teacher metrics.
   - Confirm `model_best.pth`, `code_state.json`, and any generated evaluation plots (e.g., `results.png`) exist under `RUN_DIR`.

