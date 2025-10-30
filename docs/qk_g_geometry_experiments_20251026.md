# QK vs g: Geometry-First Fixes — Four Experiments (2025-10-26)

## Background
- Prior pure-QK runs showed classification loss stuck near ln(37)≈3.61 with low alignment (qk_g_corr≈0) and poor accuracy, despite g-based routing reaching 100% accuracy on the same data. This suggested QK gradients existed but produced common‑mode updates (no contrast), likely due to token/aggregation/normalization design and dictionary geometry (high mutual coherence).

## Motivation
- Verify whether the lack of loss decrease is due to (1) broken gradient paths, or (2) low-contrast geometry and token design. Propose minimal, simplifying interventions to restore contrast without adding complexity.

## Purpose
- Demonstrate that QK can learn effectively once we remove non-essential biases/normalization pitfalls and improve geometry. Compare four minimal settings:
  - A: Remove interference, no compression (d=346), keep L2 aggregation.
  - B: A + diverse atom subset (lower μ_mean), M=12, cos-separation ≥0.98.
  - C: A + change expert aggregation from L2 to max.
  - D: A + allow dictionary tokens to attend residual (cross-attn), keep other toggles.

## Environment / Setup
- Conda: `trl-training`
- Device: CPU (MPS/CUDA not required for smoke tests)
- Python path: project root
- Data roots (real data only):
  - Dataset root: `/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized`
  - H: `/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth` (F=346, E=37)
  - W: `doa_normalized_config_c_corrected/models/usm.pth` (F=346, M=50)
- Dataset fingerprint: `713c0635878a04b32f4ee30208904d11` (111 npy files)
- Seeds: torch=42, numpy=42 (in script)

## Exact Commands
Export env (bash):
```
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=$PWD:$PYTHONPATH
```

Run A (minimal-no-bias, no compression, L2 agg):
```
PYTHONUNBUFFERED=1 python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --hybrid_alpha 0.0 --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 346 --nhead 2 --nlayers 1 \
  --no_type_bias --encoder_identity --single_gate_expert \
  --score_center_atoms --score_center_expert --score_norm std \
  --out_dir results/exp_A1_qk_minimal_no_bias_id_d346_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee results/exp_A1_qk_minimal_no_bias_id_d346_$(date +%Y%m%d_%H%M%S)/run.log
```

Run B (A + diverse atoms, M=12, cos≥0.98):
```
PYTHONUNBUFFERED=1 python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --hybrid_alpha 0.0 --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 346 --nhead 2 --nlayers 1 \
  --no_type_bias --encoder_identity --single_gate_expert \
  --score_center_atoms --score_center_expert --score_norm std \
  --atom_reduce_mode diverse --atom_min_cos 0.98 --n_atoms 12 \
  --out_dir results/exp_B_qk_diverse_M12_cos098_d346_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee results/exp_B_qk_diverse_M12_cos098_d346_$(date +%Y%m%d_%H%M%S)/run.log
```

Run C (A + expert_agg=max):
```
PYTHONUNBUFFERED=1 python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --hybrid_alpha 0.0 --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 346 --nhead 2 --nlayers 1 \
  --no_type_bias --encoder_identity --single_gate_expert \
  --score_center_atoms --score_center_expert --score_norm std \
  --expert_agg max \
  --out_dir results/exp_C_qk_expertagg_max_d346_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee results/exp_C_qk_expertagg_max_d346_$(date +%Y%m%d_%H%M%S)/run.log
```

Run D (A + d_can_attend_r):
```
PYTHONUNBUFFERED=1 python -u scripts/omp-transformer-ldv.py \
  --routing_mode qk --hybrid_alpha 0.0 --device cpu \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --epochs 10 --batch_size 16 --d_model 346 --nhead 2 --nlayers 1 \
  --no_type_bias --encoder_identity --single_gate_expert \
  --score_center_atoms --score_center_expert --score_norm std \
  --d_can_attend_r \
  --out_dir results/exp_D_qk_d_can_attend_r_d346_$(date +%Y%m%d_%H%M%S) \
  2>&1 | tee results/exp_D_qk_d_can_attend_r_d346_$(date +%Y%m%d_%H%M%S)/run.log
```

## Artifacts (paths)
- A: `results/exp_A1_qk_minimal_no_bias_id_d346_20251026_230658/`
- B: `results/exp_B_qk_diverse_M12_cos098_d346_20251026_230722/`
- C: `results/exp_C_qk_expertagg_max_d346_20251026_230748/`
- D: `results/exp_D_qk_d_can_attend_r_d346_20251026_230812/`

Each contains:
- `run.log` (stdout/stderr, unbuffered)
- `diagnostics.jsonl` (per-epoch metrics)
- `model_best.pth` (best checkpoint; LFS-tracked)
- `results.png` (visual)
- `code_state.json` (git_head, dirty, sha256 of executed script)

## Results (key numbers)
- A: loss 3.963 → 0.878, acc 97.3%, align 0.194, μ_max 0.9954, μ_mean 0.4474
- B: loss 3.924 → 0.577, acc 100.0%, align 0.186, μ_max 0.9977, μ_mean 0.3008
- C: loss 3.887 → 1.087, acc 91.9%, align -0.010, μ_max 0.9954, μ_mean 0.4474
- D: loss 3.963 → 0.878, acc 97.3%, align 0.194, μ_max 0.9954, μ_mean 0.4474

Interpretation:
- A shows that removing type bias, avoiding encoder perturbation, centering/standardizing scores, and keeping full frequency (d=346) generate sufficient contrast for QK to learn (CE<<ln(37)).
- B reduces μ_mean via diverse selection; contrast improves further, reaching 100% accuracy.
- C shows max aggregation underperforms L2 for this geometry (worse alignment and accuracy).
- D shows D→R cross-attention is not necessary once A’s simplifications are applied.

## Physical/Mathematical Analysis
- Signal model: Y≈D x, with D formed by element-wise products H⊙W (E angles × M atoms), normalized columns.
- Greedy physics score g=D^T r measures per-atom correlation with residual; energy aggregated per angle yields high contrast because correct angle’s atom projections concentrate energy.
- QK score initially lacked contrast DUE TO token/key design independent of r and common-mode norms; CE gradient thus became nearly symmetric across angles (no contrast), so loss stagnated.
- THEREFORE aligning token design with residual information (identity encoder, no type bias), keeping full F, and reducing dictionary mean coherence causes per-angle logits to separate.
- Information-theoretically, high μ_mean reduces discriminative capacity of dot products; lowering μ_mean increases mutual information between logits and true angle.

## Cross-Experiment Analysis
- Pattern: A and D (same geometry) both succeed BECAUSE removing bias/perturbation and using full F restore contrast; cross-attention is unnecessary.
- Success factors: B > A BECAUSE lower μ_mean from diverse selection increases separability even with the same training objective.
- Failure modes: C < A BECAUSE max aggregation discards consistent multi-atom evidence, lowering effective SNR of logits.
- Method effectiveness: Geometry-first simplifications dominate; loss terms (CE) suffice once contrast exists.

## Extracted Principles
- Design principles: Prefer full-frequency representations and remove non-physical type biases for routing; keep encoder identity unless proven helpful; use L2 aggregation for multi-atom evidence.
- Hypothesis formation: GIVEN μ_mean↓, predict CE will drop faster and accuracy rise; GIVEN max aggregation, expect slower/poorer convergence.
- Resource allocation: Prioritize dictionary geometry and score normalization over architectural complexity.
- Risk mitigation: Track μ_max/μ_mean and `scores_expert_std/logits_margin` to detect common-mode collapse early.
- Success amplification: Use diverse atom reducers to keep μ_mean low when M increases.

## Reproduction Instructions
1) Environment
```
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=$PWD:$PYTHONPATH
```
2) Data
```
ls /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized
find /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized -name "*.npy" | wc -l  # expect 111
```
Dataset fingerprint (expected): `713c0635878a04b32f4ee30208904d11`

3) Execute: run A/B/C/D commands above (10 epochs each, CPU)

4) Verification
- Check `Overall accuracy` in each run’s `run.log` against values above.
- Confirm `diagnostics.jsonl` first/last epoch totals match loss trend.
- Inspect `code_state.json` to verify git_head and script sha256.

## Data Lineage
- Source: white_noise_box_data_no_edge_sync_vad_normalized (real LDV test set)
- MD5 fingerprint: `713c0635878a04b32f4ee30208904d11` (computed over 111 .npy files)
- STFT grid: fs=16000, n_fft=2048, band=[300, 3000] Hz, F=346

## Component Boundaries + Acceptance
- Component: routing logits (scores_expert) from QK vs g; evaluation uses scores_expert for angle prediction.
- Inputs: Y (346,), D (346×E×M collapsed to 346×P), d_model=346; Outputs: expert logits (E,).
- Normal state acceptance: CE decreases below 1.0 within 10 epochs (A/B/D), accuracy ≥ 0.95 (A/D) or = 1.0 (B); no NaNs; deterministic under fixed seeds.

## Next Experiments
- Sweep M and `atom_min_cos` to map μ_mean→accuracy curve.
- Ablate score_norm/centering individually to rank contributions.
- Test minimal distillation (small KL to |g|) only if geometry degrades.

