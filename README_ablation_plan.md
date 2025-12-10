# OMP / Transformer Ablation Plan (exp/omp-ablation-20251209)
## Background
- Baseline: commit `af6a483` (Speech260 val-split, seeds 1–5, dataset MD5 `f563984848ae49b4443378c4ef720a51`, 20 epochs, MPS) reached val accuracies 0.946–0.975 with seed-aware eval.
- Motivation: quantify which components are essential for accuracy (Transformer, OMP selection, shared heads) and identify any redundant capacity.

## Objectives
- Measure accuracy deltas when freezing/removing components while keeping data, seed handling, and logging identical to the baseline run.
- Enforce “no-fallback” policy: exact angle matches, identical STFT grids, no silent coercions; abort on mismatches.

## Candidate Ablations (one run per row)
- Baseline rerun: identical to `af6a483` to verify reproducibility on this machine.
- Freeze Transformer weights: train only non-Transformer modules (optimizer scope excludes Transformer parameters).
- Freeze OMP selection parameters: lock OMP scoring/reduction module; train remaining parts.
- Freeze output heads only: lock classifier/projection heads; train backbone + OMP.
- Disable OMP sparsity step: replace selection with pass-through (if code path exists); otherwise add a guarded flag and fail fast if unsupported.

## Smoke & Functional Tests (real data only)
- Smoke: minimal subset (e.g., Speech260 val split angles {0, 30, 60}, 1 clip/angle, seed 42) must run end-to-end on MPS, print device, finish without errors; log to `results/<run>/run.log`.
- Functional: full val-split evaluation with seed reuse in atom reduction; assert non-degenerate accuracy (>0.1) and no NaN; include numeric diagnostics JSONL.
- Data fingerprint: `find . -name "*.npy" -exec md5sum {} \; | sort | md5sum > results/<run>/data_fingerprint.txt`.
- Subset manifest: record angles, clip ids, and seed in `results/<run>/subset_manifest.json`.

## Command Templates (exact values to fill per run)
```bash
# Env
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209:$PYTHONPATH

# Training (example: freeze Transformer)
RUN_DIR=results/omp_ablate_freeze_transformer_seed42_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN_DIR"
conda run -n trl-training python -u scripts/omp-transformer-ldv.py \
  --accelerator mps \
  --seed 42 \
  --out_dir "$RUN_DIR" \
  --freeze_transformer true \
  --use_angle_pairs \
  > "$RUN_DIR/run.log" 2>&1

# Evaluation (seed-aware atom reduction)
conda run -n trl-training python -u scripts/eval_omp_transformer_split.py \
  --run_dir "$RUN_DIR" \
  --device mps \
  --subset both \
  --seed 42 \
  > "$RUN_DIR/posthoc_eval.log" 2>&1

# Fingerprint + manifest (after data load decision)
find /Users/sbplab/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad_normalized -name "*.npy" -exec md5sum {} \; | sort | md5sum > "$RUN_DIR/data_fingerprint.txt"
```

## Logging Requirements
- Persist numeric diagnostics to `results/<run>/numeric_diagnostics.jsonl` (STFT grid, IS divergence deltas, mix stats, baseline_k, ratio quantiles).
- Save `code_state.json` with `git_head`, `dirty`, and SHA256 of executed files.
- Capture acceptance flags (smoke/functional pass) in `results/<run>/summary.json`.

## Acceptance to Commit Results
- Each ablation run must include: code + artifacts (logs, metrics, diagnostics, fingerprints, subset manifest).
- Document physical/mathematical analysis, cross-experiment comparisons (≥3 experiments with hashes), extracted principles, and reproduction steps before committing.
