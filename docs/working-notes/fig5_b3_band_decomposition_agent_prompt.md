# Agent Prompt: Implement Fig5 B3 Band Decomposition (Equivalence-Preserving)

You are implementing a frequency-band decomposition for **Figure 5 panel B3** that is **mathematically reconstructible** back to the original full-band quantities.

This task is interpretability-critical: do **not** change the meaning of the plotted quantities.

## Read first (source of truth)
Open and follow: `docs/working-notes/fig5_b3_band_decomposition_spec.md`

That spec defines:
- the exact quantities to decompose,
- the correct-by-construction math,
- band definitions,
- guardrails,
- required outputs,
- acceptance tolerances.

## Why this exists (causal context)
- v1/v3 B3 uses **model-native** `scores_expert` (Transformer QK routing over angles) vs **physics** `g_energy_expert`.
- v9/v10 introduced banded plots by switching the green curve to a **projection-based proxy**; that breaks equivalence to the original B3.
- We want banded views **without redefining the quantity**:
  - physics baseline must be decomposed with a sign-aligned rule (equivalent to `g_energy_expert`)
  - model-native score must be decomposed via attribution (Integrated Gradients) to be additive across frequency bins/bands.

## Task (deliverables)

### 1) Add a new script
Create: `scripts/fig5_b3_band_decomposition.py`

It must:
- accept `--run_dir` (default: `results/omp_transformer_speech260_trainval_split_full_20251115_082341`)
- accept `--out_dir` (default: `results/fig5_b3_band_decomp_<timestamp>/`)
- accept `--device` (`cpu|mps|cuda`, default: `cpu`)
- accept `--n_steps` for IG (default: `2048`)
- accept `--ig_baseline` (`mean_val|zero`, default: `mean_val`)
- accept `--ig_method` (`trapezoid|simpson`, default: `trapezoid`)
- accept `--sample_idx` (optional; if omitted, auto-select using the same representative-case logic as `generate_figure5_atomic.py`)
- accept `--smooth_window` (odd int; default: `1` meaning off; visualization only)
- accept `--smooth_pad` (`reflect|edge`, default: `reflect`; visualization only)
- accept `--compare_norm` (`per_curve_minmax|shared_minmax|none`, default: `per_curve_minmax`; visualization only)

Hard constraints:
- Never write artifacts to repo root; everything goes under `--out_dir`.
- No fallbacks: if shapes/angles/frequency grid/bands don’t match spec, raise with a clear error.
- Use float64 for reconstruction checks (store float32 outputs only if you also store float64 checks).
- Model must run in `eval()` for IG (dropout off).

### 2) Implement two decompositions (exact methods)
Implement exactly as in `docs/working-notes/fig5_b3_band_decomposition_spec.md`:

#### A) Physics baseline (`g_energy_expert`) decomposition
For a chosen sample `y`:
1) Compute `g_full = D.T @ y` (P,)
2) For each band mask `Bk`, compute `g_k = D[Bk,:].T @ y[Bk]`
3) Convert to equivalent decomposition of `|g_full|` using sign alignment:
   - `c_k = sign(g_full) * g_k` (P,)
   - reshape to `(E,M)` and sum over atoms -> `G_k` (E,)
4) Validate: `g_energy_full == sum_k G_k` within tolerance.

#### B) Model-native `scores_expert` decomposition via Integrated Gradients (IG)
Goal: decompose the **step-0** `scores_expert` vector across frequency bands while preserving reconstruction.

Requirements:
- Rebuild the trained model from `run_dir/code_state.json` and load `run_dir/model_best.pth`.
  - Reuse patterns from `scripts/eval_omp_transformer_split.py`.
- Use `model.eval()`.
- Extract step-0 `scores_expert` exactly like `scripts/visualize_omp_transformer_routing.py`:
  - call `model(y, D, train_mode=True)`
  - read `model.last_outputs['scores_expert']`

IG procedure (single sample):
- Baseline (recommended, fixed): `y0 = unit_norm(mean(Y_val))` computed from `modal_routing_val.npz:Y_val`
- For t=1..n_steps:
  - `alpha = t/n_steps`
  - `y_alpha = y0 + alpha*(y - y0)` with `requires_grad_(True)`
  - run forward and get `se_alpha` (E,)
  - compute gradients `∂se_alpha[e]/∂y_alpha` for all `e` (E times)
  - accumulate average gradient per expert
- Compute per-bin IG: `IG[e,f] = (y[f]-y0[f]) * avg_grad[e,f]`
- Band sums: `IG_band[k,e] = sum_{f in Bk} IG[e,f]`
- Reconstruction: `se_full[e] ≈ se_base[e] + sum_k IG_band[k,e]`

Validation:
- Must meet the reconstruction tolerances in the spec.

### 3) Write artifacts and plots
Under `--out_dir`, write:
- `run.log` (unbuffered logging recommended)
- `band_defs.json` (bin indices per band)
- `decomposition.npz` containing all arrays required by the spec
- `checks.json` with reconstruction errors
- `code_state.json` binding artifacts to executed code + inputs (git head/dirty + SHA256)
- plots:
  - Five comparison line plots (each includes QK + PHYSICS + True DoA marker):
    - `Fig5_B3_LINE_300_3000.pdf`
    - `Fig5_B3_LINE_300_500.pdf`
    - `Fig5_B3_LINE_500_1000.pdf`
    - `Fig5_B3_LINE_1000_2000.pdf`
    - `Fig5_B3_LINE_2000_3000.pdf`

Plot labeling must be explicit:
- Physics band curves: “sign-aligned contribution (Σ bands reconstructs g_energy_expert)”
- QK band curves: “Integrated Gradients attribution (Δ; base + Σ bands reconstructs scores_expert)”

## Execution instructions (must be included in logs)
Use the project environment conventions:

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace:$PYTHONPATH

RUN="results/fig5_b3_band_decomp_$(date +%Y%m%d_%H%M%S)"
python -u scripts/fig5_b3_band_decomposition.py \
  --run_dir results/omp_transformer_speech260_trainval_split_full_20251115_082341 \
  --device cpu \
  --ig_baseline mean_val \
  --ig_method trapezoid \
  --n_steps 2048 \
  --out_dir "$RUN"
```

If MPS is available on your machine, you may switch `--device mps` for speed.

Optional: capture the full stdout/stderr stream (the script also writes `$RUN/run.log`):

```bash
mkdir -p "$RUN"
PYTHONUNBUFFERED=1 python scripts/fig5_b3_band_decomposition.py \
  --run_dir results/omp_transformer_speech260_trainval_split_full_20251115_082341 \
  --device cpu \
  --ig_baseline mean_val \
  --ig_method trapezoid \
  --n_steps 2048 \
  --out_dir "$RUN" \
  2>&1 | tee "$RUN/stdout_stderr.log"
```

## Acceptance (definition of “done”)
All items in `docs/working-notes/fig5_b3_band_decomposition_spec.md` **Acceptance checklist** must pass, especially:
- Physics reconstruction error < 1e-6 (float64)
- IG reconstruction error < 1e-3 with `ig_baseline=mean_val`, `ig_method=trapezoid`, and `n_steps>=2048`
- Outputs are correctly labeled to avoid the v9/v10 interpretability trap.

## Explicit “don’t do this” list (pitfalls)
- Do NOT compute `abs()` inside each band and sum if you claim equivalence to `g_energy_expert`.
- Do NOT call the v10 proxy “model-native QK”; it is not `scores_expert`.
- Do NOT renormalize `D` per band.
- Do NOT use overlapping bands or bands that don’t cover `[300,3000]`.
- Do NOT write any artifacts to repo root.
