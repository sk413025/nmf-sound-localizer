# Acceptance Report: RTG-OMP (Complexity Cost) — Smoke Run

## 1) Executive Summary

- Goal: Make RTG influential by encoding `lambda_cost` in the teacher (penalty-OMP) and observe RTG-dependent behavior in the student.
- Outcome: **FAIL (partial success)**.
- Key numbers:
  - Teacher monotonicity (from trajectories): **Spearman ρ = -1.00** (PASS)
- Student RTG sensitivity: **max(action_change_rate_vs_ref) = 0.5421**, **max(logits_kl_mean_vs_ref) = 2.0174** (PASS)
- Eval steps trend: **steps_used_mean increases with lambda_c** (expected to decrease; FAIL)
- Interpretation: Teacher is correct (higher λ → fewer steps), but **student evaluation shows reversed step trend**, likely due to RTG1 mismatch and/or RTG0 mapping direction in eval.

---

## 2) Version, Environment, Repro Metadata

- `git_head`: `9bafdc61a6927892467c56592f31da6838fb3e25`
- Working tree: **dirty**
  - Modified: `scripts/h_exploration/generate_lag_omp.py`, `scripts/h_exploration/train_dt_lag_seq_rtg.py`
  - Added: `scripts/h_exploration/run_lambda_override_grid_eval.py`, `scripts/h_exploration/check_rtgomp_acceptance.py`
  - Added docs: `docs/rtgomp_complexity_cost_spec.md`, `docs/rtgomp_complexity_cost_acceptance_report_template.md`, etc.
- Conda env: `trl-training`
- Python: `3.12.2`
- Device: `mps`
- Seeds:
  - Data generation seed: `0`
  - Training seed: `0`
  - Eval seed: `N/A` (deterministic given data + model)

---

## 3) Data Lineage (Real Data Only)

- Mic root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- LDV root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Subset manifest: `results/rtgomp_lambda_cost_smoke_20260120_175600/subset_manifest.json`
- Selection: `first 3 clip pairs in dataset order (all angles)`
- Fingerprint: `1146bc6b77ff5272a458e5173b91abb2` (see manifest)

---

## 4) Experiment Configuration

- `Tw`: `32`
- `max_lag`: `50` → `M=101`
- `K_max`: `16`
- `gain`: `100.0`
- Teacher mode: `penalty_omp`
- `lambda_c_values`: `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2]`
- `min_k`: `1`
- RTG semantics:
  - `rtg0`: `lambda_cost_logc_norm`
  - `rtg1`: `remaining_steps_fraction`
- Eval: `num_clips=1` (reduced for runtime)

---

## 5) Exact Commands (Copy/Paste Reproduction)

### 5.1 Generate teacher trajectories
```bash
PYTHONPATH=. python -u scripts/h_exploration/generate_lag_omp.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --out_dir results/rtgomp_lambda_cost_smoke_20260120_175600/data \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --variants_per_clip 1 --max_items 3 --all_angles \
  --teacher_mode penalty_omp \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --min_k 1 --seed 0
```

### 5.2 Train student
```bash
PYTHONPATH=. python -u scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/rtgomp_lambda_cost_smoke_20260120_175600/data/lag_trajectories.pt \
  --out_dir results/rtgomp_lambda_cost_smoke_20260120_175600/model \
  --epochs 5 --batch_size 128 --lr 5e-4 \
  --rtg_dim 2 --rtg_mode lambda_cost \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --use_stop_action --seed 0
```

### 5.3 Evaluate lambda grid
```bash
PYTHONPATH=. python -u scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_smoke_20260120_175600/model/dt_freq_aware_best.pth \
  --subset_manifest results/rtgomp_lambda_cost_smoke_20260120_175600/subset_manifest.json \
  --out_dir results/rtgomp_lambda_cost_smoke_20260120_175600/eval \
  --hop_length 160 --max_lag 50 --max_k 16 --tw 32 --gain 100.0 \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --num_clips 1 --use_stop_action
```

### 5.4 Acceptance check
```bash
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid results/rtgomp_lambda_cost_smoke_20260120_175600/eval/lambda_grid.json \
  --out_json results/rtgomp_lambda_cost_smoke_20260120_175600/eval/acceptance_check.json
```

---

## 6) Artifact Index

- Teacher:
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/data/lag_trajectories.pt`
- Training:
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/model/dt_freq_aware_best.pth`
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/run.log`
- Evaluation:
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/eval/lambda_grid.json`
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/eval/acceptance_check.json`
- Provenance:
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/subset_manifest.json`
  - `results/rtgomp_lambda_cost_smoke_20260120_175600/train/diagnostics.json`

---

## 7) Acceptance Checks (Pass/Fail)

### 7.1 Teacher monotonicity (from trajectories)
- `lambda_c` vs `valid_len` (mean per λ):
  - `1e-4 → 14.01`, `3e-4 → 12.10`, `1e-3 → 10.01`, `3e-3 → 8.21`, `1e-2 → 6.35`
- Spearman ρ = **-1.00** → **PASS**

### 7.2 Student sensitivity (from `acceptance_check.json`)
- `max(action_change_rate_vs_ref) = 0.5421` → **PASS**
- `max(logits_kl_mean_vs_ref) = 2.0116` → **PASS**

### 7.3 Student step trend (expected: λ↑ → steps_used↓)
- Observed (from eval grid): `λ↑` → `steps_used_mean↑` (reverse of expected)
- Spearman (λ, steps) from eval grid: **+1.00**
- **FAIL** (inconsistent with teacher and spec expectation)

### 7.4 Trade-off existence
- `steps_range = 0.5001`, `capture_range = 0.00326` → **PASS**

Overall: **FAIL** due to reversed step monotonicity in student eval.

---

## 8) Baseline Comparison

Baseline RTG-ineffective evidence:
- Commit: `d843ed3`
- Artifact: `results/exp_interspeech_gru2_tw32_lag50_k16_ep50/rtg_grid/rtg0_rtg1_override_grid.json`

Comparison:
- Baseline RTG grid exhibits near-zero variation across RTG.
- New method shows strong **behavioral sensitivity** (action change + logits KL), but **step direction is reversed** in student eval.

---

## 9) Root Cause Analysis (Why FAIL)

Observed: Teacher trajectories correctly enforce **λ↑ → shorter** (valid_len decreases), but student eval shows the opposite (λ↑ → longer).

Most likely causes:
1) **RTG1 semantics mismatch**  
   - Training uses `remaining_steps_fraction` based on **sequence length**, while eval uses `remaining_steps_fraction` based on **fixed max_k**.  
   - This changes RTG1 distribution and can invert learned behavior.

2) **RTG0 directionality mismatch**  
   - We mapped `rtg0 = normalize(log10(lambda_c))`.  
   - If the model implicitly treats larger RTG0 as “higher target/quality”, it may extend steps for larger RTG0, opposing the intended “cost”.

3) **Short training (5 epochs)**  
   - Model may not fully learn the correct mapping of RTG to stopping policy, especially with STOP token.

Additional evidence (from `train/diagnostics.json`):
- `rtg_embed` grad norm mean: `0.0293` (non-zero)
- RTG ablation loss: `base=3.2166`, `shuffle=3.2465`, `zero=3.6487`
  - RTG is *used* (zeroing RTG worsens loss), so the issue is likely **semantic directionality**, not dead input.

---

## 10) Fix Plan (Next Steps)

Minimal fixes (next run):
1) **Align RTG1 semantics**  
   - Use `rtg_dim=1` (RTG0 only), or  
   - Change eval RTG1 to match training definition (`remaining_steps = (seq_len - k)/seq_len`), possibly by estimating `seq_len` from teacher or using a proxy.

2) **Flip RTG0 mapping**  
   - Define `rtg0 = 1 - normalize(log10(lambda_c))` so larger λ → smaller RTG0.  
   - This aligns with “higher RTG means more willing to act” semantics.

3) **Increase epochs to 10–20**  
   - Check whether student step monotonicity becomes correct with stronger training.

If these fixes fail:
- Increase STOP loss weight or enforce STOP prediction with a margin.
- Add teacher lookahead/risk penalty to make early action differences more learnable.
