# Spec: RTG-OMP via Complexity Cost (Model Selection Penalty)

This is the **canonical, self-contained engineering spec** for making RTG influential by modifying the **teacher/oracle (OMP)** to solve a **family** of physically meaningful objectives:

\[
\min_{S,w}\ \|y - D_S w\|_2^2 + \lambda |S|
\]

The key idea is to make RTG encode the **complexity penalty** `λ` (or its normalized form), so the teacher’s optimal behavior becomes RTG-dependent. This yields datasets where `P(a|s,RTG)` is not reducible to `P(a|s)` and the student can learn RTG usage.

This spec is written so an engineer can implement it without reading any other document.

---

## 0. Repo / Baselines / Constraints

### 0.1 Baseline commits (for reproducible comparison)
- Baseline start (recommended branching point): `205a6ae`
- RTG-ineffective evidence point: `d843ed3`

### 0.2 Existing scripts (current pipeline)
- Teacher trajectory generation: `scripts/h_exploration/generate_lag_omp.py`
- Student training: `scripts/h_exploration/train_dt_lag_seq_rtg.py`
- RTG override eval (baseline): `scripts/h_exploration/run_rtg_override_grid_eval.py`
- Smoke script (baseline): `scripts/h_exploration/run_smoke_gru2_tw32_lag50_k16.sh`

### 0.3 Output location policy
All new artifacts MUST be written under `results/<run_name>/...`. No outputs at repo root.

---

## 1. Problem Statement / Non-Goals

### 1.1 Problem statement
We want RTG to measurably affect decisions in the lag-selection pursuit problem. In baseline deterministic OMP, the next action is effectively a function of the current correlation state, so the student can legally ignore RTG.

We will introduce a physically interpretable cost of complexity (number of selected lag paths), yielding an RTG-conditioned teacher:
- high cost → stop earlier / prefer only high-gain atoms
- low cost → continue longer / accept smaller gains

### 1.2 Success definition (must be testable)
We consider RTG “effective” if, on a fixed real-data subset:
1) **Teacher**: `steps_used` is a monotone function of RTG (λ), and varies substantially across RTG values.
2) **Student**: RTG sweeps induce a measurable change in actions/logits, beyond baseline noise.
3) **Task trade-off**: `(final_capture, steps_used)` shows a Pareto curve across λ values (not a single saturated point).

### 1.3 Non-goals (for this iteration)
- Not optimizing absolute best capture; first goal is **RTG controllability** + reproducibility.
- Not redesigning the full model architecture; keep changes minimal.
- Not changing dataset roots or adding synthetic data (real data only).

---

## 2. Definitions (Symbols, Units, Shapes)

This section defines the exact quantities used by the teacher and student.

### 2.1 Windowed per-frequency formulation
For each clip, for each window start `t0`, for each frequency bin `f`:

- Target vector: `y ∈ C^{Tw}` (LDV STFT frames at freq `f`, within window `Tw`)
- Dictionary: `D ∈ C^{Tw×M}` (Mic STFT frames shifted by lag candidates)
  - `M = 2*max_lag + 1`
  - each column corresponds to one lag atom
- Selected set: `S ⊂ {0..M-1}`
- LS projection: `w_hat = argmin_w || y - D_S w ||_2^2`
- Residual: `r = y - D_S w_hat`

### 2.2 Energies (absolute, not ratio-based)
- Residual energy: `E(r) = ||r||_2^2` (units: squared STFT magnitude-sum over time)
- Initial energy: `E0 = E(y)`

### 2.3 Marginal gain (the teacher’s decision statistic)
At step `k`, let `S_k` and residual `r_k`. If we add an atom and refit by LS:
- `E_k = E(r_k)`
- `E_{k+1} = E(r_{k+1})`
- **Marginal gain**: `ΔE_k = E_k - E_{k+1}` (units: energy)

### 2.4 Capture (for reporting only)
Capture is reported as:
- `capture = 1 - E_res_final / max(E0, eps_energy)`
where `eps_energy` is a tiny stabilizer used ONLY for reporting, not for teacher decisions.

### 2.5 Complexity cost (RTG semantics)
We define a family of objectives:
\[
\min_{S,w}\ E(y - D_S w) + \lambda |S|
\]
- `λ` has units of energy.
- RTG encodes `λ` (or a normalized representation of `λ`).

We implement greedy pursuit approximating this objective via a threshold on `ΔE`.

---

## 3. Teacher Algorithm Spec (Penalty-OMP with STOP)

### 3.1 High-level behavior
For each `(clip, window, freq bin)` we generate a trajectory under a chosen λ:
- iteratively select lag atoms (like OMP)
- stop early when “adding another path is not worth it” given λ

This introduces **variable horizon** and provides RTG-dependent supervision.

### 3.2 Required teacher modes
Add a new `--teacher_mode` to `scripts/h_exploration/generate_lag_omp.py`:
- `omp` (baseline behavior; deterministic argmax for exactly `K_max` steps)
- `penalty_omp` (new; STOP based on complexity cost λ)

Default MUST remain `omp` to preserve baseline compatibility.

### 3.3 Decision rule (penalty_omp)
At each step `k` (0-indexed), do:
1) Compute correlations from the current residual:
   - `abs_corrs = |D_norm^H r_k|` with masking of already-selected atoms.
2) Choose the greedy atom:
   - `a_k = argmax(abs_corrs)`
3) Refit and update residual:
   - solve LS on selected atoms and update `r_{k+1}`
4) Compute marginal gain:
   - `ΔE_k = E(r_k) - E(r_{k+1})`
5) STOP rule:
   - if `k >= min_k` AND `ΔE_k <= λ`: stop after this step **OR** stop before taking this step (choose one; see 3.4)

### 3.4 STOP timing (must be consistent)
Choose one STOP convention and implement it consistently across:
- teacher generation
- student labels
- evaluation

**Spec choice (recommended): “Stop-after-evaluating-current-step”**
- We always take the greedy action at step `k`, update residual, compute `ΔE_k`.
- If `ΔE_k <= λ`, then we emit a STOP token at step `k+1` and terminate.

Rationale:
- ensures `ΔE_k` is well-defined and logged for the selected action
- avoids needing candidate-level `ΔE(j)` for all `j` in the first iteration

### 3.5 Numerical stability requirements
Teacher must be robust to ill-conditioning:
- LS solution uses `torch.linalg.lstsq`.
- If `ΔE_k < 0` due to numeric issues, clamp to `0` for stopping logic and logging.
- Always mask repeated selections (no duplicates in S).

---

## 4. Trajectory Data Spec (torch.save `.pt` schema)

### 4.1 File format and location
Teacher output is a `torch.save(list_of_blocks)` file:
- Path: `results/<run_name>/data/lag_trajectories.pt`

Each element `block` is a dict for one `(clip_idx, window_start)` containing all frequency bins as a batch.

### 4.2 Required keys (backward compatible)
To keep existing scripts working, the following keys MUST remain:
- `corrs`: `torch.float16` of shape `(F, K, M)` where:
  - `F`: number of frequency bins
  - `K`: sequence length for that block (variable)
  - `M`: number of lag atoms
- `actions`: integer tensor of shape `(F, K)` containing action ids

### 4.3 New required keys for penalty-OMP
For `teacher_mode=penalty_omp`, add:
- `lambda_abs`: `torch.float32` of shape `(F,)` or scalar broadcastable
  - the actual λ (energy units) used for the trajectory
- `lambda_c`: `torch.float32` of shape `(F,)` or scalar
  - the dimensionless scaling coefficient `c` used in `λ = c * E0` (see 5.2)
- `E0`: `torch.float32` of shape `(F,)`
  - initial energy per frequency bin for that window
- `E_res`: `torch.float32` of shape `(F, K)`
  - residual energies per step (after each LS update)
- `deltaE`: `torch.float32` of shape `(F, K)`
  - marginal gains per step (aligned with `actions` steps)
- `valid_len`: integer tensor of shape `(F,)`
  - number of valid steps for each frequency bin within the block

### 4.4 STOP token labeling (required for RTG controllability)
To let the student learn when to stop, we include an explicit STOP action:
- define STOP action id: `STOP_ID = M`
- therefore action space size: `M + 1`

Encoding rule:
- For each frequency bin sequence, append one STOP action at the end.
- Example: if a bin stops after selecting `k_stop` lag atoms, its `actions` is length `k_stop + 1` with the last token being `STOP_ID`.

Implementation detail:
- `corrs` at the STOP step can be a zero vector OR a repeat of the last correlation state; choose zero for simplicity and document it.

### 4.5 Padding/masking (training-time)
Variable-length sequences are padded in the collate function:
- pad `actions` with `-100` (ignore_index)
- pad `corrs` with `0.0`
- pad `rtg` with `0.0`

---

## 5. Teacher λ Definition (RTG→λ mapping)

### 5.1 Design requirement
`λ` must be comparable to `ΔE` (both are energies). Avoid ratio-based teacher decisions that can be dominated by arbitrary epsilons.

### 5.2 First-iteration choice (recommended)
Use **relative-to-signal** scaling per `(freq, window)`:
- `λ = c * E0`

where:
- `E0 = ||y||^2` per frequency bin/window
- `c` is a dimensionless coefficient controlled by RTG

### 5.3 λ sweep values (must be in-distribution)
Teacher generation must sweep multiple `c` values so the student sees RTG variation:
- default sweep (log-scale): `c ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`

This list must be configurable via CLI:
- `--lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2"`

### 5.4 min_k (avoid trivial immediate STOP)
Add CLI:
- `--min_k` (default `1`)

Meaning:
- do not allow STOP before at least `min_k` lag atoms have been selected

---

## 6. Student Training Spec (Learning RTG-dependent behavior)

### 6.1 RTG semantics for the student
We redefine RTG0 to encode `c` (or `λ`) rather than “remaining energy ratio”.

For each trajectory:
- `rtg0 = normalize_log_c(c)` (constant across all steps in that sequence)
- `rtg1 = remaining_steps_fraction = (K_max - k) / K_max` (optional; keep to match existing 2D RTG interface)

**Normalization spec (must be deterministic):**
- parse `c` as float
- compute `logc = log10(c)`
- choose fixed bounds from the configured sweep:
  - `logc_min = log10(min(c_values))`
  - `logc_max = log10(max(c_values))`
- map to `[0,1]`:
  - `rtg0 = (logc - logc_min) / (logc_max - logc_min)`
- clamp `[0,1]`

### 6.2 Model output dimension
Because we add STOP:
- output logits dimension MUST be `M + 1`
- STOP token id is `M`

### 6.3 Training loss / masking
Use cross-entropy with padding ignore index:
- actions padded with `-100`
- `nn.CrossEntropyLoss(ignore_index=-100)`

### 6.4 Required modifications to training script
Modify `scripts/h_exploration/train_dt_lag_seq_rtg.py`:
- add `--rtg_mode lambda_cost` (new choice)
- add `--lambda_c_values` (string list) for normalization bounds
- update dataset loader to read `block["lambda_c"]` (or infer from `lambda_abs/E0`)
- update model head to output `M+1` if `--use_stop_action` is enabled
- log RTG ablations and grad norms (see 6.5)

### 6.5 Required training-time diagnostics (to rule out “dead RTG”)
Each training run MUST write:
- `results/<run_name>/train/diagnostics.json`
containing at least:
- `rtg_embed_grad_norm_mean`
- `state_embed_grad_norm_mean`
- `freq_embed_grad_norm_mean`
- `rtg_ablation_metrics`:
  - train/val loss when RTG is intact
  - train/val loss when RTG0 is shuffled within batch
  - (optional) action_change_rate on a small held-out batch under RTG sweep

Acceptance intuition:
- if teacher behavior depends on RTG, shuffling RTG should worsen imitation.

---

## 7. Evaluation Spec (Making RTG effect observable)

### 7.1 Why baseline eval is insufficient
If evaluation always runs to `K_max` and reports only final capture, then RTG effects that primarily control **stopping** can be washed out.

We require evaluation to include:
- behavior-sensitive metrics
- cost/quality trade-off metrics

### 7.2 New evaluation script (recommended)
Add a new script:
- `scripts/h_exploration/run_lambda_override_grid_eval.py`

Purpose:
- sweep `rtg0` as lambda_cost in-distribution values
- optionally sweep `rtg1`
- evaluate both “actions/logits sensitivity” and “capture vs steps”

### 7.3 Required metrics (definitions)
Let `λ_ref` be a reference RTG setting (e.g., smallest `c`).

For each evaluated window/frequency (or aggregated batch), compute:

1) `action_change_rate`
- Compare argmax action sequences between `λ` and `λ_ref` for the same inputs.
- Definition: fraction of positions where `a_k(λ) != a_k(λ_ref)` over valid (non-pad) positions, excluding positions after STOP in each sequence.

2) `logits_kl_mean`
- Compute softmax distributions `p_k(λ)` and `p_k(λ_ref)` over actions at each step.
- `KL_k = KL(p_k(λ_ref) || p_k(λ))` (choose direction and keep consistent)
- Report mean over valid steps and items.

3) `steps_used_mean`
- Number of non-STOP lag selections until STOP (teacher-style horizon).

4) `final_capture_mean`
- Compute capture using absolute energies:
  - `capture = 1 - E_res_final / max(E0, eps_energy)`
  - `eps_energy` is for reporting only; must be logged.

### 7.4 Output schema (JSON)
Write:
- `results/<run_name>/eval/lambda_grid.json`

Schema:
```json
{
  "config": {
    "run_name": "...",
    "git_head": "...",
    "subset_manifest_path": "...",
    "mic_root": "...",
    "ldv_root": "...",
    "tw": 32,
    "max_lag": 50,
    "K_max": 16,
    "gain": 100.0,
    "lambda_c_values": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "rtg0_semantics": "lambda_cost_logc_norm",
    "rtg1_semantics": "remaining_steps_fraction",
    "eps_energy": 1e-12,
    "seed": 0
  },
  "grid": [
    {
      "lambda_c": 0.0001,
      "rtg0": 0.0,
      "summary": {
        "steps_used_mean": 12.3,
        "final_capture_mean": 0.93,
        "action_change_rate_vs_ref": 0.18,
        "logits_kl_mean_vs_ref": 0.42
      }
    }
  ]
}
```

---

## 8. CLI Spec (Flags, Defaults, Example Commands)

### 8.1 Teacher generation (`generate_lag_omp.py`)
Add flags:
- `--teacher_mode {omp,penalty_omp}` (default `omp`)
- `--lambda_c_values "<csv>"` (required for `penalty_omp`)
- `--min_k <int>` (default `1`)
- `--out_dir` (already exists; required)

Example:
```bash
PYTHONPATH=. python scripts/h_exploration/generate_lag_omp.py \
  --mic_root <REAL_MIC_ROOT> \
  --ldv_root <REAL_LDV_ROOT> \
  --all_angles \
  --tw 32 --max_lag 50 --max_k 16 --gain 100 \
  --teacher_mode penalty_omp \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --min_k 1 \
  --out_dir results/<run_name>/data
```

### 8.2 Training (`train_dt_lag_seq_rtg.py`)
Add flags:
- `--rtg_mode {teacher_final,target1,lambda_cost}` (default remains `target1`)
- `--lambda_c_values "<csv>"` (required when `rtg_mode=lambda_cost`)
- `--use_stop_action` (bool; required for RTG controllability via STOP)

Example:
```bash
PYTHONPATH=. python scripts/h_exploration/train_dt_lag_seq_rtg.py \
  --data_path results/<run_name>/data/lag_trajectories.pt \
  --out_dir results/<run_name>/model \
  --epochs 50 --batch_size 256 --lr 1e-3 --seed 0 \
  --rtg_dim 2 \
  --rtg_mode lambda_cost \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --use_stop_action
```

### 8.3 Evaluation (`run_lambda_override_grid_eval.py`)
Example:
```bash
PYTHONPATH=. python scripts/h_exploration/run_lambda_override_grid_eval.py \
  --mic_root <REAL_MIC_ROOT> --ldv_root <REAL_LDV_ROOT> \
  --ckpt_path results/<run_name>/model/dt_freq_aware_best.pth \
  --subset_manifest results/<run_name>/subset_manifest.json \
  --out_dir results/<run_name>/eval \
  --tw 32 --max_lag 50 --max_k 16 --gain 100 \
  --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
  --num_clips 3
```

---

## 9. Acceptance Criteria (Pass/Fail)

### 9.1 Teacher acceptance (must pass before training)
On a fixed subset:
- `steps_used` varies across `lambda_c_values`
- monotonic trend: higher `c` → fewer steps
- recommended check:
  - Spearman correlation `ρ(lambda_c, steps_used_mean) <= -0.6`

### 9.2 Student RTG sensitivity acceptance
On in-distribution λ sweep:
- `action_change_rate_vs_ref` significantly > baseline noise
  - target: `>= 0.05` (5%) on the evaluation subset
- `logits_kl_mean_vs_ref` significantly > 0
  - target: `>= 0.05` (scale depends; use baseline relative gap)

### 9.3 Task trade-off acceptance
Across λ sweep, `(steps_used_mean, final_capture_mean)` must not collapse to a single point:
- lower cost (small `c`) should yield higher capture and/or more steps
- higher cost (large `c`) should yield fewer steps and typically lower capture

---

## 10. Failure Modes & Debug Playbook

### 10.1 All sequences stop immediately
Symptoms:
- `steps_used_mean` near `min_k` for all λ
Likely causes:
- `c` too large
Fix:
- reduce `c` sweep range by 10×, or increase `min_k`

### 10.2 All sequences run to `K_max`
Symptoms:
- `steps_used_mean` near `K_max` for all λ
Likely causes:
- `c` too small
Fix:
- increase `c` sweep range by 10×

### 10.3 Teacher varies steps, but student ignores RTG
Symptoms:
- teacher `steps_used` monotone, but student `action_change_rate` and `KL` ~ 0
Likely causes:
- RTG fed to student does not match teacher semantics (wrong normalization or wrong field read)
- STOP token not implemented (student cannot express “stop”)
Fix:
- verify `rtg_mode=lambda_cost` is used and reads `lambda_c`
- ensure output head includes STOP (`M+1`)
- ensure STOP token exists in labels

### 10.4 RTG sweep changes output but only because of OOD
Symptoms:
- large sensitivity but only at RTG values not seen in training
Fix:
- evaluate only in-distribution `lambda_c_values`
- include the evaluation sweep values in training sweep

### 10.5 Metric artifacts (epsilon domination)
Symptoms:
- capture behaves inconsistently in low-energy bands
Fix:
- teacher decisions must use absolute energies `E0/E_res/deltaE` only
- log `E0` distribution and avoid using `+1e-6` denominators for decision logic

