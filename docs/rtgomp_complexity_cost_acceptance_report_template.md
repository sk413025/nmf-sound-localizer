# Acceptance Report Template: RTG-OMP (Complexity Cost)

This template defines the **required deliverable** after implementing and running the RTG-OMP complexity-cost experiment. It is designed to be:
- **Reproducible** (exact commands + subset manifest + fingerprints)
- **Verifiable** (numbers can be recomputed from artifacts without trusting prose)
- **Comparable** to the RTG-ineffective baseline

Save the filled report at: `results/<run_name>/ACCEPTANCE_REPORT.md`

---

## 1) Executive Summary (≤12 lines)

- Goal: Make RTG influential by encoding `lambda_cost` in the teacher (penalty-OMP).
- Outcome: `PASS` / `FAIL` (and why).
- Key metrics (report exact numbers):
  - Teacher Spearman: `rho(lambda_c, steps_used_mean) = ____`
  - Student sensitivity: `max(action_change_rate_vs_ref) = ____`, `max(logits_kl_mean_vs_ref) = ____`
  - Trade-off: `range(steps_used_mean) = ____`, `range(final_capture_mean) = ____`
- Baseline comparison (same subset): describe the delta vs baseline RTG grid (should be near-zero there).

---

## 2) Version, Environment, Repro Metadata

- `git_head`: `<hash>`
- Working tree: `clean` / `dirty` (if dirty, list modified files)
- Conda env: `trl-training`
- Python: `python --version` output
- Device: `mps` / `cpu`
- Seeds:
  - Data generation seed: `____`
  - Training seed: `____`
  - Eval seed: `____`

---

## 3) Data Lineage (Real Data Only)

- Mic root: `<path>`
- LDV root: `<path>`
- Subset manifest: `results/<run_name>/subset_manifest.json`
- Subset selection procedure (must match manifest): `<describe>`
- Fingerprint command + expected output:
  - Command: `<paste>`
  - Expected fingerprint: `<hash>`

---

## 4) Experiment Configuration (What was fixed)

Report the exact values used:
- `Tw`: `____`
- `max_lag`: `____` → `M = 2*max_lag+1 = ____`
- `K_max`: `____`
- `gain`: `____`
- `teacher_mode`: `penalty_omp`
- `lambda_c_values`: `[____]` (must match training + eval)
- `min_k`: `____`
- RTG semantics:
  - `rtg0`: `lambda_cost_logc_norm`
  - `rtg1`: `remaining_steps_fraction` (if used)

---

## 5) Exact Commands (Copy/Paste Reproduction)

### 5.1 Generate teacher trajectories
```bash
<paste exact command>
```

Expected artifacts:
- `results/<run_name>/data/lag_trajectories.pt`

### 5.2 Train student (DT)
```bash
<paste exact command>
```

Expected artifacts:
- `results/<run_name>/model/dt_freq_aware_best.pth`
- `results/<run_name>/train/diagnostics.json`

### 5.3 Evaluate RTG/λ grid
```bash
<paste exact command>
```

Expected artifacts:
- `results/<run_name>/eval/lambda_grid.json`

---

## 6) Artifact Index (Paths must exist)

- Teacher:
  - `results/<run_name>/data/lag_trajectories.pt`
- Training:
  - `results/<run_name>/train/run.log`
  - `results/<run_name>/train/diagnostics.json`
  - `results/<run_name>/model/dt_freq_aware_best.pth`
- Evaluation:
  - `results/<run_name>/eval/lambda_grid.json`
  - (Optional) `results/<run_name>/eval/plots/`
- Provenance:
  - `results/<run_name>/run.log`
  - `results/<run_name>/subset_manifest.json`

---

## 7) Acceptance Checks (Pass/Fail Table)

Fill this table with numbers **copied from `lambda_grid.json` and/or the checker output**.

| Check | Expected | Observed | Pass? |
|---|---|---:|:---:|
| Teacher monotonicity | `rho <= -0.6` (recommended) | `____` | `PASS/FAIL` |
| Teacher non-degenerate | not all `steps_used_mean` equal | `min=____ max=____` | `PASS/FAIL` |
| Student RTG sensitivity | `max(action_change_rate_vs_ref) >= 0.05` | `____` | `PASS/FAIL` |
| Student logit shift | `max(logits_kl_mean_vs_ref) > 0` | `____` | `PASS/FAIL` |
| Trade-off exists | not a single point | `Δsteps=____ Δcap=____` | `PASS/FAIL` |

---

## 8) Baseline Comparison (RTG-ineffective)

Specify the baseline artifacts you compare against and why they qualify:
- Baseline commit: `205a6ae`
- RTG-ineffective evidence commit/artifact: `d843ed3` (or an equivalent reproduced baseline run)

Baseline artifact(s) to cite:
- `<path to baseline rtg grid json>`

Comparison summary:
- Baseline `action_change_rate` and `logits_KL` should be near-zero across RTG sweeps.
- New method should show clear deviation above baseline noise floor.

---

## 9) Verification: How to Confirm This Report is Correct

### 9.1 Run the automated checker
```bash
PYTHONPATH=. python scripts/h_exploration/check_rtgomp_acceptance.py \
  --lambda_grid results/<run_name>/eval/lambda_grid.json \
  --out_json results/<run_name>/eval/acceptance_check.json
```

Attach:
- `results/<run_name>/eval/acceptance_check.json`

### 9.2 Independent recomputation (no re-training)
- Recompute key metrics by re-parsing `lambda_grid.json` and verifying:
  - Spearman `rho(lambda_c, steps_used_mean)`
  - `max(action_change_rate_vs_ref)`, `max(logits_kl_mean_vs_ref)`

### 9.3 Full rerun reproducibility (strongest)
- Re-run the exact commands in Section 5 on the same machine/env and confirm:
  - The directionality and acceptance checks remain consistent
  - Any differences are within stated tolerances

---

## 10) Failures, Root Cause, Next Steps (Required even if PASS)

If any check fails:
- Root cause hypothesis (use BECAUSE / THEREFORE)
- Evidence (link to specific artifacts/log lines)
- Minimal fix (smallest change likely to make it pass)
- Fundamental fix (if minimal fails)

If PASS:
- What changed relative to baseline BECAUSE of the RTG-conditioned teacher design?
- Next experiment to strengthen evidence (e.g., larger subset, more lambda values, or add lookahead if early actions still unchanged).

