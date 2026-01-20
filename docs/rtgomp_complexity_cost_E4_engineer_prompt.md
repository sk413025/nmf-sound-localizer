# Prompt: Execute E4 DAgger-lite (Exposure-Bias Mitigation)

You are the algorithm engineer. Execute **E4 DAgger-lite** end-to-end with **real data** and produce a results commit.

## Read these first
1) `docs/rtgomp_complexity_cost_spec.md`
2) `docs/rtgomp_complexity_cost_E4_plan.md`
3) `docs/rtgomp_complexity_cost_E4_spec.md`
4) `docs/rtgomp_complexity_cost_E4_acceptance_report_template.md`

## Objective
Reduce exposure bias so **free-rollout** monotonicity becomes correct:

```
lambda_c ↑  ⇒  steps_used_mean ↓  (target Spearman ρ <= -0.6)
```

## Hard requirements
- One commit = one executed experiment.
- Commit must include **code + artifacts + report**.
- All outputs under `results/<run_name>/`.
- No synthetic data.
- Fail fast on errors; no silent fallbacks.

## Required outputs
The run must produce:
- `results/<run_name>/run.log`
- `results/<run_name>/subset_manifest.json`
- `results/<run_name>/code_state.json`
- `results/<run_name>/data/lag_trajectories.pt`
- `results/<run_name>/data/dagger_trajectories.pt`
- `results/<run_name>/data/merged_trajectories.pt`
- `results/<run_name>/model/dt_freq_aware_best.pth`
- `results/<run_name>/train/diagnostics.json`
- `results/<run_name>/eval/lambda_grid.json`
- `results/<run_name>/eval/acceptance_check.json`
- `results/<run_name>/ACCEPTANCE_REPORT.md` (fill the E4 template)

## Implementation notes (minimal)
If no DAgger collector exists yet, create one with the smallest surface area:
- `scripts/h_exploration/run_dagger_collect.py`
- inputs: `--ckpt_path`, `--mic_root`, `--ldv_root`, `--lambda_c_values`, `--out_dir`, `--max_k`, `--tw`, `--gain`, `--min_k`
- outputs: `dagger_trajectories.pt` with fields defined in the E4 spec

Add a merge helper if needed:
- `scripts/h_exploration/merge_dagger_data.py`
- outputs: `merged_trajectories.pt`

## Execution steps (must follow)
1) Generate teacher trajectories (penalty-OMP)
2) Collect DAgger data (student rollout + teacher labels)
3) Merge datasets
4) Train on merged data (rtg1_mode=max_k)
5) Free-rollout eval (rollout_mode=free)
6) Acceptance check
7) Write E4 acceptance report
8) Commit code + results

## Report checklist (must include)
- Background / Motivation / Purpose / Expected
- Setup (env, device, data roots, subset + fingerprint)
- Exact commands
- Artifacts list
- Results with numeric values
- Interpretation using BECAUSE/THEREFORE
- Next steps

## Final response format
After commit, report:
- Run name and path
- Acceptance metrics (rho, steps_range, capture_range, action_change, KL)
- PASS/FAIL
- Next step suggestion (one line)

