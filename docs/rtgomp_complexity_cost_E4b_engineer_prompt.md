# Prompt: Execute E4b (High-Ratio DAgger-lite)

You are the algorithm engineer. Execute **E4b** end-to-end with real data and produce a results commit.

## Read these first
1) `docs/rtgomp_complexity_cost_spec.md`
2) `docs/rtgomp_complexity_cost_E4b_plan.md`
3) `docs/rtgomp_complexity_cost_E4b_spec.md`
4) `docs/rtgomp_complexity_cost_E4b_acceptance_report_template.md`

## Objective
Increase DAgger ratio and coverage to widen the **free-rollout** step range.

Target:
- `spearman(lambda_c, steps_used_mean) <= -0.6`
- `steps_range >= 0.10`

## Hard requirements
- One commit = one executed experiment.
- Commit includes code + artifacts + acceptance report.
- Outputs under `results/<run_name>/`.
- Real data only.

## Required parameter deltas from E4
- DAgger ratio: **3.0**
- DAgger coverage: **num_clips = 3**
- DAgger stride: **stride = 128**

## Execution steps
1) Generate teacher trajectories (penalty-OMP).
2) Collect DAgger data with `num_clips=3` and `stride=128`.
   - Use `--log_every_windows 25` (or 1) so long CPU-bound labeling does not look “stuck”.
3) Merge datasets with `--dagger_ratio 3.0`.
4) Train on merged data.
5) Free-rollout eval (`rollout_mode=free`).
6) Acceptance check.
7) Write `ACCEPTANCE_REPORT.md` from template.
8) Commit results.

## Required outputs
See the template in `docs/rtgomp_complexity_cost_E4b_acceptance_report_template.md`.

## Final response format
After commit, report:
- Run name and path
- Acceptance metrics (rho, steps_range, capture_range, action_change, KL)
- PASS / PARTIAL / FAIL
- Next step suggestion
