# Plan: E4g — Full (Non-Smoke) Evaluation on a Larger Real Subset

E4c/E4d/E4e/E4f established a **minimal** evaluation workflow (typically `num_pairs=3`) to validate semantics and
prevent silent bugs. However, `num_pairs=3` is still a **smoke-scale** subset and is not sufficient for stable,
paper-quality conclusions.

This plan defines E4g: a larger-subset, evaluation-only run that reuses the validated E4-series checkpoint and
evaluation scripts, but scales the subset size while preserving strict reproducibility (manifest + fingerprint).

---

## Background

- E4c extended validation A/B/C is executed on `num_pairs=3`, mainly to prove:
  - A) teacher physics sanity (OMP > Random),
  - B) free vs teacher-forced is interpretable,
  - C) lambda-cost (RTG0) control is non-degenerate.
- E4e is planned to fix `teacher_forced` semantics and step indexing so B becomes a meaningful comparator.
- E4f provides an evaluation-only baseline (OMP vs DTmin vs Random), and surfaces early-STOP underperformance at high K
  in the stop-action branch.

E4g answers: do the same qualitative conclusions hold beyond the 3-pair subset, and with what numeric stability?

---

## Prerequisites (MUST)

- E4e evaluation semantics fix is applied:
  - `teacher_forced` means: teacher evolves residual only for non-stopped freqs; student STOP is the only stop rule.
  - `steps_used_mean` is reported as a count in `[1, max_k]`, not a 0-based index.

If E4e is not done, E4g MUST NOT be run; otherwise B is not interpretable at scale.

---

## Experiment Design (E4g)

### Subset

- Use real data only.
- Use a deterministic, manifest-driven subset:
  - Selection: "first N clip pairs in dataset order (all angles)"
  - `N` default: 12 (adjustable if runtime is too high)
  - Write `subset_manifest.json` with per-file MD5 and dataset fingerprint.

### Runs (single run directory)

Use one run directory:
- `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/`

Within it:

1) **A) OMP vs Random (larger subset)**
   - Same metric as earlier A checks, but with `--num_clips N` and a fixed seed for reproducibility.

2) **B/C) Lambda-grid evaluation**
   - B) Free rollout vs teacher-forced rollout using the E4e semantics.
   - C) Lambda-grid acceptance on free rollout.

3) **Baseline table (E4f-style)**
   - DTmin vs OMP K-sweep on the same subset and checkpoint (stop-action evaluator branch when applicable).
   - Combine with OMP vs Random into a single JSON table for paper reuse.

---

## Acceptance Targets (E4g)

E4g PASS if:
- A: OMP > Random for all K in `{1,2,4,8,16}` and `Gap(K=16) >= 0.05`.
- B: Both free and teacher-forced show `lambda_c ↑ => steps_used_mean ↓` with `spearman <= -0.6`.
- C: Free rollout passes the E4c acceptance thresholds.

Baseline (E4f-style) is descriptive and MUST be reported, but is not a PASS gate for E4g.

---

## Expected Runtime (order-of-magnitude)

The lambda-grid evaluation dominates runtime and scales roughly with `N` (pairs) and the number of windows per clip.
If `N=12` is too slow, reduce to `N=6` and record that decision explicitly in the acceptance report.

---

## Required Artifacts

Under `results/rtgomp_lambda_cost_E4g_full_eval_<timestamp>/`:
- `subset_manifest.json`
- `A_verify_omp_superiority/run.log`
- `B_free/lambda_grid.json`, `B_free/run.log`
- `B_teacher_forced/lambda_grid.json`, `B_teacher_forced/run.log`
- `C_free/lambda_grid.json`, `C_free/acceptance_check.json`, `C_free/run.log`
- `baseline_dt_vs_omp_stop_action/eval_stats.pt`, `baseline_dt_vs_omp_stop_action/run.log` (if stop-action branch)
- `summary/*.json` (parsed summaries)
- `ACCEPTANCE_REPORT.md` (filled from the E4g template)

