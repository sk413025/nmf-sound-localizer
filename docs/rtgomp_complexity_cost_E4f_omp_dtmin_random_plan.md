# Plan: E4f — OMP vs DTmin vs Random Energy Capture Baseline (E4d/E4e Context)

E4d and E4e focus on evaluation semantics (teacher-forced correctness). However, there is no E4-series run that reports the direct energy-capture comparison **OMP vs DTmin vs Random** on the same subset and checkpoint. This plan defines a minimal evaluation-only follow-up to answer that question with reproducible numbers.

---

## Background

- E4c/E4d/E4e use the same checkpoint and subset manifest; the evaluation fixes do not change model weights.
- The question "how much energy is explained by OMP vs DTmin vs Random" is currently answered only by older GRU2 reports, not by the E4-series context.

---

## Goal

Produce a **K-sweep** energy-capture baseline for the E4-series checkpoint on the same real-data subset:
- OMP (oracle)
- DTmin (student)
- Random (weak baseline)

Outputs must be captured under `results/rtgomp_lambda_cost_E4f_omp_dtmin_random_<timestamp>/` with logs and summary JSON for reuse in papers.

---

## Hypotheses

1) OMP must dominate Random for all K values BECAUSE greedy correlation + least-squares projection is a principled pursuit method.
2) DTmin must be competitive with OMP (>= ~80% at K=8) BECAUSE the model was trained on OMP trajectories.
3) At K=16, DTmin should converge near OMP BECAUSE the dictionary becomes saturated with the full lag budget.
---

## Minimal Change Policy

Evaluation only; **no training or model changes**. Use existing scripts:
- `verify_omp_superiority.py` (OMP vs Random K-sweep)
- `scripts/eval_energy_capture_generic.py` (DTmin vs OMP K-sweep)

If dataset order cannot be verified to match the subset manifest, stop and document the mismatch; do not silently change the selection.

---

## Experiment Design (E4f)

1) Validate that the dataset order matches the subset manifest (first 3 clip pairs).
2) **Checkpoint compatibility preflight**: verify checkpoint head/output dimensions match `M_lags = 2*max_lag+1`.
   - If the head has `M_lags + 1`, follow the stop‑action spec:
     - `docs/rtgomp_complexity_cost_E4f_omp_dtmin_random_stop_action_k_sweep_spec.md`
3) Run OMP vs Random K-sweep (A) using `mic_root/ldv_root` from the subset manifest.
4) Run DTmin vs OMP capture (B) using the same manifest roots and the compatible checkpoint.
5) Generate a combined summary table (C) for K in `{1,2,4,8,16}`.
Artifacts:
- `A_verify_omp_superiority/run.log` (OMP vs Random numbers)
- `B_dt_vs_omp/eval_stats.pt` + `B_dt_vs_omp/run.log`
- `summary/omp_vs_random_k_sweep.json`
- `summary/dt_vs_omp_k_sweep.json`
- `summary/omp_dtmin_random_k_sweep.json`
- `subset_manifest.json`, `env_info.json`, `code_state.json`

---

## Acceptance Targets (E4f)

PASS if all:
- OMP > Random for all K, and `gap(K=16) >= 0.05`.
- DTmin capture is finite and in `[0, 1]` for all K.
- DTmin is competitive with OMP:
  - `abs(DT - OMP) <= 0.02` at `K=16` (DT may exceed OMP).
  - `DT/OMP >= 0.80` at `K=8`.
- DTmin >= Random for all K.
If any fail, E4f is FAIL and must be explained causally.
