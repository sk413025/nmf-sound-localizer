# Algorithm Engineer Agent Prompt -- Execute E4m-Speech (Dispersion Diagnostics + Calibration Readiness on Speech WAV)

You are an algorithm engineer working in this repository:

  /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2

You have **no assumed background knowledge** in signal processing or RL.

Your mission:

1) Implement **E4m-Speech** dispersion diagnostics:
   - GCC-PHAT (required)
   - Phase-slope group delay (required): global + 3 sub-bands + fit diagnostics
   - Method agreement + dispersion metrics
2) Run the required evaluation sequence on **speech WAV only** (no .npy).
3) Produce complete, reproducible artifacts under `results/<run>/`.
4) Fill acceptance reports (English) with causal interpretation (BECAUSE/THEREFORE).
5) Prepare an **atomic Results commit** (code + docs + executed artifacts). No planning-only commit.

---

## Non-negotiables (must follow)

From AGENTS.md:
- Real data only. Missing roots => FAIL fast and document prerequisites. No synthetic stand-ins.
- No silent fallbacks or coercions (no resampling; fail on fs mismatch).
- Record exact commands and logs (`tee -a`).
- Run sequentially with a lockdir per OUT_DIR.
- Artifacts must be under `results/<run>/` and sufficient to reproduce and verify.
- All writing must be in English.
- Do not create planning-only commits. The E4m docs + code + executed artifacts must be committed atomically together.

---

## Source of truth (read these first)

1) Spec (definitions, invariants, acceptance):
   - docs/rtgomp_dispersion_E4m_speech_spec.md

2) Plan (run order, commands):
   - docs/rtgomp_dispersion_E4m_speech_plan.md

3) Acceptance report template:
   - docs/rtgomp_dispersion_E4m_speech_acceptance_report_template.md

Do not invent new metric definitions. If code behavior disagrees with the spec, fix the code (and rerun).

---

## Required dataset (speech WAV only)

Use these roots exactly:
- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Guardrails:
- `subset_manifest.json` must contain only `.wav` paths. If any `.npy` appears, STOP and fix.
- If WAV fs != 16000, FAIL (no resampling fallback).

---

## Required checkpoint (DT)

Use:
- results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth

---

## Required evaluator to extend

Evaluator script:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

E4m requires extending the existing E4l subsample diagnostics:
- Keep GCC-PHAT implementation.
- Add phase-slope group delay (global + 3 sub-bands) and fit diagnostics.
- Extend:
  - results/<run>/subsample_delay_diagnostics.jsonl
  - results/<run>/summary/subsample_delay_diagnostics_summary.json

Required E4m metrics:
- phase_slope_tau_hat_ms, phase_slope_r2, phase_slope_fit_rmse_rad, phase_slope_num_bins_used
- per-band tau/r2/rmse + tau_band_spread_ms
- tau_agreement_ms = phase_slope_tau_hat_ms - gcc_phat_tau_hat_ms

No fallbacks:
- If insufficient bins or ill-posed fit: output nulls + counters; do not invent values.

---

## Required lambda grid (5 points; exact)

Use exactly:
- 1e-5,3e-5,1e-4,2e-4,3e-4

---

## Execution workflow (must run in this order)

0) Preflight (fail fast):
   - verify roots exist and len(dataset)==416

1) Smoke (paired):
   - mode=smoke, num_pairs=1
   - write_delay_diagnostics=1 (coarse)
   - write_subsample_delay_diagnostics=1 (fine; gcc_phat + phase_slope)

2) Functional (scale_check_subset, 48 pairs):
   - paired (positive path)
   - mispair_shift1 (guardrail diagnostic)

3) Full dataset (paired):
   - mode=full_dataset, num_pairs=len(dataset)=416

Always:
- use lockdir + tee -a
- device=cpu (recommended)
- conda env trl-training
- PYTHONPATH=.
- MPLCONFIGDIR=/tmp/mpl

---

## Mandatory artifacts per run

Under each `results/<run>/` directory, ensure these exist:

- subset_manifest.json
- run.log
- integrity_diagnostics.jsonl
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- delay_diagnostics.jsonl
- summary/delay_diagnostics_summary.json
- subsample_delay_diagnostics.jsonl (extended with phase_slope + dispersion fields)
- summary/subsample_delay_diagnostics_summary.json (extended)
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled template)

Optional:
- per_sample.jsonl (not required for E4m; avoid huge artifacts unless explicitly requested)

---

## code_state.json (mandatory)

Create it manually after each run and include at least:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py
- scripts/h_exploration/dataset_lag.py

---

## Results commit (mandatory)

After all required runs pass and reports are filled:
- Create a single atomic commit that includes:
  - E4m docs (spec/plan/template/prompt)
  - code changes
  - all results/<run>/ artifacts for E4m (use `git add -f results/<run>/` because results/ is ignored)

No planning-only commit is allowed.

