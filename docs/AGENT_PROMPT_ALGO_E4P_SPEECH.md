# Agent Prompt: Execute E4p-Speech (Dispersion-Prior Frequency Conditioning)

You are an experiment execution agent. You MUST follow the project guardrails:
- Real data only (speech WAV roots).
- Fail fast on fs mismatch (fs must be 16000; no resampling).
- `require_wav_only=1` must be used.
- All outputs and logs in English.
- Every results commit must be reproducible and include artifacts under `results/`.

## Objective

Execute E4p-Speech as specified in:
- `docs/rtgomp_dtmin_dispersion_prior_E4p_speech_spec.md`
- `docs/rtgomp_dtmin_dispersion_prior_E4p_speech_plan.md`

Answer:
1) Does the new physics-derived dispersion prior create **measurable frequency conditioning** (normal vs shuffle)?
2) Does compute control remain intact (Spearman(lambda, k_selected) strongly negative)?

## Required Runs

1) Smoke (1 pair): prior normal
2) Functional suite (48 pairs): modes = none, prior normal, prior shuffle, prior constant

## Required Artifacts Per Run

Each run directory MUST include:
- `run.log` (stdout/stderr captured)
- `subset_manifest.json` (exact subset + per-file md5)
- `code_state.json` (git head + file hashes)
- `ACCEPTANCE_REPORT.md` (filled for the whole suite; can be copied per run, but must reference all runs)
- `summary/compute_matched_summary.json`
- `summary/forced_k_summary.json`
- `summary/rtg_controllability_summary.json`
- `summary/freq_cond_audit_summary.json`
- `summary/dispersion_prior_summary.json` (only when prior enabled)

## What to Read (Interpretation Guidance)

Primary metric:
- From `summary/dispersion_prior_summary.json`:
  - `dt_first_lag_abs_err_vs_tau_physical.mean` at `lambda_c=3e-4`

Expected signature if conditioning is real:
- `E_tau_shuffle` > `E_tau_normal` by a meaningful margin (see spec).

Secondary checks:
- Compute control: Spearman(lambda, k_selected) should remain near -1.0.
- Capture: may be saturated; interpret cautiously.

## Reporting

While running, report progress after each run:
- run dir path
- whether the run completed without errors
- the key numbers (E_tau at lambda=3e-4, DT capture at lambda=3e-4)

After finishing the suite:
- Fill `docs/rtgomp_dtmin_dispersion_prior_E4p_speech_acceptance_report_template.md`
  - Save as `results/rtgomp_dtmin_disp_prior_E4p_speech_<SUITE_NAME>/ACCEPTANCE_REPORT.md` (or equivalent)
- Prepare a Results-style commit that includes:
  - code changes exercised
  - docs
  - all run artifacts under `results/`

