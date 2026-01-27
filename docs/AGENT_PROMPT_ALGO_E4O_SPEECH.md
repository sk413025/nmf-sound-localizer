# Algorithm Engineer Agent Prompt — Execute E4o-Speech (DTmin Frequency Conditioning Audit)

You are the Algorithm Engineer agent responsible for executing **E4o-Speech** end-to-end and reporting results.

Hard constraints:
- All content and artifacts MUST be in English.
- Real data only (speech WAV). No synthetic data. No resampling.
- Fail fast on guardrail violations.
- No planning-only commits: if you commit, commit code + artifacts + filled acceptance report atomically.

Source of truth:
- Spec: `docs/rtgomp_dtmin_freq_cond_E4o_speech_spec.md`
- Plan: `docs/rtgomp_dtmin_freq_cond_E4o_speech_plan.md`
- Acceptance template: `docs/rtgomp_dtmin_freq_cond_E4o_speech_acceptance_report_template.md`

---

## Mission

Determine whether the current DTmin (`SeqDT_FreqAware`) uses frequency conditioning (`freq_idx`/`freq_embed`) in a materially meaningful way, by running inference-time ablations and quantifying the deltas.

You must produce:
- Executed artifacts under `results/<run>/`
- A filled `results/<run>/ACCEPTANCE_REPORT.md` for each run
- A short, causal interpretation that classifies the outcome as:
  - `FREQ_COND_USED` / `FREQ_COND_IGNORED` / `INCONCLUSIVE`

---

## Required progress reporting (do this as you work)

After each milestone, post a progress update with:
- What you just did (exact command or code change)
- What you observed (key log line(s) + whether guardrails passed)
- What you will do next
- If blocked: the exact error + hypothesized root cause

Milestones:
1) Preflight passed (roots + dataset length + checkpoint exists).
2) Code changes implemented (flags added; summary JSON written).
3) Smoke run completed (normal mode).
4) Functional suite completed (4 modes on scale_check_subset).
5) Acceptance reports filled; classification finalized.

---

## Step 1 — Preflight (Fail Fast)

Follow `docs/rtgomp_dtmin_freq_cond_E4o_speech_plan.md` section 1 exactly.

If any check fails:
- STOP immediately
- Report the failure and which prerequisite is missing
- Do NOT add fallbacks (no resampling, no alternative roots)

---

## Step 2 — Implement required evaluator changes

Edit `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`:
- Add CLI flags:
  - `--freq_cond_mode` (normal/shuffle/constant/zero_embed)
  - `--freq_cond_seed`
  - `--freq_cond_constant_idx`
- Apply the ablation ONLY to the `freq_ids` passed into DT inference.
- Add artifact `summary/freq_cond_audit_summary.json` per spec.

Fail-fast requirements:
- invalid `freq_cond_mode` => fail
- constant_idx out of [0,1024] => fail

After implementation:
- Run a minimal import check (`python -m py_compile ...`) or a smoke run (preferred) before launching the suite.

---

## Step 3 — Execute the run suite (smoke + functional)

Run the minimum suite from the plan:
- Smoke: 1 pair, `freq_cond_mode=normal`
- Functional (48 pairs, paired): `normal`, `shuffle`, `constant`, `zero_embed`

Critical requirements:
- Each run directory must include:
  - `run.log`
  - `subset_manifest.json`
  - `summary/compute_matched_summary.json`
  - `summary/rtg_controllability_summary.json`
  - `summary/freq_cond_audit_summary.json`
  - `code_state.json`
  - `ACCEPTANCE_REPORT.md` (filled using the template)

If any run fails:
- STOP the suite
- Report the failure and the exact exception + stack trace from `run.log`

---

## Step 4 — Interpret results (must be causal and quantitative)

Your analysis must explicitly compute deltas vs the `normal` run:
- Use the target metric:
  - `DT_cm = compute_matched_capture_mean["dt"]` at `lambda_c=3e-4`
- Compute:
  - Δshuffle, Δconst, Δzero as defined in the spec

Classification rules (must follow the spec):
- `FREQ_COND_USED` if any delta ≥ 0.05 and consistent across ≥4/5 lambdas
- `FREQ_COND_IGNORED` if all deltas ≤ 0.01 and controllability unchanged
- otherwise `INCONCLUSIVE`

Interpretation guidance (examples of causal language):
- “DT_cm drops under shuffle BECAUSE the policy relies on `freq_embed` to disambiguate regimes; THEREFORE freq conditioning is used.”
- “DT_cm remains unchanged under zero_embed BECAUSE state correlations already encode enough frequency identity; THEREFORE freq conditioning is likely ignored.”
- “Only high band degrades BECAUSE phase behavior changes with frequency; THIS IMPLIES per-band follow-up may be justified.”

Do not hand-wave:
- Always cite the exact numbers and paths to the JSON summaries.

---

## Step 5 — Close-out checklist (before committing anything)

Verify:
- No missing required artifacts.
- All acceptance reports are filled (English).
- `code_state.json` is present and includes sha256s for edited files.
- You can reproduce each run with `conda run -n trl-training` and `PYTHONPATH=.` commands.

If you make a commit:
- Commit code + results + docs together (atomic).
- Commit message must include background/motivation/purpose/setup/commands/artifacts/results/analysis/next steps.

