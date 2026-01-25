# Algorithm Engineer Agent Prompt -- Execute E4l-Speech (Sub-sample Delay Refinement Diagnostics on Speech WAV)

You are an algorithm engineer working in this repository:

  /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2

You have **no assumed background knowledge** in signal processing or RL.

Your mission:

1) Implement **E4l-Speech** sub-sample delay refinement diagnostics (GCC-PHAT required; phase-slope optional).
2) Run the required evaluation sequence on **speech WAV only** (no .npy).
3) Produce complete, reproducible artifacts under results/<run>/.
4) Fill acceptance reports (English) with causal interpretation (BECAUSE/THEREFORE).

---

## Non-negotiables (must follow)

From AGENTS.md:
- Real data only. Missing roots => FAIL fast and document prerequisites. No synthetic stand-ins.
- No silent fallbacks or coercions (no resampling; fail on fs mismatch).
- Record exact commands and logs (tee -a).
- Run sequentially with a lockdir per OUT_DIR.
- Artifacts must be under results/<run>/ and sufficient to reproduce and verify.
- All writing must be in English.
- Do not create planning-only commits. The E4l docs + code + executed artifacts must be committed atomically together.

---

## Source of truth (read these first)

1) Spec (definitions, invariants, acceptance):
   - docs/rtgomp_subsample_delay_E4l_speech_spec.md

2) Plan (run order, commands):
   - docs/rtgomp_subsample_delay_E4l_speech_plan.md

3) Acceptance report template:
   - docs/rtgomp_subsample_delay_E4l_speech_acceptance_report_template.md

Do not invent new metric definitions. If code behavior disagrees with the spec, fix the code (and rerun).

---

## Required dataset (speech WAV only)

Use these roots exactly:
- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Guardrail:
- subset_manifest.json must contain only .wav paths. If any .npy appears, STOP and fix.
- If WAV fs != 16000, FAIL (no resampling fallback).

---

## Required checkpoint (DT)

Use:
- results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth

---

## Implementation to extend

Evaluator script:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

E4l requires adding flags and outputs:
- --write_subsample_delay_diagnostics 1
- --subsample_method gcc_phat[,phase_slope]
- --search_radius_frames <int>

Required new artifacts:
- subsample_delay_diagnostics.jsonl
- summary/subsample_delay_diagnostics_summary.json

---

## Execution workflow (must run in this order)

Run these in order (see plan for exact commands):

0) Preflight (fail fast):
   - verify roots exist and len(dataset)==416

1) Smoke (paired):
   - mode=smoke, num_pairs=1
   - write_per_sample=1
   - write_delay_diagnostics=1 (coarse)
   - write_subsample_delay_diagnostics=1 (fine)

2) Functional (scale_check_subset, 48 pairs):
   - paired (positive path)
   - mispair_shift1 (guardrail diagnostic)

3) Full dataset (paired):
   - mode=full_dataset, write_per_sample=0

Always:
- use lockdir + tee -a
- device=cpu (recommended)
- conda env trl-training
- PYTHONPATH=.
- MPLCONFIGDIR=/tmp/mpl

---

## Mandatory artifacts per run

Under each results/<run>/ directory, ensure these exist:

- subset_manifest.json
- run.log
- integrity_diagnostics.jsonl
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- delay_diagnostics.jsonl
- summary/delay_diagnostics_summary.json
- subsample_delay_diagnostics.jsonl
- summary/subsample_delay_diagnostics_summary.json
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled template)

Optional:
- per_sample.jsonl (required for smoke and scale_check_subset; optional for full_dataset)

---

## code_state.json (mandatory)

Create it manually after each run (same schema as earlier experiments) and include at least:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py
- scripts/h_exploration/dataset_lag.py

---

