# Algorithm Engineer Agent Prompt -- Execute E4h-Speech (Paper-Grade Evaluation on Speech WAV)

You are an algorithm engineer working in this repository:

  /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2

You have **no assumed background knowledge** in signal processing or RL. This prompt includes all required context.

Your mission:

1) Run the paper-grade evaluation **E4h-Speech** on the speech WAV dataset only.
2) Produce complete, reproducible artifacts under results/<run>/.
3) Write an acceptance report that interprets the results with causal language.

E4h-Speech outputs two complementary result families:

A) Compute-matched WITH STOP (main paper result)
   - DT vs OMP vs Random compared at matched compute (k_selected).

B) Forced-K NO STOP (ablation)
   - DT vs OMP vs Random compared at fixed K in {1,2,4,8,16}.

---

## Non-negotiables (must follow)

From AGENTS.md:
- Real data only. Missing roots => FAIL fast, document prerequisites. No synthetic stand-ins.
- No silent fallbacks or coercions.
- Record exact commands and logs (tee -a).
- Run sequentially with a lockdir per OUT_DIR.
- Artifacts must be under results/<run>/ and must be sufficient to reproduce and verify.
- All writing must be in English.
- Do not create planning-only commits. Any new/updated docs must be committed together with executed results (atomic).

---

## Source of truth (read these files first)

1) Spec (definitions, invariants, acceptance):
   - docs/rtgomp_complexity_cost_E4h_speech_paper_eval_spec.md

2) Plan (run order, commands, debug playbook):
   - docs/rtgomp_complexity_cost_E4h_speech_paper_eval_plan.md

3) Acceptance report template (must fill and save under results/<run>/):
   - docs/rtgomp_complexity_cost_E4h_speech_paper_eval_acceptance_report_template.md

Do not invent new metric definitions. If code behavior disagrees with the spec, fix the code (and rerun).

---

## Background (what you are evaluating)

You will evaluate how well a policy selects lag indices to reconstruct LDV STFT windows from lagged MIC STFT windows.

Key objects (per sample):
- Choose a time window start t0 and a window length Tw=32 STFT frames.
- For each frequency bin f in the band [300,3000] Hz:
  - Target vector Y_f is LDV STFT over Tw frames.
  - Dictionary D_f contains M=101 candidate lag atoms (MIC STFT windows shifted by lags in [-50,+50]).

Selection methods:
- OMP: greedy oracle selection (unique lags).
- Random: baseline selection (paper baseline: unique sampling without replacement).
- DT: learned policy that selects lags and may STOP early.

Metric:
- capture = 1 - E_res / max(E0, eps)
  - E0 = ||Y||^2
  - E_res = ||Y - projection onto selected atoms||^2
  - capture must be in [0,1] in correct least-squares projection (out-of-range indicates a failure).

Compute matching:
- True compute cost is number of selected atoms (LS solves), not "decision steps including STOP".
- Therefore compute-match OMP/Random to DT using k_selected.

---

## Required dataset (speech only)

Use these roots exactly:
- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

The dataset enumerator (DoALagDataset with angle=None) pairs MIC and LDV wavs by filename (MIC -> LDV replacement).

Critical guardrail:
- subset_manifest.json must contain only .wav paths (no .npy). If any .npy appears, you ran the wrong dataset.

---

## Required checkpoint (DTmin)

Default:
- results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth

The evaluator checks compatibility:
- max_lag=50 -> M=101
- STOP head -> action_dim=102

If the evaluator reports mismatch, stop and fix the checkpoint/flags.

---

## Implementation to run

Evaluator script:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

You will run it multiple times with different modes and out_dir values.

---

## Execution workflow (must run in this order)

Always:
- use conda env trl-training
- export PYTHONPATH=.
- export MPLCONFIGDIR=/tmp/mpl
- use a lockdir OUT_DIR/.lock
- capture logs via tee -a OUT_DIR/run.log

### Step 0: Preflight (fail fast)

Run the dataset length check from the plan (prints num_pairs, first/last pair).
Record it in your acceptance report.

### Step 1: Smoke run (required)

Run E4h-Speech smoke:
- mode=smoke
- num_pairs=1
- random_trials=3
- random_sampling=without_replacement
- write_per_sample=1
- device=cpu (recommended)

Acceptance:
- must complete
- must satisfy integrity guardrails (no out-of-range capture)

### Step 2: Functional tests (required)

2.1 Positive-path:
- mode=scale_check_subset (48 pairs)
- random_trials>=3
- random_sampling=without_replacement
- write_per_sample=1
- device=cpu

2.2 Guardrail diagnostic (non-paper baseline):
- same subset but random_sampling=with_replacement
- expected outcome: may FAIL due to capture out-of-range; must report duplicate rate and worst violation
- do not silently clamp

### Step 3: Full dataset (paper numbers)

Only after scale_check_subset passes:
- mode=full_dataset (all pairs in dataset order)
- random_trials=1 (recommended)
- write_per_sample=0 (recommended; too large otherwise)
- device=cpu

---

## Mandatory artifacts per run

Under each results/<run>/ directory, ensure these exist:
- subset_manifest.json
- run.log
- integrity_diagnostics.jsonl
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- code_state.json (you must create this; see plan)
- ACCEPTANCE_REPORT.md (filled template)

Optional:
- per_sample.jsonl (required for smoke and scale_check_subset; optional for full_dataset)

---

## How to write code_state.json (mandatory)

The evaluator does not auto-write this file. Create it manually after each run.

Minimum schema:
{
  "git_head": "...",
  "dirty": true/false,
  "files": { "scripts/h_exploration/run_rtgomp_e4h_paper_eval.py": "sha256...", ... }
}

Include at least:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py
- scripts/h_exploration/dataset_lag.py

---

## Completion criteria

You are done when you have:
- a PASS scale_check_subset run on speech (paper baseline)
- a PASS full_dataset run on speech (paper baseline)
- a completed diagnostic guardrail run (documented; may FAIL as expected)
- acceptance reports for each run (English, causal interpretation)

If anything fails, do not hide it:
- commit or save the failure artifacts and explain root cause hypotheses.

