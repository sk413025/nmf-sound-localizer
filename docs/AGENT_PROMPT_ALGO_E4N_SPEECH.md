# Algorithm Engineer Agent Prompt -- Execute E4n-Speech (Phase-2d Phase Equalization Calibration + E4m Re-validation)

You are an algorithm engineer working in this repository:

  /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/exp-interspeech-GRU2

You have **no assumed background knowledge** in signal processing or RL.

Your mission:

1) Implement **E4n-Speech** phase equalization calibration:
   - Fit a per-frequency unit-magnitude complex correction (phase_eq) on paired speech WAV.
   - Save phase_eq artifacts under `results/<run>/phase_eq/`.
2) Extend the existing evaluator to **apply** phase_eq consistently:
   - Apply to LDV STFT used by the RL/capture evaluation.
   - Apply inside the waveform-domain subsample GCC-PHAT / phase-slope path (FFT-domain multiply).
3) Run the required sequence:
   - preflight
   - FIT: smoke + functional + full_dataset (paired only)
   - EVAL (apply phase_eq from fit full_dataset): smoke + functional paired + functional mispair_shift1 + full_dataset paired
4) Produce complete reproducible artifacts under `results/<run>/` for every run.
5) Fill `ACCEPTANCE_REPORT.md` (English; BECAUSE/THEREFORE causal interpretation).
6) Prepare an **atomic Results commit** (docs + code + executed artifacts). No planning-only commit.

---

## Non-negotiables (must follow)

From AGENTS.md:
- Real data only. Missing roots => FAIL fast and document prerequisites. No synthetic stand-ins.
- No silent fallbacks or coercions (no resampling; fail on fs mismatch).
- Record exact commands and logs (`tee -a`).
- Run sequentially with a lockdir per OUT_DIR.
- Artifacts must be under `results/<run>/` and sufficient to reproduce and verify.
- All writing must be in English.
- Do not create planning-only commits. E4n docs + code + executed artifacts must be committed atomically.

Hard dataset guardrails:
- WAV only (`require_wav_only=1`); if any `.npy` appears in `subset_manifest.json`, STOP and fix.
- WAV sample rate must be exactly 16000; if not, FAIL (no resampling).

---

## Source of truth (read these first)

1) Spec:
   - docs/rtgomp_phase_eq_E4n_speech_spec.md

2) Plan:
   - docs/rtgomp_phase_eq_E4n_speech_plan.md

3) Acceptance template:
   - docs/rtgomp_phase_eq_E4n_speech_acceptance_report_template.md

Do not invent new metric definitions. If code behavior disagrees with the spec, fix the code (and rerun).

---

## Required dataset (speech WAV only)

Use these roots exactly:
- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

---

## Required evaluator + checkpoint (EVAL runs)

Evaluator:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

Checkpoint (DT):
- results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth

Lambda grid (exact):
- 1e-5,3e-5,1e-4,2e-4,3e-4

---

## Required E4n fitter (to implement)

Create:
- scripts/h_exploration/fit_phase_eq_e4n_speech.py

It must:
- enforce wav-only + fs=16000
- compute GCC-PHAT tau per window (tau_center=0; radius=search_radius_frames*hop_length)
- estimate `phase_eq_rfft` (nfft_fit=32768) via weighted circular mean after removing the delay term
- downsample to STFT bins -> `phase_eq_stft` (1025 bins)
- write:
  - results/<run>/phase_eq/phase_eq.npz
  - results/<run>/phase_eq/phase_eq_fit_summary.json
  - standard artifacts (subset_manifest.json, run.log, code_state.json, ACCEPTANCE_REPORT.md)

No fallbacks:
- NaN/Inf => FAIL
- fs mismatch => FAIL
- non-wav path => FAIL

---

## Extend evaluator to apply phase_eq (required)

In scripts/h_exploration/run_rtgomp_e4h_paper_eval.py:
- Add flags:
  - --apply_phase_eq (0/1)
  - --phase_eq_path <path to phase_eq.npz>
- When enabled:
  - Multiply ldv_stft by phase_eq_stft before evaluation.
  - Multiply LDV rFFT spectrum by phase_eq_rfft inside GCC-PHAT / phase-slope path so dispersion metrics reflect calibration.
- Hard-fail on any shape mismatch or non-unit-magnitude.

---

## Execution workflow (must run in this order)

Environment prelude:
```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

1) Preflight: verify roots + len(dataset)=416.

2) FIT runs (paired only):
- smoke (num_pairs=1)
- functional scale_check_subset (num_pairs=48)
- full_dataset (num_pairs=416) -> produces phase_eq used for EVAL

3) EVAL runs (apply phase_eq from FIT full_dataset):
- smoke paired (1)
- functional paired (48)
- functional mispair_shift1 (48)
- full_dataset paired (416)

Always:
- lockdir + tee -a
- device=cpu for EVAL runs (recommended)

---

## Mandatory artifacts per run

All runs:
- subset_manifest.json (wav-only)
- run.log
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled; English; causal)

FIT runs additionally:
- phase_eq/phase_eq.npz
- phase_eq/phase_eq_fit_summary.json

EVAL runs additionally:
- integrity_diagnostics.jsonl
- summary/*.json including subsample_delay_diagnostics_summary.json
- subsample_delay_diagnostics.jsonl (must reflect calibrated signal)

---

## Results commit (mandatory)

After all runs pass and reports are filled:
- Create one atomic Results commit including:
  - E4n docs (spec/plan/template/prompt)
  - code changes (fitter + evaluator extension)
  - all results/<run>/ artifacts (use `git add -f`)
