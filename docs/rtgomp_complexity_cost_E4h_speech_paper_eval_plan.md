# Plan: E4h-Speech -- Paper-Grade Evaluation (Compute-Matched + Forced-K Ablation)

This plan defines how to execute **E4h-Speech**, the paper-grade evaluation of DT vs OMP vs Random on the **speech WAV
dataset only**.

The intended executor is an algorithm engineer who has no prior context. Follow this plan exactly and do not "fill in"
missing behavior with assumptions.

Source of truth for definitions and acceptance thresholds:
- docs/rtgomp_complexity_cost_E4h_speech_paper_eval_spec.md
- docs/rtgomp_complexity_cost_E4h_speech_paper_eval_acceptance_report_template.md

Implementation that will be executed:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

---

## 0) Why This Plan Exists (Motivation)

We previously executed E4h on a white-noise NPY dataset. That creates a train/eval domain shift relative to the DT
checkpoint that was trained on speech WAV pairs, which can make STOP/RTG controllability appear weak or saturated.

E4h-Speech fixes this by running the same paper-grade evaluation on the speech dataset roots used for DT training.

Paper goal:
- Demonstrate RTG controllability (lambda_c controls compute).
- Compare DT vs Random and DT vs OMP under fair compute matching.

---

## 1) Non-Negotiables (Operational Guardrails)

From AGENTS.md (must comply):
- Real data only. If speech roots are missing: FAIL fast and document prerequisites. No synthetic substitutes.
- No silent fallbacks/coercions. Prefer explicit validation and clear error messages.
- All artifacts under results/<run>/ with logs, manifests, fingerprints, acceptance report.
- Every run must have exact reproduction commands recorded.
- Use conda env: trl-training.
- Run sequentially, with lockdirs, and log via tee -a.
- All documentation must be written in English.
- Do not commit planning-only changes. (Docs created/updated here must be committed together with executed results.)

---

## 2) Fixed Inputs (Default; Do Not Change Unless Justified)

### 2.1 Speech dataset roots (required)

- MIC root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- LDV root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Expected dataset length with DoALagDataset(angle=None, hop_length=160):
- len(dataset) == 416 pairs on this machine.

### 2.2 Checkpoint (DTmin)

Default checkpoint for E4h-Speech:
- results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth

### 2.3 Evaluator parameters (fixed)

- hop_length = 160
- fs = 16000
- n_fft = 2048
- freq_min = 300
- freq_max = 3000
- max_lag = 50  -> M_lags = 101
- Tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = 1e-4,3e-4,1e-3,3e-3,1e-2
- Random baseline (paper): without_replacement

Recommended devices:
- Prefer cpu for stability and reproducibility.
- mps is allowed for DT inference but may stall; if used, document it explicitly.

---

## 3) Preflight Checklist (Fail Fast)

Run these checks before any long evaluation. Record outputs in the run log or acceptance report.

### 3.1 Verify speech roots exist

Confirm both directories exist and contain WAV files:
- /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav
- /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav

### 3.2 Verify dataset pairing and length

Run:
  PYTHONPATH=. conda run -n trl-training python - <<'PY'
  from scripts.h_exploration.dataset_lag import DoALagDataset
  mic='/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC'
  ldv='/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV'
  ds=DoALagDataset(mic,ldv,angle=None,hop_length=160)
  print('num_pairs',len(ds))
  print('first_pair',ds.clips[0])
  print('last_pair',ds.clips[-1])
  PY

Acceptance:
- num_pairs is stable (expected 416 on this machine).
- first/last pairs are WAV paths under the speech roots.

If this fails:
- Stop. Fix roots. Do not proceed.

### 3.3 Verify checkpoint compatibility (max_lag and STOP head)

The evaluator already checks:
- state_embed input dim == M_lags (101)
- head output dim == M_lags + 1 (102)

If mismatch:
- Stop. You are using the wrong checkpoint or wrong max_lag.

---

## 4) Run Sequence (Smoke -> Functional -> Scale-check -> Full dataset)

Important: always run with:
- a lockdir: OUT_DIR/.lock
- logs captured: 2>&1 | tee -a OUT_DIR/run.log

Also recommended to avoid matplotlib cache warnings:
- export MPLCONFIGDIR=/tmp/mpl

### 4.1 Common shell prelude (copy/paste)

  source ~/.zshrc
  conda activate trl-training
  export PYTHONPATH=.
  export MPLCONFIGDIR=/tmp/mpl

### 4.2 Step A: Smoke test (required; real speech; tiny)

Purpose:
- Validate end-to-end execution and guardrails quickly.

Config:
- mode = smoke
- num_pairs = 1
- random_trials = 3
- write_per_sample = 1 (required for smoke)
- device = cpu (recommended)

Command template:
  OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_smoke_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$OUT_DIR/summary"
  LOCKDIR="$OUT_DIR/.lock"
  if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
  trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

  PYTHONPATH=. conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
    --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
    --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
    --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth \
    --out_dir "$OUT_DIR" \
    --mode smoke --num_pairs 1 \
    --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
    --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
    --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
    --random_trials 3 --random_sampling without_replacement --seed 0 \
    --write_per_sample 1 --device cpu \
    2>&1 | tee -a "$OUT_DIR/run.log"

Post-step required actions:
- Generate code_state.json (see Section 5).
- Fill ACCEPTANCE_REPORT.md using the template.

Acceptance:
- Must complete without error.
- Must satisfy all hard guardrails (spec Section 7.1).

### 4.3 Step B: Functional tests (required; real speech)

We require two functional runs on the deterministic 48-pair subset.

#### B1) Positive-path functional run (paper baseline)

Config:
- mode = scale_check_subset (48 pairs)
- random_trials = 3
- random_sampling = without_replacement (paper baseline)
- write_per_sample = 1 (required on scale-check)
- device = cpu (recommended)

Command:
  OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_scale_check_subset_$(date +%Y%m%d_%H%M%S)"
  ... (same lockdir pattern) ...
  PYTHONPATH=. conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
    --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
    --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
    --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth \
    --out_dir "$OUT_DIR" \
    --mode scale_check_subset --num_pairs 48 \
    --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
    --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
    --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
    --random_trials 3 --random_sampling without_replacement --seed 0 \
    --write_per_sample 1 --device cpu \
    2>&1 | tee -a "$OUT_DIR/run.log"

Acceptance:
- Must PASS hard guardrails.
- Must produce all summary JSONs and diagnostics files.

#### B2) Guardrail diagnostic run (non-paper baseline)

Purpose:
- Demonstrate the evaluator reports duplicate rates and does not silently clamp capture.

Config:
- Same as B1 but random_sampling = with_replacement
- write_per_sample can be 0 (optional); integrity_diagnostics.jsonl must exist

Expected outcome:
- This run may FAIL (capture out-of-range) due to duplicates. That is acceptable if explicitly labeled diagnostic and
  the failure is well-diagnosed.

### 4.4 Step C: Full speech dataset (paper numbers)

Only run full_dataset after scale_check_subset passes.

Config:
- mode = full_dataset (num_pairs must equal len(dataset))
- random_trials = 1 (recommended to reduce runtime)
- write_per_sample = 0 (recommended; too large otherwise)

Command:
  OUT_DIR="results/rtgomp_lambda_cost_E4h_speech_paper_eval_full_dataset_$(date +%Y%m%d_%H%M%S)"
  ... (same lockdir pattern) ...
  PYTHONPATH=. conda run --no-capture-output -n trl-training python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
    --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
    --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
    --ckpt_path results/rtgomp_lambda_cost_E4c_stopstatefix_ratio3_20260121_030820/model/dt_freq_aware_best.pth \
    --out_dir "$OUT_DIR" \
    --mode full_dataset \
    --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
    --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
    --lambda_c_values "1e-4,3e-4,1e-3,3e-3,1e-2" \
    --random_trials 1 --random_sampling without_replacement --seed 0 \
    --write_per_sample 0 --device cpu \
    2>&1 | tee -a "$OUT_DIR/run.log"

Post-step:
- Generate code_state.json.
- Fill ACCEPTANCE_REPORT.md.
- Extract paper tables from summary JSONs (report in acceptance report).

---

## 5) Required Provenance: code_state.json (must create for every run)

The evaluator does not auto-write code_state.json, but results commits require it.

Create:
  results/<run>/code_state.json

Minimal required schema:
{
  "git_head": "<hash>",
  "dirty": true/false,
  "files": { "path": "sha256hex", ... }
}

Use this command (run from repo root):
  python - <<'PY'
  import hashlib, json, subprocess
  from pathlib import Path
  def sha256(p: Path) -> str:
      h = hashlib.sha256()
      with p.open("rb") as f:
          for b in iter(lambda: f.read(1024*1024), b""):
              h.update(b)
      return h.hexdigest()
  git_head = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()
  dirty = subprocess.check_output(["git","status","--porcelain"]).decode().strip() != ""
  files = [
      "scripts/h_exploration/run_rtgomp_e4h_paper_eval.py",
      "scripts/h_exploration/dataset_lag.py",
  ]
  out = {"git_head": git_head, "dirty": dirty, "files": {p: sha256(Path(p)) for p in files}}
  print(json.dumps(out, indent=2))
  PY

Redirect the printed JSON into <run_dir>/code_state.json.

---

## 6) What To Do If Something Fails (Debug Playbook)

### 6.1 If capture out-of-range occurs in paper baseline (without replacement)

This is a hard failure. Do not proceed to full_dataset.

Steps:
- Check results/<run>/integrity_diagnostics.jsonl for per-sample entries with E0/E_res and identifiers.
- Confirm whether the failure is in DT, OMP, or Random.
- If OMP monotonicity violations > 0, suspect an evaluator bug (projection update/masking).
- If DT forced-K duplicates > 0, suspect masking failure.
- If only Random fails and random_sampling is without_replacement, that indicates a bug; Random should be unique.

### 6.2 If RTG controllability looks weak in the mean

Do not rely only on mean. Check:
- P(k_selected < max_k) vs lambda
- histogram and tail behavior

If still weak on speech:
- This may indicate STOP calibration or RTG scaling is insufficient even in-domain.
- Propose a follow-up training experiment (out of scope for E4h-Speech evaluation) and document it in Next Steps.

### 6.3 If MPS stalls or is unstable

Switch to cpu for DT inference and document it.

---

## 7) Deliverables Checklist (What Must Exist Before Declaring Done)

For each run directory:
- subset_manifest.json exists and validates (no missing files, no MD5 mismatches)
- run.log exists with exact command line
- summary JSONs exist:
  - summary/compute_matched_summary.json
  - summary/forced_k_summary.json
  - summary/rtg_controllability_summary.json
- integrity_diagnostics.jsonl exists
- code_state.json exists
- ACCEPTANCE_REPORT.md exists and is fully filled (English, causal language)

E4h-Speech is complete when:
- scale_check_subset PASS (paper baseline)
- full_dataset PASS (paper baseline)
- guardrail diagnostic run executed (may FAIL as expected; must be documented)

