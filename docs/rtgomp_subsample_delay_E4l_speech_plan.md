# Plan: E4l-Speech -- Sub-sample Delay Refinement Diagnostics (GCC-PHAT / Phase-Slope) Under Compute Control

This plan defines how to execute **E4l-Speech** end-to-end, producing complete, reproducible artifacts.

Source of truth:
- docs/rtgomp_subsample_delay_E4l_speech_spec.md
- docs/rtgomp_subsample_delay_E4l_speech_acceptance_report_template.md

Implementation to run:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py (extended with E4l flags)

---

## 0) Non-Negotiables (Operational Guardrails)

From AGENTS.md:
- Real data only. If speech roots are missing: FAIL fast and document prerequisites. No synthetic substitutes.
- No silent fallbacks/coercions. Prefer explicit validation and clear error messages.
- Run sequentially with lockdirs, and log via tee -a.
- All artifacts under results/<run>/ (logs, manifests, fingerprints, summaries, acceptance report).
- All writing must be in English.
- Do not create planning-only commits. (These docs must be committed together with the executed E4l results commit.)

---

## 1) Fixed Inputs (Do Not Change)

### 1.1 Speech roots (wav only)

- mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Expected dataset length:
- len(dataset) == 416 pairs (DoALagDataset(angle=None, hop_length=160))

### 1.2 Checkpoint (DT)

Use the calibrated speech checkpoint:
- results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth

### 1.3 Lambda grid (5 points; required)

- lambda_c_values = 1e-5,3e-5,1e-4,2e-4,3e-4

### 1.4 E4l sub-sample settings (defaults)

- subsample_method = gcc_phat
- search_radius_frames = 2
- require_wav_only = 1
- pairing_mode = paired / mispair_shift1

---

## 2) Preflight Checklist (Fail Fast)

### 2.1 Verify speech roots exist and contain WAVs

```bash
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav | head
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav | head
```

If either glob fails: STOP.

### 2.2 Verify dataset pairing and length

```bash
PYTHONPATH=. conda run -n trl-training python - <<'PY'
from scripts.h_exploration.dataset_lag import DoALagDataset
mic='/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC'
ldv='/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV'
ds=DoALagDataset(mic,ldv,angle=None,hop_length=160)
print('num_pairs',len(ds))
print('first_pair',ds.clips[0])
print('last_pair',ds.clips[-1])
PY
```

Acceptance:
- num_pairs is stable (expected 416 on this machine).
- first/last pairs are WAV paths under the speech roots.

---

## 3) Implementation Work (Must Be Done Before Running Long Jobs)

E4l requires extending the evaluator to compute sub-sample delay diagnostics.

### 3.1 Required evaluator changes

In scripts/h_exploration/run_rtgomp_e4h_paper_eval.py:
- Add flags:
  - --write_subsample_delay_diagnostics 1/0
  - --subsample_method gcc_phat[,phase_slope]
  - --search_radius_frames <int>
- Read WAV waveforms for each clip (fail-fast if fs != 16000).
- For each window (clip_idx, window_idx, start_t) and each lambda_c:
  - Compute coarse lag (DT and OMP median first lag frames).
  - Extract waveform segment aligned to STFT window.
  - Run GCC-PHAT within +/- search_radius_frames around the coarse estimate.
  - Emit subsample_delay_diagnostics.jsonl and summary/subsample_delay_diagnostics_summary.json.

No fallbacks:
- Do not resample.
- Do not fabricate estimates when undefined; write null + counters.

### 3.2 Smoke-level sanity check (before running)

Run E4l smoke (paired, num_pairs=1) and confirm:
- subsample_delay_diagnostics.jsonl exists and is non-empty
- summary/subsample_delay_diagnostics_summary.json exists
- wav-only guardrail passes

---

## 4) Common Environment Prelude (Copy/Paste)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

Recommended device:
- Use `--device cpu` for stability.

---

## 5) Run Sequence (Smoke -> Functional -> Full Dataset)

Always:
- use lockdir OUT_DIR/.lock
- capture logs via tee -a OUT_DIR/run.log

### 5.1 Smoke (paired; required)

```bash
OUT_DIR="results/rtgomp_subsample_delay_E4l_speech_smoke_paired_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode smoke --num_pairs 1 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 1 --device cpu \
  --pairing_mode paired --require_wav_only 1 \
  --write_delay_diagnostics 1 \
  --write_subsample_delay_diagnostics 1 --subsample_method gcc_phat --search_radius_frames 2 \
  |& tee -a "$OUT_DIR/run.log"
```

Post-step required actions:
- Create code_state.json.
- Fill ACCEPTANCE_REPORT.md using the E4l template.

### 5.2 Functional (scale_check_subset; required)

Run two functional runs:

1) paired (positive path)
2) mispair_shift1 (guardrail)

Use:
- mode=scale_check_subset --num_pairs 48
- write_per_sample=1
- write_subsample_delay_diagnostics=1

### 5.3 Full dataset (paired; required)

Use:
- mode=full_dataset
- write_per_sample=0 (recommended)
- write_subsample_delay_diagnostics=1

---

## 6) Mandatory Artifacts Per Run

Under results/<run>/:
- subset_manifest.json
- run.log
- integrity_diagnostics.jsonl
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- delay_diagnostics.jsonl (from E4k-style)
- summary/delay_diagnostics_summary.json
- subsample_delay_diagnostics.jsonl (E4l-specific)
- summary/subsample_delay_diagnostics_summary.json (E4l-specific)
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled template)

---

## 7) code_state.json (Mandatory)

Create code_state.json after each run (same as E4k):

Minimum schema:
```json
{
  "git_head": "...",
  "dirty": true,
  "files": {
    "scripts/h_exploration/run_rtgomp_e4h_paper_eval.py": "sha256...",
    "scripts/h_exploration/dataset_lag.py": "sha256..."
  }
}
```

---

