# Plan: E4k-Speech -- Delay & Phase-Consistency Diagnostics (Paired vs Mispair; 5-Point Lambda Grid)

This plan defines how to execute **E4k-Speech** end-to-end, producing complete, reproducible artifacts.

Source of truth:
- docs/rtgomp_delay_diagnostics_E4k_speech_spec.md
- docs/rtgomp_delay_diagnostics_E4k_speech_acceptance_report_template.md

Implementation to run:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

---

## 0) Non-Negotiables (Operational Guardrails)

From AGENTS.md:
- Real data only. If speech roots are missing: FAIL fast and document prerequisites. No synthetic substitutes.
- No silent fallbacks/coercions. Prefer explicit validation and clear error messages.
- Run sequentially with lockdirs, and log via tee -a.
- All artifacts under results/<run>/ (logs, manifests, fingerprints, summaries, acceptance report).
- All writing must be in English.
- Do not create planning-only commits. Any new/updated docs must be committed together with executed results (atomic).

---

## 1) Fixed Inputs (Do Not Change)

### 1.1 Speech roots (wav only)

- mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Expected dataset length check (on this machine):
- len(dataset) == 416 pairs (DoALagDataset(angle=None, hop_length=160))

### 1.2 Checkpoint (DT)

Use the calibrated speech checkpoint:
- results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth

### 1.3 Evaluator parameters (fixed)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq_min = 300
- freq_max = 3000
- max_lag = 50 (M_lags = 101)
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2
- lambda_c_values = 1e-5,3e-5,1e-4,2e-4,3e-4

### 1.4 Pairing modes (two separate runs)

- paired
- mispair_shift1 (guardrail diagnostic)

---

## 2) Preflight Checklist (Fail Fast)

### 2.1 Verify speech roots exist and contain WAVs

```bash
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav | head
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav | head
```

If either glob fails: STOP (missing prerequisite).

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
- num_pairs is stable (expected 416 on this machine)
- first/last pairs are WAV paths under the speech roots

---

## 3) Common Environment Prelude (Copy/Paste)

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

Recommended device:
- Use `--device cpu` for stability.

---

## 4) Run Sequence (Smoke -> Functional -> Full Dataset)

Always use:
- lockdir: OUT_DIR/.lock
- tee: `|& tee -a OUT_DIR/run.log`

### 4.1 Smoke (paired; required)

Purpose:
- Confirm end-to-end execution, guardrails, and that delay diagnostics files are produced.

Config:
- mode = smoke
- num_pairs = 1
- random_trials = 3
- random_sampling = without_replacement
- write_per_sample = 1 (small; required for smoke)
- write_delay_diagnostics = 1 (required for E4k)
- require_wav_only = 1 (hard guardrail)
- pairing_mode = paired

Command:
```bash
OUT_DIR="results/rtgomp_delay_diag_E4k_speech_smoke_paired_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
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
  --pairing_mode paired --require_wav_only 1 --write_delay_diagnostics 1 \
  |& tee -a "$OUT_DIR/run.log"
```

Post-step required actions:
- Create code_state.json (Section 5).
- Fill ACCEPTANCE_REPORT.md using the E4k template.

### 4.2 Functional (scale_check_subset; required)

Run two functional runs on the deterministic 48-pair subset:

#### 4.2.1 Positive path (paired)

Config:
- mode = scale_check_subset (48 pairs)
- pairing_mode = paired
- write_per_sample = 1
- write_delay_diagnostics = 1

#### 4.2.2 Guardrail diagnostic (mispair_shift1)

Config:
- mode = scale_check_subset (48 pairs)
- pairing_mode = mispair_shift1
- write_per_sample = 1
- write_delay_diagnostics = 1

Commands (paired then mispair):
```bash
# Paired
OUT_DIR="results/rtgomp_delay_diag_E4k_speech_scale_check_subset_paired_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode scale_check_subset --num_pairs 48 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 1 --device cpu \
  --pairing_mode paired --require_wav_only 1 --write_delay_diagnostics 1 \
  |& tee -a "$OUT_DIR/run.log"

# Mispair (guardrail)
OUT_DIR="results/rtgomp_delay_diag_E4k_speech_scale_check_subset_mispair_shift1_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode scale_check_subset --num_pairs 48 \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 3 --random_sampling without_replacement --seed 0 \
  --write_per_sample 1 --device cpu \
  --pairing_mode mispair_shift1 --require_wav_only 1 --write_delay_diagnostics 1 \
  |& tee -a "$OUT_DIR/run.log"
```

For each run:
- Generate code_state.json.
- Fill an acceptance report.

### 4.3 Full dataset (paired; required for paper-grade summary + diagnostics)

Config:
- mode = full_dataset
- pairing_mode = paired
- random_trials = 1 (recommended)
- write_per_sample = 0 (recommended; too large otherwise)
- write_delay_diagnostics = 1 (required; window-level only, should be manageable)

Command:
```bash
OUT_DIR="results/rtgomp_delay_diag_E4k_speech_full_dataset_paired_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir "$OUT_DIR" \
  --mode full_dataset \
  --hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2 \
  --lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4" \
  --random_trials 1 --random_sampling without_replacement --seed 0 \
  --write_per_sample 0 --device cpu \
  --pairing_mode paired --require_wav_only 1 --write_delay_diagnostics 1 \
  |& tee -a "$OUT_DIR/run.log"
```

Post-step required actions:
- Create code_state.json.
- Fill ACCEPTANCE_REPORT.md (must reference delay diagnostics summary).

---

## 5) How to Create code_state.json (Mandatory)

The evaluator does not auto-write code_state.json. Create it manually after each run.

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

Recommended commands:
```bash
OUT_DIR="results/<your_run_dir>"
git rev-parse HEAD
git status --porcelain
python - <<'PY'
import hashlib, json, pathlib, subprocess
out_dir = pathlib.Path("results/<your_run_dir>")
files = [
  "scripts/h_exploration/run_rtgomp_e4h_paper_eval.py",
  "scripts/h_exploration/dataset_lag.py",
]
def sha256(p):
  h=hashlib.sha256()
  with open(p,"rb") as f:
    for b in iter(lambda: f.read(1024*1024), b""):
      h.update(b)
  return h.hexdigest()
git_head = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
dirty = subprocess.check_output(["git","status","--porcelain"], text=True).strip() != ""
obj = {"git_head": git_head, "dirty": dirty, "files": {p: sha256(p) for p in files}}
(out_dir/"code_state.json").write_text(json.dumps(obj, indent=2))
print("wrote", out_dir/"code_state.json")
PY
```

---

