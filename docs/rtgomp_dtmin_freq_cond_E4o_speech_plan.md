# Plan: E4o-Speech — DTmin Frequency Conditioning Audit

This plan executes E4o-Speech end-to-end:
1) Extend the evaluator with frequency-conditioning ablations.
2) Run a small suite (smoke + scale_check_subset) to classify whether `freq_idx` is used.
3) (Optional) Run per-band evaluations to quantify band sensitivity.

Source of truth:
- `docs/rtgomp_dtmin_freq_cond_E4o_speech_spec.md`

Non-negotiables:
- Real data only; WAV only; fs must be 16000; no resampling.
- All writing and artifacts in English.
- Every run uses lockdir + `tee -a` and produces artifacts under `results/<run>/`.

---

## 0) Fixed Inputs (Do Not Change)

Speech roots:
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

DT checkpoint:
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Lambda grid:
- `1e-5,3e-5,1e-4,2e-4,3e-4`

Evaluator:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

---

## 1) Preflight (Fail Fast)

### 1.1 Verify speech roots exist and contain WAVs

```bash
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav | head
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav | head
```

If either glob fails: STOP.

### 1.2 Verify dataset length is stable (expected 416)

```bash
PYTHONPATH=. conda run --no-capture-output -n trl-training python - <<'PY'
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
- `num_pairs == 416`
- first/last pairs are `.wav` under the speech roots.

### 1.3 Verify checkpoint exists

```bash
test -f results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth
```

If missing: STOP.

---

## 2) Common Environment Prelude

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

Recommended device:
- `--device cpu`

---

## 3) Implement Required Code Changes (Before Running)

In `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`:
- Add flags: `--freq_cond_mode`, `--freq_cond_seed`, `--freq_cond_constant_idx`.
- Apply ablation to `freq_ids` passed into DT inference only (do not modify physics / OMP / Random).
- Write `summary/freq_cond_audit_summary.json` in each run.

---

## 4) Run Sequence (Minimum Suite)

Each run uses:
- lockdir: `OUT_DIR/.lock`
- log capture: `|& tee -a OUT_DIR/run.log`

Shared args (copy/paste block; keep identical across runs):
- `--mic_root ... --ldv_root ...`
- `--ckpt_path .../dt_freq_aware_best.pth`
- `--hop_length 160 --fs 16000 --n_fft 2048 --freq_min 300 --freq_max 3000`
- `--max_lag 50 --max_k 16 --tw 32 --gain 100.0 --rtg_dim 2`
- `--lambda_c_values "1e-5,3e-5,1e-4,2e-4,3e-4"`
- `--random_trials 3 --random_sampling without_replacement --seed 0`
- `--write_per_sample 0 --device cpu`
- `--pairing_mode paired --require_wav_only 1`
- `--write_delay_diagnostics 1`

### 4.1 Smoke (paired; num_pairs=1; normal)

```bash
OUT_DIR="results/rtgomp_dtmin_freq_cond_E4o_speech_smoke_normal_$(date +%Y%m%d_%H%M%S)"
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
  --write_per_sample 0 --device cpu \
  --pairing_mode paired --require_wav_only 1 \
  --write_delay_diagnostics 1 \
  --freq_cond_mode normal --freq_cond_seed 0 --freq_cond_constant_idx 200 \
  |& tee -a "$OUT_DIR/run.log"
```

### 4.2 Functional suite (paired; scale_check_subset; num_pairs=48)

Run 4 times with different `--freq_cond_mode` values:
- `normal`
- `shuffle` (use `--freq_cond_seed 0`)
- `constant` (use `--freq_cond_constant_idx 200`)
- `zero_embed`

Naming convention for out_dir:
- `results/rtgomp_dtmin_freq_cond_E4o_speech_scale48_<MODE>_<timestamp>/`

---

## 5) Optional: Band Sensitivity Runs (normal only)

Repeat scale_check_subset (48 pairs) with:
- band_lo: `--freq_min 300 --freq_max 900`
- band_mid: `--freq_min 900 --freq_max 1800`
- band_hi: `--freq_min 1800 --freq_max 3000`

Keep `--freq_cond_mode normal`.

---

## 6) Mandatory Artifacts Per Run

Each run must include:
- `subset_manifest.json`
- `run.log`
- `code_state.json` (manual)
- `ACCEPTANCE_REPORT.md` (filled; English)
- `summary/compute_matched_summary.json`
- `summary/rtg_controllability_summary.json`
- `summary/freq_cond_audit_summary.json` (new)

---

## 7) code_state.json (Mandatory)

After each run, write `results/<run>/code_state.json`, including at minimum:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py
- scripts/h_exploration/dataset_lag.py

