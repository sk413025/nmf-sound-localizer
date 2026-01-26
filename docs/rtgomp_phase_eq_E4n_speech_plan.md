# Plan: E4n-Speech -- Phase Equalization Calibration (Phase-2d) + E4m Re-validation

This plan executes E4n-Speech end-to-end:
1) Fit a reproducible LDV phase equalizer on paired speech WAV.
2) Apply the phase equalizer and re-run the E4m evaluator (dispersion diagnostics) on the same 5-point lambda grid.

Source of truth:
- `docs/rtgomp_phase_eq_E4n_speech_spec.md`

Non-negotiables (operational):
- Real data only; WAV only; fs must be 16000; no resampling.
- Every run uses lockdir + `tee -a` and produces artifacts under `results/<run>/`.
- All writing in English.
- No planning-only commits: docs + executed artifacts must be committed atomically in the Results commit.

---

## 0) Fixed Inputs (Do Not Change)

Speech roots:
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

DT checkpoint (eval runs):
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Lambda grid (eval runs; exact):
- `1e-5,3e-5,1e-4,2e-4,3e-4`

E4m evaluator (for re-validation):
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

E4n fitter (to implement):
- `scripts/h_exploration/fit_phase_eq_e4n_speech.py`

---

## 1) Preflight (Fail Fast)

### 1.1 Verify speech roots exist and contain WAVs

```bash
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC/*.wav | head
ls -1 /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV/*.wav | head
```

If either glob fails: STOP.

### 1.2 Verify dataset length is stable

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

---

## 2) Common Environment Prelude

```bash
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/mpl
```

Recommended device for eval runs:
- `--device cpu`

---

## 3) Implement Required Code Changes (Before Running Long Jobs)

### 3.1 Create fitter: `scripts/h_exploration/fit_phase_eq_e4n_speech.py`

Must:
- Load speech WAV dataset (DoALagDataset with angle=None, hop_length=160).
- Enforce wav-only and fs==16000 (fail fast).
- Iterate windows using the evaluator’s valid_starts rule.
- Compute GCC-PHAT tau per window (center=0, radius=search_radius_frames*hop_length).
- Compute weighted circular mean to estimate `phase_eq_rfft` (nfft_fit=32768).
- Downsample to STFT bins using exact bin alignment:
  - require `nfft_fit % n_fft == 0`
  - `stride = nfft_fit // n_fft` (expected 16)
  - take `phase_eq_stft[k] = phase_eq_rfft[stride * k]` to produce 1025 bins.
- Write artifacts under `results/<run>/phase_eq/` as required by the spec.

### 3.2 Extend evaluator to apply phase_eq

In `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`:
- Add CLI flags:
  - `--phase_eq_path <path-to-phase_eq.npz>` (optional)
  - `--apply_phase_eq 0/1` (default 0)
- When `apply_phase_eq=1`:
  - Load `phase_eq_stft` and multiply `ldv_stft` by it before evaluation.
  - Load `phase_eq_rfft` and apply it inside the subsample GCC/phase-slope path by multiplying the LDV rFFT spectrum before computing cross_power / GCC.
- Hard-fail on shape mismatches or non-unit-magnitude.

---

## 4) Run Sequence

Each run uses:
- lockdir: `OUT_DIR/.lock`
- log capture: `|& tee -a OUT_DIR/run.log`

### 4.1 FIT: smoke (paired; num_pairs=1)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_fit_smoke_paired_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/fit_phase_eq_e4n_speech.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --out_dir "$OUT_DIR" \
  --mode smoke --num_pairs 1 \
  --pairing_mode paired --require_wav_only 1 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --search_radius_frames 2 --seed 0 \
  |& tee -a "$OUT_DIR/run.log"
```

### 4.2 FIT: functional (paired; scale_check_subset; num_pairs=48)

Same as smoke but:
- `--mode scale_check_subset --num_pairs 48`

### 4.3 FIT: full_dataset (paired; num_pairs=416)

Same as smoke but:
- `--mode full_dataset`

Record the produced phase eq path:
- `PHASE_EQ_PATH="<fit_full_dataset_out>/phase_eq/phase_eq.npz"`

### 4.4 RE-VALIDATION (E4m-style eval with apply): smoke (paired; num_pairs=1)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_eval_smoke_paired_$(date +%Y%m%d_%H%M%S)"
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
  --write_subsample_delay_diagnostics 1 --subsample_method gcc_phat,phase_slope --search_radius_frames 2 \
  --apply_phase_eq 1 --phase_eq_path "$PHASE_EQ_PATH" \
  |& tee -a "$OUT_DIR/run.log"
```

### 4.5 RE-VALIDATION: functional (paired; 48)

Same as 4.4 but:
- `--mode scale_check_subset --num_pairs 48`
- `--pairing_mode paired`

### 4.6 RE-VALIDATION: functional (mispair_shift1; 48)

Same as 4.4 but:
- `--mode scale_check_subset --num_pairs 48`
- `--pairing_mode mispair_shift1`

### 4.7 RE-VALIDATION: full_dataset (paired; 416)

Same as 4.4 but:
- `--mode full_dataset`
- `--pairing_mode paired`

---

## 5) Mandatory Artifacts Per Run

Every FIT run must include:
- subset_manifest.json (wav-only)
- run.log
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled, English)
- phase_eq/phase_eq.npz
- phase_eq/phase_eq_fit_summary.json

Every EVAL run must include the full E4m artifact set plus:
- code_state.json (manual)
- ACCEPTANCE_REPORT.md (filled, English)
- and must record the exact `phase_eq_path` used.

---

## 6) code_state.json (Mandatory)

After each run, write `results/<run>/code_state.json` (manual), including at least:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py
- scripts/h_exploration/fit_phase_eq_e4n_speech.py
- scripts/h_exploration/dataset_lag.py

---

## 7) Results Commit (Mandatory)

After all required runs complete and reports are filled:
- Create a single atomic Results commit containing:
  - E4n docs (spec/plan/template/prompt)
  - code changes (fitter + evaluator extension)
  - all E4n run artifacts under results/<run>/ (use `git add -f`)
