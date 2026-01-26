# Acceptance Report: E4n-Speech -- Phase Equalization FIT (smoke, paired)

## 1) Executive Summary

- Run: `results/rtgomp_phase_eq_E4n_speech_fit_smoke_paired_20260126_123636/`
- Run type: FIT
- Mode: smoke
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1 (FIT): Does the fitter run end-to-end on real speech WAV and produce valid phase_eq artifacts? PASS.

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- MPLCONFIGDIR: `/tmp/mpl`

- code_state.json: `results/rtgomp_phase_eq_E4n_speech_fit_smoke_paired_20260126_123636/code_state.json`

- Subset manifest: `results/rtgomp_phase_eq_E4n_speech_fit_smoke_paired_20260126_123636/subset_manifest.json`
  - num_pairs = 1
  - wav-only: YES
  - any .npy: NO

## 3) Exact Commands (Required)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_fit_smoke_paired_20260126_123636"
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

## 4) Results (Required)

Hard guardrails:
- wav-only manifest: PASS
- fs mismatch encountered: NO
- any NaN/Inf in outputs: NO

Artifacts:
- `phase_eq/phase_eq.npz`: YES
- `phase_eq/phase_eq_fit_summary.json`: YES

From `phase_eq/phase_eq_fit_summary.json`:
- num_windows_total / used = 5 / 5
- fraction_defined_tau = 1.0
- boundary_hit_rate = 0.0
- gcc_psr_p50 = 60.19
- inband_defined_fraction_stft = 1.0
- phase_eq_unit_mag_max_err = 4.03e-08

Decision:
- PASS BECAUSE this smoke run verifies the fitter starts, enforces wav-only + fs=16000, and produces numerically valid phase_eq artifacts.

Notes:
- Deeper calibration effectiveness is evaluated in the full_dataset EVAL report:
  - `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/ACCEPTANCE_REPORT.md`

