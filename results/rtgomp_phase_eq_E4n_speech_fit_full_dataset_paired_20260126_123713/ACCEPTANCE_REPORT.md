# Acceptance Report: E4n-Speech -- Phase Equalization FIT (full_dataset, paired)

## 1) Executive Summary

- Run: `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/`
- Run type: FIT
- Mode: full_dataset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1 (FIT): Did we estimate a valid, reproducible phase equalizer (unit magnitude, wav-only, stable GCC)? PASS.

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

### 2.1 Environment

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- Device(s): cpu (fit stage runs on CPU-only numpy/scipy)
- MPLCONFIGDIR: `/tmp/mpl`

### 2.2 Code provenance (Required)

- code_state.json: `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/code_state.json`
  - git_head: `18a82aac2d8a5c391bd31b830d9754cffecec6fa`
  - dirty: `true`
  - sha256 files:
    - `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`: `1a1b0c4ccb4eed0cf0d3140b3beb62cff633d9e02400f7aac49245964685d911`
    - `scripts/h_exploration/fit_phase_eq_e4n_speech.py`: `5125bed0dbe84bf1a88649d7550423f00c7638207d0dd0a599e452a665d1e85b`
    - `scripts/h_exploration/dataset_lag.py`: `6858eca5d523d5823dd5f8623c03863d65fb103f75ba01b279f01717437cd5eb`

### 2.3 Data lineage (Speech-only; Required)

- mic_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- Pairing mode: `paired`
- Subset manifest: `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/subset_manifest.json`
  - num_pairs = 416
  - fingerprint_md5 = `13356fdb74d2acb7e85361a4dfe5c3d2`
  - Domain validation:
    - all paths end with .wav: YES
    - any .npy paths: NO

### 2.4 Fixed parameters (Must be explicit)

- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- search_radius_frames = 2
- tau_center_samples = 0 (design choice)
- nfft_fit = 32768

## 3) Exact Commands (Required)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/fit_phase_eq_e4n_speech.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --out_dir "$OUT_DIR" \
  --mode full_dataset \
  --pairing_mode paired --require_wav_only 1 \
  --fs 16000 --hop_length 160 --n_fft 2048 --freq_min 300 --freq_max 3000 \
  --max_lag 50 --tw 32 --search_radius_frames 2 --seed 0 \
  |& tee -a "$OUT_DIR/run.log"
```

## 4) Results (Required)

### 4.0 Hard guardrails

- wav-only manifest: PASS
- fs mismatch encountered: NO
- any NaN/Inf in outputs: NO
- phase_eq artifacts exist: PASS
  - `phase_eq/phase_eq.npz`: YES
  - `phase_eq/phase_eq_fit_summary.json`: YES

Decision:
- PASS BECAUSE wav-only and fs=16000 are enforced and the required phase_eq artifacts were written.

### 4.1 FIT stage: phase_eq validity

From `phase_eq/phase_eq_fit_summary.json`:
- num_windows_total = 1974
- num_windows_used (tau defined) = 1974
- fraction_defined (tau) = 1.0
- boundary_hit_rate = 0.0
- gcc_psr_p50/p90 = 53.87 / 84.25
- inband_defined_fraction_stft = 1.0
- phase_eq_unit_mag_max_err = 4.13e-08

Decision:
- PASS BECAUSE (1) GCC-PHAT tau is defined for all windows with zero boundary hits, and (2) phase_eq is unit-magnitude with full in-band coverage; THEREFORE the equalizer is numerically valid and reproducible.

## 5) Notes / Link to EVAL

This FIT run produces the phase_eq used by all E4n EVAL runs:
- `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/phase_eq/phase_eq.npz`

Interpretation of whether this calibration improves dispersion is reported in:
- `results/rtgomp_phase_eq_E4n_speech_eval_full_dataset_paired_20260126_125101/ACCEPTANCE_REPORT.md`

