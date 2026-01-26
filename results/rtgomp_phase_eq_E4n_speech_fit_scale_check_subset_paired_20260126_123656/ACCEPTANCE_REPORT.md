# Acceptance Report: E4n-Speech -- Phase Equalization FIT (scale_check_subset, paired)

## 1) Executive Summary

- Run: `results/rtgomp_phase_eq_E4n_speech_fit_scale_check_subset_paired_20260126_123656/`
- Run type: FIT
- Mode: scale_check_subset
- Pairing mode: paired
- Outcome: PASS

Primary questions answered:
- Q1 (FIT): Is phase_eq fitting stable beyond smoke scale (48 pairs) on real speech WAV? PASS.

Dataset domain statement (required):
- This run uses the speech WAV dataset only (no .npy files in the manifest): YES

## 2) Setup (Required)

- Conda env: `trl-training`
- Python: `Python 3.11.13`
- MPLCONFIGDIR: `/tmp/mpl`

- code_state.json: `results/rtgomp_phase_eq_E4n_speech_fit_scale_check_subset_paired_20260126_123656/code_state.json`

- Subset manifest: `results/rtgomp_phase_eq_E4n_speech_fit_scale_check_subset_paired_20260126_123656/subset_manifest.json`
  - num_pairs = 48
  - wav-only: YES
  - any .npy: NO

## 3) Exact Commands (Required)

```bash
OUT_DIR="results/rtgomp_phase_eq_E4n_speech_fit_scale_check_subset_paired_20260126_123656"
mkdir -p "$OUT_DIR/summary"
LOCKDIR="$OUT_DIR/.lock"
if ! mkdir "$LOCKDIR" 2>/dev/null; then echo "ERROR: lock exists: $LOCKDIR" >&2; exit 1; fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

MPLCONFIGDIR=/tmp/mpl PYTHONPATH=. conda run --no-capture-output -n trl-training python -u scripts/h_exploration/fit_phase_eq_e4n_speech.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --out_dir "$OUT_DIR" \
  --mode scale_check_subset --num_pairs 48 \
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
- num_windows_total / used = 234 / 234
- fraction_defined_tau = 1.0
- boundary_hit_rate = 0.0
- gcc_psr_p50/p90 = 58.19 / 82.20
- inband_defined_fraction_stft = 1.0
- phase_eq_unit_mag_max_err = 4.18e-08

Decision:
- PASS BECAUSE phase_eq fitting remains stable at functional-test scale with full tau coverage, no boundary hits, and unit-magnitude equalizer.

Notes:
- The phase_eq used for re-validation is produced by the full_dataset FIT run:
  - `results/rtgomp_phase_eq_E4n_speech_fit_full_dataset_paired_20260126_123713/`

