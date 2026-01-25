# Spec: E4k-Speech -- Delay & Phase-Consistency Diagnostics (Compute-Conditioned DT vs OMP; Paired vs Mispair Guardrail)

This spec defines **E4k-Speech**, a **Phase-2a diagnostic** experiment that bridges Phase-1 "reconstruction + compute
controllability" to the ultimate project goal: **enabling MIC+LDV time alignment / TDoA**.

E4k-Speech does **not** claim that TDoA is solved. Instead, it answers a necessary prerequisite question:

> Can we derive a stable, frequency-consistent **coarse delay estimate** (in STFT-frame lags) from our lag-selection
> policy, and does that estimate degrade when we reduce compute (via lambda_c / RTG), and does it fail under a
> deliberately wrong MIC↔LDV pairing (guardrail mispair)?

The intended executor is an engineer with **no prior signal processing or RL background**. This spec includes all
definitions needed to implement and validate the experiment.

Implementation to be executed:
- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py (extended with E4k-specific flags/output)

---

## 0) Phase Context (Why This Experiment Exists)

### 0.1 Original research motivation (high-level)

We want to use a near-source LDV measurement to compensate for low-SNR far-field microphone data and still obtain a
meaningful time alignment signal for downstream TDoA.

However, LDV signals often exhibit frequency-dependent phase distortions (material resonances, dispersion), which can
make naive cross-correlation / TDoA unstable.

### 0.2 What Phase-1 established (already done)

Phase-1 (E4h/E4j-Speech) established:
- The MIC→LDV relation in the STFT domain is **learnable** (DT approximates OMP and beats Random).
- Compute is **controllable** via lambda_c / RTG (STOP calibration provides monotonic control of mean k_selected).

### 0.3 What Phase-2a (this spec) must establish

Phase-2a must establish whether:
- There exists a **stable coarse alignment** between MIC and LDV that is consistent across frequency bins.
- Reducing compute (higher lambda_c) makes this alignment less stable (or not).
- A deliberate wrong pairing breaks the alignment diagnostics (guardrail).

This informs whether it is justified to proceed to Phase-2b/3:
- Sub-sample delay refinement (e.g., GCC-PHAT / phase-slope) and
- TDoA validation with geometry/ground truth.

---

## 1) Minimal Primer (Only What You Need)

### 1.1 STFT frames and "lag in frames"

We compute STFT with:
- fs = 16000 Hz
- hop_length = 160 samples

One STFT frame step corresponds to:

  hop_seconds = hop_length / fs = 160 / 16000 = 0.01 seconds = 10 ms

Therefore, a lag of +1 frame corresponds to +10 ms of delay (coarse).

**Important limitation:** This frame-level lag is too coarse to be a final TDoA. E4k-Speech treats it as a coarse
alignment diagnostic only.

### 1.2 The lag dictionary and selection

For each clip, for each time window (Tw frames), and for each frequency bin f in the band [300, 3000] Hz:

- Target: LDV STFT window Y_f (shape: Tw)
- Dictionary: MIC STFT windows shifted by lags in [-max_lag, +max_lag]

We select lag indices (atoms) and solve least squares to explain Y_f.

### 1.3 "First selected lag" as a coarse delay proxy

At the first selection step (k=1), both OMP and DT choose a single lag atom that best reduces the residual.

Define:
- M_lags = 2*max_lag + 1
- Lag_Min = -max_lag

Action index a in [0, M_lags-1] maps to a lag in frames:

  lag_frames = a + Lag_Min

For DT with STOP:
- If DT chooses STOP at the first step for a frequency bin, the "first lag" is undefined for that bin.

We will compute statistics across frequency bins within a time window:
- If the system behaves like a near-pure delay, the first-lag distribution should be narrow across frequency.
- If the LDV exhibits strong frequency-dependent phase distortion, the first-lag distribution may be wide (high
  dispersion).

---

## 2) What E4k-Speech Produces (Outputs and Metrics)

E4k-Speech adds a **window-level delay diagnostics** output on top of the existing E4h-Speech capture/compute summaries.

### 2.1 Per-window JSONL diagnostics (required)

For each (clip_idx, window_idx) and for each lambda_c in the grid, write exactly one JSON object to:
- results/<run>/delay_diagnostics.jsonl

Each row must include at minimum:

- Identifiers:
  - pairing_mode: "paired" or "mispair_shift1"
  - clip_idx: int
  - window_idx: int
  - lambda_c: float
  - start_t: int (STFT frame index of window start)

- Fixed config (for reproducibility):
  - fs, hop_length, n_fft, freq_min, freq_max, max_lag, tw, max_k
  - F_band: number of frequency bins in [freq_min, freq_max]

- DT (free rollout with STOP):
  - dt_stop0_frac: fraction of band frequency bins that chose STOP at the first decision step
  - dt_first_lag_defined_frac: 1 - dt_stop0_frac
  - dt_first_lag_median_frames: median of first lag (across frequency bins where defined)
  - dt_first_lag_mad_frames: median absolute deviation of first lag (across defined bins)
  - dt_first_lag_abs_le_1_frac: fraction of defined bins with |lag| <= 1
  - dt_first_lag_abs_le_2_frac: fraction of defined bins with |lag| <= 2
  - dt_k_selected_mean: mean k_selected across frequency bins (compute-aligned, window-level)
  - dt_steps_decision_mean: mean steps_decision across frequency bins (policy-step-aligned, window-level)
  - dt_capture_mean: mean capture across band frequency bins (window-level)

- OMP (step-1 only, no STOP):
  - omp_first_lag_median_frames
  - omp_first_lag_mad_frames

- DT vs OMP agreement (diagnostic):
  - dt_vs_omp_first_lag_match_frac: fraction of DT-defined bins where DT first lag equals OMP first lag

**No fallbacks:**
- If a metric is undefined (e.g., DT defined bins = 0), write null and log counts explicitly. Do not substitute zeros.

### 2.2 Summary JSON (required)

Write:
- results/<run>/summary/delay_diagnostics_summary.json

This must aggregate the per-window JSONL into one row per lambda_c, reporting:
- number of windows used
- mean/std and p50/p90 for dt_first_lag_mad_frames
- mean for dt_stop0_frac and dt_k_selected_mean
- mean for dt_vs_omp_first_lag_match_frac

---

## 3) Experimental Conditions

### 3.1 Dataset domain (speech WAV only; required)

Allowed dataset roots (do not change):
- mic_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root = /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Guardrail:
- subset_manifest.json must contain only `.wav` paths (no `.npy`). Any `.npy` is a hard failure.

### 3.2 Pairing modes

We evaluate two pairing modes (two separate runs):

1) paired (paper-like, correct pairing)
   - MIC and LDV are paired by filename rule (MIC -> LDV).

2) mispair_shift1 (guardrail diagnostic; still real data)
   - MIC files remain in the paired dataset order.
   - LDV files are shifted by +1 index in that same order (cyclic wrap).
   - This deliberately breaks synchronization/content alignment.

Expectation:
- mispair_shift1 should degrade capture AND increase lag dispersion (dt_first_lag_mad_frames), compared to paired.

### 3.3 Lambda grid (compute knob; required)

Use exactly this 5-point grid:
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

We will use the trend across this grid to test whether delay stability degrades as compute decreases.

---

## 4) Hard Guardrails (FAIL FAST)

E4k-Speech inherits all E4h-Speech hard guardrails:

- Missing files: FAIL
- MD5 mismatches vs manifest: FAIL
- NaN/Inf in any capture: FAIL
- capture out-of-range (< -1e-6 or > 1+1e-6) under paper baseline: FAIL
- OMP monotonicity violations (residual energy increases beyond tolerance): FAIL
- DT duplicate actions under forced-K: FAIL

Additional E4k-Speech guardrails:
- Manifest wav-only requirement: FAIL if any path does not end with ".wav" when require_wav_only=1.
- delay_diagnostics.jsonl must be produced when write_delay_diagnostics=1: FAIL if missing.

---

## 5) Acceptance Criteria (What Counts as PASS)

E4k-Speech is primarily a **diagnostic**. Therefore, PASS is defined as:

### 5.1 Pipeline validity (required for PASS)

- All hard guardrails pass on the paired run (smoke + functional + full_dataset).
- delay_diagnostics.jsonl exists and has one row per (clip window, lambda_c).
- delay_diagnostics_summary.json exists and matches the JSONL counts.

### 5.2 Non-degeneracy gates (paired run; required for PASS)

At lambda_c = 3e-4 (the highest penalty in this grid):
- Mean compute must be in the intended regime:
  - k_selected_mean (global, from compute_matched_summary.json) is approximately ~12 (no strict tolerance).
- DT must not collapse to randomness:
  - DT - Random (compute-matched capture mean) must be > 0 at lambda_c = 3e-4.

### 5.3 Guardrail separation (paired vs mispair; PASS_WITH_WARNINGS allowed)

We expect mispair_shift1 to have worse delay stability:
- For at least 4/5 lambdas:
  - dt_first_lag_mad_p50 (mispair) > dt_first_lag_mad_p50 (paired)

If this is not satisfied, the run is still considered **PIPELINE PASS** but should be reported as
PASS_WITH_WARNINGS, with analysis of why the guardrail did not separate (e.g., metric not sensitive, dataset highly
redundant, or mispair not sufficiently adversarial).

---

## 6) Required Artifacts (Per Run)

Each run must write under results/<run>/:

- subset_manifest.json
- run.log
- integrity_diagnostics.jsonl
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- delay_diagnostics.jsonl (E4k-specific; required when enabled)
- summary/delay_diagnostics_summary.json (E4k-specific; required when enabled)
- code_state.json (manually generated)
- ACCEPTANCE_REPORT.md (filled from the E4k template)

---

