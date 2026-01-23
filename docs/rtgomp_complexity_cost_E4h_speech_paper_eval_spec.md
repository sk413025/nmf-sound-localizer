# Spec: E4h-Speech -- Paper-Grade DT vs OMP vs Random (Compute-Matched + Forced-K)

This spec defines **E4h-Speech**, the paper-grade evaluation that produces defensible comparisons among:

- **DTmin**: the trained Decision Transformer checkpoint (lag-selection policy with STOP).
- **OMP**: oracle greedy pursuit upper bound (unique lag selection, no STOP).
- **Random**: a baseline lag selector (paper baseline = unique sampling without replacement).

E4h-Speech is the **same evaluation design as E4h**, but it is **restricted to the speech dataset** so that:

- The evaluation is **in-domain** relative to DT training (speech WAV pairs), and
- RTG/STOP controllability is interpretable (no train/eval domain shift).

This document is written for an engineer with **no prior signal processing or RL background**. It is intended to be
fully sufficient to implement, run, and validate E4h-Speech without external context.

---

## 0) What Problem Are We Solving?

We observe two synchronized sensors:

- A microphone (MIC): produces an acoustic signal x(t).
- A laser Doppler vibrometer (LDV): produces a vibration signal y(t).

We work in the STFT domain (time-frequency). For each frequency bin f, we model the LDV STFT over a short window as a
linear combination of lagged MIC STFT segments:

  y_f[t : t+Tw] ~= sum_{m in selected lags} w_{f,m} * x_f[t - lag_m : t - lag_m + Tw]

Key idea: the best lags are not known a priori. We want a policy that **selects a small subset of lags** to explain y.

This is a sparse approximation / pursuit problem.

---

## 1) Core Concepts (Minimal Primer)

### 1.1 STFT and windows

- The Short-Time Fourier Transform (STFT) converts a 1D waveform into a 2D complex array:
  - time index: STFT frame t (a windowed chunk of samples)
  - frequency index: f (0..n_fft/2)

- We will use a fixed **window length** Tw in STFT frames. For each clip:
  - MIC STFT: mic_stft[t, f] (complex)
  - LDV STFT: ldv_stft[t, f] (complex)

### 1.2 Dictionary atoms = lagged MIC windows

Fix a time window start t0 (in STFT frames), a window length Tw, and a lag range:

- max_lag: L
- lags: [-L, ..., 0, ..., +L]
- number of lags (atoms): M = 2*L + 1

For a given frequency bin f, the target vector is:

  Y_f = ldv_stft[t0 : t0+Tw, f]   (shape: Tw)

For each lag ell in [-L, L], define a dictionary atom:

  d_{f,ell} = mic_stft[t0-ell : t0-ell+Tw, f]  (shape: Tw)

Stack all atoms into a dictionary matrix:

  D_f = [d_{f,ell_1}, ..., d_{f,ell_M}]  (shape: Tw x M)

### 1.3 Least-squares projection and the capture metric

Given a selected set S of lag indices, let A be the submatrix of D_f with columns in S:

  A = D_f[:, S]

Compute the least-squares fit:

  w* = argmin_w ||Y_f - A w||_2
  Y_hat = A w*
  r = Y_f - Y_hat

Define energies:

  E0    = ||Y_f||_2^2
  E_res = ||r||_2^2

Define capture (how much energy we explained):

  capture = 1 - E_res / max(E0, eps_energy)

Interpretation:
- capture = 0 means we explained nothing (E_res ~= E0).
- capture = 1 means perfect explanation (E_res ~= 0).

#### 1.3.1 Why capture must be in [0, 1] (first principles)

In exact arithmetic, least-squares with Y_hat = projection of Y onto span(A) is an orthogonal projection. Orthogonal
projection cannot increase residual energy:

  E_res <= E0

Therefore capture must satisfy:

  0 <= capture <= 1

If we observe capture < 0 or capture > 1 (beyond tiny tolerance), something is wrong:
- duplicate atoms / rank deficiency -> ill-conditioned least squares
- wrong indexing / mismatched K in compute-matching
- device/dtype issues
- numeric instability

E4h-Speech treats any significant out-of-range capture as a **hard failure** for paper validity.

---

## 2) What Is Being Compared?

We compare three selection strategies for each sample:

### 2.1 OMP (Oracle Matching Pursuit) -- upper bound

Algorithm (per frequency):
1) Start with residual r0 = Y.
2) At step k, select the lag whose atom has maximum correlation with the current residual.
3) Add the atom to the active set (unique selection only).
4) Recompute the least-squares projection using all selected atoms, update residual.
5) Repeat up to max_k steps.

Properties:
- Unique selection: never pick the same lag twice.
- Residual energy should be monotonically non-increasing with k (up to small numeric tolerance).

### 2.2 Random -- baseline

Two variants (must be explicitly labeled):

1) Paper baseline: **without replacement** (unique lags)
   - For each frequency, sample K unique lag indices uniformly from {0..M-1}.

2) Diagnostic (non-paper): with replacement
   - Allows duplicates, which can cause ill-conditioned LS and capture < 0.
   - Must report duplicate rate and any out-of-range capture counts. Do not silently clamp.

### 2.3 DTmin -- learned policy with STOP

DT is trained to imitate a cost-conditioned teacher (penalty-OMP). At evaluation, DT receives:

- state: abs correlations between normalized dictionary atoms and current residual
  - abs_corrs = |D_norm^H r|  (shape: M)
- context: frequency embedding (freq_idx)
- RTG (Return-to-Go) conditioning:
  - rtg0: normalized log10(lambda_c) in [0,1]
  - rtg1: remaining_steps_fraction in [0,1] (if rtg_dim=2)

DT outputs logits over actions:
- action 0..M-1 = select a lag
- action M = STOP

Decision rule: argmax(logits) with masking to prevent duplicates.

---

## 3) Two Evaluation Families (Both Are Required)

E4h-Speech must output two result families:

### 3.1 Compute-matched (WITH STOP) -- main paper result

Goal: compare DT vs OMP vs Random at **matched compute**.

Key ambiguity: "steps" vs "compute".

- Each selected lag requires an expensive least-squares solve.
- STOP is a decision but does not add an atom.

Therefore we define:

A) k_selected (compute-aligned)
- The number of lags selected by DT before stopping or reaching max_k.
- Equivalent to number of LS projections performed.

B) steps_decision (policy-step-aligned)
- The number of decisions including STOP, if STOP occurs.
- If DT selects k atoms and then chooses STOP, steps_decision = k + 1.
- If DT never chooses STOP, steps_decision = max_k.

Compute-matching rule:
- For each sample, run DT free rollout to obtain k_selected.
- Evaluate OMP and Random **at exactly the same k_selected** (per-sample, not just matched means).

This answers the paper question:
"At the compute the policy actually uses, does DT beat Random and approach OMP?"

### 3.2 Forced-K (NO STOP) -- ablation

Goal: measure lag-selection quality independent of STOP.

Rule:
- Force DT to pick exactly K lags (mask STOP, mask duplicates).
- Run OMP for exactly K.
- Run Random for exactly K.

K values: {1, 2, 4, 8, 16} (filtered by max_k).

This answers:
"If STOP behavior confounds results, is DT's lag ranking still good?"

---

## 4) Dataset Requirements (Speech-Only)

### 4.1 Speech dataset roots (required for E4h-Speech)

We must use the speech WAV dataset (boy1) for all E4h-Speech runs:

- mic_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- ldv_root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV

Pairing rule (implemented by DoALagDataset with angle=None):
- mic file: MIC/<name>.wav
- ldv file: LDV/<name>.wav, where the filename is derived by replacing "MIC" with "LDV".

Dataset order:
- Sorted by MIC wav filename (lexicographic).
- Full dataset length is expected to be 416 pairs on this machine. If it differs, document why and treat it as a data
  version change.

### 4.2 Guardrail: do not mix domains

E4h-Speech must not accidentally run on the white-noise NPY dataset.

Validation rule:
- All file paths listed in subset_manifest.json must end with ".wav".
- If any ".npy" appears, FAIL the run and fix the roots.

---

## 5) Fixed Parameters (Must Match Training/Evaluation Conventions)

Unless explicitly justified and tested, use:

- STFT:
  - fs = 16000
  - n_fft = 2048
  - hop_length = 160
  - band = [freq_min, freq_max] = [300, 3000] Hz
- Dictionary:
  - max_lag = 50  -> M = 101
  - Tw = 32 (STFT frames)
  - max_k = 16
  - gain = 100.0 (scale mic/ldv STFT equally)
- Policy conditioning:
  - rtg_dim = 2
  - use_stop_action = true
  - lambda_c_values = 1e-4, 3e-4, 1e-3, 3e-3, 1e-2

Rationale (why these matter):
- Changing hop_length/Tw changes the window geometry and therefore the dictionary atoms and optimal lags.
- Changing max_lag changes action_dim and must match the checkpoint.
- Changing the frequency band changes which bins are evaluated; this changes stop-rate and averages.

---

## 6) Required Outputs (Artifacts)

Under:

  results/rtgomp_lambda_cost_E4h_speech_paper_eval_<mode>_<timestamp>/

Must create:
- subset_manifest.json (with per-file MD5 and fingerprint)
- run.log (stdout/stderr captured via tee)
- integrity_diagnostics.jsonl (always written; contains out-of-range diagnostics)
- summary/compute_matched_summary.json
- summary/forced_k_summary.json
- summary/rtg_controllability_summary.json
- ACCEPTANCE_REPORT.md (filled from the acceptance template)
- code_state.json (git_head + dirty + sha256 for executed files)

Optional (required for smoke and scale_check_subset; optional for full_dataset):
- per_sample.jsonl (can be huge on full dataset; if disabled, justify)

---

## 6.0 Summary JSON field reference (what to read for the acceptance report)

The evaluator writes three summary JSON files under results/<run>/summary/.

You must copy numbers from these JSONs into ACCEPTANCE_REPORT.md (do not retype from memory).

### A) summary/compute_matched_summary.json

Top-level keys:
- config: run configuration (roots, ckpt, fixed params, lambda grid, random baseline settings, device, etc.)
- integrity: guardrail counts and min/max stats
- rows: list of per-lambda summary rows

Each element of rows contains at least:
- lambda_c
- dt_capture_mean / omp_capture_mean / random_capture_mean
- dt_over_omp_mean
- dt_minus_random_mean
- k_selected_mean / k_selected_std
- steps_decision_mean / steps_decision_std

### B) summary/forced_k_summary.json

Top-level keys:
- config (same schema as compute_matched_summary)
- integrity (same schema)
- rows: list of per-K rows

Each element of rows contains at least:
- k
- dt_capture_mean / omp_capture_mean / random_capture_mean
- dt_over_omp_mean
- dt_minus_random_mean

### C) summary/rtg_controllability_summary.json

Keys:
- lambda_c_values (list)
- k_selected_mean (list aligned with lambda_c_values)
- k_selected_std (list aligned)
- k_selected_range
- k_selected_quantiles: p50/p90/p99 lists
- k_selected_histogram: dict of lambda->hist counts for k in {0..max_k}
- p_k_selected_lt_max_k (list)
- steps_decision_mean / steps_decision_std / steps_decision_range
- spearman_lambda_k_selected
- spearman_lambda_steps_decision

---

## 6.1 Exact evaluation semantics (how samples are generated)

This matters because the "number of samples" in summaries depends on:
- number of clips (num_pairs),
- number of valid windows per clip,
- number of frequency bins in the evaluation band.

The evaluator constructs samples as follows (high-level):

1) Load one clip pair (mic_wav, ldv_wav), compute STFTs:
   - mic_stft shape: (T, F)
   - ldv_stft shape: (T, F)
   - F must equal n_fft/2 + 1 (for n_fft=2048, F=1025). If not, the run must fail.

2) Define lag bounds:
   - Lag_Min = -max_lag
   - Lag_Max = +max_lag
   - M_lags = 2*max_lag + 1

3) Enumerate valid window starts t0 so that every lagged slice exists:
   - start_limit = Lag_Max
   - end_limit = T_total - Tw + Lag_Min
   - valid_starts = range(start_limit, end_limit, Tw)

4) For each window start t0 and each frequency bin f in the selected frequency band:
   - Y_f = ldv_stft[t0 : t0+Tw, f]
   - For each lag ell in [-max_lag, +max_lag]:
       d_{f,ell} = mic_stft[t0-ell : t0-ell+Tw, f]
   - This produces one sample (clip_idx, window_idx, f).

Frequency band selection:
- The evaluator constructs the frequency grid:
    freqs = linspace(0, fs/2, n_fft/2+1)
- It selects all bins whose frequency is within [freq_min, freq_max].
- The number of selected bins is F_band, and this is what num_samples_total increments by per window.

Important consequence:
- num_samples_total in integrity summaries is "number of (window, frequency) samples processed", not number of clips.

---

## 6.2 Determinism and seeding

The evaluation must be reproducible. Report all seeds explicitly.

Random baseline seeding:
- For each (clip_idx, window_idx), the evaluator sets:
    rand_seed = seed + clip_idx * 100000 + window_idx
  and uses a numpy Generator with this seed to sample random lag indices.

Implication:
- Given the same dataset order, subset manifest, and --seed, the random baseline is deterministic.

Note:
- DT and OMP are deterministic given the same inputs and argmax decisions (no sampling).

---

## 7) Acceptance Criteria (E4h-Speech)

E4h-Speech is "paper-grade": acceptance is about correctness + interpretability, not just high numbers.

### 7.1 Hard guardrails (must PASS)

- No missing files (manifest validation).
- No MD5 mismatches (manifest validation).
- No NaN/inf in reported statistics.
- num_capture_out_of_range_total == 0 for paper baseline Random without replacement.
- num_dt_duplicate_actions_forced_k == 0.
- num_omp_monotonicity_violations == 0 (beyond tolerance).
- Speech-only validation: all manifest file paths end with ".wav".

### 7.2 RTG controllability (must be demonstrated)

Primary metric (compute-aligned):
- spearman(lambda_c, k_selected_mean) <= -0.6 (recommended)
- k_selected_range >= 0.10 (recommended non-degenerate)

Distributional metrics (required even if the mean moves):
- P(k_selected < max_k) at min/max lambda (should increase with lambda)
- k_selected quantiles p50/p90/p99

If the mean effect is small but stop-rate increases materially, the run can still be acceptable, but the report must
explain the saturation mechanism.

### 7.3 DT vs Random and DT vs OMP (paper claims)

At low penalty (lambda_c = min):
- Compute-matched DT must beat Random:
  - DT_capture_mean - Random_capture_mean > 0
- DT should be close to OMP (not necessarily equal):
  - DT/OMP reasonably high (report value; thresholds are paper-dependent)

Forced-K ablation should show:
- DT is consistently above Random across K, and close to OMP as K increases.

---

## 8) Implementation Reference (What Code Runs This)

The evaluator implementation is:

- scripts/h_exploration/run_rtgomp_e4h_paper_eval.py

E4h-Speech does not require new algorithms beyond configuring this evaluator to use the speech dataset roots and saving
results to new run directories with the "E4h_speech" prefix.
