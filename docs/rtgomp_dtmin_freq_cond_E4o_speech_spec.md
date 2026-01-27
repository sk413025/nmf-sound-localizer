# Spec: E4o-Speech — DTmin Frequency Conditioning Audit (Conditional vs “Per-Band” Behavior)

This spec defines **E4o-Speech**.

E4o-Speech is a targeted diagnostic experiment to answer a simple question:

> Is the current **DTmin / `SeqDT_FreqAware`** actually using **frequency as a condition** (via `freq_idx` / `freq_embed`) to behave “frequency-aware”, or is frequency conditioning effectively unused / degenerate (or acting like a per-frequency lookup)?

All content and artifacts MUST be in English.

Implementation targets:
- Extend evaluator with frequency-conditioning ablations:
  - `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

Non-goals:
- E4o does **not** attempt to “fix dispersion” (that is E4m/E4n territory).
- E4o does **not** redesign the model architecture. It measures what the current checkpoint does.

---

## 0) Why This Exists (Background + Motivation)

We use **DTmin** to approximate an **OMP-like** lag-selection policy for MIC→LDV phase alignment under a compute knob (`lambda_c`).

The model class is named `SeqDT_FreqAware`, which includes a `freq_embed(freq_idx)` added to the state+RTG embedding. This is intended to make a **single model** behave differently across frequency bins.

However, it is currently unclear whether:
1) the model truly uses `freq_idx` as a meaningful condition (good), or
2) `freq_idx` is ignored (model is effectively frequency-agnostic), or
3) `freq_idx` behaves like a per-frequency lookup table (limited sharing; can behave like “one model per bin” in practice).

This matters because:
- We interpret DT behavior across frequency as “physics-aware” only if conditioning is real.
- If conditioning is weak/ignored, we should not attribute frequency-dependent behavior to the model.
- If the behavior is effectively “per-band”, then training separate models per band might be justified, but it increases complexity and must be motivated by evidence.

Therefore, E4o performs **inference-time ablations** that break or remove frequency conditioning and measures how much performance degrades.

---

## 1) Core Questions (Must Answer)

E4o-Speech answers:

1) **Does `freq_idx` matter at inference time?**
   - If we shuffle `freq_idx` across frequencies (wrong conditioning), do capture and controllability degrade?

2) **Is the model relying on frequency embedding vs purely on state correlations?**
   - If we force a constant `freq_idx`, does performance degrade?
   - If we zero-out the frequency embedding, does performance degrade?

3) **Is the effect band-dependent?**
   - Are ablation deltas larger in [1800,3000] Hz than in [300,900] Hz?

Outcome classification (required in acceptance report):
- `FREQ_COND_USED`: ablations cause clear degradation.
- `FREQ_COND_IGNORED`: ablations cause negligible change.
- `INCONCLUSIVE`: changes are within noise / unstable.

---

## 2) Definitions (Make Explicit)

### 2.1 “Single conditional model” vs “per-band models”

- **Single conditional model (current DTmin)**: one checkpoint, one parameter set, and a conditioning input `freq_idx`.
  - In code: `SeqDT_FreqAware.freq_embed = nn.Embedding(max_freq, d_model)`
  - Forward: `emb = layer_norm(state_emb + rtg_emb + freq_emb(freq_idx))`

- **Per-band model (the “violent” alternative)**: train separate DT checkpoints on disjoint frequency ranges.
  - E4o does not train per-band models; it determines whether this follow-up is justified.

### 2.2 Frequency conditioning ablations (E4o)

Given `freq_ids` (shape `(F_band,)`) passed to the model:
- `normal`: pass true `freq_ids` (no change).
- `shuffle`: apply a fixed permutation to `freq_ids` (deterministic by seed).
- `constant`: replace all `freq_ids` with a constant `freq_const_idx`.
- `zero_embed`: temporarily set `model.freq_embed.weight[:] = 0` for DT inference (restore after).

Key constraint:
- These are **inference-time** changes only. No retraining.

---

## 3) Dataset + Hard Guardrails

### 3.1 Speech WAV-only roots (required; do not change)

Allowed roots:
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

Hard guardrails:
- Real data only.
- `require_wav_only=1`: any non-`.wav` path => FAIL.
- WAV sample rate must be exactly `fs=16000`: any mismatch => FAIL (no resampling).

### 3.2 Pairing modes

E4o uses paired only (positive path):
- `pairing_mode=paired`

Optional guardrail:
- `pairing_mode=mispair_shift1` can be run to ensure frequency-conditioning ablations do not accidentally “improve” a mispair case.

---

## 4) Fixed Parameters (Must Match E4n/E4m Defaults)

Common:
- fs = 16000
- hop_length = 160
- n_fft = 2048
- freq band = [300, 3000] Hz
- max_lag = 50
- tw = 32
- max_k = 16
- gain = 100.0
- rtg_dim = 2

Compute knob (same 5-point grid; exact values):
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

Checkpoint (DT; required):
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Evaluator:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

Recommended device:
- `--device cpu` (device stability > speed for this diagnostic)

---

## 5) Required Implementation (Evaluator Extension)

In `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`:

### 5.1 Add CLI flags (required)

- `--freq_cond_mode` with allowed values:
  - `normal` (default)
  - `shuffle`
  - `constant`
  - `zero_embed`
- `--freq_cond_seed` (int, default 0): used for `shuffle`.
- `--freq_cond_constant_idx` (int, default 0): used for `constant`.

### 5.2 Apply the ablation in exactly one place (required)

Apply the chosen mode to the **`freq_ids` tensor passed into DT inference**, without changing:
- the physics (dictionary build, residual update),
- OMP baseline,
- random baseline,
- or frequency selection / band mask itself.

Implementation requirement:
- For `shuffle`, generate a single permutation per run (seeded) and reuse it for all windows so the ablation is deterministic.
- For `zero_embed`, ensure weights are restored after inference so the script does not leave global state mutated.

### 5.3 Logging / artifacts (required)

For every run, write:
- `summary/freq_cond_audit_summary.json` containing:
  - freq_cond_mode, seed, constant_idx
  - band definition: freq_min/freq_max (Hz), start_bin/end_bin (bins used)
  - key DT metrics at each lambda:
    - k_selected_mean
    - compute_matched_capture_mean (DT/OMP/Random)
    - DT_minus_Random (compute matched)
  - one-line classification suggestion (computed later in report is OK, but include raw numbers here)

No fallback behavior:
- constant_idx must be in `[0, 1024]` for n_fft=2048; otherwise FAIL.

---

## 6) Acceptance Criteria (PASS / PASS_WITH_WARNINGS / FAIL)

E4o is a **measurement** experiment: “no degradation” is not a failure if it is clearly measured.

### 6.1 Hard run validity (all runs)

PASS requirements:
- All hard guardrails pass (wav-only; fs=16000; no resampling).
- No NaN/Inf in summary metrics.
- All required artifacts exist:
  - `summary/compute_matched_summary.json`
  - `summary/rtg_controllability_summary.json`
  - `summary/freq_cond_audit_summary.json`
  - `subset_manifest.json`, `run.log`, `code_state.json`, `ACCEPTANCE_REPORT.md`

### 6.2 Classification quality (scale_check_subset, paired)

Define the target metric:
- `DT_cm = compute_matched_capture_mean["dt"]` at `lambda_c = 3e-4`

Compute deltas:
- `Δshuffle = DT_cm(normal) - DT_cm(shuffle)`
- `Δconst  = DT_cm(normal) - DT_cm(constant)`
- `Δzero   = DT_cm(normal) - DT_cm(zero_embed)`

Classification rules:
- `FREQ_COND_USED` if **any** of {Δshuffle, Δconst, Δzero} >= 0.05 (absolute) AND the sign is consistent (ablation does not improve) for >=4/5 lambdas.
- `FREQ_COND_IGNORED` if all of {Δshuffle, Δconst, Δzero} <= 0.01 (absolute) AND k-selected controllability is unchanged within tolerance.
- Otherwise `INCONCLUSIVE`.

Outcome mapping:
- PASS: classification is `FREQ_COND_USED` or `FREQ_COND_IGNORED`.
- PASS_WITH_WARNINGS: classification is `INCONCLUSIVE` due to high variance or unstable compute control.
- FAIL: missing artifacts, guardrail failure, or crashes.

---

## 7) Required Runs (Minimum Suite)

All runs must use lockdir + `tee -a` and write artifacts under `results/<run>/`.

1) Smoke (paired; num_pairs=1; normal):
- Validates the new CLI flags do not break the evaluator.

2) Functional suite (paired; scale_check_subset; num_pairs=48):
- normal
- shuffle
- constant
- zero_embed

Optional:
3) Band sensitivity (paired; scale_check_subset; num_pairs=48; normal):
- [300,900] Hz
- [900,1800] Hz
- [1800,3000] Hz

Each run directory must contain a filled `ACCEPTANCE_REPORT.md` in English with BECAUSE/THEREFORE causal interpretation.

