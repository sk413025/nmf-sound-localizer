# Spec: E4o-Speech — DTmin Per-Freq Dispersion Validation

This spec defines **E4o-Speech** (per-freq dispersion validation variant).

E4o-Speech validates whether DTmin's per-frequency lag outputs can directly serve as dispersion correction, eliminating the need for separate phase equalization fitting.

All content and artifacts MUST be in English.

Implementation targets:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

---

## 0) Background: From First Principles

### The Original Goal

E4m/E4n experiments may have deviated from the original objective:
- **Original question**: Can DTmin solve the dispersion problem?
- **E4m/E4n approach**: Aggregate DTmin per-freq outputs into median, then fit separate phase_eq

### Key Insight

OMP and DTmin **operate independently on each frequency**, outputting `dt_first_ids[f]` which is effectively an estimate of τ(f):

```python
# In run_rtgomp_e4h_paper_eval.py
D = torch.zeros(F, tw, M_lags)  # Each frequency has its own dictionary
dt_first_ids: (F,)              # Each frequency selects its first lag
```

This means we can derive phase calibration directly from DTmin output, without separate phase_eq fitting:

$$G_{\text{DTmin}}(f) = e^{+j 2\pi f \cdot \tau_{\text{DTmin}}(f)}$$

where $\tau_{\text{DTmin}}(f) = \text{dt\_first\_ids}[f] \times \text{hop\_length} / f_s$

---

## 1) Core Questions (Must Answer)

E4o-Speech answers:

1. **Can DTmin per-freq lags directly correct dispersion?**
   - Compare phase_slope_r2 across phase_eq sources

2. **Per-window vs Aggregated**: Which granularity works better?
   - `dtmin_perfreq_perwin`: Instant calibration per window
   - `dtmin_perfreq_agg`: Statistical aggregation before calibration

3. **How does DTmin-derived phase_eq compare to E4n fitting?**
   - If DTmin works well: Simpler pipeline, no separate fitting step
   - If DTmin fails: DTmin optimizes for energy capture, not delay precision

---

## 2) Definitions

### 2.1 Phase Equalization Sources

| Source | Description | Computation |
|--------|-------------|-------------|
| `none` | No phase_eq | Baseline |
| `fit_e4n` | E4n fitting | Load from pre-computed `.npz` file |
| `dtmin_perfreq_perwin` | DTmin per-freq | Derive from current window's `dt_first_ids` |
| `dtmin_perfreq_agg` | DTmin per-freq | Pass 1: collect all lags; Pass 2: apply median |

### 2.2 Phase Equalization Formula

```python
def derive_phase_eq_from_perfreq_lags(
    dt_first_ids: np.ndarray,  # (F,) int array
    freqs_hz: np.ndarray,      # (F,) float array
    hop_length: int,
    fs: int,
    lag_min: int,
) -> np.ndarray:
    """Returns (F,) complex64 phase_eq."""
    lags = dt_first_ids + lag_min
    tau_sec = lags * hop_length / fs
    defined_mask = dt_first_ids >= 0

    phase_eq = np.ones(len(freqs_hz), dtype=np.complex64)
    phase_eq[defined_mask] = np.exp(
        1j * 2 * np.pi * freqs_hz[defined_mask] * tau_sec[defined_mask]
    )
    return phase_eq
```

### 2.3 Two-Pass Aggregation (dtmin_perfreq_agg)

**Pass 1**: Collect per-freq lags
```python
for clip_idx in range(num_pairs):
    for window_idx in range(num_windows):
        dt_first_ids = compute_dt_free_rollout(...)
        all_first_ids.append(dt_first_ids)
```

**Aggregation**: Compute median per frequency
```python
agg_lags[f] = np.nanmedian(all_first_ids[:, f])  # NaN for undefined (-1)
```

**Pass 2**: Apply aggregated phase_eq and evaluate

---

## 3) Dataset + Hard Guardrails

### 3.1 Speech WAV-only roots (required)

Allowed roots:
- mic_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- ldv_root = `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`

Hard guardrails:
- Real data only.
- `require_wav_only=1`: any non-`.wav` path => FAIL.
- WAV sample rate must be exactly `fs=16000`: any mismatch => FAIL.

### 3.2 Pairing mode

E4o uses paired only:
- `pairing_mode=paired`

---

## 4) Fixed Parameters (Match E4n/E4m Defaults)

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

Compute knob (same 5-point grid):
- lambda_c_values = [1e-5, 3e-5, 1e-4, 2e-4, 3e-4]

Checkpoint (DT):
- `results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth`

Evaluator:
- `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py`

---

## 5) Required Implementation

### 5.1 CLI Flags (Added)

```
--phase_eq_source {none, fit_e4n, dtmin_perfreq_perwin, dtmin_perfreq_agg}
```

Default: `none`

### 5.2 Backward Compatibility

- `--apply_phase_eq=1` with `--phase_eq_source=none` maps to `fit_e4n`
- `--apply_phase_eq=1` with `--phase_eq_source=fit_e4n` is valid
- `--apply_phase_eq=1` with other sources => FAIL

### 5.3 Implementation Details

1. **`derive_phase_eq_from_perfreq_lags`**: New helper function
2. **Pass 1 (agg mode only)**: Collect per-freq lags before main loop
3. **Phase_eq application**: In subsample diagnostics section

---

## 6) Validation Metrics

| Metric | Description | Expected Behavior |
|--------|-------------|-------------------|
| `phase_slope_r2` | Post-calibration phase linearity | Higher = better calibration |
| `tau_band_spread_ms` | Sub-band delay consistency | Lower = less dispersion |
| `gcc_phat_psr` | Correlation peak sharpness | Higher = cleaner alignment |
| `abs_tau_agreement_ms` | GCC-PHAT vs phase-slope consistency | Lower = more agreement |

---

## 7) Acceptance Criteria

### 7.1 Hard Run Validity

PASS requirements:
- All hard guardrails pass (wav-only; fs=16000)
- No NaN/Inf in summary metrics
- All required artifacts exist

### 7.2 Outcome Classification

**If DTmin effectively estimates dispersion**:
- `dtmin_perfreq_*` phase_slope_r2 >= `fit_e4n` phase_slope_r2
- tau_band_spread_ms reduces
- No extra fitting step needed

**If DTmin does NOT effectively estimate dispersion**:
- `dtmin_perfreq_*` phase_slope_r2 << `fit_e4n` phase_slope_r2
- DTmin optimizes for energy capture, not delay precision
- Keep existing phase_eq fitting approach

---

## 8) Required Runs

### 8.1 Smoke Test (1 pair)

Validate code correctness:
```bash
python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mode smoke --num_pairs 1 \
  --phase_eq_source none \
  --write_subsample_delay_diagnostics 1 \
  --subsample_method gcc_phat,phase_slope \
  ...
```

### 8.2 Scale Check (48 pairs)

Four conditions:
1. `--phase_eq_source none` (baseline)
2. `--phase_eq_source fit_e4n --phase_eq_path <path>`
3. `--phase_eq_source dtmin_perfreq_perwin`
4. `--phase_eq_source dtmin_perfreq_agg`

### 8.3 Full Dataset (416 pairs)

Complete validation with all four conditions.

---

## 9) Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Smoothing | None | Try raw data first |
| Computation granularity | Both (perwin + agg) | Compare approaches |
| Representative lambda | `lambda_c_values[0]` | Simplicity; can extend later |

---

## 10) Expected Results Summary

| Condition | phase_slope_r2 | tau_band_spread_ms | Interpretation |
|-----------|---------------|-------------------|----------------|
| `none` | Low | High | No calibration |
| `fit_e4n` | High | Low | External fitting |
| `dtmin_perfreq_perwin` | ? | ? | Instant DTmin |
| `dtmin_perfreq_agg` | ? | ? | Aggregated DTmin |

If `dtmin_perfreq_agg` approaches `fit_e4n`, DTmin can replace separate fitting.
If not, DTmin is optimized for energy, not delay estimation.

---

## 11) Key Files

| File | Modification |
|------|--------------|
| `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py` | Add `--phase_eq_source`, `derive_phase_eq_from_perfreq_lags`, two-pass aggregation |
| `docs/rtgomp_dtmin_perfreq_E4o_speech_spec.md` | This spec |
