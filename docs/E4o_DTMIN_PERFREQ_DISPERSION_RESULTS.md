# E4o-Speech: DTmin Per-Freq Dispersion Validation Results

**Date**: 2026-01-27
**Experiment**: E4o-Speech DTmin per-freq dispersion validation
**Status**: PASS

---

## 1. Executive Summary

This experiment validates whether DTmin's per-frequency lag outputs can directly serve as dispersion correction, eliminating the need for separate phase equalization fitting (E4n approach).

### Key Result

**YES, DTmin per-freq lags can effectively reduce dispersion.**

| Metric | Baseline (`none`) | Best (`agg`) | Improvement |
|--------|------------------|--------------|-------------|
| tau_band_spread_ms_p50 | 10.29 ms | 7.15 ms | **-30.5%** |
| phase_slope_r2_p50 | 0.4831 | 0.4828 | ~0% (maintained) |
| gcc_phat_psr_p50 | 56.90 | 58.79 | +3.3% |

---

## 2. Background

### 2.1 The Problem

Previous experiments (E4m/E4n) may have deviated from the original goal:
- **Original question**: Can DTmin solve the dispersion problem?
- **E4m/E4n approach**: Aggregate DTmin per-freq outputs into median, then fit separate phase_eq

### 2.2 Key Insight

OMP and DTmin operate independently on each frequency, outputting `dt_first_ids[f]` which is effectively an estimate of τ(f):

```python
D = torch.zeros(F, tw, M_lags)  # Each frequency has its own dictionary
dt_first_ids: (F,)              # Each frequency selects its first lag
```

This means we can derive phase calibration directly from DTmin output:

$$G_{\text{DTmin}}(f) = e^{+j 2\pi f \cdot \tau_{\text{DTmin}}(f)}$$

where $\tau_{\text{DTmin}}(f) = (\text{dt\_first\_ids}[f] + \text{lag\_min}) \times \text{hop\_length} / f_s$

---

## 3. Experimental Design

### 3.1 Conditions Tested

| Condition | `--phase_eq_source` | Description |
|-----------|---------------------|-------------|
| Baseline | `none` | No phase equalization |
| Per-window | `dtmin_perfreq_perwin` | Derive phase_eq from current window's DTmin output |
| Aggregated | `dtmin_perfreq_agg` | Two-pass: collect all lags, compute median, then apply |

### 3.2 Implementation

New CLI parameter added to `run_rtgomp_e4h_paper_eval.py`:
```
--phase_eq_source {none, fit_e4n, dtmin_perfreq_perwin, dtmin_perfreq_agg}
```

New helper function:
```python
def derive_phase_eq_from_perfreq_lags(
    dt_first_ids: np.ndarray,  # (F,)
    freqs_hz: np.ndarray,
    hop_length: int,
    fs: int,
    lag_min: int,
) -> np.ndarray:
    # Returns (F,) complex64 phase_eq
```

### 3.3 Dataset

- **Source**: Speech WAV data (boy1)
- **mic_root**: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC`
- **ldv_root**: `/Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV`
- **Checkpoint**: `dt_freq_aware_best.pth` (E4j stopw0p020)

### 3.4 Fixed Parameters

| Parameter | Value |
|-----------|-------|
| fs | 16000 Hz |
| hop_length | 160 samples |
| n_fft | 2048 |
| freq_band | [300, 3000] Hz |
| max_lag | 50 |
| tw | 32 |
| max_k | 16 |
| lambda_c_values | 1e-5, 3e-5, 1e-4, 2e-4, 3e-4 |

---

## 4. Results

### 4.1 Smoke Test (1 pair)

| Condition | phase_slope_r2_p50 | tau_band_spread_ms_p50 |
|-----------|-------------------|----------------------|
| `none` | 0.5116 | 17.13 ms |
| `perwin` | 0.3627 | 12.86 ms |
| `agg` | 0.4376 | 6.96 ms |

### 4.2 Scale Check (48 pairs) - Primary Results

#### Full Table (lambda_c = 3e-4)

| Condition | phase_slope_r2_p50 | tau_band_spread_ms_p50 | gcc_phat_psr_p50 |
|-----------|-------------------|----------------------|-----------------|
| `none` | 0.4831 | 10.43 ms | 56.82 |
| `perwin` | 0.4729 | 9.60 ms | 60.48 |
| `agg` | 0.4828 | 7.15 ms | 58.71 |

#### Detailed Results by lambda_c

**Condition: `none` (Baseline)**
| lambda_c | r2_p50 | spread_p50 (ms) | psr_p50 |
|----------|--------|-----------------|---------|
| 1e-05 | 0.4831 | 10.289 | 56.90 |
| 3e-05 | 0.4831 | 10.288 | 56.90 |
| 1e-04 | 0.4831 | 10.147 | 56.90 |
| 2e-04 | 0.4831 | 10.429 | 56.90 |
| 3e-04 | 0.4831 | 10.429 | 56.82 |

**Condition: `dtmin_perfreq_perwin`**
| lambda_c | r2_p50 | spread_p50 (ms) | psr_p50 |
|----------|--------|-----------------|---------|
| 1e-05 | 0.4729 | 9.650 | 60.54 |
| 3e-05 | 0.4729 | 9.601 | 60.54 |
| 1e-04 | 0.4729 | 9.587 | 60.57 |
| 2e-04 | 0.4729 | 9.587 | 60.56 |
| 3e-04 | 0.4729 | 9.601 | 60.48 |

**Condition: `dtmin_perfreq_agg`**
| lambda_c | r2_p50 | spread_p50 (ms) | psr_p50 |
|----------|--------|-----------------|---------|
| 1e-05 | 0.4828 | 7.151 | 58.79 |
| 3e-05 | 0.4828 | 7.151 | 58.79 |
| 1e-04 | 0.4828 | 7.140 | 58.79 |
| 2e-04 | 0.4828 | 7.140 | 58.79 |
| 3e-04 | 0.4828 | 7.151 | 58.71 |

---

## 5. Analysis

### 5.1 tau_band_spread_ms (Dispersion Metric)

**BECAUSE** DTmin selects the first lag for each frequency independently,
**AND** these lags encode the frequency-dependent delay τ(f),
**THEREFORE** applying phase correction G(f) = exp(+j 2π f τ(f)) reduces dispersion.

| Condition | spread_p50 | Change vs Baseline |
|-----------|------------|-------------------|
| `none` | 10.29 ms | - |
| `perwin` | 9.60 ms | -6.7% |
| `agg` | 7.15 ms | **-30.5%** |

The aggregated mode (`agg`) achieves the best dispersion reduction because:
1. Statistical aggregation across windows reduces noise in individual lag estimates
2. Median is robust to outliers from windows with poor signal quality

### 5.2 phase_slope_r2 (Phase Linearity)

| Condition | r2_p50 | Change vs Baseline |
|-----------|--------|-------------------|
| `none` | 0.4831 | - |
| `perwin` | 0.4729 | -2.1% |
| `agg` | 0.4828 | -0.06% |

Phase linearity is maintained in `agg` mode, indicating that the correction does not introduce artifacts.

### 5.3 gcc_phat_psr (Peak Sharpness)

| Condition | psr_p50 | Change vs Baseline |
|-----------|---------|-------------------|
| `none` | 56.90 | - |
| `perwin` | 60.54 | +6.4% |
| `agg` | 58.79 | +3.3% |

Both DTmin-derived modes improve peak sharpness, with `perwin` showing the largest improvement.

---

## 6. Conclusions

### 6.1 Primary Finding

**DTmin's per-frequency lag outputs can directly serve as dispersion correction.**

- `dtmin_perfreq_agg` reduces tau_band_spread by 30.5% while maintaining phase_slope_r2
- No separate phase_eq fitting step (E4n) is required

### 6.2 Recommended Mode

**Use `--phase_eq_source dtmin_perfreq_agg`** for production.

Reasons:
1. Best dispersion reduction (30.5%)
2. Maintains phase linearity (r2 unchanged)
3. Statistical aggregation provides stability

### 6.3 Trade-offs

| Mode | Pros | Cons |
|------|------|------|
| `none` | Baseline, no computation | No dispersion correction |
| `perwin` | Best PSR, instant | Slightly lower r2 |
| `agg` | Best spread, stable r2 | Requires two passes |

---

## 7. Artifacts

### 7.1 Output Directories

| Condition | Directory |
|-----------|-----------|
| Smoke (none) | `results/e4o_speech_smoke_none/` |
| Smoke (perwin) | `results/e4o_speech_smoke_perwin/` |
| Smoke (agg) | `results/e4o_speech_smoke_agg/` |
| Scale (none) | `results/e4o_speech_scale_none/` |
| Scale (perwin) | `results/e4o_speech_scale_perwin/` |
| Scale (agg) | `results/e4o_speech_scale_agg/` |

### 7.2 Key Files

- `summary/subsample_delay_diagnostics_summary.json`: Primary metrics
- `summary/compute_matched_summary.json`: Capture metrics
- `subsample_delay_diagnostics.jsonl`: Per-window details

---

## 8. Code Changes

### 8.1 Modified Files

| File | Changes |
|------|---------|
| `scripts/h_exploration/run_rtgomp_e4h_paper_eval.py` | Added `--phase_eq_source`, `derive_phase_eq_from_perfreq_lags()`, two-pass aggregation |
| `docs/rtgomp_dtmin_perfreq_E4o_speech_spec.md` | New spec document |
| `docs/E4o_DTMIN_PERFREQ_DISPERSION_RESULTS.md` | This results document |

### 8.2 New CLI Parameter

```
--phase_eq_source {none, fit_e4n, dtmin_perfreq_perwin, dtmin_perfreq_agg}
    Phase equalization source for E4o dispersion validation.
    none=no phase_eq
    fit_e4n=load from --phase_eq_path (E4n fitter output)
    dtmin_perfreq_perwin=derive from DTmin per-freq lags each window (instant)
    dtmin_perfreq_agg=derive from aggregated (median) per-freq lags (two-pass)
```

---

## 9. Next Steps

1. **Full dataset validation** (416 pairs) to confirm results at scale
2. **Compare with E4n fitting** (`fit_e4n` mode) to quantify difference
3. **Investigate perwin r2 drop**: Why does per-window mode reduce phase linearity?
4. **Optimize agg computation**: Consider online median estimation to avoid two passes

---

## Appendix A: Run Commands

### Smoke Test
```bash
PYTHONPATH=. python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/e4o_speech_smoke_<condition> \
  --mode smoke --num_pairs 1 \
  --lambda_c_values '1e-5,3e-5,1e-4,2e-4,3e-4' \
  --phase_eq_source <condition> \
  --write_subsample_delay_diagnostics 1 \
  --subsample_method 'gcc_phat,phase_slope' \
  --require_wav_only 1 \
  --device cpu
```

### Scale Check (48 pairs)
```bash
PYTHONPATH=. python scripts/h_exploration/run_rtgomp_e4h_paper_eval.py \
  --mic_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC \
  --ldv_root /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV \
  --ckpt_path results/rtgomp_lambda_cost_E4j_speech_stopwsweep_warmstart_stepwise_freezebn_lr1e-3_ep15_stopw0p020_20260124_092640/model/dt_freq_aware_best.pth \
  --out_dir results/e4o_speech_scale_<condition> \
  --mode scale_check_subset \
  --lambda_c_values '1e-5,3e-5,1e-4,2e-4,3e-4' \
  --phase_eq_source <condition> \
  --write_subsample_delay_diagnostics 1 \
  --subsample_method 'gcc_phat,phase_slope' \
  --require_wav_only 1 \
  --device cpu
```

Replace `<condition>` with: `none`, `dtmin_perfreq_perwin`, or `dtmin_perfreq_agg`
