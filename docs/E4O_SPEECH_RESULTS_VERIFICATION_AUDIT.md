# E4o-Speech Results Verification Audit (Design + Artifacts + Metrics)

**Purpose**: Provide a concrete, reproducible procedure to validate that the E4o-Speech experiment:
1) is **well-posed** under the physics/math framing,
2) uses **consistent** data/configuration across conditions, and
3) reports **correct** summary numbers derived from the stored per-window artifacts.

This audit does **not** claim any downstream localization benefit by itself; it only verifies that the E4o dispersion metrics are computed and reported correctly.

---

## 1) What E4o Claims (Audit Target)

E4o tests whether **DTmin’s per-frequency first-lag outputs** can directly construct a **unit-magnitude phase equalizer** \(G(f)\) that reduces measured “dispersion” (frequency-dependent delay spread), without requiring a separate E4n fitter.

Key observable: **tau-band spread** derived from phase-slope group-delay estimates across subbands.

---

## 2) Artifacts Under Audit

Primary E4o scale runs (48 pairs):
- `results/e4o_speech_scale_none/`
- `results/e4o_speech_scale_perwin/`
- `results/e4o_speech_scale_agg/`

Key files in each directory:
- `subset_manifest.json`: exact files + per-file MD5; includes `fingerprint_md5`.
- `subsample_delay_diagnostics.jsonl`: per-window diagnostic rows.
- `summary/subsample_delay_diagnostics_summary.json`: aggregated metrics (uses P2 quantile estimator).
- `summary/compute_matched_summary.json`: compute-matched capture metrics (energy-based).

---

## 3) Design Sanity (Physics/Math)

### 3.1 Phase equalization is unit-magnitude
E4o uses an all-pass correction (per frequency bin):
\[
G(f) = e^{+j2\pi f \hat{\tau}(f)}.
\]
This is a **phase-only rotation**; it does not change magnitudes.

### 3.2 Why compute-matched capture should not change
Capture metrics are based on residual **energy** \(||Y - D\hat{a}||^2\).
Multiplying \(Y\) by a unit-magnitude complex factor rotates the complex residual but does not change its magnitude:
\[
||e^{j\theta}r||^2 = ||r||^2.
\]
Therefore, `compute_matched_summary.json` should remain invariant across `phase_eq_source` conditions (this is an internal consistency check).

---

## 4) Artifact Integrity Checks (Must Pass)

### 4.1 Confirm identical data subset across conditions
Verify `subset_manifest.json` is identical and `fingerprint_md5` matches:
- Expect: `fingerprint_md5=739a181c331f347614090fffe6f4b491`

### 4.2 Confirm row counts are consistent
For the scale set:
- `num_windows_total` should be `234`.
- Total subsample rows should be `234 windows × 5 lambdas × 2 sources = 2340`.

These are recorded in `summary/subsample_delay_diagnostics_summary.json` under `integrity`.

---

## 5) Metric Correctness Checks (Must Pass)

### 5.1 Understand how summaries are computed
`summary/subsample_delay_diagnostics_summary.json` is produced by streaming accumulation during evaluation:
- Quantiles are computed by the **P2 estimator** (approximate but deterministic).
- `phase_slope_r2` quantiles are updated only when:
  - `phase_slope_tau_hat_ms` is defined **and**
  - `undefined_reason` is `null`.

### 5.2 Recompute quantiles from JSONL and match the stored summary
We provide a script that replays `subsample_delay_diagnostics.jsonl` in-order and reproduces:
- `psr_p50`
- `tau_band_spread_ms_p50`
- `phase_slope_r2_p50`
including the filtering logic above.

Script:
- `scripts/h_exploration/audit_e4o_speech_results.py`

Run it on the scale set:
```bash
PYTHONPATH=. python scripts/h_exploration/audit_e4o_speech_results.py \
  --lambda_c 3e-4 \
  --run_dirs \
    results/e4o_speech_scale_none \
    results/e4o_speech_scale_perwin \
    results/e4o_speech_scale_agg
```

Expected outcome:
- All subsample summary checks PASS (dt + omp).
- Compute-matched capture invariance PASS (dt/omp capture + k_selected identical within tolerance).

---

## 6) Observed Consistency Signals (What “Looks Right”)

These are sanity signals that support correctness (not “proof of best method”):
- `tau_band_spread_ms_p50` decreases for `dtmin_perfreq_agg` vs `none`.
- `phase_slope_fraction_defined` should not decrease dramatically (ideally increases).
- `boundary_hit_rate` should remain near 0 for stable search windows.
- `compute_matched_summary.json` invariance should hold across conditions.

---

## 7) Known Methodological Caveats (To Document in Papers)

1) **Aggregation uses a representative lambda**: current implementation derives the aggregated phase_eq from `lambda_c_values[0]` for simplicity. This is acceptable if explicitly stated, and can be extended if needed.
2) **Perwin STFT application**: current implementation applies per-window phase equalization in the *subsample (GCC-PHAT) path*; it does not currently apply a per-window phase_eq to the STFT tensor for the OMP/DT paths. This does not affect the dispersion diagnostic audit, but should be clarified if perwin is used in a paper.
3) **Calibration vs evaluation split**: `dtmin_perfreq_agg` is a two-pass calibration on the same dataset subset. If reviewers are concerned about leakage, add a held-out evaluation where phase_eq is fit on one subset and evaluated on another.

---

## 8) Next Audit Step (Beyond Correctness)

After correctness is established, the next step for publication is **utility**:
- demonstrate downstream benefit (TDOA/DOA/localization) on held-out data.

This is separate from the “E4o results are computed correctly” question addressed here.

