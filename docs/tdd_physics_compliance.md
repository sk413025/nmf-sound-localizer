# TDD Guide for Physics/Math Compliance (Evaluation & Recommendations)

This document summarizes the current testing status of (1) transfer-function (`H`) handling and (2) the NMF-based localization pipeline, and proposes a TDD workflow to keep the implementation aligned with the stated physics and math assumptions.

---

## 1. Current Status (vs. TDD)

### What is already covered well

- **Transfer-function estimation**: integration tests synthesize STFT-domain data using `Y = H ⊙ X`, then verify that `DataProcessor.estimate_transfer_functions` recovers `H` (within a tolerance) for the 500–1500 Hz band with the expected mean-normalization + global scaling behavior.
- **Mixing-matrix structure**: unit tests check `A = [diag(H_d) W]` and ensure frequency-weighting is applied consistently (i.e., weighting both `A` and `Y`).
- **Normalization**: `TransferFunctionProcessor.process_transfer_functions` is tested for frequency limiting and reference-angle normalization (e.g., `H[:, ref] ≈ 1` while preserving inter-angle ratios).
- **Angle separability**: tests use “L2-normalized frequency-shape similarity” (off-diagonal correlation) as a separability proxy, and verify correct angle-index wrap-around (0/360).

### Gaps and risks

- The workflow is still closer to “adding tests after implementation” than “spec-first TDD” (red → green → refactor).
- Several physics/maths invariances are not yet systematically encoded as tests:
  - **Scale invariance**: predictable behavior under scaling of `H` and/or `W` (and the corresponding compensation in `X`, depending on the update rules).
  - **STFT parameter sensitivity**: how changes in `n_fft / hop / window` affect `H`, within an acceptable tolerance.
  - **β-divergence behavior**: for Euclidean (`β=2`), the loss should be non-increasing (allowing tiny numerical jitter); other β values need defined tolerances.
  - **Contrast enhancement (engineering heuristic)** must preserve core invariances (non-negativity, angle ordering stability).
- Real measured “golden” datasets are not yet used as regression thresholds / drift monitors.

**Conclusion**: the project is “moving toward TDD” but not fully following a spec-first loop. The next step is to convert the core physical and mathematical assumptions into explicit, measurable acceptance tests.

---

## 2. TDD Workflow: From Principles to Tests

### Principles

- Convert each principle into a measurable acceptance condition (metric + threshold + units/dimensions).
- Write failing tests first (red), implement the minimal change to pass (green), then refactor while keeping tests green.
- Separate **invariance tests** (must hold by definition/theory) from **engineering-strategy tests** (must not break invariances).

### Test layers

- **Unit tests (pure functions)**: frequency limits, normalization (mean/reference), contrast enhancement, mixing-matrix construction.
- **Property-based tests**: scale invariance, monotonicity trends, ordering stability, numeric stability.
- **Integration tests**: ground-truth `X` + `H` → synthesize `Y` (STFT domain) → run full estimation → compare to expected values/tolerances.
- **Regression tests (measured data)**: fixed dataset + thresholds to detect drift across changes.

---

## 3. Recommended Test Checklist (Principle-First)

### H estimation and STFT settings

- **Synthetic recovery**: generate `Y = H ⊙ X`, then `estimate_transfer_functions` should recover `H` (NRMSE/MAE/correlation thresholds).
- **Parameter sensitivity**: perturb `n_fft / hop / window`; `H` differences should remain within tolerance (e.g., NRMSE ≤ 0.05).
- **Units and dimensions**: after masking, frequency-bin indices must match the intended Hz range (e.g., 500–1500 Hz); `H` is dimensionless (complex ratio).

### Frequency banding and weighting consistency

- Frequency limiting: `apply_frequency_limit` must match the DataProcessor pipeline exactly (correct bin alignment).
- Weighting consistency: when applying frequency weights, weighting both `A` and `Y` should preserve the estimated angle ordering (e.g., top-1 unchanged or within ±1 angle step).

### Normalization and contrast enhancement

- Reference normalization: after selecting a reference angle, `H[:, ref] ≈ 1` while preserving inter-angle ratios (per-frequency MAE threshold).
- Mean vs reference: in controlled synthetic data, the choice of normalization should not introduce systematic angle bias.
- Contrast enhancement must preserve:
  - non-negativity,
  - angle ordering stability (Kendall τ / Spearman ρ threshold),
  - controlled dynamic-range change (within a specified range).

### Mixing matrix `A` structure

- Exact structural check: `A = [diag(H_d) W]` (and Frobenius-norm consistency when block normalization is enabled).
- Scale behavior: scaling `H` or `W` should lead to the expected compensating behavior in `X` (tolerance depends on the update rule).

### NMF and β-divergence

- Loss behavior: for `β=2`, loss should be non-increasing over iterations (allowing <1e-6 jitter).
- Numeric stability: no NaN/Inf, ε-projection works, bounded outputs under extreme inputs.

### Angle separability and mapping

- Shape-based separability: after L2 normalization per angle, off-diagonal correlation should be low for separable `H`.
- Angle indexing: `get_direction_index` handles wrap-around correctly; nearest-angle mapping is correct.

### Measured-data regression (once a path is fixed)

- Correlation/NRMSE between estimated `H` and measured `H` exceeds a threshold.
- Direction accuracy meets a minimum bar under fixed spacing and SNR (example: spacing=45°, SNR=20 dB → top-1 ≥ 60%).

---

## 4. Thresholds and Reproducibility

### Example thresholds

- `H` recovery: MAE ≤ 1e-2 or NRMSE ≤ 0.05; per-frequency correlation ≥ 0.95.
- Weighting consistency: top-1 angle unchanged or angle error ≤ 1× angle grid step.
- Contrast enhancement: Kendall τ / Spearman ρ ≥ 0.9; min value ≥ 0; dynamic-range change within ±10% of the design target.
- Loss (`β=2`): non-increasing over 10–20 iterations (allow tiny jitter).

### Reproducibility rules

- Fix RNG seeds.
- For noisy tests, use repeated trials and decide by mean/quantiles.
- Document units, dimensions, and angle grid explicitly in every test.

---

## 5. Note: Adjusting Synthetic Test Scenarios (Without “Teaching to the Test”)

- Root issue: a rank-1 synthetic `H(f,θ) = s(θ) × g(f)` becomes identical across angles after per-angle L2 normalization, so off-diagonal correlation ≈ 1, contradicting “separable” expectations.
- Fix principle: do not loosen thresholds; instead, generate synthetic `H` with real angle–frequency interaction so that different angles truly have different normalized frequency shapes.
- Condition-number assertions:
  - The current implementation computes SVD on *unnormalized* `H`, making it highly scale-sensitive and a poor proxy for “shape separability”.
  - If condition number is used, compute it after normalization or use a more stable alternative (e.g., eigenvalue ratios of a Gram matrix).

---

## 6. Next Actions (TDD Roadmap)

- Immediate:
  - Add property tests for **scale invariance** and **β=2 loss monotonicity**.
  - Add tests for STFT-parameter sensitivity and contrast-enhancement ordering stability.
- Mid-term:
  - Introduce a measured “golden” dataset with regression thresholds and CI integration.
  - Add a normalized condition-number (or alternative) test and update implementation if needed.
- Long-term:
  - Standardize a “principle → test → minimal implementation → refactor” workflow (templates + examples).
  - Add drift monitoring for key metrics across versions.

---

## 7. Code References

- Estimation pipeline: `nmf_localizer/core/data_processor.py` (`estimate_transfer_functions`)
- Transfer-function processing: `nmf_localizer/core/transfer_functions.py` (frequency limiting, normalization, contrast enhancement, separability analysis)
- NMF + mixing matrix: `nmf_localizer/core/localizer.py` (`_construct_mixing_matrix`, `factorize`)
- Tests: `tests/test_transfer_function_pipeline.py`, `tests/test_transfer_functions.py`

---

If useful, we can convert each checklist item into an explicit failing test first (red), then apply minimal implementation changes to make it pass (green), and only then refactor.
