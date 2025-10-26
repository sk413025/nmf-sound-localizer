# QK-Routing Evidence Assessment: Commit 70ba576 Analysis

**Date**: 2025-10-26  
**Commit**: 70ba5761b78b6f6ccc4eddaf0da9b8c05fa1b9e6  
**Experiment**: QK-routing, 10 epochs, DoADataset (F=346, E=37, M=8)

---

## Executive Summary

**Question**: Can we verify all gradient flow hypotheses from commit 70ba576 data?

**Answer**: **Partial verification only - 50% fully verified, 20% unverifiable**

- ✅ **5/10 hypotheses VERIFIED** with direct measurements
- ⚠️ **3/10 hypotheses INFERRED** from indirect evidence  
- ❌ **2/10 hypotheses HYPOTHESIZED** - no supporting data

**Critical Finding**: The two most important bottleneck hypotheses (g sparsity 96.6%, L2 pooling gradient vanishing) **cannot be verified** due to missing diagnostic data.

---

## Hypothesis Verification Matrix

| ID | Hypothesis | Evidence Available | Verdict | Data Source |
|----|-----------|-------------------|---------|-------------|
| H1 | Gradients flow to attention | ✅ Yes | ✅ VERIFIED | `grad_norms['Wq/Wk/encoder']` |
| H2 | Gradients too small | ⚠️ Indirect | ⚠️ INFERRED | `grad_norms × lr` (calculated) |
| H3 | QK direction wrong (uncorrelated with g) | ✅ Yes | ✅ VERIFIED | `qk_g_corr_pearson_mean` |
| H4 | Double softmax dilution (99.93%) | ⚠️ Partial | ⚠️ INFERRED | `w_e_entropy` only |
| H5 | L2 pooling gradient vanishing | ❌ No | ❓ HYPOTHESIZED | Missing `scores_atoms/expert` |
| H6 | g sparsity suppresses 96.6% gradients | ❌ No | ❓ HYPOTHESIZED | Missing `g_stats` |
| H7 | Transformer depth bottleneck | ✅ Yes | ✅ VERIFIED (WRONG) | `grad_norms['encoder']` |
| H8 | High coherence gradient degeneracy | ⚠️ Partial | ⚠️ INFERRED | μ=0.9995 + P_D grad |
| H9 | Teacher supervision ineffective | ✅ Yes | ✅ VERIFIED | `teacher_acc` vs `qk_top1_match` |
| H10 | Parameters barely move (<3%) | ✅ Yes | ✅ VERIFIED (WRONG) | `param_delta` |

---

## Detailed Hypothesis Analysis

### ✅ H1: Gradients Flow to Attention Layers (VERIFIED)

**Hypothesis**: Backpropagation successfully delivers gradients to Wq, Wk, and encoder parameters.

**Evidence**:
```json
{
  "grad_norms.Wq": {
    "epoch_1": 3.345e-03,
    "epoch_10": 1.009e-02,
    "median": 6.330e-03,
    "min": 2.308e-03,
    "max": 1.168e-02
  },
  "grad_norms.Wk": {
    "epoch_1": 4.650e-03,
    "epoch_10": 2.014e-02,
    "median": 1.339e-02,
    "min": 3.116e-03,
    "max": 2.137e-02
  },
  "grad_norms.encoder": {
    "epoch_1": 2.450e-02,
    "epoch_10": 7.677e-02,
    "median": 3.503e-02,
    "min": 1.474e-02,
    "max": 7.677e-02
  }
}
```

**Verdict**: ✅ **VERIFIED**
- All gradient norms are non-zero
- Gradients grow 3-5× over 10 epochs (Wq: 3.3e-3 → 1.0e-2)
- Direct measurement from `torch.nn.utils.clip_grad_norm_`

**Conclusion**: Backprop works correctly. Gradients reach attention layers.

---

### ⚠️ H2: Gradients Too Small for Effective Learning (INFERRED)

**Hypothesis**: Gradient magnitudes are too small to cause significant parameter updates.

**Evidence**:
```python
# Calculation (no direct measurement)
Δw = grad_norm × learning_rate
Wq: 6.33e-3 × 3e-3 = 1.90e-5
Wk: 1.34e-2 × 3e-3 = 4.02e-5
encoder: 3.50e-2 × 3e-3 = 1.05e-4

# Relative update (assuming ||w|| ≈ 1.0)
Wq: 1.90e-5 / 1.0 = 0.0019% per epoch
Wk: 4.02e-5 / 1.0 = 0.0040% per epoch
encoder: 1.05e-4 / 1.0 = 0.0105% per epoch
```

**Missing Data**:
- ❌ Actual parameter norms `||Wq||`, `||Wk||`, `||encoder||`
- ❌ Direct measurement of `||θ_t - θ_0||`

**Verdict**: ⚠️ **INFERRED**
- Calculated from `grad × lr`, not directly measured
- Assumes parameter norms ~1.0 (typical Xavier initialization)
- Conclusion plausible but lacks ground truth

---

### ✅ H3: QK Direction Wrong (Uncorrelated with g) (VERIFIED)

**Hypothesis**: QK attention scores are uncorrelated with physics signal g, indicating random walk.

**Evidence**:
```json
{
  "qk_g_corr_pearson_mean": [
    0.0512,  // epoch 1
    0.0100,  // epoch 2
   -0.0115,  // epoch 3
    0.0225,  // epoch 4
   -0.0043,  // epoch 5
   -0.0176,  // epoch 6
    0.0169,  // epoch 7
   -0.0042,  // epoch 8
    0.0033,  // epoch 9
   -0.0069   // epoch 10
  ],
  "mean": 0.0059,
  "std": 0.0192,
  "range": [-0.0176, 0.0512]
}
```

**Verdict**: ✅ **VERIFIED**
- Pearson correlation oscillates around zero (≈ 0.006 ± 0.019)
- No systematic trend toward alignment
- Direct measurement via `torch.corrcoef(qk_scores.flatten(), g.flatten())`

**Conclusion**: QK attention produces random scores with respect to physics signal. Direction is fundamentally wrong.

---

### ⚠️ H4: Double Softmax Dilution (99.93%) (INFERRED)

**Hypothesis**: Two consecutive softmax operations dilute gradients by factor of 0.026² ≈ 0.00068.

**Theoretical Prediction**:
```python
# When w ≈ uniform(1/N), softmax gradient:
∂softmax_i/∂α_i ≈ w_i(1 - w_i) ≈ (1/N)(1 - 1/N)

# For E=37 experts:
∂w_e/∂α ≈ 1/37 × 36/37 ≈ 0.026

# For M=8 atoms:
∂w_a/∂α ≈ 1/8 × 7/8 ≈ 0.109

# Combined:
0.026 × 0.109 ≈ 0.0028 (99.7% loss)
```

**Available Evidence**:
```json
{
  "w_e_entropy_mean": 2.299,  // epoch 1
  "max_entropy": log(37) = 3.611,
  "uniformity": 2.299 / 3.611 = 63.6%
}
```

**Missing Data**:
- ❌ `w_a_entropy_mean` (atom-level routing entropy)
- ❌ `scores_atoms` distribution (min/max/mean/std)
- ❌ `scores_expert` distribution
- ❌ Direct softmax gradient measurements

**Verdict**: ⚠️ **INFERRED**
- `w_e_entropy` suggests moderate dispersion (63.6% of max)
- Theory predicts 99.7% gradient loss if both softmaxes are uniform
- Lacks direct measurement of `scores` or `w_a` to confirm

**Conclusion**: Likely true based on entropy, but needs complete routing statistics to verify.

---

### ❓ H5: L2 Pooling Gradient Vanishing (HYPOTHESIZED)

**Hypothesis**: L2 norm pooling causes gradient vanishing when `scores_atoms ≈ 0`.

**Theoretical Prediction**:
```python
# Forward:
scores_expert = sqrt(sum(scores_atoms²))

# Gradient:
∂scores_expert/∂scores_atoms = scores_atoms / scores_expert

# If scores_atoms ≈ 0 (zero-centered after normalization):
# → Numerator ≈ 0
# → ∂/∂scores_atoms ≈ 0 (vanishing)
```

**Missing Data**:
- ❌ `scores_atoms_min/max/mean/std`
- ❌ `scores_expert_min/max/mean/std`
- ❌ Distribution analysis: are scores zero-centered?

**Verdict**: ❓ **HYPOTHESIZED**
- Pure theoretical prediction
- **Zero empirical evidence**
- Cannot verify or reject without `scores` statistics

**Conclusion**: This is a **critical gap**. L2 pooling is a core bottleneck claim but lacks any supporting data.

---

### ❓ H6: g Sparsity Suppresses 96.6% Gradients (HYPOTHESIZED)

**Hypothesis**: Physics signal g is sparse (only ~10/296 atoms have |g| > threshold), suppressing 96.6% of gradients.

**Theoretical Prediction**:
```python
# Gradient to k:
∂L/∂k = ∂L/∂weighted × (q * g)

# If g[e, a] ≈ 0 for most atoms:
# → Gradient vanishes for those atoms

# Estimated sparsity:
total_atoms = E × M = 37 × 8 = 296
significant_atoms ≈ 10 (physics-based estimate)
suppressed_ratio = (296 - 10) / 296 = 96.6%
```

**Missing Data**:
- ❌ `g_min/max/mean/std`
- ❌ `g_near_zero_ratio` (e.g., `(|g| < 0.01).mean()`)
- ❌ `g_p50/p95/p99` (percentiles)
- ❌ `g_top10_mean` (verify "only ~10 atoms" claim)

**Verdict**: ❓ **HYPOTHESIZED**
- Pure theoretical prediction based on physics intuition
- **Zero empirical evidence**
- The "96.6%" figure is an estimate, not measured

**Conclusion**: This is the **most critical missing data**. g sparsity is central to explaining why gradients don't work, but we have no measurements.

---

### ✅ H7: Transformer Depth Bottleneck (VERIFIED - BUT WRONG!)

**Hypothesis**: 6-layer TransformerEncoder causes exponential gradient attenuation (0.1-0.3 per layer).

**Evidence**:
```json
{
  "grad_norms": {
    "Wq": 6.33e-03,
    "Wk": 1.34e-02,
    "encoder": 3.50e-02  // LARGER than Wq/Wk!
  },
  "ratio": {
    "encoder / Wq": 3.50e-02 / 6.33e-03 = 5.5×,
    "encoder / Wk": 3.50e-02 / 1.34e-02 = 2.6×
  }
}
```

**Verdict**: ✅ **VERIFIED (hypothesis was WRONG)**
- Encoder gradients are **larger** than Wq/Wk, not smaller
- Data **contradicts** the bottleneck theory
- Encoder is NOT the primary gradient bottleneck

**Conclusion**: This is an important **negative result**. The original hypothesis was based on typical deep network behavior, but data proves otherwise. Likely due to residual connections + multiple gradient paths (MHA + FFN) summing constructively.

---

### ⚠️ H8: High Coherence Gradient Degeneracy (INFERRED)

**Hypothesis**: Dictionary coherence μ=0.9995 causes low effective rank, leading to gradient degeneracy.

**Available Evidence**:
```json
{
  "coherence_P_D": 0.9995,  // Known from preprocessing
  "grad_norms.P_D": 1.73e-03,  // Very small
  "grad_norms.P_R": 1.76e-04   // Even smaller
}
```

**Theory**:
```python
# High coherence → P_D^T P_D has near-zero eigenvalues
# → Gradient matrix ill-conditioned
# → Small gradients in low-rank subspace
```

**Missing Data**:
- ❌ Singular value spectrum of P_D
- ❌ Effective rank (e.g., `sum(σ) / max(σ)`)
- ❌ Condition number `max(σ) / min(σ)`

**Verdict**: ⚠️ **INFERRED**
- μ=0.9995 is known (very high)
- P_D gradient is very small (1.73e-03, ~100× smaller than encoder)
- Indirect support for hypothesis, but lacks spectral analysis

**Conclusion**: Plausible based on coherence + small gradients, but needs rank analysis to confirm degeneracy mechanism.

---

### ✅ H9: Teacher Supervision Ineffective (VERIFIED)

**Hypothesis**: Despite perfect teacher accuracy, QK attention does not learn to match teacher selections.

**Evidence**:
```json
{
  "teacher_acc_subset": 1.0,  // Teacher is perfect (all 10 epochs)
  "qk_top1_match_rate": [
    0.0450,  // epoch 1 (4.5%)
    0.0270,  // epoch 2 (2.7%)
    0.0450,  // epoch 3
    0.0270,  // epoch 4
    0.0360,  // epoch 5
    0.0450,  // epoch 6
    0.0180,  // epoch 7
    0.0270,  // epoch 8
    0.0270,  // epoch 9
    0.0270   // epoch 10
  ],
  "mean_match_rate": 0.0309  // ~3%, essentially random (1/37 = 2.7%)
}
```

**Verdict**: ✅ **VERIFIED**
- Teacher achieves 100% accuracy (perfect supervision signal)
- QK top-1 match rate ≈ 3% (random guessing baseline: 1/37 = 2.7%)
- No improvement over 10 epochs
- Direct measurement via `(qk_pred == teacher_pred).float().mean()`

**Conclusion**: Teacher supervision completely fails. QK ignores the supervision signal and continues random selection.

---

### ✅ H10: Parameters Barely Move (<3%) (VERIFIED - BUT WRONG!)

**Hypothesis**: Parameters change by <3% over training, indicating near-frozen weights.

**Evidence**:
```json
{
  "param_delta": {
    "Wq": [0.617, 0.463, 0.301, 0.257, 0.320, 0.302, 0.463, 0.474, 0.315, 0.345],
    "Wk": [0.604, 0.433, 0.291, 0.234, 0.276, 0.292, 0.366, 0.349, 0.297, 0.336],
    "encoder": [3.070, 2.457, 2.011, 1.653, 1.413, 1.241, 1.281, 1.110, 0.899, 0.981]
  },
  "final_delta": {
    "Wq": 0.345,     // 34.5% change!
    "Wk": 0.336,     // 33.6% change!
    "encoder": 0.981 // 98.1% change!
  }
}
```

**Verdict**: ✅ **VERIFIED (hypothesis was WRONG)**
- Direct measurement via `||θ_t - θ_0||`
- Wq/Wk change by 30-35% (assuming ||θ_0|| ≈ 1.0)
- Encoder changes by ~98%
- **Much larger than predicted <3%**

**Conclusion**: Parameters DO move significantly. The "frozen parameters" hypothesis is **rejected**. The problem is not lack of updates, but **wrong direction** of updates (H3).

---

## Evidence Strength Statistics

| Category | Count | Percentage |
|----------|-------|------------|
| ✅ VERIFIED (direct measurement) | 5/10 | 50% |
| ⚠️ INFERRED (indirect evidence) | 3/10 | 30% |
| ❓ HYPOTHESIZED (no data) | 2/10 | 20% |

**Verifiability**: 80% (8/10 can be assessed with current or indirect data)  
**Data Completeness**: 50% (only 5/10 have direct measurements)

---

## Critical Missing Data

### 🔴 CRITICAL (blocks core hypothesis verification)

**1. Physics Signal g Statistics**

Required to verify **H6: g sparsity suppresses 96.6% gradients**

```python
g_stats = {
    'g_min': float(g.min()),
    'g_max': float(g.max()),
    'g_mean': float(g.abs().mean()),
    'g_std': float(g.std()),
    'g_p50': float(torch.median(g.abs())),
    'g_p95': float(torch.quantile(g.abs(), 0.95)),
    'g_p99': float(torch.quantile(g.abs(), 0.99)),
    'g_near_zero_ratio': float((g.abs() < 0.01).float().mean()),
    'g_top10_mean': float(torch.topk(g.abs().flatten(), k=10).values.mean()),
}
```

**Why critical**: The "96.6% suppression" claim is a central bottleneck explanation, but we have **zero** measurements of g distribution.

---

**2. Scores Distribution (atoms & expert)**

Required to verify **H5: L2 pooling gradient vanishing**

```python
scores_stats = {
    'scores_atoms_min': float(scores_atoms.min()),
    'scores_atoms_max': float(scores_atoms.max()),
    'scores_atoms_mean': float(scores_atoms.mean()),
    'scores_atoms_std': float(scores_atoms.std()),
    'scores_expert_min': float(scores_expert.min()),
    'scores_expert_max': float(scores_expert.max()),
    'scores_expert_mean': float(scores_expert.mean()),
    'scores_expert_std': float(scores_expert.std()),
}
```

**Why critical**: L2 pooling gradient vanishing is a core bottleneck claim, but we cannot verify if `scores_atoms ≈ 0` without distribution data.

---

### 🟡 HIGH (strongly recommended for complete analysis)

**3. Atom-level Routing Entropy**

Required to complete **H4: Double softmax dilution** verification

```python
w_a_entropy_mean = float(np.mean([
    entropy(w_a[e].detach().cpu().numpy()) 
    for e in range(E)
]))
```

**Why high**: We have `w_e_entropy`, but lack `w_a_entropy` to verify both softmax stages are near-uniform.

---

**4. Direct Parameter Norms**

Required to strengthen **H2: Gradients too small**

```python
param_norms = {
    'Wq_norm': float(model.Wq.weight.norm()),
    'Wk_norm': float(model.Wk.weight.norm()),
    'encoder_norm': float(sum(p.norm() for p in model.encoder.parameters())),
}
```

**Why high**: Currently using `param_delta / assumed_norm`, but direct norms would eliminate assumptions.

---

### 🟢 MEDIUM (useful but not essential)

**5. Per-layer Gradient Decomposition**

Would provide finer-grained analysis of **H7: Transformer depth**

```python
layer_grads = {
    f'encoder_layer_{i}_grad': float(get_layer_grad_norm(model.encoder.layers[i]))
    for i in range(num_layers)
}
```

---

**6. Spectral Analysis of P_D**

Would complete **H8: Coherence degeneracy** verification

```python
U, S, V = torch.svd(P_D)
coherence_stats = {
    'singular_values': S.tolist(),
    'effective_rank': float((S.sum() / S.max())),
    'condition_number': float(S.max() / S.min()),
}
```

---

## Key Findings Summary

### ✅ What We CAN Verify (5/10)

1. **Gradients flow** → YES (grad_norms prove it)
2. **Direction wrong** → YES (qk_g_corr ≈ 0)
3. **Teacher fails** → YES (teacher_acc=1 but QK doesn't follow)
4. **Parameters move** → YES (Δθ = 30-35%, hypothesis was wrong)
5. **Encoder not bottleneck** → YES (encoder grad > Wq/Wk, hypothesis was wrong)

### ❌ What We CANNOT Verify (2/10)

1. **g sparsity 96.6%** → NO DATA (missing g_stats)
2. **L2 pooling vanishing** → NO DATA (missing scores distribution)

### ⚠️ What We Can INFER (3/10)

1. **Gradients too small** → CALCULATED (grad × lr, no direct param norms)
2. **Double softmax 99.93%** → PARTIAL (w_e_entropy only, missing w_a)
3. **Coherence degeneracy** → PARTIAL (μ known + P_D grad small, but no spectral analysis)

---

## Evidence Chain Weaknesses

**Problem 1: Cannot quantify g sparsity**
- **Claim**: "Element-wise multiplication with g suppresses 96.6% of gradients"
- **Evidence**: ❌ None - pure theoretical estimate
- **Impact**: Cannot verify if g is actually sparse or uniformly distributed

**Problem 2: Cannot observe scores distribution**
- **Claim**: "L2 pooling causes gradient vanishing because scores_atoms ≈ 0"
- **Evidence**: ❌ None - pure theoretical prediction
- **Impact**: Cannot verify if scores are indeed zero-centered

**Problem 3: Incomplete routing analysis**
- **Claim**: "Double softmax dilutes gradients by 99.93%"
- **Evidence**: ⚠️ Partial - only w_e_entropy available
- **Impact**: Cannot confirm both softmax stages are uniform

---

## Answer to User Question

**Question**: "從 commit 70ba576 有辦法證實你的假說嗎？"

**Answer**: **部分可以，但 2 個最關鍵的假說無法證實**

### Summary Table

| Status | Count | Hypotheses |
|--------|-------|------------|
| ✅ Fully verified | 5/10 | H1, H3, H7*, H9, H10* |
| ⚠️ Partially verified | 3/10 | H2, H4, H8 |
| ❌ Unverifiable | 2/10 | **H5, H6** |

*Note: H7 and H10 were verified but hypothesis was WRONG (contradicted by data)*

### What This Means

**Strong conclusions** (can state with confidence):
- ✅ Gradients flow but point in wrong direction
- ✅ Teacher supervision completely fails
- ✅ Parameters update significantly but in wrong direction
- ✅ Encoder is NOT the bottleneck (contrary to hypothesis)

**Weak conclusions** (must state with caveats):
- ⚠️ "Gradients too small" → LIKELY but not directly measured
- ⚠️ "Double softmax 99.93%" → PLAUSIBLE but incomplete evidence
- ⚠️ "Coherence degeneracy" → SUGGESTED but lacks spectral analysis

**Cannot conclude** (pure theory without evidence):
- ❌ "96.6% gradient suppression from g sparsity" → **NO DATA**
- ❌ "L2 pooling gradient vanishing" → **NO DATA**

---

## Recommendations

### For Scientific Rigor

**In the gradient flow analysis report (`qk_routing_gradient_flow_analysis.md`):**

1. **Label evidence strength** for each claim:
   - ✅ VERIFIED: Direct measurement from logs
   - ⚠️ INFERRED: Calculated or indirect evidence
   - ❓ HYPOTHESIZED: Theoretical prediction, no data

2. **Add "Data Limitations" section** documenting what cannot be verified

3. **Downgrade unsupported claims**:
   - "96.6% suppression" → "**Estimated** 96.6% (no direct measurement)"
   - "L2 pooling vanishes" → "**Hypothesized** to cause vanishing (no scores data)"

### For Future Experiments

**Must add these diagnostics** to verify all hypotheses:

```python
# In forward pass, log:
diagnostic_stats = {
    # 🔴 CRITICAL
    'g_min': float(g.min()),
    'g_max': float(g.max()),
    'g_mean': float(g.abs().mean()),
    'g_near_zero_ratio': float((g.abs() < 0.01).float().mean()),
    'scores_atoms_mean': float(scores_atoms.mean()),
    'scores_expert_mean': float(scores_expert.mean()),
    
    # 🟡 HIGH
    'w_a_entropy_mean': float(np.mean([entropy(w_a[e]) for e in range(E)])),
    'Wq_norm': float(model.Wq.weight.norm()),
}
```

**Impact**: With these additions, we could achieve **100% verification rate** (10/10 hypotheses with direct or strong indirect evidence).

---

## Conclusions

**Current State**: 
- Evidence sufficient to prove **core problem** (gradients flow but wrong direction)
- Evidence **insufficient** to prove **why** direction is wrong (bottleneck mechanisms)

**Path Forward**:
1. **Document current limitations** in analysis report with evidence labels
2. **Add missing diagnostics** (g_stats, scores_stats, w_a_entropy)
3. **Re-run experiment** with enhanced logging
4. **Update analysis** with complete evidence chain

**Timeline Estimate**:
- Code modifications: 30 minutes (add logging)
- Re-run experiment: ~10 minutes (10 epochs, CPU)
- Analysis update: 1 hour (verify hypotheses, update report)
- **Total**: ~2 hours to complete verification

---

## Appendix: Raw Data Extract

### Gradient Norms (all 10 epochs)

```python
grad_norms_wq = [
    3.345e-03, 2.308e-03, 4.378e-03, 9.556e-03, 8.773e-03,
    8.118e-03, 1.168e-02, 6.314e-03, 5.924e-03, 1.009e-02
]

grad_norms_wk = [
    4.650e-03, 3.116e-03, 7.455e-03, 2.137e-02, 1.936e-02,
    1.816e-02, 2.027e-02, 1.332e-02, 1.346e-02, 2.014e-02
]

grad_norms_encoder = [
    2.450e-02, 1.474e-02, 2.349e-02, 5.235e-02, 4.934e-02,
    4.498e-02, 5.393e-02, 3.427e-02, 3.580e-02, 7.677e-02
]
```

### QK-g Correlation (all 10 epochs)

```python
qk_g_corr = [
    0.0512, 0.0100, -0.0115, 0.0225, -0.0043,
   -0.0176, 0.0169, -0.0042, 0.0033, -0.0069
]

mean = 0.0059
std = 0.0192
```

### Parameter Deltas (all 10 epochs)

```python
param_delta_wq = [
    0.617, 0.463, 0.301, 0.257, 0.320,
    0.302, 0.463, 0.474, 0.315, 0.345
]

param_delta_wk = [
    0.604, 0.433, 0.291, 0.234, 0.276,
    0.292, 0.366, 0.349, 0.297, 0.336
]

param_delta_encoder = [
    3.070, 2.457, 2.011, 1.653, 1.413,
    1.241, 1.281, 1.110, 0.899, 0.981
]
```

### Classification Loss (all 10 epochs)

```python
class_loss = [
    3.6106, 3.6066, 3.6100, 3.6106, 3.6034,
    3.6013, 3.6018, 3.6105, 3.6143, 3.5997
]

mean = 3.6069
std = 0.0047
random_baseline = log(37) = 3.6109
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-26  
**Author**: Gradient Flow Analysis  
**Related Documents**: `qk_routing_gradient_flow_analysis.md`
