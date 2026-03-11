# Polar Plot Architecture Analysis and Frequency Decomposition Notes

> This document records the data sources for Master Figure Panel B3 (polar plot), the model architecture mapping, and a feasibility analysis for frequency decomposition.

## 1. Neural Network Architecture Overview (Conceptual)

High-level flow:

1. Inputs:
   - Y (signal spectrum), shape (346,)
   - D (physical dictionary), shape (346, 296) where 296 = 37 experts * 8 atoms
2. Token embedding:
   - Residual token: t_R = P_R(Y) + type_R, shape (d,)
   - Dictionary tokens: T_D = P_D(D) + type_D, shape (296, d)
   - Combined token sequence: T = [t_R; T_D], shape (297, d)
3. Transformer encoder with mask:
   - Residual token attends to all tokens
   - Dictionary tokens attend only to themselves
4. QK attention scores (model-native):
   - scores_atoms = (Wk(H_D) @ Wq(h_R)) / sqrt(d), shape (296,)
   - reshape to (37, 8)
   - scores_expert = sqrt(sum(|scores_atoms|^2) over 8 atoms), shape (37,)
5. Physics baseline (OMP):
   - g_energy = |D.T @ Y|, shape (296,)
   - reshape to (37, 8), sum over atoms -> g_energy_expert, shape (37,)

## 2. Data Shape Reference

### 2.1 Input data (dict_data)

| Variable | Shape | Description |
| --- | --- | --- |
| `D` | (346, 296) | Physical dictionary: frequency x total atoms |
| `H` | (346, 37) | Angle transfer functions: frequency x angle |
| `W_reduced` | (346, 8) | Speech bases (reduced): frequency x bases |
| `angles` | (37,) | Angle values [5 deg, 10 deg, ..., 185 deg] |

### 2.2 Network outputs (routing_data)

| Variable | Shape | Source | Description |
| --- | --- | --- | --- |
| `Y_val` | (1924, 346) | Input | Input signal spectra |
| `labels` | (1924,) | Ground truth | True angle index (0-36) |
| `scores_atoms` | (1924, 37, 8) | QK attention | Attention score per atom |
| `scores_expert` | (1924, 37) | QK attention | L2-pooled score per expert |
| `g_energy_expert` | (1924, 37) | Physics (non-NN) | OMP energy |D.T @ Y| |

### 2.3 Key dimensions

| Symbol | Value | Meaning |
| --- | --- | --- |
| N_samples | 1924 | Validation set samples |
| N_freqs | 346 | Frequency bins (300-3000 Hz) |
| N_experts | 37 | Number of angles (5 deg to 185 deg, step 5 deg) |
| N_atoms | 8 | Atoms per angle |
| N_total | 296 | Total atoms = 37 * 8 |

## 3. Polar Plot Data Sources (B3)

- Orange curve (Traditional OMP)
  - Variable: `routing_data['g_energy_expert']`
  - Shape: (1924, 37) -> select one sample -> (37,)
  - Source: |D.T @ Y| summed over atoms
  - Meaning: pure physics projection energy

- Green curve (Physics-Aware AI)
  - Variable: `routing_data['scores_expert']`
  - Shape: (1924, 37) -> select one sample -> (37,)
  - Source: QK attention = (Wk(H_D) @ Wq(h_R)) / sqrt(d)
  - Meaning: transformer-learned attention score

- Green dashed marker (Ground truth)
  - Variable: `routing_data['labels']`
  - Meaning: true angle

## 4. Why the Original Polar Plot is Full-Band

### 4.1 Design rationale

The original design uses full-band aggregated scores:

```python
# Original code (create_master_figure.py, lines 348-351)
physics_scores = routing_data['g_energy_expert'][sample_idx, :]  # (37,)
qk_scores = routing_data['scores_expert'][sample_idx, :]         # (37,)
```

These scores already aggregate over all frequencies:

| Method | Aggregation |
| --- | --- |
| Physics (OMP) | g_energy = |D.T @ Y| sums across 346 frequency bins |
| QK attention | scores_atoms come from a transformer that ingests the full-band projection |

Design considerations:
1. Simplicity: a single plot shows the final angle selection contrast.
2. Focus: highlights discrimination differences between QK and OMP.
3. Complexity: frequency decomposition increases reader burden.

### 4.2 Technical constraint

QK attention scores are not frequency-separable by construction:
- The query h_R is a contextual embedding from the transformer encoder.
- There is no direct per-frequency path to split QK scores.

To decompose by frequency, an indirect method is required:

```python
# Indirect computation: QK weights x dictionary energy in a band
qk_freq_contribution = D[freq_mask, :] @ scores_atoms.flatten()
```

## 5. Pros/Cons of Frequency Decomposition

### 5.1 Full-band aggregation (original)

| Pros | Cons |
| --- | --- |
| Simple and direct | No visibility into frequency mechanisms |
| Directly reflects model decision | Cannot explain why QK is more accurate |
| Fits main-figure layout | May miss important physical insight |
| No extra assumptions | Cannot test frequency selectivity |

### 5.2 Frequency decomposition (500/1000/2000 Hz)

| Pros | Cons |
| --- | --- |
| Reveals which bands carry discriminative signal | More complex figure |
| Tests physics hypothesis: low vs high frequency sensitivity | Requires choosing representative bands |
| Improves interpretability | Indirect computation, not model-native |
| Supports physics-AI narrative | Adds analysis assumptions |

### 5.3 Recommended usage

| Scenario | Recommendation |
| --- | --- |
| Main paper figure | Keep full-band aggregation (simple) |
| Supplementary figure | Add frequency decomposition (deeper analysis) |
| Reviewer response | Use frequency decomposition to answer mechanism questions |
| Talks / presentations | Choose based on audience background |

## 6. Frequency Decomposition Methods

### 6.1 Physics (OMP) band score

```python
def physics_freq_score(Y, D, target_freq, bandwidth=100):
    """
    Compute OMP energy for a target frequency band.
    """
    freqs = np.linspace(300, 3000, 346)
    freq_mask = (freqs >= target_freq - bandwidth / 2) & \
                (freqs <= target_freq + bandwidth / 2)

    D_band = D[freq_mask, :]           # (n_band, 296)
    Y_band = Y[freq_mask]              # (n_band,)

    g_energy_band = np.abs(D_band.T @ Y_band)  # (296,)
    g_energy_expert = g_energy_band.reshape(37, 8).sum(axis=1)  # (37,)

    return g_energy_expert
```

### 6.2 QK attention band score (indirect)

```python
def qk_freq_score(Y, D, scores_atoms, target_freq, bandwidth=100):
    """
    Compute the effective QK contribution in a target frequency band.

    Idea: QK selects atoms; measure their weighted response in the band.
    """
    freqs = np.linspace(300, 3000, 346)
    freq_mask = (freqs >= target_freq - bandwidth / 2) & \
                (freqs <= target_freq + bandwidth / 2)

    D_band = D[freq_mask, :]  # (n_band, 296)
    Y_band = Y[freq_mask]     # (n_band,)

    qk_scores = np.zeros(37)
    for expert_idx in range(37):
        atom_slice = slice(expert_idx * 8, (expert_idx + 1) * 8)
        D_expert_band = D_band[:, atom_slice]      # (n_band, 8)
        qk_weights = scores_atoms[expert_idx, :]   # (8,)

        # QK-weighted band response
        weighted_response = D_expert_band @ qk_weights  # (n_band,)
        qk_scores[expert_idx] = np.abs(weighted_response @ Y_band)

    return qk_scores
```

## 7. Expected Results and Hypothesis

If frequency decomposition is applied, expected observations:

| Band | Physics (OMP) | QK attention | Physical interpretation |
| --- | --- | --- | --- |
| 500 Hz (low) | Similar across angles | Possibly more discriminative | Long wavelength -> lower angle sensitivity |
| 1000 Hz (mid) | Moderate spread | Moderate focus | Mid-band has moderate resolution |
| 2000 Hz (high) | More discriminative | Most discriminative | Short wavelength -> higher angle sensitivity |

Core hypothesis: QK attention learns to emphasize higher frequencies because they are more angle-sensitive.

## 8. Related Code Locations

| File | Content |
| --- | --- |
| `scripts/create_master_figure.py` | Original master figure generation |
| `scripts/omp-transformer.py` | FullTransformerRoutedSoftOMP model definition |
| `doa_rl/model/transformer.py` | Base transformer architecture |

## 9. Reference: Commit 872aa65

This commit adjusted the master figure to Nature Communications formatting:
- Removed Pearson r textbox (moved to caption)
- Removed cyan arrow annotations
- Reduced font size to 6-10 pt
- Reduced panel spacing

Polar plot data sources and computation logic did not change.

Document created: 2025-01-19
Related commit: 872aa65373f4be4b3cd69774b3656facb2e33e61
