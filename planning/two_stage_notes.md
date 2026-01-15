# Two-Stage Learning for Cross-Sensor Speech Localization

> Research design document for Interspeech 2026 submission

---

## H (Transfer Function)

| Property | Description |
|----------|-------------|
| **Source** | White noise recordings @ 37 angles (0°-180°, 5° step) |
| **Form** | `H_cal(f, θ) ∈ ℂ` (angle-indexed complex transfer function) |
| **Property** | Content-independent estimate under an LTI assumption (white-noise excitation) |

**Key insight**: `H_cal` is measured (pre-computed), not learned. It captures the *angle-dependent* Mic→LDV transfer characteristics of the specific sensor+environment setup.

---

## Physical Model and Assumptions

We use the following *explicit* assumptions when talking about “physics” in this project:

- **Fixed geometry**: sensor placement and source direction are static within a recording.
- **Linear, time-invariant (LTI) approximation**: within a clip, the Mic→LDV mapping can be approximated as an LTI system.
- **Small-signal regime**: the LDV + vibro-acoustic response is treated as linear around the operating point.

### Time-domain model (per angle)

Let `x(t)` be the microphone waveform and `y_θ(t)` be the LDV waveform recorded at direction `θ`. We assume

`y_θ(t) = (h_θ * x)(t) + ε(t)`,

where `h_θ` is the (unknown) impulse response and `ε(t)` is measurement noise / mismatch.

### STFT-domain model (narrowband approximation)

Let `X(f, n)` and `Y_θ(f, n)` be complex STFTs (frequency bin `f`, frame index `n`). Under the usual narrowband approximation,

`Y_θ(f, n) ≈ H_θ(f) · X(f, n)`,

where `H_θ(f)` is a complex transfer function (magnitude + phase).

### How `H_cal(f, θ)` is estimated from white noise

With white-noise excitation, we estimate `H_cal(f, θ)` by time-averaging cross/auto spectra, e.g.

`H_cal(f, θ) = S_yx(f, θ) / (S_xx(f, θ) + δ)`.

This is “content-independent” only in the sense that the estimator targets the *system* under the LTI approximation.

---

## Two-Stage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Learn Mic→LDV Speech Transformation (single θ₀)   │
│ ─────────────────────────────────────────                    │
│ • Input: Mic speech spectrogram                              │
│ • Output: Reconstructed LDV speech spectrogram               │
│ • Method: Freq-Aware DTMin (OMP distillation)                │
│ • Learns: Frequency-dependent transformation features        │
│           (without angle information)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: Direction Estimation Fine-tuning (introduce H)     │
│ ─────────────────────────────────────────                    │
│ • Input: Stage 1 encoder output + H(f, θ₁...θ₃₇)           │
│ • Output: Direction classification (37-way)                  │
│ • Method: Direction Head learns feature ↔ H correspondence  │
│ • Validates: Stage 1 features are useful for downstream     │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Logic

| Component | What it learns | What it excludes |
|-----------|----------------|------------------|
| Stage 1 | Mic→LDV frequency transformation features | Angle information |
| H | Angle-dependent frequency response | Speech content |
| Stage 2 | Feature ↔ H correspondence | - |

### Why this design works

1. **Stage 1 features** are content-aware but angle-agnostic
2. **H** is content-agnostic but angle-aware
3. **Combining both** provides complete content + angle information

---

## Stage 1: Details

### Training Setup
- **Fixed angle**: Single angle θ₀ (no angle labels used)
- **Task**: Mic speech → LDV speech reconstruction
- **Loss**: Energy reduction (minimize reconstruction residual)

### Method: Freq-Aware DTMin
- OMP teacher provides optimal lag selection per frequency bin
- DTMin learns to imitate OMP across multiple frequencies
- Frequency embedding resolves phase-wrapping ambiguity (φ(f) = -2π·f·τ mod 2π)

### Output
- Encoder that produces frequency-dependent transformation features
- 97.11% Energy Reduction achieved (commit 2067cec)

---

## Stage 2: Details (To be implemented)

### Architecture
```
Mic Speech ──→ [Stage 1 Encoder] ──→ Features z
                 (frozen/fine-tune)        │
                                           ↓
                              ┌────────────────────────┐
                              │   Direction Head       │
                              │   f(z, H(θ₁...θ₃₇))   │
                              └────────────────────────┘
                                           │
                                           ↓
                                Direction logits (37-way)
```

### Ablation Experiments
| Variant | Description | Purpose |
|---------|-------------|---------|
| A: Frozen | Freeze Stage 1 encoder | Validate feature quality directly |
| B: Fine-tune | Fine-tune entire model | Best direction accuracy |
| C: End-to-End | No Stage 1, train from scratch | Prove pretraining value |

### Evaluation Metrics
- Top-1 Accuracy (37 classes)
- Accuracy@10° (±2 classes tolerance)

---

## Paper Presentation

| Section | Content |
|---------|---------|
| Method | Stage 1: Freq-Aware DTMin + Stage 2: Direction Head |
| Results | Energy Reduction (97.11%) + Direction Accuracy (not yet measured) |
| Ablation | Frozen vs Fine-tune vs End-to-End |

---

## Original Notes (preserved)

```
兩階段
1. 先只學聲學特徵，可以完整 H (白噪音) 但加上 masking
  1.1 先想成像 SVD 有很多不同 mode ， OMP 要去學挑哪些 mode 可以最快重建 LDV (residual 最小)
  1.2 同一句話，男女生頻率會不一樣，所以必須要多組不同 OMP 才可能涵蓋
  1.3 用 DTMin 一次學起來
2. fine tuning 方向，這時候就不用 masking

因為方向識別看成是下游應用，驗證第一階段真的有學到好的轉換特徵
方向識別是作為驗證器
```
