# Two-Stage Learning for Cross-Sensor Speech Localization

> Research design document for Interspeech 2026 submission

---

## H (Transfer Function)

| Property | Description |
|----------|-------------|
| **Source** | White noise recordings @ 37 angles (0°-180°, 5° step) |
| **Form** | H(f, θ) - independent frequency response per angle |
| **Property** | Content-independent (white noise excitation) |

**Key insight**: H is pre-computed, not learned. It captures the angle-dependent acoustic properties of the Mic-to-LDV system.

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
- Frequency embedding resolves phase ambiguity (Δφ = 2πf·Δτ)

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
| Results | Energy Reduction (97.11%) + Direction Accuracy (TBD) |
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
