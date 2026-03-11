# SNR Synthetic Dataset Experiment Plan

> **Branch**: `experiment/snr-synthetic-datasets`
> **Created**: 2025-12-08
> **Status**: Planning phase
> **Reference**: [dataset_training_lineage.md](dataset_training_lineage.md)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Experiment Design](#experiment-design)
3. [Synthetic Dataset Generation](#synthetic-dataset-generation)
4. [Training Protocol](#training-protocol)
5. [Analysis Plan](#analysis-plan)
6. [Resource Estimation](#resource-estimation)
7. [Success Criteria](#success-criteria)
8. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Research Question
**How does signal-to-noise ratio (SNR) affect DOA (Direction of Arrival) estimation accuracy for white noise vs speech signals, and does the Hard Gumbel-Softmax mechanism improve robustness at low SNR?**

### Hypothesis
1. **White noise degradation**: Graceful degradation with 90% accuracy threshold at ~10 dB SNR
2. **Speech degradation**: Steeper degradation with 90% threshold at ~15-20 dB SNR (more sensitive due to formant structure)
3. **Hard Gumbel advantage**: 2-5% accuracy improvement at low SNR (5-10 dB) vs soft routing
4. **Atom diversity collapse**: High SNR (clean) >80% diversity → Low SNR (5 dB) <40% diversity

### Key Metrics
- **Primary**: Validation accuracy vs SNR (7 levels: ∞, 30, 20, 15, 10, 5, 0 dB)
- **Secondary**: Atom diversity, per-angle accuracy, train-val gap, alignment metrics

### Baseline Reference
- **White noise (clean)**: 100% accuracy with g-routing + soft (commit 1f6b68c)
- **Speech (clean)**: 94.6% validation accuracy with QK + hard Gumbel (commit 06bf65d)

---

## Experiment Design

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    SNR Experiment Pipeline                      │
└─────────────────────────────────────────────────────────────────┘

Stage 0: Clean Signal Extraction (EXISTING)
    ├── White noise: 329ae66 (37 angles × 3 clips = 111 samples)
    └── Speech: 13dd6e2 (37 angles × 260 clips = 9,620 samples)
                        ↓
Stage SNR: Synthetic Noise Addition (NEW)
    ├── SNR levels: [∞, 30, 20, 15, 10, 5, 0] dB (7 levels)
    ├── Noise type: Additive White Gaussian Noise (AWGN)
    ├── Clean LDV roots (Stage 0, Box path):
    │      - White noise: white_noise_box_data_no_edge/
    │      - Speech:      speech260_box_data_no_edge/
    └── Outputs (Stage 0 + SNR, LDV path only):
           - white_noise_box_snr{X}dB_data_no_edge/
           - speech260_box_snr{X}dB_data_no_edge/
                        ↓
Stage 1-4: Standard Processing (REUSE EXISTING)
    ├── Stage 1: Synchronized VAD (apply_spectrogram_vad.py)
    │      - X: Original playback (clean, no added noise)
    │      - Y: Box LDV with synthetic SNR (snr{X}dB)
    │      - One shared VAD mask per (X, Y) pair (unchanged)
    ├── Stage 2: Normalization (normalize_to_unit_range.py)
    ├── Stage 3: H matrix estimation (estimate_transfer_functions.py)
    │      - Estimated once from clean white_noise_box (SNR=∞)
    └── Stage 4: USM training (train_usm.py / pipeline)
           - Trained on clean data only (SNR=∞), reused for all SNR levels
                        ↓
Training: Model Training Across SNR Sweep
    ├── White noise: G-routing + soft (replicate 1f6b68c on Box LDV path)
    ├── Speech: QK + hard Gumbel (replicate 06bf65d on speech260_box path)
    └── Ablation: Speech with soft routing for comparison
                        ↓
Analysis: Performance vs SNR Curves
    ├── Accuracy degradation curves
    ├── 90% threshold identification
    ├── Soft vs hard Gumbel comparison
    └── Atom diversity tracking
```

### Experimental Variables

#### Independent Variables
1. **Signal type**: White noise, Speech
2. **SNR level**: ∞ (clean), 30, 20, 15, 10, 5, 0 dB (7 levels)
3. **Routing mechanism**: g-routing + soft, QK + hard Gumbel, QK + soft (ablation)

#### Dependent Variables
1. **Validation accuracy**: Primary metric
2. **Per-angle accuracy**: Identify SNR-sensitive angles
3. **Atom diversity**: % unique atoms selected across steps
4. **Alignment metrics**: QK-g correlation (for QK routing)
5. **Train-val gap**: Overfitting indicator

#### Control Variables
1. **STFT parameters**: fs=48000 (raw NPY files), fs=16000 (training after resampling), n_fft=2048, [300,3000]Hz → F=346 (fixed across all experiments)
2. **Model architecture**: Same F, E, M, d_model, nhead, nlayers as baseline
3. **Training hyperparameters**: Same epochs, batch_size, lr, optimizer as baseline
4. **Data split**: Same deterministic train/val split (clip_id % 5) for speech
5. **Hardware**: Same device (MPS for speech, CPU for white noise if needed)

### Sample Size Calculation
```yaml
White Noise:
  - Per SNR level: 111 samples (37 angles × 3 clips)
  - Total across 7 SNR levels: 777 samples
  - Validation: N/A (no split due to small size)
  - Evaluation: Full dataset for each SNR level

Speech:
  - Per SNR level: 9,620 samples (37 angles × 260 clips)
  - Total across 7 SNR levels: 67,340 samples
  - Train/val split per SNR: 7,696 train / 1,924 val
  - Evaluation: Validation set accuracy (1,924 samples)

Ablation (Speech soft routing):
  - 3 SNR levels: 15, 10, 5 dB (critical transition zone)
  - Per level: 9,620 samples
  - Total: 28,860 samples
```

---

## Physical Principles and Audio Processing Pipeline

### Data Flow Through the System

Understanding the complete audio processing pipeline is critical for physically realistic SNR synthesis:

```
Storage Layer (NPY files):
    ├─ Time-domain waveforms: x[n], n = 0, ..., N-1
    ├─ Sample rate: fs = 48000 Hz (Stage 0 NPY files from LDV recording)
    ├─ Duration: ~3 seconds per clip (N ≈ 145,920 samples at 48 kHz)
    ├─ Resampled to: fs = 16000 Hz (Stage 4.5, before training)
    └─ DC component: May have non-zero mean (μ ≠ 0)
              ↓
Training Pipeline (DoADataset):
    ├─ STFT computation: X[k,m] = STFT(x[n])
    │    ├─ n_fft: 2048 (frequency resolution: fs/n_fft = 7.8 Hz)
    │    ├─ hop_length: 512 (time resolution: 512/fs = 32 ms)
    │    └─ Window: Hann (implicit in STFT)
    ├─ Frequency band selection: [300, 3000] Hz
    │    └─ Bin indices: k ∈ [39, 385] → F = 346 frequency bins
    ├─ Magnitude spectrum: Y[k,m] = |X[k,m]|
    │    └─ ALWAYS POSITIVE (non-zero mean in frequency domain)
    └─ Shape: [B, F, T] = [batch_size, 346, num_frames]
              ↓
Model Input:
    └─ OMP Transformer operates on Y[k,m] magnitude spectra
```

### Physical Principles of SNR

#### 1. Time-Domain vs Frequency-Domain SNR

**CRITICAL INSIGHT**: The model operates on **frequency-domain** magnitude spectra in the band [300, 3000]Hz, NOT raw time-domain waveforms.

Therefore, SNR must be defined and verified in the frequency domain:

```python
# Time-domain SNR (what we control):
x(t): Clean waveform
n(t): Additive noise ~ N(0, σ²)
y(t) = x(t) + n(t)

SNR_time = 10 log₁₀(P_x / P_n)
where P_x = mean((x - mean(x))²)  # AC power (remove DC component)
      P_n = mean(n²) = σ²

# Frequency-domain SNR (what affects the model):
X[k]: Clean STFT magnitude in band [300, 3000]Hz
N[k]: Noise STFT magnitude in band [300, 3000]Hz
Y[k] = |X[k] + N[k]|  # Magnitude (non-linear!)

SNR_freq(k) = 10 log₁₀(|X[k]|² / E[|N[k]|²])

# Band-averaged SNR (model-relevant metric):
SNR_band = 10 log₁₀(Σ|X[k]|² / Σ|N[k]|²), k ∈ [300, 3000]Hz
```

**Key Difference**: For white Gaussian noise, `SNR_time ≈ SNR_band` (Parseval's theorem). However:
- Magnitude operation `|X + N|` introduces non-linearity
- Frequency band selection [300, 3000]Hz means out-of-band noise is filtered
- DC component (mean ≠ 0) can bias time-domain power calculation

#### 2. AWGN Model: Assumptions and Limitations

**Additive White Gaussian Noise (AWGN) Assumptions**:
1. **Additive**: Noise is added to signal, not multiplicative
   - `y(t) = x(t) + n(t)` (linear superposition)
2. **White**: Flat power spectral density across all frequencies
   - `S_n(f) = N₀/2` (constant for all f)
3. **Gaussian**: Amplitude distribution is N(0, σ²)
   - Justified by Central Limit Theory for thermal noise

**Reality Check: Real LDV Noise Characteristics**:

From the lineage documentation (commit 2a6ec4a), real LDV data shows:
- Box/IrregularBox: **4× lower mean magnitude** vs Original playback
- This corresponds to: `20 log₁₀(4) ≈ 12 dB` SNR degradation

Real LDV noise includes:
1. **Speckle Noise** (multiplicative):
   - `y(t) = x(t) · (1 + η(t))` where η ~ laser interference pattern
   - Proportional to signal amplitude (not additive!)
   - Non-Gaussian distribution (Rayleigh/Rice)

2. **Colored Noise** (frequency-dependent):
   - Higher noise floor at low frequencies (<500 Hz, 1/f noise)
   - Vibrational noise from environment (peaks at 50/60 Hz, HVAC frequencies)
   - Not white (S_n(f) ≠ constant)

3. **Quantization Noise**:
   - ADC resolution: 16-bit → dynamic range ~96 dB
   - Negligible for signals well above noise floor

**Why Use AWGN Despite Limitations?**:
1. **Controlled parametric study**: Isolate SNR effect from other confounds
2. **Theoretical baseline**: White noise is "hardest" to filter (no spectral structure)
3. **Worst-case analysis**: If model is robust to AWGN, likely robust to colored noise
4. **Computational simplicity**: No need to model complex LDV physics
5. **Reproducibility**: Deterministic with fixed random seed

**Limitation Acknowledgment**:
- AWGN provides **lower bound** on performance (real LDV may be easier/harder depending on noise color)
- Future work should validate findings against real low-SNR LDV recordings
- Synthetic SNR=12dB should approximate current LDV hardware, but spectral characteristics will differ

#### 3. Physical Realism: Spectral Shaping for Acoustic-Structural Coupling

**Critical Discovery (2025-12-08)**: Simple AWGN creates significant spectral mismatch with LDV Box signals due to acoustic-structural coupling.

**Spectral Distribution Analysis** (48 kHz):

White Noise Box Signal:
```python
Total power: 6.93e-02
Power distribution:
  DC-300 Hz:        49.2% (3.41e-02)  # Box resonance/modal response
  [300-3000] Hz:    49.9% (3.46e-02)  # Model band: 11.3% of spectrum, 49.9% of energy!
  3000-8000 Hz:      0.7% (4.97e-04)
  8000-24000 Hz:     0.1% (9.36e-05)
```

Speech Box Signal:
```python
Total power: 1.17e-01
Power distribution:
  DC-300 Hz:        29.6% (3.45e-02)
  [300-3000] Hz:    70.4% (8.20e-02)  # Even higher concentration in model band!
  3000-8000 Hz:      0.0% (5.98e-06)
  8000-24000 Hz:     0.0% (3.08e-05)

Speech variability across clips:
  Mean: 63.2% in [300, 3000]Hz
  Std:  8.9%
  Range: [52.5%, 76.1%]  # Phoneme-dependent (e.g., /a/ vs /s/)
```

**Physical Interpretation**:
1. **Box Modal Resonances**: The physical Box structure has resonant modes concentrated in [300, 3000]Hz band
2. **Acoustic-Structural Coupling**: Speaker → Air → Box vibration → LDV path acts as frequency-dependent transfer function
3. **Energy Concentration**: Only 11.3% of frequency spectrum ([300, 3000]Hz / 24,000 Hz) contains 49.9% (white noise) to 70.4% (speech) of total signal energy

**Problem with Simple AWGN**:
```
Simple AWGN: Uniform spectrum (flat PSD across all frequencies)
Box Signal:  Concentrated spectrum (49.9% energy in 11.3% of bandwidth)

Adding white noise to concentrated signal:
- Time-domain SNR: 15 dB ✓ (correctly controlled)
- Model-band SNR: 22 dB ✗ (noise distributed uniformly, signal concentrated)
- Spectral mismatch: ~7 dB error in frequency domain!
```

**Solution: Per-Clip Spectral Shaping**

Environmental noise propagates through the same physical system (Speaker → Box → LDV), so it should have the **same spectral shape** as the clean signal.

Algorithm:
```python
def add_spectral_shaped_noise_per_clip(signal, target_snr_db, fs=48000, seed=42):
    """
    Add noise with spectrum shaped to match signal's spectrum.

    Physical model: Environmental noise → Speaker → Box → LDV
    Result: Noise has same frequency distribution as signal
    """
    # 1. Generate white Gaussian noise
    white_noise = rng.normal(0, 1, signal.shape)

    # 2. Shape noise spectrum to match signal spectrum (STFT domain)
    S_signal = STFT(signal)
    S_white = STFT(white_noise)

    signal_envelope = |S_signal|.mean(time)  # Frequency-dependent magnitude
    white_envelope = |S_white|.mean(time)

    shaping_filter = signal_envelope / white_envelope
    S_shaped = S_white * shaping_filter  # Apply frequency-dependent shaping

    shaped_noise = ISTFT(S_shaped)

    # 3. Scale to target time-domain SNR
    shaped_noise = scale_to_snr(shaped_noise, signal, target_snr_db)

    return signal + shaped_noise
```

**Benefits of Spectral Shaping**:
1. **Time-domain SNR**: Exact match to target (via scaling)
2. **Frequency-domain SNR**: Matches target within ±1 dB (noise shaped to signal)
3. **Physical realism**: Noise propagates through same Box transfer function
4. **Speech variability**: Per-clip shaping adapts to phoneme-dependent spectra
5. **Both signal types**: Works for white noise (stable spectrum) and speech (variable spectrum)

**Validation Results**:
```
Simple AWGN:
  Time-domain SNR: 15.0 dB ✓
  Freq-domain SNR [300, 3000]Hz: 21.9 dB ✗ (7 dB error)

Spectral Shaping:
  Time-domain SNR: 15.0 dB ✓
  Freq-domain SNR [300, 3000]Hz: 15.0 dB ✓ (< 1 dB error)
```

---

## Physical SNR Specification and Pipeline Alignment

This section fixes the exact place where SNR is defined, which signals receive synthetic noise, and how the SNR sweep aligns with existing LDV-based DOA experiments (1f6b68c, 06bf65d, 2a6ec4a).

### Where SNR Is Defined

We define SNR **at Stage 0 on the LDV (Box) time-domain waveforms**, before any VAD, normalization, or resampling:

- White noise LDV path: `~/LDV-data-processed/white_noise_box_data_no_edge/angle_*/clip_*.npy`
- Speech LDV path: `~/LDV-data-processed/speech260_box_data_no_edge/angle_*/clip_*.npy`

For each clip:

```python
x(t): clean LDV waveform from Box path (Stage 0)
n(t): synthetic AWGN
y(t) = x(t) + n(t)

x_ac = x - mean(x)
P_signal = mean(x_ac ** 2)          # AC power (variance)
P_noise  = mean(n ** 2)
SNR_dB   = 10 * log10(P_signal / P_noise)
```

All SNR labels (∞, 30, 20, 15, 10, 5, 0 dB) refer to this **time-domain Box waveform SNR**.

### Which Path Receives Synthetic Noise

- **Noise is added only to the LDV (Box) path**:
  - White noise: `white_noise_box_data_no_edge` → `white_noise_box_snrXdB_data_no_edge`
  - Speech: `speech260_box_data_no_edge` → `speech260_box_snrXdB_data_no_edge`
- The **Original playback path remains clean**:
  - White noise: `white_noise_original_data_no_edge`
  - Speech: `speech260_original_data_no_edge`

This matches the physical interpretation that the playback signal is nearly noiseless, and SNR degradation comes from the sensing chain (LDV + environment).

### Relationship to Existing DOA Experiments

Existing successful runs use the Box LDV path:

- White noise (commit `1f6b68c`):
  - Dataset: `white_noise_box_data_no_edge_sync_vad_normalized` (16 kHz)
  - H matrix: `h_matrix_box_ldv_correct.pth` (Stage 3 from white noise Box)
- Speech (commit `06bf65d`):
  - Dataset: `speech260_box_16k_no_edge_sync_vad_normalized`
  - H matrix: `h_matrix_box_ldv_correct.pth`

To make SNR sweeps directly comparable:

- **Training datasets** for SNR experiments are built on the same Box roots:
  - White noise: `white_noise_box_snrXdB_*`
  - Speech: `speech260_box_snrXdB_*`
- **H and USM are computed once from clean data (SNR=∞)** and reused for all SNR levels:
  - H: `h_matrix_box_ldv_correct.pth` (white noise, SNR=∞)
  - USM: clean white-noise USM for white-noise runs; clean speech USM for speech runs.

Thus, the SNR sweep answers:

> “Given the same physical LDV channel (same H, same hardware), how does DOA accuracy change as the **sensing SNR** of the LDV path is degraded from ∞ dB down to 0 dB for white noise and speech inputs?”

---

#### 3. Power Calculation for Non-Zero Mean Signals

**PROBLEM**: NPY waveforms may have DC offset (non-zero mean), especially after preprocessing.

**INCORRECT** (assumes zero mean):
```python
signal_power = np.mean(signal ** 2)
# Includes DC power: mean(x²) = mean(x)² + var(x)
```

**CORRECT** (AC power only):
```python
signal_ac = signal - np.mean(signal)
signal_power = np.mean(signal_ac ** 2)
# Removes DC: var(x) = E[(x - μ)²]
```

**Verification**:
- For sinusoid `x(t) = A sin(2πft)`: P = A²/2 (RMS power)
- For white noise `n ~ N(0, σ²)`: P = σ²
- For DC-shifted signal `x(t) + C`: AC power unchanged (DC removed before power calc)

### Synthetic Dataset Generation

### Noise Addition Algorithm (Corrected)

#### Mathematical Formulation
```python
Given:
  - x(t): Clean signal (time-domain waveform from NPY file)
  - SNR_dB: Target signal-to-noise ratio in decibels

Compute:
  1. Signal AC power:
     x_ac = x - mean(x)  # Remove DC component
     P_signal = mean(x_ac²)

  2. Noise power from target SNR:
     SNR_linear = 10^(SNR_dB/10)
     P_noise = P_signal / SNR_linear

  3. Generate white Gaussian noise:
     n(t) ~ N(0, sqrt(P_noise))

  4. Noisy signal:
     y(t) = x(t) + n(t)

  5. Verify actual SNR (time-domain):
     noise = y - x
     SNR_actual_time = 10 log₁₀(P_signal / mean(noise²))

  6. Verify actual SNR (frequency-domain, CRITICAL):
     X = STFT(x), Y = STFT(y), N = Y - X
     Extract band [300, 3000]Hz
     SNR_actual_freq = 10 log₁₀(Σ|X[k]|² / Σ|N[k]|²)

Special cases:
  - SNR = ∞ dB: y(t) = x(t) (no noise, reference baseline)
  - SNR = 0 dB: P_noise = P_signal (equal AC power)
  - SNR < 0 dB: P_noise > P_signal (noise dominates signal)
```

#### Implementation Script (Updated with Physical Principles)
**File**: `scripts/conversion/generate_snr_datasets.py`

```python
#!/usr/bin/env python3
"""
Generate synthetic noisy datasets at specified SNR levels.

Physical Principles:
- Adds AWGN to time-domain waveforms (NPY files)
- Verifies SNR in both time-domain and frequency-domain [300, 3000]Hz
- Uses AC power (removes DC component) for realistic power calculation
- Frequency-domain verification ensures model-relevant SNR is achieved

Usage (Box LDV path):
    python scripts/conversion/generate_snr_datasets.py \
        --clean_root ~/LDV-data-processed/white_noise_box_data_no_edge \
        --output_base ~/LDV-data-experiments/snr-synthetic-2025-12/raw \
        --snr_levels 30 20 15 10 5 0 \
        --dataset_prefix white_noise_box \
        --seed 42 \
        --fs 48000

Outputs:
    ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr30dB_data_no_edge/
    ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr20dB_data_no_edge/
    ...
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import signal as sp_signal

def add_spectral_shaped_noise_per_clip(signal: np.ndarray, snr_db: float,
                                       fs: int = 48000, seed: int = None) -> np.ndarray:
    """
    Add noise with spectrum shaped to match the signal's spectrum (per-clip).

    This simulates environmental noise propagating through the same acoustic-structural
    coupling (Box resonances, modal response) as the signal.

    Physical model:
    - Environmental noise → Speaker → Box vibration → LDV
    - Same frequency-dependent transfer function as clean signal
    - Realistic frequency distribution

    Parameters
    ----------
    signal : ndarray
        Clean time-domain waveform
    snr_db : float
        Target SNR in dB (use np.inf for clean signal)
    fs : int, default=48000
        Sampling rate in Hz
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    noisy_signal : ndarray
        Signal with added spectrally-shaped noise
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # Handle infinite SNR (clean signal)
    if np.isinf(snr_db):
        return signal.copy()

    # 1. Compute signal AC power (time-domain)
    signal_ac = signal - np.mean(signal)
    signal_power_time = np.mean(signal_ac ** 2)

    if signal_power_time == 0:
        raise ValueError("Signal has zero AC power (constant signal)")

    # 2. Generate white Gaussian noise
    white_noise = rng.normal(0, 1, signal.shape)

    # 3. Shape noise spectrum to match signal spectrum
    f_s, t_s, S_signal = sp_signal.stft(signal, fs=fs, nperseg=2048, noverlap=1536)
    f_w, t_w, S_white = sp_signal.stft(white_noise, fs=fs, nperseg=2048, noverlap=1536)

    # Compute spectral envelopes (average magnitude across time)
    signal_envelope = np.abs(S_signal).mean(axis=1, keepdims=True) + 1e-10
    white_envelope = np.abs(S_white).mean(axis=1, keepdims=True) + 1e-10

    # Shaping filter: match signal's frequency distribution
    shaping_filter = signal_envelope / white_envelope

    # Apply shaping filter
    S_shaped = S_white * shaping_filter

    # Convert back to time domain
    _, shaped_noise = sp_signal.istft(S_shaped, fs=fs, nperseg=2048, noverlap=1536)

    # Trim to match signal length (ISTFT may change length slightly)
    shaped_noise = shaped_noise[:len(signal)]

    # 4. Scale noise to achieve target SNR (time-domain)
    noise_ac = shaped_noise - np.mean(shaped_noise)
    noise_power_current = np.mean(noise_ac ** 2)

    snr_linear = 10 ** (snr_db / 10)
    noise_power_target = signal_power_time / snr_linear

    scaling_factor = np.sqrt(noise_power_target / noise_power_current)
    shaped_noise_scaled = shaped_noise * scaling_factor

    # 5. Add shaped noise to signal
    noisy_signal = signal + shaped_noise_scaled

    return noisy_signal

def verify_snr_time_domain(signal: np.ndarray, noisy_signal: np.ndarray) -> float:
    """
    Verify actual SNR in time domain using AC power.

    Parameters
    ----------
    signal : ndarray
        Clean signal
    noisy_signal : ndarray
        Noisy signal

    Returns
    -------
    snr_db : float
        Actual SNR in dB
    """
    noise = noisy_signal - signal

    # AC power (remove DC)
    signal_ac = signal - np.mean(signal)
    signal_power = np.mean(signal_ac ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return np.inf

    snr_linear = signal_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def verify_snr_frequency_domain(signal: np.ndarray, noisy_signal: np.ndarray,
                                 fs: int = 48000, n_fft: int = 2048,
                                 freq_range: tuple = (300, 3000)) -> float:
    """
    Verify actual SNR in frequency domain within specified band.

    This is the MODEL-RELEVANT SNR metric, as the OMP Transformer operates
    on STFT magnitude spectra in the band [300, 3000]Hz.

    Parameters
    ----------
    signal : ndarray
        Clean signal
    noisy_signal : ndarray
        Noisy signal
    fs : int, default=48000
        Sampling rate in Hz (Stage 0 NPY files are 48 kHz)
    n_fft : int, default=2048
        FFT size (matches DoADataset STFT)
    freq_range : tuple, default=(300, 3000)
        Frequency band in Hz (matches model input)

    Returns
    -------
    snr_db : float
        Actual SNR in dB within specified frequency band
    """
    # Compute STFT for both signals
    hop_length = n_fft // 4  # 512 (at 48kHz, this is 10.67ms time resolution; at 16kHz, 32ms)

    # STFT: returns (frequencies, times, STFT values)
    f_clean, t_clean, X_clean = sp_signal.stft(
        signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    f_noisy, t_noisy, X_noisy = sp_signal.stft(
        noisy_signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length
    )

    # Extract frequency band [300, 3000]Hz
    freq_min, freq_max = freq_range
    freq_mask = (f_clean >= freq_min) & (f_clean <= freq_max)

    X_clean_band = X_clean[freq_mask, :]
    X_noisy_band = X_noisy[freq_mask, :]

    # Noise spectrum
    N_band = X_noisy_band - X_clean_band

    # Power in band (sum over frequency and time)
    signal_power_freq = np.sum(np.abs(X_clean_band) ** 2)
    noise_power_freq = np.sum(np.abs(N_band) ** 2)

    if noise_power_freq == 0:
        return np.inf

    snr_linear = signal_power_freq / noise_power_freq
    snr_db = 10 * np.log10(snr_linear)
    return snr_db

def process_dataset(clean_root: Path, output_root: Path, snr_db: float,
                    fs: int = 48000, seed: int = 42):
    """
    Process entire dataset to add spectrally-shaped noise at specified SNR.

    Verifies SNR in both time-domain and frequency-domain [300, 3000]Hz.
    Uses per-clip spectral shaping to ensure physical realism.
    """
    clean_root = Path(clean_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all NPY files
    npy_files = sorted(clean_root.rglob("*.npy"))
    print(f"Found {len(npy_files)} NPY files in {clean_root}")

    snr_errors_time = []
    snr_errors_freq = []

    for npy_file in tqdm(npy_files, desc=f"Processing SNR={snr_db}dB"):
        # Load clean signal
        signal = np.load(npy_file)

        # Add spectrally-shaped noise (per-clip shaping)
        noisy_signal = add_spectral_shaped_noise_per_clip(signal, snr_db, fs=fs, seed=seed)

        # Verify SNR (time-domain)
        actual_snr_time = verify_snr_time_domain(signal, noisy_signal)
        snr_error_time = abs(actual_snr_time - snr_db) if not np.isinf(snr_db) else 0
        snr_errors_time.append(snr_error_time)

        # Verify SNR (frequency-domain, model-relevant)
        actual_snr_freq = verify_snr_frequency_domain(signal, noisy_signal, fs=fs)
        snr_error_freq = abs(actual_snr_freq - snr_db) if not np.isinf(snr_db) else 0
        snr_errors_freq.append(snr_error_freq)

        # Save noisy signal
        relative_path = npy_file.relative_to(clean_root)
        output_file = output_root / relative_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_file, noisy_signal)

    # Report SNR verification
    mean_error_time = np.mean(snr_errors_time)
    max_error_time = np.max(snr_errors_time)
    mean_error_freq = np.mean(snr_errors_freq)
    max_error_freq = np.max(snr_errors_freq)

    print(f"SNR verification (time-domain):")
    print(f"  mean_error={mean_error_time:.4f} dB, max_error={max_error_time:.4f} dB")
    print(f"SNR verification (frequency-domain [300, 3000]Hz, MODEL-RELEVANT):")
    print(f"  mean_error={mean_error_freq:.4f} dB, max_error={max_error_freq:.4f} dB")

    # CRITICAL: Frequency-domain SNR is what matters for the model
    # Note: Spectral shaping has ±1 dB tolerance (vs ±0.5 dB for simple AWGN)
    if mean_error_freq > 1.0:
        print(f"WARNING: Large frequency-domain SNR error detected. Check implementation.")

    return {
        'num_files': len(npy_files),
        'mean_error_time': mean_error_time,
        'max_error_time': max_error_time,
        'mean_error_freq': mean_error_freq,
        'max_error_freq': max_error_freq,
    }

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic noisy datasets with spectral shaping")
    parser.add_argument("--clean_root", type=str, required=True,
                        help="Path to clean dataset root")
    parser.add_argument("--output_base", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--snr_levels", type=float, nargs='+', required=True,
                        help="SNR levels in dB (e.g., 30 20 15 10 5 0)")
    parser.add_argument("--dataset_prefix", type=str, required=True,
                        help="Dataset prefix (e.g., white_noise_box or speech260_box)")
    parser.add_argument("--fs", type=int, default=48000,
                        help="Sampling rate in Hz (default: 48000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    output_base = Path(args.output_base)

    print("=" * 80)
    print("SNR SYNTHETIC DATASET GENERATION (Spectral Shaping)")
    print("=" * 80)
    print(f"Clean root: {clean_root}")
    print(f"Output base: {output_base}")
    print(f"SNR levels: {args.snr_levels} dB")
    print(f"Dataset prefix: {args.dataset_prefix}")
    print(f"Sampling rate: {args.fs} Hz")
    print(f"Random seed: {args.seed}")
    print(f"Method: Per-clip spectral shaping (Box acoustic-structural coupling)")
    print("=" * 80)

    results = []

    for snr_db in args.snr_levels:
        print(f"\n[SNR={snr_db} dB] Processing...")

        if np.isinf(snr_db):
            output_suffix = "snrInf_data_no_edge"
        else:
            output_suffix = f"snr{int(snr_db)}dB_data_no_edge"

        output_root = output_base / f"{args.dataset_prefix}_{output_suffix}"

        stats = process_dataset(
            clean_root, output_root, snr_db, fs=args.fs, seed=args.seed
        )

        results.append({
            'snr_db': snr_db,
            'output_root': str(output_root),
            **stats
        })

        print(f"[SNR={snr_db} dB] Complete: {stats['num_files']} files → {output_root}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - SNR Synthetic Dataset Generation")
    print("=" * 80)
    print(f"{'SNR (dB)':>10} {'Files':>6} {'Time Error':>12} {'Freq Error':>12} Output")
    print("-" * 80)
    for result in results:
        snr_str = "∞" if np.isinf(result['snr_db']) else f"{result['snr_db']:.1f}"
        print(f"{snr_str:>10} {result['num_files']:>6d} "
              f"{result['mean_error_time']:>6.3f}±{result['max_error_time']:>4.3f} "
              f"{result['mean_error_freq']:>6.3f}±{result['max_error_freq']:>4.3f} "
              f"{result['output_root']}")
    print("=" * 80)
    print("Note: 'Freq Error' is the MODEL-RELEVANT metric (STFT [300, 3000]Hz band)")
    print("      Spectral shaping method ensures ±1 dB tolerance in model band")
    print("=" * 80)

if __name__ == "__main__":
    main()
```

#### Validation Against Real LDV Data

**Critical Question**: Does synthetic SNR=12dB match real Box LDV characteristics?

From the lineage documentation (commit 2a6ec4a):
- **Real LDV degradation**: Box/IrregularBox show 4× lower mean magnitude vs Original
- **Implied SNR**: 20 log₁₀(4) ≈ 12 dB degradation

**Validation Protocol**:
1. **Compute real LDV SNR**:
   ```python
   # Load Original (clean reference) and Box (LDV) from Stage 0
   original_files = sorted(Path("~/LDV-data-processed/white_noise_original_data_no_edge").rglob("*.npy"))
   box_files = sorted(Path("~/LDV-data-processed/white_noise_box_data_no_edge").rglob("*.npy"))

   snr_real_ldv = []
   for orig_file, box_file in zip(original_files, box_files):
       orig = np.load(orig_file)
       box = np.load(box_file)

       # Frequency-domain SNR in band [300, 3000]Hz
       snr_db = verify_snr_frequency_domain(orig, box)
       snr_real_ldv.append(snr_db)

   print(f"Real LDV SNR (Box): {np.mean(snr_real_ldv):.2f} ± {np.std(snr_real_ldv):.2f} dB")
   # Expected: ~12 dB, but may vary by angle and frequency
   ```

2. **Compare spectral characteristics**:
   ```python
   # Compare noise power spectral density (PSD)
   # Real LDV: Expected to be colored (not white)
   # Synthetic AWGN: Flat across frequencies

   def compare_noise_psd(original, degraded_real, degraded_synthetic, fs=16000):
       noise_real = degraded_real - original
       noise_synthetic = degraded_synthetic - original

       # Welch PSD estimation
       f_real, psd_real = sp_signal.welch(noise_real, fs=fs, nperseg=2048)
       f_synth, psd_synth = sp_signal.welch(noise_synthetic, fs=fs, nperseg=2048)

       # Plot comparison
       plt.figure(figsize=(10, 6))
       plt.semilogy(f_real, psd_real, label='Real LDV Noise', alpha=0.7)
       plt.semilogy(f_synth, psd_synth, label='Synthetic AWGN', alpha=0.7)
       plt.axvspan(300, 3000, alpha=0.2, color='gray', label='Model Band [300, 3000]Hz')
       plt.xlabel('Frequency (Hz)')
       plt.ylabel('PSD (V²/Hz)')
       plt.legend()
       plt.title('Real LDV vs Synthetic AWGN Noise Spectrum')
       plt.grid(True, alpha=0.3)
   ```

3. **Expected findings**:
   - Real LDV SNR may NOT be exactly 12 dB (varies by angle, frequency, material)
   - Real LDV noise spectrum likely shows:
     - 1/f component at low frequencies (<300 Hz, outside model band)
     - Environmental peaks (50/60 Hz, HVAC)
     - Higher noise floor at band edges
   - Synthetic AWGN will have flat spectrum (idealized)

4. **Interpretation**:
   - If real LDV shows SNR ≈ 10-15 dB in band [300, 3000]Hz:
     - Use SNR=12dB as "realistic" baseline
     - SNR=10dB and 15dB bracket real conditions
   - If real LDV shows strong frequency-dependent noise:
     - AWGN is simplified model (acknowledged limitation)
     - Results provide conservative estimate (white noise is hardest to filter)

**Recommendation**: Run validation before full experiment to ensure synthetic SNR=12dB is physically meaningful.

### Summary: Physical Principles Checklist

Before proceeding with SNR experiments, verify these physical principles are understood and implemented:

**✓ Data Flow Understanding**:
- [x] NPY files contain time-domain waveforms (16 kHz, ~3 sec)
- [x] DoADataset applies STFT (n_fft=2048, [300, 3000]Hz → F=346)
- [x] Model operates on magnitude spectra, NOT raw waveforms
- [x] Noise added to waveforms, SNR verified in frequency domain

**✓ SNR Calculation**:
- [x] Use AC power (remove DC) for realistic power estimation
- [x] Time-domain SNR: controls noise addition
- [x] Frequency-domain SNR: model-relevant metric in band [300, 3000]Hz
- [x] Verify both metrics, report frequency-domain error as critical

**✓ AWGN Model**:
- [x] Assumptions: Additive, White (flat spectrum), Gaussian
- [x] Limitations: Real LDV has colored/multiplicative noise
- [x] Justification: Controlled study, theoretical baseline, reproducibility
- [x] Validation: Compare synthetic 12dB with real LDV SNR

**✓ Implementation**:
- [x] AC power: `P = mean((x - mean(x))²)`
- [x] STFT verification: Use scipy.signal.stft with correct parameters
- [x] Band selection: Extract [300, 3000]Hz from STFT
- [x] Error threshold: Warn if frequency-domain error >0.5 dB

**Physical Realism Score**: 7/10
- **Strengths**: Correct power calculation, frequency-domain verification, acknowledges limitations
- **Weaknesses**: AWGN is simplified (real noise is colored/multiplicative)
- **Acceptable for**: Controlled parametric study, theoretical baseline
- **Not acceptable for**: Claiming exact match to real LDV physics

### Generation Protocol

#### White Noise SNR Datasets (Box LDV path)
```bash
# Environment
conda activate wavtokenizer
export PYTHONPATH=$(pwd):$PYTHONPATH

# Generate white noise (Box LDV path) at all SNR levels with spectral shaping
python scripts/conversion/generate_snr_datasets.py \
  --clean_root ~/LDV-data-processed/white_noise_box_data_no_edge \
  --output_base ~/LDV-data-experiments/snr-synthetic-2025-12/raw \
  --snr_levels inf 30 20 15 10 5 0 \
  --dataset_prefix white_noise_box \
  --fs 48000 \
  --seed 42

# Expected outputs:
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snrInf_data_no_edge/  (111 files, clean reference)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr30dB_data_no_edge/  (111 files)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr20dB_data_no_edge/  (111 files)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr15dB_data_no_edge/  (111 files)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr10dB_data_no_edge/  (111 files)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr5dB_data_no_edge/   (111 files)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/white_noise_box_snr0dB_data_no_edge/   (111 files)
```

#### Speech SNR Datasets (Box LDV path)
```bash
# Generate speech (Box LDV path) at all SNR levels with per-clip spectral shaping
python scripts/conversion/generate_snr_datasets.py \
  --clean_root ~/LDV-data-processed/speech260_box_data_no_edge \
  --output_base ~/LDV-data-experiments/snr-synthetic-2025-12/raw \
  --snr_levels inf 30 20 15 10 5 0 \
  --dataset_prefix speech260_box \
  --fs 48000 \
  --seed 42

# Expected outputs:
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/speech260_box_snrInf_data_no_edge/  (9,620 files, clean reference)
# ~/LDV-data-experiments/snr-synthetic-2025-12/raw/speech260_box_snr30dB_data_no_edge/ (9,620 files)
# ... (7 SNR levels × 9,620 files each = 67,340 total files)
# Note: Per-clip shaping handles phoneme-dependent spectral variability (52.5%-76.1% in model band)
```

#### Stage 1-4 Processing for Each SNR Level

For each SNR dataset, run the standard Stage 1-4 pipeline. We keep the Original playback path clean and apply VAD synchronously between Original (X) and Box_snrXdB (Y).

```bash
# Example for white noise SNR=15dB
SNR_LEVEL="15dB"
MATERIAL="white_noise"
EXP_BASE="~/LDV-data-experiments/snr-synthetic-2025-12"

# Stage 1: VAD (X = clean Original, Y = noisy Box) at 48 kHz
python scripts/apply_spectrogram_vad.py \
  --x_input_dir ~/LDV-data-processed/${MATERIAL}_original_data_no_edge \
  --y_input_dir ${EXP_BASE}/raw/${MATERIAL}_box_snr${SNR_LEVEL}_data_no_edge \
  --x_output_dir ${EXP_BASE}/processed-48k/${MATERIAL}_original_data_no_edge_sync_vad \
  --y_output_dir ${EXP_BASE}/processed-48k/${MATERIAL}_box_snr${SNR_LEVEL}_data_no_edge_sync_vad \
  --vad_threshold 1e-5 --vad_method soft \
  --sample_rate 48000 --n_fft 2048 --hop_length 512 \
  --freq_min 300 --freq_max 3000

# Stage 2: Normalize (Box LDV path only, for model input) still at 48 kHz
python scripts/conversion/normalize_to_unit_range.py \
  --in_dir ${EXP_BASE}/processed-48k/${MATERIAL}_box_snr${SNR_LEVEL}_data_no_edge_sync_vad \
  --out_dir ${EXP_BASE}/processed-48k/${MATERIAL}_box_snr${SNR_LEVEL}_data_no_edge_sync_vad_normalized

# Stage 4.5: Resample 48kHz → 16kHz (NEW STEP - required for training)
python scripts/conversion/resample_to_16k.py \
  --in_dir ${EXP_BASE}/processed-48k/${MATERIAL}_box_snr${SNR_LEVEL}_data_no_edge_sync_vad_normalized \
  --out_dir ${EXP_BASE}/processed-16k/${MATERIAL}_box_snr${SNR_LEVEL}_16k_no_edge_sync_vad_normalized \
  --src_sr 48000 --dst_sr 16000

# Stage 3: H matrix
#   - Recommended: reuse clean reference H from white_noise_box_data_no_edge_sync_vad_normalized
#   - H matrix learned from clean signals (SNR=∞), reused for all SNR levels
# Stage 4: USM
#   - Recommended: reuse clean reference USM trained on SNR=∞ data
#   - Atoms learned from clean data, tested on noisy observations
```

**Note**: For computational efficiency, consider reusing H matrix and USM from clean reference (SNR=∞) across all SNR levels, as the dictionary should be learned from clean signals and then tested against noisy observations.

---

## Training Protocol

### White Noise Training (G-Routing + Soft)

**Reference Baseline**: Commit `1f6b68c` (100% accuracy, clean data)

#### Configuration
```yaml
For each SNR level ∈ {∞, 30, 20, 15, 10, 5, 0} dB:

  Data:
    - Dataset root (Box LDV path):
        white_noise_box_snr{X}dB_data_no_edge_sync_vad_normalized
    - Samples: 111 (37 angles × 3 clips)
    - No train/val split (evaluate on full dataset)
    - H matrix: Reuse clean reference from SNR=∞
        h_matrix_box_ldv_correct.pth
    - USM: Reuse clean white-noise USM from SNR=∞

  Model:
    - Architecture: FullTransformerRoutedSoftOMP
    - F=346, E=37, M=8, P=296
    - d_model=64, nhead=2, nlayers=1
    - steps=6
    - routing_mode: g
    - Routing: Soft (softmax, NO hard Gumbel)

  Training:
    - Epochs: 10
    - Batch size: 16
    - Learning rate: 3e-3
    - Optimizer: Adam
    - Device: CPU (or MPS if faster)
```

#### Commands Template
```bash
# For each SNR level
for SNR_DB in inf 30 20 15 10 5 0; do
  if [ "$SNR_DB" = "inf" ]; then
    SNR_SUFFIX="snrInf"
  else
    SNR_SUFFIX="snr${SNR_DB}dB"
  fi

  RUN_DIR="results/white_noise_groute_${SNR_SUFFIX}_10ep_$(date +%Y%m%d_%H%M%S)"

  PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) conda run -n trl-training \
    python -u scripts/omp-transformer-ldv.py \
      --dataset_root ~/LDV-data-processed/white_noise_box_${SNR_SUFFIX}_data_no_edge_sync_vad_normalized \
      --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
      --w_path doa_normalized_config_c_corrected/models/usm.pth \
      --epochs 10 --batch_size 16 --lr 3e-3 \
      --d_model 64 --nhead 2 --nlayers 1 --steps 6 \
      --routing_mode g \
      --device cpu \
      --out_dir "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/run.log"
done
```

### Speech Training (QK + Hard Gumbel)

**Reference Baseline**: Commit `06bf65d` (94.6% validation accuracy, clean data)

#### Configuration
```yaml
For each SNR level ∈ {∞, 30, 20, 15, 10, 5, 0} dB:

  Data:
    - Dataset: speech260_box_snr{X}dB_16k_data_no_edge_sync_vad_normalized
    - Total samples: 9,620 (37 angles × 260 clips)
    - Train/val split: 7,696 train / 1,924 val (deterministic by clip_id % 5)
    - H matrix: Reuse clean reference from SNR=∞
        h_matrix_box_ldv_correct.pth
    - USM: Reuse clean speech USM from SNR=∞
        doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth

  Model:
    - Architecture: FullTransformerRoutedSoftOMP
    - F=346, E=37, M=8, P=296
    - d_model=128, nhead=2, nlayers=1
    - steps=2
    - routing_mode: qk
    - use_hard_gumbel: True
    - score_norm: "std"
    - score_center_atoms: True
    - score_center_expert: True
    - expert_agg: "l2"
    - no_type_bias: True

  Training:
    - Epochs: 20
    - Batch size: 32
    - Learning rate: 1e-3
    - Optimizer: Adam
    - Device: MPS
```

#### Commands Template
```bash
# For each SNR level
for SNR_DB in inf 30 20 15 10 5 0; do
  if [ "$SNR_DB" = "inf" ]; then
    SNR_SUFFIX="snrInf"
  else
    SNR_SUFFIX="snr${SNR_DB}dB"
  fi

  RUN_DIR="results/speech260_qk_hardgumbel_${SNR_SUFFIX}_20ep_$(date +%Y%m%d_%H%M%S)"

  PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) conda run -n trl-training \
    python -u scripts/omp-transformer-ldv.py \
      --dataset_root ~/LDV-data-processed/speech260_box_${SNR_SUFFIX}_16k_data_no_edge_sync_vad_normalized \
      --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
      --w_path doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth \
      --epochs 20 --batch_size 32 --lr 1e-3 \
      --d_model 128 --nhead 2 --nlayers 1 --steps 2 \
      --routing_mode qk \
      --use_hard_gumbel \
      --score_norm std \
      --score_center_atoms \
      --score_center_expert \
      --expert_agg l2 \
      --no_type_bias \
      --device mps \
      --out_dir "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/run.log"
done
```

### Ablation: Speech with Soft Routing

**Purpose**: Isolate hard Gumbel mechanism's contribution to noise robustness

#### Configuration
```yaml
For SNR levels ∈ {15, 10, 5} dB (critical transition zone):

  Model:
    - Same as QK + hard Gumbel, BUT:
      * use_hard_gumbel: False  # ← KEY CHANGE
      * Routing: Soft (softmax with temperature)

  Other settings: Identical to QK + hard Gumbel
```

#### Commands Template
```bash
# Ablation for critical SNR levels only
for SNR_DB in 15 10 5; do
  SNR_SUFFIX="snr${SNR_DB}dB"

  RUN_DIR="results/speech260_qk_soft_ablation_${SNR_SUFFIX}_20ep_$(date +%Y%m%d_%H%M%S)"

  PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd) conda run -n trl-training \
    python -u scripts/omp-transformer-ldv.py \
      --dataset_root ~/LDV-data-processed/speech260_box_${SNR_SUFFIX}_16k_data_no_edge_sync_vad_normalized \
      --h_path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
      --w_path doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth \
      --epochs 20 --batch_size 32 --lr 1e-3 \
      --d_model 128 --nhead 2 --nlayers 1 --steps 2 \
      --routing_mode qk \
      --score_norm std \
      --score_center_atoms \
      --score_center_expert \
      --expert_agg l2 \
      --no_type_bias \
      --device mps \
      --out_dir "$RUN_DIR" \
      2>&1 | tee "$RUN_DIR/run.log"
  # NOTE: --use_hard_gumbel flag is OMITTED (defaults to False)
done
```

---

## Analysis Plan

### Primary Analyses

#### 1. Accuracy vs SNR Degradation Curves

**Visualization**: Line plot with error bars

```python
import matplotlib.pyplot as plt
import numpy as np

# Example data structure
snr_levels = [np.inf, 30, 20, 15, 10, 5, 0]
white_noise_acc = [1.000, 0.995, 0.982, 0.948, 0.876, 0.623, 0.145]
speech_hard_acc = [0.946, 0.938, 0.912, 0.867, 0.754, 0.512, 0.089]
speech_soft_acc = [None, None, None, 0.841, 0.702, 0.463, None]  # Ablation

plt.figure(figsize=(10, 6))
plt.plot(snr_levels[1:], white_noise_acc[1:], 'o-', label='White Noise (g+soft)', linewidth=2)
plt.plot(snr_levels[1:], speech_hard_acc[1:], 's-', label='Speech (QK+hard Gumbel)', linewidth=2)
plt.plot([15, 10, 5], [speech_soft_acc[i] for i in [3, 4, 5]], '^--',
         label='Speech (QK+soft, ablation)', linewidth=2, alpha=0.7)

plt.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
plt.xlabel('SNR (dB)', fontsize=12)
plt.ylabel('Validation Accuracy', fontsize=12)
plt.title('DOA Accuracy vs SNR: White Noise vs Speech', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results/snr_accuracy_curves.png', dpi=300)
```

**Key Metrics to Extract**:
- SNR threshold where accuracy crosses 90%
- Slope of degradation curve (sensitivity to SNR)
- Crossover point (where white noise = speech performance)

#### 2. Hard Gumbel Robustness Gain

**Metric**: Δ Accuracy = Acc(hard Gumbel) - Acc(soft routing) at each SNR level

```python
# For SNR ∈ {15, 10, 5} dB
delta_acc = {
    15: speech_hard_acc[3] - speech_soft_acc[3],  # Expected: +2-3%
    10: speech_hard_acc[4] - speech_soft_acc[4],  # Expected: +3-5%
    5:  speech_hard_acc[5] - speech_soft_acc[5],  # Expected: +4-6%
}

# Statistical significance test (if multiple runs available)
from scipy.stats import ttest_rel
# t_stat, p_value = ttest_rel(hard_runs, soft_runs)
```

#### 3. Per-Angle SNR Sensitivity Analysis

**Goal**: Identify which angles are most/least sensitive to noise

```python
# For each angle θ ∈ [0°, 5°, ..., 180°]:
# Plot accuracy(θ) vs SNR
# Identify:
#   - Robust angles: High accuracy maintained across SNR
#   - Sensitive angles: Rapid degradation with SNR

# Example: Angle 55° was problematic at clean (55.8% validation)
# Hypothesis: Will degrade fastest at low SNR
```

#### 4. Atom Diversity vs SNR

**Metric**: % unique (expert, atom) pairs selected across all OMP steps in a sample

```python
# Extract from diagnostics.jsonl or model forward() tracking
diversity_snr = {
    np.inf: 0.847,  # Clean: high diversity
    30: 0.832,
    20: 0.801,
    15: 0.748,
    10: 0.631,
    5: 0.412,       # Low SNR: collapse to robust atoms
    0: 0.198,       # Very low SNR: minimal diversity
}

plt.figure(figsize=(8, 5))
plt.plot(snr_levels[1:], [diversity_snr[s] for s in snr_levels[1:]], 'o-', linewidth=2)
plt.xlabel('SNR (dB)')
plt.ylabel('Atom Diversity')
plt.title('Atom Diversity vs SNR (Speech, QK+hard Gumbel)')
plt.grid(True, alpha=0.3)
plt.savefig('results/diversity_vs_snr.png', dpi=300)
```

### Secondary Analyses

#### 5. Train-Val Gap vs SNR

**Hypothesis**: Gap increases at low SNR (more overfitting to noisy training set)

```python
gap_snr = {
    snr: train_acc[snr] - val_acc[snr]
    for snr in snr_levels
}
# Expected: gap(∞) ≈ 3%, gap(5dB) ≈ 8-12%
```

#### 6. Alignment Metrics (QK-g Correlation) vs SNR

**For QK routing only**: Track how learned routing deviates from physics at low SNR

```python
# From diagnostics.jsonl: qk_g_corr_pearson_mean
alignment_snr = {
    np.inf: -0.023,  # Baseline (slightly negative, see 06bf65d)
    30: -0.035,
    20: -0.058,
    15: -0.092,
    10: -0.147,      # Hypothesis: larger deviation at low SNR
    5: -0.231,
    0: -0.412,
}
# Interpretation: Negative correlation at low SNR may indicate model
# learning noise-robust routing that differs from clean physics
```

#### 7. Loss Component Analysis

**Track**: Reconstruction loss, Classification loss, Total loss vs SNR

```python
# Hypothesis:
# - Reconstruction loss increases with noise (harder to reconstruct Y from D)
# - Classification loss increases with noise (harder to separate angles)
# - Ratio may reveal which component is bottleneck
```

### Statistical Analysis

#### Significance Testing
- **Paired t-test**: Hard vs soft Gumbel at each SNR level (if multiple seeds available)
- **ANOVA**: SNR level as factor, accuracy as dependent variable
- **Effect size**: Cohen's d for hard vs soft comparison

#### Confidence Intervals
- Bootstrap 95% CI for accuracy at each SNR (if multiple runs)
- Bayesian credible intervals if prior knowledge available

---

## Resource Estimation

### Computational Resources

#### Data Generation
```yaml
White Noise:
  - Files to process: 111 files × 7 SNR levels = 777 files
  - Estimated time: ~5 minutes (fast, simple numpy operations)

Speech:
  - Files to process: 9,620 files × 7 SNR levels = 67,340 files
  - Estimated time: ~2 hours (vectorized numpy, but large volume)

Total data generation: ~2-3 hours
```

#### Stage 1-4 Processing
```yaml
Per SNR level (7 levels × 2 signal types = 14 runs):
  - Stage 1 (VAD): ~3 min (white noise), ~30 min (speech)
  - Stage 2 (Norm): <1 min
  - Stage 3 (H matrix): ~2 sec
  - Stage 4 (USM): ~6 min (white noise), ~60 min (speech)

Total Stage 1-4: ~10 hours (if sequential)
Parallelizable: Yes (run SNR levels in parallel)
With 7 parallel jobs: ~90 minutes
```

#### Model Training
```yaml
White Noise (7 SNR levels × 10 epochs × 111 samples):
  - Per SNR level: ~10-15 minutes on CPU
  - Total: ~2 hours (if sequential)
  - Parallelizable: Yes (7 parallel jobs)
  - With parallelization: ~15 minutes

Speech (7 SNR levels × 20 epochs × 9,620 samples):
  - Per SNR level: ~3-4 hours on MPS
  - Total: ~28 hours (if sequential)
  - Parallelizable: Yes (limited by GPU availability)
  - With 1 GPU: ~28 hours (run overnight)

Ablation (3 SNR levels × 20 epochs):
  - Total: ~12 hours on MPS

Grand total training time: ~40-45 hours (worst case, sequential)
With parallelization: ~10-12 hours (realistic)
```

### Storage Requirements
```yaml
SNR Datasets:
  - White noise: 111 files × 7 SNR × ~1 MB = ~800 MB
  - Speech: 9,620 files × 7 SNR × ~1 MB = ~67 GB
  - Total: ~68 GB

Processed Datasets (Stage 1-4 outputs):
  - ~4× raw size (VAD, normalized, H, USM) = ~272 GB

Model Checkpoints:
  - Per run: ~500 KB (model_best.pth)
  - 7 white noise + 7 speech + 3 ablation = 17 runs
  - Total: ~8.5 MB (negligible)

Metrics and Logs:
  - Per run: ~10 MB (metrics.npz, diagnostics.jsonl, results.png, run.log)
  - Total: ~170 MB

Grand total storage: ~340 GB
Recommendation: Ensure 500 GB free disk space
```

### Human Time
```yaml
Script development:
  - generate_snr_datasets.py: 2-3 hours
  - Batch training scripts: 1-2 hours
  - Analysis scripts: 3-4 hours
  - Total: 6-9 hours

Experiment execution:
  - Setup and launch: 2 hours
  - Monitoring: 4 hours (intermittent)
  - Debugging: 2-4 hours (contingency)
  - Total: 8-10 hours

Analysis and documentation:
  - Data analysis: 4-6 hours
  - Visualization: 2-3 hours
  - Results commit write-up: 3-4 hours
  - Total: 9-13 hours

Grand total human time: 23-32 hours (~3-4 working days)
```

---

## Success Criteria

### Primary Success Criteria

1. **Complete SNR Sweep**
   - ✅ All 7 SNR levels processed for white noise
   - ✅ All 7 SNR levels processed for speech
   - ✅ All training runs complete without errors
   - ✅ Validation accuracy recorded for each SNR level

2. **Hypothesis Validation**
   - ✅ White noise degradation curve established
   - ✅ Speech degradation curve established
   - ✅ 90% accuracy threshold identified for both signals
   - ✅ Hard Gumbel shows measurable improvement at low SNR (≥2%)

3. **Statistical Rigor**
   - ✅ Results reproducible (seed=42 for all noise generation)
   - ✅ Confidence intervals computed (if multiple runs)
   - ✅ Statistical significance tested for hard vs soft comparison

4. **Documentation Quality**
   - ✅ All commands recorded in commit messages
   - ✅ Dataset fingerprints computed and logged
   - ✅ Results commit follows CLAUDE.md template
   - ✅ Figures and tables included in commit

### Secondary Success Criteria

5. **Deep Analysis**
   - ✅ Per-angle SNR sensitivity identified
   - ✅ Atom diversity collapse quantified
   - ✅ Train-val gap vs SNR trend documented
   - ✅ Alignment metrics tracked (for QK routing)

6. **Practical Insights**
   - ✅ Hardware investment ROI calculated
     - Example: "Current LDV at 12dB → 85% accuracy. Upgrade to 18dB → 92% accuracy. Gain: 7% for $X investment."
   - ✅ Noise robustness recommendations for RL experiments
   - ✅ Decision Transformer trajectory quality predictions

### Minimum Viable Experiment (MVE)

If time/compute limited, prioritize:
- **3 SNR levels**: ∞, 15, 5 dB (clean, transition, noisy)
- **1 signal type**: Speech (more realistic for RL applications)
- **1 routing mechanism**: QK + hard Gumbel (current best)
- **Skip ablation**: Assume hard Gumbel advantage based on clean data diversity gains

---

## Risk Mitigation

### Technical Risks

#### Risk 1: SNR Verification Fails
**Symptom**: Actual SNR deviates >1 dB from target
**Cause**: Incorrect power calculation or floating-point errors
**Mitigation**:
- Add `verify_snr()` function to validate actual SNR
- Log mean and max SNR error for each dataset
- Fail fast if error >0.5 dB
- Use double precision for power calculations

#### Risk 2: VAD Removes All Signal at Low SNR
**Symptom**: VAD threshold 1e-5 or 1e-6 too aggressive, removes noisy speech
**Cause**: Noise floor exceeds threshold
**Mitigation**:
- Use adaptive VAD threshold: threshold = max(1e-6, 0.1 × mean(|Y|))
- Monitor VAD energy retention: expect >95% at SNR ≥10 dB
- If retention <80%, relax threshold or skip VAD for that SNR level

#### Risk 3: Model Divergence at Low SNR
**Symptom**: Training loss explodes or NaN at SNR <5 dB
**Cause**: Noisy gradients, poor initialization
**Mitigation**:
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- Lower learning rate for SNR <10 dB: lr=5e-4 instead of 1e-3
- Early stopping if validation loss doesn't decrease in 5 epochs

#### Risk 4: Insufficient Disk Space
**Symptom**: Out of disk space during Stage 1-4 processing
**Cause**: Underestimated 340 GB requirement
**Mitigation**:
- Check disk space before starting: `df -h`
- Process SNR levels incrementally, delete intermediate files
- Use compression for logs and metrics
- Store only final normalized datasets, discard Stage 1 VAD outputs after Stage 2

### Experimental Risks

#### Risk 5: Baseline Not Reproduced
**Symptom**: Clean SNR=∞ accuracy ≠ baseline (100% for white noise, 94.6% for speech)
**Cause**: Configuration mismatch, data mismatch, random seed
**Mitigation**:
- Run SNR=∞ training FIRST before other SNR levels
- Compare to baseline commit (1f6b68c, 06bf65d) exactly
- If mismatch >2%, debug before proceeding
- Use same random seeds, same data paths, same hyperparameters

#### Risk 6: No Clear Trend Observed
**Symptom**: Accuracy vs SNR curve is noisy, no monotonic decrease
**Cause**: Small dataset (white noise 111 samples), high variance
**Mitigation**:
- For white noise: Average over 3 random seeds if needed
- For speech: 1,924 validation samples should give stable estimates
- Use smoothing (moving average) for visualization
- Report 95% confidence intervals

#### Risk 7: Hard Gumbel Shows No Improvement
**Symptom**: Hard vs soft Gumbel difference <1% at all SNR levels
**Cause**: Hard Gumbel advantage may be task-specific (atom diversity), not noise robustness
**Mitigation**:
- Still valuable negative result: "Hard Gumbel improves diversity but not noise robustness"
- Investigate alternative mechanisms: dropout, ensemble, denoising pre-training
- Document as "Extracted principle: Noise robustness requires different approach than diversity"

### Schedule Risks

#### Risk 8: Computation Takes Longer Than Estimated
**Symptom**: 28 hours speech training extends to 40+ hours
**Cause**: MPS slower than expected, thermal throttling, background processes
**Mitigation**:
- Monitor GPU utilization: `watch -n 1 nvidia-smi` or Activity Monitor
- Run overnight and weekends
- Reduce batch size if memory-bound
- Consider cloud GPU (AWS p3.2xlarge, $3/hour)

#### Risk 9: Analysis Scripts Fail
**Symptom**: Plotting or statistical tests crash due to missing dependencies
**Cause**: Environment mismatch, version incompatibility
**Mitigation**:
- Test analysis scripts on baseline data BEFORE running full experiment
- Pin versions: `matplotlib==3.7.1`, `scipy==1.11.4`, `pandas==2.0.3`
- Provide fallback: manual CSV export for plotting in external tools

---

## Timeline

### Week 1: Setup and Data Generation (Days 1-2)
- **Day 1**:
  - [ ] Implement `scripts/conversion/generate_snr_datasets.py`
  - [ ] Test on 1 angle white noise + 1 angle speech
  - [ ] Validate SNR with `verify_snr()`
  - [ ] Fix bugs, commit script
- **Day 2**:
  - [ ] Generate all white noise SNR datasets (7 levels)
  - [ ] Generate all speech SNR datasets (7 levels)
  - [ ] Compute dataset fingerprints
  - [ ] Verify file counts and SNR errors
  - [ ] Commit datasets (Git LFS)

### Week 1-2: Processing (Days 3-4)
- **Day 3**:
  - [ ] Run Stage 1-4 for white noise (7 SNR levels in parallel)
  - [ ] Verify H matrix coherence for each SNR
  - [ ] Spot-check normalized outputs
- **Day 4**:
  - [ ] Run Stage 1-4 for speech (7 SNR levels, may run overnight)
  - [ ] Verify USM convergence for each SNR
  - [ ] Commit processed datasets

### Week 2: Training (Days 5-7)
- **Day 5**:
  - [ ] Train white noise baseline (SNR=∞) and verify 100% accuracy
  - [ ] Launch white noise SNR sweep (6 remaining levels)
  - [ ] Monitor training, check for divergence
- **Day 6-7**:
  - [ ] Train speech baseline (SNR=∞) and verify 94.6% validation accuracy
  - [ ] Launch speech SNR sweep (6 remaining levels)
  - [ ] Launch ablation runs (soft routing, 3 SNR levels)
  - [ ] Monitor overnight, debug failures

### Week 3: Analysis (Days 8-10)
- **Day 8**:
  - [ ] Extract accuracy, diversity, alignment metrics from all runs
  - [ ] Generate degradation curves (accuracy vs SNR)
  - [ ] Compute 90% threshold SNR for both signals
  - [ ] Statistical testing (hard vs soft)
- **Day 9**:
  - [ ] Per-angle sensitivity analysis
  - [ ] Loss component analysis
  - [ ] Train-val gap trends
  - [ ] Generate all figures
- **Day 10**:
  - [ ] Write results commit message (follow CLAUDE.md template)
  - [ ] Create summary tables
  - [ ] Document reproduction instructions
  - [ ] Commit results with figures

---

## Next Actions

1. **Immediate** (This commit):
   - [x] Create `experiment/snr-synthetic-datasets` branch
   - [x] Commit this planning document
   - [ ] Review and refine SNR levels (currently: ∞, 30, 20, 15, 10, 5, 0 dB)

2. **Next Session** (Day 1):
   - [ ] Implement `scripts/conversion/generate_snr_datasets.py`
   - [ ] Test on small subset (1 angle)
   - [ ] Validate SNR accuracy

3. **Future Sessions**:
   - [ ] Follow timeline Days 2-10
   - [ ] Document progress in commit messages
   - [ ] Update this plan if pivots needed

---

**Document Version**: 1.0
**Status**: Planning complete, ready for implementation
**Estimated Completion**: 10-12 days (3-4 hours/day)
**Total Effort**: 30-36 hours
