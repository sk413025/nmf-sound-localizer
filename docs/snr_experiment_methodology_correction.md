# SNR Experiment Methodology Correction: From Source Noise to Sensor Noise

**Date:** December 9, 2025  
**Status:** Investigation Complete / Correction Planned  
**Author:** GitHub Copilot (Gemini 3 Pro)

## 1. Background & Motivation

**Context:**  
Phase 6 of the project focuses on evaluating the robustness of the OMP-Transformer model (and the physical G-Routing mechanism) against noise. We used a "White Noise" dataset as the source signal.

**The Anomaly:**  
Upon running the SNR sweep (Inf, 30dB, ..., 0dB), we observed:
1.  **100% Accuracy** across ALL levels, even at 0dB.
2.  **Identical Reconstruction Error** (`rec_loss ≈ 0.0006`) across all levels.
3.  **Identical Loss Curves** in training/evaluation logs.

**Why this is impossible:**  
In a valid SNR experiment, as noise increases (SNR decreases), the input signal $y$ becomes distorted ($y \approx y_{clean} + n$). The dictionary $D$ is fixed (learned from clean data). Therefore, the reconstruction error $||y - D x||^2$ *must* increase because the noise $n$ cannot be perfectly represented by the sparse combination of atoms in $D$. Constant reconstruction error implies the model is seeing effectively the same input.

## 2. Investigation Process (Reproduction Steps)

We performed a systematic "Process of Elimination" to identify the root cause.

### Step 1: Verify Dataset Paths (Rule out "Reading Wrong Files")
*   **Hypothesis:** The evaluation script might be hardcoded to read the clean (`Inf`) dataset regardless of the argument.
*   **Action:** Checked logs (`snr_transformer_evaluation.log`).
*   **Observation:** Logs showed correct paths:
    ```
    Dataset root: .../processed-48k/white_noise_box_snr0dB_sync_vad_normalized
    ```
*   **Conclusion:** Script is pointing to the correct folders.

### Step 2: Verify Raw Data Differences (Rule out "Files are Identical")
*   **Hypothesis:** The dataset generation script might have failed to write new files, leaving copies of clean data.
*   **Action:** Created `debug_dataset_loading.py` to load `clip_000.npy` from `Inf` and `0dB` folders.
*   **Observation:**
    *   Raw Mean Difference: `3.16e-05` (Non-zero).
    *   The files are binary different.
*   **Conclusion:** The files contain different data.

### Step 3: Verify Normalized Data Differences (The Smoking Gun)
*   **Hypothesis:** The difference exists in amplitude but not in shape. Since the model normalizes inputs ($y \leftarrow y / ||y||$), amplitude differences are erased.
*   **Action:** Calculated difference between normalized vectors in `debug_dataset_loading.py`.
*   **Observation:**
    *   Normalized Mean Difference: `0.0014` (~0.1%).
    *   This is extremely small for a 0dB signal (where noise power = signal power).
*   **Conclusion:** The **Spectral Shape** of the noisy signal is almost identical to the clean signal.

### Step 4: Code Audit of Noise Generation
*   **Action:** Examined `scripts/conversion/generate_snr_datasets.py`.
*   **Finding:**
    ```python
    # 3. Shape noise spectrum to match signal spectrum
    f_s, t_s, S_signal = sp_signal.stft(signal, ...)
    f_w, t_w, S_white = sp_signal.stft(white_noise, ...)
    
    # Shaping filter: match signal's frequency distribution
    shaping_filter = signal_envelope / white_envelope
    S_shaped = S_white * shaping_filter
    ```
*   **Root Cause:** The script explicitly forces the noise to have the exact same frequency envelope as the signal before adding it.

## 3. Physical Analysis: Source Noise vs. Sensor Noise

The root cause lies in the physical interpretation of "Noise".

### Scenario A: Source Noise (Current Implementation)
*   **Physics:** The noise exists *in the source* (e.g., a noisy audio file played through the speaker).
*   **Propagation:** $(S_{source} + N_{source}) \xrightarrow{\text{Box Transfer Function } H} Y_{LDV}$
*   **Math:** $Y = (S + N) * H = S*H + N*H$.
*   **Effect:** Since $S$ and $N$ are both white noise (flat spectrum), $(S+N)$ is just a louder white noise. After passing through the Box ($H$), both components acquire the *same* spectral shape (Box resonances).
*   **Model View:** The model sees $k \cdot (S*H)$. After normalization, this equals $S*H$. The noise is invisible.

### Scenario B: Sensor Noise (Correct Implementation for Robustness)
*   **Physics:** The noise exists *in the measurement* (e.g., LDV sensor thermal noise, ambient interference).
*   **Propagation:** $S_{source} \xrightarrow{H} Y_{signal}$; $N_{sensor}$ adds directly.
*   **Math:** $Y = (S * H) + N_{sensor}$.
*   **Effect:** $S*H$ has peaks and valleys (Box resonances). $N_{sensor}$ is flat (white).
*   **Result:** The noise fills in the spectral valleys of the signal. The spectral shape changes significantly.
*   **Model View:** The input is no longer just a scaled version of the clean signal. The "contrast" of the spectral features is reduced. This challenges the sparse coding model.

## 4. Correction Plan

We must switch from simulating Source Noise to simulating Sensor Noise.

### 1. Modify Generation Script
*   **File:** `scripts/conversion/generate_snr_datasets.py`
*   **Change:** Remove the spectral shaping step. Add white noise directly to the signal.
*   **Note:** Since the input `clean_root` contains files that are already recorded/processed (i.e., they are $S*H$), adding white noise to them correctly simulates Sensor Noise ($Y = Y_{clean} + N$).

### 2. Regenerate Datasets
*   Re-run the generation script for all SNR levels.
*   Re-run the batch processing (VAD + Normalization).

### 3. Re-evaluate
*   Run the `batch_evaluate_white_noise_snr_transformer.sh` script again.
*   Expect to see accuracy degradation at lower SNRs (e.g., < 10dB).

## 5. Commit Lineage Plan

**Commit 1: Documentation of Methodology Error**
*   **File:** `docs/snr_experiment_methodology_correction.md` (This file)
*   **Message:** `Docs: Document SNR experiment methodology error (Source vs Sensor noise)`
*   **Purpose:** Preserve the lesson learned. "Negative results" and "Methodology bugs" are valuable knowledge.

**Commit 2: Fix Noise Generation Logic**
*   **File:** `scripts/conversion/generate_snr_datasets.py`
*   **Change:** Remove spectral shaping logic.
*   **Message:** `Fix: Switch SNR generation from Source Noise to Sensor Noise`
*   **Details:** "Previously, noise was shaped to match signal spectrum, which normalization cancelled out. Now adding flat white noise to simulate sensor noise."

**Commit 3: Regenerate and Re-evaluate (Results)**
*   **Files:** `results/white_noise_transformer/**`, `docs/snr_experiment_phase6_white_noise_results.md`
*   **Message:** `Results: Phase 6 White Noise SNR Sweep (Sensor Noise)`
*   **Details:** Updated results with physically valid noise model.
