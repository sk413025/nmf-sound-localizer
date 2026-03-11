# Phase 6: White Noise SNR Experiment Results (Sensor Noise Model)

## 1. Executive Summary
This experiment validates the "Sensor Noise" methodology for SNR testing. Unlike previous attempts where noise was added to the source (and subsequently cancelled by normalization), this experiment adds white noise to the microphone signals ($Y = S + N$).

**Result:** The model shows expected performance degradation as SNR decreases, confirming the validity of the test methodology.

| SNR Level | Accuracy | Status |
|-----------|----------|--------|
| Baseline (Inf) | 100.00% | ✅ Robust |
| 30 dB | 100.00% | ✅ Robust |
| 20 dB | 100.00% | ✅ Robust |
| 15 dB | 100.00% | ✅ Robust |
| 10 dB | 91.89% | ⚠️ Degrading |
| 5 dB | 37.84% | ❌ Failed |
| 0 dB | 16.22% | ❌ Failed |

## 2. Methodology Correction
### Previous Flaw (Source Noise)
- **Method:** $Y = (S + N) * H$
- **Issue:** Normalization $\frac{Y}{||Y||}$ cancelled the noise scale because the noise was convolved and scaled with the signal.
- **Outcome:** 100% accuracy at all SNR levels (invalid test).

### Corrected Method (Sensor Noise)
- **Method:** $Y = (S * H) + N$
- **Implementation:** Additive White Gaussian Noise (AWGN) applied to the microphone array signals.
- **Normalization:** $\frac{S*H + N}{||S*H + N||}$ preserves the Signal-to-Noise ratio.

## 3. Experimental Setup
- **Model:** `omp-transformer-ldv.py` (FullTransformerRoutedSoftOMP)
- **Training Data:** Clean speech (no noise augmentation).
- **Test Data:** 111 clips per SNR level (White Noise).
- **Processing Pipeline:**
    1. **Raw Audio (48kHz)**: `white_noise_snr_{level}_raw`
    2. **VAD (Sync X-Y)**: `apply_spectrogram_vad.py`
    3. **Normalization**: `normalize_to_unit_range.py`
    4. **Evaluation**: `omp-transformer-ldv.py` (Internal 16kHz resampling)

## 4. Detailed Analysis
- **High SNR (>15dB)**: The Transformer model is highly robust to white noise, maintaining perfect accuracy.
- **Breaking Point (10dB)**: Performance begins to drop significantly. This suggests the model's attention mechanism starts failing to distinguish the directional signal from the noise floor.
- **Failure Zone (<5dB)**: The model collapses. At 0dB (Signal power = Noise power), accuracy is 16.2%, which is near random guess for this dataset.

## 5. Conclusion
The "Sensor Noise" model is the correct approach for evaluating robustness. The Transformer model demonstrates strong resilience down to 15dB SNR but requires specific noise-robust training or denoising front-ends to handle environments with SNR < 10dB.
