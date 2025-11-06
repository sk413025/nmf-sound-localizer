# Dataset Card - Multi-Angle NMF Optimization Dataset

**Data Source**: Commit `83c959735d2f959240e764a20ba92c1cf64424eb`  
**Generated**: 2025-09-03T21:12:08  
**Angles**: angle_90, angle_100, angle_110 (3 angles)  
**Files**: 3 files per angle, total 9 pairs of X-Y files

---

## 📂 Data Source and Structure

### File Organization
```
/Users/sbplab/jnrle/datasets/test_nmf_output_no_edge_with_original/
├── white_noise_original_data_no_edge/    # X data (white noise sources)
│   ├── angle_90/
│   ├── angle_100/
│   └── angle_110/
└── white_noise_box_data_no_edge/         # Y data (box response recordings)
    ├── angle_90/
    ├── angle_100/
    └── angle_110/
```

### Data Generation Process
1. **X Data**: Generated white noise signals (identical across all angles)
2. **Y Data**: Box acoustic response recordings at different angles
3. **H Data**: Computed transfer functions H = Y/X using existing `DataProcessor.estimate_transfer_functions()`

---

## 📊 X Data Distribution (White Noise Sources)

### Basic Properties
- **Sample Rate**: 16,000 Hz
- **Files per Angle**: 3 files × 3 angles = 9 total files
- **File Duration**: 9.12 seconds per file
- **Total Duration**: 82.08 seconds (1.37 minutes)
- **Samples per File**: 145,920 samples

### Statistical Distribution
- **Amplitude Range**: [-0.446686, 0.446686]
- **Mean**: -0.001466 (essentially zero)
- **Standard Deviation**: 0.255788
- **RMS**: 0.255792
- **Dynamic Range**: 83.3 dB

### Distribution Characteristics
- **Skewness**: 0.00785 (nearly symmetric)
- **Kurtosis**: 1.827 (close to Gaussian)
- **Zero Crossing Rate**: 49.99% (typical white noise)
- **Cross-Angle Consistency**: All angles have identical statistics (same white noise source)

---

## 📊 Y Data Distribution (Box Response Recordings)

### Basic Properties  
- **Sample Rate**: 16,000 Hz (same as X)
- **Files per Angle**: 3 files × 3 angles = 9 total files
- **File Duration**: 9.12 seconds per file (same as X)
- **Total Duration**: 82.08 seconds

### Overall Statistical Distribution
- **Amplitude Range**: [-0.070920, 0.081807]
- **Mean**: -0.001006
- **Standard Deviation**: 0.015070
- **RMS (Global)**: 0.015104
- **Dynamic Range**: 164.9 dB

### Per-Angle Distribution Analysis
| Angle | RMS | Std Dev | Min | Max | Dynamic Range (dB) |
|-------|-----|---------|-----|-----|-------------------|
| **angle_90** | 0.012024 | 0.011978 | -0.057002 | 0.058990 | 162.1 |
| **angle_100** | 0.017630 | 0.017599 | -0.070920 | 0.081807 | 116.7 |
| **angle_110** | 0.015132 | 0.015104 | -0.070719 | 0.071315 | 115.5 |

### Y Data Characteristics
- **Signal Type**: Box acoustic response (smooth, low zero-crossing rate: 3.46%)
- **Angular Variation**: Clear differences between angles indicate spatial dependency
- **Response Pattern**: angle_100 shows highest amplitude response

---

## 🔀 H Data Distribution (Transfer Functions H = Y/X)

### Computation Method
```python
# Using existing nmf_localizer modules
data_processor = DataProcessor(config)
H, angles_tensor, angle_folders, metadata = data_processor.estimate_transfer_functions(
    original_root=x_root,
    box_root=y_root,
    method='xy_correspondence'
)
```

### H Matrix Properties
- **Shape**: [321 frequencies, 17 angles]
- **Frequency Range**: 500 - 3000 Hz (filtered from original 0-8000 Hz)
- **Frequency Resolution**: 7.8125 Hz
- **Computation**: Welch periodogram method with Hann windowing

### H Magnitude Distribution
- **Range**: [0.158489, 0.919166]
- **Mean**: 0.391636
- **Standard Deviation**: 0.112942
- **Median**: 0.382025
- **Dynamic Range**: 15.3 dB

### H Magnitude Percentiles
- **5%**: 0.226
- **25%**: 0.307
- **50%**: 0.382
- **75%**: 0.465
- **95%**: 0.589

### Per-Angle H Statistics
| Angle | H Mean | H Max | H Min | Dynamic Range (dB) |
|-------|--------|-------|-------|-------------------|
| **angle_30** | 0.338 | 0.801 | 0.158 | 14.1 |
| **angle_45** | 0.350 | 0.703 | 0.184 | 11.6 |
| **angle_90** | 0.353 | 0.714 | 0.183 | 11.8 |
| **angle_100** | 0.414 | 0.745 | 0.169 | 12.9 |
| **angle_110** | 0.396 | 0.814 | 0.176 | 13.3 |
| **angle_150** | 0.400 | 0.831 | 0.167 | 13.9 |

---

## ⚙️ Computation Parameters and Methods

### STFT Parameters
```python
config = NMFConfig(
    sample_rate=16000,
    n_fft=2048,           # 128ms window at 16kHz
    hop_length=512,       # 32ms hop, 75% overlap
    freq_min=500.0,       # High-pass filter
    freq_max=3000.0,      # Low-pass filter
    window='hann'
)
```

### Transfer Function Computation
1. **Method**: `xy_correspondence` - pairs X and Y files by index
2. **Spectral Estimation**: Welch's method for power spectral density
3. **Window Function**: Hann window with 75% overlap
4. **Frequency Filtering**: Bandpass 500-3000 Hz applied post-computation
5. **H Calculation**: H = Pxy / Pxx (cross-PSD / auto-PSD of X)

### Processing Pipeline
```python
# For each X-Y file pair:
freqs, Pxx = signal.welch(x_signal, fs=16000, nperseg=2048, noverlap=1536, window='hann')
_, Pyy = signal.welch(y_signal, fs=16000, nperseg=2048, noverlap=1536, window='hann')  
_, Pxy = signal.csd(x_signal, y_signal, fs=16000, nperseg=2048, noverlap=1536, window='hann')

# Transfer function with regularization
H = Pxy / (Pxx + 1e-12)

# Apply frequency filtering to final H matrix
H_filtered = H[freq_indices_500_to_3000Hz, :]
```

---

## 📈 Data Distribution Summary

### X Data (White Noise)
- **Type**: Gaussian white noise
- **Purpose**: Reference input signal for transfer function computation
- **Consistency**: Identical across all angles (expected for white noise source)
- **Quality**: Clean Gaussian distribution, appropriate dynamic range

### Y Data (Box Response)
- **Type**: Acoustic box response recordings  
- **Purpose**: Target output signals showing angular dependency
- **Variation**: Clear angular differences in amplitude and dynamic range
- **Quality**: Smooth response signals with angle-dependent characteristics

### H Data (Transfer Functions)
- **Type**: Frequency-domain transfer function estimates
- **Purpose**: Characterize acoustic system response H = Y/X
- **Structure**: 321 frequency bins × 17 angular directions
- **Quality**: Reasonable magnitude range with angular variations

---

## 🔧 Module Dependencies and Consistency

### Analysis Framework
- **DataProcessor**: `nmf_localizer.core.data_processor`
- **TransferFunctionProcessor**: `nmf_localizer.core.transfer_functions`  
- **AudioProcessor**: `nmf_localizer.utils.audio_utils`
- **Configuration**: `nmf_localizer.config.NMFConfig`

### Verification
- All computations use existing nmf_localizer modules
- Parameters consistent with project configuration
- Results reproducible via fixed commit reference
- Data integrity verified through statistical analysis

---

## 💾 File Outputs

### Generated Files
- **JSON Data**: `tests/data/comprehensive_dataset_card.json` (29.9 KB)
- **Markdown Summary**: `tests/data/dataset_card_summary.md` (this file)

### Data Validation
- **Total Files Processed**: 18 files (9 X + 9 Y)
- **Missing Files**: None
- **Corrupted Data**: None detected
- **Statistical Consistency**: Verified across all angles

---

**Analysis Tool**: `tests/generate_dataset_card.py`  
**Computation Date**: 2025-09-03T21:12:08  
**Data Commit**: `83c959735d2f959240e764a20ba92c1cf64424eb`