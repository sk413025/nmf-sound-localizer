# Complete Architecture and Data Flow Analysis

## Overview
This document provides a comprehensive analysis of the data flow from raw audio to direction prediction, detailing all dimension transformations and network architectures involved.

**Last Updated**: 2025-10-04 - Added precise runtime dimensions from actual execution logs

## 1. Raw Audio Input
- **Input Format**: Mono waveform (.npy files)
- **Dimensions**: `(T,)` where T = number of audio samples
- **Sampling Rate**: 48000 Hz (verified from runtime logs)
- **Typical Duration**: 2.0 seconds = 96,000 samples
- **Storage Structure**:
  ```
  root/
  ├── angle_0/
  │   ├── clip_000.npy  # Shape: (T,)
  │   ├── clip_001.npy
  │   └── ...
  ├── angle_15/
  │   └── ...
  ```

## 2. STFT Processing (AudioProcessor)

### 2.1 STFT Computation
```python
# In nmf_localizer/utils/audio_utils.py
freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
    wav,           # Shape: (T,) - raw waveform
    fs=48000,      # Sampling frequency
    nperseg=2048,  # FFT size
    noverlap=1536, # 75% overlap (default)
    window='hann'
)
```

**Output Dimensions (Verified from Runtime):**
- `freqs`: `(1025,)` where F_full = nperseg//2 + 1 = 1025 frequency bins
- `times`: `(189,)` for 2-second audio with 75% overlap
- `stft`: `(1025, 189)` complex-valued STFT
- `magnitude`: `(1025, 189)` magnitude spectrogram

### 2.2 Frequency Band Filtering
```python
# Extract 300-3000 Hz band
mask = (freqs >= 300.0) & (freqs <= 3000.0)
mag_band = magnitude[mask, :]  # Shape: (F, N)
```

**Output Dimensions (Verified from Runtime):**
- `mag_band`: `(116, 189)`
- **Precisely 116 frequency bins** covering 300-3000 Hz (confirmed from logs)
- N = 189 time frames for 2-second audio

## 3. Tokenization Pipeline

### 3.1 Available Tokenizers

#### A. PatchTokenizer (Primary Tokenizer - Verified from Logs)
```python
PatchTokenizer(Fp=16, Np=10, n_levels=16)
```
- **Input**: `Y` shape `(116, 189)` magnitude spectrogram
- **Process**:
  1. Convert to log scale: `logY = log(max(Y, 1e-12))`
  2. Divide into patches: Grid of `(7, 18)` where:
     - `Lf = 116 // 16 = 7` frequency patches
     - `Lt = 189 // 10 = 18` time patches (floor division)
  3. Quantize each patch mean to 16 levels
- **Output**: List of tokens like `["<P_0_0_4>", "<P_0_1_4>", "<P_0_2_4>", ...]`
- **Number of tokens**: **126 tokens** (7 × 18 patches)

#### B. LeafTokenizer (Verified from Logs)
```python
LeafTokenizer(sr=48000, n_mels=64, n_levels=16)
```
- **Input**: `wav` shape `(96000,)` raw waveform
- **Process**:
  1. Apply Mel spectrogram (64 mel bins)
  2. Extract top-4 frequency bins per time frame
  3. Quantize to 16 levels
- **Output**: Tokens like `["<LEAF_61:8_57:8_62:8_53:8>", ...]`
- **Number of tokens**: **481 tokens** (time frames from mel spectrogram)

#### C. ScatterTokenizer
```python
ScatterTokenizer(sr=16000, J=6, Q=8, T=2**14, n_levels=16)
```
- **Input**: `wav` shape `(T,)` raw waveform
- **Process**:
  1. Apply scattering transform (wavelet-based)
  2. Quantize scattering coefficients
- **Output**: Tokens like `["<SC_0_5>", "<SC_1_8>", ...]`

### 3.2 Direction Projection Tokens (Optional)
```python
direction_projection_tokens(YF, H, alpha=1.0, topM=None)
```
- **Input**:
  - `YF`: `(F,)` mean spectrum
  - `H`: `(F, D)` transfer functions
- **Process**: Project spectrum onto direction basis
- **Output**: Tokens like `["<R_15:20>", "<R_30:18>", ...]`

## 4. Text Processing and Vocabulary

### 4.1 Vocabulary Building (Verified from Logs)
```python
vocab = Vocab()
vocab.build(token_lists)  # Build from all training tokens
```
- **Vocabulary Size**: **129 tokens** for PatchTokenizer (126 patch tokens + 3 special)
- **Special tokens**: `[PAD]`, `[CLS]`, `[SEP]` (verified from code)
- Token IDs start from 0 ([PAD]), 1 ([CLS]), 2 ([SEP])

### 4.2 Token Encoding (Verified from Runtime)
```python
ids = vocab.encode(tokens, add_cls=True)  # Convert tokens to IDs
input_ids, attn_mask = pad_sequences([ids], vocab.pad_id)
```
- **Sequence Length**: 127 tokens (126 patch tokens + 1 [CLS] token)
- **Output Shape**:
  - `input_ids`: `(1, 127)` for single sample
  - `attn_mask`: `(1, 127)` binary mask (all 1s for full sequence)

## 5. TransformerPolicy Network

### 5.1 Architecture Parameters (Verified Configuration)
```python
TransformerPolicy(
    vocab_size=129,              # Exact vocabulary size for PatchTokenizer
    n_dirs=24,                   # 24 directions (0°, 15°, ..., 345°)
    d_model=256,                 # Hidden dimension
    nhead=8,                     # Attention heads
    num_layers=2,                # Transformer layers
    dim_ff=512,                  # Feedforward dimension
    dropout=0.1,
    max_len=512
)
```
**Total Parameters**: ~1.74M parameters (estimated)

### 5.2 Forward Pass Flow

#### Step 1: Token Embedding
```python
tok_embed = nn.Embedding(vocab_size, d_model)
x = tok_embed(input_ids)  # (B, L) → (B, L, 256)
```

#### Step 2: Position Embedding
```python
pos_embed = nn.Embedding(max_len, d_model)
pos = torch.arange(L).expand(B, L)
x = x + pos_embed(pos)  # Add positional info: (B, L, 256)
```

#### Step 3: Transformer Encoder
```python
encoder = nn.TransformerEncoder(...)
h = encoder(x, src_key_padding_mask=~attn_mask)  # (B, L, 256)
```

#### Step 4: Normalization and Pooling
```python
h = norm(h)  # LayerNorm: (B, L, 256)
# Mean pooling over sequence
pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / attn_mask.sum(dim=1, keepdim=True)
# Result: (B, 256)
```

#### Step 5: Classification Head
```python
head = nn.Linear(d_model, n_dirs)
logits = head(pooled)  # (B, 256) → (B, D)
```

### 5.3 Output
- **Logits Shape**: `(B, D)` where D = number of directions
- **Prediction**: `argmax(logits)` gives predicted direction index
- **Probability**: `softmax(logits)` gives probability distribution over directions

## 6. Complete Data Flow Summary (With Precise Runtime Dimensions)

```
1. Raw Audio Input
   Shape: (96,000) samples [2 seconds @ 48kHz]
   ↓
2. STFT Processing
   Shape: (96,000) → (1025, 189) → (116, 189) [after 300-3000 Hz filtering]
   F = 116 frequency bins, N = 189 time frames
   ↓
3. PatchTokenizer
   Shape: (116, 189) → 126 tokens [7×18 patch grid]
   ↓
4. Vocabulary Encoding
   Shape: 126 tokens → (1, 127) token IDs [with [CLS] token]
   ↓
5. TransformerPolicy
   a. Embedding: (1, 127) → (1, 127, 256)
   b. Transformer: (1, 127, 256) → (1, 127, 256)
   c. Pooling: (1, 127, 256) → (1, 256)
   d. Classification: (1, 256) → (1, 24)
   ↓
6. Direction Prediction
   Shape: (1, 24) logits → scalar direction index [0-23]
```

## 7. Key Dimensions Reference (Verified from Runtime)

| Stage | Variable | Shape | Description |
|-------|----------|-------|-------------|
| Raw Audio | `wav` | `(96,000)` | 2s @ 48kHz |
| STFT | `magnitude` | `(1025, 189)` | Full spectrum |
| Filtered | `Y` | `(116, 189)` | 300-3000 Hz band |
| Patches | `patches` | `7 × 18` | Patch grid |
| Tokens | `tokens` | `126` | PatchTokenizer output |
| Vocabulary | `vocab` | `129` | Including 3 special tokens |
| Encoded | `input_ids` | `(1, 127)` | With [CLS] token |
| Embedding | `x` | `(1, 127, 256)` | Token + position embeddings |
| Transformer | `h` | `(1, 127, 256)` | After 2 layers |
| Pooled | `pooled` | `(1, 256)` | Mean pooling |
| Output | `logits` | `(1, 24)` | 24 directions |

## 8. Transfer Functions and Source Dictionary

### Transfer Functions H (Verified Dimensions)
- **Shape**: `(116, 24)`
  - F = 116 frequency bins (matches filtered spectrogram exactly)
  - D = 24 directions (15° increments: 0°, 15°, 30°, ..., 345°)
- **Purpose**: Encodes acoustic response at each direction
- **Usage**: Optional direction projection tokens
- **File**: `h_matrix_80_150_freq_300_3000.pth`

### Source Dictionary W
- **Shape**: `(F, K)` where:
  - F = frequency bins
  - K = number of source components
- **Purpose**: NMF basis for source separation
- **Usage**: In advantage computation for RL training

## 9. Precise Runtime Dimension Summary

Based on actual execution logs from the system, here are the exact dimensions at each stage:

### Audio Processing Pipeline
1. **Input Audio**: `(96,000,)` - 2 seconds @ 48kHz
2. **STFT Output**: `(1025, 189)` - 1025 frequency bins, 189 time frames
3. **Filtered Spectrogram**: `(116, 189)` - Precisely 116 bins for 300-3000 Hz

### Tokenization Pipeline
1. **PatchTokenizer Grid**: `7 × 18 = 126 patches`
2. **Token Sequence**: 126 patch tokens + 1 [CLS] = 127 total
3. **Vocabulary Size**: 129 (126 unique patches + 3 special tokens)

### Neural Network Pipeline
1. **Input IDs**: `(1, 127)` - Single sample, 127 tokens
2. **Embeddings**: `(1, 127, 256)` - 256-dimensional embeddings
3. **After Transformer**: `(1, 127, 256)` - Maintains sequence length
4. **After Pooling**: `(1, 256)` - Aggregated representation
5. **Output Logits**: `(1, 24)` - 24 possible directions

### Memory and Computational Requirements
- **Model Parameters**: ~1.74M total
- **Sequence Length**: Fixed at 127 for PatchTokenizer
- **Batch Processing**: Typically B=1 for inference, variable for training

## 10. Important Notes

1. **No BYOL in Current Architecture**: The current system uses a Transformer-based policy network, not BYOL (Bootstrap Your Own Latent). BYOL references in project docs may refer to planned future work or previous experiments.

2. **Direction Prediction Method**: The system predicts direction through:
   - Tokenizing spectrograms into discrete symbols
   - Learning token patterns associated with each direction
   - Using Transformer to map token sequences to direction probabilities

3. **Tokenization is Key**: The choice of tokenizer critically affects:
   - Information preservation
   - Sequence length
   - Model complexity
   - Training efficiency

4. **Reinforcement Learning Context**: The system uses:
   - Advantage computation based on NMF localization quality
   - GRPO (Group Relative Policy Optimization) for training
   - On-policy buffer for experience collection