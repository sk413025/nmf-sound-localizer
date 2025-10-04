# Complete Architecture and Data Flow Analysis

## Overview
This document provides a comprehensive analysis of the data flow from raw audio to direction prediction, detailing all dimension transformations and network architectures involved.

## 1. Raw Audio Input
- **Input Format**: Mono waveform (.npy files)
- **Dimensions**: `(T,)` where T = number of audio samples
- **Sampling Rate**: Configurable (default: 16000 Hz or 48000 Hz)
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

**Output Dimensions:**
- `freqs`: `(F_full,)` where F_full = nperseg//2 + 1 = 1025 frequency bins
- `times`: `(N,)` where N = number of time frames (depends on T and overlap)
- `stft`: `(F_full, N)` complex-valued STFT
- `magnitude`: `(F_full, N)` magnitude spectrogram

### 2.2 Frequency Band Filtering
```python
# Extract 300-3000 Hz band
mask = (freqs >= 300.0) & (freqs <= 3000.0)
mag_band = magnitude[mask, :]  # Shape: (F, N)
```

**Output Dimensions:**
- `mag_band`: `(F, N)` where F ≈ 140-150 bins (depends on fs and nperseg)
- For fs=48000, nperseg=2048: F ≈ 140 bins covering 300-3000 Hz

## 3. Tokenization Pipeline

### 3.1 Available Tokenizers

#### A. PatchTokenizer
```python
PatchTokenizer(Fp=16, Np=10, n_levels=16)
```
- **Input**: `Y` shape `(F, N)` magnitude spectrogram
- **Process**:
  1. Convert to log scale: `logY = log(max(Y, 1e-12))`
  2. Divide into patches: Grid of `(Lf, Lt)` where:
     - `Lf = F // Fp` (frequency patches)
     - `Lt = N // Np` (time patches)
  3. Quantize each patch to 16 levels
- **Output**: List of tokens like `["<P_0_0_5>", "<P_0_1_8>", ...]`
- **Number of tokens**: `Lf * Lt`

#### B. LeafTokenizer
```python
LeafTokenizer(sr=16000, n_mels=64, n_levels=16)
```
- **Input**: `wav` shape `(T,)` raw waveform
- **Process**:
  1. Apply Mel spectrogram or LEAF frontend
  2. Extract top-4 frequency bins per time frame
  3. Quantize to 16 levels
- **Output**: Tokens like `["<LEAF_12:5_8:3_2:1_0:0>", ...]`
- **Number of tokens**: Equal to number of time frames

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

### 4.1 Vocabulary Building
```python
vocab = Vocab()
vocab.build(token_lists)  # Build from all training tokens
```
- **Vocabulary Size**: Depends on tokenizer and data
- Includes special tokens: `<PAD>`, `<CLS>`, `<EOS>`, `<UNK>`

### 4.2 Token Encoding
```python
ids = vocab.encode(tokens, add_cls=True)  # Convert tokens to IDs
input_ids, attn_mask = pad_sequences([ids], vocab.pad_id)
```
- **Output Shape**:
  - `input_ids`: `(B, L)` where L = padded sequence length
  - `attn_mask`: `(B, L)` binary mask

## 5. TransformerPolicy Network

### 5.1 Architecture Parameters
```python
TransformerPolicy(
    vocab_size=len(vocab.itos),  # e.g., 1000-5000 tokens
    n_dirs=H.shape[1],           # e.g., 24 directions (0°, 15°, ..., 345°)
    d_model=256,                 # Hidden dimension
    nhead=8,                     # Attention heads
    num_layers=2,                # Transformer layers
    dim_ff=512,                  # Feedforward dimension
    dropout=0.1,
    max_len=512
)
```

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

## 6. Complete Data Flow Summary

```
1. Raw Audio Input
   Shape: (T,) samples
   ↓
2. STFT Processing
   Shape: (T,) → (F_full, N) → (F, N) [after band filtering]
   F ≈ 140 bins, N = time frames
   ↓
3. Tokenization
   Shape: (F, N) → List[str] of length K tokens
   K depends on tokenizer (e.g., Lf*Lt for PatchTokenizer)
   ↓
4. Vocabulary Encoding
   Shape: List[str] → (1, L) token IDs [L = padded length]
   ↓
5. TransformerPolicy
   a. Embedding: (1, L) → (1, L, 256)
   b. Transformer: (1, L, 256) → (1, L, 256)
   c. Pooling: (1, L, 256) → (1, 256)
   d. Classification: (1, 256) → (1, D)
   ↓
6. Direction Prediction
   Shape: (1, D) logits → scalar direction index
```

## 7. Key Dimensions Reference

| Stage | Variable | Shape | Description |
|-------|----------|-------|-------------|
| Raw Audio | `wav` | `(T,)` | T = audio samples |
| STFT | `magnitude` | `(F_full, N)` | F_full = 1025, N = time frames |
| Filtered | `Y` | `(F, N)` | F ≈ 140 (300-3000 Hz) |
| Tokens | `tokens` | `List[str]` | Variable length K |
| Encoded | `input_ids` | `(B, L)` | B = batch, L = padded length |
| Embedding | `x` | `(B, L, 256)` | 256 = d_model |
| Transformer | `h` | `(B, L, 256)` | Hidden states |
| Pooled | `pooled` | `(B, 256)` | Aggregated features |
| Output | `logits` | `(B, D)` | D = n_dirs (e.g., 24) |

## 8. Transfer Functions and Source Dictionary

### Transfer Functions H
- **Shape**: `(F, D)` where:
  - F = frequency bins (matches spectrogram)
  - D = number of directions
- **Purpose**: Encodes acoustic response at each direction
- **Usage**: Optional direction projection tokens

### Source Dictionary W
- **Shape**: `(F, K)` where:
  - F = frequency bins
  - K = number of source components
- **Purpose**: NMF basis for source separation
- **Usage**: In advantage computation for RL training

## 9. Important Notes

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