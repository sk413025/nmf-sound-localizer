#!/usr/bin/env python
"""Detailed analysis of patches and their physical meaning in the Transformer pipeline."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import project modules
from doa_rl.features import PatchTokenizer
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab
from nmf_localizer.utils.audio_utils import AudioProcessor


def create_test_spectrogram():
    """Create a test spectrogram with known patterns."""
    # Create synthetic audio with directional pattern
    fs = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))

    # Simulate sound from 45 degrees - stronger in certain frequencies
    freq1, freq2 = 1000, 2000  # Hz
    wav = np.sin(2 * np.pi * freq1 * t) + 0.5 * np.sin(2 * np.pi * freq2 * t)
    wav += 0.1 * np.random.randn(len(wav))  # Add noise

    # Compute STFT
    freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=2048, noverlap=1536, window='hann'
    )

    # Filter to 300-3000 Hz band
    mask = (freqs >= 300.0) & (freqs <= 3000.0)
    mag_band = magnitude[mask, :]

    return mag_band, freqs[mask], times


def analyze_patches(Y, freqs, times):
    """Analyze patches and their physical meaning."""

    print("=" * 80)
    print("PATCH ANALYSIS: Physical Meaning and Structure")
    print("=" * 80)

    # Initialize tokenizer
    Fp, Np = 16, 10  # Patch dimensions
    n_levels = 16
    tokenizer = PatchTokenizer(Fp=Fp, Np=Np, n_levels=n_levels)

    F, N = Y.shape
    print(f"\n1. SPECTROGRAM DIMENSIONS:")
    print(f"   Input spectrogram Y: {Y.shape}")
    print(f"   - F = {F} frequency bins ({freqs[0]:.1f} - {freqs[-1]:.1f} Hz)")
    print(f"   - N = {N} time frames ({times[0]:.3f} - {times[-1]:.3f} seconds)")

    # Calculate patch grid
    Lf = F // Fp  # Number of frequency patches
    Lt = N // Np  # Number of time patches

    print(f"\n2. PATCH GRID STRUCTURE:")
    print(f"   Patch size: Fp={Fp} frequency bins × Np={Np} time frames")
    print(f"   Patch grid: {Lf} × {Lt} = {Lf * Lt} total patches")

    # Physical dimensions of each patch
    freq_per_patch = (freqs[-1] - freqs[0]) / Lf
    time_per_patch = (times[-1] - times[0]) / Lt

    print(f"\n3. PHYSICAL DIMENSIONS PER PATCH:")
    print(f"   Frequency coverage: ~{freq_per_patch:.1f} Hz per patch")
    print(f"   Time duration: ~{time_per_patch:.3f} seconds per patch")
    print(f"   Each patch represents: {Fp} freq bins × {Np} time frames = {Fp*Np} values")

    # Extract and analyze actual patches
    print(f"\n4. PATCH EXTRACTION AND QUANTIZATION:")
    logY = np.log(np.maximum(Y, 1e-12))

    patches_data = []
    patches_tokens = []

    for i in range(Lf):
        for j in range(Lt):
            # Extract patch
            patch = logY[i * Fp:(i + 1) * Fp, j * Np:(j + 1) * Np]

            # Calculate patch statistics
            patch_mean = float(patch.mean())
            patch_std = float(patch.std())

            # Quantize to level (same as tokenizer)
            level = int(np.clip((patch_mean + 15) / 30 * (n_levels - 1), 0, n_levels - 1))

            # Create token
            token = f"<P_{i}_{j}_{level}>"

            patches_data.append({
                'patch': patch,
                'mean': patch_mean,
                'std': patch_std,
                'level': level,
                'token': token,
                'freq_idx': i,
                'time_idx': j,
                'freq_range': (freqs[i * Fp], freqs[min((i + 1) * Fp - 1, F - 1)]),
                'time_range': (times[j * Np], times[min((j + 1) * Np - 1, N - 1)])
            })
            patches_tokens.append(token)

    # Show example patches
    print("\n   Example patches (first 3):")
    for idx in range(min(3, len(patches_data))):
        p = patches_data[idx]
        print(f"   Patch {idx}: {p['token']}")
        print(f"     - Position: freq_patch={p['freq_idx']}, time_patch={p['time_idx']}")
        print(f"     - Frequency: {p['freq_range'][0]:.1f} - {p['freq_range'][1]:.1f} Hz")
        print(f"     - Time: {p['time_range'][0]:.3f} - {p['time_range'][1]:.3f} sec")
        print(f"     - Mean energy: {p['mean']:.2f} (log scale)")
        print(f"     - Quantized level: {p['level']}/15")

    return patches_data, patches_tokens


def analyze_transformer_processing(patches_tokens):
    """Analyze how Transformer processes patch tokens."""

    print("\n" + "=" * 80)
    print("TRANSFORMER PROCESSING OF PATCHES")
    print("=" * 80)

    # Build vocabulary
    vocab = Vocab()
    vocab.build([patches_tokens])

    print(f"\n1. TOKEN VOCABULARY:")
    print(f"   Total unique tokens: {len(vocab.itos)}")
    print(f"   Token sequence length: {len(patches_tokens)}")

    # Encode tokens
    ids = vocab.encode(patches_tokens, add_cls=True)

    print(f"\n2. TOKEN ENCODING:")
    print(f"   Original tokens: {len(patches_tokens)}")
    print(f"   Encoded IDs: {len(ids)} (including [CLS])")
    print(f"   Sample encoding: {patches_tokens[0]} → ID {ids[1]}")  # Skip CLS

    # Create input tensors
    input_ids = torch.tensor([ids], dtype=torch.long)
    attn_mask = torch.ones_like(input_ids)

    # Initialize model
    model = TransformerPolicy(
        vocab_size=len(vocab.itos),
        n_dirs=24,
        d_model=256,
        nhead=8,
        num_layers=2
    )

    print(f"\n3. TRANSFORMER ARCHITECTURE:")
    print(f"   Input: {input_ids.shape} token IDs")
    print(f"   Embedding dim: 256")
    print(f"   Attention heads: 8 (32 dims per head)")
    print(f"   Layers: 2")

    with torch.no_grad():
        # Get embeddings
        tok_embed = model.tok_embed(input_ids)
        pos_embed = model.pos_embed(torch.arange(input_ids.shape[1]).unsqueeze(0))
        embeddings = tok_embed + pos_embed

        print(f"\n4. EMBEDDING STAGE:")
        print(f"   Token embeddings: {tok_embed.shape}")
        print(f"   Position embeddings: {pos_embed.shape}")
        print(f"   Combined: {embeddings.shape}")

        # Through transformer
        key_padding_mask = attn_mask.eq(0)
        h = model.encoder(embeddings, src_key_padding_mask=key_padding_mask)

        print(f"\n5. TRANSFORMER ENCODING:")
        print(f"   After transformer: {h.shape}")
        print(f"   Each token now has 256-dim contextual representation")

        # Pooling and classification
        h = model.norm(h)
        denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = model.head(pooled)

        print(f"\n6. OUTPUT STAGE:")
        print(f"   After pooling: {pooled.shape}")
        print(f"   Direction logits: {logits.shape}")
        print(f"   Predicted direction: {torch.argmax(logits, dim=-1).item()}")

    return model, input_ids, embeddings


def explain_patch_transformer_relationship():
    """Explain the relationship between patches and Transformer."""

    print("\n" + "=" * 80)
    print("PATCH-TRANSFORMER RELATIONSHIP: Physical Meaning")
    print("=" * 80)

    print("""
1. PATCH AS ACOUSTIC FINGERPRINT:
   Each patch captures a local time-frequency pattern:
   - Frequency range: ~385 Hz (e.g., 300-685 Hz for patch 0)
   - Time window: ~0.105 seconds
   - Physical meaning: Energy distribution in that time-freq region
   - Quantization: Reduces to 16 discrete energy levels

2. TOKEN ENCODING:
   <P_i_j_k> where:
   - i = frequency position (0-6): Which frequency band
   - j = time position (0-17): Which time window
   - k = energy level (0-15): How much energy in that region

   Example: <P_3_8_12>
   - Patch at middle frequency (band 3/7)
   - Middle time (frame 8/18)
   - High energy (level 12/15)

3. TRANSFORMER ATTENTION MECHANISM:
   The Transformer learns relationships between patches:

   a) Temporal Patterns:
      - Patches at same frequency but different times
      - Learns sound movement/evolution patterns
      - Example: Energy moving from P_3_0 → P_3_1 → P_3_2 (time progression)

   b) Spectral Patterns:
      - Patches at same time but different frequencies
      - Learns harmonic structure
      - Example: P_0_5, P_1_5, P_2_5 (frequency stack at time 5)

   c) Diagonal Patterns:
      - Frequency-time trajectories
      - Learns Doppler effects or pitch changes
      - Example: P_0_0 → P_1_1 → P_2_2 (rising frequency over time)

4. DIRECTION PREDICTION:
   Different directions create distinct patch patterns:

   Direction 0°:   High energy in patches P_2_*, P_3_* (mid frequencies)
   Direction 90°:  High energy in patches P_0_*, P_1_* (low frequencies)
   Direction 180°: High energy in patches P_5_*, P_6_* (high frequencies)

   The Transformer learns these direction-specific patterns through:
   - Self-attention: Which patches co-occur for each direction
   - Position encoding: Spatial arrangement of patches
   - Pooling: Aggregate all patch information into direction vote

5. ATTENTION WEIGHTS INTERPRETATION:
   When predicting direction, attention weights show:
   - Which patches are most informative
   - How patches vote together for a direction
   - Temporal coherence (adjacent time patches)
   - Spectral coherence (adjacent frequency patches)
    """)

    print("\n" + "=" * 80)
    print("MATHEMATICAL FORMULATION")
    print("=" * 80)

    print("""
Given spectrogram Y ∈ ℝ^(F×N), patches are computed as:

1. Patch Extraction:
   P_ij = Y[i·Fp:(i+1)·Fp, j·Np:(j+1)·Np] ∈ ℝ^(Fp×Np)

2. Feature Extraction:
   f_ij = mean(log(P_ij)) ∈ ℝ

3. Quantization:
   q_ij = round((f_ij + 15) / 30 × 15) ∈ {0,1,...,15}

4. Token Generation:
   t_ij = "<P_" + str(i) + "_" + str(j) + "_" + str(q_ij) + ">"

5. Transformer Processing:
   E = TokenEmbed(t) + PosEmbed(pos) ∈ ℝ^(L×d)
   H = TransformerEncoder(E) ∈ ℝ^(L×d)
   z = MeanPool(H) ∈ ℝ^d
   logits = Linear(z) ∈ ℝ^D

Where:
- L = Lf × Lt (total patches)
- d = 256 (embedding dimension)
- D = 24 (directions)
    """)


def visualize_patch_structure():
    """Create visualization of patch structure."""

    print("\n" + "=" * 80)
    print("PATCH GRID VISUALIZATION")
    print("=" * 80)

    print("""
    Spectrogram Y (116 × 189) divided into patches:

    Time →
    ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
  F │ P00│ P01│ P02│ P03│ P04│ P05│ P06│ P07│ P08│ P09│ P10│ P11│ P12│ P13│ P14│ P15│ P16│ P17│
  r ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  e │ P10│ P11│ P12│ P13│ P14│ P15│ P16│ P17│ P18│ P19│P110│P111│P112│P113│P114│P115│P116│P117│
  q ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  u │ P20│ P21│ P22│ P23│ P24│ P25│ P26│ P27│ P28│ P29│P210│P211│P212│P213│P214│P215│P216│P217│
  e ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  n │ P30│ P31│ P32│ P33│ P34│ P35│ P36│ P37│ P38│ P39│P310│P311│P312│P313│P314│P315│P316│P317│
  c ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
  y │ P40│ P41│ P42│ P43│ P44│ P45│ P46│ P47│ P48│ P49│P410│P411│P412│P413│P414│P415│P416│P417│
  ↓ ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
    │ P50│ P51│ P52│ P53│ P54│ P55│ P56│ P57│ P58│ P59│P510│P511│P512│P513│P514│P515│P516│P517│
    ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
    │ P60│ P61│ P62│ P63│ P64│ P65│ P66│ P67│ P68│ P69│P610│P611│P612│P613│P614│P615│P616│P617│
    └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘

    Each patch Pij:
    - Covers 16 frequency bins × 10 time frames
    - Represents ~385 Hz × 0.105 seconds
    - Encoded as token <P_i_j_level>
    - Becomes 256-dim embedding in Transformer

    Total: 7 × 18 = 126 patches → 126 tokens → 126 embeddings
    """)


def main():
    """Run complete patch analysis."""

    # Create test spectrogram
    Y, freqs, times = create_test_spectrogram()

    # Analyze patches
    patches_data, patches_tokens = analyze_patches(Y, freqs, times)

    # Analyze Transformer processing
    model, input_ids, embeddings = analyze_transformer_processing(patches_tokens)

    # Explain relationships
    explain_patch_transformer_relationship()

    # Visualize structure
    visualize_patch_structure()

    print("\n" + "=" * 80)
    print("SUMMARY: Patches are local time-frequency energy patterns that become")
    print("discrete tokens for the Transformer to learn direction-specific patterns")
    print("=" * 80)


if __name__ == "__main__":
    main()