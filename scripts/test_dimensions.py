#!/usr/bin/env python
"""Test script to capture precise dimensions and shapes through the pipeline."""

import logging
import numpy as np
import torch
from pathlib import Path

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer, LeafTokenizer, direction_projection_tokens
from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor

def test_dimensions():
    """Run through the pipeline and log all dimensions."""

    # Test data paths
    test_root = "/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
    tf_path = "h_matrix_80_150_freq_300_3000.pth"

    # Check if test data exists, otherwise create synthetic data
    if not Path(test_root).exists():
        logger.warning(f"Test data not found at {test_root}, using synthetic data")
        test_root = None

    # 1. Create synthetic audio data for testing
    logger.info("=" * 60)
    logger.info("1. RAW AUDIO DIMENSIONS")
    logger.info("=" * 60)

    fs = 48000  # Sampling rate
    duration = 2.0  # seconds
    T = int(fs * duration)
    wav = np.random.randn(T) * 0.1
    logger.info(f"Raw audio shape: {wav.shape} = ({T:,} samples)")
    logger.info(f"Duration: {duration} seconds at {fs} Hz")

    # 2. STFT Processing
    logger.info("\n" + "=" * 60)
    logger.info("2. STFT PROCESSING")
    logger.info("=" * 60)

    nperseg = 2048
    noverlap = int(nperseg * 0.75)

    freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=nperseg, noverlap=noverlap, window='hann'
    )

    logger.info(f"STFT parameters:")
    logger.info(f"  - nperseg (FFT size): {nperseg}")
    logger.info(f"  - noverlap: {noverlap} ({100*noverlap/nperseg:.1f}% overlap)")
    logger.info(f"  - hop_length: {nperseg - noverlap}")
    logger.info(f"\nOutput dimensions:")
    logger.info(f"  - freqs shape: {freqs.shape} (frequency bins)")
    logger.info(f"  - times shape: {times.shape} (time frames)")
    logger.info(f"  - stft shape: {stft.shape} (complex)")
    logger.info(f"  - magnitude shape: {magnitude.shape}")

    # 3. Frequency band filtering
    logger.info("\n" + "=" * 60)
    logger.info("3. FREQUENCY BAND FILTERING")
    logger.info("=" * 60)

    freq_min, freq_max = 300.0, 3000.0
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :]

    logger.info(f"Frequency range: {freq_min} - {freq_max} Hz")
    logger.info(f"Original frequency bins: {len(freqs)}")
    logger.info(f"Selected frequency bins: {mask.sum()}")
    logger.info(f"Filtered magnitude shape: {mag_band.shape}")
    logger.info(f"  F = {mag_band.shape[0]} frequency bins")
    logger.info(f"  N = {mag_band.shape[1]} time frames")

    Y = mag_band  # For tokenization

    # 4. Tokenization
    logger.info("\n" + "=" * 60)
    logger.info("4. TOKENIZATION")
    logger.info("=" * 60)

    # Test PatchTokenizer
    logger.info("\n4.1 PatchTokenizer:")
    patch_tok = PatchTokenizer(Fp=16, Np=10, n_levels=16)
    patch_tokens = patch_tok(Y)
    Lf = Y.shape[0] // 16
    Lt = Y.shape[1] // 10
    logger.info(f"  Input Y shape: {Y.shape}")
    logger.info(f"  Patch grid: ({Lf}, {Lt}) = {Lf}x{Lt} patches")
    logger.info(f"  Output tokens: {len(patch_tokens)} tokens")
    logger.info(f"  Sample tokens: {patch_tokens[:3]}")

    # Test LeafTokenizer
    logger.info("\n4.2 LeafTokenizer:")
    leaf_tok = LeafTokenizer(sr=fs, n_mels=64, n_levels=16)
    leaf_tokens = leaf_tok(wav)
    logger.info(f"  Input wav shape: {wav.shape}")
    logger.info(f"  Output tokens: {len(leaf_tokens)} tokens")
    logger.info(f"  Sample tokens: {leaf_tokens[:2]}")

    # 5. Load transfer functions (if available)
    logger.info("\n" + "=" * 60)
    logger.info("5. TRANSFER FUNCTIONS AND DIRECTION TOKENS")
    logger.info("=" * 60)

    if Path(tf_path).exists():
        H, angles = load_H(tf_path)
        logger.info(f"Transfer function H shape: {H.shape}")
        logger.info(f"  F = {H.shape[0]} frequency bins")
        logger.info(f"  D = {H.shape[1]} directions")
        logger.info(f"Angles shape: {angles.shape}")
        logger.info(f"Angle values: {angles[:5].tolist()}... (showing first 5)")

        # Test direction projection tokens
        YF = Y.mean(axis=1)  # Mean spectrum
        dir_tokens = direction_projection_tokens(YF, H.numpy(), alpha=1.0, topM=5)
        logger.info(f"\nDirection projection tokens:")
        logger.info(f"  Input YF shape: {YF.shape} (mean spectrum)")
        logger.info(f"  Output tokens: {len(dir_tokens)} tokens")
        logger.info(f"  Sample tokens: {dir_tokens}")

        n_dirs = H.shape[1]
    else:
        logger.warning(f"Transfer function file not found: {tf_path}")
        n_dirs = 24  # Default
        H = None

    # 6. Vocabulary and encoding
    logger.info("\n" + "=" * 60)
    logger.info("6. VOCABULARY AND ENCODING")
    logger.info("=" * 60)

    # Build vocabulary from tokens
    all_tokens = [patch_tokens, leaf_tokens]
    if H is not None:
        all_tokens.append(patch_tokens + dir_tokens)

    vocab = Vocab()
    vocab.build(all_tokens)

    logger.info(f"Vocabulary size: {len(vocab.itos)}")
    logger.info(f"Special tokens: {[vocab.cls_token, vocab.eos_token, vocab.pad_token, vocab.unk_token]}")

    # Encode tokens
    ids = vocab.encode(patch_tokens, add_cls=True)
    logger.info(f"\nEncoding example:")
    logger.info(f"  Original tokens: {len(patch_tokens)} tokens")
    logger.info(f"  Encoded IDs: {len(ids)} IDs (with CLS)")
    logger.info(f"  Sample IDs: {ids[:10]}")

    # Padding
    input_ids, attn_mask = pad_sequences([ids], vocab.pad_id)
    logger.info(f"\nPadding:")
    logger.info(f"  Padded input_ids shape: {input_ids.shape}")
    logger.info(f"  Attention mask shape: {attn_mask.shape}")

    # 7. TransformerPolicy network
    logger.info("\n" + "=" * 60)
    logger.info("7. TRANSFORMER POLICY NETWORK")
    logger.info("=" * 60)

    policy = TransformerPolicy(
        vocab_size=len(vocab.itos),
        n_dirs=n_dirs,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_ff=512,
        dropout=0.1,
        max_len=512
    )

    logger.info(f"Model configuration:")
    logger.info(f"  vocab_size: {len(vocab.itos)}")
    logger.info(f"  n_dirs: {n_dirs}")
    logger.info(f"  d_model: 256")
    logger.info(f"  nhead: 8")
    logger.info(f"  num_layers: 2")
    logger.info(f"  dim_ff: 512")

    # Forward pass with dimension tracking
    logger.info(f"\nForward pass dimensions:")
    B, L = input_ids.shape
    logger.info(f"  Input shape: {input_ids.shape} (B={B}, L={L})")

    with torch.no_grad():
        # Get intermediate outputs by modifying forward pass
        tok_embed = policy.tok_embed(input_ids)
        logger.info(f"  After token embedding: {tok_embed.shape}")

        pos_idx = torch.arange(L, device=input_ids.device) % policy.pos_embed.num_embeddings
        pos = pos_idx.unsqueeze(0).expand(B, L)
        pos_embed = policy.pos_embed(pos)
        x = tok_embed + pos_embed
        logger.info(f"  After position embedding: {x.shape}")

        key_padding_mask = attn_mask.eq(0)
        h = policy.encoder(x, src_key_padding_mask=key_padding_mask)
        logger.info(f"  After transformer encoder: {h.shape}")

        h = policy.norm(h)
        logger.info(f"  After layer norm: {h.shape}")

        denom = attn_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (h * attn_mask.unsqueeze(-1)).sum(dim=1) / denom
        logger.info(f"  After pooling: {pooled.shape}")

        logits = policy.head(pooled)
        logger.info(f"  Final logits: {logits.shape}")

    # 8. Output interpretation
    logger.info("\n" + "=" * 60)
    logger.info("8. OUTPUT INTERPRETATION")
    logger.info("=" * 60)

    probs = torch.softmax(logits, dim=-1)
    predicted_idx = torch.argmax(logits, dim=-1)

    logger.info(f"Output logits shape: {logits.shape}")
    logger.info(f"Probability distribution shape: {probs.shape}")
    logger.info(f"Predicted direction index: {predicted_idx.item()}")

    if H is not None:
        predicted_angle = angles[predicted_idx.item()].item()
        logger.info(f"Predicted angle: {predicted_angle:.1f} degrees")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DIMENSION FLOW SUMMARY")
    logger.info("=" * 60)
    logger.info(f"""
    1. Audio: ({T:,},) samples
    2. STFT: ({len(freqs)}, {len(times)}) = (1025, {len(times)})
    3. Filtered: ({mag_band.shape[0]}, {mag_band.shape[1]}) frequency band
    4. Tokens: {len(patch_tokens)} patch tokens
    5. Encoded: {len(ids)} token IDs
    6. Padded: ({B}, {L}) = {input_ids.shape}
    7. Embedding: ({B}, {L}, 256)
    8. Transformer: ({B}, {L}, 256)
    9. Pooled: ({B}, 256)
    10. Logits: ({B}, {n_dirs}) → prediction
    """)

    return {
        'wav_shape': wav.shape,
        'stft_shape': stft.shape,
        'magnitude_shape': magnitude.shape,
        'filtered_shape': mag_band.shape,
        'n_tokens': len(patch_tokens),
        'vocab_size': len(vocab.itos),
        'input_shape': input_ids.shape,
        'logits_shape': logits.shape,
        'n_dirs': n_dirs
    }

if __name__ == "__main__":
    results = test_dimensions()
    print("\nTest completed successfully!")