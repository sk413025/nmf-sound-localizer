"""Patch-token aware Hugging Face tokenizer construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

# Patch grid configuration matches PatchTokenizer defaults (Fp=16, Np=10 on 116x189 spectrogram)
_PATCH_FREQ_BANDS = 7
_PATCH_TIME_WINDOWS = 18
_PATCH_LEVELS = 16

# Special tokens for causal LM usage
_PAD_TOKEN = "<PAD>"
_BOS_TOKEN = "<BOS>"
_EOS_TOKEN = "<EOS>"
_UNK_TOKEN = "<UNK>"


def _generate_patch_tokens() -> Iterable[str]:
    for i in range(_PATCH_FREQ_BANDS):
        for j in range(_PATCH_TIME_WINDOWS):
            for level in range(_PATCH_LEVELS):
                yield f"<P_{i}_{j}_{level}>"


def _generate_nmf_atom_tokens(n_atoms: int = 64) -> Iterable[str]:
    """Generate NMF atom tokens in format <AT_atom_id:level>."""
    for atom_id in range(n_atoms):
        for level in range(_PATCH_LEVELS):
            yield f"<AT_{atom_id}:{level}>"


def _generate_direction_projection_tokens() -> Iterable[str]:
    """Generate direction projection tokens in format <R_angle:level>.
    
    Covers angles 0-180 in 5-degree increments (37 angles total).
    """
    for angle in range(0, 181, 5):
        for level in range(_PATCH_LEVELS):
            yield f"<R_{angle:03d}:{level}>"


def _build_vocab(
    direction_tokens: Sequence[str],
    enable_extended: bool = False,
    n_atoms: int = 64,
) -> List[str]:
    """Build vocabulary with optional multi-modal tokens.
    
    Args:
        direction_tokens: Direction tokens <D_angle> from training angles
        enable_extended: If True, include NMF atom and direction projection tokens
        n_atoms: Number of NMF atoms (only used if enable_extended=True)
    
    Returns:
        List of all vocabulary tokens
    """
    vocab: List[str] = []
    
    # Special tokens
    vocab.extend([_PAD_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _UNK_TOKEN])
    
    # Patch tokens (always included)
    vocab.extend(_generate_patch_tokens())
    
    # Extended multi-modal tokens (optional)
    if enable_extended:
        # NMF Atom tokens: <AT_atom_id:level>
        vocab.extend(_generate_nmf_atom_tokens(n_atoms))
        
        # Direction Projection tokens: <R_angle:level>
        vocab.extend(_generate_direction_projection_tokens())
    
    # Direction tokens: <D_angle>
    vocab.extend(direction_tokens)
    
    return vocab


def build_patch_tokenizer(
    direction_angles: Sequence[float],
    save_dir: Path | None = None,
    enable_extended_vocab: bool = False,
    n_atoms: int = 64,
) -> PreTrainedTokenizerFast:
    """Create (and optionally persist) a WordLevel tokenizer for patch + direction tokens.
    
    Args:
        direction_angles: Sequence of direction angles for <D_angle> tokens
        save_dir: Optional directory to save tokenizer files
        enable_extended_vocab: If True, include multi-modal tokens (NMF atoms, direction projections)
        n_atoms: Number of NMF atoms (only used if enable_extended_vocab=True)
    
    Returns:
        PreTrainedTokenizerFast tokenizer with configured vocabulary
        
    Notes:
        Extended vocabulary adds:
        - NMF Atom tokens: <AT_atom_id:level> (n_atoms × 16 levels)
        - Direction Projection tokens: <R_angle:level> (37 angles × 16 levels)
        Total expansion: ~1,600 tokens (64 atoms × 16 + 37 angles × 16)
    """

    direction_angles = sorted({int(round(angle)) for angle in direction_angles})
    direction_tokens = tuple(f"<D_{angle:03d}>" for angle in direction_angles)
    vocab_list = _build_vocab(
        direction_tokens,
        enable_extended=enable_extended_vocab,
        n_atoms=n_atoms,
    )
    vocab = {tok: idx for idx, tok in enumerate(vocab_list)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=_UNK_TOKEN))
    tokenizer.pre_tokenizer = WhitespaceSplit()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=_BOS_TOKEN,
        eos_token=_EOS_TOKEN,
        pad_token=_PAD_TOKEN,
        unk_token=_UNK_TOKEN,
    )
    # Set a sane finite max length to avoid OverflowError in fast tokenizers
    # when downstream code uses max_length=tokenizer.model_max_length.
    # Patch/dir prompts are well below this bound.
    hf_tokenizer.model_max_length = 8192
    hf_tokenizer.padding_side = "left"

    # Attach angle metadata for downstream helpers
    hf_tokenizer.direction_tokens = direction_tokens
    hf_tokenizer.direction_token_to_angle = {
        token: angle for token, angle in zip(direction_tokens, direction_angles)
    }

    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        hf_tokenizer.save_pretrained(save_path)

    return hf_tokenizer


def _get_direction_mapping(tokenizer: PreTrainedTokenizerFast) -> Dict[str, float]:
    mapping = getattr(tokenizer, "direction_token_to_angle", None)
    if mapping is None:
        raise ValueError("Tokenizer is missing direction token metadata; rebuild with build_patch_tokenizer().")
    return mapping


def direction_token_to_angle(token_id: int, tokenizer: PreTrainedTokenizerFast) -> float:
    token = tokenizer.convert_ids_to_tokens(token_id)
    mapping = _get_direction_mapping(tokenizer)
    if token not in mapping:
        raise ValueError(f"Token id {token_id} ('{token}') is not a registered direction token")
    return mapping[token]


def direction_token_ids(tokenizer: PreTrainedTokenizerFast) -> Sequence[int]:
    mapping = _get_direction_mapping(tokenizer)
    return tokenizer.convert_tokens_to_ids(list(mapping.keys()))
