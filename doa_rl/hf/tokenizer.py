"""Patch-token aware Hugging Face tokenizer construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
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


def _build_vocab(direction_tokens: Sequence[str]) -> List[str]:
    vocab: List[str] = []
    vocab.extend([_PAD_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _UNK_TOKEN])
    vocab.extend(_generate_patch_tokens())
    vocab.extend(direction_tokens)
    return vocab


def build_patch_tokenizer(
    direction_angles: Sequence[float],
    save_dir: Path | None = None,
) -> PreTrainedTokenizerFast:
    """Create (and optionally persist) a WordLevel tokenizer for patch + direction tokens."""

    direction_angles = sorted({int(round(angle)) for angle in direction_angles})
    direction_tokens = tuple(f"<D_{angle:03d}>" for angle in direction_angles)
    vocab_list = _build_vocab(direction_tokens)
    vocab = {tok: idx for idx, tok in enumerate(vocab_list)}

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=_UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=_BOS_TOKEN,
        eos_token=_EOS_TOKEN,
        pad_token=_PAD_TOKEN,
        unk_token=_UNK_TOKEN,
    )

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
