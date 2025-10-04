"""Generation-time constraints for direction-only decoding."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor


class DirectionLogitsProcessor(LogitsProcessor):
    """Mask logits so that only direction tokens remain selectable."""

    def __init__(self, allowed_token_ids):
        super().__init__()
        self.allowed = torch.tensor(sorted(set(allowed_token_ids)), dtype=torch.long)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.allowed.device != scores.device:
            self.allowed = self.allowed.to(scores.device)
        mask_value = torch.finfo(scores.dtype).min
        masked = torch.full_like(scores, mask_value)
        masked[:, self.allowed] = scores[:, self.allowed]
        return masked

