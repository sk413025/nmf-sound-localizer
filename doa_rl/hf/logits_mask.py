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


class NoRepeatDirectionLogitsProcessor(DirectionLogitsProcessor):
    """Mask logits to allow only unseen direction tokens from the allowed set."""

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if self.allowed.device != scores.device:
            self.allowed = self.allowed.to(scores.device)
        mask_value = torch.finfo(scores.dtype).min
        masked = torch.full_like(scores, mask_value)

        batch_size = scores.shape[0]
        for i in range(batch_size):
            row_tokens = input_ids[i]
            seen_mask = torch.isin(row_tokens, self.allowed)
            seen = row_tokens[seen_mask]
            if seen.numel() > 0:
                seen_unique = torch.unique(seen)
                allowed_mask = ~torch.isin(self.allowed, seen_unique)
                allowed_ids = self.allowed[allowed_mask]
                if allowed_ids.numel() == 0:
                    allowed_ids = self.allowed
            else:
                allowed_ids = self.allowed
            masked[i, allowed_ids] = scores[i, allowed_ids]
        return masked
