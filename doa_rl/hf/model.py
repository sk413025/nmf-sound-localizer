"""Causal LM builder aligned with our patch-token policy dimensions."""

from __future__ import annotations

from typing import Tuple, Optional

from copy import deepcopy
from dataclasses import dataclass

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerBase
from transformers.utils import ModelOutput
from trl import AutoModelForCausalLMWithValueHead

import torch
from types import MethodType


@dataclass
class ValueHeadModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple] = None
    value: Optional[torch.FloatTensor] = None


def _patch_value_head_forward(model: AutoModelForCausalLMWithValueHead) -> AutoModelForCausalLMWithValueHead:
    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        kwargs.setdefault("return_dict", True)
        kwargs["output_hidden_states"] = True
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = outputs.hidden_states
        last_hidden = hidden_states[-1]
        value = self.v_head(last_hidden).squeeze(-1)
        logits = outputs.logits
        loss = getattr(outputs, "loss", None)
        return ValueHeadModelOutput(
            logits=logits,
            loss=loss,
            hidden_states=hidden_states,
            past_key_values=getattr(outputs, "past_key_values", None),
            value=value,
        )

    model.forward = MethodType(forward, model)
    model.is_gradient_checkpointing = False
    model.supports_gradient_checkpointing = False
    return model


def build_value_head_model(
    tokenizer: PreTrainedTokenizerBase,
    d_model: int = 256,
    n_layer: int = 2,
    n_head: int = 8,
    n_positions: int = 512,
) -> Tuple[AutoModelForCausalLMWithValueHead, AutoModelForCausalLMWithValueHead]:
    """Create policy and frozen reference models sharing initial weights."""

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=d_model,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    base_policy = GPT2LMHeadModel(config)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(base_policy)
    policy.generation_config = policy.pretrained_model.generation_config
    policy.base_model_prefix = "pretrained_model"
    policy = _patch_value_head_forward(policy)
    policy.score = policy.v_head

    reference = deepcopy(policy)
    reference.requires_grad_(False)

    return policy, reference
