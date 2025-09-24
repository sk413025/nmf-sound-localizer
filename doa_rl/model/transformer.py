from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerPolicy(nn.Module):
    """Token-level Transformer encoder with D-way classification head.

    Inputs: input_ids (B, L), attention_mask (B, L)
    Output: logits (B, D)
    """

    def __init__(self, vocab_size: int, n_dirs: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 2, dim_ff: int = 512, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.n_dirs = n_dirs
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_ff, dropout=dropout,
                                               batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_dirs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        pos_idx = torch.arange(L, device=input_ids.device) % self.pos_embed.num_embeddings
        pos = pos_idx.unsqueeze(0).expand(B, L)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        key_padding_mask = attention_mask.eq(0)
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (h * attention_mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(pooled)
        return logits

    def dist(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(input_ids, attention_mask)
        return torch.distributions.Categorical(logits=logits)

    def act(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        d = self.dist(input_ids, attention_mask)
        a = d.sample()
        return a, d.log_prob(a)

    @staticmethod
    def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
        p = F.softmax(logits_p, dim=-1)
        log_p = F.log_softmax(logits_p, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        kl = torch.sum(p * (log_p - log_q), dim=-1)
        return kl

