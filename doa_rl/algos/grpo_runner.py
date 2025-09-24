from typing import List, Dict
import re
import time
import logging
import numpy as np
import torch

from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from doa_rl.training.buffer import GroupOnPolicyBuffer
from doa_rl.training.grpo_trainer import GRPOTrainer
from doa_rl.advantage import AdvantageComputer


_TOKEN_RE = re.compile(r"<[^>]+>")
logger = logging.getLogger(__name__)


def _extract_tokens(prompt: str) -> List[str]:
    return _TOKEN_RE.findall(prompt)


class GRPORunner:
    def __init__(self, dir_vocab: List[str]):
        self.dir_vocab = dir_vocab

    def train(self, ds, exs: List[Dict], dir_vocab: List[str], J: int = 1, G: int = 4,
              d_model: int = 256, nhead: int = 8, layers: int = 2,
              adv: AdvantageComputer = None, lr: float = 1e-3,
              beta_kl: float = 0.01, clip_eps: float = 0.2, epochs: int = 4) -> Dict[str, float]:
        start_time = time.time()
        logger.info("GRPORunner: building vocab from %d samples (G=%d)", len(ds), G)
        token_lists = [[* _extract_tokens(r["prompt"]) ] for r in ds]
        vocab = Vocab(); vocab.build(token_lists)
        logger.info("GRPORunner: vocab_size=%d, n_dirs=%d, epochs=%d", len(vocab.itos), len(dir_vocab), epochs)
        policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=len(dir_vocab),
                                   d_model=d_model, nhead=nhead, num_layers=layers)
        trainer = GRPOTrainer(policy, lr=lr, clip_eps=clip_eps, beta_kl=beta_kl)

        buf = GroupOnPolicyBuffer()
        last_info: Dict[str, float] = {}
        for epoch in range(epochs):
            t0 = time.time()
            buf.clear()
            for j, (sample, extra) in enumerate(zip(ds, exs)):
                ids = vocab.encode(_extract_tokens(sample["prompt"]))
                input_ids, attn = pad_sequences([ids], vocab.pad_id)
                Y = extra["Y"]
                if isinstance(Y, np.ndarray):
                    Y_t = torch.from_numpy(np.asarray(Y, dtype=np.float32))
                else:
                    Y_t = Y
                out = adv(Y_t) if adv is not None else {"A": torch.zeros(len(dir_vocab))}
                A = out["A"]
                logits_old = None
                for _ in range(G):
                    with torch.no_grad():
                        logits = policy(input_ids, attn)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        logp = dist.log_prob(action)
                        adv_val = A[action.item()].view(1)
                    if logits_old is None:
                        logits_old = logits
                    buf.add(logits=logits_old.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                            advantage_raw=adv_val.squeeze(0), group_id=extra.get('path','unknown'),
                            input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))
            batch = buf.to_tensors()
            last_info = trainer.update(batch, epochs=epochs)
            logger.info("GRPORunner: epoch=%d policy_loss=%.6f kl=%.6f elapsed=%.2fs",
                        epoch, float(last_info.get("policy_loss", 0.0)), float(last_info.get("kl", 0.0)), time.time() - t0)
        logger.info("GRPORunner finished in %.2fs", time.time() - start_time)
        return last_info
