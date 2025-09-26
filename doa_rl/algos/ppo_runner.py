from typing import List, Dict, Tuple
import re
import time
import logging
import numpy as np
import torch

from doa_rl.model.transformer import TransformerPolicy
from doa_rl.text import Vocab, pad_sequences
from doa_rl.training.buffer import OnPolicyBuffer
from doa_rl.training.ppo_trainer import PPOTrainer
from doa_rl.advantage import AdvantageComputer


_TOKEN_RE = re.compile(r"<[^>]+>")
logger = logging.getLogger(__name__)


def _extract_tokens(prompt: str) -> List[str]:
    return _TOKEN_RE.findall(prompt)


class PPORunner:
    def __init__(self, dir_vocab: List[str]):
        self.dir_vocab = dir_vocab

    def train(self, ds, exs: List[Dict], dir_vocab: List[str], J: int = 1,
              d_model: int = 256, nhead: int = 8, layers: int = 2,
              adv: AdvantageComputer = None, lr: float = 1e-3,
              clip_eps: float = 0.2, target_kl: float = 0.02, entropy_coef: float = 0.0,
              epochs: int = 4) -> Dict[str, float]:
        start_time = time.time()
        logger.info("PPORunner: building vocab from %d samples", len(ds))
        token_lists = [[* _extract_tokens(r["prompt"]) ] for r in ds]
        vocab = Vocab(); vocab.build(token_lists)
        logger.info("PPORunner: vocab_size=%d, n_dirs=%d, epochs=%d", len(vocab.itos), len(dir_vocab), epochs)
        policy = TransformerPolicy(vocab_size=len(vocab.itos), n_dirs=len(dir_vocab),
                                   d_model=d_model, nhead=nhead, num_layers=layers)
        trainer = PPOTrainer(policy, lr=lr, clip_eps=clip_eps, target_kl=target_kl, entropy_coef=entropy_coef)

        buf = OnPolicyBuffer()
        last_info: Dict[str, float] = {}
        for epoch in range(epochs):
            t0 = time.time()
            buf.clear()
            adv_action_raw_vals = []
            for i, (sample, extra) in enumerate(zip(ds, exs)):
                ids = vocab.encode(_extract_tokens(sample["prompt"]))
                input_ids, attn = pad_sequences([ids], vocab.pad_id)
                # Compute advantage per sample using provided AdvantageComputer
                Y = extra["Y"]
                if isinstance(Y, np.ndarray):
                    Y_t = torch.from_numpy(Y.astype("float32"))
                else:
                    Y_t = Y
                with torch.no_grad():
                    logits = policy(input_ids, attn)
                    dist = torch.distributions.Categorical(logits=logits)
                    pi_vec = torch.softmax(logits, dim=-1).squeeze(0).cpu()
                out = adv(Y_t, pi=pi_vec) if adv is not None else {"A": torch.zeros(len(dir_vocab))}
                A = out["A"]
                if logger.isEnabledFor(logging.DEBUG) and i == 0:
                    try:
                        y_np = Y if isinstance(Y, np.ndarray) else Y.detach().cpu().numpy()
                        logger.debug(
                            "PPORunner[epoch %d]: first-sample Y stats: shape=%s min=%.4g max=%.4g mean=%.4g std=%.4g",
                            epoch, tuple(y_np.shape), float(np.min(y_np)), float(np.max(y_np)), float(np.mean(y_np)), float(np.std(y_np))
                        )
                        a_np = A.detach().cpu().numpy()
                        logger.debug(
                            "PPORunner[epoch %d]: first A stats: D=%d min=%.4g max=%.4g mean=%.4g std=%.4g neg_frac=%.3f",
                            epoch, a_np.size, float(a_np.min()), float(a_np.max()), float(a_np.mean()), float(a_np.std()), float((a_np < 0).mean())
                        )
                    except Exception:
                        pass
                with torch.no_grad():
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    adv_val = A[action.item()].view(1)
                    adv_action_raw_vals.append(float(adv_val.item()))
                buf.add(logits=logits.squeeze(0), action=action.squeeze(0), logp=logp.squeeze(0),
                        advantage=adv_val.squeeze(0), input_ids=input_ids.squeeze(0), attention_mask=attn.squeeze(0))
                if (i + 1) % 10 == 0:
                    logger.debug("PPORunner: buffered %d samples", i + 1)
            batch = buf.to_tensors()
            if logger.isEnabledFor(logging.DEBUG):
                try:
                    arr = np.asarray(adv_action_raw_vals, dtype=np.float32)
                    logger.debug(
                        "PPORunner[epoch %d]: raw advantage(action) stats before norm: n=%d min=%.4g max=%.4g mean=%.4g std=%.4g",
                        epoch, arr.size, float(arr.min()) if arr.size else float('nan'), float(arr.max()) if arr.size else float('nan'), float(arr.mean()) if arr.size else float('nan'), float(arr.std()) if arr.size else float('nan')
                    )
                    adv_std = batch.get("adv")
                    if adv_std is not None:
                        adv_np = adv_std.detach().cpu().numpy()
                        logger.debug(
                            "PPORunner[epoch %d]: standardized advantage stats: n=%d min=%.4g max=%.4g mean=%.4g std=%.4g",
                            epoch, adv_np.size, float(adv_np.min()), float(adv_np.max()), float(adv_np.mean()), float(adv_np.std())
                        )
                except Exception:
                    pass
            last_info = trainer.update(batch, epochs=epochs)
            logger.info("PPORunner: epoch=%d policy_loss=%.6f kl=%.6f elapsed=%.2fs",
                        epoch, float(last_info.get("policy_loss", 0.0)), float(last_info.get("kl", 0.0)), time.time() - t0)
        logger.info("PPORunner finished in %.2fs", time.time() - start_time)
        return last_info
