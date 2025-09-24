from typing import Dict
import torch
import torch.nn.functional as F

from doa_rl.model.transformer import TransformerPolicy


class PPOTrainer:
    def __init__(self, policy: TransformerPolicy, lr: float = 1e-3,
                 clip_eps: float = 0.2, target_kl: float = 0.02,
                 entropy_coef: float = 0.0, device: str = "cpu"):
        self.policy = policy.to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.entropy_coef = entropy_coef
        self.device = device

    def update(self, batch: Dict[str, torch.Tensor], epochs: int = 4) -> Dict[str, float]:
        actions = batch["actions"].to(self.device)
        logp_old = batch["logp_old"].to(self.device)
        adv = batch["adv"].to(self.device)

        is_text = ("input_ids" in batch)
        if is_text:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            logits_old = batch["logits_old"].to(self.device)
        else:
            phi = batch["phi"].to(self.device)
            logits_old = batch["logits_old"].to(self.device)

        info_last = {}
        for _ in range(epochs):
            if is_text:
                logits = self.policy(input_ids, attention_mask)
                dist = torch.distributions.Categorical(logits=logits)
            else:
                logits = self.policy(phi)
                dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(actions)
            ratio = torch.exp(logp - logp_old)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            entropy = torch.mean(dist.entropy())
            kl = torch.mean(TransformerPolicy.kl_divergence(logits_old, logits))
            loss = policy_loss - self.entropy_coef * entropy

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()

            info_last = {
                "loss": loss.item(),
                "policy_loss": policy_loss.item(),
                "entropy": float(entropy.item()),
                "kl": float(kl.item()),
            }
            if kl.item() > self.target_kl:
                break
        return info_last

