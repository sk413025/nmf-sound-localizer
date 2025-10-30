from typing import Any, Dict, List
import torch


class OnPolicyBuffer:
    def __init__(self):
        self.storage: List[Dict[str, torch.Tensor]] = []

    def add(self, features: torch.Tensor = None, logits: torch.Tensor = None, action: torch.Tensor = None,
            logp: torch.Tensor = None, advantage: torch.Tensor = None,
            input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None):
        rec: Dict[str, torch.Tensor] = {}
        if features is not None:
            rec["phi"] = features.detach().cpu()
        if input_ids is not None:
            rec["input_ids"] = input_ids.detach().cpu()
        if attention_mask is not None:
            rec["attention_mask"] = attention_mask.detach().cpu()
        if logits is not None:
            rec["logits_old"] = logits.detach().cpu()
        if action is not None:
            rec["actions"] = action.detach().cpu()
        if logp is not None:
            rec["logp_old"] = logp.detach().cpu()
        if advantage is not None:
            rec["adv"] = advantage.detach().cpu()
        self.storage.append(rec)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        batch = {k: torch.stack([x[k] for x in self.storage], dim=0) for k in self.storage[0].keys()}
        if "adv" in batch:
            adv = batch["adv"]
            batch["adv"] = (adv - adv.mean()) / (adv.std() + 1e-8)
        return batch

    def clear(self):
        self.storage.clear()


class GroupOnPolicyBuffer:
    def __init__(self):
        self.storage: List[Dict[str, Any]] = []

    def add(self, features: torch.Tensor = None, logits: torch.Tensor = None, action: torch.Tensor = None,
            logp: torch.Tensor = None, advantage_raw: torch.Tensor = None, group_id: Any = None,
            input_ids: torch.Tensor = None, attention_mask: torch.Tensor = None):
        rec: Dict[str, Any] = {}
        if features is not None:
            rec["phi"] = features.detach().cpu()
        if input_ids is not None:
            rec["input_ids"] = input_ids.detach().cpu()
        if attention_mask is not None:
            rec["attention_mask"] = attention_mask.detach().cpu()
        if logits is not None:
            rec["logits_old"] = logits.detach().cpu()
        if action is not None:
            rec["actions"] = action.detach().cpu()
        if logp is not None:
            rec["logp_old"] = logp.detach().cpu()
        if advantage_raw is not None:
            rec["adv_raw"] = advantage_raw.detach().cpu()
        if group_id is not None:
            rec["gid"] = group_id
        self.storage.append(rec)

    def to_tensors(self) -> Dict[str, torch.Tensor]:
        from collections import defaultdict
        groups = defaultdict(list)
        for i, r in enumerate(self.storage):
            groups[r["gid"]].append(i)
        adv = torch.zeros(len(self.storage))
        for _, idxs in groups.items():
            vals = torch.stack([self.storage[i]["adv_raw"] for i in idxs])
            mu = vals.mean(); sd = vals.std()
            adv_group = (vals - mu) / (sd + 1e-8)
            for off, i in enumerate(idxs):
                adv[i] = adv_group[off]

        keys = [k for k in ["phi", "input_ids", "attention_mask", "logits_old", "actions", "logp_old"] if k in self.storage[0]]
        batch = {k: torch.stack([r[k] for r in self.storage], dim=0) for k in keys}
        batch["adv"] = adv
        return batch

    def clear(self):
        self.storage.clear()

