#!/usr/bin/env python3
"""Train DTMin (Decision Transformer Minimum) on angle-range shards.

DTMin learns to imitate OMP teacher's hierarchical trajectory decisions:
- Input: Residual embeddings h_seq from OMP steps
- Supervision: Teacher's (expert, atom) choices at each step
- Training: BC (Behavioral Cloning) or AWR (Advantage-Weighted Regression)
- Evaluation: Joint accuracy (trajectory matching) + Voted accuracy (angle classification)

NOTE: This is TRUE DTMin - imitates teacher trajectory, not direct angle classification.
For direct angle supervision, use label_mode='angle' (non-standard, see ARCHITECTURE_VIOLATION.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from doa_rl.domain_randomization import AngleRangeDataset, load_shard_summary

LOG = logging.getLogger("train_angle_range_dtmin")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    h_seq = torch.stack([b["h_seq"] for b in batch])
    expert_gt = torch.stack([b["expert_gt"] for b in batch])
    atom_gt = torch.stack([b["atom_gt"] for b in batch])
    angle_gt = torch.tensor([b["angle_gt"] for b in batch], dtype=torch.long)
    result = {"h_seq": h_seq, "expert_gt": expert_gt, "atom_gt": atom_gt, "angle_gt": angle_gt}
    # Include RTG sequences if available (for Physics-Informed DT)
    if "rtg_seq" in batch[0] and batch[0]["rtg_seq"] is not None:
        result["rtg_seq"] = torch.stack([b["rtg_seq"] for b in batch])
    if "step_seq" in batch[0] and batch[0]["step_seq"] is not None:
        result["step_seq"] = torch.stack([b["step_seq"] for b in batch])
    if "actual_length" in batch[0]:
        result["actual_length"] = torch.tensor([b["actual_length"] for b in batch], dtype=torch.long)
    if "angle_deg" in batch[0] and batch[0]["angle_deg"] is not None:
        result["angle_deg"] = torch.tensor([b["angle_deg"] for b in batch], dtype=torch.float)
    return result


class TinyDTMin(nn.Module):
    """DTMin: Decision Transformer Minimum for OMP trajectory imitation.

    Architecture:
    - TransformerEncoder: Process residual sequence embeddings
    - Expert head: Predict which angle/expert to select (E-way classification)
    - Atom head: Predict which atom within expert to select (M-way classification)

    Training objective (label_mode='teacher'):
    - Expert loss: CE(expert_pred, expert_omp)  # Imitate teacher's expert choice
    - Atom loss: CE(atom_pred, atom_omp)        # Imitate teacher's atom choice

    Evaluation:
    - Joint accuracy: P(expert=teacher AND atom=teacher)  # Trajectory matching
    - Voted accuracy: majority_vote(experts) == angle_gt  # Localization quality

    Physics-Informed DT extensions:
    - RTG projection: Condition on return-to-go for controllable generation
    - Causal mask: Enforce sequential decision making (position i can only attend to j <= i)
    """

    def __init__(self, d_model: int, nhead: int, nlayers: int, E: int, M: int, max_K: int,
                 dropout: float = 0.1, rtg_dim: int = 2, use_rtg: bool = False):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos = nn.Embedding(max_K, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.expert_head = nn.Linear(d_model, E)
        self.atom_head = nn.Linear(d_model, M)

        # Physics-Informed DT extensions
        self.use_rtg = use_rtg
        self.max_K = max_K
        if use_rtg:
            self.rtg_proj = nn.Sequential(
                nn.Linear(rtg_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

    def _generate_causal_mask(self, K: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask.

        Returns: (K, K) mask where mask[i,j] = True means position i
                 cannot attend to position j (j > i is masked)
        """
        mask = torch.triu(torch.ones(K, K, device=device), diagonal=1).bool()
        return mask

    def forward(self, h_seq: torch.Tensor, rtg_seq: torch.Tensor = None,
                use_causal_mask: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = h_seq.shape
        device = h_seq.device

        # Positional embedding
        pos_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        h = h_seq + self.pos(pos_idx)

        # RTG conditioning (Physics-Informed DT)
        if self.use_rtg and rtg_seq is not None:
            rtg_embed = self.rtg_proj(rtg_seq)
            h = h + rtg_embed

        # Causal mask for sequential decision making
        mask = self._generate_causal_mask(K, device) if use_causal_mask else None

        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        expert_logits = self.expert_head(h)
        atom_logits = self.atom_head(h)
        return expert_logits, atom_logits


def compute_targets(batch: Dict[str, torch.Tensor], label_mode: str) -> torch.Tensor:
    """Compute supervision targets for expert head.
    
    label_mode='teacher' (DTMin standard): Imitate OMP teacher's expert choices
    label_mode='angle' (non-standard): Direct angle supervision (see ARCHITECTURE_VIOLATION.md)
    """
    if label_mode == "angle":
        # WARNING: This bypasses trajectory imitation - NOT true DTMin!
        angle = batch["angle_gt"]
        B, K, _ = batch["h_seq"].shape
        return angle.view(B, 1).expand(-1, K)
    # Standard DTMin: supervise with teacher's expert choices
    return batch["expert_gt"]


def _mask_from_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    B, K, _ = batch["h_seq"].shape
    if "actual_length" in batch:
        lengths = torch.as_tensor(batch["actual_length"], device=device)
        arange = torch.arange(K, device=device).unsqueeze(0)
        return arange < lengths.unsqueeze(1)
    return torch.ones((B, K), device=device, dtype=torch.bool)


def compute_metrics(expert_logits: torch.Tensor, atom_logits: torch.Tensor, batch: Dict[str, torch.Tensor], label_mode: str) -> Dict[str, float]:
    target_expert = compute_targets(batch, label_mode)
    expert_pred = expert_logits.argmax(dim=-1)
    atom_pred = atom_logits.argmax(dim=-1)
    mask = _mask_from_batch(batch, expert_logits.device)
    atom_mask = mask & (batch["atom_gt"] != -100)

    expert_acc = ((expert_pred == target_expert) & mask).float().sum().item() / mask.sum().clamp(min=1).item()
    atom_acc = ((atom_pred == batch["atom_gt"]) & atom_mask).float().sum().item() / atom_mask.sum().clamp(min=1).item()
    joint_acc = ((expert_pred == target_expert) & (atom_pred == batch["atom_gt"]) & atom_mask).float().sum().item() / atom_mask.sum().clamp(min=1).item()
    voted_preds = []
    lengths = batch.get("actual_length", None)
    for i in range(expert_pred.shape[0]):
        length_i = int(lengths[i].item()) if lengths is not None else expert_pred.shape[1]
        counts = torch.bincount(expert_pred[i, :length_i], minlength=expert_logits.shape[-1])
        voted_preds.append(int(torch.argmax(counts).item()))
    voted_preds = torch.tensor(voted_preds, device=expert_pred.device)
    voted_gt = batch["angle_gt"]
    voted_acc = (voted_preds == voted_gt).float().mean().item()
    metrics = {
        "expert_acc": expert_acc,
        "atom_acc": atom_acc,
        "joint_acc": joint_acc,
        "voted_acc": voted_acc,
    }
    return metrics


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, mode: str, awr_beta: float, awr_w_max: float, awr_normalize: bool, label_mode: str, use_atom_loss: bool, angle_weights: Dict[float, float] | None, use_rtg: bool = False, use_causal_mask: bool = False) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        h_seq = batch["h_seq"].to(device)
        expert_gt = batch["expert_gt"].to(device)
        atom_gt = batch["atom_gt"].to(device)
        angle_gt = batch["angle_gt"].to(device)
        # RTG conditioning for Physics-Informed DT
        rtg_seq = batch.get("rtg_seq")
        if rtg_seq is not None and use_rtg:
            rtg_seq = rtg_seq.to(device)
        else:
            rtg_seq = None
        target_expert = compute_targets({"h_seq": h_seq, "expert_gt": expert_gt, "angle_gt": angle_gt}, label_mode)
        mask = _mask_from_batch({"h_seq": h_seq, **batch}, device)
        atom_mask = mask & (atom_gt != -100)

        optimizer.zero_grad()
        expert_logits, atom_logits = model(h_seq, rtg_seq=rtg_seq, use_causal_mask=use_causal_mask)

        expert_ce = F.cross_entropy(
            expert_logits.transpose(1, 2), target_expert, reduction="none", ignore_index=-100
        )
        expert_loss_per_sample = (expert_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if use_atom_loss:
            atom_ce = F.cross_entropy(
                atom_logits.transpose(1, 2), atom_gt, reduction="none", ignore_index=-100
            )
            atom_loss_per_sample = (atom_ce * atom_mask).sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1)
        else:
            atom_loss_per_sample = torch.zeros_like(expert_loss_per_sample)
        loss_per_sample = expert_loss_per_sample + atom_loss_per_sample
        if angle_weights:
            w = torch.tensor(
                [angle_weights.get(float(a.item()), 1.0) for a in batch.get("angle_deg", batch["angle_gt"])],
                device=device,
                dtype=loss_per_sample.dtype,
            )
            loss_per_sample = loss_per_sample * w

        if mode == "awr":
            B, K = expert_gt.shape
            angle_expanded = angle_gt.view(B, 1).expand(-1, K)
            returns = ((target_expert == angle_expanded) & mask).float().sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            baseline = returns.mean().detach()
            adv = returns - baseline
            weights = torch.exp(adv / awr_beta).clamp(max=awr_w_max)
            if awr_normalize:
                weights = weights / (weights.mean() + 1e-8)
            loss = (loss_per_sample * weights).mean()
        else:
            loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label_mode: str, use_atom_loss: bool, angle_weights: Dict[float, float] | None, use_rtg: bool = False, use_causal_mask: bool = False) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    metrics_accum: Dict[str, float] = Counter()
    for batch in loader:
        h_seq = batch["h_seq"].to(device)
        expert_gt = batch["expert_gt"].to(device)
        atom_gt = batch["atom_gt"].to(device)
        angle_gt = batch["angle_gt"].to(device)
        # RTG conditioning for Physics-Informed DT
        rtg_seq = batch.get("rtg_seq")
        if rtg_seq is not None and use_rtg:
            rtg_seq = rtg_seq.to(device)
        else:
            rtg_seq = None
        target_expert = compute_targets({"h_seq": h_seq, "expert_gt": expert_gt, "angle_gt": angle_gt}, label_mode)
        mask = _mask_from_batch({"h_seq": h_seq, **batch}, device)
        atom_mask = mask & (atom_gt != -100)
        expert_logits, atom_logits = model(h_seq, rtg_seq=rtg_seq, use_causal_mask=use_causal_mask)
        expert_ce = F.cross_entropy(
            expert_logits.transpose(1, 2), target_expert, reduction="none", ignore_index=-100
        )
        expert_loss_per_sample = (expert_ce * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if use_atom_loss:
            atom_ce = F.cross_entropy(
                atom_logits.transpose(1, 2), atom_gt, reduction="none", ignore_index=-100
            )
            atom_loss_per_sample = (atom_ce * atom_mask).sum(dim=1) / atom_mask.sum(dim=1).clamp_min(1)
        else:
            atom_loss_per_sample = torch.zeros_like(expert_loss_per_sample)
        loss_per_sample = expert_loss_per_sample + atom_loss_per_sample
        if angle_weights:
            w = torch.tensor(
                [angle_weights.get(float(a.item()), 1.0) for a in batch.get("angle_deg", batch["angle_gt"])],
                device=device,
                dtype=loss_per_sample.dtype,
            )
            loss_per_sample = loss_per_sample * w
        loss = loss_per_sample.mean()
        total_loss += loss.item()
        total_batches += 1
        batch_on_device = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        batch_on_device["h_seq"] = h_seq
        batch_on_device["expert_gt"] = expert_gt
        batch_on_device["atom_gt"] = atom_gt
        batch_on_device["angle_gt"] = angle_gt
        m = compute_metrics(expert_logits, atom_logits, batch_on_device, label_mode)
        for k, v in m.items():
            metrics_accum[k] += v
    avg_metrics = {k: v / max(total_batches, 1) for k, v in metrics_accum.items()}
    avg_metrics["loss"] = total_loss / max(total_batches, 1)
    return avg_metrics


def expected_angles(shard_dirs: Sequence[Path]) -> List[int]:
    angles: List[int] = []
    for d in shard_dirs:
        summary = load_shard_summary(d)
        angles.extend(summary.angles)
    return sorted(set(angles))


def parse_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train DTMin (Decision Transformer Minimum) on angle-range shards. Supports BC (Behavioral Cloning) or AWR (Advantage-Weighted Regression) for OMP trajectory imitation.")
    ap.add_argument("--shard-root", type=Path, default=Path("results/angle_range_shards"), help="Root containing shard subdirs.")
    ap.add_argument("--shard-names", type=str, default="full,low,mid,high", help="Comma-separated shard names to load.")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--mode", type=str, default="bc", choices=["bc", "awr"], help="Training mode: 'bc' (standard imitation) or 'awr' (advantage-weighted).")
    ap.add_argument("--awr-beta", type=float, default=0.5)
    ap.add_argument("--awr-w-max", type=float, default=4.0)
    ap.add_argument("--awr-normalize", action="store_true")
    ap.add_argument("--expected-voted", type=float, default=0.6, help="Threshold to flag low voted accuracy.")
    ap.add_argument("--require-voted-threshold", action="store_true", help="Raise error if voted accuracy is below threshold.")
    ap.add_argument("--label-mode", type=str, default="teacher", choices=["angle", "teacher"], 
                    help="DTMin supervision mode. 'teacher' (default): imitate OMP trajectory. 'angle': direct angle classification (non-standard, see ARCHITECTURE_VIOLATION.md).")
    ap.add_argument("--disable-atom-loss", action="store_true",
                    help="Skip atom supervision. WARNING: Disabling breaks DTMin's hierarchical learning - only use when teacher atoms are severely unreliable.")
    ap.add_argument("--log-file", type=Path, default=None, help="Optional JSON log output.")
    ap.add_argument("--angle-weight-json", type=Path, default=None, help="Optional JSON mapping angle_deg -> weight (float). Loss will be scaled per sample.")
    # Physics-Informed DT extensions
    ap.add_argument("--use-rtg", action="store_true",
                    help="Enable RTG (Return-to-Go) conditioning for Physics-Informed DT mode.")
    ap.add_argument("--use-causal-mask", action="store_true",
                    help="Use causal attention mask for sequential decision making.")
    ap.add_argument("--rtg-dim", type=int, default=2,
                    help="RTG embedding dimension (default: 2 for [cumulative_reward, remaining_steps]).")
    return ap


def main() -> None:
    args = parse_args().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    angle_weights = None
    if args.angle_weight_json:
        angle_weights = {float(k): float(v) for k, v in json.loads(Path(args.angle_weight_json).read_text()).items()}

    shard_dirs = [args.shard_root / name for name in args.shard_names.split(",") if name.strip()]
    for d in shard_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Shard dir not found: {d}")

    dataset = AngleRangeDataset(shard_dirs)
    N, K, d_model_detected = dataset.shape_spec()
    coverage = dataset.coverage()
    expected = expected_angles(shard_dirs)
    missing_angles = [a for a in expected if a not in coverage]
    LOG.info("="*80)
    LOG.info("DTMin Training Configuration")
    LOG.info("="*80)
    LOG.info("Dataset loaded: N=%d K=%d d_model=%d shards=%s", N, K, d_model_detected, [d.name for d in shard_dirs])
    LOG.info("Coverage: angles=%d, missing=%s", len(coverage), missing_angles)
    LOG.info("Training mode: %s", args.mode.upper())
    LOG.info("Supervision: label_mode='%s' %s", args.label_mode,
             "(TRUE DTMin - trajectory imitation)" if args.label_mode == "teacher" else "(NON-STANDARD - direct angle supervision)")
    LOG.info("Atom loss: %s %s", "ENABLED" if not args.disable_atom_loss else "DISABLED",
             "" if not args.disable_atom_loss else "(WARNING: Breaks hierarchical learning!)")
    LOG.info("Physics-Informed DT: RTG=%s, Causal Mask=%s",
             "ENABLED" if args.use_rtg else "DISABLED",
             "ENABLED" if args.use_causal_mask else "DISABLED")
    LOG.info("="*80)
    if missing_angles:
        raise ValueError(f"Missing angles in dataset: {missing_angles}")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = max(len(dataset) - val_size, 1)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    E = int(torch.max(torch.stack([sample["expert_gt"].max() for sample in dataset.samples])).item()) + 1
    M = int(torch.max(torch.stack([sample["atom_gt"].max() for sample in dataset.samples])).item()) + 1

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_atom_loss = not args.disable_atom_loss
    model = TinyDTMin(
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.layers,
        E=E,
        M=M,
        max_K=K,
        dropout=args.dropout,
        rtg_dim=args.rtg_dim,
        use_rtg=args.use_rtg,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.mode,
            args.awr_beta,
            args.awr_w_max,
            args.awr_normalize,
            args.label_mode,
            use_atom_loss,
            angle_weights,
            use_rtg=args.use_rtg,
            use_causal_mask=args.use_causal_mask,
        )
        val_metrics = evaluate(
            model, val_loader, device, args.label_mode, use_atom_loss, angle_weights,
            use_rtg=args.use_rtg, use_causal_mask=args.use_causal_mask,
        )
        LOG.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | voted=%.3f | expert=%.3f | atom=%.3f | joint=%.3f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            val_metrics["voted_acc"],
            val_metrics["expert_acc"],
            val_metrics["atom_acc"],
            val_metrics["joint_acc"],
        )
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], **val_metrics})

    final_voted = history[-1]["voted_acc"]
    if final_voted < args.expected_voted:
        msg = f"Voted accuracy {final_voted:.3f} below expected threshold {args.expected_voted}"
        if args.require_voted_threshold:
            raise RuntimeError(msg)
        LOG.warning(msg)

    if args.log_file:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        config_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
        with args.log_file.open("w") as f:
            json.dump({"config": config_dump, "history": history, "coverage": coverage}, f, indent=2)


if __name__ == "__main__":
    main()
