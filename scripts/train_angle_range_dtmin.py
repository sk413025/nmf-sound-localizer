#!/usr/bin/env python3
"""Train a tiny DTMin-style model on angle-range shards (BC or AWR)."""

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
    return {"h_seq": h_seq, "expert_gt": expert_gt, "atom_gt": atom_gt, "angle_gt": angle_gt}


class TinyDTMin(nn.Module):
    """Minimal DTMin-style encoder with dual heads."""

    def __init__(self, d_model: int, nhead: int, nlayers: int, E: int, M: int, max_K: int, dropout: float = 0.1):
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

    def forward(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = h_seq.shape
        pos_idx = torch.arange(K, device=h_seq.device).unsqueeze(0).expand(B, K)
        h = h_seq + self.pos(pos_idx)
        h = self.encoder(h)
        h = self.norm(h)
        expert_logits = self.expert_head(h)
        atom_logits = self.atom_head(h)
        return expert_logits, atom_logits


def compute_metrics(expert_logits: torch.Tensor, atom_logits: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    expert_pred = expert_logits.argmax(dim=-1)
    atom_pred = atom_logits.argmax(dim=-1)
    expert_acc = (expert_pred == batch["expert_gt"]).float().mean().item()
    atom_acc = (atom_pred == batch["atom_gt"]).float().mean().item()
    joint_acc = ((expert_pred == batch["expert_gt"]) & (atom_pred == batch["atom_gt"])).float().mean().item()
    voted_preds = []
    for i in range(expert_pred.shape[0]):
        counts = torch.bincount(expert_pred[i], minlength=expert_logits.shape[-1])
        voted_preds.append(int(torch.argmax(counts).item()))
    voted_preds = torch.tensor(voted_preds, device=expert_pred.device)
    voted_gt = batch["angle_gt"]
    voted_acc = (voted_preds == voted_gt).float().mean().item()
    return {
        "expert_acc": expert_acc,
        "atom_acc": atom_acc,
        "joint_acc": joint_acc,
        "voted_acc": voted_acc,
    }


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, mode: str, awr_beta: float, awr_w_max: float, awr_normalize: bool) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        h_seq = batch["h_seq"].to(device)
        expert_gt = batch["expert_gt"].to(device)
        atom_gt = batch["atom_gt"].to(device)
        angle_gt = batch["angle_gt"].to(device)

        optimizer.zero_grad()
        expert_logits, atom_logits = model(h_seq)

        expert_loss = F.cross_entropy(expert_logits.view(-1, expert_logits.shape[-1]), expert_gt.view(-1))
        atom_loss = F.cross_entropy(atom_logits.view(-1, atom_logits.shape[-1]), atom_gt.view(-1))
        loss_per_sample = (expert_loss + atom_loss).view(1)

        if mode == "awr":
            B, K = expert_gt.shape
            angle_expanded = angle_gt.view(B, 1).expand(-1, K)
            returns = (expert_gt == angle_expanded).float().mean(dim=1)
            baseline = returns.mean().detach()
            adv = returns - baseline
            weights = torch.exp(adv / awr_beta).clamp(max=awr_w_max)
            if awr_normalize:
                weights = weights / (weights.mean() + 1e-8)
            loss = (loss_per_sample * weights.mean()).mean()
        else:
            loss = loss_per_sample.mean()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    return {"loss": total_loss / max(total_batches, 1)}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    metrics_accum: Dict[str, float] = Counter()
    for batch in loader:
        h_seq = batch["h_seq"].to(device)
        expert_gt = batch["expert_gt"].to(device)
        atom_gt = batch["atom_gt"].to(device)
        expert_logits, atom_logits = model(h_seq)
        loss = (
            F.cross_entropy(expert_logits.view(-1, expert_logits.shape[-1]), expert_gt.view(-1))
            + F.cross_entropy(atom_logits.view(-1, atom_logits.shape[-1]), atom_gt.view(-1))
        )
        total_loss += loss.item()
        total_batches += 1
        m = compute_metrics(expert_logits, atom_logits, {k: v.to(device) for k, v in batch.items()})
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train tiny DTMin on angle-range shards (BC or AWR).")
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
    ap.add_argument("--mode", type=str, default="bc", choices=["bc", "awr"])
    ap.add_argument("--awr-beta", type=float, default=0.5)
    ap.add_argument("--awr-w-max", type=float, default=4.0)
    ap.add_argument("--awr-normalize", action="store_true")
    ap.add_argument("--expected-voted", type=float, default=0.6, help="Threshold to flag low voted accuracy.")
    ap.add_argument("--require-voted-threshold", action="store_true", help="Raise error if voted accuracy is below threshold.")
    ap.add_argument("--log-file", type=Path, default=None, help="Optional JSON log output.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    shard_dirs = [args.shard_root / name for name in args.shard_names.split(",") if name.strip()]
    for d in shard_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Shard dir not found: {d}")

    dataset = AngleRangeDataset(shard_dirs)
    N, K, d_model_detected = dataset.shape_spec()
    coverage = dataset.coverage()
    expected = expected_angles(shard_dirs)
    missing_angles = [a for a in expected if a not in coverage]
    LOG.info("Dataset loaded: N=%d K=%d d_model=%d shards=%s", N, K, d_model_detected, [d.name for d in shard_dirs])
    LOG.info("Coverage: angles=%d, missing=%s", len(coverage), missing_angles)
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
    model = TinyDTMin(
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.layers,
        E=E,
        M=M,
        max_K=K,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.mode, args.awr_beta, args.awr_w_max, args.awr_normalize)
        val_metrics = evaluate(model, val_loader, device)
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
