from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ShardSummary:
    path: Path
    samples: int
    angles: List[int]
    clips_per_angle: int


def load_shard_summary(shard_dir: Path) -> ShardSummary:
    summary_path = shard_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json in {shard_dir}")
    with summary_path.open() as f:
        summary = json.load(f)
    return ShardSummary(
        path=shard_dir,
        samples=int(summary["samples"]),
        angles=[int(a) for a in summary["angles"]],
        clips_per_angle=int(summary.get("clips_per_angle", 0)),
    )


class AngleRangeDataset(Dataset):
    """Simple loader for embeddings.npz shards produced by the angle-range generator."""

    def __init__(self, shard_dirs: Sequence[Path]):
        self.shard_dirs = [Path(p) for p in shard_dirs]
        if not self.shard_dirs:
            raise ValueError("AngleRangeDataset requires at least one shard directory")
        self.samples: List[Dict] = []
        self.angles: List[int] = []
        self._load_all()

    def _load_all(self) -> None:
        for shard_dir in self.shard_dirs:
            npz_path = shard_dir / "embeddings.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing embeddings.npz in {shard_dir}")
            data = np.load(npz_path)
            required = ["h_seq", "expert_gt", "atom_gt", "angle_gt"]
            for key in required:
                if key not in data:
                    raise ValueError(f"{npz_path} missing key '{key}'")
            h_seq = data["h_seq"]
            expert_gt = data["expert_gt"]
            atom_gt = data["atom_gt"]
            angle_gt = data["angle_gt"]
            angle_deg_gt = data["angle_deg_gt"] if "angle_deg_gt" in data else None
            actual_length = data["actual_length"] if "actual_length" in data else np.full(h_seq.shape[0], h_seq.shape[1], dtype=np.int64)
            # Load RTG and step sequences for Physics-Informed DT
            rtg_seq = data["rtg_seq"] if "rtg_seq" in data else None
            step_seq = data["step_seq"] if "step_seq" in data else None
            if h_seq.shape[0] != expert_gt.shape[0] or h_seq.shape[0] != angle_gt.shape[0]:
                raise ValueError(f"Shape mismatch in {npz_path}: h_seq={h_seq.shape} expert_gt={expert_gt.shape} angle_gt={angle_gt.shape}")
            for i in range(h_seq.shape[0]):
                angle_deg_val = float(angle_deg_gt[i]) if angle_deg_gt is not None else None
                sample = {
                    "h_seq": torch.from_numpy(h_seq[i]).float(),
                    "expert_gt": torch.from_numpy(expert_gt[i]).long(),
                    "atom_gt": torch.from_numpy(atom_gt[i]).long(),
                    "angle_gt": int(angle_gt[i]),
                    "angle_deg": angle_deg_val,
                    "actual_length": int(actual_length[i]),
                    "shard": shard_dir.name,
                }
                # Add RTG and step sequences if available
                if rtg_seq is not None:
                    sample["rtg_seq"] = torch.from_numpy(rtg_seq[i]).float()
                if step_seq is not None:
                    sample["step_seq"] = torch.from_numpy(step_seq[i]).float()
                self.samples.append(sample)
            # Track coverage
            summary = load_shard_summary(shard_dir)
            self.angles.extend(summary.angles)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]

    def coverage(self) -> Dict[float, int]:
        """Return counts per angle in degrees for quick sanity checks."""
        counts: Dict[float, int] = {}
        for sample in self.samples:
            angle = sample["angle_deg"] if sample.get("angle_deg") is not None else sample["angle_gt"]
            counts[angle] = counts.get(angle, 0) + 1
        return counts

    def shape_spec(self) -> Tuple[int, int, int]:
        """Return (N, K, d_model) for diagnostics."""
        if not self.samples:
            return (0, 0, 0)
        h0 = self.samples[0]["h_seq"]
        return len(self.samples), h0.shape[0], h0.shape[1]
