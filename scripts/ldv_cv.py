from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


def parse_clip_id(path: str) -> int:
    """Extract the integer clip id from a dataset sample path."""
    fname = os.path.basename(path)
    stem, _ = os.path.splitext(fname)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse clip id from path: {path}")
    return int(digits)


def compute_dataset_fingerprint(dataset_root: str) -> str:
    """Compute a deterministic MD5 fingerprint of all .npy files in a dataset."""
    root = Path(dataset_root)
    npy_files = sorted(root.rglob("*.npy"))
    hasher = hashlib.md5()
    for npy_file in npy_files:
        with open(npy_file, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def build_outer_cv_split(
    metadata: Iterable[Dict[str, Any]],
    n_folds: int = 5,
    test_fold: int = 0,
    val_fold: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Build deterministic outer-CV train/validation/test masks from clip ids.

    Fold assignment is stratified by angle because each clip id is shared across
    angle folders and we use `clip_id % n_folds` as the fold index.
    """
    if n_folds < 3:
        raise ValueError(f"n_folds must be >= 3 for train/val/test splitting, got {n_folds}")
    if not 0 <= test_fold < n_folds:
        raise ValueError(f"test_fold must be in [0, {n_folds - 1}], got {test_fold}")
    if val_fold is None:
        val_fold = (test_fold + 1) % n_folds
    if not 0 <= val_fold < n_folds:
        raise ValueError(f"val_fold must be in [0, {n_folds - 1}], got {val_fold}")
    if val_fold == test_fold:
        raise ValueError("val_fold must differ from test_fold")

    metadata = list(metadata)
    n_samples = len(metadata)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    per_angle: Dict[float, Dict[str, Any]] = {}

    for idx, item in enumerate(metadata):
        path = item.get("path")
        angle_deg = float(item.get("angle_deg", 0.0))
        if path is None:
            raise ValueError(f"Missing 'path' in metadata entry at index {idx}")

        clip_id = parse_clip_id(path)
        fold_id = clip_id % n_folds

        stats = per_angle.setdefault(
            angle_deg,
            {
                "train": 0,
                "val": 0,
                "test": 0,
                "fold_counts": [0 for _ in range(n_folds)],
            },
        )
        stats["fold_counts"][fold_id] += 1

        if fold_id == test_fold:
            test_indices.append(idx)
            stats["test"] += 1
        elif fold_id == val_fold:
            val_indices.append(idx)
            stats["val"] += 1
        else:
            train_indices.append(idx)
            stats["train"] += 1

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            "Outer-CV split produced an empty subset "
            f"(train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)})."
        )

    train_mask = torch.zeros(n_samples, dtype=torch.bool)
    val_mask = torch.zeros(n_samples, dtype=torch.bool)
    test_mask = torch.zeros(n_samples, dtype=torch.bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    split_info: Dict[str, Any] = {
        "event": "outer_cv_split",
        "n_folds": int(n_folds),
        "test_fold": int(test_fold),
        "val_fold": int(val_fold),
        "train_folds": [fold for fold in range(n_folds) if fold not in (test_fold, val_fold)],
        "total_samples": int(n_samples),
        "train_samples": int(train_mask.sum().item()),
        "val_samples": int(val_mask.sum().item()),
        "test_samples": int(test_mask.sum().item()),
        "per_angle": {
            int(angle): {
                "train": int(stats["train"]),
                "val": int(stats["val"]),
                "test": int(stats["test"]),
                "fold_counts": [int(v) for v in stats["fold_counts"]],
            }
            for angle, stats in sorted(per_angle.items())
        },
    }
    return train_mask, val_mask, test_mask, split_info


def print_outer_cv_split(split_info: Dict[str, Any]) -> None:
    """Pretty-print the deterministic outer-CV split summary."""
    print(
        "\n=== Outer Cross-Validation Split "
        f"(n_folds={split_info['n_folds']}, test_fold={split_info['test_fold']}, "
        f"val_fold={split_info['val_fold']}) ==="
    )
    print(f"  Total samples: {split_info['total_samples']}")
    print(f"  Train samples: {split_info['train_samples']}")
    print(f"  Val samples:   {split_info['val_samples']}")
    print(f"  Test samples:  {split_info['test_samples']}")
    for angle in sorted(split_info["per_angle"].keys()):
        stats = split_info["per_angle"][angle]
        print(
            f"  Angle {int(angle):3d}°: "
            f"train={stats['train']:3d}, val={stats['val']:3d}, test={stats['test']:3d}, "
            f"folds={stats['fold_counts']}"
        )
