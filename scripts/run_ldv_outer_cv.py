#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_fold_accuracy(mode: str, fold_dir: Path) -> dict:
    if mode == "transformer":
        metrics_path = fold_dir / "metrics.npz"
        data = np.load(metrics_path, allow_pickle=True)
        return {
            "test_accuracy": float(data["test_accuracy"]) if "test_accuracy" in data else float(data["best_accuracy"]),
            "best_val_accuracy": float(data["best_val_accuracy"]) if "best_val_accuracy" in data else None,
            "best_epoch": int(data["best_epoch"]),
        }

    results_path = fold_dir / "results.json"
    with open(results_path, "r") as f:
        data = json.load(f)
    return {
        "test_accuracy": float(data["overall_accuracy"]) / 100.0,
        "best_val_accuracy": None,
        "best_epoch": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic outer-fold CV for LDV transformer or greedy baseline scripts."
    )
    parser.add_argument("--mode", choices=["transformer", "greedy"], required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--out_dir", type=str, default=None, help="Base output directory for fold subdirectories")
    parser.add_argument("--skip_completed", action="store_true", help="Skip folds whose output artifact already exists")
    args, remainder = parser.parse_known_args()

    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/{args.mode}_ldv_outercv_{timestamp}"

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    script_name = "omp-transformer-ldv.py" if args.mode == "transformer" else "eval_greedy_with_doadataset.py"
    script_path = REPO_ROOT / "scripts" / script_name
    required_artifact = "metrics.npz" if args.mode == "transformer" else "results.json"

    fold_summaries = []
    for fold in range(args.n_folds):
        fold_dir = out_dir / f"fold_{fold}"
        artifact_path = fold_dir / required_artifact
        if args.skip_completed and artifact_path.exists():
            print(f"[skip] fold {fold}: found {artifact_path}")
        else:
            cmd = [
                sys.executable,
                str(script_path),
                "--n_folds",
                str(args.n_folds),
                "--test_fold",
                str(fold),
                "--val_fold",
                str((fold + 1) % args.n_folds),
                "--out_dir",
                args.out_dir,
                *remainder,
            ]
            print(f"[run] fold {fold}: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Missing fold artifact after fold {fold}: {artifact_path}")

        fold_summary = {"fold": fold}
        fold_summary.update(load_fold_accuracy(args.mode, fold_dir))
        fold_summaries.append(fold_summary)

    accuracies = np.array([item["test_accuracy"] for item in fold_summaries], dtype=float)
    summary = {
        "mode": args.mode,
        "n_folds": args.n_folds,
        "folds": fold_summaries,
        "mean_test_accuracy": float(np.mean(accuracies)),
        "std_test_accuracy": float(np.std(accuracies, ddof=0)),
    }

    summary_path = out_dir / "outer_cv_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] wrote outer-CV summary to {summary_path}")


if __name__ == "__main__":
    main()
