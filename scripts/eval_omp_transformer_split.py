#!/usr/bin/env python3
"""
Post-hoc evaluation for OMP Transformer Speech260 train/val split.

Given a finished training run directory (with code_state.json and model_best.pth),
reloads the model and data, re-applies the deterministic train/val split based on
clip ids, and computes separate metrics on the train and validation subsets without
retraining the model.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import importlib


def build_model_from_args(mod, H_final, W_final, angles, args_dict, device: str):
    """Recreate FullTransformerRoutedSoftOMP using the same hyperparameters as training."""
    F = H_final.shape[0]
    E = len(angles)
    M = int(args_dict.get("n_atoms", 8))
    d_model = int(args_dict.get("d_model", 64))
    nhead = int(args_dict.get("nhead", 2))
    nlayers = int(args_dict.get("nlayers", 1))

    # Ensure nhead divides d_model, mirroring training main()
    if d_model % nhead != 0:
        for candidate in [2, 4, 8, 1]:
            if d_model % candidate == 0:
                nhead = candidate
                break
        else:
            nhead = 1

    model = mod.FullTransformerRoutedSoftOMP(
        F=F,
        E=E,
        M=M,
        d=d_model,
        nhead=nhead,
        nlayers=nlayers,
        steps=int(args_dict.get("steps", 2)),
        top_e=int(args_dict.get("top_e", 2)),
        L=int(args_dict.get("top_l", 2)),
        tau_e=0.5,
        tau_a=0.2,
        eta=0.5,
        routing="gumbel",
        routing_mode=args_dict.get("routing_mode", "qk"),
        hybrid_alpha=float(args_dict.get("hybrid_alpha", 0.5)),
        routing_e=args_dict.get("routing_activation_e", "gumbel"),
        routing_a=args_dict.get("routing_activation_a", "gumbel"),
        score_norm_mode=args_dict.get("score_norm", "none"),
        score_reg_weight=float(args_dict.get("score_reg_weight", 0.0)),
    ).to(device)

    # Apply toggles (mirror training main())
    model.no_type_bias = bool(args_dict.get("no_type_bias", False))
    model.encoder_identity = bool(args_dict.get("encoder_identity", False))
    model.single_gate_expert = bool(args_dict.get("single_gate_expert", False))
    model.score_center_atoms = bool(args_dict.get("score_center_atoms", False))
    model.score_center_expert = bool(args_dict.get("score_center_expert", False))
    model.expert_agg = args_dict.get("expert_agg", "l2")
    model.d_can_attend_r = bool(args_dict.get("d_can_attend_r", False))
    model.use_hard_gumbel = bool(args_dict.get("use_hard_gumbel", False))

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc train/val eval for OMP Transformer Speech260 run (no retraining)."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run directory (must contain code_state.json and model_best.pth).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to run evaluation on: cpu|mps|cuda (default: mps).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["train", "val", "both"],
        default="both",
        help="Which subset(s) to evaluate: train, val, or both (default: both).",
    )

    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    code_state_path = run_dir / "code_state.json"
    ckpt_path = run_dir / "model_best.pth"
    if not code_state_path.is_file():
        raise SystemExit(f"Missing code_state.json in run_dir: {code_state_path}")
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing model_best.pth in run_dir: {ckpt_path}")

    with code_state_path.open("r") as f:
        code_state = json.load(f)
    args_dict = code_state.get("args", {})

    # Import training module dynamically (omp-transformer-ldv.py)
    omp_mod = importlib.import_module("scripts.omp-transformer-ldv")

    # Rebuild data pipeline
    h_path = args_dict.get("h_path")
    w_path = args_dict.get("w_path")
    dataset_root = args_dict.get("dataset_root")
    if not (h_path and w_path and dataset_root):
        raise SystemExit("code_state.args is missing h_path, w_path, or dataset_root.")

    print("Reloading H/W and dataset using training configuration...")
    H_raw, W_raw, angles = omp_mod.load_raw_ldv_matrices(h_path, w_path, device="cpu")

    atom_reduce_mode = args_dict.get("atom_reduce_mode", "kmeans")
    n_atoms = int(args_dict.get("n_atoms", 8))
    atom_min_cos = float(args_dict.get("atom_min_cos", 0.98))
    random_state = int(args_dict.get("seed", 42))

    if atom_reduce_mode == "kmeans":
        W_reduced, kmeans_labels, _ = omp_mod.reduce_atoms_kmeans(
            W_raw, n_clusters=n_atoms, random_state=random_state
        )
    else:
        W_reduced, kmeans_labels, _ = omp_mod.reduce_atoms(
            W_raw,
            mode=atom_reduce_mode,
            n_clusters=n_atoms,
            random_state=random_state,
            min_cos=atom_min_cos,
        )

    H_final = H_raw
    W_final = W_reduced
    D, idx2angle = omp_mod.build_dictionary(H_final, W_final, angles)

    Y_samples, labels, metadata = omp_mod.load_ldv_samples(
        dataset_root, H_final, W_final, angles, device="cpu"
    )

    # Re-apply deterministic train/val split
    train_mask, val_mask, split_stats = omp_mod.split_train_val_by_clip_id(metadata, val_mod=5)
    Y_train = Y_samples[train_mask]
    labels_train = labels[train_mask]
    Y_val = Y_samples[val_mask]
    labels_val = labels[val_mask]

    print("\n[Eval] Using split from split_train_val_by_clip_id (clip_id % 5).")
    print(
        f"  Train subset: {int(train_mask.sum().item())} samples, "
        f"Val subset: {int(val_mask.sum().item())} samples"
    )

    # Build model and load checkpoint
    device = torch.device(args.device if torch.cuda.is_available() or args.device != "cuda" else "cpu")
    model = build_model_from_args(omp_mod, H_final, W_final, angles, args_dict, device=device)
    # Load checkpoint produced by training run (trusted source in this repo)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate requested subsets
    results = {}
    if args.subset in ("train", "both"):
        print("\n[Eval] Evaluating TRAIN subset...")
        metrics_train = omp_mod.evaluate(model, D, Y_train, labels_train, idx2angle, device=device)
        results["train"] = metrics_train
        print(
            f"  Train accuracy: {metrics_train['accuracy']:.3f} "
            f"({metrics_train['accuracy']*100:.1f}%)"
        )

    if args.subset in ("val", "both"):
        print("\n[Eval] Evaluating VAL subset...")
        metrics_val = omp_mod.evaluate(model, D, Y_val, labels_val, idx2angle, device=device)
        results["val"] = metrics_val
        print(
            f"  Val accuracy:   {metrics_val['accuracy']:.3f} "
            f"({metrics_val['accuracy']*100:.1f}%)"
        )

    # Save metrics to run_dir
    out_npz = run_dir / "posthoc_eval_metrics.npz"
    np.savez(
        out_npz,
        train_accuracy=results.get("train", {}).get("accuracy", None),
        val_accuracy=results.get("val", {}).get("accuracy", None),
        train_confusion=results.get("train", {}).get("confusion_matrix", None),
        val_confusion=results.get("val", {}).get("confusion_matrix", None),
        train_per_angle=results.get("train", {}).get("per_angle_accuracy", None),
        val_per_angle=results.get("val", {}).get("per_angle_accuracy", None),
    )
    print(f"\n[Eval] Saved post-hoc metrics to: {out_npz}")


if __name__ == "__main__":
    main()
