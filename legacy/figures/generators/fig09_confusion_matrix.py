"""Supplementary Figure 9 — Confusion Matrix Heatmaps.

Panels:
  (a) Confusion matrix heatmaps: Baseline vs No Transformer
  (b) Angle 55 distribution comparison
  (c) Angle 100 distribution comparison
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    load_paths,
    DOUBLE_COL_MM,
)


def _load_metrics(path: Path):
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def _plot_heatmaps(baseline_cm, no_trans_cm, angles, out_path: Path) -> list[Path]:
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25, left=0.08, right=0.95, bottom=0.15, top=0.85)

    for ax_idx, (cm, title) in enumerate([(baseline_cm, "Baseline (Transformer)"), (no_trans_cm, "No Transformer")]):
        ax = fig.add_subplot(gs[ax_idx])
        im = ax.imshow(cm, interpolation="nearest", cmap="viridis", origin="upper")
        ax.set_title(title, fontsize=8, fontweight="bold")

        tick_indices = np.arange(0, len(angles), 5)
        tick_labels = [f"{angles[i]:.0f}" for i in tick_indices]
        ax.set_xticks(tick_indices); ax.set_xticklabels(tick_labels, rotation=45, fontsize=6)
        ax.set_yticks(tick_indices); ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_xlabel("Predicted Atom Index", fontsize=7)
        ax.set_ylabel("True DOA (Angle)", fontsize=7)

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    paths = save_outputs(fig, out_path)
    plt.close(fig)
    return paths


def _plot_distribution_comparison(baseline_row, no_trans_row, angles, target_angle, out_path: Path) -> list[Path]:
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=65)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, left=0.08, right=0.95, bottom=0.2, top=0.85)

    target_idx = int(np.where(np.isclose(angles, target_angle))[0][0])

    for ax_idx, (row, title) in enumerate([(baseline_row, f"Baseline @ {target_angle:.0f}\u00b0"), (no_trans_row, f"No Transformer @ {target_angle:.0f}\u00b0")]):
        ax = fig.add_subplot(gs[ax_idx])
        total = row.sum()
        probs = row / total if total > 0 else row
        atom_indices = np.arange(len(angles))

        colors = np.array(["#7f7f7f"] * len(angles))
        colors[target_idx] = "#d62728"

        ax.vlines(x=atom_indices, ymin=0, ymax=probs, colors=colors, linewidth=1.0, alpha=0.8)
        ax.scatter(atom_indices, probs, color=colors, s=10, zorder=3, edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5)

        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel("Predicted Atom Index", fontsize=7)
        ax.set_ylabel("Probability", fontsize=7)
        ax.set_ylim(0, 1.05)

        tick_indices = np.arange(0, len(angles), 5)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f"{angles[i]:.0f}\u00b0" for i in tick_indices], rotation=45, fontsize=6)
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(axis="y", linestyle="--", alpha=0.3, linewidth=0.5)

        correct_prob = probs[target_idx]
        ax.text(target_idx, correct_prob + 0.05, f"{correct_prob:.1%}", ha="center", va="bottom", color="#d62728", fontsize=6, fontweight="bold")

    paths = save_outputs(fig, out_path)
    plt.close(fig)
    return paths


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 9 panels. Returns list of output file paths."""
    set_nature_rcparams()

    paths_cfg = load_paths()
    baseline_path = data_root / paths_cfg["confusion_matrix"]["baseline"]
    no_trans_path = data_root / paths_cfg["confusion_matrix"]["no_transformer"]

    baseline_data = _load_metrics(baseline_path)
    no_trans_data = _load_metrics(no_trans_path)

    if baseline_data is None or no_trans_data is None:
        print(f"[fig09] SKIP: Confusion matrix data not found")
        return []

    baseline_cm = baseline_data["confusion_matrix"]
    no_trans_cm = no_trans_data["confusion_matrix"]
    angles = baseline_data["angles"]

    all_paths: list[Path] = []
    all_paths.extend(_plot_heatmaps(baseline_cm, no_trans_cm, angles, output_dir / "fig09_heatmaps"))

    for target_angle, suffix in [(55.0, "angle55"), (100.0, "angle100")]:
        target_idx = np.where(np.isclose(angles, target_angle))[0]
        if len(target_idx) > 0:
            idx = target_idx[0]
            all_paths.extend(_plot_distribution_comparison(baseline_cm[idx], no_trans_cm[idx], angles, target_angle, output_dir / f"fig09_{suffix}"))

    print(f"[fig09] Generated {len(all_paths)} files")
    return all_paths
