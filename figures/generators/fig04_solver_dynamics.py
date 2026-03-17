"""Figure 4 — Solver Dynamics: Architecture + Training + Ablation + Per-angle.

Panel (a): Architecture diagram (external JPG asset).
This generator produces the NEW data panels:
  (b) Training convergence curves (loss + accuracy over epochs)
  (c) Clean-condition ablation comparison (strip/dot chart)
  (d) Per-angle accuracy profile (37-point bar chart)

Data: metrics.npz (train_history, per_angle_accuracy), figure4_data.json (ablation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    DOUBLE_COL_MM,
    SEMANTIC_PALETTE,
)


# ---------------------------------------------------------------------------
# Ablation variant styling
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = [
    ("Baseline",         SEMANTIC_PALETTE["learned"],   "Baseline"),
    ("No Type Bias",     "#CC79A7",                     "No type bias"),
    ("No Transformer",   SEMANTIC_PALETTE["highlight"], "No transformer"),
    ("Fixed Heuristic",  SEMANTIC_PALETTE["ablation"],  "Fixed heuristic"),
    ("G-Fixed",          "#56B4E9",                     "G-fixed"),
    ("G-Teacher",        "#F0E442",                     "G-teacher"),
    ("Dense Routing",    "#000000",                     "Dense routing"),
]


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig04_panel_manifest.json"
    payload = {
        "figure_id": "fig04",
        "composite_asset": "figures/output/fig04_solver_dynamics.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 4 data panels (b, c, d)."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]
    metrics_path = run_dir / "metrics.npz"

    if not metrics_path.exists():
        print(f"[fig04] SKIP: metrics not found at {metrics_path}")
        return []

    # Load primary run metrics
    metrics = dict(np.load(metrics_path, allow_pickle=True))
    train_history = metrics["train_history"]  # array of dicts
    per_angle_acc = metrics["per_angle_accuracy"]
    angles = metrics["angles"]
    best_epoch = int(metrics["best_epoch"])

    # Load ablation data
    agg_path = data_root / paths_cfg["ablation_sweep"]["aggregated_json"]
    ablation_data: dict = {}
    if agg_path.exists():
        with open(agg_path) as f:
            ablation_data = json.load(f).get("ablation", {})

    # Extract training curves
    epochs = np.arange(1, len(train_history) + 1)
    total_loss = np.array([h["total_loss"] for h in train_history])
    class_loss = np.array([h["class_loss"] for h in train_history])
    teacher_acc = np.array([h["teacher_acc_subset"] for h in train_history])

    # -----------------------------------------------------------------------
    # Build composite figure (3 panels in a row)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.0, 1.2, 1.0],
        wspace=0.45, left=0.06, right=0.97, bottom=0.18, top=0.88,
    )

    # --- Panel (b): Training convergence ---
    ax_b = fig.add_subplot(gs[0, 0])
    ax_b.plot(epochs, total_loss, "-o", markersize=2, linewidth=0.9,
              color=SEMANTIC_PALETTE["learned"], label="Total loss")
    ax_b.plot(epochs, class_loss, "-s", markersize=2, linewidth=0.9,
              color=SEMANTIC_PALETTE["ablation"], label="Class loss")
    ax_b.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.6,
                 alpha=0.7, label=f"Best epoch ({best_epoch})")
    ax_b.set_xlabel("Epoch", fontsize=6)
    ax_b.set_ylabel("Loss", fontsize=6)
    ax_b.set_title("Training convergence", fontsize=6.5)
    ax_b.legend(fontsize=6, frameon=False, loc="upper right")
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)

    # Secondary axis for accuracy
    ax_b2 = ax_b.twinx()
    ax_b2.plot(epochs, teacher_acc, "-^", markersize=2, linewidth=0.9,
               color=SEMANTIC_PALETTE["physics"], alpha=0.7)
    ax_b2.set_ylabel("Subset accuracy", fontsize=6, color=SEMANTIC_PALETTE["physics"])
    ax_b2.tick_params(axis="y", labelcolor=SEMANTIC_PALETTE["physics"], labelsize=6)
    add_panel_label(ax_b, "b", x=-0.15, y=1.06)

    # --- Panel (c): Clean-condition ablation strip chart ---
    ax_c = fig.add_subplot(gs[0, 1])
    y_positions = []
    y_labels = []
    for i, (variant_key, color, display_name) in enumerate(ABLATION_VARIANTS):
        seeds = ablation_data.get(variant_key, [])
        if not seeds:
            continue
        y_pos = len(y_positions)
        y_positions.append(y_pos)
        y_labels.append(display_name)

        seeds_arr = np.array(seeds)
        mean_val = np.mean(seeds_arr)

        # Individual seed points
        ax_c.scatter(seeds_arr, [y_pos] * len(seeds_arr),
                     color=color, s=12, zorder=3, alpha=0.7, edgecolors="none")
        # Mean bar
        ax_c.plot([mean_val, mean_val], [y_pos - 0.3, y_pos + 0.3],
                  color=color, linewidth=1.5, zorder=4)

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(y_labels, fontsize=6)
    ax_c.set_xlabel("Accuracy (clean)", fontsize=6)
    ax_c.set_title("Ablation comparison", fontsize=6.5)
    ax_c.set_xlim(0, 1.05)
    ax_c.grid(axis="x", linestyle="--", alpha=0.3)
    ax_c.invert_yaxis()
    add_panel_label(ax_c, "c", x=-0.18, y=1.06)

    # --- Panel (d): Per-angle accuracy profile ---
    ax_d = fig.add_subplot(gs[0, 2])
    bar_colors = [SEMANTIC_PALETTE["learned"]] * len(angles)
    # Highlight angles below mean
    mean_acc = np.mean(per_angle_acc)
    for i in range(len(per_angle_acc)):
        if per_angle_acc[i] < mean_acc - 0.1:
            bar_colors[i] = SEMANTIC_PALETTE["ablation"]

    ax_d.bar(angles, per_angle_acc, width=4, color=bar_colors, alpha=0.8,
             edgecolor="none")
    ax_d.axhline(mean_acc, color="black", linestyle="--", linewidth=0.6,
                 alpha=0.7, label=f"Mean = {mean_acc:.2f}")
    ax_d.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_ylabel("Accuracy", fontsize=6)
    ax_d.set_title("Per-angle accuracy", fontsize=6.5)
    ax_d.set_ylim(0, 1.05)
    ax_d.legend(fontsize=6, frameon=False, loc="lower left")
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_d, "d", x=-0.12, y=1.06)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig04_solver_dynamics")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig04_solver_dynamics_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel b standalone
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_b.add_subplot(111)
    ax.plot(epochs, total_loss, "-o", markersize=3, linewidth=1.0,
            color=SEMANTIC_PALETTE["learned"], label="Total loss")
    ax.plot(epochs, class_loss, "-s", markersize=3, linewidth=1.0,
            color=SEMANTIC_PALETTE["ablation"], label="Class loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=5, frameon=False)
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig04_panel_b_convergence"))
    plt.close(fig_b)

    # Panel c standalone
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_c.add_subplot(111)
    y_positions_s = []
    y_labels_s = []
    for i, (variant_key, color, display_name) in enumerate(ABLATION_VARIANTS):
        seeds = ablation_data.get(variant_key, [])
        if not seeds:
            continue
        y_pos = len(y_positions_s)
        y_positions_s.append(y_pos)
        y_labels_s.append(display_name)
        seeds_arr = np.array(seeds)
        ax.scatter(seeds_arr, [y_pos] * len(seeds_arr),
                   color=color, s=15, zorder=3, alpha=0.7, edgecolors="none")
        ax.plot([np.mean(seeds_arr)] * 2, [y_pos - 0.3, y_pos + 0.3],
                color=color, linewidth=2.0, zorder=4)
    ax.set_yticks(y_positions_s)
    ax.set_yticklabels(y_labels_s, fontsize=6)
    ax.set_xlabel("Accuracy (clean)")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.invert_yaxis()
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.20, right=0.95, bottom=0.12, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig04_panel_c_ablation"))
    plt.close(fig_c)

    # Panel d standalone
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_d.add_subplot(111)
    ax.bar(angles, per_angle_acc, width=4, color=SEMANTIC_PALETTE["learned"],
           alpha=0.8, edgecolor="none")
    ax.axhline(mean_acc, color="black", linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig04_panel_d_perangle"))
    plt.close(fig_d)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "b",
                "title": "Training convergence",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_convergence.pdf",
                "provenance_mode": "data_backed",
                "description": "Training loss and subset accuracy over 20 epochs, best epoch marked.",
            },
            {
                "panel_id": "c",
                "title": "Clean-condition ablation",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_ablation.pdf",
                "provenance_mode": "data_backed",
                "description": "Strip chart of 7-variant ablation at clean SNR with individual seeds and mean bars.",
            },
            {
                "panel_id": "d",
                "title": "Per-angle accuracy profile",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_perangle.pdf",
                "provenance_mode": "data_backed",
                "description": "37-point bar chart showing accuracy per angle; below-mean angles highlighted.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig04] Generated {len(all_paths)} files "
          f"(best_epoch={best_epoch}, mean_acc={mean_acc:.3f})")
    return all_paths
