"""Figure 5 — Structure Alignment (Correlation + Selection Probability).

Two-panel master figure:
  (a) Global correlation: 2 stacked 37x37 heatmaps
      - H matrix physical structure (top)
      - QK learned structure (bottom)
  (b) Selection probability: 2 stacked 37x37 heatmaps
      - OMP selection probability (top)
      - QK selection probability (bottom)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    load_paths,
    DOUBLE_COL_MM,
)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _load_data(run_dir: Path) -> tuple[dict, dict]:
    routing_data = dict(np.load(run_dir / "modal_routing_val.npz", allow_pickle=True))
    dict_data = dict(np.load(run_dir / "dictionary.npz", allow_pickle=True))
    return routing_data, dict_data


def _compute_global_correlation(routing_data: dict, dict_data: dict) -> tuple[np.ndarray, np.ndarray, float]:
    scores_expert = routing_data["scores_expert"]
    H = dict_data["H"]
    expert_corr = np.corrcoef(scores_expert.T)
    H_corr = np.corrcoef(H.T)
    structure_corr = np.corrcoef(H_corr.flatten(), expert_corr.flatten())[0, 1]
    return expert_corr, H_corr, structure_corr


def _compute_selection_probability(routing_data: dict) -> tuple[np.ndarray, np.ndarray]:
    scores_expert = routing_data["scores_expert"]
    g_energy_expert = routing_data["g_energy_expert"]
    labels = routing_data["labels"]

    n_angles = 37
    physics_prob = np.zeros((n_angles, n_angles))
    qk_prob = np.zeros((n_angles, n_angles))

    for true_angle in range(n_angles):
        mask = labels == true_angle
        if mask.sum() == 0:
            continue
        for expert_idx in range(n_angles):
            physics_prob[true_angle, expert_idx] = (np.argmax(g_energy_expert[mask], axis=1) == expert_idx).mean()
            qk_prob[true_angle, expert_idx] = (np.argmax(scores_expert[mask], axis=1) == expert_idx).mean()

    return physics_prob, qk_prob


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 5. Returns list of output file paths."""
    set_nature_rcparams()

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    if not (run_dir / "modal_routing_val.npz").exists():
        print(f"[fig05] SKIP: routing data not found at {run_dir}")
        return []

    routing_data, dict_data = _load_data(run_dir)
    expert_corr, H_corr, pearson_r = _compute_global_correlation(routing_data, dict_data)
    physics_prob, qk_prob = _compute_selection_probability(routing_data)
    angles = dict_data["angles"]

    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]

    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=160)
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35, left=0.08, right=0.95, bottom=0.06, top=0.92)

    # Panel (a): Global Correlation
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 0], hspace=0.30)

    ax_a1 = fig.add_subplot(gs_a[0, 0])
    im_a1 = ax_a1.imshow(H_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax_a1.set_title("H Matrix Physical Structure", fontsize=7, fontweight="bold")
    ax_a1.set_xlabel("Angle Index j", fontsize=6)
    ax_a1.set_ylabel("Angle Index i", fontsize=6)
    ax_a1.set_xticks(tick_positions); ax_a1.set_xticklabels(tick_labels, fontsize=5)
    ax_a1.set_yticks(tick_positions); ax_a1.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im_a1, ax=ax_a1, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=5); cbar.ax.tick_params(labelsize=5)

    ax_a2 = fig.add_subplot(gs_a[1, 0])
    im_a2 = ax_a2.imshow(expert_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax_a2.set_title("QK Learned Structure", fontsize=7, fontweight="bold")
    ax_a2.set_xlabel("Angle Index j", fontsize=6)
    ax_a2.set_ylabel("Angle Index i", fontsize=6)
    ax_a2.set_xticks(tick_positions); ax_a2.set_xticklabels(tick_labels, fontsize=5)
    ax_a2.set_yticks(tick_positions); ax_a2.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im_a2, ax=ax_a2, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", fontsize=5); cbar.ax.tick_params(labelsize=5)
    ax_a2.text(0.5, -0.22, f"Pearson r = {pearson_r:.3f}", transform=ax_a2.transAxes, fontsize=6, ha="center", style="italic")
    ax_a1.text(-0.15, 1.12, "a", transform=ax_a1.transAxes, fontsize=10, fontweight="bold", va="top")

    # Panel (b): Selection Probability
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], hspace=0.30)
    vmax_unified = max(physics_prob.max(), qk_prob.max())

    ax_b1 = fig.add_subplot(gs_b[0, 0])
    im_b1 = ax_b1.imshow(physics_prob, cmap="viridis", aspect="equal", extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified, interpolation="nearest")
    ax_b1.set_title("Traditional OMP", fontsize=7, fontweight="bold")
    ax_b1.set_ylabel("True DOA", fontsize=6); ax_b1.set_xlabel("Selected Expert", fontsize=6)
    ax_b1.set_xticks(tick_positions); ax_b1.set_xticklabels(tick_labels, fontsize=5)
    ax_b1.set_yticks(tick_positions); ax_b1.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im_b1, ax=ax_b1, fraction=0.046, pad=0.04)
    cbar.set_label("P(select)", fontsize=5); cbar.ax.tick_params(labelsize=5)

    ax_b2 = fig.add_subplot(gs_b[1, 0])
    im_b2 = ax_b2.imshow(qk_prob, cmap="viridis", aspect="equal", extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified, interpolation="nearest")
    ax_b2.set_title("Physics-Aware AI", fontsize=7, fontweight="bold")
    ax_b2.set_ylabel("True DOA", fontsize=6); ax_b2.set_xlabel("Selected Expert", fontsize=6)
    ax_b2.set_xticks(tick_positions); ax_b2.set_xticklabels(tick_labels, fontsize=5)
    ax_b2.set_yticks(tick_positions); ax_b2.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im_b2, ax=ax_b2, fraction=0.046, pad=0.04)
    cbar.set_label("P(select)", fontsize=5); cbar.ax.tick_params(labelsize=5)
    ax_b1.text(-0.15, 1.12, "b", transform=ax_b1.transAxes, fontsize=10, fontweight="bold", va="top")

    all_paths = save_outputs(fig, output_dir / "fig05_structure_alignment")
    plt.close(fig)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
