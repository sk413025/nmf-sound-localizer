"""Figure 5 — Performance + Structure Alignment (merged).

Panel (a): 3-line SNR sweep (Baseline vs No Transformer vs Fixed Heuristic)
           with std shading.
Panel (b): Side-by-side H_corr vs QK_corr heatmaps (37x37, RdBu_r).
Panel (c): Side-by-side OMP vs AI selection probability (37x37, viridis).

Data: figure4_data.json + modal_routing_val.npz + dictionary.npz.
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
# Constants (from old fig04_snr_ablation)
# ---------------------------------------------------------------------------
SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "\u221e"]

# 3 curves: Baseline (learned), No Transformer (highlight/ablation), Fixed Heuristic (OMP)
SNR_VARIANTS = [
    ("Baseline",        SEMANTIC_PALETTE["learned"],   "Physics-aware AI"),
    ("No Transformer",  SEMANTIC_PALETTE["highlight"],  "No transformer"),
    ("Fixed Heuristic", SEMANTIC_PALETTE["ablation"],  "OMP baseline"),
]


# ---------------------------------------------------------------------------
# Computation helpers (ported from old fig05_structure_alignment)
# ---------------------------------------------------------------------------

def _compute_global_correlation(
    routing_data: dict, dict_data: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores_expert = routing_data["scores_expert"]
    H = dict_data["H"]
    expert_corr = np.corrcoef(scores_expert.T)
    H_corr = np.corrcoef(H.T)
    structure_corr = np.corrcoef(H_corr.flatten(), expert_corr.flatten())[0, 1]
    return expert_corr, H_corr, structure_corr


def _compute_selection_probability(
    routing_data: dict,
) -> tuple[np.ndarray, np.ndarray]:
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
            physics_prob[true_angle, expert_idx] = (
                np.argmax(g_energy_expert[mask], axis=1) == expert_idx
            ).mean()
            qk_prob[true_angle, expert_idx] = (
                np.argmax(scores_expert[mask], axis=1) == expert_idx
            ).mean()

    return physics_prob, qk_prob


def _plot_snr_panel(ax, snr_curves: dict, variants: list[tuple]) -> None:
    """Plot multi-variant SNR degradation curves on the given axes."""
    x_snr = np.arange(len(SNR_ORDER))
    for variant, color, label in variants:
        variant_levels = snr_curves.get(variant, {})
        means, stds = [], []
        for snr_key in SNR_ORDER:
            vals = variant_levels.get(snr_key, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(x_snr, means_arr, "-o", color=color, markersize=3,
                linewidth=1.0, label=label, zorder=3)
        ax.fill_between(x_snr, means_arr - stds_arr, means_arr + stds_arr,
                        color=color, alpha=0.15, zorder=1)
    ax.set_xticks(x_snr)
    ax.set_xticklabels(SNR_DISPLAY)
    ax.set_xlabel("SNR (dB)", fontsize=6)
    ax.set_ylabel("Accuracy", fontsize=6)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=4.5, frameon=False, loc="lower left")


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig05_panel_manifest.json"
    payload = {
        "figure_id": "fig05",
        "composite_asset": "figures/output/fig05_performance_structure.pdf",
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
    """Generate Figure 5 — Performance + Structure Alignment."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    # Load SNR data
    agg_path = data_root / paths_cfg["ablation_sweep"]["aggregated_json"]
    snr_data: dict = {}
    if agg_path.exists():
        with open(agg_path) as f:
            snr_data = json.load(f)

    # Load routing data
    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"
    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig05] SKIP: routing data not found at {run_dir}")
        return []

    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    angles = dict_data["angles"]

    expert_corr, H_corr, pearson_r = _compute_global_correlation(
        routing_data, dict_data
    )
    physics_prob, qk_prob = _compute_selection_probability(routing_data)

    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]
    snr_curves = snr_data.get("snr", {})

    # -----------------------------------------------------------------------
    # Build composite figure
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=120)
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[0.8, 1.1, 1.1],
        wspace=0.35, left=0.07, right=0.96, bottom=0.08, top=0.92,
    )

    # --- Panel (a): SNR sweep (3 curves — P1-3) ---
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_snr_panel(ax_a, snr_curves, SNR_VARIANTS)
    ax_a.set_title("Noise robustness", fontsize=6.5)
    add_panel_label(ax_a, "a", x=-0.15, y=1.06)

    # --- Panel (b): H_corr vs QK_corr ---
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1], hspace=0.30,
    )

    ax_b1 = fig.add_subplot(gs_b[0, 0])
    im1 = ax_b1.imshow(H_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b1.set_title("H physical structure", fontsize=6, fontweight="bold")
    ax_b1.set_xticks(tick_positions)
    ax_b1.set_xticklabels(tick_labels, fontsize=5)
    ax_b1.set_yticks(tick_positions)
    ax_b1.set_yticklabels(tick_labels, fontsize=5)
    ax_b1.set_xlabel("Angle (\u00b0)", fontsize=5)
    ax_b1.set_ylabel("Angle (\u00b0)", fontsize=5)
    cbar = plt.colorbar(im1, ax=ax_b1, fraction=0.046, pad=0.04)
    cbar.set_label("Corr", fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_b1, "b", x=-0.2, y=1.12)

    ax_b2 = fig.add_subplot(gs_b[1, 0])
    im2 = ax_b2.imshow(expert_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b2.set_title("QK learned structure", fontsize=6, fontweight="bold")
    ax_b2.set_xticks(tick_positions)
    ax_b2.set_xticklabels(tick_labels, fontsize=5)
    ax_b2.set_yticks(tick_positions)
    ax_b2.set_yticklabels(tick_labels, fontsize=5)
    ax_b2.set_xlabel("Angle (\u00b0)", fontsize=5)
    ax_b2.set_ylabel("Angle (\u00b0)", fontsize=5)
    cbar = plt.colorbar(im2, ax=ax_b2, fraction=0.046, pad=0.04)
    cbar.set_label("Corr", fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    ax_b2.text(0.5, -0.25, f"Pearson r = {pearson_r:.3f}",
               transform=ax_b2.transAxes, fontsize=5.5, ha="center",
               style="italic")

    # --- Panel (c): Selection probability ---
    gs_c = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 2], hspace=0.30,
    )
    vmax_unified = max(physics_prob.max(), qk_prob.max())

    ax_c1 = fig.add_subplot(gs_c[0, 0])
    im3 = ax_c1.imshow(physics_prob, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                        interpolation="nearest")
    ax_c1.set_title("OMP selection", fontsize=6, fontweight="bold")
    ax_c1.set_ylabel("True DOA", fontsize=5)
    ax_c1.set_xlabel("Expert", fontsize=5)
    ax_c1.set_xticks(tick_positions)
    ax_c1.set_xticklabels(tick_labels, fontsize=5)
    ax_c1.set_yticks(tick_positions)
    ax_c1.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im3, ax=ax_c1, fraction=0.046, pad=0.04)
    cbar.set_label("P(select)", fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_c1, "c", x=-0.2, y=1.12)

    ax_c2 = fig.add_subplot(gs_c[1, 0])
    im4 = ax_c2.imshow(qk_prob, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                        interpolation="nearest")
    ax_c2.set_title("Physics-aware AI", fontsize=6, fontweight="bold")
    ax_c2.set_ylabel("True DOA", fontsize=5)
    ax_c2.set_xlabel("Expert", fontsize=5)
    ax_c2.set_xticks(tick_positions)
    ax_c2.set_xticklabels(tick_labels, fontsize=5)
    ax_c2.set_yticks(tick_positions)
    ax_c2.set_yticklabels(tick_labels, fontsize=5)
    cbar = plt.colorbar(im4, ax=ax_c2, fraction=0.046, pad=0.04)
    cbar.set_label("P(select)", fontsize=5)
    cbar.ax.tick_params(labelsize=5)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig05_performance_structure")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig05_performance_structure_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel a standalone (3 curves)
    fig_a = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_a.add_subplot(111)
    _plot_snr_panel(ax, snr_curves, SNR_VARIANTS)
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig05_panel_a_snr_sweep"))
    plt.close(fig_a)

    # Panel b standalone
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=120)
    gs_bs = gridspec.GridSpec(2, 1, figure=fig_b, hspace=0.30,
                              left=0.08, right=0.95, bottom=0.08, top=0.94)
    ax1 = fig_b.add_subplot(gs_bs[0, 0])
    ax1.imshow(H_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax1.set_title("H physical structure", fontsize=7, fontweight="bold")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=5)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=5)
    add_panel_label(ax1, "b")
    ax2 = fig_b.add_subplot(gs_bs[1, 0])
    ax2.imshow(expert_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax2.set_title("QK learned structure", fontsize=7, fontweight="bold")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=5)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=5)
    ax2.text(0.5, -0.22, f"Pearson r = {pearson_r:.3f}",
             transform=ax2.transAxes, fontsize=6, ha="center", style="italic")
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig05_panel_b_correlation"))
    plt.close(fig_b)

    # Panel c standalone
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=120)
    gs_cs = gridspec.GridSpec(2, 1, figure=fig_c, hspace=0.30,
                              left=0.08, right=0.95, bottom=0.08, top=0.94)
    ax1 = fig_c.add_subplot(gs_cs[0, 0])
    ax1.imshow(physics_prob, cmap="viridis", aspect="equal",
               extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
               interpolation="nearest")
    ax1.set_title("OMP selection", fontsize=7, fontweight="bold")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=5)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=5)
    add_panel_label(ax1, "c")
    ax2 = fig_c.add_subplot(gs_cs[1, 0])
    ax2.imshow(qk_prob, cmap="viridis", aspect="equal",
               extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
               interpolation="nearest")
    ax2.set_title("Physics-aware AI", fontsize=7, fontweight="bold")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=5)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=5)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig05_panel_c_selection"))
    plt.close(fig_c)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "SNR sweep",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_a_snr_sweep.pdf",
                "provenance_mode": "data_backed",
                "description": "Three-curve SNR degradation (physics-aware AI vs no transformer vs OMP baseline).",
            },
            {
                "panel_id": "b",
                "title": "Structure alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_correlation.pdf",
                "provenance_mode": "data_backed",
                "description": "H-matrix physical vs QK learned correlation heatmaps.",
            },
            {
                "panel_id": "c",
                "title": "Selection probability",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_selection.pdf",
                "provenance_mode": "data_backed",
                "description": "OMP vs physics-aware AI selection probability heatmaps.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
