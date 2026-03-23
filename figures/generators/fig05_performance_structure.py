"""Figure 5 — Performance + Structure Alignment + Interpretability (6 panels).

Panel (a): 3-line SNR sweep (Baseline vs No Transformer vs Fixed Heuristic)
Panel (b): Side-by-side H_corr vs QK_corr heatmaps (37x37, RdBu_r)
Panel (c): Side-by-side OMP vs AI selection probability (37x37, viridis)
Panel (d): Confusion matrices (baseline vs no-transformer) - promoted from Supp
Panel (e): Angle-specific routing at 2 representative angles - promoted from Supp
Panel (f): Per-angle diagonal concentration (baseline vs ablation)

Data: figure4_data.json + modal_routing_val.npz + dictionary.npz + confusion metrics.
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
# Constants
# ---------------------------------------------------------------------------
SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "\u221e"]

SNR_VARIANTS = [
    ("Baseline",        SEMANTIC_PALETTE["learned"],   "Physics-aware AI"),
    ("No Transformer",  SEMANTIC_PALETTE["highlight"],  "No transformer"),
    ("Fixed Heuristic", SEMANTIC_PALETTE["ablation"],  "OMP baseline"),
]


# ---------------------------------------------------------------------------
# Computation helpers
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
    ax.legend(fontsize=6, frameon=False, loc="lower left")


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
    """Generate Figure 5 — Performance + Structure + Interpretability (6 panels)."""
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

    # Load confusion matrix data (panels d-f)
    cm_cfg = paths_cfg.get("confusion_matrix", {})
    baseline_cm_path = data_root / cm_cfg.get("baseline", "")
    no_trans_cm_path = data_root / cm_cfg.get("no_transformer", "")

    baseline_cm = None
    no_trans_cm = None
    baseline_per_angle = None
    no_trans_per_angle = None

    if baseline_cm_path.exists():
        bdata = dict(np.load(baseline_cm_path, allow_pickle=True))
        baseline_cm = bdata["confusion_matrix"]
        baseline_per_angle = bdata["per_angle_accuracy"]
    if no_trans_cm_path.exists():
        ntdata = dict(np.load(no_trans_cm_path, allow_pickle=True))
        no_trans_cm = ntdata["confusion_matrix"]
        no_trans_per_angle = ntdata["per_angle_accuracy"]

    # -----------------------------------------------------------------------
    # Build composite figure (2 rows x 3 columns)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=170)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.22, wspace=0.28,
        left=0.06, right=0.98, bottom=0.06, top=0.97,
    )

    # --- Row 1 ---

    # Panel (a): SNR sweep
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_snr_panel(ax_a, snr_curves, SNR_VARIANTS)
    ax_a.set_title("Noise robustness", fontsize=6.5)
    add_panel_label(ax_a, "a", x=-0.15, y=1.06)

    # Panel (b): H_corr vs QK_corr
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 1],
        width_ratios=[1.0, 0.05],
        hspace=0.12, wspace=0.05,
    )

    ax_b1 = fig.add_subplot(gs_b[0, 0])
    im1 = ax_b1.imshow(H_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b1.set_title("H physical structure", fontsize=5.8)
    ax_b1.set_xticks(tick_positions)
    ax_b1.set_xticklabels([])
    ax_b1.set_yticks(tick_positions)
    ax_b1.set_yticklabels(tick_labels, fontsize=5)
    ax_b1.set_ylabel("Angle (\u00b0)", fontsize=5)
    ax_b1.tick_params(axis="both", length=2)
    add_panel_label(ax_b1, "b", x=-0.2, y=1.12)

    ax_b2 = fig.add_subplot(gs_b[1, 0], sharex=ax_b1, sharey=ax_b1)
    im2 = ax_b2.imshow(expert_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b2.set_title("QK learned structure", fontsize=5.8)
    ax_b2.set_xticks(tick_positions)
    ax_b2.set_xticklabels(tick_labels, fontsize=5)
    ax_b2.set_yticks(tick_positions)
    ax_b2.set_yticklabels(tick_labels, fontsize=5)
    ax_b2.set_xlabel("Angle (\u00b0)", fontsize=5)
    ax_b2.set_ylabel("Angle (\u00b0)", fontsize=5)
    ax_b2.tick_params(axis="both", length=2)
    cax_b = fig.add_subplot(gs_b[:, 1])
    cbar = plt.colorbar(im2, cax=cax_b)
    cbar.set_label("Corr", fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    ax_b2.text(0.03, 0.05, f"r = {pearson_r:.3f}",
               transform=ax_b2.transAxes, fontsize=5.3, style="italic")

    # Panel (c): Selection probability
    gs_c = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs[0, 2],
        width_ratios=[1.0, 0.05],
        hspace=0.12, wspace=0.05,
    )
    vmax_unified = max(physics_prob.max(), qk_prob.max())

    ax_c1 = fig.add_subplot(gs_c[0, 0])
    im3 = ax_c1.imshow(physics_prob, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                        interpolation="nearest")
    ax_c1.set_title("OMP selection", fontsize=5.8)
    ax_c1.set_ylabel("True DOA", fontsize=5)
    ax_c1.set_xticks(tick_positions)
    ax_c1.set_xticklabels([])
    ax_c1.set_yticks(tick_positions)
    ax_c1.set_yticklabels(tick_labels, fontsize=5)
    ax_c1.tick_params(axis="both", length=2)
    add_panel_label(ax_c1, "c", x=-0.2, y=1.12)

    ax_c2 = fig.add_subplot(gs_c[1, 0], sharex=ax_c1, sharey=ax_c1)
    im4 = ax_c2.imshow(qk_prob, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                        interpolation="nearest")
    ax_c2.set_title("Physics-aware AI", fontsize=5.8)
    ax_c2.set_ylabel("True DOA", fontsize=5)
    ax_c2.set_xlabel("Expert", fontsize=5)
    ax_c2.set_xticks(tick_positions)
    ax_c2.set_xticklabels(tick_labels, fontsize=5)
    ax_c2.set_yticks(tick_positions)
    ax_c2.set_yticklabels(tick_labels, fontsize=5)
    ax_c2.tick_params(axis="both", length=2)
    cax_c = fig.add_subplot(gs_c[:, 1])
    cbar = plt.colorbar(im4, cax=cax_c)
    cbar.set_label("P(select)", fontsize=5)
    cbar.ax.tick_params(labelsize=5)

    # --- Row 2 ---

    # Panel (d): Confusion matrices (baseline vs no-transformer)
    if baseline_cm is not None and no_trans_cm is not None:
        gs_d = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs[1, 0],
            width_ratios=[1.0, 0.05],
            hspace=0.12, wspace=0.05,
        )

        # Normalize confusion matrices to row-probabilities
        baseline_cm_norm = baseline_cm.astype(float)
        row_sums = baseline_cm_norm.sum(axis=1, keepdims=True)
        baseline_cm_norm = np.divide(baseline_cm_norm, row_sums,
                                      where=row_sums > 0)

        no_trans_cm_norm = no_trans_cm.astype(float)
        row_sums_nt = no_trans_cm_norm.sum(axis=1, keepdims=True)
        no_trans_cm_norm = np.divide(no_trans_cm_norm, row_sums_nt,
                                      where=row_sums_nt > 0)

        ax_d1 = fig.add_subplot(gs_d[0, 0])
        ax_d1.imshow(baseline_cm_norm, cmap="viridis", aspect="equal",
                     interpolation="nearest", vmin=0, vmax=1)
        ax_d1.set_title("Baseline", fontsize=5.8)
        ax_d1.set_xticks(tick_positions)
        ax_d1.set_xticklabels([])
        ax_d1.set_yticks(tick_positions)
        ax_d1.set_yticklabels(tick_labels, fontsize=5)
        ax_d1.set_ylabel("True", fontsize=5)
        ax_d1.tick_params(axis="both", length=2)
        add_panel_label(ax_d1, "d", x=-0.20, y=1.12)

        ax_d2 = fig.add_subplot(gs_d[1, 0], sharex=ax_d1, sharey=ax_d1)
        im_d2 = ax_d2.imshow(no_trans_cm_norm, cmap="viridis", aspect="equal",
                              interpolation="nearest", vmin=0, vmax=1)
        ax_d2.set_title("No transformer", fontsize=5.8)
        ax_d2.set_xticks(tick_positions)
        ax_d2.set_xticklabels(tick_labels, fontsize=5)
        ax_d2.set_yticks(tick_positions)
        ax_d2.set_yticklabels(tick_labels, fontsize=5)
        ax_d2.set_xlabel("Predicted", fontsize=5)
        ax_d2.set_ylabel("True", fontsize=5)
        ax_d2.tick_params(axis="both", length=2)
        cax_d = fig.add_subplot(gs_d[:, 1])
        cbar = plt.colorbar(im_d2, cax=cax_d)
        cbar.set_label("P(pred|true)", fontsize=5)
        cbar.ax.tick_params(labelsize=5)
    else:
        ax_d_placeholder = fig.add_subplot(gs[1, 0])
        ax_d_placeholder.text(0.5, 0.5, "Confusion matrix\ndata unavailable",
                              transform=ax_d_placeholder.transAxes,
                              ha="center", va="center", fontsize=7)
        ax_d_placeholder.set_axis_off()
        add_panel_label(ax_d_placeholder, "d", x=-0.1, y=1.06)

    # Panel (e): Angle-specific routing at 2 representative angles
    if baseline_cm is not None and no_trans_cm is not None:
        gs_e = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[1, 1], hspace=0.22,
        )

        representative_angles = [55.0, 100.0]
        for row_idx, target_angle in enumerate(representative_angles):
            target_idx = int(np.argmin(np.abs(angles - target_angle)))

            ax_e = fig.add_subplot(gs_e[row_idx, 0])

            # Baseline distribution
            bl_row = baseline_cm[target_idx].astype(float)
            bl_total = bl_row.sum()
            bl_probs = bl_row / bl_total if bl_total > 0 else bl_row

            # No-transformer distribution
            nt_row = no_trans_cm[target_idx].astype(float)
            nt_total = nt_row.sum()
            nt_probs = nt_row / nt_total if nt_total > 0 else nt_row

            x_pos = np.arange(len(angles))
            ax_e.bar(x_pos - 0.15, bl_probs, width=0.3, alpha=0.7,
                     color=SEMANTIC_PALETTE["learned"], label="Baseline")
            ax_e.bar(x_pos + 0.15, nt_probs, width=0.3, alpha=0.7,
                     color=SEMANTIC_PALETTE["ablation"], label="No trans.")
            ax_e.axvline(target_idx, color="lime", linewidth=0.8,
                         linestyle="--", alpha=0.8)

            ax_e.set_title(f"Routing @ {angles[target_idx]:.0f}\u00b0",
                           fontsize=5.6)
            if row_idx == 1:
                ax_e.set_xlabel("Expert index", fontsize=5)
            ax_e.set_ylabel("P(predict)", fontsize=5)
            ax_e.set_xticks(tick_positions)
            ax_e.set_xticklabels(tick_labels, fontsize=5.5)
            ax_e.tick_params(axis="y", labelsize=5.5)
            if row_idx == 0:
                ax_e.legend(fontsize=6, frameon=False, loc="upper right")
                add_panel_label(ax_e, "e", x=-0.20, y=1.12)
    else:
        ax_e_placeholder = fig.add_subplot(gs[1, 1])
        ax_e_placeholder.text(0.5, 0.5, "Angle routing\ndata unavailable",
                              transform=ax_e_placeholder.transAxes,
                              ha="center", va="center", fontsize=7)
        ax_e_placeholder.set_axis_off()
        add_panel_label(ax_e_placeholder, "e", x=-0.1, y=1.06)

    # Panel (f): Per-angle diagonal concentration (baseline vs ablation)
    if baseline_per_angle is not None and no_trans_per_angle is not None:
        ax_f = fig.add_subplot(gs[1, 2])
        ax_f.plot(angles, baseline_per_angle, "-o", markersize=2,
                  linewidth=0.9, color=SEMANTIC_PALETTE["learned"],
                  label="Baseline")
        ax_f.plot(angles, no_trans_per_angle, "-s", markersize=2,
                  linewidth=0.9, color=SEMANTIC_PALETTE["ablation"],
                  label="No transformer")
        ax_f.fill_between(angles, baseline_per_angle, no_trans_per_angle,
                          alpha=0.15, color=SEMANTIC_PALETTE["highlight"])
        ax_f.set_xlabel("Angle (\u00b0)", fontsize=6)
        ax_f.set_ylabel("P(correct)", fontsize=6)
        ax_f.set_title("Per-angle accuracy", fontsize=6.5)
        ax_f.set_ylim(0, 1.05)
        ax_f.legend(fontsize=6, frameon=False, loc="lower left")
        ax_f.grid(axis="y", linestyle="--", alpha=0.3)

        # Annotate improvement count
        n_improved = np.sum(baseline_per_angle > no_trans_per_angle)
        ax_f.text(0.95, 0.05, f"{n_improved}/{len(angles)} improved",
                  transform=ax_f.transAxes, ha="right", va="bottom",
                  fontsize=6, style="italic")
        add_panel_label(ax_f, "f", x=-0.15, y=1.06)
    else:
        ax_f_placeholder = fig.add_subplot(gs[1, 2])
        ax_f_placeholder.text(0.5, 0.5, "Per-angle data\nunavailable",
                              transform=ax_f_placeholder.transAxes,
                              ha="center", va="center", fontsize=7)
        ax_f_placeholder.set_axis_off()
        add_panel_label(ax_f_placeholder, "f", x=-0.1, y=1.06)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig05_performance_structure")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig05_performance_structure_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel a standalone
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

    # Panel d standalone (confusion matrices)
    if baseline_cm is not None and no_trans_cm is not None:
        fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
        gs_ds = fig_d.add_gridspec(1, 2, wspace=0.25, left=0.08, right=0.95,
                                    bottom=0.15, top=0.85)
        for ax_idx, (cm, title) in enumerate([
            (baseline_cm, "Baseline"),
            (no_trans_cm, "No Transformer"),
        ]):
            ax = fig_d.add_subplot(gs_ds[ax_idx])
            cm_norm = cm.astype(float)
            rs = cm_norm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_norm, rs, where=rs > 0)
            ax.imshow(cm_norm, cmap="viridis", aspect="equal",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(title, fontsize=7, fontweight="bold")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=5)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=5)
            ax.set_xlabel("Predicted", fontsize=6)
            if ax_idx == 0:
                ax.set_ylabel("True", fontsize=6)
        add_panel_label(fig_d.axes[0], "d")
        all_paths.extend(save_outputs(fig_d, panel_dir / "fig05_panel_d_confusion"))
        plt.close(fig_d)

    # Panel e standalone (angle-specific routing)
    if baseline_cm is not None and no_trans_cm is not None:
        fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=100)
        gs_es = gridspec.GridSpec(2, 1, figure=fig_e, hspace=0.35,
                                  left=0.08, right=0.95, bottom=0.10, top=0.92)
        for row_idx, target_angle in enumerate([55.0, 100.0]):
            target_idx = int(np.argmin(np.abs(angles - target_angle)))
            ax = fig_e.add_subplot(gs_es[row_idx, 0])
            bl_row = baseline_cm[target_idx].astype(float)
            bl_probs = bl_row / bl_row.sum() if bl_row.sum() > 0 else bl_row
            nt_row = no_trans_cm[target_idx].astype(float)
            nt_probs = nt_row / nt_row.sum() if nt_row.sum() > 0 else nt_row
            x_pos = np.arange(len(angles))
            ax.bar(x_pos - 0.15, bl_probs, width=0.3, alpha=0.7,
                   color=SEMANTIC_PALETTE["learned"], label="Baseline")
            ax.bar(x_pos + 0.15, nt_probs, width=0.3, alpha=0.7,
                   color=SEMANTIC_PALETTE["ablation"], label="No trans.")
            ax.axvline(target_idx, color="lime", linewidth=0.8, linestyle="--")
            ax.set_title(f"Routing @ {angles[target_idx]:.0f}\u00b0", fontsize=7)
            ax.set_xlabel("Expert index")
            ax.set_ylabel("P(predict)")
            if row_idx == 0:
                ax.legend(fontsize=5, frameon=False)
        add_panel_label(fig_e.axes[0], "e")
        all_paths.extend(save_outputs(fig_e, panel_dir / "fig05_panel_e_routing"))
        plt.close(fig_e)

    # Panel f standalone (per-angle diagonal concentration)
    if baseline_per_angle is not None and no_trans_per_angle is not None:
        fig_f = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
        ax = fig_f.add_subplot(111)
        ax.plot(angles, baseline_per_angle, "-o", markersize=3, linewidth=1.0,
                color=SEMANTIC_PALETTE["learned"], label="Baseline")
        ax.plot(angles, no_trans_per_angle, "-s", markersize=3, linewidth=1.0,
                color=SEMANTIC_PALETTE["ablation"], label="No transformer")
        ax.fill_between(angles, baseline_per_angle, no_trans_per_angle,
                        alpha=0.15, color=SEMANTIC_PALETTE["highlight"])
        ax.set_xlabel("Angle (\u00b0)")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=5, frameon=False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax, "f")
        fig_f.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
        all_paths.extend(save_outputs(fig_f, panel_dir / "fig05_panel_f_diagonal"))
        plt.close(fig_f)

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
            {
                "panel_id": "d",
                "title": "Confusion matrices",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_confusion.pdf",
                "provenance_mode": "data_backed",
                "description": "Baseline vs no-transformer confusion matrices showing diagonal sharpening.",
            },
            {
                "panel_id": "e",
                "title": "Angle-specific routing",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_routing.pdf",
                "provenance_mode": "data_backed",
                "description": "Routing distributions at 55 and 100 degrees comparing baseline vs no-transformer.",
            },
            {
                "panel_id": "f",
                "title": "Per-angle diagonal concentration",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_f_diagonal.pdf",
                "provenance_mode": "data_backed",
                "description": "37-point P(correct) comparison showing transformer routing benefit per angle.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
