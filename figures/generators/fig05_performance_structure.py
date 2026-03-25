"""Figure 5 — Performance + Structure Alignment + Routing Attribution (6 panels).

Panel (a): 4-line SNR sweep benchmark (guided solver vs router-bypass vs OMP baseline vs dense routing)
Panel (b): Physical-vs-learned correlation structure (H vs QK)
Panel (c): Classical-reference confusion diagnostic (OMP baseline vs guided solver)
Panel (d): Confusion matrices (dense routing vs router-bypass)
Panel (e): Representative-angle conditional output distributions (guided solver vs router-bypass)
Panel (f): Per-angle decoder accuracy benchmark (guided solver vs router-bypass vs OMP baseline vs dense routing)

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
    SEMANTIC_PALETTE,
)
from figures.naming import get_bound_label
from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "\u221e"]
DENSE_COLOR = "#4A4A4A"

SNR_VARIANTS = [
    ("No Type Bias",    SEMANTIC_PALETTE["learned"],   get_bound_label("fig05", "a", "No Type Bias", label_type="short")),
    ("No Transformer",  SEMANTIC_PALETTE["ablation"],  get_bound_label("fig05", "a", "No Transformer", label_type="short")),
    ("Fixed Heuristic", SEMANTIC_PALETTE["classical"], get_bound_label("fig05", "a", "Fixed Heuristic", label_type="short")),
    ("Dense Routing",   DENSE_COLOR,                   get_bound_label("fig05", "a", "Dense Routing", label_type="short")),
]

FIG05_GENERATOR = figure_section("fig05", "generator")
FIG05_OUTER_GRID = dict(FIG05_GENERATOR["outer_grid"])
FIG05_TOP_ROW = dict(FIG05_GENERATOR["top_row"])
FIG05_BOTTOM_ROW = dict(FIG05_GENERATOR["bottom_row"])
FIG05_HEATMAP_STACK = dict(FIG05_GENERATOR["heatmap_stack"])
FIG05_ROUTING_STACK = dict(FIG05_GENERATOR["routing_stack"])
FIG05_STANDALONE = dict(FIG05_GENERATOR["standalone"])


def _titlecase_short_label(label: str) -> str:
    """Promote a compact label for title-like positions without mangling all-caps."""
    return label if label.isupper() else label[:1].upper() + label[1:]


def _set_gid(ax, gid: str):
    """Assign a stable layout/audit identifier to an axes and return it."""
    ax.set_gid(gid)
    return ax


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


def _row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def _centered_moving_average(series: np.ndarray, window: int = 3) -> np.ndarray:
    """Display-only centered moving average without circular wraparound."""
    series = np.asarray(series, dtype=float)
    if window <= 1:
        return series.copy()
    if window % 2 == 0:
        raise ValueError("window must be odd for centered moving average")

    half = window // 2
    smoothed = np.empty_like(series, dtype=float)
    for idx in range(series.size):
        lo = max(0, idx - half)
        hi = min(series.size, idx + half + 1)
        smoothed[idx] = float(series[lo:hi].mean())
    return smoothed


def _plot_snr_panel(
    ax,
    snr_curves: dict,
    variants: list[tuple],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
) -> None:
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
    ax.set_xlabel("SNR (dB)", fontsize=axis_label_pt)
    ax.set_ylabel("Accuracy", fontsize=axis_label_pt)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    legend_ncol = 2 if len(variants) > 3 else 1
    ax.legend(
        fontsize=legend_pt,
        frameon=False,
        loc="lower left",
        ncol=legend_ncol,
        columnspacing=0.8,
        handletextpad=0.4,
    )


def _save_panel_manifest(
    panel_dir: Path,
    panel_specs: list[dict],
    typography: dict[str, float],
) -> Path:
    manifest_path = panel_dir / "fig05_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig05",
        "composite_asset": "figures/output/fig05_performance_structure.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": typography,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 5 — Performance + Structure + Routing Attribution (6 panels)."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    annotation_pt = typography["annotation"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    panel_c_omp_full = get_bound_label("fig05", "c", "omp_side", label_type="full")
    panel_c_solver_full = get_bound_label(
        "fig05", "c", "learned_side", label_type="full"
    )
    cm_dense_short = get_bound_label("fig05", "d", "dense_cm", label_type="short")
    cm_no_transformer_short = get_bound_label(
        "fig05", "d", "no_transformer_cm", label_type="short"
    )
    cm_dense_full = get_bound_label("fig05", "d", "dense_cm", label_type="full")
    cm_no_transformer_full = get_bound_label(
        "fig05", "d", "no_transformer_cm", label_type="full"
    )
    routing_solver_short = get_bound_label(
        "fig05", "e", "baseline_dist", label_type="short"
    )
    routing_no_transformer_short = get_bound_label(
        "fig05", "e", "no_transformer_dist", label_type="short"
    )
    routing_solver_full = get_bound_label(
        "fig05", "e", "baseline_dist", label_type="full"
    )
    routing_no_transformer_full = get_bound_label(
        "fig05", "e", "no_transformer_dist", label_type="full"
    )
    diag_solver_short = get_bound_label("fig05", "f", "baseline_line", label_type="short")
    diag_no_transformer_short = get_bound_label(
        "fig05", "f", "no_transformer_line", label_type="short"
    )
    diag_omp_short = get_bound_label("fig05", "f", "omp_line", label_type="short")
    diag_dense_short = get_bound_label("fig05", "f", "dense_line", label_type="short")
    diag_solver_full = get_bound_label("fig05", "f", "baseline_line", label_type="full")
    diag_no_transformer_full = get_bound_label(
        "fig05", "f", "no_transformer_line", label_type="full"
    )
    diag_omp_full = get_bound_label("fig05", "f", "omp_line", label_type="full")
    diag_dense_full = get_bound_label("fig05", "f", "dense_line", label_type="full")
    snr_solver_full = get_bound_label("fig05", "a", "No Type Bias", label_type="full")
    snr_no_transformer_full = get_bound_label(
        "fig05", "a", "No Transformer", label_type="full"
    )
    snr_omp_full = get_bound_label("fig05", "a", "Fixed Heuristic", label_type="full")
    snr_dense_full = get_bound_label("fig05", "a", "Dense Routing", label_type="full")

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

    expert_corr, H_corr, pearson_r = _compute_global_correlation(routing_data, dict_data)

    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]
    snr_curves = snr_data.get("snr", {})

    # Load confusion matrix data (panels c-e)
    cm_cfg = paths_cfg.get("confusion_matrix", {})
    omp_baseline_cm_path = data_root / cm_cfg.get("omp_baseline", "")
    baseline_cm_path = data_root / cm_cfg.get("baseline", "")
    no_trans_cm_path = data_root / cm_cfg.get("no_transformer", "")
    dense_cm_path = data_root / cm_cfg.get("dense_routing", "")
    panel_f_summary_path = data_root / paths_cfg["fig05_panel_f_summary"]

    omp_baseline_cm = None
    baseline_cm = None
    no_trans_cm = None
    dense_cm = None
    panel_f_summary = None

    if omp_baseline_cm_path.exists():
        odata = dict(np.load(omp_baseline_cm_path, allow_pickle=True))
        omp_baseline_cm = odata["confusion_matrix"]
    if baseline_cm_path.exists():
        bdata = dict(np.load(baseline_cm_path, allow_pickle=True))
        baseline_cm = bdata["confusion_matrix"]
    if no_trans_cm_path.exists():
        ntdata = dict(np.load(no_trans_cm_path, allow_pickle=True))
        no_trans_cm = ntdata["confusion_matrix"]
    if dense_cm_path.exists():
        ddata = dict(np.load(dense_cm_path, allow_pickle=True))
        dense_cm = ddata["confusion_matrix"]
    if not panel_f_summary_path.exists():
        raise FileNotFoundError(
            f"Required Fig. 5f summary artifact not found: {panel_f_summary_path}"
        )
    panel_f_summary = dict(np.load(panel_f_summary_path, allow_pickle=True))

    # -----------------------------------------------------------------------
    # Build composite figure as two narrative rows:
    # top row = benchmark + context, bottom row = matched ablation block.
    # -----------------------------------------------------------------------
    fig = make_figure(
        width_mm=FIG05_GENERATOR["composite_width_mm"],
        height_mm=FIG05_GENERATOR["composite_height_mm"],
    )
    gs = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=FIG05_OUTER_GRID["height_ratios"],
        hspace=FIG05_OUTER_GRID["hspace"],
        left=FIG05_OUTER_GRID["left"],
        right=FIG05_OUTER_GRID["right"],
        bottom=FIG05_OUTER_GRID["bottom"],
        top=FIG05_OUTER_GRID["top"],
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[0, 0],
        width_ratios=FIG05_TOP_ROW["width_ratios"],
        wspace=FIG05_TOP_ROW["wspace"],
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs[1, 0],
        width_ratios=FIG05_BOTTOM_ROW["width_ratios"],
        wspace=FIG05_BOTTOM_ROW["wspace"],
    )

    # --- Row 1 ---

    # Panel (a): SNR sweep
    ax_a = _set_gid(fig.add_subplot(gs_top[0, 0]), "fig05.panel_a.main")
    _plot_snr_panel(
        ax_a,
        snr_curves,
        SNR_VARIANTS,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax_a.set_title("Noise robustness benchmark", fontsize=title_pt)
    add_panel_label(ax_a, "a", x=-0.15, y=1.06)

    # Panel (b): H_corr vs QK_corr
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_top[0, 1],
        width_ratios=[1.0, FIG05_HEATMAP_STACK["colorbar_ratio"]],
        hspace=FIG05_HEATMAP_STACK["hspace"],
        wspace=FIG05_HEATMAP_STACK["wspace"],
    )

    ax_b1 = _set_gid(fig.add_subplot(gs_b[0, 0]), "fig05.panel_b.top")
    im1 = ax_b1.imshow(H_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b1.set_title("Physical structure (H)", fontsize=title_pt)
    ax_b1.set_xticks(tick_positions)
    ax_b1.set_xticklabels([])
    ax_b1.set_yticks(tick_positions)
    ax_b1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_b1.set_ylabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_b1.tick_params(axis="both", length=2)
    add_panel_label(ax_b1, "b", x=-0.2, y=1.12)

    ax_b2 = _set_gid(fig.add_subplot(gs_b[1, 0], sharex=ax_b1, sharey=ax_b1), "fig05.panel_b.bottom")
    im2 = ax_b2.imshow(expert_corr, cmap="RdBu_r", aspect="equal",
                        vmin=-1.0, vmax=1.0)
    ax_b2.set_title("Learned routing (QK)", fontsize=title_pt)
    ax_b2.set_xticks(tick_positions)
    ax_b2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_b2.set_yticks(tick_positions)
    ax_b2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_b2.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_b2.set_ylabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_b2.tick_params(axis="both", length=2)
    cax_b = _set_gid(fig.add_subplot(gs_b[:, 1]), "fig05.panel_b.colorbar")
    cbar = plt.colorbar(im2, cax=cax_b)
    cbar.set_label("Corr", fontsize=colorbar_label_pt)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    ax_b2.text(0.03, 0.05, f"r = {pearson_r:.3f}",
               transform=ax_b2.transAxes, fontsize=annotation_pt, style="italic")

    # Panel (c): Confusion matrices (OMP baseline vs guided solver)
    gs_c = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=gs_top[0, 2],
        width_ratios=[1.0, FIG05_HEATMAP_STACK["colorbar_ratio"]],
        hspace=FIG05_HEATMAP_STACK["hspace"],
        wspace=FIG05_HEATMAP_STACK["wspace"],
    )
    if omp_baseline_cm is None or baseline_cm is None:
        raise FileNotFoundError(
            "Fig. 5c requires both paths.yaml[confusion_matrix.omp_baseline] and "
            "paths.yaml[confusion_matrix.baseline] metrics."
        )
    omp_baseline_cm_norm = _row_normalize_matrix(omp_baseline_cm)
    baseline_cm_norm = _row_normalize_matrix(baseline_cm)

    ax_c1 = _set_gid(fig.add_subplot(gs_c[0, 0]), "fig05.panel_c.top")
    im3 = ax_c1.imshow(omp_baseline_cm_norm, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=1.0,
                        interpolation="nearest")
    ax_c1.set_title("OMP baseline", fontsize=title_pt)
    ax_c1.set_ylabel("True DOA", fontsize=axis_label_pt)
    ax_c1.set_xticks(tick_positions)
    ax_c1.set_xticklabels([])
    ax_c1.set_yticks(tick_positions)
    ax_c1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_c1.tick_params(axis="both", length=2)
    add_panel_label(ax_c1, "c", x=-0.2, y=1.12)

    ax_c2 = _set_gid(fig.add_subplot(gs_c[1, 0], sharex=ax_c1, sharey=ax_c1), "fig05.panel_c.bottom")
    im4 = ax_c2.imshow(baseline_cm_norm, cmap="viridis", aspect="equal",
                        extent=[0, 37, 37, 0], vmin=0, vmax=1.0,
                        interpolation="nearest")
    ax_c2.set_title("Guided solver", fontsize=title_pt)
    ax_c2.set_ylabel("True DOA", fontsize=axis_label_pt)
    ax_c2.set_xlabel("Predicted DOA", fontsize=axis_label_pt)
    ax_c2.set_xticks(tick_positions)
    ax_c2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_c2.set_yticks(tick_positions)
    ax_c2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_c2.tick_params(axis="both", length=2)
    cax_c = _set_gid(fig.add_subplot(gs_c[:, 1]), "fig05.panel_c.colorbar")
    cbar = plt.colorbar(im4, cax=cax_c)
    cbar.set_label("P(pred|true)", fontsize=colorbar_label_pt)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)

    # --- Row 2 ---

    # Panel (d): Confusion matrices (dense routing vs router-bypass)
    if dense_cm is not None and no_trans_cm is not None:
        gs_d = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=gs_bottom[0, 0],
            width_ratios=[1.0, FIG05_HEATMAP_STACK["colorbar_ratio"]],
            hspace=FIG05_HEATMAP_STACK["hspace"],
            wspace=FIG05_HEATMAP_STACK["wspace"],
        )

        dense_cm_norm = _row_normalize_matrix(dense_cm)
        no_trans_cm_norm = _row_normalize_matrix(no_trans_cm)

        ax_d1 = _set_gid(fig.add_subplot(gs_d[0, 0]), "fig05.panel_d.top")
        ax_d1.imshow(dense_cm_norm, cmap="viridis", aspect="equal",
                     interpolation="nearest", vmin=0, vmax=1)
        ax_d1.set_title(_titlecase_short_label(cm_dense_short), fontsize=title_pt)
        ax_d1.set_xticks(tick_positions)
        ax_d1.set_xticklabels([])
        ax_d1.set_yticks(tick_positions)
        ax_d1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
        ax_d1.set_ylabel("True DOA", fontsize=axis_label_pt)
        ax_d1.tick_params(axis="both", length=2)
        add_panel_label(ax_d1, "d", x=-0.20, y=1.12)

        ax_d2 = _set_gid(fig.add_subplot(gs_d[1, 0], sharex=ax_d1, sharey=ax_d1), "fig05.panel_d.bottom")
        im_d2 = ax_d2.imshow(no_trans_cm_norm, cmap="viridis", aspect="equal",
                              interpolation="nearest", vmin=0, vmax=1)
        ax_d2.set_title(_titlecase_short_label(cm_no_transformer_short), fontsize=title_pt)
        ax_d2.set_xticks(tick_positions)
        ax_d2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
        ax_d2.set_yticks(tick_positions)
        ax_d2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
        ax_d2.set_xlabel("Predicted DOA", fontsize=axis_label_pt)
        ax_d2.set_ylabel("True DOA", fontsize=axis_label_pt)
        ax_d2.tick_params(axis="both", length=2)
        cax_d = _set_gid(fig.add_subplot(gs_d[:, 1]), "fig05.panel_d.colorbar")
        cbar = plt.colorbar(im_d2, cax=cax_d)
        cbar.set_label("P(pred|true)", fontsize=colorbar_label_pt)
        cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    else:
        ax_d_placeholder = fig.add_subplot(gs[1, 0])
        ax_d_placeholder.text(0.5, 0.5, "Confusion matrix\ndata unavailable",
                              transform=ax_d_placeholder.transAxes,
                              ha="center", va="center", fontsize=annotation_pt)
        ax_d_placeholder.set_axis_off()
        add_panel_label(ax_d_placeholder, "d", x=-0.1, y=1.06)

    # Panel (e): Conditional output profiles at 2 representative angles
    if baseline_cm is not None and no_trans_cm is not None:
        gs_e = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs_bottom[0, 1], hspace=FIG05_ROUTING_STACK["hspace"],
        )

        representative_angles = [55.0, 100.0]
        profile_ymax = 0.0
        profile_rows = []
        for target_angle in representative_angles:
            target_idx = int(np.argmin(np.abs(angles - target_angle)))
            bl_row = baseline_cm[target_idx].astype(float)
            nt_row = no_trans_cm[target_idx].astype(float)
            bl_probs = bl_row / bl_row.sum() if bl_row.sum() > 0 else bl_row
            nt_probs = nt_row / nt_row.sum() if nt_row.sum() > 0 else nt_row
            profile_ymax = max(profile_ymax, float(bl_probs.max()), float(nt_probs.max()))
            profile_rows.append((target_idx, bl_probs, nt_probs))
        profile_ymax = min(1.0, profile_ymax * 1.10)

        for row_idx, (target_idx, bl_probs, nt_probs) in enumerate(profile_rows):

            panel_gid = "fig05.panel_e.top" if row_idx == 0 else "fig05.panel_e.bottom"
            ax_e = _set_gid(fig.add_subplot(gs_e[row_idx, 0]), panel_gid)

            ax_e.plot(
                angles,
                bl_probs,
                "-o",
                markersize=2.5,
                linewidth=0.9,
                color=SEMANTIC_PALETTE["learned"],
                label=routing_solver_short,
                zorder=3,
            )
            ax_e.plot(
                angles,
                nt_probs,
                "--s",
                markersize=2.3,
                linewidth=0.9,
                color=SEMANTIC_PALETTE["ablation"],
                label=routing_no_transformer_short,
                zorder=2,
            )
            ax_e.axvline(float(angles[target_idx]), color=SEMANTIC_PALETTE["physics"], linewidth=0.8,
                         linestyle="--", alpha=0.8)

            ax_e.set_title(f"Conditional output @ {angles[target_idx]:.0f}\u00b0",
                           fontsize=title_pt)
            if row_idx == 1:
                ax_e.set_xlabel("Predicted angle (\u00b0)", fontsize=axis_label_pt)
            ax_e.set_ylabel("P(pred|true)", fontsize=axis_label_pt)
            ax_e.set_xticks(angles[tick_positions])
            ax_e.set_xticklabels(tick_labels, fontsize=tick_label_pt)
            ax_e.set_xlim(float(angles[0]), float(angles[-1]))
            ax_e.set_ylim(0, profile_ymax)
            ax_e.tick_params(axis="both", labelsize=tick_label_pt)
            ax_e.grid(axis="y", linestyle="--", alpha=0.25)
            if row_idx == 0:
                ax_e.legend(fontsize=legend_pt, frameon=False, loc="upper right")
                add_panel_label(ax_e, "e", x=-0.20, y=1.12)
    else:
        ax_e_placeholder = fig.add_subplot(gs[1, 1])
        ax_e_placeholder.text(0.5, 0.5, "Angle routing\ndata unavailable",
                              transform=ax_e_placeholder.transAxes,
                              ha="center", va="center", fontsize=annotation_pt)
        ax_e_placeholder.set_axis_off()
        add_panel_label(ax_e_placeholder, "e", x=-0.1, y=1.06)

    # Panel (f): Per-angle decoder accuracy benchmark
    if panel_f_summary is not None:
        f_angles = np.asarray(panel_f_summary["angles"], dtype=float)
        if not np.array_equal(np.asarray(angles, dtype=float), f_angles):
            raise ValueError("Fig. 5f summary angle grid does not match the figure angle grid")
        guided_mean = _centered_moving_average(
            np.asarray(panel_f_summary["guided_mean"], dtype=float), window=3
        )
        router_bypass_mean = _centered_moving_average(
            np.asarray(panel_f_summary["router_bypass_mean"], dtype=float), window=3
        )
        omp_mean = _centered_moving_average(
            np.asarray(panel_f_summary["omp_mean"], dtype=float), window=3
        )
        dense_mean = _centered_moving_average(
            np.asarray(panel_f_summary["dense_mean"], dtype=float), window=3
        )
        ax_f = _set_gid(fig.add_subplot(gs_bottom[0, 2]), "fig05.panel_f.main")
        ax_f.plot(
            angles,
            guided_mean,
            "-",
            linewidth=1.15,
            color=SEMANTIC_PALETTE["learned"],
            label=diag_solver_short,
            zorder=4,
        )
        ax_f.plot(
            angles,
            router_bypass_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["ablation"],
            label=diag_no_transformer_short,
            zorder=3,
        )
        ax_f.plot(
            angles,
            omp_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["classical"],
            label=diag_omp_short,
            zorder=2,
        )
        ax_f.plot(
            angles,
            dense_mean,
            ":",
            linewidth=1.05,
            color=DENSE_COLOR,
            label=diag_dense_short,
            zorder=1,
        )
        ax_f.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
        ax_f.set_ylabel("Mean P(correct)", fontsize=axis_label_pt)
        ax_f.set_title("Per-angle decoder accuracy", fontsize=title_pt)
        ax_f.set_ylim(0, 1.05)
        ax_f.legend(
            fontsize=legend_pt,
            frameon=False,
            loc="lower left",
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.4,
        )
        ax_f.tick_params(axis="both", labelsize=tick_label_pt)
        ax_f.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax_f, "f", x=-0.15, y=1.06)
    else:
        ax_f_placeholder = fig.add_subplot(gs[1, 2])
        ax_f_placeholder.text(0.5, 0.5, "Per-angle data\nunavailable",
                              transform=ax_f_placeholder.transAxes,
                              ha="center", va="center", fontsize=annotation_pt)
        ax_f_placeholder.set_axis_off()
        add_panel_label(ax_f_placeholder, "f", x=-0.1, y=1.06)

    # Save composite
    all_paths = save_outputs(
        fig,
        output_dir / "fig05_performance_structure",
        typography=typography,
    )
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig05_performance_structure_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel a standalone
    fig_a = make_figure(
        width_mm=FIG05_STANDALONE["a"]["width_mm"],
        height_mm=FIG05_STANDALONE["a"]["height_mm"],
    )
    ax = fig_a.add_subplot(111)
    _plot_snr_panel(
        ax,
        snr_curves,
        SNR_VARIANTS,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(**FIG05_STANDALONE["a"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_a,
            panel_dir / "fig05_panel_a_snr_sweep",
            typography=typography,
        )
    )
    plt.close(fig_a)

    # Panel b standalone
    fig_b = make_figure(
        width_mm=FIG05_STANDALONE["b"]["width_mm"],
        height_mm=FIG05_STANDALONE["b"]["height_mm"],
    )
    fig_b_grid = FIG05_STANDALONE["b"]["grid"]
    gs_bs = gridspec.GridSpec(
        2,
        1,
        figure=fig_b,
        hspace=fig_b_grid["hspace"],
        left=fig_b_grid["left"],
        right=fig_b_grid["right"],
        bottom=fig_b_grid["bottom"],
        top=fig_b_grid["top"],
    )
    ax1 = fig_b.add_subplot(gs_bs[0, 0])
    ax1.imshow(H_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax1.set_title("Physical structure (H)", fontsize=title_pt, fontweight="bold")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    add_panel_label(ax1, "b")
    ax2 = fig_b.add_subplot(gs_bs[1, 0])
    ax2.imshow(expert_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax2.set_title("Learned routing (QK)", fontsize=title_pt, fontweight="bold")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax2.text(0.5, -0.22, f"Pearson r = {pearson_r:.3f}",
             transform=ax2.transAxes, fontsize=annotation_pt, ha="center", style="italic")
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig05_panel_b_correlation",
            typography=typography,
        )
    )
    plt.close(fig_b)

    # Panel c standalone
    fig_c = make_figure(
        width_mm=FIG05_STANDALONE["c"]["width_mm"],
        height_mm=FIG05_STANDALONE["c"]["height_mm"],
    )
    fig_c_grid = FIG05_STANDALONE["c"]["grid"]
    gs_cs = gridspec.GridSpec(
        2,
        1,
        figure=fig_c,
        hspace=fig_c_grid["hspace"],
        left=fig_c_grid["left"],
        right=fig_c_grid["right"],
        bottom=fig_c_grid["bottom"],
        top=fig_c_grid["top"],
    )
    ax1 = fig_c.add_subplot(gs_cs[0, 0])
    ax1.imshow(omp_baseline_cm_norm, cmap="viridis", aspect="equal",
               extent=[0, 37, 37, 0], vmin=0, vmax=1.0,
               interpolation="nearest")
    ax1.set_title("OMP baseline", fontsize=title_pt, fontweight="bold")
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    add_panel_label(ax1, "c")
    ax2 = fig_c.add_subplot(gs_cs[1, 0])
    ax2.imshow(baseline_cm_norm, cmap="viridis", aspect="equal",
               extent=[0, 37, 37, 0], vmin=0, vmax=1.0,
               interpolation="nearest")
    ax2.set_title("Guided solver", fontsize=title_pt, fontweight="bold")
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    all_paths.extend(
        save_outputs(
            fig_c,
            panel_dir / "fig05_panel_c_selection",
            typography=typography,
        )
    )
    plt.close(fig_c)

    # Panel d standalone (confusion matrices)
    if dense_cm is not None and no_trans_cm is not None:
        fig_d = make_figure(
            width_mm=FIG05_STANDALONE["d"]["width_mm"],
            height_mm=FIG05_STANDALONE["d"]["height_mm"],
        )
        fig_d_grid = FIG05_STANDALONE["d"]["grid"]
        gs_ds = fig_d.add_gridspec(
            1,
            2,
            wspace=fig_d_grid["wspace"],
            left=fig_d_grid["left"],
            right=fig_d_grid["right"],
            bottom=fig_d_grid["bottom"],
            top=fig_d_grid["top"],
        )
        for ax_idx, (cm, title) in enumerate([
            (dense_cm, _titlecase_short_label(cm_dense_short)),
            (no_trans_cm, _titlecase_short_label(cm_no_transformer_short)),
        ]):
            ax = fig_d.add_subplot(gs_ds[ax_idx])
            cm_norm = cm.astype(float)
            rs = cm_norm.sum(axis=1, keepdims=True)
            cm_norm = np.divide(cm_norm, rs, where=rs > 0)
            ax.imshow(cm_norm, cmap="viridis", aspect="equal",
                      interpolation="nearest", vmin=0, vmax=1)
            ax.set_title(title, fontsize=title_pt, fontweight="bold")
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
            ax.set_xlabel("Predicted DOA", fontsize=axis_label_pt)
            if ax_idx == 0:
                ax.set_ylabel("True DOA", fontsize=axis_label_pt)
        add_panel_label(fig_d.axes[0], "d")
        all_paths.extend(
            save_outputs(
                fig_d,
                panel_dir / "fig05_panel_d_confusion",
                typography=typography,
            )
        )
        plt.close(fig_d)

    # Panel e standalone (conditional output profiles)
    if baseline_cm is not None and no_trans_cm is not None:
        fig_e = make_figure(
            width_mm=FIG05_STANDALONE["e"]["width_mm"],
            height_mm=FIG05_STANDALONE["e"]["height_mm"],
        )
        fig_e_grid = FIG05_STANDALONE["e"]["grid"]
        gs_es = gridspec.GridSpec(
            2,
            1,
            figure=fig_e,
            hspace=fig_e_grid["hspace"],
            left=fig_e_grid["left"],
            right=fig_e_grid["right"],
            bottom=fig_e_grid["bottom"],
            top=fig_e_grid["top"],
        )
        representative_angles = [55.0, 100.0]
        profile_ymax = 0.0
        profile_rows = []
        for target_angle in representative_angles:
            target_idx = int(np.argmin(np.abs(angles - target_angle)))
            bl_row = baseline_cm[target_idx].astype(float)
            nt_row = no_trans_cm[target_idx].astype(float)
            bl_probs = bl_row / bl_row.sum() if bl_row.sum() > 0 else bl_row
            nt_probs = nt_row / nt_row.sum() if nt_row.sum() > 0 else nt_row
            profile_ymax = max(profile_ymax, float(bl_probs.max()), float(nt_probs.max()))
            profile_rows.append((target_idx, bl_probs, nt_probs))
        profile_ymax = min(1.0, profile_ymax * 1.10)

        for row_idx, (target_idx, bl_probs, nt_probs) in enumerate(profile_rows):
            ax = fig_e.add_subplot(gs_es[row_idx, 0])
            ax.plot(
                angles,
                bl_probs,
                "-o",
                markersize=2.8,
                linewidth=0.95,
                color=SEMANTIC_PALETTE["learned"],
                label=routing_solver_short,
            )
            ax.plot(
                angles,
                nt_probs,
                "--s",
                markersize=2.5,
                linewidth=0.95,
                color=SEMANTIC_PALETTE["ablation"],
                label=routing_no_transformer_short,
            )
            ax.axvline(float(angles[target_idx]), color=SEMANTIC_PALETTE["physics"], linewidth=0.8, linestyle="--")
            ax.set_title(f"Conditional output @ {angles[target_idx]:.0f}\u00b0", fontsize=title_pt)
            ax.set_xlabel("Predicted angle (\u00b0)", fontsize=axis_label_pt)
            ax.set_ylabel("P(pred|true)", fontsize=axis_label_pt)
            ax.set_xticks(angles[tick_positions])
            ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
            ax.set_xlim(float(angles[0]), float(angles[-1]))
            ax.set_ylim(0, profile_ymax)
            ax.tick_params(axis="both", labelsize=tick_label_pt)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            if row_idx == 0:
                ax.legend(fontsize=legend_pt, frameon=False)
        add_panel_label(fig_e.axes[0], "e")
        all_paths.extend(
            save_outputs(
                fig_e,
                panel_dir / "fig05_panel_e_routing",
                typography=typography,
            )
        )
        plt.close(fig_e)

    # Panel f standalone (per-angle decoder benchmark)
    if panel_f_summary is not None:
        f_angles = np.asarray(panel_f_summary["angles"], dtype=float)
        guided_mean = _centered_moving_average(
            np.asarray(panel_f_summary["guided_mean"], dtype=float), window=3
        )
        router_bypass_mean = _centered_moving_average(
            np.asarray(panel_f_summary["router_bypass_mean"], dtype=float), window=3
        )
        omp_mean = _centered_moving_average(
            np.asarray(panel_f_summary["omp_mean"], dtype=float), window=3
        )
        dense_mean = _centered_moving_average(
            np.asarray(panel_f_summary["dense_mean"], dtype=float), window=3
        )
        fig_f = make_figure(
            width_mm=FIG05_STANDALONE["f"]["width_mm"],
            height_mm=FIG05_STANDALONE["f"]["height_mm"],
        )
        ax = fig_f.add_subplot(111)
        ax.plot(f_angles, guided_mean, "-", linewidth=1.2,
                color=SEMANTIC_PALETTE["learned"], label=diag_solver_short, zorder=4)
        ax.plot(f_angles, router_bypass_mean, "-", linewidth=1.1,
                color=SEMANTIC_PALETTE["ablation"], label=diag_no_transformer_short, zorder=3)
        ax.plot(f_angles, omp_mean, "-", linewidth=1.1,
                color=SEMANTIC_PALETTE["classical"], label=diag_omp_short, zorder=2)
        ax.plot(f_angles, dense_mean, ":", linewidth=1.1,
                color=DENSE_COLOR, label=diag_dense_short, zorder=1)
        ax.set_xlabel("Angle (\u00b0)")
        ax.set_ylabel("Mean P(correct)")
        ax.set_title("Per-angle decoder accuracy", fontsize=title_pt, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=legend_pt, frameon=False, loc="lower left", ncol=2, columnspacing=0.8, handletextpad=0.4)
        ax.tick_params(axis="both", labelsize=tick_label_pt)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax, "f")
        fig_f.subplots_adjust(**FIG05_STANDALONE["f"]["subplots_adjust"])
        all_paths.extend(
            save_outputs(
                fig_f,
                panel_dir / "fig05_panel_f_diagonal",
                typography=typography,
            )
        )
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
                "description": f"Four-curve SNR degradation ({snr_solver_full} vs {snr_no_transformer_full} vs {snr_omp_full} vs {snr_dense_full}).",
            },
            {
                "panel_id": "b",
                "title": "Structure alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_correlation.pdf",
                "provenance_mode": "data_backed",
                "description": "Physical-dictionary and learned-router correlation heatmaps providing the physical interpretability anchor.",
            },
            {
                "panel_id": "c",
                "title": "Classical confusion reference",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_selection.pdf",
                "provenance_mode": "data_backed",
                "description": f"Row-normalized confusion maps comparing {panel_c_omp_full} versus {panel_c_solver_full} as the classical context panel.",
            },
            {
                "panel_id": "d",
                "title": "Dense-vs-bypass confusion",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_confusion.pdf",
                "provenance_mode": "data_backed",
                "description": f"Row-normalized confusion matrices comparing {cm_dense_full} against {cm_no_transformer_full}.",
            },
            {
                "panel_id": "e",
                "title": "Representative-angle conditional outputs",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_routing.pdf",
                "provenance_mode": "data_backed",
                "description": f"Representative conditional output distributions at 55 and 100 degrees comparing {routing_solver_full} vs {routing_no_transformer_full}.",
            },
            {
                "panel_id": "f",
                "title": "Per-angle decoder accuracy",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_f_diagonal.pdf",
                "provenance_mode": "data_backed",
                "description": f"Five-seed clean mean P(correct) comparison across {diag_solver_full}, {diag_no_transformer_full}, {diag_omp_full}, and {diag_dense_full}, shown as a 3-angle centered moving-average display.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
