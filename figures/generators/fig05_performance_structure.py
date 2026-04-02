"""Figure 5 — Performance + manifold alignment + decoding behavior (5 panels).

Panel (a): 4-line SNR sweep benchmark (guided solver vs router-bypass vs OMP baseline vs dense routing)
Panel (b): Unified confusion-family block (OMP baseline vs guided solver vs dense routing vs router-bypass)
Panel (c): Measured manifold vs guided neighborhood structure
Panel (d): Four-angle conditional output distributions (guided solver vs router-bypass)
Panel (e): Per-angle decoder accuracy benchmark (guided solver vs router-bypass vs OMP baseline vs dense routing)

Data: figure4_data.json + modal_routing_val.npz + dictionary.npz + confusion metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

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
FIG05_COMPOSITE_GRID = dict(FIG05_GENERATOR["composite_grid"])
FIG05_TOP_ROW = dict(FIG05_GENERATOR["top_row"])
FIG05_BOTTOM_ROW = dict(FIG05_GENERATOR["bottom_row"])
FIG05_HEATMAP_STACK = dict(FIG05_GENERATOR["heatmap_stack"])
FIG05_ROUTING_STACK = dict(FIG05_GENERATOR["routing_stack"])
FIG05_STANDALONE = dict(FIG05_GENERATOR["standalone"])
FIG05_ROUTING_ANGLES = [55.0, 70.0, 95.0, 100.0]
FIG05_CORR_NORM = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=1.0)
FIG05_CORR_TICKS = [-0.1, 0.0, 0.5, 1.0]


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


def _prepare_conditional_output_profiles(
    guided_cm: np.ndarray,
    bypass_cm: np.ndarray,
    angles: np.ndarray,
    representative_angles: list[float],
) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], float]:
    """Return normalized representative rows and a shared y-limit."""
    profile_ymax = 0.0
    profile_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for target_angle in representative_angles:
        target_idx = int(np.argmin(np.abs(angles - target_angle)))
        guided_row = guided_cm[target_idx].astype(float)
        bypass_row = bypass_cm[target_idx].astype(float)
        guided_probs = guided_row / guided_row.sum() if guided_row.sum() > 0 else guided_row
        bypass_probs = bypass_row / bypass_row.sum() if bypass_row.sum() > 0 else bypass_row
        profile_ymax = max(
            profile_ymax,
            float(guided_probs.max()),
            float(bypass_probs.max()),
        )
        profile_rows.append((target_idx, guided_probs, bypass_probs))
    return profile_rows, min(1.0, profile_ymax * 1.10)


def _smoothed_mean_and_sem(summary: dict, prefix: str, n_runs: int = 5) -> tuple[np.ndarray, np.ndarray]:
    mean = _centered_moving_average(np.asarray(summary[f"{prefix}_mean"], dtype=float), window=3)
    std = _centered_moving_average(np.asarray(summary[f"{prefix}_std"], dtype=float), window=3)
    return mean, std / np.sqrt(float(n_runs))


def _plot_confusion_matrix(
    ax,
    matrix_norm: np.ndarray,
    title: str,
    *,
    title_pt: float,
    axis_label_pt: float,
    tick_label_pt: float,
    tick_positions: list[int],
    tick_labels: list[str],
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
    fontweight: str | None = None,
):
    im = ax.imshow(
        matrix_norm,
        cmap="viridis",
        aspect="equal",
        extent=[0, 37, 37, 0],
        vmin=0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=title_pt, fontweight=fontweight)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    if show_xticklabels:
        ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
        ax.tick_params(axis="x", labelbottom=True, labelsize=tick_label_pt, pad=1.0)
        for label in ax.get_xticklabels():
            label.set_visible(True)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)
    if show_yticklabels:
        ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
        ax.yaxis.set_ticks_position("left")
        ax.tick_params(axis="y", labelleft=True, labelsize=tick_label_pt, pad=1.0)
        for label in ax.get_yticklabels():
            label.set_visible(True)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelleft=False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_label_pt)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_label_pt)
    ax.tick_params(axis="both", length=2)
    return im


def _annotate_left_axis_values(
    ax,
    *,
    tick_positions: list[int],
    tick_labels: list[str],
    tick_label_pt: float,
    x: float = -0.06,
) -> None:
    """Render robust left-side numeric values when shared-axis tick labels get suppressed."""
    y0, y1 = ax.get_ylim()
    if y0 == y1:
        return
    for pos, label in zip(tick_positions, tick_labels):
        y_norm = (float(pos) - y0) / (y1 - y0)
        ax.text(
            x,
            y_norm,
            label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=tick_label_pt,
            clip_on=False,
        )


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
        fontsize=max(legend_pt - 0.45, 5.8),
        frameon=False,
        loc="lower left",
        ncol=1,
        columnspacing=0.6,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
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
    """Generate Figure 5 — performance, manifold alignment, and decoding behavior."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    annotation_pt = typography["annotation"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    panel_b_omp_full = get_bound_label("fig05", "b", "omp_side", label_type="full")
    panel_b_solver_full = get_bound_label(
        "fig05", "b", "learned_side", label_type="full"
    )
    panel_b_omp_short = get_bound_label("fig05", "b", "omp_side", label_type="short")
    panel_b_solver_short = get_bound_label(
        "fig05", "b", "learned_side", label_type="short"
    )
    panel_b_dense_short = get_bound_label("fig05", "b", "dense_side", label_type="short")
    panel_b_no_transformer_short = get_bound_label(
        "fig05", "b", "no_transformer_side", label_type="short"
    )
    panel_b_dense_full = get_bound_label("fig05", "b", "dense_side", label_type="full")
    panel_b_no_transformer_full = get_bound_label(
        "fig05", "b", "no_transformer_side", label_type="full"
    )
    routing_solver_short = get_bound_label(
        "fig05", "d", "baseline_dist", label_type="short"
    )
    routing_no_transformer_short = get_bound_label(
        "fig05", "d", "no_transformer_dist", label_type="short"
    )
    routing_solver_full = get_bound_label(
        "fig05", "d", "baseline_dist", label_type="full"
    )
    routing_no_transformer_full = get_bound_label(
        "fig05", "d", "no_transformer_dist", label_type="full"
    )
    diag_solver_short = get_bound_label("fig05", "e", "baseline_line", label_type="short")
    diag_no_transformer_short = get_bound_label(
        "fig05", "e", "no_transformer_line", label_type="short"
    )
    diag_omp_short = get_bound_label("fig05", "e", "omp_line", label_type="short")
    diag_dense_short = get_bound_label("fig05", "e", "dense_line", label_type="short")
    diag_solver_full = get_bound_label("fig05", "e", "baseline_line", label_type="full")
    diag_no_transformer_full = get_bound_label(
        "fig05", "e", "no_transformer_line", label_type="full"
    )
    diag_omp_full = get_bound_label("fig05", "e", "omp_line", label_type="full")
    diag_dense_full = get_bound_label("fig05", "e", "dense_line", label_type="full")
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
            f"Required Fig. 5 per-angle summary artifact not found: {panel_f_summary_path}"
        )
    panel_f_summary = dict(np.load(panel_f_summary_path, allow_pickle=True))

    omp_baseline_cm_norm = None
    baseline_cm_norm = None
    no_trans_cm_norm = None
    dense_cm_norm = None
    if omp_baseline_cm is not None:
        omp_baseline_cm_norm = _row_normalize_matrix(omp_baseline_cm)
    if baseline_cm is not None:
        baseline_cm_norm = _row_normalize_matrix(baseline_cm)
    if no_trans_cm is not None:
        no_trans_cm_norm = _row_normalize_matrix(no_trans_cm)
    if dense_cm is not None:
        dense_cm_norm = _row_normalize_matrix(dense_cm)

    # -----------------------------------------------------------------------
    # Build composite figure as a 5-panel, two-row journal layout:
    # top row = benchmark + unified confusion family + structure anchor;
    # bottom row = four-angle conditional outputs + per-angle benchmark.
    # -----------------------------------------------------------------------
    fig = make_figure(
        width_mm=FIG05_GENERATOR["composite_width_mm"],
        height_mm=FIG05_GENERATOR["composite_height_mm"],
    )
    gs_outer = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=FIG05_COMPOSITE_GRID["height_ratios"],
        hspace=FIG05_COMPOSITE_GRID["hspace"],
        left=FIG05_COMPOSITE_GRID["left"],
        right=FIG05_COMPOSITE_GRID["right"],
        bottom=FIG05_COMPOSITE_GRID["bottom"],
        top=FIG05_COMPOSITE_GRID["top"],
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1,
        3,
        subplot_spec=gs_outer[0, 0],
        width_ratios=FIG05_TOP_ROW["width_ratios"],
        wspace=FIG05_TOP_ROW["wspace"],
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_outer[1, 0],
        width_ratios=FIG05_BOTTOM_ROW["width_ratios"],
        wspace=FIG05_BOTTOM_ROW["wspace"],
    )

    # Panel (a): SNR sweep benchmark
    ax_a = _set_gid(fig.add_subplot(gs_top[0, 0]), "fig05.panel_a.main")
    _plot_snr_panel(
        ax_a,
        snr_curves,
        SNR_VARIANTS,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax_a.set_title("Noise robustness", fontsize=title_pt)
    add_panel_label(ax_a, "a", x=-0.15, y=1.02)

    # Panel (b): unified confusion-family block
    if (
        omp_baseline_cm_norm is None
        or baseline_cm_norm is None
        or dense_cm_norm is None
        or no_trans_cm_norm is None
    ):
        raise FileNotFoundError(
            "Fig. 5b requires OMP-baseline, guided-solver, dense-routing, and "
            "router-bypass confusion metrics."
        )
    ax_b_panel = _set_gid(fig.add_subplot(gs_top[0, 1]), "fig05.panel_b.block")
    ax_b_panel.set_axis_off()
    ax_b_panel.set_xticks([])
    ax_b_panel.set_yticks([])
    gs_b = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=gs_top[0, 1],
        width_ratios=[1.0, 1.0, FIG05_HEATMAP_STACK["colorbar_ratio"]],
        hspace=FIG05_HEATMAP_STACK["hspace"],
        wspace=FIG05_HEATMAP_STACK["wspace"],
    )

    ax_b1 = _set_gid(fig.add_subplot(gs_b[0, 0]), "fig05.b.top_left")
    im_b = _plot_confusion_matrix(
        ax_b1,
        omp_baseline_cm_norm,
        _titlecase_short_label(panel_b_omp_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        show_xticklabels=False,
        show_yticklabels=False,
        fontweight=None,
    )
    _annotate_left_axis_values(
        ax_b1,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        tick_label_pt=tick_label_pt,
    )

    ax_b2 = _set_gid(fig.add_subplot(gs_b[0, 1], sharex=ax_b1, sharey=ax_b1), "fig05.b.top_right")
    _plot_confusion_matrix(
        ax_b2,
        baseline_cm_norm,
        _titlecase_short_label(panel_b_solver_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        show_xticklabels=False,
        show_yticklabels=False,
    )

    ax_b3 = _set_gid(fig.add_subplot(gs_b[1, 0], sharex=ax_b1, sharey=ax_b1), "fig05.b.bottom_left")
    _plot_confusion_matrix(
        ax_b3,
        dense_cm_norm,
        _titlecase_short_label(panel_b_dense_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        show_yticklabels=False,
    )
    _annotate_left_axis_values(
        ax_b3,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        tick_label_pt=tick_label_pt,
    )

    ax_b4 = _set_gid(fig.add_subplot(gs_b[1, 1], sharex=ax_b1, sharey=ax_b1), "fig05.b.bottom_right")
    _plot_confusion_matrix(
        ax_b4,
        no_trans_cm_norm,
        _titlecase_short_label(panel_b_no_transformer_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        show_yticklabels=False,
    )

    cax_b = _set_gid(fig.add_subplot(gs_b[:, 2]), "fig05.b.colorbar")
    cbar = plt.colorbar(im_b, cax=cax_b)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    add_panel_label(ax_b_panel, "b", x=-0.04, y=1.02)
    ax_b_panel.text(
        0.50,
        -0.08,
        "Predicted DOA",
        va="center",
        ha="center",
        transform=ax_b_panel.transAxes,
        fontsize=axis_label_pt,
    )

    # Panel (c): measured manifold vs guided neighborhood map
    ax_c_panel = _set_gid(fig.add_subplot(gs_top[0, 2]), "fig05.panel_c.block")
    ax_c_panel.set_axis_off()
    ax_c_panel.set_xticks([])
    ax_c_panel.set_yticks([])
    gs_c = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=gs_top[0, 2],
        width_ratios=[1.0, FIG05_HEATMAP_STACK["colorbar_ratio"]],
        hspace=0.38,
        wspace=0.10,
    )

    ax_c1 = _set_gid(fig.add_subplot(gs_c[0, 0]), "fig05.panel_c.top")
    im_c = ax_c1.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_c1.set_title("Measured manifold", fontsize=title_pt, pad=6.0)
    ax_c1.set_xticks(tick_positions)
    ax_c1.set_xticklabels([])
    ax_c1.set_yticks(tick_positions)
    ax_c1.set_yticklabels([])
    ax_c1.tick_params(axis="both", length=2)

    ax_c2 = _set_gid(
        fig.add_subplot(gs_c[1, 0], sharex=ax_c1, sharey=ax_c1),
        "fig05.panel_c.bottom",
    )
    ax_c2.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_c2.set_title("Guided map", fontsize=title_pt, pad=6.0)
    ax_c2.set_xticks(tick_positions)
    ax_c2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_c2.set_yticks(tick_positions)
    ax_c2.set_yticklabels([])
    ax_c2.tick_params(axis="both", length=2)
    ax_c2.text(
        0.03,
        0.05,
        f"r = {pearson_r:.3f}",
        transform=ax_c2.transAxes,
        fontsize=annotation_pt,
        style="italic",
    )
    cax_c = _set_gid(fig.add_subplot(gs_c[:, 1]), "fig05.panel_c.colorbar")
    cbar_c = plt.colorbar(im_c, cax=cax_c)
    cbar_c.set_ticks(FIG05_CORR_TICKS)
    cbar_c.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar_c.ax.tick_params(labelsize=colorbar_tick_pt)
    cbar_c.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    add_panel_label(ax_c_panel, "c", x=-0.005, y=1.02)
    ax_c_panel.text(
        0.40,
        -0.08,
        "Angle (\u00b0)",
        va="center",
        ha="center",
        transform=ax_c_panel.transAxes,
        fontsize=axis_label_pt,
    )

    # Panel (d): Conditional output profiles at 4 representative angles
    if baseline_cm is not None and no_trans_cm is not None:
        ax_d_panel = _set_gid(fig.add_subplot(gs_bottom[0, 0]), "fig05.panel_d.block")
        ax_d_panel.set_axis_off()
        ax_d_panel.set_xticks([])
        ax_d_panel.set_yticks([])
        gs_d = gridspec.GridSpecFromSubplotSpec(
            2,
            2,
            subplot_spec=gs_bottom[0, 0],
            hspace=FIG05_ROUTING_STACK["hspace"],
            wspace=FIG05_ROUTING_STACK["wspace"],
        )
        profile_rows, profile_ymax = _prepare_conditional_output_profiles(
            baseline_cm,
            no_trans_cm,
            angles,
            FIG05_ROUTING_ANGLES,
        )

        first_ax = None
        for idx, (target_idx, bl_probs, nt_probs) in enumerate(profile_rows):
            row_idx, col_idx = divmod(idx, 2)
            inner_gid = (
                "fig05.d.top_left" if idx == 0 else
                "fig05.d.top_right" if idx == 1 else
                "fig05.d.bottom_left" if idx == 2 else
                "fig05.d.bottom_right"
            )
            if first_ax is None:
                ax_d = _set_gid(fig.add_subplot(gs_d[row_idx, col_idx]), inner_gid)
                first_ax = ax_d
            else:
                ax_d = _set_gid(
                    fig.add_subplot(gs_d[row_idx, col_idx], sharex=first_ax, sharey=first_ax),
                    inner_gid,
                )

            ax_d.plot(
                angles,
                bl_probs,
                "-o",
                markersize=2.5,
                linewidth=0.9,
                color=SEMANTIC_PALETTE["learned"],
                label=routing_solver_short,
                zorder=3,
            )
            ax_d.plot(
                angles,
                nt_probs,
                "--s",
                markersize=2.3,
                linewidth=0.9,
                color=SEMANTIC_PALETTE["ablation"],
                label=routing_no_transformer_short,
                zorder=2,
            )
            ax_d.axvline(
                float(angles[target_idx]),
                color=SEMANTIC_PALETTE["physics"],
                linewidth=0.8,
                linestyle="--",
                alpha=0.8,
            )
            ax_d.set_title(
                f"{angles[target_idx]:.0f}\u00b0",
                fontsize=title_pt,
                pad=2.0,
            )
            ax_d.set_xticks(angles[tick_positions])
            if row_idx == 0:
                ax_d.set_xticklabels([])
            else:
                ax_d.set_xticklabels(tick_labels, fontsize=tick_label_pt)
                ax_d.set_xlabel(
                    "Predicted angle (\u00b0)",
                    fontsize=axis_label_pt,
                    labelpad=-0.2,
                )
            ax_d.set_xlim(float(angles[0]), float(angles[-1]))
            ax_d.set_ylim(0, profile_ymax)
            ax_d.tick_params(axis="both", labelsize=tick_label_pt)
            ax_d.grid(axis="y", linestyle="--", alpha=0.25)
            if col_idx == 1:
                ax_d.tick_params(labelleft=False)
            if idx == 0:
                ax_d.legend(
                    fontsize=legend_pt,
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.98),
                    borderaxespad=0.0,
                )
        add_panel_label(ax_d_panel, "d", x=-0.04, y=1.02)
    else:
        ax_d_placeholder = fig.add_subplot(gs_bottom[0, 0])
        ax_d_placeholder.text(
            0.5,
            0.5,
            "Angle routing\ndata unavailable",
            transform=ax_d_placeholder.transAxes,
            ha="center",
            va="center",
            fontsize=annotation_pt,
        )
        ax_d_placeholder.set_axis_off()
        add_panel_label(ax_d_placeholder, "d", x=-0.1, y=1.06)

    # Panel (e): Per-angle decoder accuracy benchmark
    if panel_f_summary is not None:
        f_angles = np.asarray(panel_f_summary["angles"], dtype=float)
        if not np.array_equal(np.asarray(angles, dtype=float), f_angles):
            raise ValueError("Fig. 5e summary angle grid does not match the figure angle grid")
        guided_mean, guided_sem = _smoothed_mean_and_sem(panel_f_summary, "guided")
        router_bypass_mean, router_bypass_sem = _smoothed_mean_and_sem(panel_f_summary, "router_bypass")
        omp_mean, omp_sem = _smoothed_mean_and_sem(panel_f_summary, "omp")
        dense_mean, dense_sem = _smoothed_mean_and_sem(panel_f_summary, "dense")
        ax_e = _set_gid(fig.add_subplot(gs_bottom[0, 1]), "fig05.panel_e.main")
        ax_e.fill_between(
            angles,
            np.clip(guided_mean - guided_sem, 0.0, 1.0),
            np.clip(guided_mean + guided_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["learned"],
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
        ax_e.fill_between(
            angles,
            np.clip(router_bypass_mean - router_bypass_sem, 0.0, 1.0),
            np.clip(router_bypass_mean + router_bypass_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["ablation"],
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        ax_e.fill_between(
            angles,
            np.clip(omp_mean - omp_sem, 0.0, 1.0),
            np.clip(omp_mean + omp_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["classical"],
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        ax_e.fill_between(
            angles,
            np.clip(dense_mean - dense_sem, 0.0, 1.0),
            np.clip(dense_mean + dense_sem, 0.0, 1.0),
            color=DENSE_COLOR,
            alpha=0.06,
            linewidth=0,
            zorder=0,
        )
        ax_e.plot(
            angles,
            guided_mean,
            "-",
            linewidth=1.15,
            color=SEMANTIC_PALETTE["learned"],
            label=diag_solver_short,
            zorder=4,
        )
        ax_e.plot(
            angles,
            router_bypass_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["ablation"],
            label=diag_no_transformer_short,
            zorder=3,
        )
        ax_e.plot(
            angles,
            omp_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["classical"],
            label=diag_omp_short,
            zorder=2,
        )
        ax_e.plot(
            angles,
            dense_mean,
            ":",
            linewidth=1.05,
            color=DENSE_COLOR,
            label=diag_dense_short,
            zorder=1,
        )
        ax_e.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt, labelpad=-0.2)
        ax_e.set_title("Decoder accuracy", fontsize=title_pt)
        ax_e.set_ylim(0, 1.05)
        ax_e.legend(
            fontsize=legend_pt,
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(0.98, 0.02),
            ncol=1,
            columnspacing=0.6,
            handletextpad=0.4,
        )
        ax_e.tick_params(axis="both", labelsize=tick_label_pt)
        ax_e.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax_e, "e", x=-0.02, y=1.01)
    else:
        ax_e_placeholder = fig.add_subplot(gs_bottom[0, 2])
        ax_e_placeholder.text(
            0.5,
            0.5,
            "Per-angle data\nunavailable",
            transform=ax_e_placeholder.transAxes,
            ha="center",
            va="center",
            fontsize=annotation_pt,
        )
        ax_e_placeholder.set_axis_off()
        add_panel_label(ax_e_placeholder, "e", x=-0.1, y=1.06)

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

    # Panel b standalone (unified confusion family)
    fig_b = make_figure(
        width_mm=FIG05_STANDALONE["b"]["width_mm"],
        height_mm=FIG05_STANDALONE["b"]["height_mm"],
    )
    fig_b_grid = FIG05_STANDALONE["b"]["grid"]
    gs_bs = gridspec.GridSpec(
        2,
        3,
        figure=fig_b,
        width_ratios=[1.0, 1.0, fig_b_grid["colorbar_ratio"]],
        hspace=fig_b_grid["hspace"],
        wspace=fig_b_grid["wspace"],
        left=fig_b_grid["left"],
        right=fig_b_grid["right"],
        bottom=fig_b_grid["bottom"],
        top=fig_b_grid["top"],
    )
    ax1 = fig_b.add_subplot(gs_bs[0, 0])
    im_b = _plot_confusion_matrix(
        ax1,
        omp_baseline_cm_norm,
        _titlecase_short_label(panel_b_omp_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        ylabel="True DOA",
        show_xticklabels=False,
        show_yticklabels=False,
        fontweight="bold",
    )
    _annotate_left_axis_values(
        ax1,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        tick_label_pt=tick_label_pt,
    )
    add_panel_label(ax1, "b")
    ax2 = fig_b.add_subplot(gs_bs[0, 1], sharex=ax1, sharey=ax1)
    _plot_confusion_matrix(
        ax2,
        baseline_cm_norm,
        _titlecase_short_label(panel_b_solver_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        show_xticklabels=False,
        show_yticklabels=False,
        fontweight="bold",
    )
    ax3 = fig_b.add_subplot(gs_bs[1, 0], sharex=ax1, sharey=ax1)
    _plot_confusion_matrix(
        ax3,
        dense_cm_norm,
        _titlecase_short_label(panel_b_dense_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        xlabel="Predicted DOA",
        ylabel="True DOA",
        show_yticklabels=False,
        fontweight="bold",
    )
    _annotate_left_axis_values(
        ax3,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        tick_label_pt=tick_label_pt,
    )
    ax4 = fig_b.add_subplot(gs_bs[1, 1], sharex=ax1, sharey=ax1)
    _plot_confusion_matrix(
        ax4,
        no_trans_cm_norm,
        _titlecase_short_label(panel_b_no_transformer_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        xlabel="Predicted DOA",
        show_yticklabels=False,
        fontweight="bold",
    )
    cax = fig_b.add_subplot(gs_bs[:, 2])
    cbar = plt.colorbar(im_b, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig05_panel_b_confusion_family",
            typography=typography,
        )
    )
    plt.close(fig_b)

    # Panel c standalone (structure alignment)
    fig_c = make_figure(
        width_mm=FIG05_STANDALONE["c"]["width_mm"],
        height_mm=FIG05_STANDALONE["c"]["height_mm"],
    )
    fig_c_grid = FIG05_STANDALONE["c"]["grid"]
    gs_cs = gridspec.GridSpec(
        2,
        2,
        figure=fig_c,
        width_ratios=[1.0, fig_c_grid["colorbar_ratio"]],
        hspace=fig_c_grid["hspace"],
        wspace=fig_c_grid["wspace"],
        left=fig_c_grid["left"],
        right=fig_c_grid["right"],
        bottom=fig_c_grid["bottom"],
        top=fig_c_grid["top"],
    )
    ax1 = fig_c.add_subplot(gs_cs[0, 0])
    im_c = ax1.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax1.set_title("Measured manifold", fontsize=title_pt, fontweight="bold", pad=6.0)
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    add_panel_label(ax1, "c")
    ax2 = fig_c.add_subplot(gs_cs[1, 0])
    ax2.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax2.set_title("Guided map", fontsize=title_pt, fontweight="bold", pad=6.0)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax2.text(0.5, -0.22, f"r = {pearson_r:.3f}",
             transform=ax2.transAxes, fontsize=annotation_pt, ha="center", style="italic")
    cax = fig_c.add_subplot(gs_cs[:, 1])
    cbar = plt.colorbar(im_c, cax=cax)
    cbar.set_ticks(FIG05_CORR_TICKS)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    all_paths.extend(
        save_outputs(
            fig_c,
            panel_dir / "fig05_panel_c_structure_alignment",
            typography=typography,
        )
    )
    plt.close(fig_c)

    # Panel d standalone (four-angle conditional output profiles)
    if baseline_cm is not None and no_trans_cm is not None:
        fig_d = make_figure(
            width_mm=FIG05_STANDALONE["d"]["width_mm"],
            height_mm=FIG05_STANDALONE["d"]["height_mm"],
        )
        fig_d_grid = FIG05_STANDALONE["d"]["grid"]
        ax_panel = fig_d.add_subplot(111)
        ax_panel.set_axis_off()
        gs_ds = gridspec.GridSpec(
            2,
            2,
            figure=fig_d,
            hspace=fig_d_grid["hspace"],
            wspace=fig_d_grid["wspace"],
            left=fig_d_grid["left"],
            right=fig_d_grid["right"],
            bottom=fig_d_grid["bottom"],
            top=fig_d_grid["top"],
        )
        profile_rows, profile_ymax = _prepare_conditional_output_profiles(
            baseline_cm,
            no_trans_cm,
            angles,
            FIG05_ROUTING_ANGLES,
        )

        first_ax = None
        for idx, (target_idx, bl_probs, nt_probs) in enumerate(profile_rows):
            row_idx, col_idx = divmod(idx, 2)
            if first_ax is None:
                ax = fig_d.add_subplot(gs_ds[row_idx, col_idx])
                first_ax = ax
            else:
                ax = fig_d.add_subplot(gs_ds[row_idx, col_idx], sharex=first_ax, sharey=first_ax)
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
            ax.set_title(
                f"{angles[target_idx]:.0f}\u00b0",
                fontsize=title_pt,
                pad=2.0,
            )
            ax.set_xticks(angles[tick_positions])
            if row_idx == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
                ax.set_xlabel(
                    "Predicted angle (\u00b0)",
                    fontsize=axis_label_pt,
                    labelpad=-0.2,
                )
            ax.set_xlim(float(angles[0]), float(angles[-1]))
            ax.set_ylim(0, profile_ymax)
            ax.tick_params(axis="both", labelsize=tick_label_pt)
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            if col_idx == 1:
                ax.tick_params(labelleft=False)
            if idx == 0:
                ax.legend(
                    fontsize=legend_pt,
                    frameon=False,
                    loc="upper right",
                    bbox_to_anchor=(0.98, 0.98),
                    borderaxespad=0.0,
                )
        add_panel_label(ax_panel, "d")
        all_paths.extend(
            save_outputs(
                fig_d,
                panel_dir / "fig05_panel_d_conditional_outputs",
                typography=typography,
            )
        )
        plt.close(fig_d)

    # Panel e standalone (per-angle decoder benchmark)
    if panel_f_summary is not None:
        f_angles = np.asarray(panel_f_summary["angles"], dtype=float)
        guided_mean, guided_sem = _smoothed_mean_and_sem(panel_f_summary, "guided")
        router_bypass_mean, router_bypass_sem = _smoothed_mean_and_sem(panel_f_summary, "router_bypass")
        omp_mean, omp_sem = _smoothed_mean_and_sem(panel_f_summary, "omp")
        dense_mean, dense_sem = _smoothed_mean_and_sem(panel_f_summary, "dense")
        fig_e = make_figure(
            width_mm=FIG05_STANDALONE["e"]["width_mm"],
            height_mm=FIG05_STANDALONE["e"]["height_mm"],
        )
        ax = fig_e.add_subplot(111)
        ax.fill_between(
            f_angles,
            np.clip(guided_mean - guided_sem, 0.0, 1.0),
            np.clip(guided_mean + guided_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["learned"],
            alpha=0.12,
            linewidth=0,
            zorder=0,
        )
        ax.fill_between(
            f_angles,
            np.clip(router_bypass_mean - router_bypass_sem, 0.0, 1.0),
            np.clip(router_bypass_mean + router_bypass_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["ablation"],
            alpha=0.10,
            linewidth=0,
            zorder=0,
        )
        ax.fill_between(
            f_angles,
            np.clip(omp_mean - omp_sem, 0.0, 1.0),
            np.clip(omp_mean + omp_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["classical"],
            alpha=0.08,
            linewidth=0,
            zorder=0,
        )
        ax.fill_between(
            f_angles,
            np.clip(dense_mean - dense_sem, 0.0, 1.0),
            np.clip(dense_mean + dense_sem, 0.0, 1.0),
            color=DENSE_COLOR,
            alpha=0.06,
            linewidth=0,
            zorder=0,
        )
        ax.plot(f_angles, guided_mean, "-", linewidth=1.2,
                color=SEMANTIC_PALETTE["learned"], label=diag_solver_short, zorder=4)
        ax.plot(f_angles, router_bypass_mean, "-", linewidth=1.1,
                color=SEMANTIC_PALETTE["ablation"], label=diag_no_transformer_short, zorder=3)
        ax.plot(f_angles, omp_mean, "-", linewidth=1.1,
                color=SEMANTIC_PALETTE["classical"], label=diag_omp_short, zorder=2)
        ax.plot(f_angles, dense_mean, ":", linewidth=1.1,
                color=DENSE_COLOR, label=diag_dense_short, zorder=1)
        ax.set_xlabel("Angle (\u00b0)", labelpad=-0.2)
        ax.set_ylabel("Mean P(correct)")
        ax.set_title("Decoder accuracy", fontsize=title_pt, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(
            fontsize=legend_pt,
            frameon=False,
            loc="lower right",
            ncol=1,
            columnspacing=0.6,
            handletextpad=0.4,
        )
        ax.tick_params(axis="both", labelsize=tick_label_pt)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax, "e")
        fig_e.subplots_adjust(**FIG05_STANDALONE["e"]["subplots_adjust"])
        all_paths.extend(
            save_outputs(
                fig_e,
                panel_dir / "fig05_panel_e_decoder_accuracy",
                typography=typography,
            )
        )
        plt.close(fig_e)

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
                "title": "Unified confusion family",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_confusion_family.pdf",
                "provenance_mode": "data_backed",
                "description": (
                    f"Unified row-normalized confusion-family block comparing {panel_b_omp_full}, "
                    f"{panel_b_solver_full}, {panel_b_dense_full}, and {panel_b_no_transformer_full}."
                ),
            },
            {
                "panel_id": "c",
                "title": "Structure alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_structure_alignment.pdf",
                "provenance_mode": "data_backed",
                "description": "Correlation heatmaps showing that the guided neighborhood map follows the measured angle manifold rather than replacing it with an arbitrary classifier pattern.",
            },
            {
                "panel_id": "d",
                "title": "Four-angle conditional outputs",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_conditional_outputs.pdf",
                "provenance_mode": "data_backed",
                "description": f"Conditional output distributions at 55, 70, 95, and 100 degrees comparing {routing_solver_full} vs {routing_no_transformer_full}.",
            },
            {
                "panel_id": "e",
                "title": "Per-angle decoder accuracy",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_decoder_accuracy.pdf",
                "provenance_mode": "data_backed",
                "description": f"Five-seed clean mean P(correct) comparison across {diag_solver_full}, {diag_no_transformer_full}, {diag_omp_full}, and {diag_dense_full}, shown as a 3-angle centered moving-average display with light ±1 s.e.m. shading.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
