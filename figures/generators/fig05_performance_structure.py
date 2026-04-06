"""Figure 5 — Prediction structure + learned alignment + decoding behavior (6 panels).

Panel (a): 4-line SNR sweep benchmark (guided solver vs router-bypass vs OMP baseline vs dense routing)
Panel (b): Per-angle clean decoder accuracy benchmark
Panel (c): Prediction-locality confusion comparison (OMP baseline vs guided solver)
Panel (d): Measured local structure
Panel (e): Learned neighborhood-emphasis map
Panel (f): Quantitative structure-alignment closure

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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    SEMANTIC_PALETTE,
    STYLE_COLORS,
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
FIG05_CONFUSION_PAIR = dict(FIG05_GENERATOR["confusion_pair"])
FIG05_QUANT_STACK = dict(FIG05_GENERATOR["quantitative_stack"])
FIG05_STANDALONE = dict(FIG05_GENERATOR["standalone"])
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


def _local_band_scores(matrix: np.ndarray, angles: np.ndarray, radius_deg: float = 15.0) -> np.ndarray:
    """Mean local-band score at each angle, excluding the self-diagonal."""
    scores = []
    for idx, angle in enumerate(np.asarray(angles, dtype=float)):
        dist = np.abs(angles - angle)
        mask = (dist <= radius_deg) & (dist > 0)
        scores.append(float(np.asarray(matrix[idx, mask], dtype=float).mean()))
    return np.asarray(scores, dtype=float)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 0:
        raise ValueError("min-max normalization requires non-constant values")
    return (values - lo) / span


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

    conf_omp_full = get_bound_label("fig05", "c", "omp_side", label_type="full")
    conf_solver_full = get_bound_label("fig05", "c", "learned_side", label_type="full")
    conf_omp_short = get_bound_label("fig05", "c", "omp_side", label_type="short")
    conf_solver_short = get_bound_label("fig05", "c", "learned_side", label_type="short")
    panel_b_solver_short = get_bound_label("fig05", "b", "baseline_line", label_type="short")
    panel_b_no_transformer_short = get_bound_label(
        "fig05", "b", "no_transformer_line", label_type="short"
    )
    panel_b_omp_short = get_bound_label("fig05", "b", "omp_line", label_type="short")
    panel_b_dense_short = get_bound_label("fig05", "b", "dense_line", label_type="short")
    panel_b_solver_full = get_bound_label("fig05", "b", "baseline_line", label_type="full")
    panel_b_no_transformer_full = get_bound_label(
        "fig05", "b", "no_transformer_line", label_type="full"
    )
    panel_b_omp_full = get_bound_label("fig05", "b", "omp_line", label_type="full")
    panel_b_dense_full = get_bound_label("fig05", "b", "dense_line", label_type="full")
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
    band_h = _local_band_scores(H_corr, angles)
    band_expert = _local_band_scores(expert_corr, angles)
    band_h_norm = _minmax_normalize(band_h)
    band_expert_norm = _minmax_normalize(band_expert)
    profile_corr = float(np.corrcoef(band_h_norm, band_expert_norm)[0, 1])
    profile_mae = float(np.mean(np.abs(band_h_norm - band_expert_norm)))

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
    f_angles = np.asarray(panel_f_summary["angles"], dtype=float)
    if not np.array_equal(np.asarray(angles, dtype=float), f_angles):
        raise ValueError("Fig. 5 per-angle summary angle grid does not match the figure angle grid")
    guided_mean, guided_sem = _smoothed_mean_and_sem(panel_f_summary, "guided")
    router_bypass_mean, router_bypass_sem = _smoothed_mean_and_sem(panel_f_summary, "router_bypass")
    omp_mean, omp_sem = _smoothed_mean_and_sem(panel_f_summary, "omp")
    dense_mean, dense_sem = _smoothed_mean_and_sem(panel_f_summary, "dense")

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
    # Build composite figure as a 6-panel, two-row journal layout:
    # top row = robustness benchmark + clean decoder accuracy + prediction locality;
    # bottom row = measured structure + learned neighborhood map + quantitative closure.
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
        3,
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

    # Panel (b): per-angle clean decoder accuracy
    ax_b = _set_gid(fig.add_subplot(gs_top[0, 1]), "fig05.panel_b.main")
    ax_b.fill_between(
            angles,
            np.clip(guided_mean - guided_sem, 0.0, 1.0),
            np.clip(guided_mean + guided_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["learned"],
            alpha=0.12,
            linewidth=0,
            zorder=0,
    )
    ax_b.fill_between(
            angles,
            np.clip(router_bypass_mean - router_bypass_sem, 0.0, 1.0),
            np.clip(router_bypass_mean + router_bypass_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["ablation"],
            alpha=0.10,
            linewidth=0,
            zorder=0,
    )
    ax_b.fill_between(
            angles,
            np.clip(omp_mean - omp_sem, 0.0, 1.0),
            np.clip(omp_mean + omp_sem, 0.0, 1.0),
            color=SEMANTIC_PALETTE["classical"],
            alpha=0.08,
            linewidth=0,
            zorder=0,
    )
    ax_b.fill_between(
            angles,
            np.clip(dense_mean - dense_sem, 0.0, 1.0),
            np.clip(dense_mean + dense_sem, 0.0, 1.0),
            color=DENSE_COLOR,
            alpha=0.06,
            linewidth=0,
            zorder=0,
    )
    ax_b.plot(
            angles,
            guided_mean,
            "-",
            linewidth=1.15,
            color=SEMANTIC_PALETTE["learned"],
            label=panel_b_solver_short,
            zorder=4,
    )
    ax_b.plot(
            angles,
            router_bypass_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["ablation"],
            label=panel_b_no_transformer_short,
            zorder=3,
    )
    ax_b.plot(
            angles,
            omp_mean,
            "-",
            linewidth=1.05,
            color=SEMANTIC_PALETTE["classical"],
            label=panel_b_omp_short,
            zorder=2,
    )
    ax_b.plot(
            angles,
            dense_mean,
            ":",
            linewidth=1.05,
            color=DENSE_COLOR,
            label=panel_b_dense_short,
            zorder=1,
    )
    ax_b.set_title("Clean decoder accuracy", fontsize=title_pt)
    ax_b.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt, labelpad=-0.2)
    ax_b.set_ylim(0, 1.05)
    ax_b.legend(
        fontsize=legend_pt,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        ncol=1,
        columnspacing=0.6,
        handletextpad=0.4,
    )
    ax_b.tick_params(axis="both", labelsize=tick_label_pt)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_b, "b", x=-0.12, y=1.02)

    # Panel (c): prediction locality comparison (guided vs OMP)
    if omp_baseline_cm_norm is None or baseline_cm_norm is None:
        raise FileNotFoundError(
            "Fig. 5c requires guided-solver and OMP-baseline confusion metrics."
        )
    ax_c_panel = _set_gid(fig.add_subplot(gs_top[0, 2]), "fig05.panel_c.block")
    ax_c_panel.set_axis_off()
    ax_c_panel.set_xticks([])
    ax_c_panel.set_yticks([])
    ax_c_panel.text(
        0.46,
        1.005,
        "Prediction locality",
        transform=ax_c_panel.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_pt,
    )
    ax_c_panel.text(
        0.46,
        0.972,
        "Row-normalized clean confusion",
        transform=ax_c_panel.transAxes,
        ha="center",
        va="top",
        fontsize=max(annotation_pt - 0.2, 5.8),
        color=STYLE_COLORS["muted_text"],
    )
    ax_c1 = ax_c_panel.inset_axes([0.01, 0.095, 0.445, 0.79])
    im_c = _plot_confusion_matrix(
        ax_c1,
        omp_baseline_cm_norm,
        _titlecase_short_label(conf_omp_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        ylabel="True DOA",
        show_xticklabels=True,
        show_yticklabels=False,
        fontweight="bold",
    )
    _annotate_left_axis_values(
        ax_c1,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        tick_label_pt=tick_label_pt,
    )
    ax_c2 = ax_c_panel.inset_axes([0.475, 0.095, 0.445, 0.79], sharex=ax_c1, sharey=ax_c1)
    _plot_confusion_matrix(
        ax_c2,
        baseline_cm_norm,
        _titlecase_short_label(conf_solver_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        xlabel="Predicted DOA",
        show_yticklabels=False,
        fontweight="bold",
    )
    cax_c = ax_c_panel.inset_axes([0.94, 0.095, 0.014, 0.79])
    cbar_c = plt.colorbar(im_c, cax=cax_c)
    cbar_c.ax.tick_params(labelsize=colorbar_tick_pt)
    add_panel_label(ax_c_panel, "c", x=-0.005, y=1.02)

    # Panel (d): measured local structure
    ax_d = _set_gid(fig.add_subplot(gs_bottom[0, 0]), "fig05.panel_d.main")
    im_d = ax_d.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_d.set_title("Measured local structure", fontsize=title_pt)
    ax_d.set_xticks(tick_positions)
    ax_d.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_d.set_yticks(tick_positions)
    ax_d.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_d.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_d.set_ylabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_d.tick_params(axis="both", length=2)
    add_panel_label(ax_d, "d", x=-0.12, y=1.02)

    # Panel (e): learned neighborhood-emphasis map
    ax_e = _set_gid(fig.add_subplot(gs_bottom[0, 1]), "fig05.panel_e.main")
    ax_e.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_e.set_title("Guided-solver neighborhood-emphasis", fontsize=title_pt)
    ax_e.set_xticks(tick_positions)
    ax_e.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_e.set_yticks(tick_positions)
    ax_e.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_e.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_e.tick_params(axis="both", length=2)
    cax_e = inset_axes(ax_e, width="3.6%", height="88%", loc="center right", borderpad=1.2)
    cbar_e = plt.colorbar(im_d, cax=cax_e)
    cbar_e.set_ticks(FIG05_CORR_TICKS)
    cbar_e.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar_e.ax.tick_params(labelsize=colorbar_tick_pt)
    cbar_e.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    add_panel_label(ax_e, "e", x=-0.12, y=1.02)

    # Panel (f): quantitative structure-alignment closure
    ax_f_panel = _set_gid(fig.add_subplot(gs_bottom[0, 2]), "fig05.panel_f.block")
    ax_f_panel.set_axis_off()
    ax_f_panel.set_xticks([])
    ax_f_panel.set_yticks([])
    gs_f = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs_bottom[0, 2],
        hspace=FIG05_QUANT_STACK["hspace"],
    )
    ax_f1 = _set_gid(fig.add_subplot(gs_f[0, 0]), "fig05.panel_f.top")
    ax_f1.plot(
        angles,
        band_h_norm,
        color=SEMANTIC_PALETTE["physics"],
        linewidth=1.10,
        label="Measured local-band score",
    )
    ax_f1.plot(
        angles,
        band_expert_norm,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=1.10,
        label="Learned local-band score",
    )
    ax_f1.set_title("Quantitative structure alignment", fontsize=title_pt)
    ax_f1.set_xlim(float(angles[0]), float(angles[-1]))
    ax_f1.set_ylim(-0.02, 1.02)
    ax_f1.set_xticks(angles[tick_positions])
    ax_f1.set_xticklabels([])
    ax_f1.set_yticks([0.0, 0.5, 1.0])
    ax_f1.tick_params(axis="both", labelsize=tick_label_pt)
    ax_f1.grid(axis="y", linestyle="--", alpha=0.25)
    ax_f1.legend(
        fontsize=max(legend_pt - 0.4, 5.8),
        frameon=False,
        loc="lower left",
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    ax_f1.text(
        0.03,
        0.96,
        f"matrix r = {pearson_r:.3f}",
        transform=ax_f1.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_pt,
        color=SEMANTIC_PALETTE["physics"],
    )

    ax_f2 = _set_gid(fig.add_subplot(gs_f[1, 0]), "fig05.panel_f.bottom")
    ax_f2.scatter(
        band_h_norm,
        band_expert_norm,
        s=24,
        color=STYLE_COLORS["dense_routing"],
        alpha=0.86,
    )
    ax_f2.plot([0.0, 1.0], [0.0, 1.0], color=STYLE_COLORS["guide_line"], linewidth=0.85, linestyle="--")
    ax_f2.set_xlim(-0.02, 1.02)
    ax_f2.set_ylim(-0.02, 1.02)
    ax_f2.set_xticks([0.0, 0.5, 1.0])
    ax_f2.set_yticks([0.0, 0.5, 1.0])
    ax_f2.set_xlabel("Measured score", fontsize=axis_label_pt)
    ax_f2.tick_params(axis="both", labelsize=tick_label_pt)
    ax_f2.grid(axis="both", linestyle="--", alpha=0.20)
    ax_f2.text(
        0.03,
        0.96,
        f"profile r = {profile_corr:.3f}; MAE = {profile_mae:.3f}",
        transform=ax_f2.transAxes,
        ha="left",
        va="top",
        fontsize=annotation_pt,
        color=STYLE_COLORS["neutral_text"],
    )
    add_panel_label(ax_f_panel, "f", x=-0.005, y=1.02)

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

    # Panel b standalone (per-angle clean decoder accuracy)
    fig_b = make_figure(
        width_mm=FIG05_STANDALONE["b"]["width_mm"],
        height_mm=FIG05_STANDALONE["b"]["height_mm"],
    )
    ax = fig_b.add_subplot(111)
    ax.fill_between(
        angles,
        np.clip(guided_mean - guided_sem, 0.0, 1.0),
        np.clip(guided_mean + guided_sem, 0.0, 1.0),
        color=SEMANTIC_PALETTE["learned"],
        alpha=0.12,
        linewidth=0,
        zorder=0,
    )
    ax.fill_between(
        angles,
        np.clip(router_bypass_mean - router_bypass_sem, 0.0, 1.0),
        np.clip(router_bypass_mean + router_bypass_sem, 0.0, 1.0),
        color=SEMANTIC_PALETTE["ablation"],
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )
    ax.fill_between(
        angles,
        np.clip(omp_mean - omp_sem, 0.0, 1.0),
        np.clip(omp_mean + omp_sem, 0.0, 1.0),
        color=SEMANTIC_PALETTE["classical"],
        alpha=0.08,
        linewidth=0,
        zorder=0,
    )
    ax.fill_between(
        angles,
        np.clip(dense_mean - dense_sem, 0.0, 1.0),
        np.clip(dense_mean + dense_sem, 0.0, 1.0),
        color=DENSE_COLOR,
        alpha=0.06,
        linewidth=0,
        zorder=0,
    )
    ax.plot(angles, guided_mean, "-", linewidth=1.15, color=SEMANTIC_PALETTE["learned"], label=panel_b_solver_short, zorder=4)
    ax.plot(angles, router_bypass_mean, "-", linewidth=1.05, color=SEMANTIC_PALETTE["ablation"], label=panel_b_no_transformer_short, zorder=3)
    ax.plot(angles, omp_mean, "-", linewidth=1.05, color=SEMANTIC_PALETTE["classical"], label=panel_b_omp_short, zorder=2)
    ax.plot(angles, dense_mean, ":", linewidth=1.05, color=DENSE_COLOR, label=panel_b_dense_short, zorder=1)
    ax.set_xlabel("Angle (\u00b0)", labelpad=-0.2)
    ax.set_title("Clean decoder accuracy", fontsize=title_pt, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=legend_pt, frameon=False, loc="lower right", ncol=1, columnspacing=0.6, handletextpad=0.4)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(**FIG05_STANDALONE["b"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig05_panel_b_decoder_accuracy",
            typography=typography,
        )
    )
    plt.close(fig_b)

    # Panel c standalone (prediction locality comparison)
    fig_c = make_figure(
        width_mm=FIG05_STANDALONE["c"]["width_mm"],
        height_mm=FIG05_STANDALONE["c"]["height_mm"],
    )
    fig_c_grid = FIG05_STANDALONE["c"]["grid"]
    gs_cs = gridspec.GridSpec(
        1,
        3,
        figure=fig_c,
        width_ratios=[1.0, 1.0, fig_c_grid["colorbar_ratio"]],
        wspace=fig_c_grid["wspace"],
        left=fig_c_grid["left"],
        right=fig_c_grid["right"],
        bottom=fig_c_grid["bottom"],
        top=fig_c_grid["top"],
    )
    ax1 = fig_c.add_subplot(gs_cs[0, 0])
    im_c = _plot_confusion_matrix(
        ax1,
        omp_baseline_cm_norm,
        _titlecase_short_label(conf_omp_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        ylabel="True DOA",
        show_xticklabels=True,
        show_yticklabels=False,
        fontweight="bold",
    )
    _annotate_left_axis_values(ax1, tick_positions=tick_positions, tick_labels=tick_labels, tick_label_pt=tick_label_pt)
    add_panel_label(ax1, "c")
    ax2 = fig_c.add_subplot(gs_cs[0, 1], sharex=ax1, sharey=ax1)
    _plot_confusion_matrix(
        ax2,
        baseline_cm_norm,
        _titlecase_short_label(conf_solver_short),
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        tick_positions=tick_positions,
        tick_labels=tick_labels,
        xlabel="Predicted DOA",
        show_yticklabels=False,
        fontweight="bold",
    )
    cax = fig_c.add_subplot(gs_cs[0, 2])
    cbar = plt.colorbar(im_c, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    all_paths.extend(
        save_outputs(
            fig_c,
            panel_dir / "fig05_panel_c_prediction_locality",
            typography=typography,
        )
    )
    plt.close(fig_c)

    # Panel d standalone (measured local structure)
    fig_d = make_figure(
        width_mm=FIG05_STANDALONE["d"]["width_mm"],
        height_mm=FIG05_STANDALONE["d"]["height_mm"],
    )
    ax = fig_d.add_subplot(111)
    im_d = ax.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax.set_title("Measured local structure", fontsize=title_pt, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(**FIG05_STANDALONE["d"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_d,
            panel_dir / "fig05_panel_d_measured_structure",
            typography=typography,
        )
    )
    plt.close(fig_d)

    # Panel e standalone (learned neighborhood-emphasis map)
    fig_e = make_figure(
        width_mm=FIG05_STANDALONE["e"]["width_mm"],
        height_mm=FIG05_STANDALONE["e"]["height_mm"],
    )
    ax = fig_e.add_subplot(111)
    im_e = ax.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax.set_title("Guided-solver neighborhood-emphasis", fontsize=title_pt, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_xlabel("Angle (\u00b0)")
    cax = inset_axes(ax, width="3.6%", height="88%", loc="center right", borderpad=1.2)
    cbar = plt.colorbar(im_e, cax=cax)
    cbar.set_ticks(FIG05_CORR_TICKS)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(**FIG05_STANDALONE["e"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_e,
            panel_dir / "fig05_panel_e_neighborhood_map",
            typography=typography,
        )
    )
    plt.close(fig_e)

    # Panel f standalone (quantitative structure-alignment closure)
    fig_f = make_figure(
        width_mm=FIG05_STANDALONE["f"]["width_mm"],
        height_mm=FIG05_STANDALONE["f"]["height_mm"],
    )
    fig_f_grid = FIG05_STANDALONE["f"]["grid"]
    gs_fs = gridspec.GridSpec(
        2,
        1,
        figure=fig_f,
        hspace=fig_f_grid["hspace"],
        left=fig_f_grid["left"],
        right=fig_f_grid["right"],
        bottom=fig_f_grid["bottom"],
        top=fig_f_grid["top"],
    )
    ax1 = fig_f.add_subplot(gs_fs[0, 0])
    ax1.plot(angles, band_h_norm, color=SEMANTIC_PALETTE["physics"], linewidth=1.15, label="Measured local-band score")
    ax1.plot(angles, band_expert_norm, color=SEMANTIC_PALETTE["learned"], linewidth=1.15, label="Learned local-band score")
    ax1.set_title("Quantitative structure alignment", fontsize=title_pt, fontweight="bold")
    ax1.set_xlim(float(angles[0]), float(angles[-1]))
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks(angles[tick_positions])
    ax1.set_xticklabels([])
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.set_ylabel("Normalized score", fontsize=axis_label_pt)
    ax1.tick_params(axis="both", labelsize=tick_label_pt)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.legend(fontsize=max(legend_pt - 0.4, 5.8), frameon=False, loc="lower left")
    ax1.text(0.03, 0.96, f"matrix r = {pearson_r:.3f}", transform=ax1.transAxes, ha="left", va="top", fontsize=annotation_pt)
    add_panel_label(ax1, "f")
    ax2 = fig_f.add_subplot(gs_fs[1, 0])
    ax2.scatter(band_h_norm, band_expert_norm, s=24, color=STYLE_COLORS["dense_routing"], alpha=0.86)
    ax2.plot([0.0, 1.0], [0.0, 1.0], color=STYLE_COLORS["guide_line"], linewidth=0.85, linestyle="--")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel("Measured score")
    ax2.set_ylabel("Learned score")
    ax2.tick_params(axis="both", labelsize=tick_label_pt)
    ax2.grid(axis="both", linestyle="--", alpha=0.20)
    ax2.text(0.03, 0.96, f"profile r = {profile_corr:.3f}; MAE = {profile_mae:.3f}", transform=ax2.transAxes, ha="left", va="top", fontsize=annotation_pt)
    all_paths.extend(
        save_outputs(
            fig_f,
            panel_dir / "fig05_panel_f_quant_alignment",
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
                "title": "Clean decoder accuracy",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_decoder_accuracy.pdf",
                "provenance_mode": "data_backed",
                "description": f"Five-seed clean mean P(correct) comparison across {panel_b_solver_full}, {panel_b_no_transformer_full}, {panel_b_omp_full}, and {panel_b_dense_full}, shown as a 3-angle centered moving-average display with light ±1 s.e.m. shading.",
            },
            {
                "panel_id": "c",
                "title": "Prediction locality",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_prediction_locality.pdf",
                "provenance_mode": "data_backed",
                "description": f"Row-normalized clean confusion comparison showing that {conf_solver_full} remains locally concentrated around the target neighborhood, whereas {conf_omp_full} shows broader off-diagonal fracture.",
            },
            {
                "panel_id": "d",
                "title": "Measured local structure",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_measured_structure.pdf",
                "provenance_mode": "data_backed",
                "description": "Measured correlation matrix of the calibrated fingerprint space, showing the near-diagonal local band that defines the physical neighborhood structure.",
            },
            {
                "panel_id": "e",
                "title": "Neighborhood-emphasis map",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_neighborhood_map.pdf",
                "provenance_mode": "data_backed",
                "description": "Guided neighborhood-emphasis correlation map derived from the routed-score matrix, displayed on the same angle frame and correlation scale as panel d.",
            },
            {
                "panel_id": "f",
                "title": "Quantitative alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_f_quant_alignment.pdf",
                "provenance_mode": "data_backed",
                "description": "Quantitative closure comparing measured and learned local-band structure through normalized profile overlays and concordance scatter, with full-matrix and per-angle correlation summaries.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig05] Generated {len(all_paths)} files (Pearson r={pearson_r:.3f})")
    return all_paths
