"""Figure 5 — neighborhood preservation through final prediction (7 panels).

Panel (a): Neighborhood-preservation cascade on one shared radius axis.
Panel (b): Family-level final neighborhood-preservation curves.
Panel (c): Final-prediction locality comparison (OMP baseline vs guided solver).
Panel (d): Measured neighborhood geometry.
Panel (e): Guided neighborhood geometry.
Panel (f): Quantitative neighborhood-alignment closure.
Panel (g): Downstream performance consequences (SNR sweep + clean decoder accuracy).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)
from figures.naming import get_bound_label
from figures.style import (
    SEMANTIC_PALETTE,
    STYLE_COLORS,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "\u221e"]
DENSE_COLOR = "#4A4A4A"
LOCAL_RADII_DEG = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float32)
LOCAL_CALLOUT_DEG = 15.0
FIG05_CORR_NORM = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=1.0)
FIG05_CORR_TICKS = [-0.1, 0.0, 0.5, 1.0]

FIG05_GENERATOR = figure_section("fig05", "generator")
FIG05_COMPOSITE_GRID = dict(FIG05_GENERATOR["composite_grid"])
FIG05_TOP_ROW = dict(FIG05_GENERATOR["top_row"])
FIG05_MIDDLE_ROW = dict(FIG05_GENERATOR["middle_row"])
FIG05_BOTTOM_ROW = dict(FIG05_GENERATOR["bottom_row"])
FIG05_CONFUSION_PAIR = dict(FIG05_GENERATOR["confusion_pair"])
FIG05_QUANT_STACK = dict(FIG05_GENERATOR["quantitative_stack"])
FIG05_CONSEQUENCE_STACK = dict(FIG05_GENERATOR["consequence_stack"])
FIG05_STANDALONE = dict(FIG05_GENERATOR["standalone"])


def _set_gid(ax, gid: str):
    ax.set_gid(gid)
    return ax


def _titlecase_short_label(label: str) -> str:
    return label if label.isupper() else label[:1].upper() + label[1:]


def _row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def _grouped_match_surface(Y: np.ndarray, D: np.ndarray, n_experts: int) -> np.ndarray:
    n_atoms = int(D.shape[1] // n_experts)
    grouped = np.abs(np.asarray(Y, dtype=np.float32) @ np.asarray(D, dtype=np.float32))
    return grouped.reshape(len(Y), n_experts, n_atoms).sum(axis=2).astype(np.float32)


def _mass_within_radius_rows(
    support_rows: np.ndarray,
    angles: np.ndarray,
    truth_angles: np.ndarray,
    radii_deg: np.ndarray,
    *,
    normalize_rows: bool,
) -> np.ndarray:
    rows = np.asarray(support_rows, dtype=float)
    if normalize_rows:
        rows = _row_normalize_matrix(rows)
    angles = np.asarray(angles, dtype=float)
    truth_angles = np.asarray(truth_angles, dtype=float)
    radii_deg = np.asarray(radii_deg, dtype=float)
    out = np.zeros((rows.shape[0], radii_deg.size), dtype=np.float32)
    for idx, target_angle in enumerate(truth_angles):
        distance = np.abs(angles - float(target_angle))
        for ridx, radius in enumerate(radii_deg):
            out[idx, ridx] = float(rows[idx, distance <= radius].sum())
    return out


def _curve_mean_sem(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rows, dtype=np.float32)
    mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sem = np.zeros_like(mean)
    if arr.shape[0] > 1:
        sem = (arr.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(arr.shape[0])).astype(np.float32)
    return mean, sem


def _mass_value_at_radius(curve: np.ndarray, radii_deg: np.ndarray, radius_deg: float) -> float:
    ridx = int(np.argmin(np.abs(np.asarray(radii_deg, dtype=float) - float(radius_deg))))
    return float(np.asarray(curve, dtype=float)[ridx])


def _compute_global_correlation(
    routing_data: dict,
    dict_data: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores_expert = np.asarray(routing_data["scores_expert"], dtype=float)
    H = np.asarray(dict_data["H"], dtype=float)
    expert_corr = np.corrcoef(scores_expert.T)
    H_corr = np.corrcoef(H.T)
    structure_corr = float(np.corrcoef(H_corr.flatten(), expert_corr.flatten())[0, 1])
    return expert_corr, H_corr, structure_corr


def _local_band_scores(matrix: np.ndarray, angles: np.ndarray, radius_deg: float = 15.0) -> np.ndarray:
    scores: list[float] = []
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


def _centered_moving_average(series: np.ndarray, window: int = 3) -> np.ndarray:
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
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=title_pt, fontweight=fontweight)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    if show_xticklabels:
        ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
        ax.tick_params(axis="x", labelbottom=True, labelsize=tick_label_pt, pad=1.0)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)
    if show_yticklabels:
        ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
        ax.tick_params(axis="y", labelleft=True, labelsize=tick_label_pt, pad=1.0)
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


def _plot_support_curves(
    ax,
    *,
    radii_deg: np.ndarray,
    curves: list[tuple[str, np.ndarray, np.ndarray | None, str, str, float]],
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    title: str,
    annotation_pt: float,
    show_ylabel: bool = True,
    callout_radius_deg: float | None = LOCAL_CALLOUT_DEG,
    shade_local_regime: bool = True,
) -> None:
    if shade_local_regime:
        ax.axvspan(
            0.0,
            LOCAL_CALLOUT_DEG,
            color=SEMANTIC_PALETTE["highlight"],
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )
    ax.axvline(
        LOCAL_CALLOUT_DEG,
        color=STYLE_COLORS["guide_line"],
        linewidth=0.8,
        linestyle=":",
        alpha=0.9,
        zorder=1,
    )
    for label, mean, sem, color, linestyle, linewidth in curves:
        mean = np.asarray(mean, dtype=float)
        if sem is not None:
            sem = np.asarray(sem, dtype=float)
            ax.fill_between(
                radii_deg,
                np.clip(mean - sem, 0.0, 1.0),
                np.clip(mean + sem, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            radii_deg,
            mean,
            marker="o",
            markersize=2.8,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_title(title, fontsize=annotation_pt + 0.5)
    ax.set_xlim(float(radii_deg[0]), float(radii_deg[-1]))
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks(radii_deg)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Neighborhood radius (\u00b0)", fontsize=axis_label_pt)
    if show_ylabel:
        ax.set_ylabel("Mass within radius", fontsize=axis_label_pt)
    ax.tick_params(axis="both", labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(
        fontsize=max(legend_pt - 0.35, 5.8),
        frameon=False,
        loc="lower right",
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    if callout_radius_deg is not None:
        ax.text(
            0.02,
            0.94,
            f"within {int(callout_radius_deg)}\u00b0",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(annotation_pt - 0.2, 5.8),
            color=STYLE_COLORS["muted_text"],
        )


def _plot_snr_panel(
    ax,
    snr_curves: dict,
    variants: list[tuple[str, str, str]],
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
            stds.append(np.std(vals) if vals else 0.0)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        ax.plot(x_snr, means_arr, "-o", color=color, markersize=3.0, linewidth=1.0, label=label, zorder=3)
        ax.fill_between(x_snr, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.14, zorder=1)
    ax.set_xticks(x_snr)
    ax.set_xticklabels(SNR_DISPLAY)
    ax.set_xlabel("SNR (dB)", fontsize=axis_label_pt)
    ax.set_ylabel("Accuracy", fontsize=axis_label_pt)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        fontsize=max(legend_pt - 0.4, 5.8),
        frameon=False,
        loc="lower left",
        ncol=1,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def _plot_clean_accuracy_panel(
    ax,
    *,
    angles: np.ndarray,
    guided_mean: np.ndarray,
    guided_sem: np.ndarray,
    router_bypass_mean: np.ndarray,
    router_bypass_sem: np.ndarray,
    omp_mean: np.ndarray,
    omp_sem: np.ndarray,
    dense_mean: np.ndarray,
    dense_sem: np.ndarray,
    labels: dict[str, str],
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
) -> None:
    curves = [
        (guided_mean, guided_sem, SEMANTIC_PALETTE["learned"], labels["guided"], "-", 1.15, 0.12),
        (router_bypass_mean, router_bypass_sem, SEMANTIC_PALETTE["ablation"], labels["router_bypass"], "-", 1.05, 0.10),
        (omp_mean, omp_sem, SEMANTIC_PALETTE["classical"], labels["omp"], "-", 1.05, 0.08),
        (dense_mean, dense_sem, DENSE_COLOR, labels["dense"], ":", 1.05, 0.06),
    ]
    for mean, sem, color, label, linestyle, linewidth, alpha in curves:
        ax.fill_between(
            angles,
            np.clip(mean - sem, 0.0, 1.0),
            np.clip(mean + sem, 0.0, 1.0),
            color=color,
            alpha=alpha,
            linewidth=0.0,
            zorder=0,
        )
        ax.plot(angles, mean, linestyle=linestyle, linewidth=linewidth, color=color, label=label, zorder=3)
    ax.set_title("Clean decoder accuracy", fontsize=axis_label_pt + 0.3)
    ax.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        fontsize=max(legend_pt - 0.45, 5.8),
        frameon=False,
        loc="lower right",
        ncol=1,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict], typography: dict[str, float]) -> Path:
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
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 5 — structure-first bridge, alignment, and consequences."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    annotation_pt = typography["annotation"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    # Labels
    family_b_solver_short = get_bound_label("fig05", "b", "Baseline", label_type="short")
    family_b_router_short = get_bound_label("fig05", "b", "No Transformer", label_type="short")
    family_b_omp_short = get_bound_label("fig05", "b", "Fixed Heuristic", label_type="short")
    family_b_dense_short = get_bound_label("fig05", "b", "Dense Routing", label_type="short")
    perf_solver_short = get_bound_label("fig05", "g", "baseline_line", label_type="short")
    perf_router_short = get_bound_label("fig05", "g", "no_transformer_line", label_type="short")
    perf_omp_short = get_bound_label("fig05", "g", "omp_line", label_type="short")
    perf_dense_short = get_bound_label("fig05", "g", "dense_line", label_type="short")
    conf_omp_short = get_bound_label("fig05", "c", "omp_side", label_type="short")
    conf_solver_short = get_bound_label("fig05", "c", "learned_side", label_type="short")

    # Load primary artifacts
    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"
    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig05] SKIP: routing data not found at {run_dir}")
        return []

    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    angles = np.asarray(dict_data["angles"], dtype=float)
    labels = np.asarray(routing_data["labels"], dtype=int)
    Y_val = np.asarray(routing_data["Y_val"], dtype=np.float32)
    D = np.asarray(dict_data["D"], dtype=np.float32)

    expert_corr, H_corr, pearson_r = _compute_global_correlation(routing_data, dict_data)
    band_h = _local_band_scores(H_corr, angles)
    band_expert = _local_band_scores(expert_corr, angles)
    band_h_norm = _minmax_normalize(band_h)
    band_expert_norm = _minmax_normalize(band_expert)
    profile_corr = float(np.corrcoef(band_h_norm, band_expert_norm)[0, 1])
    profile_mae = float(np.mean(np.abs(band_h_norm - band_expert_norm)))

    # Stage-0 speech support
    speech_grouped = _grouped_match_surface(Y_val, D, len(angles))
    speech_stage0_rows = _mass_within_radius_rows(
        speech_grouped,
        angles,
        angles[labels],
        LOCAL_RADII_DEG,
        normalize_rows=True,
    )
    speech_stage0_mean, speech_stage0_sem = _curve_mean_sem(speech_stage0_rows)

    # Fig. 4 first-step neighborhood contraction artifact
    fig04_mechanics_path = data_root / paths_cfg["fig04_stepwise_mechanics"]
    if not fig04_mechanics_path.exists():
        raise FileNotFoundError(f"Required Fig. 4 mechanics artifact not found: {fig04_mechanics_path}")
    fig04_mechanics = dict(np.load(fig04_mechanics_path, allow_pickle=True))
    fig04_radii = np.asarray(fig04_mechanics["aligned_radius_deg"], dtype=float)
    match_idx = [int(np.argmin(np.abs(fig04_radii - radius))) for radius in LOCAL_RADII_DEG]
    step1_mean = np.asarray(fig04_mechanics["aligned_cum_mass_delta_mean"], dtype=float)[match_idx]
    step1_sem = np.asarray(fig04_mechanics["aligned_cum_mass_delta_sem"], dtype=float)[match_idx]

    # Final family confusion locality
    cm_cfg = paths_cfg["confusion_matrix"]
    confusion_paths = {
        "omp": data_root / cm_cfg["omp_baseline"],
        "guided": data_root / cm_cfg["baseline"],
        "router_bypass": data_root / cm_cfg["no_transformer"],
        "dense": data_root / cm_cfg["dense_routing"],
    }
    confusion_rows: dict[str, np.ndarray] = {}
    confusion_norm: dict[str, np.ndarray] = {}
    for key, path in confusion_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Required Fig. 5 confusion artifact not found: {path}")
        cm = np.asarray(np.load(path, allow_pickle=True)["confusion_matrix"], dtype=float)
        confusion_norm[key] = _row_normalize_matrix(cm)
        confusion_rows[key] = _mass_within_radius_rows(
            confusion_norm[key],
            angles,
            angles,
            LOCAL_RADII_DEG,
            normalize_rows=False,
        )
    family_curves = {key: _curve_mean_sem(rows) for key, rows in confusion_rows.items()}

    # Clean and noisy performance summaries
    summary_path = data_root / paths_cfg["fig05_panel_f_summary"]
    if not summary_path.exists():
        raise FileNotFoundError(f"Required Fig. 5 per-angle summary artifact not found: {summary_path}")
    panel_f_summary = dict(np.load(summary_path, allow_pickle=True))
    if not np.array_equal(np.asarray(panel_f_summary["angles"], dtype=float), angles):
        raise ValueError("Fig. 5 per-angle summary angle grid does not match the figure angle grid")
    guided_mean, guided_sem = _smoothed_mean_and_sem(panel_f_summary, "guided")
    router_bypass_mean, router_bypass_sem = _smoothed_mean_and_sem(panel_f_summary, "router_bypass")
    omp_mean, omp_sem = _smoothed_mean_and_sem(panel_f_summary, "omp")
    dense_mean, dense_sem = _smoothed_mean_and_sem(panel_f_summary, "dense")

    agg_path = data_root / paths_cfg["ablation_sweep"]["aggregated_json"]
    snr_curves: dict = {}
    if agg_path.exists():
        with open(agg_path, encoding="utf-8") as f:
            snr_curves = json.load(f).get("snr", {})

    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]

    support_bridge_npz = output_dir / "fig05_performance_structure_panels" / "fig05_neighborhood_bridge.npz"
    support_bridge_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        support_bridge_npz,
        radii_deg=LOCAL_RADII_DEG.astype(np.float32),
        speech_stage0_mean=speech_stage0_mean.astype(np.float32),
        speech_stage0_sem=speech_stage0_sem.astype(np.float32),
        guided_step1_mean=np.asarray(step1_mean, dtype=np.float32),
        guided_step1_sem=np.asarray(step1_sem, dtype=np.float32),
        guided_final_mean=np.asarray(family_curves["guided"][0], dtype=np.float32),
        guided_final_sem=np.asarray(family_curves["guided"][1], dtype=np.float32),
        router_bypass_final_mean=np.asarray(family_curves["router_bypass"][0], dtype=np.float32),
        router_bypass_final_sem=np.asarray(family_curves["router_bypass"][1], dtype=np.float32),
        omp_final_mean=np.asarray(family_curves["omp"][0], dtype=np.float32),
        omp_final_sem=np.asarray(family_curves["omp"][1], dtype=np.float32),
        dense_final_mean=np.asarray(family_curves["dense"][0], dtype=np.float32),
        dense_final_sem=np.asarray(family_curves["dense"][1], dtype=np.float32),
    )

    # Composite layout: top row (a,b), middle row (c,d,e), bottom row (f,g)
    fig = make_figure(
        width_mm=FIG05_GENERATOR["composite_width_mm"],
        height_mm=FIG05_GENERATOR["composite_height_mm"],
    )
    gs_outer = gridspec.GridSpec(
        3,
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
        1, 2, subplot_spec=gs_outer[0, 0], width_ratios=FIG05_TOP_ROW["width_ratios"], wspace=FIG05_TOP_ROW["wspace"]
    )
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[1, 0], width_ratios=FIG05_MIDDLE_ROW["width_ratios"], wspace=FIG05_MIDDLE_ROW["wspace"]
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[2, 0], width_ratios=FIG05_BOTTOM_ROW["width_ratios"], wspace=FIG05_BOTTOM_ROW["wspace"]
    )

    # Panel a
    ax_a = _set_gid(fig.add_subplot(gs_top[0, 0]), "fig05.panel_a.main")
    cascade_label_15 = _mass_value_at_radius(speech_stage0_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG)
    step1_label_15 = _mass_value_at_radius(step1_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG)
    final_guided_label_15 = _mass_value_at_radius(family_curves["guided"][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG)
    _plot_support_curves(
        ax_a,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (f"speech stage-0 ({cascade_label_15:.2f})", speech_stage0_mean, speech_stage0_sem, SEMANTIC_PALETTE["physics"], "--", 1.05),
            (f"first guided step ({step1_label_15:.2f})", step1_mean, step1_sem, SEMANTIC_PALETTE["learned"], "-", 1.20),
            (f"final guided prediction ({final_guided_label_15:.2f})", family_curves["guided"][0], family_curves["guided"][1], STYLE_COLORS["guide_line"], "-", 1.10),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Neighborhood-preservation cascade",
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax_a, "a", x=-0.15, y=1.02)

    # Panel b
    ax_b = _set_gid(fig.add_subplot(gs_top[0, 1]), "fig05.panel_b.main")
    _plot_support_curves(
        ax_b,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (f"{family_b_solver_short} ({_mass_value_at_radius(family_curves['guided'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["guided"][0], family_curves["guided"][1], SEMANTIC_PALETTE["learned"], "-", 1.20),
            (f"{family_b_router_short} ({_mass_value_at_radius(family_curves['router_bypass'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["router_bypass"][0], family_curves["router_bypass"][1], SEMANTIC_PALETTE["ablation"], "-", 1.05),
            (f"{family_b_omp_short} ({_mass_value_at_radius(family_curves['omp'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["omp"][0], family_curves["omp"][1], SEMANTIC_PALETTE["classical"], "-", 1.05),
            (f"{family_b_dense_short} ({_mass_value_at_radius(family_curves['dense'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["dense"][0], family_curves["dense"][1], DENSE_COLOR, ":", 1.05),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Final neighborhood preservation by family",
        annotation_pt=annotation_pt,
        show_ylabel=False,
    )
    add_panel_label(ax_b, "b", x=-0.12, y=1.02)

    # Panel c
    ax_c_panel = _set_gid(fig.add_subplot(gs_mid[0, 0]), "fig05.panel_c.block")
    ax_c_panel.set_axis_off()
    ax_c_panel.text(
        0.46,
        1.005,
        "Final prediction locality",
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
        confusion_norm["omp"],
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
    _annotate_left_axis_values(ax_c1, tick_positions=tick_positions, tick_labels=tick_labels, tick_label_pt=tick_label_pt)
    ax_c2 = ax_c_panel.inset_axes([0.475, 0.095, 0.445, 0.79], sharex=ax_c1, sharey=ax_c1)
    _plot_confusion_matrix(
        ax_c2,
        confusion_norm["guided"],
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

    # Panel d
    ax_d = _set_gid(fig.add_subplot(gs_mid[0, 1]), "fig05.panel_d.main")
    im_d = ax_d.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_d.set_title("Measured neighborhood geometry", fontsize=title_pt)
    ax_d.set_xticks(tick_positions)
    ax_d.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_d.set_yticks(tick_positions)
    ax_d.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_d.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_d.set_ylabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_d.tick_params(axis="both", length=2)
    add_panel_label(ax_d, "d", x=-0.12, y=1.02)

    # Panel e
    ax_e = _set_gid(fig.add_subplot(gs_mid[0, 2]), "fig05.panel_e.main")
    ax_e.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_e.set_title("Guided neighborhood geometry", fontsize=title_pt)
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

    # Panel f (stacked)
    ax_f_panel = _set_gid(fig.add_subplot(gs_bottom[0, 0]), "fig05.panel_f.block")
    ax_f_panel.set_axis_off()
    gs_f = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0, 0], hspace=FIG05_QUANT_STACK["hspace"])
    ax_f1 = _set_gid(fig.add_subplot(gs_f[0, 0]), "fig05.panel_f.top")
    ax_f1.plot(angles, band_h_norm, color=SEMANTIC_PALETTE["physics"], linewidth=1.10, label="Measured local-band score")
    ax_f1.plot(angles, band_expert_norm, color=SEMANTIC_PALETTE["learned"], linewidth=1.10, label="Learned local-band score")
    ax_f1.set_title("Quantitative neighborhood alignment", fontsize=title_pt)
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
    ax_f2.scatter(band_h_norm, band_expert_norm, s=24, color=STYLE_COLORS["dense_routing"], alpha=0.86)
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

    # Panel g (stacked consequences)
    ax_g_panel = _set_gid(fig.add_subplot(gs_bottom[0, 1]), "fig05.panel_g.block")
    ax_g_panel.set_axis_off()
    gs_g = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0, 1], hspace=FIG05_CONSEQUENCE_STACK["hspace"])
    ax_g1 = _set_gid(fig.add_subplot(gs_g[0, 0]), "fig05.panel_g.top")
    _plot_snr_panel(
        ax_g1,
        snr_curves,
        [
            ("No Type Bias", SEMANTIC_PALETTE["learned"], perf_solver_short),
            ("No Transformer", SEMANTIC_PALETTE["ablation"], perf_router_short),
            ("Fixed Heuristic", SEMANTIC_PALETTE["classical"], perf_omp_short),
            ("Dense Routing", DENSE_COLOR, perf_dense_short),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax_g1.set_title("Noise robustness", fontsize=axis_label_pt + 0.3)
    ax_g2 = _set_gid(fig.add_subplot(gs_g[1, 0]), "fig05.panel_g.bottom")
    _plot_clean_accuracy_panel(
        ax_g2,
        angles=angles,
        guided_mean=guided_mean,
        guided_sem=guided_sem,
        router_bypass_mean=router_bypass_mean,
        router_bypass_sem=router_bypass_sem,
        omp_mean=omp_mean,
        omp_sem=omp_sem,
        dense_mean=dense_mean,
        dense_sem=dense_sem,
        labels={
            "guided": perf_solver_short,
            "router_bypass": perf_router_short,
            "omp": perf_omp_short,
            "dense": perf_dense_short,
        },
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    add_panel_label(ax_g_panel, "g", x=-0.005, y=1.02)

    all_paths = save_outputs(fig, output_dir / "fig05_performance_structure", typography=typography)
    plt.close(fig)

    panel_dir = output_dir / "fig05_performance_structure_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Standalone panels
    # a
    fig_a = make_figure(width_mm=FIG05_STANDALONE["a"]["width_mm"], height_mm=FIG05_STANDALONE["a"]["height_mm"])
    ax = fig_a.add_subplot(111)
    _plot_support_curves(
        ax,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (f"speech stage-0 ({cascade_label_15:.2f})", speech_stage0_mean, speech_stage0_sem, SEMANTIC_PALETTE["physics"], "--", 1.05),
            (f"first guided step ({step1_label_15:.2f})", step1_mean, step1_sem, SEMANTIC_PALETTE["learned"], "-", 1.20),
            (f"final guided prediction ({final_guided_label_15:.2f})", family_curves["guided"][0], family_curves["guided"][1], STYLE_COLORS["guide_line"], "-", 1.10),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Neighborhood-preservation cascade",
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(**FIG05_STANDALONE["a"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig05_panel_a_neighborhood_cascade", typography=typography))
    plt.close(fig_a)

    # b
    fig_b = make_figure(width_mm=FIG05_STANDALONE["b"]["width_mm"], height_mm=FIG05_STANDALONE["b"]["height_mm"])
    ax = fig_b.add_subplot(111)
    _plot_support_curves(
        ax,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (f"{family_b_solver_short} ({_mass_value_at_radius(family_curves['guided'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["guided"][0], family_curves["guided"][1], SEMANTIC_PALETTE["learned"], "-", 1.20),
            (f"{family_b_router_short} ({_mass_value_at_radius(family_curves['router_bypass'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["router_bypass"][0], family_curves["router_bypass"][1], SEMANTIC_PALETTE["ablation"], "-", 1.05),
            (f"{family_b_omp_short} ({_mass_value_at_radius(family_curves['omp'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["omp"][0], family_curves["omp"][1], SEMANTIC_PALETTE["classical"], "-", 1.05),
            (f"{family_b_dense_short} ({_mass_value_at_radius(family_curves['dense'][0], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})", family_curves["dense"][0], family_curves["dense"][1], DENSE_COLOR, ":", 1.05),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Final neighborhood preservation by family",
        annotation_pt=annotation_pt,
        show_ylabel=False,
    )
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(**FIG05_STANDALONE["b"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig05_panel_b_family_neighborhood", typography=typography))
    plt.close(fig_b)

    # c
    fig_c = make_figure(width_mm=FIG05_STANDALONE["c"]["width_mm"], height_mm=FIG05_STANDALONE["c"]["height_mm"])
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
        confusion_norm["omp"],
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
        confusion_norm["guided"],
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
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig05_panel_c_prediction_locality", typography=typography))
    plt.close(fig_c)

    # d
    fig_d = make_figure(width_mm=FIG05_STANDALONE["d"]["width_mm"], height_mm=FIG05_STANDALONE["d"]["height_mm"])
    ax = fig_d.add_subplot(111)
    im = ax.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax.set_title("Measured neighborhood geometry", fontsize=title_pt, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(**FIG05_STANDALONE["d"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig05_panel_d_measured_structure", typography=typography))
    plt.close(fig_d)

    # e
    fig_e = make_figure(width_mm=FIG05_STANDALONE["e"]["width_mm"], height_mm=FIG05_STANDALONE["e"]["height_mm"])
    ax = fig_e.add_subplot(111)
    im = ax.imshow(expert_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax.set_title("Guided neighborhood geometry", fontsize=title_pt, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_xlabel("Angle (\u00b0)")
    cax = inset_axes(ax, width="3.6%", height="88%", loc="center right", borderpad=1.2)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_ticks(FIG05_CORR_TICKS)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(**FIG05_STANDALONE["e"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig05_panel_e_neighborhood_map", typography=typography))
    plt.close(fig_e)

    # f
    fig_f = make_figure(width_mm=FIG05_STANDALONE["f"]["width_mm"], height_mm=FIG05_STANDALONE["f"]["height_mm"])
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
    ax1.set_title("Quantitative neighborhood alignment", fontsize=title_pt, fontweight="bold")
    ax1.set_xlim(float(angles[0]), float(angles[-1]))
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_xticks(angles[tick_positions])
    ax1.set_xticklabels([])
    ax1.set_yticks([0.0, 0.5, 1.0])
    ax1.tick_params(axis="both", labelsize=tick_label_pt)
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.legend(fontsize=max(legend_pt - 0.4, 5.8), frameon=False, loc="lower left", handlelength=1.4, handletextpad=0.35, borderaxespad=0.2)
    ax1.text(0.03, 0.96, f"matrix r = {pearson_r:.3f}", transform=ax1.transAxes, ha="left", va="top", fontsize=annotation_pt)
    add_panel_label(ax1, "f")
    ax2 = fig_f.add_subplot(gs_fs[1, 0])
    ax2.scatter(band_h_norm, band_expert_norm, s=24, color=STYLE_COLORS["dense_routing"], alpha=0.86)
    ax2.plot([0.0, 1.0], [0.0, 1.0], color=STYLE_COLORS["guide_line"], linewidth=0.85, linestyle="--")
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_xticks([0.0, 0.5, 1.0])
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel("Measured score", fontsize=axis_label_pt)
    ax2.tick_params(axis="both", labelsize=tick_label_pt)
    ax2.grid(axis="both", linestyle="--", alpha=0.20)
    ax2.text(0.03, 0.96, f"profile r = {profile_corr:.3f}; MAE = {profile_mae:.3f}", transform=ax2.transAxes, ha="left", va="top", fontsize=annotation_pt)
    all_paths.extend(save_outputs(fig_f, panel_dir / "fig05_panel_f_quant_alignment", typography=typography))
    plt.close(fig_f)

    # g
    fig_g = make_figure(width_mm=FIG05_STANDALONE["g"]["width_mm"], height_mm=FIG05_STANDALONE["g"]["height_mm"])
    fig_g_grid = FIG05_STANDALONE["g"]["grid"]
    gs_gs = gridspec.GridSpec(
        2,
        1,
        figure=fig_g,
        hspace=fig_g_grid["hspace"],
        left=fig_g_grid["left"],
        right=fig_g_grid["right"],
        bottom=fig_g_grid["bottom"],
        top=fig_g_grid["top"],
    )
    ax1 = fig_g.add_subplot(gs_gs[0, 0])
    _plot_snr_panel(
        ax1,
        snr_curves,
        [
            ("No Type Bias", SEMANTIC_PALETTE["learned"], perf_solver_short),
            ("No Transformer", SEMANTIC_PALETTE["ablation"], perf_router_short),
            ("Fixed Heuristic", SEMANTIC_PALETTE["classical"], perf_omp_short),
            ("Dense Routing", DENSE_COLOR, perf_dense_short),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax1.set_title("Noise robustness", fontsize=axis_label_pt + 0.3)
    add_panel_label(ax1, "g")
    ax2 = fig_g.add_subplot(gs_gs[1, 0])
    _plot_clean_accuracy_panel(
        ax2,
        angles=angles,
        guided_mean=guided_mean,
        guided_sem=guided_sem,
        router_bypass_mean=router_bypass_mean,
        router_bypass_sem=router_bypass_sem,
        omp_mean=omp_mean,
        omp_sem=omp_sem,
        dense_mean=dense_mean,
        dense_sem=dense_sem,
        labels={
            "guided": perf_solver_short,
            "router_bypass": perf_router_short,
            "omp": perf_omp_short,
            "dense": perf_dense_short,
        },
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    all_paths.extend(save_outputs(fig_g, panel_dir / "fig05_panel_g_performance_consequences", typography=typography))
    plt.close(fig_g)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Neighborhood-preservation cascade",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_a_neighborhood_cascade.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared radius-based mass-within-neighborhood curves linking speech stage-0 support, first guided-step contraction, and final guided prediction locality.",
            },
            {
                "panel_id": "b",
                "title": "Final neighborhood preservation by family",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_family_neighborhood.pdf",
                "provenance_mode": "data_backed",
                "description": "Family-level final clean mass-within-radius curves derived from row-normalized confusion matrices, showing that guided prediction retains the strongest local concentration.",
            },
            {
                "panel_id": "c",
                "title": "Final prediction locality",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_prediction_locality.pdf",
                "provenance_mode": "data_backed",
                "description": "Row-normalized clean confusion comparison showing that guided solver remains concentrated near the target neighborhood, whereas the OMP baseline fractures more broadly off diagonal.",
            },
            {
                "panel_id": "d",
                "title": "Measured neighborhood geometry",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_measured_structure.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle-angle correlation matrix of calibrated H, showing the near-diagonal local band that defines the measured neighborhood geometry.",
            },
            {
                "panel_id": "e",
                "title": "Guided neighborhood geometry",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_neighborhood_map.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle-angle correlation map of the guided score surface, displayed on the same angle frame and correlation scale as panel d.",
            },
            {
                "panel_id": "f",
                "title": "Quantitative neighborhood alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_f_quant_alignment.pdf",
                "provenance_mode": "data_backed",
                "description": "Quantitative closure comparing measured and guided neighborhood geometry through normalized profile overlays and concordance scatter, with full-matrix and per-angle correlation summaries.",
            },
            {
                "panel_id": "g",
                "title": "Performance consequences",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_g_performance_consequences.pdf",
                "provenance_mode": "data_backed",
                "description": "Stacked noise-robustness and clean per-angle decoder-accuracy summaries for the same family comparison shown in panel b.",
            },
        ],
        typography,
    )
    all_paths.append(manifest)
    all_paths.append(support_bridge_npz)
    return all_paths


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    paths = load_paths()
    generate(REPO_ROOT, REPO_ROOT / paths["output_dir"])
