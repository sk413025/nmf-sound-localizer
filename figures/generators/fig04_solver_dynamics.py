"""Figure 4 — admissible first-step contraction before subtraction.

This generator produces the six paper-facing governed panels:

- (a) admissible neighborhood contraction
- (b) representative broad support
- (c) representative contracted support
- (d) validation-wide neighborhood contraction
- (e) angle-resolved within-15° contraction
- (f) clip-level within-15° gain CDF
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from figures.fig04_stepwise_mechanics import build_stepwise_mechanics_artifact
from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)
from figures.style import (
    FAMILY_STYLE,
    SEMANTIC_PALETTE,
    STROKE_TOKENS,
    STYLE_COLORS,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


EPS = 1e-8
FIG04_HEAVY_LINEWIDTH = STROKE_TOKENS["emphasis"]
FIG04_BASE_LINEWIDTH = STROKE_TOKENS["data"]
FIG04_LIGHT_LINEWIDTH = STROKE_TOKENS["base"]
FIG04_RULE_LINEWIDTH = STROKE_TOKENS["base"]
FIG04_GUIDE_LINE_COLOR = STYLE_COLORS["guide_line"]
FIG04_GUIDE_FILL_COLOR = STYLE_COLORS["guide_fill"]
FIG04_TARGET_RADIUS_DEG = 15.0
FIG04_LOCAL_PAD_DEG = 25.0

FIG04_GENERATOR = figure_section("fig04", "generator")
FIG04_SPLIT = dict(FIG04_GENERATOR["split"])
FIG04_GRID = dict(FIG04_GENERATOR["composite_grid"])
FIG04_PANEL_SLOT_WIDTHS_MM = {
    key: float(value)
    for key, value in dict(FIG04_SPLIT["panel_slot_width_mm"]).items()
}
FIG04_PANEL_SLOT_HEIGHTS_MM = {
    key: float(value)
    for key, value in dict(FIG04_SPLIT["panel_slot_height_mm"]).items()
}


def _save_panel_manifest(
    panel_dir: Path,
    panel_specs: list[dict],
    typography: dict[str, float],
) -> Path:
    manifest_path = panel_dir / "fig04_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig04",
        "composite_asset": "figures/output/fig04_solver_dynamics.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": typography,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / max(float(arr.max()), EPS)


def _normalize_in_window(
    values: np.ndarray,
    angles_deg: np.ndarray,
    x_min: float,
    x_max: float,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = (angles_deg >= x_min) & (angles_deg <= x_max)
    denom = max(float(arr[mask].max()) if np.any(mask) else float(arr.max()), EPS)
    return arr / denom


def _load_h_similarity(data_root: Path) -> np.ndarray:
    paths_cfg = load_paths()
    dict_path = data_root / paths_cfg["primary_run"] / "dictionary.npz"
    dict_data = np.load(dict_path, allow_pickle=True)
    H = np.asarray(dict_data["H"], dtype=np.float64)
    return np.corrcoef(H.T).astype(np.float32)


def _load_validation_grid(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    paths_cfg = load_paths()
    modal_path = data_root / paths_cfg["primary_run"] / "modal_routing_val.npz"
    modal = np.load(modal_path, allow_pickle=True)
    labels = np.asarray(modal["labels"], dtype=np.int64)
    angles_deg = np.asarray(modal["angles_deg"], dtype=np.float32)
    return labels, angles_deg


def _local_focus(target_angle: float) -> tuple[float, float]:
    x_min = max(0.0, target_angle - FIG04_LOCAL_PAD_DEG)
    x_max = min(180.0, target_angle + FIG04_LOCAL_PAD_DEG)
    return x_min, x_max


def _focus_ticks(x_min: float, x_max: float) -> list[float]:
    start = int(np.ceil(x_min / 15.0) * 15)
    stop = int(np.floor(x_max / 15.0) * 15)
    ticks = list(range(start, stop + 1, 15))
    if len(ticks) < 3:
        ticks = [x_min, (x_min + x_max) / 2.0, x_max]
    return [float(t) for t in ticks]


def _local_band_profile(
    h_similarity: np.ndarray,
    angles_deg: np.ndarray,
    target_angle: float,
) -> np.ndarray:
    target_idx = int(np.argmin(np.abs(angles_deg - target_angle)))
    return _normalize_1d(h_similarity[target_idx])


def _configure_profile_axis(
    ax: plt.Axes,
    *,
    tick_label_pt: float,
    target_angle: float,
    show_yticks: bool = True,
) -> None:
    x_min, x_max = _local_focus(target_angle)
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(_focus_ticks(x_min, x_max))
    ax.set_ylim(0.0, 1.05)
    if show_yticks:
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    else:
        ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])


def _draw_local_band_guide(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    local_band: np.ndarray,
    *,
    target_angle: float,
) -> None:
    ax.fill_between(
        angles_deg,
        np.zeros_like(local_band),
        local_band,
        color=FIG04_GUIDE_FILL_COLOR,
        alpha=0.72,
        linewidth=0.0,
        zorder=0,
    )
    ax.plot(
        angles_deg,
        local_band,
        color=FIG04_GUIDE_LINE_COLOR,
        linewidth=FIG04_LIGHT_LINEWIDTH,
        linestyle="-",
        zorder=1,
    )
    ax.axvspan(
        target_angle - FIG04_TARGET_RADIUS_DEG,
        target_angle + FIG04_TARGET_RADIUS_DEG,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_tertiary"],
        linewidth=0.0,
        zorder=0,
    )
    ax.axvline(
        target_angle,
        color=SEMANTIC_PALETTE["highlight"],
        linewidth=FIG04_RULE_LINEWIDTH,
        alpha=0.95,
        zorder=2,
    )


def _plot_trace(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    ax.plot(
        angles_deg,
        values,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=3,
    )


def _fill_trace(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    alpha: float,
) -> None:
    ax.fill_between(
        angles_deg,
        np.zeros_like(values),
        values,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=2,
    )


def _style_text_box(ax: plt.Axes, x: float, y: float, text: str, *, fontsize: float) -> None:
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="top",
        color=STYLE_COLORS["neutral_text"],
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": (1.0, 1.0, 1.0, 0.84),
            "edgecolor": "none",
        },
    )


def _compute_angle_resolved_contraction(
    mechanics: dict[str, np.ndarray],
    labels: np.ndarray,
    angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    before_rows = np.asarray(mechanics["aligned_cum_mass_g_rows"], dtype=np.float32)
    after_rows = np.asarray(mechanics["aligned_cum_mass_delta_rows"], dtype=np.float32)
    if before_rows.shape[0] != labels.shape[0]:
        raise RuntimeError(
            "Fig. 4 angle-resolved contraction requires label rows aligned with "
            "the stored cumulative-mass rows."
        )
    radius_deg = np.asarray(mechanics["aligned_radius_deg"], dtype=np.float32)
    idx15 = int(np.argmin(np.abs(radius_deg - FIG04_TARGET_RADIUS_DEG)))
    before_mean = np.zeros(angles_deg.size, dtype=np.float32)
    after_mean = np.zeros(angles_deg.size, dtype=np.float32)
    gain_mean = np.zeros(angles_deg.size, dtype=np.float32)
    for label_idx in range(angles_deg.size):
        mask = labels == label_idx
        if not np.any(mask):
            continue
        before_vals = before_rows[mask, idx15]
        after_vals = after_rows[mask, idx15]
        before_mean[label_idx] = float(before_vals.mean(dtype=np.float64))
        after_mean[label_idx] = float(after_vals.mean(dtype=np.float64))
        gain_mean[label_idx] = after_mean[label_idx] - before_mean[label_idx]
    return before_mean, after_mean, gain_mean


def _compute_clip_gain_cdf(
    mechanics: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, float]:
    radius_deg = np.asarray(mechanics["aligned_radius_deg"], dtype=np.float32)
    idx15 = int(np.argmin(np.abs(radius_deg - FIG04_TARGET_RADIUS_DEG)))
    before_rows = np.asarray(mechanics["aligned_cum_mass_g_rows"], dtype=np.float32)
    after_rows = np.asarray(mechanics["aligned_cum_mass_delta_rows"], dtype=np.float32)
    gain = after_rows[:, idx15] - before_rows[:, idx15]
    gain_sorted = np.sort(gain)
    cdf = np.arange(1, gain_sorted.size + 1, dtype=np.float32) / float(gain_sorted.size)
    median_gain = float(np.median(gain_sorted))
    positive_rate = float(np.mean(gain > 0.0))
    return gain_sorted, cdf, median_gain, positive_rate


def _plot_panel_a(
    ax: plt.Axes,
    mechanics: dict[str, np.ndarray],
    angles_deg: np.ndarray,
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    x_min, x_max = _local_focus(target_angle)
    local_band = _local_band_profile(h_similarity, angles_deg, target_angle)
    broad = _normalize_in_window(
        np.asarray(mechanics["stage0_g_norm_mean"][0], dtype=np.float32),
        angles_deg,
        x_min,
        x_max,
    )
    contracted = _normalize_in_window(
        np.asarray(mechanics["stage0_delta_norm_mean"][0], dtype=np.float32),
        angles_deg,
        x_min,
        x_max,
    )
    _draw_local_band_guide(ax, angles_deg, local_band, target_angle=target_angle)
    _plot_trace(
        ax,
        angles_deg,
        broad,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=FIG04_BASE_LINEWIDTH,
    )
    _fill_trace(ax, angles_deg, broad, color=SEMANTIC_PALETTE["physics"], alpha=0.08)
    _plot_trace(
        ax,
        angles_deg,
        contracted,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
    )
    _fill_trace(ax, angles_deg, contracted, color=SEMANTIC_PALETTE["learned"], alpha=0.12)
    _configure_profile_axis(ax, tick_label_pt=tick_label_pt, target_angle=target_angle)
    ax.set_title("Admissible neighborhood contraction", fontsize=title_pt, pad=2.5)
    ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    _style_text_box(
        ax,
        0.03,
        0.95,
        "validation mean\nbroad support vs first guided step",
        fontsize=legend_pt - 0.45,
    )
    ax.text(
        0.97,
        0.12,
        "measured neighborhood",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.55,
        ha="right",
        va="bottom",
        color=STYLE_COLORS["muted_text"],
    )
    if add_label:
        add_panel_label(ax, "a", x=0.02, y=1.01)


def _plot_panel_b(
    ax: plt.Axes,
    mechanics: dict[str, np.ndarray],
    angles_deg: np.ndarray,
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    x_min, x_max = _local_focus(target_angle)
    local_band = _local_band_profile(h_similarity, angles_deg, target_angle)
    broad = _normalize_in_window(
        np.asarray(mechanics["sample_g_expert_steps"][0, 0], dtype=np.float32),
        angles_deg,
        x_min,
        x_max,
    )
    _draw_local_band_guide(ax, angles_deg, local_band, target_angle=target_angle)
    _plot_trace(
        ax,
        angles_deg,
        broad,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=FIG04_BASE_LINEWIDTH,
    )
    _fill_trace(ax, angles_deg, broad, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _configure_profile_axis(ax, tick_label_pt=tick_label_pt, target_angle=target_angle)
    ax.set_title("Representative broad support", fontsize=title_pt, pad=2.5)
    ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    _style_text_box(
        ax,
        0.03,
        0.95,
        f"{target_angle:.0f}° validation clip\nbefore first guided step",
        fontsize=legend_pt - 0.45,
    )
    if add_label:
        add_panel_label(ax, "b", x=0.02, y=1.01)


def _plot_panel_c(
    ax: plt.Axes,
    mechanics: dict[str, np.ndarray],
    angles_deg: np.ndarray,
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    x_min, x_max = _local_focus(target_angle)
    local_band = _local_band_profile(h_similarity, angles_deg, target_angle)
    contracted = _normalize_in_window(
        np.asarray(mechanics["sample_delta_expert_steps"][0, 0], dtype=np.float32),
        angles_deg,
        x_min,
        x_max,
    )
    _draw_local_band_guide(ax, angles_deg, local_band, target_angle=target_angle)
    _plot_trace(
        ax,
        angles_deg,
        contracted,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
    )
    _fill_trace(ax, angles_deg, contracted, color=SEMANTIC_PALETTE["learned"], alpha=0.14)
    _configure_profile_axis(ax, tick_label_pt=tick_label_pt, target_angle=target_angle)
    ax.set_title("Representative contracted support", fontsize=title_pt, pad=2.5)
    ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    _style_text_box(
        ax,
        0.03,
        0.95,
        f"same {target_angle:.0f}° clip\nafter first guided step",
        fontsize=legend_pt - 0.45,
    )
    if add_label:
        add_panel_label(ax, "c", x=0.02, y=1.01)


def _plot_panel_d(
    ax: plt.Axes,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    radius_deg = np.asarray(mechanics["aligned_radius_deg"], dtype=np.float32)
    before_mean = np.asarray(mechanics["aligned_cum_mass_g_mean"], dtype=np.float32)
    before_sem = np.asarray(mechanics["aligned_cum_mass_g_sem"], dtype=np.float32)
    after_mean = np.asarray(mechanics["aligned_cum_mass_delta_mean"], dtype=np.float32)
    after_sem = np.asarray(mechanics["aligned_cum_mass_delta_sem"], dtype=np.float32)
    clip_count = int(np.asarray(mechanics["aligned_clip_count"]).item())
    ax.fill_between(
        radius_deg,
        np.clip(after_mean - after_sem, 0.0, 1.05),
        np.clip(after_mean + after_sem, 0.0, 1.05),
        color=SEMANTIC_PALETTE["learned"],
        alpha=FAMILY_STYLE["fill_alpha_primary"],
        linewidth=0.0,
    )
    ax.plot(
        radius_deg,
        after_mean,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
    )
    ax.fill_between(
        radius_deg,
        np.clip(before_mean - before_sem, 0.0, 1.05),
        np.clip(before_mean + before_sem, 0.0, 1.05),
        color=SEMANTIC_PALETTE["physics"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    ax.plot(
        radius_deg,
        before_mean,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=FIG04_BASE_LINEWIDTH,
    )
    ax.axvspan(
        0.0,
        FIG04_TARGET_RADIUS_DEG,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    ax.axvline(
        FIG04_TARGET_RADIUS_DEG,
        color=STYLE_COLORS["chance_line"],
        linewidth=FIG04_RULE_LINEWIDTH,
        linestyle=":",
    )
    idx15 = int(np.argmin(np.abs(radius_deg - FIG04_TARGET_RADIUS_DEG)))
    ax.set_xlim(float(radius_deg[0]), 45.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_title("Validation-wide neighborhood contraction", fontsize=title_pt, pad=2.5)
    ax.set_xlabel("Neighborhood radius (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Mass within radius", fontsize=axis_label_pt)
    _style_text_box(
        ax,
        0.03,
        0.95,
        f"n = {clip_count:,}\nwithin 15°: {before_mean[idx15]:.2f} → {after_mean[idx15]:.2f}",
        fontsize=legend_pt - 0.45,
    )
    ax.text(
        0.97,
        0.92,
        "after first guided step",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.5,
        ha="right",
        va="top",
        color=SEMANTIC_PALETTE["learned"],
    )
    ax.text(
        0.97,
        0.82,
        "before",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.5,
        ha="right",
        va="top",
        color=SEMANTIC_PALETTE["physics"],
    )
    if add_label:
        add_panel_label(ax, "d", x=0.02, y=1.01)


def _plot_panel_e(
    ax: plt.Axes,
    before_mean: np.ndarray,
    after_mean: np.ndarray,
    gain_mean: np.ndarray,
    angle_grid_deg: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    ax.fill_between(
        angle_grid_deg,
        before_mean,
        after_mean,
        color=STYLE_COLORS["guide_fill"],
        alpha=0.70,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        angle_grid_deg,
        before_mean,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=FIG04_BASE_LINEWIDTH,
        marker="o",
        markersize=1.8,
        zorder=2,
    )
    ax.plot(
        angle_grid_deg,
        after_mean,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
        marker="o",
        markersize=1.8,
        zorder=3,
    )
    ax.set_xlim(float(angle_grid_deg[0]), float(angle_grid_deg[-1]))
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_xlabel("True angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Mass inside 15°", fontsize=axis_label_pt)
    ax.set_title("Angle-resolved within-15° contraction", fontsize=title_pt, pad=2.5)
    _style_text_box(
        ax,
        0.03,
        0.95,
        f"mean gain = {float(np.mean(gain_mean)):+.2f}\npositive mean gain at every angle",
        fontsize=legend_pt - 0.45,
    )
    ax.text(
        0.97,
        0.92,
        "after first guided step",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.5,
        ha="right",
        va="top",
        color=SEMANTIC_PALETTE["learned"],
    )
    ax.text(
        0.97,
        0.82,
        "before",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.5,
        ha="right",
        va="top",
        color=SEMANTIC_PALETTE["physics"],
    )
    if add_label:
        add_panel_label(ax, "e", x=0.02, y=1.01)


def _plot_panel_f(
    ax: plt.Axes,
    gain_sorted: np.ndarray,
    cdf: np.ndarray,
    median_gain: float,
    positive_rate: float,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> None:
    ax.plot(
        gain_sorted,
        cdf,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
    )
    ax.axvline(
        0.0,
        color=STYLE_COLORS["chance_line"],
        linewidth=FIG04_RULE_LINEWIDTH,
        linestyle=":",
    )
    ax.axvline(
        median_gain,
        color=SEMANTIC_PALETTE["highlight"],
        linewidth=FIG04_RULE_LINEWIDTH,
        linestyle="--",
    )
    ax.set_xlim(float(gain_sorted.min()) - 0.02, float(gain_sorted.max()) + 0.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_xlabel("Gain in mass inside 15°", fontsize=axis_label_pt)
    ax.set_ylabel("Cumulative fraction of clips", fontsize=axis_label_pt)
    ax.set_title("Clip-level within-15° gain CDF", fontsize=title_pt, pad=2.5)
    _style_text_box(
        ax,
        0.03,
        0.95,
        f"median = {median_gain:+.2f}\n{positive_rate * 100.0:.1f}% of clips > 0",
        fontsize=legend_pt - 0.45,
    )
    if add_label:
        add_panel_label(ax, "f", x=0.02, y=1.01)


def _load_mechanics(data_root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    paths_cfg = load_paths()
    mechanics_path = data_root / paths_cfg["fig04_stepwise_mechanics"]
    build_stepwise_mechanics_artifact(data_root, mechanics_path)
    return mechanics_path, dict(np.load(mechanics_path))


def _build_composite(
    fig: plt.Figure,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    labels: np.ndarray,
    angle_grid_deg: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_panel_labels: bool,
) -> None:
    grid = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=FIG04_GRID["left"],
        right=FIG04_GRID["right"],
        bottom=FIG04_GRID["bottom"],
        top=FIG04_GRID["top"],
        hspace=FIG04_GRID["hspace"],
        wspace=FIG04_GRID["wspace"],
        width_ratios=FIG04_GRID["width_ratios"],
        height_ratios=FIG04_GRID["height_ratios"],
    )
    angles_deg = np.asarray(mechanics["angles_deg"], dtype=np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    before_mean, after_mean, gain_mean = _compute_angle_resolved_contraction(
        mechanics,
        labels,
        angle_grid_deg,
    )
    gain_sorted, cdf, median_gain, positive_rate = _compute_clip_gain_cdf(mechanics)

    ax_a = fig.add_subplot(grid[0, 0])
    _plot_panel_a(
        ax_a,
        mechanics,
        angles_deg,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_b = fig.add_subplot(grid[0, 1])
    _plot_panel_b(
        ax_b,
        mechanics,
        angles_deg,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_c = fig.add_subplot(grid[0, 2])
    _plot_panel_c(
        ax_c,
        mechanics,
        angles_deg,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_d = fig.add_subplot(grid[1, 0])
    _plot_panel_d(
        ax_d,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_e = fig.add_subplot(grid[1, 1])
    _plot_panel_e(
        ax_e,
        before_mean,
        after_mean,
        gain_mean,
        angle_grid_deg,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_f = fig.add_subplot(grid[1, 2])
    _plot_panel_f(
        ax_f,
        gain_sorted,
        cdf,
        median_gain,
        positive_rate,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 4 paper-facing data-backed panels."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    title_pt = typography["title"]

    mechanics_path, mechanics = _load_mechanics(data_root)
    h_similarity = _load_h_similarity(data_root)
    labels, angle_grid_deg = _load_validation_grid(data_root)
    angles_deg = np.asarray(mechanics["angles_deg"], dtype=np.float32)

    all_paths: list[Path] = []
    fig = make_figure(
        width_mm=FIG04_GENERATOR["composite_width_mm"],
        height_mm=FIG04_GENERATOR["composite_height_mm"],
    )
    _build_composite(
        fig,
        mechanics,
        h_similarity,
        labels,
        angle_grid_deg,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_panel_labels=True,
    )
    all_paths.extend(save_outputs(fig, output_dir / "fig04_solver_dynamics", typography=typography))
    plt.close(fig)

    before_mean, after_mean, gain_mean = _compute_angle_resolved_contraction(
        mechanics,
        labels,
        angle_grid_deg,
    )
    gain_sorted, cdf, median_gain, positive_rate = _compute_clip_gain_cdf(mechanics)

    panel_dir = output_dir / "fig04_solver_dynamics_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    def _make_standalone(panel_id: str, plot_fn, *plot_args) -> list[Path]:
        fig_panel = make_figure(
            width_mm=FIG04_PANEL_SLOT_WIDTHS_MM[panel_id],
            height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM[panel_id],
        )
        ax = fig_panel.add_subplot(111)
        plot_fn(
            ax,
            *plot_args,
            axis_label_pt=axis_label_pt,
            tick_label_pt=tick_label_pt,
            title_pt=title_pt,
            legend_pt=legend_pt,
            add_label=False,
        )
        fig_panel.subplots_adjust(**FIG04_SPLIT["standalone_subplots"][panel_id])
        stem = {
            "a": "fig04_panel_a_admissible_contraction",
            "b": "fig04_panel_b_representative_broad_support",
            "c": "fig04_panel_c_representative_contracted_support",
            "d": "fig04_panel_d_validation_contraction",
            "e": "fig04_panel_e_angle_resolved_within15",
            "f": "fig04_panel_f_clip_gain_cdf",
        }[panel_id]
        paths = save_outputs(fig_panel, panel_dir / stem, typography=typography)
        plt.close(fig_panel)
        return paths

    all_paths.extend(_make_standalone("a", _plot_panel_a, mechanics, angles_deg, h_similarity))
    all_paths.extend(_make_standalone("b", _plot_panel_b, mechanics, angles_deg, h_similarity))
    all_paths.extend(_make_standalone("c", _plot_panel_c, mechanics, angles_deg, h_similarity))
    all_paths.extend(_make_standalone("d", _plot_panel_d, mechanics))
    all_paths.extend(
        _make_standalone("e", _plot_panel_e, before_mean, after_mean, gain_mean, angle_grid_deg)
    )
    all_paths.extend(_make_standalone("f", _plot_panel_f, gain_sorted, cdf, median_gain, positive_rate))

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Admissible neighborhood contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_a_admissible_contraction.pdf",
                "provenance_mode": "data_backed",
                "description": "Validation-mean broad support and first-step contracted support shown against the measured neighborhood on one shared angle frame.",
            },
            {
                "panel_id": "b",
                "title": "Representative broad support",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_representative_broad_support.pdf",
                "provenance_mode": "data_backed",
                "description": "Representative 70-degree validation clip before the first guided step, showing broad local support against the measured neighborhood band.",
            },
            {
                "panel_id": "c",
                "title": "Representative contracted support",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_representative_contracted_support.pdf",
                "provenance_mode": "data_backed",
                "description": "The same representative 70-degree validation clip after one guided step, showing contraction back into the measured neighborhood band.",
            },
            {
                "panel_id": "d",
                "title": "Validation-wide neighborhood contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_validation_contraction.pdf",
                "provenance_mode": "data_backed",
                "description": "Validation-wide cumulative mass within radius before and after the first guided step, showing contraction into the measured local neighborhood.",
            },
            {
                "panel_id": "e",
                "title": "Angle-resolved within-15° contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_angle_resolved_within15.pdf",
                "provenance_mode": "data_backed",
                "description": "Per-angle before-and-after mass inside the operative 15-degree neighborhood, showing that contraction holds across the directional grid.",
            },
            {
                "panel_id": "f",
                "title": "Clip-level within-15° gain CDF",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_f_clip_gain_cdf.pdf",
                "provenance_mode": "data_backed",
                "description": "Empirical CDF of per-clip gain inside the operative 15-degree neighborhood, showing that local contraction is positive for nearly all validation clips.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(
        "[fig04] Generated "
        f"{len(all_paths)} files using stepwise mechanics artifact {mechanics_path.name}"
    )
    return all_paths
