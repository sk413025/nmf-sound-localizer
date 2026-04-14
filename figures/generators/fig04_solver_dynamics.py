"""Figure 4 — neighborhood-preserving admissibility before subtraction.

This generator produces the five paper-facing governed panels:

- (a) neighborhood admissibility strip
- (b) broad initial match
- (c) local contraction
- (d) validation-wide neighborhood contraction
- (e) within-15-degree mass gain
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
FIG04_MARKER_EDGEWIDTH = STROKE_TOKENS["grid"]
FIG04_POINT_MARKER_SIZE = FAMILY_STYLE["compact_marker_pt"] ** 2
FIG04_MEAN_MARKER_SIZE = FAMILY_STYLE["standard_marker_pt"]
FIG04_GUIDE_LINE_COLOR = STYLE_COLORS["guide_line"]
FIG04_GUIDE_FILL_COLOR = STYLE_COLORS["guide_fill"]
FIG04_TARGET_RADIUS_DEG = 15.0

FIG04_GENERATOR = figure_section("fig04", "generator")
FIG04_SPLIT = dict(FIG04_GENERATOR["split"])
FIG04_GRID = dict(FIG04_GENERATOR["composite_grid"])
FIG04_STANDALONE = dict(FIG04_SPLIT["standalone_subplots"])
FIG04_PANEL_SLOT_WIDTHS_MM = {
    key: float(value)
    for key, value in dict(FIG04_SPLIT["panel_slot_width_mm"]).items()
}
FIG04_PANEL_SLOT_HEIGHTS_MM = {
    key: float(value)
    for key, value in dict(FIG04_SPLIT["panel_slot_height_mm"]).items()
}
FIG04_MIDDLE_ROW = dict(FIG04_GENERATOR["middle_row"])
FIG04_BOTTOM_ROW = dict(FIG04_GENERATOR["bottom_row"])


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
    show_yticks: bool = True,
    local_focus: tuple[float, float] | None = None,
) -> None:
    if local_focus is None:
        ax.set_xlim(0.0, 180.0)
        ax.set_xticks([0, 45, 90, 135, 180])
    else:
        x_min, x_max = local_focus
        ax.set_xlim(x_min, x_max)
        ax.set_xticks([45, 60, 75, 90])
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
    guide_label: bool = False,
    legend_pt: float | None = None,
) -> None:
    ax.fill_between(
        angles_deg,
        np.zeros_like(local_band),
        local_band,
        color=FIG04_GUIDE_FILL_COLOR,
        alpha=0.65,
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
    if guide_label:
        ax.text(
            0.98,
            0.10,
            "measured neighborhood in H",
            transform=ax.transAxes,
            fontsize=legend_pt if legend_pt is not None else 5.6,
            ha="right",
            va="bottom",
            color=STYLE_COLORS["muted_text"],
        )


def _plot_trace(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "-",
    label: str | None = None,
) -> None:
    ax.plot(
        angles_deg,
        values,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=label,
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


def _plot_panel_a(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    outer = slot_spec.subgridspec(1, 2, width_ratios=[0.96, 1.64], wspace=0.14)
    left_ax = fig.add_subplot(outer[0, 0])
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    img = left_ax.imshow(
        h_similarity,
        extent=(0.0, 180.0, 0.0, 180.0),
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    img.set_clim(
        float(np.percentile(h_similarity, 5.0)),
        float(np.percentile(h_similarity, 99.0)),
    )
    left_ax.axvspan(
        target_angle - FIG04_TARGET_RADIUS_DEG,
        target_angle + FIG04_TARGET_RADIUS_DEG,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    left_ax.axhspan(
        target_angle - FIG04_TARGET_RADIUS_DEG,
        target_angle + FIG04_TARGET_RADIUS_DEG,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    left_ax.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=FIG04_RULE_LINEWIDTH)
    left_ax.axhline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=FIG04_RULE_LINEWIDTH)
    left_ax.set_xticks([0, 45, 90, 135, 180])
    left_ax.set_yticks([0, 45, 90, 135, 180])
    left_ax.tick_params(labelsize=tick_label_pt, length=2)
    left_ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    left_ax.set_ylabel("Angle (°)", fontsize=axis_label_pt)
    left_ax.set_title("Measured neighborhood", fontsize=title_pt, pad=2.0)
    left_ax.text(
        0.03,
        0.95,
        f"target = {target_angle:.0f}°",
        transform=left_ax.transAxes,
        fontsize=legend_pt - 0.2,
        va="top",
        ha="left",
        color="white",
    )

    right = outer[0, 1].subgridspec(1, 3, wspace=0.08)
    local_band = _local_band_profile(h_similarity, mechanics["angles_deg"], target_angle)
    stage_specs = [
        ("broad match", _normalize_1d(mechanics["stage0_g_norm_mean"][0]), SEMANTIC_PALETTE["physics"], "--", FIG04_BASE_LINEWIDTH),
        ("local gate", _normalize_1d(mechanics["stage0_w_theta_norm_mean"][0]), SEMANTIC_PALETTE["ablation"], ":", FIG04_LIGHT_LINEWIDTH),
        ("local update", _normalize_1d(mechanics["stage0_delta_norm_mean"][0]), SEMANTIC_PALETTE["learned"], "-", FIG04_HEAVY_LINEWIDTH),
    ]
    stage_axes: list[plt.Axes] = []
    focus = (45.0, 95.0)
    for idx, (title, values, color, linestyle, linewidth) in enumerate(stage_specs):
        ax = fig.add_subplot(right[0, idx])
        _draw_local_band_guide(ax, mechanics["angles_deg"], local_band, target_angle=target_angle)
        _plot_trace(ax, mechanics["angles_deg"], values, color=color, linestyle=linestyle, linewidth=linewidth)
        _fill_trace(ax, mechanics["angles_deg"], values, color=color, alpha=0.08 if idx < 2 else 0.12)
        _configure_profile_axis(ax, tick_label_pt=tick_label_pt - 0.2, show_yticks=(idx == 0), local_focus=focus)
        ax.set_title(title, fontsize=title_pt - 0.35, pad=1.2, color=STYLE_COLORS["muted_text"])
        if idx == 0:
            ax.set_ylabel("Normalized support", fontsize=axis_label_pt - 0.3)
        stage_axes.append(ax)

    if add_label:
        add_panel_label(left_ax, "a", x=0.02, y=1.01)
    return [left_ax, *stage_axes]


def _plot_panel_b(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    angles_deg = np.asarray(mechanics["angles_deg"], dtype=np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    local_band = _local_band_profile(h_similarity, angles_deg, target_angle)
    x_min, x_max = 45.0, 95.0
    broad = _normalize_in_window(
        np.asarray(mechanics["sample_g_expert_steps"][0, 0], dtype=np.float32),
        angles_deg,
        x_min,
        x_max,
    )
    _draw_local_band_guide(
        ax,
        angles_deg,
        local_band,
        target_angle=target_angle,
        guide_label=True,
        legend_pt=legend_pt - 0.2,
    )
    _plot_trace(
        ax,
        angles_deg,
        broad,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=FIG04_BASE_LINEWIDTH,
        label="broad match",
    )
    _fill_trace(ax, angles_deg, broad, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _configure_profile_axis(ax, tick_label_pt=tick_label_pt, local_focus=(x_min, x_max))
    ax.set_title("Broad initial match", fontsize=title_pt, pad=2.0)
    ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.94,
        "representative speech clip",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.2,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
    )
    ax.legend(frameon=False, fontsize=legend_pt - 0.2, loc="upper center")
    if add_label:
        add_panel_label(ax, "b", x=0.02, y=1.01)
    return [ax]


def _plot_panel_c(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    angles_deg = np.asarray(mechanics["angles_deg"], dtype=np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    local_band = _local_band_profile(h_similarity, angles_deg, target_angle)
    x_min, x_max = 45.0, 95.0
    broad = _normalize_in_window(np.asarray(mechanics["sample_g_expert_steps"][0, 0], dtype=np.float32), angles_deg, x_min, x_max)
    gate = _normalize_in_window(np.asarray(mechanics["sample_w_theta_steps"][0, 0], dtype=np.float32), angles_deg, x_min, x_max)
    update = _normalize_in_window(np.asarray(mechanics["sample_delta_expert_steps"][0, 0], dtype=np.float32), angles_deg, x_min, x_max)
    _draw_local_band_guide(ax, angles_deg, local_band, target_angle=target_angle)
    _plot_trace(ax, angles_deg, broad, color=SEMANTIC_PALETTE["physics"], linestyle="--", linewidth=FIG04_BASE_LINEWIDTH, label="broad match")
    _plot_trace(ax, angles_deg, gate, color=SEMANTIC_PALETTE["ablation"], linestyle=":", linewidth=FIG04_LIGHT_LINEWIDTH, label="local gate")
    _plot_trace(ax, angles_deg, update, color=SEMANTIC_PALETTE["learned"], linewidth=FIG04_HEAVY_LINEWIDTH, label="local update")
    _fill_trace(ax, angles_deg, gate, color=SEMANTIC_PALETTE["ablation"], alpha=0.06)
    _fill_trace(ax, angles_deg, update, color=SEMANTIC_PALETTE["learned"], alpha=0.10)
    _configure_profile_axis(ax, tick_label_pt=tick_label_pt, local_focus=(x_min, x_max))
    ax.set_title("Local contraction", fontsize=title_pt, pad=2.0)
    ax.set_xlabel("Angle (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    ax.legend(frameon=False, fontsize=legend_pt - 0.2, loc="upper right", handlelength=2.0)
    if add_label:
        add_panel_label(ax, "c", x=0.02, y=1.01)
    return [ax]


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
    g_mean = np.asarray(mechanics["aligned_cum_mass_g_mean"], dtype=np.float32)
    g_sem = np.asarray(mechanics["aligned_cum_mass_g_sem"], dtype=np.float32)
    delta_mean = np.asarray(mechanics["aligned_cum_mass_delta_mean"], dtype=np.float32)
    delta_sem = np.asarray(mechanics["aligned_cum_mass_delta_sem"], dtype=np.float32)
    clip_count = int(np.asarray(mechanics["aligned_clip_count"]).item())
    ax.fill_between(
        radius_deg,
        np.clip(delta_mean - delta_sem, 0.0, 1.05),
        np.clip(delta_mean + delta_sem, 0.0, 1.05),
        color=SEMANTIC_PALETTE["learned"],
        alpha=FAMILY_STYLE["fill_alpha_primary"],
        linewidth=0.0,
    )
    ax.plot(radius_deg, delta_mean, color=SEMANTIC_PALETTE["learned"], linewidth=FIG04_HEAVY_LINEWIDTH, label="after first guided step")
    ax.fill_between(
        radius_deg,
        np.clip(g_mean - g_sem, 0.0, 1.05),
        np.clip(g_mean + g_sem, 0.0, 1.05),
        color=SEMANTIC_PALETTE["physics"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    ax.plot(radius_deg, g_mean, color=SEMANTIC_PALETTE["physics"], linestyle="--", linewidth=FIG04_BASE_LINEWIDTH, label="before")
    ax.axvspan(0.0, FIG04_TARGET_RADIUS_DEG, color=SEMANTIC_PALETTE["highlight"], alpha=FAMILY_STYLE["fill_alpha_secondary"], linewidth=0.0)
    ax.axvline(FIG04_TARGET_RADIUS_DEG, color=STYLE_COLORS["chance_line"], linewidth=FIG04_RULE_LINEWIDTH, linestyle=":")
    idx15 = int(np.argmin(np.abs(radius_deg - FIG04_TARGET_RADIUS_DEG)))
    ax.text(
        0.03,
        0.95,
        f"validation-wide, n = {clip_count:,}\nwithin 15°: {g_mean[idx15]:.2f} → {delta_mean[idx15]:.2f}",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.2,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
    )
    ax.set_xlim(0.0, 45.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_title("Validation-wide neighborhood contraction", fontsize=title_pt, pad=2.0)
    ax.set_xlabel("Neighborhood radius (°)", fontsize=axis_label_pt)
    ax.set_ylabel("Mass within radius", fontsize=axis_label_pt)
    ax.legend(frameon=False, fontsize=legend_pt - 0.2, loc="lower right")
    if add_label:
        add_panel_label(ax, "d", x=0.02, y=1.01)


def _plot_panel_e(
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
    idx15 = int(np.argmin(np.abs(radius_deg - FIG04_TARGET_RADIUS_DEG)))
    before = np.asarray(mechanics["aligned_cum_mass_g_rows"], dtype=np.float32)[:, idx15]
    after = np.asarray(mechanics["aligned_cum_mass_delta_rows"], dtype=np.float32)[:, idx15]
    data = [before, after]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=False, showextrema=False)
    violin_colors = [SEMANTIC_PALETTE["physics"], SEMANTIC_PALETTE["learned"]]
    for body, color in zip(parts["bodies"], violin_colors, strict=False):
        body.set_facecolor(color)
        body.set_alpha(0.55)
    parts["cmeans"].set_color(STYLE_COLORS["neutral_text"])
    parts["cmeans"].set_linewidth(FIG04_BASE_LINEWIDTH)

    rng = np.random.default_rng(42)
    for xpos, values, color in zip([0, 1], data, violin_colors, strict=False):
        sample = values if values.size <= 300 else rng.choice(values, size=300, replace=False)
        jitter = rng.uniform(-0.08, 0.08, size=sample.size)
        ax.scatter(
            np.full(sample.size, xpos, dtype=np.float32) + jitter,
            sample,
            color=color,
            s=FIG04_POINT_MARKER_SIZE * 0.35,
            alpha=0.18,
            linewidths=0.0,
            zorder=3,
        )
        ax.text(
            xpos,
            min(float(np.mean(values)) + 0.06, 1.02),
            f"{np.mean(values):.2f}",
            fontsize=legend_pt,
            fontweight="bold",
            va="bottom",
            ha="center",
            color=color,
        )

    ax.axhspan(0.0, 0.2, color=SEMANTIC_PALETTE["highlight"], alpha=FAMILY_STYLE["fill_alpha_secondary"], linewidth=0.0, zorder=0)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["before", "after"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=tick_label_pt, length=0)
    ax.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_title("Within-15° mass gain", fontsize=title_pt, pad=2.0)
    ax.set_xlabel("First guided step", fontsize=axis_label_pt)
    ax.set_ylabel("Mass inside 15°", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.96,
        "validation clips",
        transform=ax.transAxes,
        fontsize=legend_pt - 0.2,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
    )
    if add_label:
        add_panel_label(ax, "e", x=0.02, y=1.01)


def _load_mechanics(data_root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    paths_cfg = load_paths()
    mechanics_path = data_root / paths_cfg["fig04_stepwise_mechanics"]
    build_stepwise_mechanics_artifact(data_root, mechanics_path)
    return mechanics_path, dict(np.load(mechanics_path))


def _build_composite(
    fig: plt.Figure,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    legend_pt: float,
    add_panel_labels: bool,
) -> None:
    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=FIG04_GRID["height_ratios"],
        hspace=FIG04_GRID["hspace"],
        left=FIG04_GRID["left"],
        right=FIG04_GRID["right"],
        bottom=FIG04_GRID["bottom"],
        top=FIG04_GRID["top"],
    )
    _plot_panel_a(
        fig,
        outer[0],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    middle = outer[1].subgridspec(
        1,
        2,
        wspace=FIG04_MIDDLE_ROW["wspace"],
        width_ratios=FIG04_MIDDLE_ROW["width_ratios"],
    )
    _plot_panel_b(
        fig,
        middle[0, 0],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    _plot_panel_c(
        fig,
        middle[0, 1],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    bottom = outer[2].subgridspec(
        1,
        2,
        wspace=FIG04_BOTTOM_ROW["wspace"],
        width_ratios=FIG04_BOTTOM_ROW["width_ratios"],
    )
    ax_d = fig.add_subplot(bottom[0, 0])
    _plot_panel_d(
        ax_d,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=add_panel_labels,
    )
    ax_e = fig.add_subplot(bottom[0, 1])
    _plot_panel_e(
        ax_e,
        mechanics,
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

    all_paths: list[Path] = []
    fig = make_figure(
        width_mm=FIG04_GENERATOR["composite_width_mm"],
        height_mm=FIG04_GENERATOR["composite_height_mm"],
    )
    _build_composite(
        fig,
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_panel_labels=True,
    )
    all_paths.extend(save_outputs(fig, output_dir / "fig04_solver_dynamics", typography=typography))
    plt.close(fig)

    panel_dir = output_dir / "fig04_solver_dynamics_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig_a = make_figure(width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["a"], height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["a"])
    gs_a = gridspec.GridSpec(1, 1, figure=fig_a, **FIG04_STANDALONE["a"])
    _plot_panel_a(
        fig_a,
        gs_a[0],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig04_panel_a_architecture_physics", typography=typography))
    plt.close(fig_a)

    fig_b = make_figure(width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["b"], height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["b"])
    gs_b = gridspec.GridSpec(1, 1, figure=fig_b, **FIG04_STANDALONE["b"])
    _plot_panel_b(
        fig_b,
        gs_b[0],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig04_panel_b_broad_match", typography=typography))
    plt.close(fig_b)

    fig_c = make_figure(width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["c"], height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["c"])
    gs_c = gridspec.GridSpec(1, 1, figure=fig_c, **FIG04_STANDALONE["c"])
    _plot_panel_c(
        fig_c,
        gs_c[0],
        mechanics,
        h_similarity,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig04_panel_c_local_gate", typography=typography))
    plt.close(fig_c)

    fig_d = make_figure(width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["d"], height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["d"])
    ax_d = fig_d.add_subplot(111)
    _plot_panel_d(
        ax_d,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    fig_d.subplots_adjust(**FIG04_STANDALONE["d"])
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig04_panel_d_residual_purification", typography=typography))
    plt.close(fig_d)

    fig_e = make_figure(width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["e"], height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["e"])
    ax_e = fig_e.add_subplot(111)
    _plot_panel_e(
        ax_e,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    fig_e.subplots_adjust(**FIG04_STANDALONE["e"])
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig04_panel_e_ablation", typography=typography))
    plt.close(fig_e)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Neighborhood admissibility strip",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_a_architecture_physics.pdf",
                "provenance_mode": "data_backed",
                "description": "Measured neighborhood in H paired with the broad-match, local-gate, and local-update profiles on the shared angle axis.",
            },
            {
                "panel_id": "b",
                "title": "Broad initial match",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_broad_match.pdf",
                "provenance_mode": "data_backed",
                "description": "Representative speech clip showing broad pre-commitment support across nearby calibrated angles above the measured neighborhood band.",
            },
            {
                "panel_id": "c",
                "title": "Local contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_local_gate.pdf",
                "provenance_mode": "data_backed",
                "description": "Representative local gate and resulting update, both confined to the physically plausible neighborhood around the target angle.",
            },
            {
                "panel_id": "d",
                "title": "Validation-wide neighborhood contraction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_residual_purification.pdf",
                "provenance_mode": "data_backed",
                "description": "Validation-wide cumulative mass within radius before and after the first guided step, showing contraction into the measured local neighborhood.",
            },
            {
                "panel_id": "e",
                "title": "Within-15° mass gain",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_ablation.pdf",
                "provenance_mode": "data_backed",
                "description": "Per-clip distribution of mass inside the measured 15-degree neighborhood before and after the first guided step.",
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

