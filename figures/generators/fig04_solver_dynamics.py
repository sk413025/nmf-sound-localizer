"""Figure 4 — Physics-guided solver mechanism.

Panel (a) is composed at the manuscript layer as a minimal orienting strip.
This generator produces the data-backed mechanism panels:

- (b) one-axis overlay of the broad match, local gate, and local update
- (c) one-axis cleanup overlay with compact summary callouts
- (d) clean decoder-family comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from figures.fig04_stepwise_mechanics import build_stepwise_mechanics_artifact
from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)
from figures.style import (
    SEMANTIC_PALETTE,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


DECODER_VARIANTS = [
    ("No Type Bias", "guided solver", SEMANTIC_PALETTE["learned"]),
    ("No Transformer", "router bypass", SEMANTIC_PALETTE["ablation"]),
    ("Fixed Heuristic", "OMP baseline", SEMANTIC_PALETTE["classical"]),
    ("Dense Routing", "dense routing", "#4A4A4A"),
]
BEFORE_UPDATE_COLOR = "#7A7A7A"
LIGHT_GATE_COLOR = "#8DCDB7"
EPS = 1e-8

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


def _configure_profile_axis(
    ax: plt.Axes,
    *,
    tick_label_pt: float,
    show_xticklabels: bool,
    show_yticks: bool,
) -> None:
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([0, 45, 90, 135, 180])
    if show_xticklabels:
        ax.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    else:
        ax.tick_params(axis="x", labelbottom=False, length=2)
    if show_yticks:
        ax.set_yticks([0.0, 1.0])
        ax.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    else:
        ax.set_yticks([])
    ax.grid(axis="y", linestyle="--", alpha=0.22)


def _draw_profile(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    *,
    color: str,
    target_angle: float,
    linestyle: str = "-",
    linewidth: float = 1.2,
    alpha_fill: float = 0.12,
) -> None:
    ax.fill_between(
        angles_deg,
        np.clip(mean - sem, 0.0, 1.05),
        np.clip(mean + sem, 0.0, 1.05),
        color=color,
        alpha=alpha_fill,
        linewidth=0.0,
    )
    ax.plot(
        angles_deg,
        mean,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    ax.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.9, alpha=0.95)


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / max(float(arr.max()), EPS)


def _normalize_2d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / max(float(arr.max()), EPS)


def _plot_bridge_block(
    fig: plt.Figure,
    slot_spec,
    *,
    matrix_theta_mode: np.ndarray,
    marginal_theta: np.ndarray,
    angles_deg: np.ndarray,
    mode_axis: np.ndarray,
    target_angle: float,
    heatmap_gid: str,
    marginal_gid: str,
    heatmap_label: str,
    marginal_label: str,
    cmap: str,
    line_color: str,
    tick_label_pt: float,
    axis_label_pt: float,
    show_xticklabels: bool,
) -> list[plt.Axes]:
    sub = slot_spec.subgridspec(2, 1, height_ratios=[1.0, 0.52], hspace=0.35)
    axes: list[plt.Axes] = []

    matrix_norm = _normalize_2d(matrix_theta_mode)
    marginal_norm = _normalize_1d(marginal_theta)
    angle_step = float(np.median(np.diff(angles_deg))) if angles_deg.size > 1 else 5.0
    extent = (
        float(angles_deg[0] - angle_step / 2.0),
        float(angles_deg[-1] + angle_step / 2.0),
        float(mode_axis[0] - 0.5),
        float(mode_axis[-1] + 0.5),
    )

    ax_heat = fig.add_subplot(sub[0, 0])
    ax_heat.set_gid(heatmap_gid)
    axes.append(ax_heat)
    ax_heat.imshow(
        matrix_norm.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax_heat.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax_heat.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.8, alpha=0.95)
    ax_heat.set_xlim(0.0, 180.0)
    ax_heat.set_xticks([0, 45, 90, 135, 180])
    ax_heat.tick_params(axis="x", labelbottom=False, length=2)
    if mode_axis.size >= 3:
        mid_idx = int(mode_axis.size // 2)
        ax_heat.set_yticks([float(mode_axis[0]), float(mode_axis[mid_idx]), float(mode_axis[-1])])
    else:
        ax_heat.set_yticks(mode_axis.astype(np.float32))
    ax_heat.tick_params(axis="y", labelsize=tick_label_pt - 0.2, length=2)
    ax_heat.set_ylabel(r"mode $m$", fontsize=axis_label_pt)
    ax_heat.text(
        0.02,
        0.92,
        heatmap_label,
        transform=ax_heat.transAxes,
        fontsize=tick_label_pt,
        color="#1A1A1A",
        va="top",
    )

    ax_curve = fig.add_subplot(sub[1, 0], sharex=ax_heat)
    ax_curve.set_gid(marginal_gid)
    axes.append(ax_curve)
    ax_curve.plot(
        angles_deg,
        marginal_norm,
        color=line_color,
        linewidth=1.15,
    )
    ax_curve.fill_between(
        angles_deg,
        0.0,
        marginal_norm,
        color=line_color,
        alpha=0.10,
        linewidth=0.0,
    )
    ax_curve.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax_curve.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.8, alpha=0.95)
    ax_curve.set_xlim(0.0, 180.0)
    ax_curve.set_ylim(0.0, 1.05)
    ax_curve.set_xticks([0, 45, 90, 135, 180])
    ax_curve.set_yticks([])
    ax_curve.grid(axis="y", linestyle="--", alpha=0.18)
    if show_xticklabels:
        ax_curve.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    else:
        ax_curve.tick_params(axis="x", labelbottom=False, length=2)
    ax_curve.text(
        0.02,
        0.70,
        marginal_label,
        transform=ax_curve.transAxes,
        fontsize=tick_label_pt - 0.2,
        color=line_color,
        va="top",
    )
    return axes


def _plot_panel_b(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    ax.set_gid("fig04.panel_b.main")
    angles_deg = mechanics["angles_deg"]
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    track_specs = [
        (
            "broad match",
            mechanics["stage0_g_norm_mean"],
            mechanics["stage0_g_norm_sem"],
            SEMANTIC_PALETTE["physics"],
            "--",
            1.35,
            0.08,
        ),
        (
            "local gate",
            mechanics["stage0_w_theta_norm_mean"],
            mechanics["stage0_w_theta_norm_sem"],
            LIGHT_GATE_COLOR,
            "-",
            1.15,
            0.10,
        ),
        (
            "local update",
            mechanics["stage0_delta_norm_mean"],
            mechanics["stage0_delta_norm_sem"],
            SEMANTIC_PALETTE["learned"],
            "-",
            1.55,
            0.12,
        ),
    ]

    for track_label, mean_arr, sem_arr, color, linestyle, linewidth, alpha_fill in track_specs:
        _draw_profile(
            ax,
            angles_deg,
            mean_arr[0],
            sem_arr[0],
            color=color,
            target_angle=target_angle,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha_fill=alpha_fill,
        )

    _configure_profile_axis(
        ax,
        tick_label_pt=tick_label_pt,
        show_xticklabels=True,
        show_yticks=True,
    )
    ax.set_ylabel("Relative response", fontsize=axis_label_pt)
    ax.set_xlabel("Angle (\N{DEGREE SIGN})", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.93,
        f"{target_angle:.0f}" + "\N{DEGREE SIGN}" + " exemplar",
        transform=ax.transAxes,
        fontsize=title_pt - 0.2,
        va="top",
        ha="left",
    )
    legend_handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, linewidth=linewidth)
        for label, _mean_arr, _sem_arr, color, linestyle, linewidth, _alpha_fill in track_specs
    ]
    ax.legend(
        legend_handles,
        [label for label, *_rest in track_specs],
        frameon=False,
        fontsize=tick_label_pt - 0.1,
        loc="upper right",
        handlelength=2.0,
        borderpad=0.2,
        labelspacing=0.25,
    )

    if add_label:
        add_panel_label(ax, "b", x=0.0, y=1.02)
    return [ax]


def _plot_panel_c(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    ax.set_gid("fig04.panel_c.main")
    angles_deg = mechanics["angles_deg"].astype(np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    sample_g_steps = mechanics["sample_g_expert_steps"].astype(np.float32)
    radius_deg = mechanics["aligned_radius_deg"].astype(np.float32)
    g_mean = mechanics["aligned_cum_mass_g_mean"].astype(np.float32)
    g_sem = mechanics["aligned_cum_mass_g_sem"].astype(np.float32)
    delta_mean = mechanics["aligned_cum_mass_delta_mean"].astype(np.float32)
    delta_sem = mechanics["aligned_cum_mass_delta_sem"].astype(np.float32)
    clip_count = int(np.asarray(mechanics["aligned_clip_count"]).item())
    sample_initial_res = mechanics["sample_initial_res_norm"].astype(np.float32)
    sample_res_norms = mechanics["sample_res_norms"].astype(np.float32)

    scale = max(float(sample_g_steps[0, 0].max()), EPS)
    g_stage0 = sample_g_steps[0, 0] / scale
    next_step = min(1, sample_g_steps.shape[1] - 1)
    g_after = sample_g_steps[0, next_step] / scale

    ax.plot(
        angles_deg,
        g_stage0,
        linestyle="--",
        linewidth=1.20,
        color=BEFORE_UPDATE_COLOR,
        label="before step",
    )
    ax.plot(
        angles_deg,
        g_after,
        linewidth=1.55,
        color="#2878B5",
        label="after step",
    )
    ax.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.9, alpha=0.95)
    _configure_profile_axis(
        ax,
        tick_label_pt=tick_label_pt,
        show_xticklabels=True,
        show_yticks=True,
    )
    ax.set_ylabel("Relative match", fontsize=axis_label_pt)
    ax.set_xlabel("Angle (\N{DEGREE SIGN})", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.93,
        f"{target_angle:.0f}" + "\N{DEGREE SIGN}",
        transform=ax.transAxes,
        fontsize=tick_label_pt,
        va="top",
        ha="left",
    )
    ax.legend(
        frameon=False,
        fontsize=legend_pt - 0.3,
        loc="upper right",
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )
    residual_ratio = np.concatenate(
        ([1.0], sample_res_norms[0] / max(float(sample_initial_res[0]), EPS))
    )
    local_radius_idx = int(np.argmin(np.abs(radius_deg - 15.0)))
    local_mass_before = float(g_mean[local_radius_idx])
    local_mass_after = float(delta_mean[local_radius_idx])
    residual_after = float(residual_ratio[1]) if residual_ratio.size > 1 else float(residual_ratio[-1])
    summary_text = "\n".join(
        [
            "after one local step",
            f"mass within 15{chr(176)}: {local_mass_before:.2f} -> {local_mass_after:.2f}",
            f"residual: 1.00 -> {residual_after:.2f}",
            f"n = {clip_count}",
        ]
    )
    ax.text(
        0.98,
        0.08,
        summary_text,
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        ha="right",
        va="bottom",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "#F7F7F3",
            "edgecolor": "#D9D9D0",
            "linewidth": 0.8,
            "alpha": 0.96,
        },
    )

    if add_label:
        add_panel_label(ax, "c", x=0.0, y=1.02)
    return [ax]


def _plot_panel_d(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    outer = fig.add_subplot(slot_spec)
    outer.set_gid("fig04.panel_d.block")
    outer.set_axis_off()
    outer.set_xticks([])
    outer.set_yticks([])
    sub = slot_spec.subgridspec(2, 1, hspace=0.42)
    axes: list[plt.Axes] = [outer]

    angles_deg = mechanics["angles_deg"].astype(np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    mode_axis = mechanics["mode_axis"].astype(np.float32)
    sample_w_theta_m_steps = mechanics["sample_w_theta_m_steps"].astype(np.float32)
    sample_delta_theta_m_steps = mechanics["sample_delta_theta_m_steps"].astype(np.float32)

    w_theta_mode = sample_w_theta_m_steps[0, 0]
    w_curve = w_theta_mode.sum(axis=1)
    delta_theta_mode = sample_delta_theta_m_steps[0, 0]
    delta_curve = np.sqrt((delta_theta_mode ** 2).sum(axis=1) + EPS)

    axes.extend(
        _plot_bridge_block(
            fig,
            sub[0, 0],
            matrix_theta_mode=w_theta_mode,
            marginal_theta=w_curve,
            angles_deg=angles_deg,
            mode_axis=mode_axis,
            target_angle=target_angle,
            heatmap_gid="fig04.panel_d.w_heatmap",
            marginal_gid="fig04.panel_d.w_curve",
            heatmap_label=r"$w_{t,\theta,m}$",
            marginal_label=r"$w_t(\theta)=\sum_m w_{t,\theta,m}$",
            cmap=BRIDGE_GATE_CMAP,
            line_color=SEMANTIC_PALETTE["learned"],
            tick_label_pt=tick_label_pt,
            axis_label_pt=axis_label_pt,
            show_xticklabels=False,
        )
    )
    axes.extend(
        _plot_bridge_block(
            fig,
            sub[1, 0],
            matrix_theta_mode=delta_theta_mode,
            marginal_theta=delta_curve,
            angles_deg=angles_deg,
            mode_axis=mode_axis,
            target_angle=target_angle,
            heatmap_gid="fig04.panel_d.delta_heatmap",
            marginal_gid="fig04.panel_d.delta_curve",
            heatmap_label=r"$|\eta\hat{\Delta x}_{t,\theta,m}|$",
            marginal_label=r"$\Delta x_t(\theta)=\|\eta\hat{\Delta x}_{t,\theta,\cdot}\|_2$",
            cmap=BRIDGE_DELTA_CMAP,
            line_color="#2878B5",
            tick_label_pt=tick_label_pt,
            axis_label_pt=axis_label_pt,
            show_xticklabels=True,
        )
    )

    outer.text(
        0.10,
        1.005,
        f"{target_angle:.0f}" + "\N{DEGREE SIGN}" + " exemplar",
        transform=outer.transAxes,
        fontsize=title_pt - 0.4,
        va="bottom",
        ha="left",
    )

    if add_label:
        add_panel_label(outer, "d", x=0.0, y=1.02)
    return axes


def _plot_panel_d_ablation(
    ax: plt.Axes,
    ablation_data: dict[str, list[float]],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    add_label: bool,
) -> None:
    ax.set_gid("fig04.panel_d.main")
    row_positions = np.arange(len(DECODER_VARIANTS) - 1, -1, -1, dtype=np.float32)
    chance_level = min(
        float(np.asarray(ablation_data.get("Dense Routing", [0.0]), dtype=np.float32).mean()),
        0.08,
    )

    ax.axvspan(0.0, chance_level, color="#F5F5F5", zorder=0)
    ax.axvline(chance_level, color="#CFCFCF", linestyle="--", linewidth=0.85, zorder=1)
    ax.hlines(row_positions, 0.0, 1.0, color="#ECECEC", linewidth=0.75, zorder=1)

    for (variant_key, display_label, color), y_pos in zip(DECODER_VARIANTS, row_positions, strict=False):
        seeds = ablation_data.get(variant_key, [])
        if not seeds:
            continue
        seed_arr = np.asarray(seeds, dtype=np.float32)
        mean_val = float(seed_arr.mean())
        sem_val = 0.0 if seed_arr.size <= 1 else float(seed_arr.std(ddof=1) / np.sqrt(seed_arr.size))
        offsets = (
            np.linspace(-0.11, 0.11, num=seed_arr.size, dtype=np.float32)
            if seed_arr.size > 1
            else np.zeros(1, dtype=np.float32)
        )
        ax.scatter(
            seed_arr,
            np.full(seed_arr.size, y_pos, dtype=np.float32) + offsets,
            color=color,
            s=18,
            alpha=0.58,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.errorbar(
            mean_val,
            y_pos,
            xerr=sem_val,
            fmt="o",
            color=color,
            markersize=5.8,
            capsize=2.2,
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            0.99,
            y_pos,
            f"{mean_val:.2f}",
            fontsize=tick_label_pt - 0.1,
            va="center",
            ha="right",
            color="#4A4A4A",
        )
        ax.text(
            min(chance_level + 0.03, 0.10),
            y_pos,
            display_label,
            fontsize=axis_label_pt - 0.3,
            va="center",
            ha="left",
            color="#404040",
            zorder=2,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, len(DECODER_VARIANTS) - 0.45)
    ax.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=tick_label_pt + 0.2, length=2)
    ax.set_yticks([])
    ax.set_xlabel("Clean Top-1 Accuracy (restricted)", fontsize=axis_label_pt, labelpad=1.2)
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.text(
        0.02,
        0.98,
        "chance floor",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        va="top",
        ha="left",
        color="#6B6B6B",
    )
    ax.text(
        0.98,
        0.98,
        "secondary check, 5 seeds",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        va="top",
        ha="right",
        color="#6B6B6B",
    )

    if add_label:
        add_panel_label(ax, "d", x=0.0, y=1.02)


def _load_ablation_data(data_root: Path) -> dict[str, list[float]]:
    paths_cfg = load_paths()
    agg_path = data_root / paths_cfg["ablation_sweep"]["aggregated_json"]
    if not agg_path.exists():
        raise FileNotFoundError(f"Fig. 4 ablation sweep not found at {agg_path}")
    with agg_path.open(encoding="utf-8") as handle:
        return json.load(handle).get("ablation", {})


def _load_mechanics(data_root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    paths_cfg = load_paths()
    mechanics_path = data_root / paths_cfg["fig04_stepwise_mechanics"]
    build_stepwise_mechanics_artifact(data_root, mechanics_path)
    return mechanics_path, dict(np.load(mechanics_path))


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 4 data-backed panels b-d."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    title_pt = typography["title"]

    mechanics_path, mechanics = _load_mechanics(data_root)
    ablation_data = _load_ablation_data(data_root)

    fig = make_figure(
        width_mm=FIG04_GENERATOR["composite_width_mm"],
        height_mm=FIG04_GENERATOR["composite_height_mm"],
    )
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=FIG04_GRID["width_ratios"],
        height_ratios=FIG04_GRID["height_ratios"],
        hspace=FIG04_GRID["hspace"],
        wspace=FIG04_GRID["wspace"],
        left=FIG04_GRID["left"],
        right=FIG04_GRID["right"],
        bottom=FIG04_GRID["bottom"],
        top=FIG04_GRID["top"],
    )

    _plot_panel_b(
        fig,
        gs[0, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    _plot_panel_c(
        fig,
        gs[1, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=False,
    )
    ax_d = fig.add_subplot(gs[:, 1])
    _plot_panel_d_ablation(
        ax_d,
        ablation_data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        add_label=False,
    )

    all_paths = save_outputs(
        fig,
        output_dir / "fig04_solver_dynamics",
        typography=typography,
    )
    plt.close(fig)

    panel_dir = output_dir / "fig04_solver_dynamics_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig_b = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["b"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["b"],
    )
    gs_b = fig_b.add_gridspec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    _plot_panel_b(
        fig_b,
        gs_b[0, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    fig_b.subplots_adjust(**FIG04_STANDALONE["b"])
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig04_panel_b_gate_qk",
            typography=typography,
        )
    )
    plt.close(fig_b)

    fig_c = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["c"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["c"],
    )
    gs_c = fig_c.add_gridspec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    _plot_panel_c(
        fig_c,
        gs_c[0, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=False,
    )
    fig_c.subplots_adjust(**FIG04_STANDALONE["c"])
    all_paths.extend(
        save_outputs(
            fig_c,
            panel_dir / "fig04_panel_c_update_residual",
            typography=typography,
        )
    )
    plt.close(fig_c)

    fig_d = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["d"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["d"],
    )
    gs_d = fig_d.add_gridspec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax_d_single = fig_d.add_subplot(gs_d[0, 0])
    _plot_panel_d_ablation(
        ax_d_single,
        ablation_data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        add_label=False,
    )
    fig_d.subplots_adjust(**FIG04_STANDALONE["d"])
    all_paths.extend(
        save_outputs(
            fig_d,
            panel_dir / "fig04_panel_d_ablation",
            typography=typography,
        )
    )
    plt.close(fig_d)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "b",
                "title": "Broad match, local gate, and local update",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_gate_qk.pdf",
                "provenance_mode": "data_backed",
                "description": f"One-axis overlay for the shared exemplar showing the broad match, the local gate, and the local update on the active {mechanics_path.name} manifold.",
            },
            {
                "panel_id": "c",
                "title": "Residual after one local step",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_update_residual.pdf",
                "provenance_mode": "data_backed",
                "description": "One-axis cleanup overlay for the shared 70-degree exemplar, with compact callouts summarizing the validation-wide mass moved within 15 degrees and the first residual-norm drop.",
            },
            {
                "panel_id": "d",
                "title": "Secondary clean-condition check",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_ablation.pdf",
                "provenance_mode": "data_backed",
                "description": "Restricted clean-condition check showing the guided solver, router-bypass, OMP baseline, and dense routing families across the shared five-seed sweep.",
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
