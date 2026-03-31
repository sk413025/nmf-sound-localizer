"""Figure 4 — Physics-guided solver mechanism.

Panel (a) remains the committed architecture artwork in the manuscript layer.
This generator produces the data-backed mechanism panels:

- (b) routing formation: ``g_t -> q_t K_t^T -> w_t``
- (c) gated update + residual correction
- (d) aggregation bridge from ``w_{t,θ,m}`` / ``ηΔx̂_{t,θ,m}`` to
  ``w_t(θ)`` / ``Δx_t(θ)``
- (e) clean routing-mechanism ablation
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
    SEMANTIC_PALETTE,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


DECODER_VARIANTS = [
    ("No Type Bias", SEMANTIC_PALETTE["learned"]),
    ("No Transformer", SEMANTIC_PALETTE["ablation"]),
    ("Fixed Heuristic", SEMANTIC_PALETTE["classical"]),
    ("Dense Routing", "#4A4A4A"),
]
QK_COLOR = "#5E8B7E"
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
ABLATION_XLABELS = ["guided", "bypass", "OMP", "dense"]
ABLATION_YLABELS = ["guided", "bypass", "OMP", "dense"]
BRIDGE_GATE_CMAP = "Greens"
BRIDGE_DELTA_CMAP = "Blues"
FIG04_STACK_HEIGHT_RATIOS = [
    float(value) for value in FIG04_GRID.get("stack_height_ratios", [43.0, 22.0])
]
FIG04_STACK_HSPACE = float(FIG04_GRID.get("stack_hspace", 0.32))


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
    outer = fig.add_subplot(slot_spec)
    outer.set_gid("fig04.panel_b.block")
    outer.set_axis_off()
    outer.set_xticks([])
    outer.set_yticks([])
    sub = slot_spec.subgridspec(3, 1, hspace=0.15)

    angles_deg = mechanics["angles_deg"]
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    track_specs = [
        (r"$g_t(\theta)$", mechanics["stage0_g_norm_mean"], mechanics["stage0_g_norm_sem"], SEMANTIC_PALETTE["physics"]),
        (r"$(q_tK_t^\top)(\theta)$", mechanics["stage0_qk_norm_mean"], mechanics["stage0_qk_norm_sem"], QK_COLOR),
        (r"$w_t(\theta)$", mechanics["stage0_w_theta_norm_mean"], mechanics["stage0_w_theta_norm_sem"], SEMANTIC_PALETTE["learned"]),
    ]

    axes: list[plt.Axes] = [outer]
    shared: plt.Axes | None = None
    for row_idx, (track_label, mean_arr, sem_arr, color) in enumerate(track_specs):
        if shared is None:
            ax = fig.add_subplot(sub[row_idx, 0])
            shared = ax
        else:
            ax = fig.add_subplot(sub[row_idx, 0], sharex=shared, sharey=shared)
        axes.append(ax)
        ax.set_gid(("fig04.panel_b.g", "fig04.panel_b.qk", "fig04.panel_b.w")[row_idx])

        _draw_profile(
            ax,
            angles_deg,
            mean_arr[0],
            sem_arr[0],
            color=color,
            target_angle=target_angle,
        )
        _configure_profile_axis(
            ax,
            tick_label_pt=tick_label_pt,
            show_xticklabels=row_idx == len(track_specs) - 1,
            show_yticks=row_idx == 0,
        )
        if row_idx == 0:
            ax.text(
                0.08,
                0.90,
                f"{target_angle:.0f}" + "\N{DEGREE SIGN}",
                transform=ax.transAxes,
                fontsize=title_pt - 0.2,
                va="top",
            )
        ax.text(
            0.02,
            0.12,
            track_label,
            transform=ax.transAxes,
            fontsize=tick_label_pt,
            color=color,
            ha="left",
            va="bottom",
        )

    if add_label:
        add_panel_label(outer, "b", x=0.0, y=1.02)
    return axes


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
    outer = fig.add_subplot(slot_spec)
    outer.set_gid("fig04.panel_c.block")
    outer.set_axis_off()
    outer.set_xticks([])
    outer.set_yticks([])
    sub = slot_spec.subgridspec(4, 1, height_ratios=[1.0, 1.0, 0.95, 0.72], hspace=0.44)
    axes: list[plt.Axes] = [outer]

    angles_deg = mechanics["angles_deg"].astype(np.float32)
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    sample_g_steps = mechanics["sample_g_expert_steps"].astype(np.float32)
    sample_delta_steps = mechanics["sample_delta_expert_steps"].astype(np.float32)
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
    delta_stage0 = sample_delta_steps[0, 0] / scale
    next_step = min(1, sample_g_steps.shape[1] - 1)
    g_after = sample_g_steps[0, next_step] / scale

    ax_update = fig.add_subplot(sub[0, 0])
    ax_update.set_gid("fig04.panel_c.update")
    axes.append(ax_update)
    ax_update.plot(
        angles_deg,
        g_stage0,
        linestyle="--",
        linewidth=1.05,
        color=SEMANTIC_PALETTE["physics"],
        label=r"$g_t(\theta)$",
    )
    ax_update.plot(
        angles_deg,
        delta_stage0,
        linewidth=1.25,
        color=SEMANTIC_PALETTE["learned"],
        label=r"$\Delta x_t(\theta)$",
    )
    ax_update.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax_update.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.9, alpha=0.95)
    _configure_profile_axis(
        ax_update,
        tick_label_pt=tick_label_pt,
        show_xticklabels=False,
        show_yticks=True,
    )
    ax_update.text(
        0.07,
        0.90,
        f"{target_angle:.0f}" + "\N{DEGREE SIGN}",
        transform=ax_update.transAxes,
        fontsize=tick_label_pt,
        va="top",
    )
    ax_update.set_ylabel(r"$g_t, \Delta x_t$", fontsize=axis_label_pt)
    ax_update.legend(
        frameon=False,
        fontsize=tick_label_pt - 0.2,
        loc="lower right",
        handlelength=1.6,
        borderpad=0.2,
        labelspacing=0.25,
    )

    ax_residual = fig.add_subplot(sub[1, 0], sharex=ax_update, sharey=ax_update)
    ax_residual.set_gid("fig04.panel_c.residual")
    axes.append(ax_residual)
    ax_residual.plot(
        angles_deg,
        g_stage0,
        linestyle="--",
        linewidth=1.0,
        color=BEFORE_UPDATE_COLOR,
        label=r"$g_t(\theta)$",
    )
    ax_residual.plot(
        angles_deg,
        g_after,
        linewidth=1.2,
        color="#2878B5",
        label=r"$g_{t+1}(\theta)$",
    )
    ax_residual.axvspan(target_angle - 5.0, target_angle + 5.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax_residual.axvline(target_angle, color=SEMANTIC_PALETTE["highlight"], linewidth=0.9, alpha=0.95)
    _configure_profile_axis(
        ax_residual,
        tick_label_pt=tick_label_pt,
        show_xticklabels=False,
        show_yticks=False,
    )
    ax_residual.set_ylabel(r"$g_t, g_{t+1}$", fontsize=axis_label_pt)
    ax_residual.legend(
        frameon=False,
        fontsize=tick_label_pt - 0.3,
        loc="upper right",
        handlelength=1.4,
        borderpad=0.2,
        labelspacing=0.22,
    )

    ax_mass = fig.add_subplot(sub[2, 0])
    ax_mass.set_gid("fig04.panel_c.mass")
    axes.append(ax_mass)
    ax_mass.fill_between(
        radius_deg,
        np.clip(g_mean - g_sem, 0.0, 1.0),
        np.clip(g_mean + g_sem, 0.0, 1.0),
        color=SEMANTIC_PALETTE["physics"],
        alpha=0.12,
        linewidth=0.0,
    )
    ax_mass.plot(
        radius_deg,
        g_mean,
        color=SEMANTIC_PALETTE["physics"],
        linestyle="--",
        linewidth=1.15,
        label=r"$g_t$",
    )
    ax_mass.fill_between(
        radius_deg,
        np.clip(delta_mean - delta_sem, 0.0, 1.0),
        np.clip(delta_mean + delta_sem, 0.0, 1.0),
        color=SEMANTIC_PALETTE["learned"],
        alpha=0.12,
        linewidth=0.0,
    )
    ax_mass.plot(
        radius_deg,
        delta_mean,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=1.35,
        label=r"$\Delta x_t$",
    )
    ax_mass.axvspan(0.0, 15.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.08)
    ax_mass.axvline(15.0, color=SEMANTIC_PALETTE["highlight"], linewidth=0.9, alpha=0.95)
    ax_mass.set_xlim(float(radius_deg[0]), float(radius_deg[-1]))
    ax_mass.set_ylim(0.0, 1.02)
    ax_mass.set_xticks(radius_deg[::2])
    ax_mass.set_yticks([0.0, 0.5, 1.0])
    ax_mass.tick_params(axis="both", labelsize=tick_label_pt, length=2)
    ax_mass.grid(axis="both", linestyle="--", alpha=0.22)
    ax_mass.set_ylabel("Local mass", fontsize=axis_label_pt)
    ax_mass.text(
        0.03,
        0.94,
        r"$P(|\Delta \theta| \leq r)$" + f"\n n = {clip_count}",
        transform=ax_mass.transAxes,
        fontsize=tick_label_pt,
        va="top",
        ha="left",
    )
    ax_mass.text(
        0.58,
        0.10,
        "0-15°",
        transform=ax_mass.transAxes,
        fontsize=tick_label_pt,
        color="#5A5A5A",
        ha="center",
    )
    ax_mass.legend(
        frameon=False,
        fontsize=legend_pt - 0.5,
        loc="lower right",
        handlelength=1.4,
        borderpad=0.2,
        labelspacing=0.25,
    )

    ax_bar = fig.add_subplot(sub[3, 0])
    ax_bar.set_gid("fig04.panel_c.bar")
    axes.append(ax_bar)
    residual_ratio = np.concatenate(
        ([1.0], sample_res_norms[0] / max(float(sample_initial_res[0]), EPS))
    )
    bar_colors = ["#D4D4D4", LIGHT_GATE_COLOR, SEMANTIC_PALETTE["learned"]][: residual_ratio.size]
    ax_bar.bar(
        np.arange(residual_ratio.size),
        residual_ratio,
        color=bar_colors,
        width=0.66,
        edgecolor="none",
    )
    ax_bar.set_ylim(0.0, 1.08)
    ax_bar.set_xticks(np.arange(residual_ratio.size))
    ax_bar.set_xticklabels([])
    ax_bar.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.25)
    ax_bar.text(
        0.03,
        0.93,
        r"$\|r_{t+k}\|_2 / \|r_t\|_2$",
        transform=ax_bar.transAxes,
        fontsize=tick_label_pt,
        va="top",
        ha="left",
    )
    for idx, stage_label in enumerate((r"$t$", r"$t+1$", r"$t+2$")[: residual_ratio.size]):
        ax_bar.text(
            idx,
            0.03,
            stage_label,
            transform=ax_bar.get_xaxis_transform(),
            fontsize=tick_label_pt,
            ha="center",
            va="bottom",
        )

    if add_label:
        add_panel_label(outer, "c", x=0.0, y=1.02)
    return axes


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


def _plot_panel_e(
    ax: plt.Axes,
    ablation_data: dict[str, list[float]],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    add_label: bool,
) -> None:
    ax.set_gid("fig04.panel_e.main")
    row_positions = np.arange(len(DECODER_VARIANTS) - 1, -1, -1, dtype=np.float32)
    for (variant_key, color), y_pos in zip(DECODER_VARIANTS, row_positions, strict=False):
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
            alpha=0.80,
            edgecolors="none",
            zorder=3,
        )
        ax.errorbar(
            mean_val,
            y_pos,
            xerr=sem_val,
            fmt="o",
            color=color,
            markersize=4.6,
            capsize=1.8,
            linewidth=1.0,
            zorder=4,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.55, len(DECODER_VARIANTS) - 0.45)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.text(
        0.98,
        0.96,
        r"$P(\mathrm{correct})$",
        transform=ax.transAxes,
        fontsize=tick_label_pt,
        va="top",
        ha="right",
        color="#4A4A4A",
    )
    for label, y_pos in zip(ABLATION_YLABELS, row_positions, strict=False):
        ax.text(
            0.10,
            y_pos,
            label,
            fontsize=tick_label_pt - 0.1,
            va="center",
            ha="left",
            color="#3A3A3A",
        )

    if add_label:
        add_panel_label(ax, "e", x=0.0, y=1.02)


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
    """Generate Figure 4 data-backed panels b-e."""
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
        1,
        3,
        figure=fig,
        width_ratios=FIG04_GRID["width_ratios"],
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
        gs[0, 1],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=False,
    )
    gs_right = gs[0, 2].subgridspec(
        2,
        1,
        height_ratios=FIG04_STACK_HEIGHT_RATIOS,
        hspace=FIG04_STACK_HSPACE,
    )
    _plot_panel_d(
        fig,
        gs_right[0, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    ax_e = fig.add_subplot(gs_right[1, 0])
    _plot_panel_e(
        ax_e,
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
    _plot_panel_d(
        fig_d,
        gs_d[0, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    fig_d.subplots_adjust(**FIG04_STANDALONE["d"])
    all_paths.extend(
        save_outputs(
            fig_d,
            panel_dir / "fig04_panel_d_aggregation_bridge",
            typography=typography,
        )
    )
    plt.close(fig_d)

    fig_e = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["e"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["e"],
    )
    gs_e = fig_e.add_gridspec(1, 1, left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax_e_single = fig_e.add_subplot(gs_e[0, 0])
    _plot_panel_e(
        ax_e_single,
        ablation_data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        add_label=False,
    )
    fig_e.subplots_adjust(**FIG04_STANDALONE["e"])
    all_paths.extend(
        save_outputs(
            fig_e,
            panel_dir / "fig04_panel_e_ablation",
            typography=typography,
        )
    )
    plt.close(fig_e)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "b",
                "title": "Routing formation",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_gate_qk.pdf",
                "provenance_mode": "data_backed",
                "description": f"Angle-conditioned stage-0 validation summaries showing how the solver transforms the physical correlation g_t into angle-local routing weights w_t through the learned QK score q_tK_t^T on the active {mechanics_path.name} manifold.",
            },
            {
                "panel_id": "c",
                "title": "Gated update and residual correction",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_update_residual.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared 70-degree portrait mechanism panel showing the gated update Δx_t, the first residual-consistent change from g_t to g_{t+1}, the validation-wide localization summary, and residual-norm descent across unrolled stages.",
            },
            {
                "panel_id": "d",
                "title": "Aggregation bridge",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_aggregation_bridge.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared 70-degree exemplar bridge panel showing the mode-resolved routing tensor w_{t,θ,m} and gated-update magnitude |ηΔx̂_{t,θ,m}| together with their angle-level reductions w_t(θ) and Δx_t(θ), making panel a's internal variables explicit on the angle axis used by panels b-c.",
            },
            {
                "panel_id": "e",
                "title": "Routing-mechanism ablation",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_e_ablation.pdf",
                "provenance_mode": "data_backed",
                "description": "Compact clean-condition ablation plot comparing the guided solver, router-bypass, OMP baseline, and dense routing families across the shared five-seed sweep.",
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
