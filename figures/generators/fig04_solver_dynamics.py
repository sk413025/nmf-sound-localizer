"""Figure 4 — Physics-guided solver mechanism.

This generator produces the four manuscript-facing data-backed panels:

- (a) measured local band in H
- (b) representative exemplar
- (c) validation-wide local concentration
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


DECODER_VARIANTS = [
    ("No Type Bias", "guided solver", SEMANTIC_PALETTE["learned"]),
    ("No Transformer", "router bypass", SEMANTIC_PALETTE["ablation"]),
    ("Fixed Heuristic", "OMP baseline", SEMANTIC_PALETTE["classical"]),
    ("Dense Routing", "dense routing", STYLE_COLORS["dense_routing"]),
]
BEFORE_UPDATE_COLOR = SEMANTIC_PALETTE["physics"]
LIGHT_GATE_COLOR = SEMANTIC_PALETTE["learned"]
EPS = 1e-8
FIG04_HEAVY_LINEWIDTH = STROKE_TOKENS["emphasis"]
FIG04_BASE_LINEWIDTH = STROKE_TOKENS["data"]
FIG04_LIGHT_LINEWIDTH = STROKE_TOKENS["base"]
FIG04_RULE_LINEWIDTH = STROKE_TOKENS["base"]
FIG04_MARKER_EDGEWIDTH = STROKE_TOKENS["grid"]
FIG04_POINT_MARKER_SIZE = FAMILY_STYLE["compact_marker_pt"] ** 2
FIG04_MEAN_MARKER_SIZE = FAMILY_STYLE["standard_marker_pt"]
FIG04_ERRORBAR_CAPSIZE = FAMILY_STYLE["compact_capsize_pt"]
FIG04_GUIDE_LINE_COLOR = STYLE_COLORS["guide_line"]
FIG04_GUIDE_FILL_COLOR = STYLE_COLORS["guide_fill"]

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
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0, 45, 90, 135, 180])
    if show_xticklabels:
        ax.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    else:
        ax.tick_params(axis="x", labelbottom=False, length=2)
    if show_yticks:
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    else:
        ax.set_yticks([])
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])


def _draw_profile(
    ax: plt.Axes,
    angles_deg: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    *,
    color: str,
    target_angle: float,
    linestyle: str = "-",
    linewidth: float = FIG04_BASE_LINEWIDTH,
    alpha_fill: float = FAMILY_STYLE["fill_alpha_primary"],
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
    ax.axvspan(
        target_angle - 5.0,
        target_angle + 5.0,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_tertiary"],
    )
    ax.axvline(
        target_angle,
        color=SEMANTIC_PALETTE["highlight"],
        linewidth=FIG04_RULE_LINEWIDTH,
        alpha=0.95,
    )


def _normalize_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return arr / max(float(arr.max()), EPS)


def _load_h_similarity(data_root: Path) -> np.ndarray:
    """Load H matrix and compute cosine similarity for the bridge inset."""
    paths_cfg = load_paths()
    dict_path = data_root / paths_cfg["primary_run"] / "dictionary.npz"
    dict_data = np.load(dict_path, allow_pickle=True)
    H = np.asarray(dict_data["H"], dtype=np.float64)
    norms = np.linalg.norm(H, axis=0, keepdims=True)
    H_normed = H / np.maximum(norms, 1e-8)
    return (H_normed.T @ H_normed).astype(np.float32)


def _plot_panel_a_local_band(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    h_similarity: np.ndarray,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    ax.set_gid("fig04.panel_a.main")
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    img = ax.imshow(
        h_similarity,
        extent=(0.0, 180.0, 0.0, 180.0),
        origin="lower",
        aspect="auto",
        cmap="magma",
        interpolation="nearest",
    )
    img.set_clim(float(np.percentile(h_similarity, 5.0)), float(np.percentile(h_similarity, 99.0)))
    ax.axvspan(
        target_angle - 5.0,
        target_angle + 5.0,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_tertiary"],
        linewidth=0.0,
    )
    ax.axvline(
        target_angle,
        color=SEMANTIC_PALETTE["highlight"],
        linewidth=FIG04_RULE_LINEWIDTH,
        alpha=0.95,
    )
    ax.axhline(
        target_angle,
        color=SEMANTIC_PALETTE["highlight"],
        linewidth=FIG04_RULE_LINEWIDTH,
        alpha=0.95,
    )
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(0.0, 180.0)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.set_title("Measured local structure", fontsize=title_pt - 0.1, pad=2.0)
    ax.set_xlabel("Angle (\N{DEGREE SIGN})", fontsize=axis_label_pt)
    ax.set_ylabel("Angle (\N{DEGREE SIGN})", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.92,
        f"target = {target_angle:.0f}" + "\N{DEGREE SIGN}",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
    )
    if add_label:
        add_panel_label(ax, "a", x=0.0, y=1.02)
    return [ax]


def _plot_panel_b(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
    h_similarity: np.ndarray | None = None,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    ax.set_gid("fig04.panel_a.main")
    angles_deg = mechanics["angles_deg"]
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    track_specs = [
        (
            "broad match",
            mechanics["stage0_g_norm_mean"],
            mechanics["stage0_g_norm_sem"],
            SEMANTIC_PALETTE["physics"],
            "--",
            FIG04_BASE_LINEWIDTH,
            FAMILY_STYLE["fill_alpha_tertiary"],
        ),
        (
            "local gate",
            mechanics["stage0_w_theta_norm_mean"],
            mechanics["stage0_w_theta_norm_sem"],
            LIGHT_GATE_COLOR,
            ":",
            FIG04_LIGHT_LINEWIDTH,
            FAMILY_STYLE["fill_alpha_secondary"],
        ),
        (
            "local update",
            mechanics["stage0_delta_norm_mean"],
            mechanics["stage0_delta_norm_sem"],
            SEMANTIC_PALETTE["learned"],
            "-",
            FIG04_HEAVY_LINEWIDTH,
            FAMILY_STYLE["fill_alpha_primary"],
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
    local_window_min = max(0.0, target_angle - 25.0)
    local_window_max = min(180.0, target_angle + 25.0)
    ax.set_xlim(local_window_min, local_window_max)
    ax.set_xticks([45, 60, 75, 90] if 45 <= local_window_min and local_window_max <= 95 else [local_window_min, target_angle, local_window_max])
    ax.set_title("Representative exemplar", fontsize=title_pt - 0.1, pad=2.0)
    ax.set_ylabel("Normalized support", fontsize=axis_label_pt)
    ax.set_xlabel("Angle (\N{DEGREE SIGN})", fontsize=axis_label_pt)
    ax.text(
        0.03,
        0.84,
        f"target = {target_angle:.0f}" + "\N{DEGREE SIGN}",
        transform=ax.transAxes,
        fontsize=title_pt - 0.2,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
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
        add_panel_label(ax, "c", x=0.0, y=1.02)
    return [ax]


def _plot_panel_a_stage_profiles(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    stage_titles = (
        ("broad match", mechanics["stage0_g_norm_mean"], mechanics["stage0_g_norm_sem"], SEMANTIC_PALETTE["physics"], "--", FIG04_BASE_LINEWIDTH),
        ("local gate", mechanics["stage0_w_theta_norm_mean"], mechanics["stage0_w_theta_norm_sem"], LIGHT_GATE_COLOR, ":", FIG04_LIGHT_LINEWIDTH),
        ("local update", mechanics["stage0_delta_norm_mean"], mechanics["stage0_delta_norm_sem"], SEMANTIC_PALETTE["learned"], "-", FIG04_HEAVY_LINEWIDTH),
    )
    angles_deg = mechanics["angles_deg"]
    target_angle = float(np.asarray(mechanics["representative_angles_deg"]).item())
    stage_spec = slot_spec.subgridspec(1, 3, wspace=0.16)
    axes: list[plt.Axes] = []

    for idx, (stage_title, mean_arr, sem_arr, color, linestyle, linewidth) in enumerate(stage_titles):
        ax = fig.add_subplot(stage_spec[0, idx])
        ax.set_gid(f"fig04.panel_a.stage_{idx}")
        _draw_profile(
            ax,
            angles_deg,
            mean_arr[0],
            sem_arr[0],
            color=color,
            target_angle=target_angle,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha_fill=FAMILY_STYLE["fill_alpha_secondary"],
        )
        ax.set_xlim(0.0, 180.0)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([0, 90, 180])
        ax.set_yticks([0.0, 1.0] if idx == 0 else [])
        ax.tick_params(axis="x", labelsize=tick_label_pt - 0.4, length=2)
        ax.tick_params(axis="y", labelsize=tick_label_pt - 0.4, length=2)
        ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
        ax.set_title(stage_title, fontsize=title_pt - 0.8, pad=1.0, color=STYLE_COLORS["muted_text"])
        if idx == 0:
            ax.set_ylabel("Support", fontsize=axis_label_pt - 0.5, labelpad=0.8)
        else:
            ax.spines["left"].set_visible(False)
        ax.set_xlabel("Angle (°)", fontsize=axis_label_pt - 0.6, labelpad=0.6)
        axes.append(ax)

    if add_label and axes:
        add_panel_label(axes[0], "a", x=0.0, y=1.02)
    return axes


def _plot_panel_c(
    fig: plt.Figure,
    slot_spec,
    mechanics: dict[str, np.ndarray],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ax = fig.add_subplot(slot_spec)
    ax.set_gid("fig04.panel_b.main")
    radius_deg = mechanics["aligned_radius_deg"].astype(np.float32)
    g_mean = mechanics["aligned_cum_mass_g_mean"].astype(np.float32)
    g_sem = mechanics["aligned_cum_mass_g_sem"].astype(np.float32)
    delta_mean = mechanics["aligned_cum_mass_delta_mean"].astype(np.float32)
    delta_sem = mechanics["aligned_cum_mass_delta_sem"].astype(np.float32)
    clip_count = int(np.asarray(mechanics["aligned_clip_count"]).item())

    ax.fill_between(
        radius_deg,
        np.clip(delta_mean - delta_sem, 0, 1),
        np.clip(delta_mean + delta_sem, 0, 1),
        color=SEMANTIC_PALETTE["learned"],
        alpha=FAMILY_STYLE["fill_alpha_primary"],
        linewidth=0,
    )
    ax.plot(
        radius_deg,
        delta_mean,
        color=SEMANTIC_PALETTE["learned"],
        linewidth=FIG04_HEAVY_LINEWIDTH,
        label="after",
    )
    ax.fill_between(
        radius_deg,
        np.clip(g_mean - g_sem, 0, 1),
        np.clip(g_mean + g_sem, 0, 1),
        color=BEFORE_UPDATE_COLOR,
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0,
    )
    ax.plot(
        radius_deg,
        g_mean,
        color=BEFORE_UPDATE_COLOR,
        linewidth=FIG04_BASE_LINEWIDTH,
        linestyle="--",
        label="before",
    )
    ax.axvspan(
        0.0,
        15.0,
        color=SEMANTIC_PALETTE["highlight"],
        alpha=FAMILY_STYLE["fill_alpha_secondary"],
        linewidth=0.0,
    )
    local_radius_idx = int(np.argmin(np.abs(radius_deg - 15.0)))
    ax.axvline(
        15.0,
        color=STYLE_COLORS["chance_line"],
        linewidth=FIG04_RULE_LINEWIDTH,
        linestyle=":",
    )
    ax.text(
        0.17,
        0.06,
        "within 15°",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.3,
        color=STYLE_COLORS["muted_text"],
        ha="center",
        va="bottom",
    )
    ax.annotate(
        f"{float(delta_mean[local_radius_idx]):.2f}",
        xy=(15.0, float(delta_mean[local_radius_idx])),
        xytext=(20, float(delta_mean[local_radius_idx]) - 0.06),
        fontsize=tick_label_pt - 0.3,
        color=SEMANTIC_PALETTE["learned"],
        arrowprops={"arrowstyle": "-", "color": STYLE_COLORS["chance_line"], "linewidth": STROKE_TOKENS["annotation"]},
    )
    ax.annotate(
        f"{float(g_mean[local_radius_idx]):.2f}",
        xy=(15.0, float(g_mean[local_radius_idx])),
        xytext=(20, float(g_mean[local_radius_idx]) + 0.06),
        fontsize=tick_label_pt - 0.3,
        color=BEFORE_UPDATE_COLOR,
        arrowprops={"arrowstyle": "-", "color": STYLE_COLORS["chance_line"], "linewidth": STROKE_TOKENS["annotation"]},
    )
    ax.set_xlim(0, 45)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 15, 30, 45])
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.set_title("Validation set", fontsize=axis_label_pt + 0.1, pad=2.0)
    ax.set_xlabel(f"Neighborhood radius ({chr(176)})", fontsize=axis_label_pt, labelpad=1.0)
    ax.set_ylabel("Mass within radius", fontsize=axis_label_pt, labelpad=1.0)
    ax.text(
        0.03,
        0.91,
        f"validation-wide, n = {clip_count:,}",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        ha="left",
        va="top",
        color=STYLE_COLORS["muted_text"],
    )
    ax.legend(
        frameon=False,
        fontsize=legend_pt - 0.3,
        loc="lower right",
        handlelength=1.5,
        labelspacing=0.2,
    )
    if add_label:
        add_panel_label(ax, "b", x=0.0, y=1.02)
    return [ax]


def _plot_panel_d_ablation(
    ax: plt.Axes,
    ablation_data: dict[str, list[float]],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    add_label: bool,
) -> None:
    ax.set_gid("fig04.panel_d.main")
    chance_level = min(
        float(np.asarray(ablation_data.get("Dense Routing", [0.0]), dtype=np.float32).mean()),
        0.08,
    )
    display_labels = ["guided", "bypass", "OMP", "dense"]
    x_positions = np.arange(len(DECODER_VARIANTS), dtype=np.float32)

    ax.axhspan(0.0, chance_level, color=STYLE_COLORS["chance_fill"], zorder=0)
    ax.axhline(
        chance_level,
        color=STYLE_COLORS["chance_line"],
        linestyle="--",
        linewidth=FIG04_RULE_LINEWIDTH,
        zorder=1,
    )

    for idx, ((variant_key, _display_label, color), x_pos) in enumerate(zip(DECODER_VARIANTS, x_positions, strict=False)):
        seeds = ablation_data.get(variant_key, [])
        if not seeds:
            continue
        seed_arr = np.asarray(seeds, dtype=np.float32)
        mean_val = float(seed_arr.mean())
        sem_val = 0.0 if seed_arr.size <= 1 else float(seed_arr.std(ddof=1) / np.sqrt(seed_arr.size))
        x_offsets = (
            np.linspace(-0.10, 0.10, num=seed_arr.size, dtype=np.float32)
            if seed_arr.size > 1
            else np.zeros(1, dtype=np.float32)
        )
        ax.scatter(
            np.full(seed_arr.size, x_pos, dtype=np.float32) + x_offsets,
            seed_arr,
            color=color,
            s=FIG04_POINT_MARKER_SIZE,
            alpha=FAMILY_STYLE["summary_point_alpha"],
            edgecolors="white",
            linewidths=FIG04_MARKER_EDGEWIDTH,
            zorder=3,
        )
        ax.errorbar(
            x_pos,
            mean_val,
            yerr=sem_val,
            fmt="o",
            color=color,
            markersize=FIG04_MEAN_MARKER_SIZE,
            capsize=FIG04_ERRORBAR_CAPSIZE,
            linewidth=FIG04_BASE_LINEWIDTH,
            zorder=4,
        )
        ax.text(
            x_pos,
            min(mean_val + 0.055, 1.01),
            f"{mean_val:.2f}",
            fontsize=tick_label_pt - 0.1,
            fontweight="bold",
            va="bottom",
            ha="center",
            color=color,
            zorder=5,
        )

    ax.set_xlim(-0.5, len(DECODER_VARIANTS) - 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="x", labelsize=tick_label_pt - 0.1, length=0)
    ax.tick_params(axis="y", labelsize=tick_label_pt, length=2)
    ax.set_title("Clean decoder family", fontsize=axis_label_pt + 0.1, pad=2.0)
    ax.set_xlabel("Decoder family", fontsize=axis_label_pt, labelpad=1.0)
    ax.set_ylabel("Clean Top-1 accuracy", fontsize=axis_label_pt - 0.1, labelpad=1.0)
    ax.grid(axis="y", linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])
    ax.text(
        0.02,
        0.98,
        "chance floor",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        va="top",
        ha="left",
        color=STYLE_COLORS["muted_text"],
    )
    ax.text(
        0.98,
        0.98,
        "5 seeds",
        transform=ax.transAxes,
        fontsize=tick_label_pt - 0.1,
        va="top",
        ha="right",
        color=STYLE_COLORS["muted_text"],
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
    """Generate Figure 4 manuscript-facing data-backed panels."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    title_pt = typography["title"]

    mechanics_path, mechanics = _load_mechanics(data_root)
    ablation_data = _load_ablation_data(data_root)
    h_sim = _load_h_similarity(data_root)

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

    _plot_panel_a_local_band(
        fig,
        gs[0, 0],
        mechanics,
        h_sim,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    _plot_panel_b(
        fig,
        gs[0, 1],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
        h_similarity=None,
    )
    _plot_panel_c(
        fig,
        gs[1, 0],
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        add_label=False,
    )
    ax_d = fig.add_subplot(gs[1, 1])
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

    fig_a = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["a"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["a"],
    )
    _plot_panel_a_local_band(
        fig_a,
        111,
        mechanics,
        h_sim,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
    )
    fig_a.subplots_adjust(**FIG04_STANDALONE["a"])
    all_paths.extend(
        save_outputs(
            fig_a,
            panel_dir / "fig04_panel_a_local_band",
            typography=typography,
        )
    )
    plt.close(fig_a)

    fig_b = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["b"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["b"],
    )
    _plot_panel_b(
        fig_b,
        111,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=False,
        h_similarity=None,
    )
    fig_b.subplots_adjust(**FIG04_STANDALONE["b"])
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig04_panel_b_representative_overlay",
            typography=typography,
        )
    )
    plt.close(fig_b)

    fig_c = make_figure(
        width_mm=FIG04_PANEL_SLOT_WIDTHS_MM["c"],
        height_mm=FIG04_PANEL_SLOT_HEIGHTS_MM["c"],
    )
    _plot_panel_c(
        fig_c,
        111,
        mechanics,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
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
    ax_d_single = fig_d.add_subplot(111)
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
                "panel_id": "a",
                "title": "Local band in H",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_a_local_band.pdf",
                "provenance_mode": "data_backed",
                "description": "Measured local overlap band from the fixed dictionary, plotted on the shared angle axis around the representative 70-degree target.",
            },
            {
                "panel_id": "b",
                "title": "Representative exemplar",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_b_representative_overlay.pdf",
                "provenance_mode": "data_backed",
                "description": "Representative 70-degree validation exemplar showing the broad match, local gate, and local update on one shared angle axis.",
            },
            {
                "panel_id": "c",
                "title": "Validation set",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_c_update_residual.pdf",
                "provenance_mode": "data_backed",
                "description": "Population cumulative-mass view showing that the local step concentrates update mass inside the matched 15-degree neighborhood across the validation set.",
            },
            {
                "panel_id": "d",
                "title": "Clean decoder family",
                "asset_path": "figures/output/fig04_solver_dynamics_panels/fig04_panel_d_ablation.pdf",
                "provenance_mode": "data_backed",
                "description": "Clean-condition comparison of guided, router-bypass, OMP baseline, and dense routing decoders across the same five-seed sweep.",
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
