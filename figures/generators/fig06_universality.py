"""Figure 6 — Cross-material recurrence under object-specific calibration.

Panel (b): Cross-material H strip across five materials.
Panel (c): Low-rank continuity summary across materials using Fig. 2-style centered-magnitude SVD.
Panel (d): Compact response-strength versus Top-1 readout summary.
Panel (e): Low-emphasis material band/code support strip.

Panel (a) remains a manual support asset and is composed downstream.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

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


MATERIAL_LABELS = {
    "a": "Acrylic plate",
    "p": "Paper cup",
    "w": "Wooden board",
    "b": "Cardboard box",
    "m": "Laptop shell",
}
MATERIAL_SHORT_LABELS = {
    "a": "Acrylic",
    "p": "Paper cup",
    "w": "Wood board",
    "b": "Cardboard",
    "m": "Laptop",
}
MATERIAL_SCREENING_LABELS = {
    "a": "Acrylic",
    "p": "Cup",
    "w": "Wood",
    "b": "Card",
    "m": "Laptop",
}
OTHER_COLOR = "#7A7A7A"
BACKUP_COLOR = SEMANTIC_PALETTE["learned"]
PRIMARY_COLOR = SEMANTIC_PALETTE["physics"]
RANK95_COLOR = "#CC7A00"
GRID_COLOR = "#D8D8D8"
PANEL_C_MATERIAL_STYLES = {
    "b": {"color": PRIMARY_COLOR, "marker": "o"},
    "w": {"color": BACKUP_COLOR, "marker": "^"},
    "a": {"color": SEMANTIC_PALETTE["highlight"], "marker": "s"},
    "p": {"color": SEMANTIC_PALETTE["ablation"], "marker": "D"},
    "m": {"color": "#CC79A7", "marker": "v"},
}

FIG06_GENERATOR = figure_section("fig06", "generator")
FIG06_GRID = dict(FIG06_GENERATOR["composite_grid"])
FIG06_SPLIT = dict(FIG06_GENERATOR["split"]["standalone"])
FIG06_MIDDLE_WIDTH_RATIOS = [
    float(FIG06_SPLIT["c"]["width_mm"]),
    float(FIG06_SPLIT["e"]["width_mm"]),
]
FIG06_TYPOGRAPHY = {
    **font_tokens(),
    **figure_section("fig06", "typography"),
}


def _load_h_matrix(h_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(str(h_path), map_location="cpu", weights_only=False)
    return payload["H"].cpu().numpy(), np.asarray(payload["angles"], dtype=float)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _build_freq_axis(
    n_freq: int,
    *,
    sample_rate: float,
    n_fft: float,
    freq_min: float,
) -> np.ndarray:
    df = sample_rate / n_fft
    k_start = int(np.ceil(freq_min / df))
    return (k_start + np.arange(n_freq)) * df


def _smooth_1d(values: np.ndarray, *, window: int = 3) -> np.ndarray:
    if window <= 1:
        return values.astype(float).copy()
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values.astype(float), kernel, mode="same")


def _centered_magnitude_matrix(h_matrix: np.ndarray) -> np.ndarray:
    magnitude = np.abs(h_matrix).astype(float)
    return magnitude - magnitude.mean(axis=1, keepdims=True)


def _rank_for_energy(cumulative_energy: np.ndarray, threshold: float) -> int:
    return int(np.searchsorted(cumulative_energy, threshold, side="left") + 1)


def _panel_c_material_style(material: str) -> dict[str, str]:
    return PANEL_C_MATERIAL_STYLES[material]


def _save_panel_manifest(
    panel_dir: Path,
    panel_specs: list[dict[str, Any]],
    typography: dict[str, float],
) -> Path:
    manifest_path = panel_dir / "fig06_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig06",
        "composite_asset": "figures/output/fig06_universality.pdf",
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


def _prepare_fig06_data(data_root: Path) -> dict[str, Any]:
    paths_cfg = load_paths()
    h_run_dir = (data_root / paths_cfg["fig06_cross_material_h_run"]).resolve()
    selection_run_dir = (data_root / paths_cfg["fig06_cross_material_selection_run"]).resolve()
    support_run_dir = (data_root / paths_cfg["fig06_cross_material_support_run"]).resolve()

    comparison_report = _load_json(selection_run_dir / "comparison_report.json")
    selection_summary = _load_json(selection_run_dir / "selection_summary.json")
    support_report = _load_json(support_run_dir / "support_report.json")
    performance_rows = _read_csv(support_run_dir / "material_performance_summary.csv")
    low_rank_rows = _read_csv(support_run_dir / "low_rank_summary.csv")

    ranking = tuple(selection_summary["ranking"])
    primary_material = selection_summary["primary_material"]
    backup_material = selection_summary["backup_material"]

    low_rank_by_material = {row["material_code"]: row for row in low_rank_rows}
    performance_by_material = {row["material_code"]: row for row in performance_rows}

    config = comparison_report["config"]
    n_freq = None
    h_matrices: dict[str, np.ndarray] = {}
    angle_map: dict[str, np.ndarray] = {}
    cumulative_energy: dict[str, np.ndarray] = {}
    rank90: dict[str, int] = {}
    rank95: dict[str, int] = {}
    spectral_envelopes: dict[str, np.ndarray] = {}
    angular_contrast_smooth: dict[str, np.ndarray] = {}
    representative_band_freq_hz: dict[str, float] = {}
    representative_band_lo_khz: dict[str, float] = {}
    representative_band_hi_khz: dict[str, float] = {}
    representative_directivity: dict[str, np.ndarray] = {}
    overall_energy: dict[str, float] = {}

    for material in ranking:
        h_path = h_run_dir / f"h_matrix_normalized_original_to_{material}.pth"
        if not h_path.exists():
            raise FileNotFoundError(f"Missing H matrix for material {material}: {h_path}")
        h_matrix, angles = _load_h_matrix(h_path)
        if n_freq is None:
            n_freq = h_matrix.shape[0]
        h_matrices[material] = h_matrix
        angle_map[material] = angles

    if n_freq is None:
        raise RuntimeError("No H matrices were loaded for Fig. 6.")

    freqs_hz = _build_freq_axis(
        n_freq,
        sample_rate=float(config["sample_rate"]),
        n_fft=float(config["n_fft"]),
        freq_min=float(config["freq_min"]),
    )
    freq_mask = (freqs_hz >= 300.0) & (freqs_hz <= 3000.0)
    valid_freq_indices = np.flatnonzero(freq_mask)
    if valid_freq_indices.size == 0:
        raise RuntimeError("No valid 300-3000 Hz bins were found for Fig. 6.")

    for material in ranking:
        h_matrix = h_matrices[material]
        singular_values = np.linalg.svd(_centered_magnitude_matrix(h_matrix), compute_uv=False)
        singular_energy = singular_values**2
        cumulative_energy[material] = np.cumsum(singular_energy) / (singular_energy.sum() + 1e-12)

        low_rank_row = low_rank_by_material[material]
        computed_rank90 = _rank_for_energy(cumulative_energy[material], 0.90)
        computed_rank95 = _rank_for_energy(cumulative_energy[material], 0.95)
        expected_rank90 = int(float(low_rank_row["rank_90_energy"]))
        expected_rank95 = int(float(low_rank_row["rank_95_energy"]))
        if computed_rank90 != expected_rank90 or computed_rank95 != expected_rank95:
            raise RuntimeError(
                f"Fig. 6 low-rank summary mismatch for material {material}: "
                f"computed ({computed_rank90}, {computed_rank95}) vs "
                f"summary ({expected_rank90}, {expected_rank95})"
            )
        rank90[material] = computed_rank90
        rank95[material] = computed_rank95

        magnitude = np.abs(h_matrix).astype(float)
        overall_energy[material] = float((magnitude**2).mean())
        envelope = magnitude.mean(axis=1)
        spectral_envelopes[material] = envelope / (envelope.max() + 1e-12)

        contrast = magnitude.std(axis=1) / (magnitude.mean(axis=1) + 1e-12)
        smoothed = _smooth_1d(contrast, window=3)
        angular_contrast_smooth[material] = smoothed

        peak_rel = int(np.argmax(smoothed[freq_mask]))
        peak_idx = int(valid_freq_indices[peak_rel])
        representative_band_freq_hz[material] = float(freqs_hz[peak_idx])
        lo = max(0, peak_idx - 1)
        hi = min(magnitude.shape[0], peak_idx + 2)
        representative_band_lo_khz[material] = float(freqs_hz[lo] / 1000.0)
        representative_band_hi_khz[material] = float(freqs_hz[hi - 1] / 1000.0)

        band_profile = magnitude[lo:hi].mean(axis=0)
        band_min = float(band_profile.min())
        band_max = float(band_profile.max())
        if band_max - band_min <= 1e-12:
            representative_directivity[material] = np.zeros_like(band_profile)
        else:
            representative_directivity[material] = (band_profile - band_min) / (band_max - band_min)

    energy_values = np.asarray([overall_energy[material] for material in ranking], dtype=float)
    top1_values = np.asarray(
        [float(performance_by_material[material]["top1_acc"]) for material in ranking],
        dtype=float,
    )

    def _normalize_across_materials(values: np.ndarray) -> np.ndarray:
        span = float(values.max() - values.min())
        if span <= 1e-12:
            return np.zeros_like(values)
        return (values - values.min()) / span

    normalized_energy = _normalize_across_materials(energy_values)
    normalized_top1 = _normalize_across_materials(top1_values)

    return {
        "comparison_report": comparison_report,
        "selection_summary": selection_summary,
        "support_report": support_report,
        "performance_rows": performance_rows,
        "performance_by_material": performance_by_material,
        "ranking": ranking,
        "primary_material": primary_material,
        "backup_material": backup_material,
        "h_matrices": h_matrices,
        "angles": angle_map,
        "freqs_khz": freqs_hz / 1000.0,
        "cumulative_energy": cumulative_energy,
        "rank90": rank90,
        "rank95": rank95,
        "spectral_envelopes": spectral_envelopes,
        "angular_contrast_smooth": angular_contrast_smooth,
        "representative_band_freq_hz": representative_band_freq_hz,
        "representative_band_lo_khz": representative_band_lo_khz,
        "representative_band_hi_khz": representative_band_hi_khz,
        "representative_directivity": representative_directivity,
        "normalized_energy": {
            material: float(normalized_energy[idx])
            for idx, material in enumerate(ranking)
        },
        "normalized_top1": {
            material: float(normalized_top1[idx])
            for idx, material in enumerate(ranking)
        },
    }


def _material_color(material: str, *, primary: str, backup: str | None) -> str:
    if material == primary:
        return PRIMARY_COLOR
    if material == backup:
        return BACKUP_COLOR
    return OTHER_COLOR


def _screening_material_style(material: str) -> dict[str, str]:
    return PANEL_C_MATERIAL_STYLES.get(material, {"color": OTHER_COLOR, "marker": "o"})


def _make_panel_block(
    fig: plt.Figure,
    slot_spec,
    *,
    label: str,
    title: str | None,
    title_pt: float,
    add_label: bool,
    label_x: float = 0.0,
    label_y: float = 1.005,
    title_x: float = 0.06,
    title_y: float = 1.005,
):
    ax_block = fig.add_subplot(slot_spec)
    ax_block.set_axis_off()
    ax_block.set_xticks([])
    ax_block.set_yticks([])
    if add_label:
        add_panel_label(ax_block, label, x=label_x, y=label_y)
    if title:
        ax_block.text(
            title_x,
            title_y,
            title,
            transform=ax_block.transAxes,
            fontsize=title_pt,
            ha="left",
            va="bottom",
            color="black",
        )
    return ax_block


def _plot_panel_b(
    fig: plt.Figure,
    slot_spec,
    data: dict[str, Any],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    colorbar_tick_pt: float,
    colorbar_label_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ranking = data["ranking"]
    ax_block = _make_panel_block(
        fig,
        slot_spec,
        label="b",
        title=None,
        title_pt=axis_label_pt,
        add_label=add_label,
    )
    heatmap_grid = slot_spec.subgridspec(
        1,
        len(ranking) + 1,
        width_ratios=[1] * len(ranking) + [0.06],
        wspace=0.08,
    )

    heatmap_axes = [fig.add_subplot(heatmap_grid[0, idx]) for idx in range(len(ranking))]
    cax = fig.add_subplot(heatmap_grid[0, len(ranking)])

    log_mats = [
        np.log10(np.clip(np.abs(data["h_matrices"][material]), 1e-8, None))
        for material in ranking
    ]
    stacked = np.concatenate([mat.ravel() for mat in log_mats])
    vmin = float(np.percentile(stacked, 2))
    vmax = float(np.percentile(stacked, 98))
    freqs_khz = data["freqs_khz"]
    y_ticks = [0.5, 1.0, 2.0, 3.0]
    image = None

    for idx, material in enumerate(ranking):
        ax = heatmap_axes[idx]
        mat = log_mats[idx]
        angles = data["angles"][material]
        image = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[angles.min(), angles.max(), freqs_khz.min(), freqs_khz.max()],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(MATERIAL_SHORT_LABELS[material], fontsize=title_pt - 0.2, pad=2)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="both", labelsize=tick_label_pt, length=2)
        if idx == 0:
            ax.set_ylabel("Freq. (kHz)", fontsize=axis_label_pt)
        else:
            ax.set_yticklabels([])
    heatmap_axes[len(heatmap_axes) // 2].set_xlabel("Angle (deg)", fontsize=axis_label_pt)

    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt, length=2)
    cbar.set_label(r"$\log_{10}|H|$", fontsize=colorbar_label_pt, labelpad=2)
    return [ax_block, *heatmap_axes, cax]


def _plot_panel_c(
    fig: plt.Figure,
    slot_spec,
    data: dict[str, Any],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ranking = data["ranking"]
    ax_block = _make_panel_block(
        fig,
        slot_spec,
        label="c",
        title=None,
        title_pt=title_pt,
        add_label=add_label,
    )
    grid = slot_spec.subgridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.18)
    ax_curve = fig.add_subplot(grid[0, 0])
    ax_rank = fig.add_subplot(grid[0, 1])

    max_modes = 8
    x = np.arange(1, max_modes + 1)
    for material in ranking:
        style = _panel_c_material_style(material)
        ax_curve.plot(
            x,
            data["cumulative_energy"][material][: len(x)],
            marker=style["marker"],
            linewidth=1.45,
            markersize=3.0,
            color=style["color"],
            label=MATERIAL_SHORT_LABELS[material],
        )
    for threshold in (0.90, 0.95):
        ax_curve.axhline(threshold, color=GRID_COLOR, linestyle="--", linewidth=0.9, zorder=0)
    ax_curve.set_xlim(1.0, max_modes + 0.2)
    ax_curve.set_ylim(0.0, 1.02)
    ax_curve.set_xticks(x)
    ax_curve.set_ylabel(r"Cumul. centered-$|H|$ energy", fontsize=axis_label_pt)
    ax_curve.set_xlabel("Modes kept", fontsize=axis_label_pt)
    ax_curve.tick_params(axis="both", labelsize=tick_label_pt, length=2)
    ax_curve.grid(True, alpha=0.18)
    ax_curve.legend(
        frameon=False,
        fontsize=legend_pt - 0.4,
        loc="lower right",
        ncol=2,
        handlelength=1.4,
        borderpad=0.1,
        labelspacing=0.15,
        columnspacing=0.8,
    )

    y = np.arange(len(ranking))
    for idx, material in enumerate(ranking):
        style = _panel_c_material_style(material)
        ax_rank.hlines(
            idx,
            float(data["rank90"][material]),
            float(data["rank95"][material]),
            color="#C8C8C8",
            linewidth=2.2,
            zorder=1,
        )
        ax_rank.scatter(
            data["rank90"][material],
            idx,
            s=24,
            color=style["color"],
            zorder=3,
            label="Rank90" if idx == 0 else None,
        )
        ax_rank.scatter(
            data["rank95"][material],
            idx,
            s=26,
            marker="s",
            color=RANK95_COLOR,
            zorder=4,
            label="Rank95" if idx == 0 else None,
        )
    ax_rank.set_title("Modes needed", fontsize=title_pt - 0.15, pad=2)
    ax_rank.set_xlim(0.5, max(max(data["rank90"].values()), max(data["rank95"].values())) + 1.0)
    ax_rank.set_yticks(y)
    ax_rank.set_yticklabels([MATERIAL_SHORT_LABELS[m] for m in ranking], fontsize=tick_label_pt)
    for label, material in zip(ax_rank.get_yticklabels(), ranking):
        label.set_color(_panel_c_material_style(material)["color"])
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Modes", fontsize=axis_label_pt)
    ax_rank.tick_params(axis="x", labelsize=tick_label_pt, length=2)
    ax_rank.grid(axis="x", linestyle="--", alpha=0.20)
    ax_rank.legend(
        frameon=False,
        fontsize=legend_pt - 0.4,
        loc="upper right",
        handletextpad=0.25,
        borderpad=0.1,
    )
    return [ax_block, ax_curve, ax_rank]


def _plot_panel_d(
    fig: plt.Figure,
    slot_spec,
    data: dict[str, Any],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ranking = data["ranking"]
    freqs_khz = data["freqs_khz"]
    ax_block = _make_panel_block(
        fig,
        slot_spec,
        label="e",
        title=None,
        title_pt=title_pt,
        add_label=add_label,
    )
    grid = slot_spec.subgridspec(
        len(ranking),
        3,
        width_ratios=[0.58, 1.60, 0.94],
        hspace=0.26,
        wspace=0.16,
    )
    axes: list[plt.Axes] = []
    row_axes: list[tuple[plt.Axes, plt.Axes, plt.Axes]] = []
    column_headers = ("Selected band", "Recovered code")
    contrast_ymax = max(float(data["angular_contrast_smooth"][material].max()) for material in ranking)
    contrast_ymax = max(contrast_ymax, 1e-6)

    def _style_row_axis(ax: plt.Axes, *, bottom_row: bool) -> None:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#C6C6C6")
        ax.spines["bottom"].set_color("#C6C6C6")
        ax.spines["left"].set_linewidth(0.55)
        ax.spines["bottom"].set_linewidth(0.55)
        ax.tick_params(axis="y", left=False, labelleft=False)
        if bottom_row:
            ax.tick_params(axis="x", labelsize=tick_label_pt, length=2, pad=1)
        else:
            ax.tick_params(axis="x", bottom=False, labelbottom=False)

    for row_idx, material in enumerate(ranking):
        tint = _material_color(
            material,
            primary=data["primary_material"],
            backup=data["backup_material"],
        )
        ax_label = fig.add_subplot(grid[row_idx, 0])
        ax_freq = fig.add_subplot(grid[row_idx, 1])
        ax_code = fig.add_subplot(grid[row_idx, 2])

        band_lo = data["representative_band_lo_khz"][material]
        band_hi = data["representative_band_hi_khz"][material]
        freq_hz = data["representative_band_freq_hz"][material]
        angles = data["angles"][material]
        band_center_khz = freq_hz / 1000.0
        bottom_row = row_idx == len(ranking) - 1
        support_alpha = 0.42 if material in {data["primary_material"], data["backup_material"]} else 0.30
        support = data["spectral_envelopes"][material]
        contrast = data["angular_contrast_smooth"][material] / contrast_ymax

        ax_label.set_axis_off()
        ax_label.text(
            0.98,
            0.5,
            MATERIAL_LABELS[material],
            transform=ax_label.transAxes,
            ha="right",
            va="center",
            fontsize=max(axis_label_pt - 0.4, 5.4),
            color="#4A4A4A",
        )

        ax_freq.axvspan(band_lo, band_hi, color=tint, alpha=0.14, zorder=0)
        ax_freq.fill_between(freqs_khz, 0, support, color="#BFC4C7", alpha=0.16)
        ax_freq.plot(
            freqs_khz,
            support,
            color="#8B8F93",
            alpha=max(support_alpha, 0.36),
            linewidth=0.9,
        )
        ax_freq.plot(
            freqs_khz,
            contrast,
            color="#3A3D40",
            linewidth=1.15,
        )
        ax_freq.axvline(band_center_khz, color=tint, linewidth=0.9, linestyle="--", alpha=0.55)
        ax_freq.set_xlim(0.3, 3.0)
        ax_freq.set_ylim(0.0, 1.02)
        ax_freq.set_xticks([0.5, 1.5, 2.5])
        ax_freq.text(
            0.98,
            0.80,
            f"{band_center_khz:.2f} kHz",
            transform=ax_freq.transAxes,
            ha="right",
            va="center",
            fontsize=max(tick_label_pt - 0.2, 5.0),
            color="black",
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
        )
        _style_row_axis(ax_freq, bottom_row=bottom_row)

        ax_code.fill_between(angles, 0, data["representative_directivity"][material], color=tint, alpha=0.10)
        ax_code.plot(
            angles,
            data["representative_directivity"][material],
            color="#3A3D40",
            linewidth=1.25,
        )
        ax_code.set_xlim(float(angles.min()), float(angles.max()))
        ax_code.set_ylim(-0.02, 1.02)
        ax_code.set_xticks([0, 90, 180])
        _style_row_axis(ax_code, bottom_row=bottom_row)

        if bottom_row:
            ax_freq.set_xlabel("Freq. (kHz)", fontsize=axis_label_pt, labelpad=1.5)
            ax_code.set_xlabel("Angle (deg)", fontsize=axis_label_pt, labelpad=1.5)

        axes.extend([ax_label, ax_freq, ax_code])
        row_axes.append((ax_label, ax_freq, ax_code))

    block_bbox = ax_block.get_position()
    x_left = (row_axes[0][0].get_position().x0 - block_bbox.x0) / block_bbox.width
    x_right = (row_axes[0][2].get_position().x1 - block_bbox.x0) / block_bbox.width
    for col_idx, header in enumerate(column_headers):
        axis_bbox = row_axes[0][col_idx + 1].get_position()
        x_center = ((axis_bbox.x0 + axis_bbox.x1) * 0.5 - block_bbox.x0) / block_bbox.width
        ax_block.text(
            x_center,
            1.015,
            header,
            transform=ax_block.transAxes,
            ha="center",
            va="bottom",
            fontsize=max(title_pt - 0.35, 5.6),
            color="#2A2A2A",
        )

    for row_idx in range(len(row_axes) - 1):
        upper_bbox = row_axes[row_idx][1].get_position()
        lower_bbox = row_axes[row_idx + 1][1].get_position()
        y_sep = ((upper_bbox.y0 + lower_bbox.y1) * 0.5 - block_bbox.y0) / block_bbox.height
        ax_block.plot(
            [x_left, x_right],
            [y_sep, y_sep],
            transform=ax_block.transAxes,
            color="#E1E1E1",
            linewidth=0.85,
            zorder=0,
            clip_on=False,
        )

    return [ax_block, *axes]


def _plot_panel_e(
    fig: plt.Figure,
    slot_spec,
    data: dict[str, Any],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ranking = data["ranking"]
    ax_block = _make_panel_block(
        fig,
        slot_spec,
        label="d",
        title=None,
        title_pt=title_pt,
        add_label=add_label,
        label_y=1.018,
    )
    ax = fig.add_subplot(slot_spec)

    y = np.arange(len(ranking))
    for idx, material in enumerate(ranking):
        style = _screening_material_style(material)
        tint = style["color"]
        marker = style["marker"]
        if material in {data["primary_material"], data["backup_material"]}:
            ax.axhspan(idx - 0.48, idx + 0.48, color=tint, alpha=0.06, zorder=0)

        row = data["performance_by_material"][material]
        energy_x = data["normalized_energy"][material]
        top1 = float(row["top1_acc"])
        top1_lo = float(row["top1_ci_low"])
        top1_hi = float(row["top1_ci_high"])
        ax.plot([energy_x, top1], [idx, idx], color="#B5B5B5", linewidth=1.25, zorder=1)
        ax.errorbar(
            top1,
            idx,
            xerr=[[top1 - top1_lo], [top1_hi - top1]],
            fmt="none",
            capsize=2.3,
            ecolor=tint,
            linewidth=1.0,
            zorder=2,
        )
        ax.scatter(
            energy_x,
            idx,
            s=72,
            facecolors="white",
            edgecolors=tint,
            linewidths=1.45,
            marker=marker,
            zorder=3,
        )
        ax.scatter(top1, idx, s=80, color=tint, marker=marker, zorder=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#AFAFAF")
    ax.spines["bottom"].set_color("#AFAFAF")
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)

    compact_tick_pt = max(tick_label_pt - 0.25, 5.0)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Norm. energy and Top-1", fontsize=axis_label_pt, labelpad=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [MATERIAL_SCREENING_LABELS[m] for m in ranking],
        fontsize=compact_tick_pt,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=tick_label_pt, length=2, pad=1)
    ax.tick_params(axis="y", length=0, pad=1)
    ax.grid(axis="x", linestyle="--", alpha=0.22)
    ax.text(
        0.98,
        1.02,
        "open = energy, filled = Top-1 ± CI",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=compact_tick_pt,
    )
    return [ax_block, ax]


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate the Fig. 6 data-backed panels and generator composite."""
    typography = dict(FIG06_TYPOGRAPHY)
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    data = _prepare_fig06_data(data_root)

    fig = make_figure(
        width_mm=FIG06_GENERATOR["composite_width_mm"],
        height_mm=FIG06_GENERATOR["composite_height_mm"],
    )
    outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        left=FIG06_GRID["left"],
        right=FIG06_GRID["right"],
        bottom=FIG06_GRID["bottom"],
        top=FIG06_GRID["top"],
        hspace=FIG06_GRID["hspace"],
        wspace=FIG06_GRID["wspace"],
        width_ratios=FIG06_GRID["width_ratios"],
        height_ratios=FIG06_GRID["height_ratios"],
    )

    _plot_panel_b(
        fig,
        outer[0, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        colorbar_tick_pt=colorbar_tick_pt,
        colorbar_label_pt=colorbar_label_pt,
        add_label=True,
    )
    middle = outer[1, 0].subgridspec(1, 2, width_ratios=FIG06_MIDDLE_WIDTH_RATIOS, wspace=0.16)
    _plot_panel_c(
        fig,
        middle[0, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=True,
    )
    _plot_panel_e(
        fig,
        middle[0, 1],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=True,
    )
    _plot_panel_d(
        fig,
        outer[2, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=True,
    )

    all_paths = save_outputs(
        fig,
        output_dir / "fig06_universality",
        typography=typography,
    )
    plt.close(fig)

    panel_dir = output_dir / "fig06_universality_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    standalone_specs = [
        (
            "b",
            "b",
            _plot_panel_b,
            "fig06_panel_b_cross_material_h",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "title_pt": title_pt,
                "colorbar_tick_pt": colorbar_tick_pt,
                "colorbar_label_pt": colorbar_label_pt,
                "add_label": True,
            },
        ),
        (
            "c",
            "c",
            _plot_panel_c,
            "fig06_panel_c_low_rank_continuity",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "legend_pt": legend_pt,
                "title_pt": title_pt,
                "add_label": True,
            },
        ),
        (
            "d",
            "e",
            _plot_panel_e,
            "fig06_panel_d_screening_consequence",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "title_pt": title_pt,
                "add_label": True,
            },
        ),
        (
            "e",
            "d",
            _plot_panel_d,
            "fig06_panel_e_material_frequency_structure",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "title_pt": title_pt,
                "add_label": True,
            },
        ),
    ]

    for panel_id, size_key, plot_fn, stem, kwargs in standalone_specs:
        fig_panel = make_figure(
            width_mm=FIG06_SPLIT[size_key]["width_mm"],
            height_mm=FIG06_SPLIT[size_key]["height_mm"],
        )
        slot = gridspec.GridSpec(1, 1, figure=fig_panel)[0, 0]
        plot_fn(fig_panel, slot, data, **kwargs)
        fig_panel.subplots_adjust(**FIG06_SPLIT[size_key]["subplots_adjust"])
        all_paths.extend(save_outputs(fig_panel, panel_dir / stem, typography=typography))
        plt.close(fig_panel)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "b",
                "title": "Cross-material H",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_b_cross_material_h.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared-normalization H strip for the five compared materials, showing that structured angle-frequency encoding is present across the full object set.",
            },
            {
                "panel_id": "c",
                "title": "Low-rank continuity",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_c_low_rank_continuity.pdf",
                "provenance_mode": "data_backed",
                "description": "Fig. 2-aligned centered-magnitude cumulative-energy curves and rank90/rank95 summaries showing that the encoder remains low-dimensional across materials.",
            },
            {
                "panel_id": "d",
                "title": "Object-conditioned readout and energy",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_screening_consequence.pdf",
                "provenance_mode": "data_backed",
                "description": "Compact normalized-energy versus Top-1 comparison with Top-1 uncertainty, showing that overall response strength alone does not explain the object-level readout ranking.",
            },
            {
                "panel_id": "e",
                "title": "Material frequency structure",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_e_material_frequency_structure.pdf",
                "provenance_mode": "data_backed",
                "description": "Five-row support strip showing the selected frequency band and recovered representative directional code for each compared material.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig06] Generated {len(all_paths)} files")
    return all_paths
