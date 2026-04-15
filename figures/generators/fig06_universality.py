"""Figure 6 — Cross-material recurrence under object-specific calibration.

Panel (a): Measured response-regime map across materials.
Panel (b): Cross-material H strip with the object-specific informative-band window.
Panel (c): Inter-angle correlation-decay summary across materials.
Panel (d): Readout summary against subspace-overlap burden and response-energy descriptors.
Panel (e): Object-specific informative-band windows and their recovered directional codes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

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
GRID_COLOR = STYLE_COLORS["chance_line"]
MATERIAL_PANEL_STYLES = {
    "a": {"color": SEMANTIC_PALETTE["physics"], "marker": "s"},
    "b": {"color": SEMANTIC_PALETTE["learned"], "marker": "o"},
    "w": {"color": SEMANTIC_PALETTE["highlight"], "marker": "^"},
    "p": {"color": SEMANTIC_PALETTE["ablation"], "marker": "D"},
    "m": {"color": STYLE_COLORS["dense_routing"], "marker": "v"},
}

FIG06_GENERATOR = figure_section("fig06", "generator")
FIG06_GRID = dict(FIG06_GENERATOR["composite_grid"])
FIG06_SPLIT = dict(FIG06_GENERATOR["split"]["standalone"])
REPO_ROOT = Path(__file__).resolve().parents[2]
PANEL_A_SOURCE = (
    REPO_ROOT
    / "figures/output/fig06_cross_material_universality_panels/fig06_panel_a_material_exemplars.png"
)
PANEL_A_SOURCE_ORDER = ("a", "p", "w", "b", "m")
FIG06_MIDDLE_WIDTH_RATIOS = [
    float(FIG06_SPLIT["c"]["width_mm"]),
    float(FIG06_SPLIT["d"]["width_mm"]),
]
FIG06_MIDDLE_ROW = dict(FIG06_GENERATOR["middle_row"])
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
    return MATERIAL_PANEL_STYLES[material]


def _trim_white_rgb(tile: np.ndarray, *, pad_px: int = 6) -> np.ndarray:
    if tile.ndim != 3 or tile.shape[2] < 3:
        return tile
    mask = np.any(tile[:, :, :3] < 245, axis=2)
    if not mask.any():
        return tile
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad_px, 0)
    y1 = min(int(ys.max()) + pad_px + 1, tile.shape[0])
    x0 = max(int(xs.min()) - pad_px, 0)
    x1 = min(int(xs.max()) + pad_px + 1, tile.shape[1])
    return tile[y0:y1, x0:x1]


def _load_panel_a_thumbnails() -> dict[str, np.ndarray]:
    if not PANEL_A_SOURCE.exists():
        raise FileNotFoundError(f"Missing Fig. 6 panel-a source asset: {PANEL_A_SOURCE}")
    src = np.asarray(Image.open(PANEL_A_SOURCE).convert("RGB"))
    h, w, _ = src.shape
    photo_strip = src[int(round(h * 0.32)) : int(round(h * 0.90)), int(round(w * 0.01)) : int(round(w * 0.99))]
    tile_w = photo_strip.shape[1] // 5
    thumbs: dict[str, np.ndarray] = {}
    for idx, material in enumerate(PANEL_A_SOURCE_ORDER):
        left = idx * tile_w
        right = photo_strip.shape[1] if idx == 4 else (idx + 1) * tile_w
        tile = photo_strip[:, left:right]
        thumbs[material] = _trim_white_rgb(tile, pad_px=4)
    return thumbs


def _correlation_decay(centered_mag: np.ndarray, angle_step_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    angle_vectors = centered_mag.T.astype(float)
    angle_vectors = angle_vectors - angle_vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(angle_vectors, axis=1, keepdims=True) + 1e-12
    normalized = angle_vectors / norms
    corr = normalized @ normalized.T
    n_angles = corr.shape[0]
    separations = np.arange(n_angles, dtype=float) * angle_step_deg
    mean_corr = np.empty(n_angles, dtype=float)
    for lag in range(n_angles):
        vals = [corr[i, i + lag] for i in range(n_angles - lag)]
        if lag > 0:
            vals.extend(corr[i + lag, i] for i in range(n_angles - lag))
        mean_corr[lag] = float(np.mean(vals))
    nonzero = mean_corr[1:]
    sep_nonzero = separations[1:]
    crossing = np.where(nonzero <= 0.0)[0]
    zero_cross_deg = float(sep_nonzero[crossing[0]]) if crossing.size else float(sep_nonzero[-1])
    return separations, mean_corr, zero_cross_deg


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
    angle_table_path = (data_root / paths_cfg["fig06_cross_material_angle_table"]).resolve()

    comparison_report = _load_json(selection_run_dir / "comparison_report.json")
    selection_summary = _load_json(selection_run_dir / "selection_summary.json")
    support_report = _load_json(support_run_dir / "support_report.json")
    performance_rows = _read_csv(support_run_dir / "material_performance_summary.csv")
    low_rank_rows = _read_csv(support_run_dir / "low_rank_summary.csv")
    angle_rows = _read_csv(angle_table_path)

    ranking = tuple(selection_summary["ranking"])
    primary_material = selection_summary["primary_material"]
    backup_material = selection_summary["backup_material"]

    low_rank_by_material = {row["material_code"]: row for row in low_rank_rows}
    performance_by_material = {row["material_code"]: row for row in performance_rows}
    angle_top1_by_material: dict[str, np.ndarray] = {}

    config = comparison_report["config"]
    n_freq = None
    h_matrices: dict[str, np.ndarray] = {}
    angle_map: dict[str, np.ndarray] = {}
    cumulative_energy: dict[str, np.ndarray] = {}
    rank90: dict[str, int] = {}
    rank95: dict[str, int] = {}
    effective_rank: dict[str, float] = {}
    spectral_envelopes: dict[str, np.ndarray] = {}
    angular_contrast_smooth: dict[str, np.ndarray] = {}
    representative_band_freq_hz: dict[str, float] = {}
    representative_band_lo_khz: dict[str, float] = {}
    representative_band_hi_khz: dict[str, float] = {}
    representative_directivity: dict[str, np.ndarray] = {}
    overall_energy: dict[str, float] = {}
    corr_decay_deg: np.ndarray | None = None
    corr_decay_by_material: dict[str, np.ndarray] = {}
    corr_zero_cross_deg: dict[str, float] = {}
    mean_top3_overlap_by_material: dict[str, float] = {}

    for material in ranking:
        h_path = h_run_dir / f"h_matrix_normalized_original_to_{material}.pth"
        if not h_path.exists():
            raise FileNotFoundError(f"Missing H matrix for material {material}: {h_path}")
        h_matrix, angles = _load_h_matrix(h_path)
        if n_freq is None:
            n_freq = h_matrix.shape[0]
        h_matrices[material] = h_matrix
        angle_map[material] = angles
        material_angle_rows = sorted(
            (row for row in angle_rows if row["material_code"] == material),
            key=lambda row: float(row["angle_deg"]),
        )
        angle_top1_by_material[material] = np.asarray(
            [float(row["angle_top1_acc"]) for row in material_angle_rows],
            dtype=float,
        )
        mean_top3_overlap_by_material[material] = float(
            np.mean([float(row["mean_top3_subspace_overlap_to_others"]) for row in material_angle_rows])
        )

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
        effective_rank[material] = float(low_rank_row["effective_rank"])

        magnitude = np.abs(h_matrix).astype(float)
        overall_energy[material] = float((magnitude**2).mean())
        envelope = magnitude.mean(axis=1)
        spectral_envelopes[material] = envelope / (envelope.max() + 1e-12)

        separations, mean_corr, zero_cross_deg = _correlation_decay(
            _centered_magnitude_matrix(h_matrix),
            angle_step_deg=float(np.median(np.diff(angles))),
        )
        if corr_decay_deg is None:
            corr_decay_deg = separations
        corr_decay_by_material[material] = mean_corr
        corr_zero_cross_deg[material] = zero_cross_deg

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
    zero_cross_values = np.asarray([corr_zero_cross_deg[material] for material in ranking], dtype=float)
    normalized_zero_cross = _normalize_across_materials(zero_cross_values)

    if corr_decay_deg is None:
        raise RuntimeError("No correlation-decay curves were computed for Fig. 6.")

    return {
        "comparison_report": comparison_report,
        "selection_summary": selection_summary,
        "support_report": support_report,
        "performance_rows": performance_rows,
        "performance_by_material": performance_by_material,
        "angle_top1_by_material": angle_top1_by_material,
        "ranking": ranking,
        "primary_material": primary_material,
        "backup_material": backup_material,
        "h_matrices": h_matrices,
        "angles": angle_map,
        "freqs_khz": freqs_hz / 1000.0,
        "cumulative_energy": cumulative_energy,
        "rank90": rank90,
        "rank95": rank95,
        "effective_rank": effective_rank,
        "spectral_envelopes": spectral_envelopes,
        "angular_contrast_smooth": angular_contrast_smooth,
        "representative_band_freq_hz": representative_band_freq_hz,
        "representative_band_lo_khz": representative_band_lo_khz,
        "representative_band_hi_khz": representative_band_hi_khz,
        "representative_directivity": representative_directivity,
        "corr_decay_deg": corr_decay_deg,
        "corr_decay_by_material": corr_decay_by_material,
        "corr_zero_cross_deg": corr_zero_cross_deg,
        "mean_top3_overlap_by_material": mean_top3_overlap_by_material,
        "normalized_energy": {
            material: float(normalized_energy[idx])
            for idx, material in enumerate(ranking)
        },
        "normalized_top1": {
            material: float(normalized_top1[idx])
            for idx, material in enumerate(ranking)
        },
        "normalized_zero_cross": {
            material: float(normalized_zero_cross[idx])
            for idx, material in enumerate(ranking)
        },
    }


def _screening_material_style(material: str) -> dict[str, str]:
    return MATERIAL_PANEL_STYLES.get(
        material,
        {"color": STYLE_COLORS["dense_routing"], "marker": "o"},
    )


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
        width_ratios=[1] * len(ranking) + [0.085],
        wspace=0.06,
    )

    heatmap_axes = [fig.add_subplot(heatmap_grid[0, idx]) for idx in range(len(ranking))]
    cax = fig.add_subplot(heatmap_grid[0, len(ranking)])
    cax.set_gid("fig06.panel_b.colorbar")

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
        ax.set_title(MATERIAL_SHORT_LABELS[material], fontsize=title_pt - 0.3, pad=0.8)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="both", labelsize=tick_label_pt, length=2, pad=0.8)
        ax.axhspan(
            data["representative_band_lo_khz"][material],
            data["representative_band_hi_khz"][material],
            facecolor="none",
            edgecolor=_screening_material_style(material)["color"],
            linewidth=STROKE_TOKENS["annotation"],
            linestyle="--",
        )
        if idx == 0:
            ax.set_ylabel("Freq. (kHz)", fontsize=axis_label_pt)
        else:
            ax.set_yticklabels([])
    heatmap_axes[len(heatmap_axes) // 2].set_xlabel(
        "Angle (deg)",
        fontsize=axis_label_pt,
        labelpad=0.4,
    )

    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt, length=2)
    cbar.set_label("log10|H|", fontsize=colorbar_label_pt, labelpad=0.6)
    return [ax_block, *heatmap_axes, cax]


def _plot_panel_a(
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
        label="a",
        title="Measured response-regime map",
        title_pt=title_pt,
        add_label=add_label,
        title_x=0.03,
        title_y=1.03,
    )
    ax = fig.add_subplot(slot_spec)
    thumbs = _load_panel_a_thumbnails()

    x_vals = np.asarray([data["corr_zero_cross_deg"][material] for material in ranking], dtype=float)
    y_vals = np.asarray([data["effective_rank"][material] for material in ranking], dtype=float)
    x_pad = max(3.0, 0.18 * float(x_vals.max() - x_vals.min() + 1e-9))
    y_pad = max(1.8, 0.22 * float(y_vals.max() - y_vals.min() + 1e-9))
    ax.set_xlim(float(x_vals.min() - x_pad), float(x_vals.max() + x_pad))
    ax.set_ylim(max(0.0, float(y_vals.min() - y_pad)), float(y_vals.max() + y_pad))
    ax.grid(True, linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])

    thumb_offsets = {
        "b": (18, -2),
        "w": (0, 18),
        "a": (-22, -2),
        "p": (-22, 0),
        "m": (20, 0),
    }
    label_offsets = {
        "b": (14, -14),
        "w": (0, -18),
        "a": (24, 8),
        "p": (-14, -14),
        "m": (14, -14),
    }
    for material in ranking:
        style = _screening_material_style(material)
        x = float(data["corr_zero_cross_deg"][material])
        y = float(data["effective_rank"][material])
        ax.scatter(
            [x],
            [y],
            s=58,
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=STROKE_TOKENS["base"],
            zorder=3,
        )
        thumb = OffsetImage(thumbs[material], zoom=0.058)
        thumb_dx, thumb_dy = thumb_offsets.get(material, (18, 0))
        ab = AnnotationBbox(
            thumb,
            (x, y),
            xybox=(thumb_dx, thumb_dy),
            boxcoords="offset points",
            frameon=True,
            bboxprops={
                "edgecolor": style["color"],
                "linewidth": STROKE_TOKENS["base"],
                "boxstyle": "round,pad=0.08",
            },
            pad=0.02,
            zorder=2,
        )
        ax.add_artist(ab)
        dx, dy = label_offsets.get(material, (0, 16))
        ax.annotate(
            MATERIAL_SHORT_LABELS[material],
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="center",
            va="bottom" if dy >= 0 else "top",
            fontsize=max(tick_label_pt - 0.2, 5.4),
            color=style["color"],
        )

    ax.set_xlabel("Correlation-decay width (deg)", fontsize=axis_label_pt, labelpad=0.2)
    ax.set_ylabel("Effective rank", fontsize=axis_label_pt)
    ax.tick_params(axis="both", labelsize=tick_label_pt, length=2, pad=0.8)
    return [ax_block, ax]


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
    ax_curve = fig.add_subplot(slot_spec)

    x = data["corr_decay_deg"]
    focus_mask = x <= 90.0
    x_focus = x[focus_mask]
    for material in ranking:
        style = _panel_c_material_style(material)
        curve = _smooth_1d(data["corr_decay_by_material"][material], window=3)[focus_mask]
        ax_curve.plot(
            x_focus,
            curve,
            marker=style["marker"],
            linewidth=STROKE_TOKENS["data"],
            markersize=FAMILY_STYLE["compact_marker_pt"],
            color=style["color"],
            label=MATERIAL_SHORT_LABELS[material],
        )
        zero_cross = float(data["corr_zero_cross_deg"][material])
        if zero_cross <= float(x_focus.max()):
            idx = int(np.argmin(np.abs(x_focus - zero_cross)))
            ax_curve.scatter(
                [zero_cross],
                [curve[idx]],
                s=22,
                marker="o",
                facecolors="white",
                edgecolors=style["color"],
                linewidths=STROKE_TOKENS["base"],
                zorder=4,
            )
    ax_curve.axhline(
        0.0,
        color=GRID_COLOR,
        linestyle="--",
        linewidth=STROKE_TOKENS["annotation"],
        zorder=0,
    )
    ax_curve.axvspan(
        0.0,
        15.0,
        color=STYLE_COLORS["highlight_fill"],
        alpha=0.55,
        zorder=-1,
    )
    ax_curve.text(
        0.04,
        0.08,
        "shaded = nearest-angle regime",
        transform=ax_curve.transAxes,
        ha="left",
        va="bottom",
        fontsize=max(tick_label_pt - 0.45, 5.0),
        color=STYLE_COLORS["muted_text"],
    )
    ax_curve.set_xlim(0.0, float(x_focus.max()))
    ax_curve.set_ylim(-0.42, 1.02)
    ax_curve.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_curve.set_ylabel("Mean centered-|H| corr.", fontsize=axis_label_pt)
    ax_curve.set_xlabel("Angular separation |Δθ| (deg)", fontsize=axis_label_pt)
    ax_curve.tick_params(axis="both", labelsize=tick_label_pt, length=2)
    ax_curve.grid(True, alpha=FAMILY_STYLE["grid_alpha"])
    ax_curve.legend(
        frameon=False,
        fontsize=legend_pt - 0.5,
        loc="upper right",
        ncol=2,
        handlelength=1.4,
        borderpad=0.1,
        labelspacing=0.15,
        columnspacing=0.8,
    )
    return [ax_block, ax_curve]


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
    ax_block = _make_panel_block(
        fig,
        slot_spec,
        label="d",
        title=None,
        title_pt=title_pt,
        add_label=add_label,
    )
    grid = slot_spec.subgridspec(2, 1, height_ratios=[1.00, 1.00], hspace=0.06)
    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    x = np.arange(len(ranking), dtype=float)
    energy = np.asarray([data["normalized_energy"][material] for material in ranking], dtype=float)
    overlap_width = np.asarray([data["corr_zero_cross_deg"][material] for material in ranking], dtype=float)
    overlap_burden = np.asarray(
        [data["mean_top3_overlap_by_material"][material] for material in ranking],
        dtype=float,
    )
    top1 = np.asarray(
        [float(data["performance_by_material"][material]["top1_acc"]) for material in ranking],
        dtype=float,
    )
    top1_lo = np.asarray(
        [float(data["performance_by_material"][material]["top1_ci_low"]) for material in ranking],
        dtype=float,
    )
    top1_hi = np.asarray(
        [float(data["performance_by_material"][material]["top1_ci_high"]) for material in ranking],
        dtype=float,
    )

    top1_color = SEMANTIC_PALETTE["physics"]
    violin = ax_top.violinplot(
        [data["angle_top1_by_material"][material] for material in ranking],
        positions=x,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(top1_color)
        body.set_edgecolor("none")
        body.set_alpha(0.12)

    ax_top.plot(
        x,
        top1,
        color=top1_color,
        linewidth=STROKE_TOKENS["annotation"],
        marker="o",
        markersize=FAMILY_STYLE["standard_marker_pt"],
        zorder=3,
    )
    ax_top.errorbar(
        x,
        top1,
        yerr=[top1 - top1_lo, top1_hi - top1],
        fmt="none",
        capsize=FAMILY_STYLE["compact_capsize_pt"],
        ecolor=top1_color,
        linewidth=STROKE_TOKENS["data"],
        zorder=3,
    )
    for ax in (ax_top, ax_bottom):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(STYLE_COLORS["guide_line"])
        ax.spines["bottom"].set_color(STYLE_COLORS["guide_line"])
        ax.spines["left"].set_linewidth(STROKE_TOKENS["base"])
        ax.spines["bottom"].set_linewidth(STROKE_TOKENS["base"])
        ax.set_xlim(-0.35, len(ranking) - 0.65)
        ax.grid(True, linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])

    ax_top.set_ylim(0.0, 1.02)
    ax_top.set_yticks([0.0, 0.5, 1.0])
    ax_top.set_ylabel("Top-1", fontsize=axis_label_pt, labelpad=1.0)
    ax_top.tick_params(axis="y", labelsize=tick_label_pt, length=2, pad=1)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    marker_sizes = 28.0 + 56.0 * energy
    label_offsets = {
        "b": (-12, 9),
        "w": (-18, -10),
        "a": (8, 12),
        "p": (10, -10),
        "m": (0, 11),
    }
    for idx, material in enumerate(ranking):
        style = _screening_material_style(material)
        ax_bottom.errorbar(
            [overlap_burden[idx]],
            [top1[idx]],
            yerr=[[top1[idx] - top1_lo[idx]], [top1_hi[idx] - top1[idx]]],
            fmt="none",
            ecolor=top1_color,
            linewidth=STROKE_TOKENS["data"],
            capsize=FAMILY_STYLE["compact_capsize_pt"],
            zorder=2,
        )
        ax_bottom.scatter(
            [overlap_burden[idx]],
            [top1[idx]],
            s=float(marker_sizes[idx]),
            marker=style["marker"],
            color=style["color"],
            edgecolors="white",
            linewidths=STROKE_TOKENS["base"],
            zorder=3,
        )
        ax_bottom.annotate(
            MATERIAL_SHORT_LABELS[material],
            (overlap_burden[idx], top1[idx]),
            xytext=label_offsets[material],
            textcoords="offset points",
            ha="center",
            va="bottom" if label_offsets[material][1] >= 0 else "top",
            fontsize=max(tick_label_pt - 0.4, 5.0),
            color=style["color"],
        )
    ax_bottom.set_xlim(float(overlap_burden.min()) - 0.005, float(overlap_burden.max()) + 0.005)
    ax_bottom.set_ylim(0.60, 0.88)
    ax_bottom.set_yticks([0.65, 0.75, 0.85])
    ax_bottom.set_xlabel("Mean top-3 subspace overlap burden", fontsize=axis_label_pt, labelpad=1.0)
    ax_bottom.set_ylabel("Top-1 mean", fontsize=axis_label_pt, labelpad=1.0)
    ax_bottom.tick_params(axis="both", labelsize=tick_label_pt, length=2, pad=1)
    ax_bottom.text(
        0.98,
        0.05,
        "area scales with norm. |H| energy",
        transform=ax_bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(tick_label_pt - 0.35, 5.0),
        color=STYLE_COLORS["muted_text"],
    )
    return [ax_block, ax_top, ax_bottom]


def _plot_panel_e(
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
        label="e",
        title="Selected bands and recovered codes",
        title_pt=title_pt,
        add_label=add_label,
        title_x=0.03,
        title_y=0.94,
    )
    ax_code = fig.add_subplot(slot_spec)
    ax_band = ax_code.twiny()
    short_labels = {
        "b": "Card",
        "w": "Wood",
        "a": "Acry.",
        "p": "Cup",
        "m": "Laptop",
    }
    for idx, material in enumerate(ranking):
        style = _screening_material_style(material)
        lo = data["representative_band_lo_khz"][material]
        hi = data["representative_band_hi_khz"][material]
        center = data["representative_band_freq_hz"][material] / 1000.0
        ax_code.plot(
            data["angles"][material],
            data["representative_directivity"][material],
            color=style["color"],
            linewidth=STROKE_TOKENS["emphasis"],
            marker=style["marker"],
            markersize=FAMILY_STYLE["compact_marker_pt"] + 0.3,
            markevery=6,
        )
        y_band = 1.08 + (len(ranking) - 1 - idx) * 0.055
        ax_band.hlines(
            y_band,
            lo,
            hi,
            color=style["color"],
            alpha=0.35,
            linewidth=max(STROKE_TOKENS["base"] * 1.8, 1.4),
            zorder=2,
        )
        ax_band.plot(
            [center],
            [y_band],
            marker=style["marker"],
            color=style["color"],
            markersize=FAMILY_STYLE["standard_marker_pt"] + 0.3,
            linewidth=0.0,
            zorder=3,
        )
        ax_band.text(
            0.28,
            y_band,
            short_labels[material],
            ha="right",
            va="center",
            fontsize=max(tick_label_pt - 0.25, 5.0),
            color=style["color"],
        )

    ax_code.set_xlim(0.0, 186.0)
    ax_code.set_ylim(0.0, 1.02)
    ax_code.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax_code.set_yticks([0.0, 0.5, 1.0])
    ax_code.set_ylabel("Recovered band code", fontsize=axis_label_pt, labelpad=1.0)
    ax_code.set_xlabel("Angle (deg)", fontsize=axis_label_pt, labelpad=1.0)
    ax_code.spines["top"].set_visible(False)
    ax_code.spines["right"].set_visible(False)
    ax_code.spines["left"].set_color(STYLE_COLORS["guide_line"])
    ax_code.spines["bottom"].set_color(STYLE_COLORS["guide_line"])
    ax_code.spines["left"].set_linewidth(STROKE_TOKENS["base"])
    ax_code.spines["bottom"].set_linewidth(STROKE_TOKENS["base"])
    ax_code.tick_params(axis="both", labelsize=tick_label_pt, length=2, pad=1)
    ax_code.grid(True, linestyle="--", alpha=FAMILY_STYLE["grid_alpha"])

    ax_band.set_xlim(0.3, 3.0)
    ax_band.set_ylim(0.0, 1.34)
    ax_band.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax_band.set_xlabel("Selected contrast band (kHz)", fontsize=axis_label_pt, labelpad=2.1)
    ax_band.tick_params(axis="x", labelsize=tick_label_pt, length=2, pad=1.0)
    ax_band.set_yticks([])
    ax_band.grid(False)
    ax_band.spines["bottom"].set_visible(False)
    ax_band.spines["left"].set_visible(False)
    ax_band.spines["right"].set_visible(False)
    ax_band.spines["top"].set_color(STYLE_COLORS["guide_line"])
    ax_band.spines["top"].set_linewidth(STROKE_TOKENS["base"])
    end_label_x = 182.0
    y_targets: list[tuple[str, float]] = []
    for material in ranking:
        y_tail = float(np.mean(data["representative_directivity"][material][-3:]))
        y_targets.append((material, y_tail))
    y_targets.sort(key=lambda item: item[1])
    adjusted_y: dict[str, float] = {}
    min_gap = 0.08
    prev_y: float | None = None
    for material, y_val in y_targets:
        placed = y_val if prev_y is None else max(y_val, prev_y + min_gap)
        adjusted_y[material] = min(placed, 0.98)
        prev_y = adjusted_y[material]
    for material in ranking:
        style = _screening_material_style(material)
        ax_code.annotate(
            MATERIAL_SHORT_LABELS[material],
            (end_label_x, adjusted_y[material]),
            ha="left",
            va="center",
            fontsize=max(legend_pt - 0.6, 5.0),
            color=style["color"],
            annotation_clip=False,
        )

    return [ax_block, ax_code, ax_band]


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
        4,
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

    _plot_panel_a(
        fig,
        outer[0, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=True,
    )
    _plot_panel_b(
        fig,
        outer[1, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        colorbar_tick_pt=colorbar_tick_pt,
        colorbar_label_pt=colorbar_label_pt,
        add_label=True,
    )
    middle = outer[2, 0].subgridspec(
        1,
        2,
        width_ratios=FIG06_MIDDLE_WIDTH_RATIOS,
        wspace=float(FIG06_MIDDLE_ROW["wspace"]),
    )
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
    _plot_panel_d(
        fig,
        middle[0, 1],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        add_label=True,
    )
    _plot_panel_e(
        fig,
        outer[3, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
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
            "a",
            "a",
            _plot_panel_a,
            "fig06_panel_a_response_regime",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "title_pt": title_pt,
                "add_label": True,
            },
        ),
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
            "d",
            _plot_panel_d,
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
            "e",
            _plot_panel_e,
            "fig06_panel_e_material_frequency_structure",
            {
                "axis_label_pt": axis_label_pt,
                "tick_label_pt": tick_label_pt,
                "legend_pt": legend_pt,
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
                "panel_id": "a",
                "title": "Measured response regime",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_a_response_regime.pdf",
                "provenance_mode": "data_backed",
                "description": "Measured response-descriptor map across the five objects, placing each material by its centered-|H| correlation-decay width and effective rank while retaining object exemplars.",
            },
            {
                "panel_id": "b",
                "title": "Cross-material H",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_b_cross_material_h.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared-normalization H strip for the five compared materials, with the selected informative-band window overlaid on each object to show where directional contrast concentrates.",
            },
            {
                "panel_id": "c",
                "title": "Local-ordering decay",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_c_low_rank_continuity.pdf",
                "provenance_mode": "data_backed",
                "description": "Mean centered-|H| correlation versus angular separation for each object, showing that directional fingerprints remain locally ordered across materials but over object-specific correlation-decay widths.",
            },
            {
                "panel_id": "d",
                "title": "Readout versus overlap burden",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_screening_consequence.pdf",
                "provenance_mode": "data_backed",
                "description": "Per-angle Top-1 distributions together with an object-level Top-1-versus-overlap scatter whose marker area tracks normalized response energy, showing descriptively that vibration amplitude alone does not order the five objects and that the mean top-3 subspace overlap burden provides a more informative comparison axis.",
            },
            {
                "panel_id": "e",
                "title": "Selected contrast band",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_e_material_frequency_structure.pdf",
                "provenance_mode": "data_backed",
                "description": "Per-object informative-band windows together with their band-limited directional codes, showing that one shared contrast-selection rule picks different frequency windows across passive structures and recovers different directional codes from those object-specific windows.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig06] Generated {len(all_paths)} files")
    return all_paths
