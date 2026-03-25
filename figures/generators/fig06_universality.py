"""Figure 6 — Cross-material universality and material screening.

Panel (b): Cross-material H-matrix strip for five everyday objects.
Panel (c): Downstream screening metrics, ranked by selection outcome.
Panel (d): Physical quality versus task accuracy scatter.

Panel (a) remains a manual support asset and is composed downstream.
"""

from __future__ import annotations

import json
from pathlib import Path

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


MATERIAL_ORDER = ("a", "p", "w", "b", "m")
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
PRIMARY_MATERIAL = "b"
BACKUP_MATERIAL = "w"
OTHER_COLOR = "#7A7A7A"
BACKUP_COLOR = SEMANTIC_PALETTE["learned"]
PRIMARY_COLOR = SEMANTIC_PALETTE["physics"]
P95_COLOR = SEMANTIC_PALETTE["ablation"]

FIG06_GENERATOR = figure_section("fig06", "generator")
FIG06_GRID = dict(FIG06_GENERATOR["composite_grid"])
FIG06_SPLIT = dict(FIG06_GENERATOR["split"]["standalone"])


def _load_h_matrix(h_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    payload = torch.load(str(h_path), map_location="cpu", weights_only=False)
    return payload["H"].cpu().numpy(), np.asarray(payload["angles"], dtype=float)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _material_color(code: str) -> str:
    if code == PRIMARY_MATERIAL:
        return PRIMARY_COLOR
    if code == BACKUP_MATERIAL:
        return BACKUP_COLOR
    return OTHER_COLOR


def _save_panel_manifest(
    panel_dir: Path,
    panel_specs: list[dict],
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


def _prepare_fig06_data(data_root: Path) -> dict:
    paths_cfg = load_paths()
    h_run_dir = (data_root / paths_cfg["fig06_cross_material_h_run"]).resolve()
    selection_run_dir = (data_root / paths_cfg["fig06_cross_material_selection_run"]).resolve()

    comparison_report = _load_json(selection_run_dir / "comparison_report.json")
    selection_summary = _load_json(selection_run_dir / "selection_summary.json")

    h_matrices: dict[str, np.ndarray] = {}
    angle_map: dict[str, np.ndarray] = {}
    for material in MATERIAL_ORDER:
        h_path = h_run_dir / f"h_matrix_normalized_original_to_{material}.pth"
        if not h_path.exists():
            raise FileNotFoundError(f"Missing H matrix for material {material}: {h_path}")
        H, angles = _load_h_matrix(h_path)
        h_matrices[material] = H
        angle_map[material] = angles

    n_freq = next(iter(h_matrices.values())).shape[0]
    config = comparison_report["config"]
    freqs_hz = _build_freq_axis(
        n_freq,
        sample_rate=float(config["sample_rate"]),
        n_fft=float(config["n_fft"]),
        freq_min=float(config["freq_min"]),
    )

    return {
        "comparison_report": comparison_report,
        "selection_summary": selection_summary,
        "h_matrices": h_matrices,
        "angles": angle_map,
        "freqs_khz": freqs_hz / 1000.0,
        "h_run_dir": h_run_dir,
        "selection_run_dir": selection_run_dir,
    }


def _plot_panel_b(
    fig: plt.Figure,
    slot_spec,
    data: dict,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    title_pt: float,
    colorbar_tick_pt: float,
    colorbar_label_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    sub = slot_spec.subgridspec(
        1,
        6,
        width_ratios=[1, 1, 1, 1, 1, 0.06],
        wspace=0.10,
    )
    heatmap_axes = [fig.add_subplot(sub[0, idx]) for idx in range(5)]
    cax = fig.add_subplot(sub[0, 5])

    log_mats = []
    for material in MATERIAL_ORDER:
        mat = np.log10(np.clip(np.abs(data["h_matrices"][material]), 1e-8, None))
        log_mats.append(mat)
    stacked = np.concatenate([mat.ravel() for mat in log_mats])
    vmin = float(np.percentile(stacked, 2))
    vmax = float(np.percentile(stacked, 98))

    image = None
    freqs_khz = data["freqs_khz"]
    y_ticks = [0.5, 1.0, 2.0, 3.0]
    for idx, material in enumerate(MATERIAL_ORDER):
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
        ax.set_title(MATERIAL_SHORT_LABELS[material], fontsize=title_pt, pad=2)
        ax.set_xticks([0, 45, 90, 135, 180])
        ax.set_yticks(y_ticks)
        ax.tick_params(axis="both", labelsize=tick_label_pt, length=2)
        if idx == 0:
            ax.set_ylabel("Freq. (kHz)", fontsize=axis_label_pt)
        else:
            ax.set_yticklabels([])
        if add_label and idx == 0:
            add_panel_label(ax, "b", x=-0.22, y=1.05)

    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt, length=2)
    cbar.set_label(r"$\log_{10}|H|$", fontsize=colorbar_label_pt, labelpad=2)

    mid_ax = heatmap_axes[2]
    mid_ax.set_xlabel("Angle (deg)", fontsize=axis_label_pt)
    return [*heatmap_axes, cax]


def _plot_panel_c(
    fig: plt.Figure,
    slot_spec,
    data: dict,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    ranking = list(data["selection_summary"]["ranking"])
    sub = slot_spec.subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.18)
    ax_top = fig.add_subplot(sub[0, 0])
    ax_bottom = fig.add_subplot(sub[1, 0], sharey=ax_top)

    y = np.arange(len(ranking))
    top1 = np.array(
        [data["comparison_report"]["materials"][m]["eval_metrics"]["top1_acc"] for m in ranking]
    )
    within10 = np.array(
        [data["comparison_report"]["materials"][m]["eval_metrics"]["acc_within_10deg"] for m in ranking]
    )
    mae = np.array(
        [data["comparison_report"]["materials"][m]["eval_metrics"]["mae_deg"] for m in ranking]
    )
    p95 = np.array(
        [data["comparison_report"]["materials"][m]["eval_metrics"]["p95_abs_err_deg"] for m in ranking]
    )

    for idx, material in enumerate(ranking):
        tint = _material_color(material)
        if material in {PRIMARY_MATERIAL, BACKUP_MATERIAL}:
            ax_top.axhspan(idx - 0.48, idx + 0.48, color=tint, alpha=0.06, zorder=0)
            ax_bottom.axhspan(idx - 0.48, idx + 0.48, color=tint, alpha=0.06, zorder=0)

    h = 0.34
    ax_top.barh(
        y - h / 2,
        top1,
        height=h,
        color=PRIMARY_COLOR,
        alpha=0.85,
        label="Top-1",
        zorder=3,
    )
    ax_top.barh(
        y + h / 2,
        within10,
        height=h,
        color=BACKUP_COLOR,
        alpha=0.80,
        label="Within 10°",
        zorder=2,
    )
    ax_top.set_xlim(0.0, 1.02)
    ax_top.set_yticks(y)
    ax_top.set_yticklabels([MATERIAL_LABELS[m] for m in ranking], fontsize=tick_label_pt)
    ax_top.invert_yaxis()
    ax_top.set_xlabel("Accuracy", fontsize=axis_label_pt)
    ax_top.set_title("Downstream screening metrics", fontsize=title_pt, loc="left", pad=2)
    ax_top.tick_params(axis="x", labelsize=tick_label_pt)
    ax_top.grid(axis="x", linestyle="--", alpha=0.25)
    ax_top.legend(
        fontsize=legend_pt,
        frameon=False,
        loc="lower right",
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    if add_label:
        add_panel_label(ax_top, "c", x=-0.18, y=1.05)

    for idx, material in enumerate(ranking):
        ax_bottom.hlines(idx, 0.0, p95[idx], color="#C8C8C8", linewidth=2.2, zorder=1)
        ax_bottom.scatter(mae[idx], idx, s=26, color=PRIMARY_COLOR, zorder=3, label="MAE" if idx == 0 else None)
        ax_bottom.scatter(
            p95[idx],
            idx,
            s=28,
            marker="s",
            color=P95_COLOR,
            zorder=4,
            label="P95 abs. error" if idx == 0 else None,
        )
    ax_bottom.set_xlim(0.0, 40.0)
    ax_bottom.set_xlabel("Error (deg)", fontsize=axis_label_pt)
    ax_bottom.set_yticks(y)
    ax_bottom.tick_params(axis="x", labelsize=tick_label_pt)
    ax_bottom.tick_params(axis="y", left=False, labelleft=False)
    ax_bottom.grid(axis="x", linestyle="--", alpha=0.25)
    ax_bottom.legend(fontsize=legend_pt, frameon=False, loc="upper right")

    return [ax_top, ax_bottom]


def _plot_panel_d(
    ax: plt.Axes,
    data: dict,
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    annotation_pt: float,
    title_pt: float,
    add_label: bool,
) -> list[plt.Axes]:
    comparison = data["comparison_report"]["materials"]
    ranking = list(data["selection_summary"]["ranking"])

    coherence = np.array([comparison[m]["h_metrics"]["mean_coherence"] for m in ranking])
    top1 = np.array([comparison[m]["eval_metrics"]["top1_acc"] for m in ranking])

    for material, x_val, y_val in zip(ranking, coherence, top1):
        ax.scatter(
            x_val,
            y_val,
            s=54,
            color=_material_color(material),
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

    label_offsets = {
        "b": (0.00025, 0.010),
        "w": (0.00025, -0.012),
        "a": (0.00025, 0.006),
        "p": (-0.0025, -0.010),
        "m": (0.00025, -0.014),
    }
    labeled_points = {"a", "w", "m"}
    for material, x_val, y_val in zip(ranking, coherence, top1):
        if material not in labeled_points:
            continue
        dx, dy = label_offsets[material]
        ax.text(
            x_val + dx,
            y_val + dy,
            MATERIAL_SHORT_LABELS[material],
            fontsize=annotation_pt,
            color=_material_color(material),
            ha="left" if dx >= 0 else "right",
            va="center",
        )

    paper_x = comparison["p"]["h_metrics"]["mean_coherence"]
    paper_y = comparison["p"]["eval_metrics"]["top1_acc"]
    box_x = comparison["b"]["h_metrics"]["mean_coherence"]
    box_y = comparison["b"]["eval_metrics"]["top1_acc"]

    ax.annotate(
        "Highest coherence",
        xy=(paper_x, paper_y),
        xytext=(paper_x - 0.0048, paper_y + 0.020),
        fontsize=annotation_pt,
        color=OTHER_COLOR,
        arrowprops={"arrowstyle": "-", "color": OTHER_COLOR, "lw": 0.7},
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "Best task",
        xy=(box_x, box_y),
        xytext=(box_x - 0.0048, box_y + 0.018),
        fontsize=annotation_pt,
        color=PRIMARY_COLOR,
        arrowprops={"arrowstyle": "-", "color": PRIMARY_COLOR, "lw": 0.7},
        ha="left",
        va="bottom",
    )

    ax.set_xlabel("Mean coherence of $H$", fontsize=axis_label_pt)
    ax.set_ylabel("Top-1 accuracy", fontsize=axis_label_pt)
    ax.set_title("Physics versus task accuracy", fontsize=title_pt, loc="left", pad=2)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_xlim(0.018, 0.0355)
    ax.set_ylim(0.64, 0.84)
    if add_label:
        add_panel_label(ax, "d", x=-0.18, y=1.05)
    return [ax]


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate the Fig. 6 data-backed panels and generator composite."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    annotation_pt = typography["annotation"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    data = _prepare_fig06_data(data_root)

    fig = make_figure(
        width_mm=FIG06_GENERATOR["composite_width_mm"],
        height_mm=FIG06_GENERATOR["composite_height_mm"],
    )
    outer = gridspec.GridSpec(
        2,
        2,
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
        outer[0, :],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        colorbar_tick_pt=colorbar_tick_pt,
        colorbar_label_pt=colorbar_label_pt,
        add_label=True,
    )
    _plot_panel_c(
        fig,
        outer[1, 0],
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=True,
    )
    ax_d = fig.add_subplot(outer[1, 1])
    _plot_panel_d(
        ax_d,
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        annotation_pt=annotation_pt,
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

    # Standalone panel b
    fig_b = make_figure(
        width_mm=FIG06_SPLIT["b"]["width_mm"],
        height_mm=FIG06_SPLIT["b"]["height_mm"],
    )
    slot_b = gridspec.GridSpec(1, 1, figure=fig_b)[0, 0]
    _plot_panel_b(
        fig_b,
        slot_b,
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        title_pt=title_pt,
        colorbar_tick_pt=colorbar_tick_pt,
        colorbar_label_pt=colorbar_label_pt,
        add_label=True,
    )
    fig_b.subplots_adjust(**FIG06_SPLIT["b"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_b,
            panel_dir / "fig06_panel_b_h_matrices",
            typography=typography,
        )
    )
    plt.close(fig_b)

    # Standalone panel c
    fig_c = make_figure(
        width_mm=FIG06_SPLIT["c"]["width_mm"],
        height_mm=FIG06_SPLIT["c"]["height_mm"],
    )
    slot_c = gridspec.GridSpec(1, 1, figure=fig_c)[0, 0]
    _plot_panel_c(
        fig_c,
        slot_c,
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title_pt=title_pt,
        add_label=True,
    )
    fig_c.subplots_adjust(**FIG06_SPLIT["c"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_c,
            panel_dir / "fig06_panel_c_material_screening",
            typography=typography,
        )
    )
    plt.close(fig_c)

    # Standalone panel d
    fig_d = make_figure(
        width_mm=FIG06_SPLIT["d"]["width_mm"],
        height_mm=FIG06_SPLIT["d"]["height_mm"],
    )
    ax_d_standalone = fig_d.add_subplot(111)
    _plot_panel_d(
        ax_d_standalone,
        data,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        annotation_pt=annotation_pt,
        title_pt=title_pt,
        add_label=True,
    )
    fig_d.subplots_adjust(**FIG06_SPLIT["d"]["subplots_adjust"])
    all_paths.extend(
        save_outputs(
            fig_d,
            panel_dir / "fig06_panel_d_physics_vs_task",
            typography=typography,
        )
    )
    plt.close(fig_d)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "b",
                "title": "Cross-material H matrices",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_b_h_matrices.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared-normalization strip showing structured angle-frequency transfer functions for five everyday objects.",
            },
            {
                "panel_id": "c",
                "title": "Material screening metrics",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_c_material_screening.pdf",
                "provenance_mode": "data_backed",
                "description": "Ranked downstream screening panel comparing accuracy and error metrics across materials.",
            },
            {
                "panel_id": "d",
                "title": "Physical quality versus task accuracy",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_physics_vs_task.pdf",
                "provenance_mode": "data_backed",
                "description": "Scatter linking mean H coherence to downstream top-1 accuracy, showing that physical quality alone does not pick the best material.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig06] Generated {len(all_paths)} files")
    return all_paths
