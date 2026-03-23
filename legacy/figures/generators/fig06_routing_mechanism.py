"""Figure 6 — Micro-level Routing Mechanism.

Two-panel master figure:
  (a) 2x2 freq x atom heatmaps (Physics true/wrong, QK true/wrong)
  (b) 5 band line plots: full band + 4 sub-band decompositions (OMP vs QK vs True DoA)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    load_paths,
    DOUBLE_COL_MM,
)

BAND_CONFIGS = (
    {"min_hz": 300, "max_hz": 3000, "label": "300-3000 Hz (Full Band)", "include_upper": True},
    {"min_hz": 300, "max_hz": 500, "label": "300-500 Hz", "include_upper": False},
    {"min_hz": 500, "max_hz": 1000, "label": "500-1000 Hz", "include_upper": False},
    {"min_hz": 1000, "max_hz": 2000, "label": "1000-2000 Hz", "include_upper": False},
    {"min_hz": 2000, "max_hz": 3000, "label": "2000-3000 Hz", "include_upper": True},
)


def _load_data(run_dir: Path) -> tuple[dict, dict]:
    routing_data = dict(np.load(run_dir / "modal_routing_val.npz", allow_pickle=True))
    dict_data = dict(np.load(run_dir / "dictionary.npz", allow_pickle=True))
    return routing_data, dict_data


def _find_representative_case(routing_data: dict, dict_data: dict) -> tuple[int, int, int]:
    scores_expert = routing_data["scores_expert"]
    g_energy = routing_data["g_energy_expert"]
    labels = routing_data["labels"]
    angles = dict_data["angles"]

    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)

    angle_145_idx = int(np.argmin(np.abs(angles - 145)))
    angle_60_idx = int(np.argmin(np.abs(angles - 60)))

    for idx in range(len(labels)):
        if labels[idx] == angle_145_idx and g_pred[idx] == angle_60_idx and qk_pred[idx] == angle_145_idx:
            return idx, angle_145_idx, angle_60_idx

    qk_fix = (qk_pred == labels) & (g_pred != labels)
    if qk_fix.sum() > 0:
        idx = int(np.where(qk_fix)[0][0])
        return idx, int(labels[idx]), int(g_pred[idx])

    return 0, int(labels[0]), int(np.argmax(g_energy[0]))


def _band_mask(freqs: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["include_upper"]:
        return (freqs >= cfg["min_hz"]) & (freqs <= cfg["max_hz"])
    return (freqs >= cfg["min_hz"]) & (freqs < cfg["max_hz"])


def _compute_band_scores(Y, D, scores_atoms, freqs, cfg):
    mask = _band_mask(freqs, cfg)
    D_band = D[mask, :]
    Y_band = Y[mask]
    atom_proj = D_band.T @ Y_band
    physics_scores = np.abs(atom_proj).reshape(37, 8).sum(axis=1)
    qk_scores = np.abs((atom_proj.reshape(37, 8) * scores_atoms).sum(axis=1))
    return physics_scores, qk_scores


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 6. Returns list of output file paths."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    if not (run_dir / "modal_routing_val.npz").exists():
        print(f"[fig06] SKIP: routing data not found at {run_dir}")
        return []

    routing_data, dict_data = _load_data(run_dir)
    angles = dict_data["angles"]
    D = dict_data["D"]

    sample_idx, true_expert, wrong_expert = _find_representative_case(routing_data, dict_data)
    Y = routing_data["Y_val"][sample_idx]
    scores_atoms_sample = routing_data["scores_atoms"][sample_idx]
    g_energy_atoms = np.abs(D.T @ Y).reshape(37, 8)

    D_true = D[:, true_expert * 8:(true_expert + 1) * 8]
    D_wrong = D[:, wrong_expert * 8:(wrong_expert + 1) * 8]
    g_true_atoms = g_energy_atoms[true_expert, :]
    g_wrong_atoms = g_energy_atoms[wrong_expert, :]

    freqs_all = np.linspace(300, 3000, D_true.shape[0])
    freq_mask = freqs_all <= 2000

    physics_true_hm = np.abs(D_true[freq_mask, :]) * g_true_atoms[np.newaxis, :]
    physics_wrong_hm = np.abs(D_wrong[freq_mask, :]) * g_wrong_atoms[np.newaxis, :]

    qk_true_atoms = scores_atoms_sample[true_expert, :]
    qk_wrong_atoms = scores_atoms_sample[wrong_expert, :]
    qk_true_hm = D_true[freq_mask, :] * qk_true_atoms[np.newaxis, :]
    qk_wrong_hm = D_wrong[freq_mask, :] * qk_wrong_atoms[np.newaxis, :]

    # Build figure
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=165)
    gs_top = gridspec.GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[0.95, 1.05],
        hspace=0.3,
        left=0.08,
        right=0.96,
        bottom=0.07,
        top=0.94,
    )

    # Panel (a): 2x2 heatmaps
    gs_a = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_top[0], hspace=0.35, wspace=0.30)
    y_extent = [0, 8, 0, 2000]
    y_ticks = [0, 500, 1000, 1500, 2000]

    vmin_phys = min(physics_true_hm.min(), physics_wrong_hm.min())
    vmax_phys = max(physics_true_hm.max(), physics_wrong_hm.max())
    vmax_qk = max(np.abs(qk_true_hm).max(), np.abs(qk_wrong_hm).max())

    heatmap_configs = [
        (gs_a[0, 0], physics_true_hm, "viridis", vmin_phys, vmax_phys, f"OMP: True ({angles[true_expert]:.0f}\u00b0)", "Energy"),
        (gs_a[0, 1], physics_wrong_hm, "viridis", vmin_phys, vmax_phys, f"OMP: Wrong ({angles[wrong_expert]:.0f}\u00b0)", "Energy"),
        (gs_a[1, 0], qk_true_hm, "RdBu_r", -vmax_qk, vmax_qk, f"QK: True ({angles[true_expert]:.0f}\u00b0)", "QK Score"),
        (gs_a[1, 1], qk_wrong_hm, "RdBu_r", -vmax_qk, vmax_qk, f"QK: Wrong ({angles[wrong_expert]:.0f}\u00b0)", "QK Score"),
    ]

    for gs_pos, data, cmap, vmin, vmax, title, cbar_label in heatmap_configs:
        ax = fig.add_subplot(gs_pos)
        im = ax.imshow(data, aspect="auto", cmap=cmap, extent=y_extent, interpolation="nearest", origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=6, fontweight="bold")
        ax.set_ylabel("Freq (Hz)", fontsize=5); ax.set_xlabel("Atom", fontsize=5)
        ax.set_yticks(y_ticks); ax.tick_params(labelsize=5)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=5); cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_linewidth(0.5)

    fig.text(0.06, 0.94, "a", fontsize=10, fontweight="bold", va="top")

    # Panel (b): 5 band line plots
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs_top[1], wspace=0.35)
    freqs = np.linspace(300, 3000, D.shape[0])
    tick_positions = [0, 9, 18, 27, 36]
    tick_vals = [angles[i] for i in tick_positions if i < len(angles)]
    tick_strs = [f"{int(v)}" for v in tick_vals]

    for col, band_cfg in enumerate(BAND_CONFIGS):
        physics_scores, qk_scores = _compute_band_scores(Y, D, scores_atoms_sample, freqs, band_cfg)
        ax = fig.add_subplot(gs_b[0, col])
        y_top = max(physics_scores.max(), qk_scores.max())
        if y_top <= 0:
            y_top = 1.0

        ax.grid(True, alpha=0.25, linestyle="--", color="gray", linewidth=0.5); ax.set_axisbelow(True)
        ax.plot(angles, physics_scores, "o-", color="coral", linewidth=0.9, markersize=1.5, alpha=0.6, label="OMP")
        ax.plot(angles, qk_scores, "-", color="darkgreen", linewidth=1.2, alpha=0.9, label="QK")

        peak_idx = np.argmax(qk_scores)
        ax.scatter([angles[peak_idx]], [qk_scores[peak_idx]], color="darkgreen", s=12, zorder=3)

        true_angle = angles[true_expert]
        ax.axvline(true_angle, color="lime", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.scatter([true_angle], [y_top * 1.05], marker="*", color="lime", s=30, edgecolors="black", linewidths=0.5, zorder=4)

        ax.set_title(band_cfg["label"], fontsize=6, fontweight="bold")
        ax.set_xlabel("Angle", fontsize=5)
        if col == 0:
            ax.set_ylabel("Score", fontsize=5)
        ax.set_xlim(angles[0], angles[-1]); ax.set_ylim(0, y_top * 1.15)
        ax.set_xticks(tick_vals); ax.set_xticklabels(tick_strs, fontsize=5, rotation=45)
        ax.tick_params(axis="y", labelsize=5)
        if col == 0:
            ax.legend(frameon=False, loc="upper right", fontsize=6, handlelength=1.0)

    fig.text(0.06, 0.47, "b", fontsize=10, fontweight="bold", va="top")

    all_paths = save_outputs(fig, output_dir / "fig06_routing_mechanism")
    plt.close(fig)

    print(f"[fig06] Generated {len(all_paths)} files (sample={sample_idx})")
    return all_paths
