"""Figure 6 — Cross-material Universality: Physics Evidence (new panels d, e).

Panels (a)-(c) are external JPG assets (photos + heatmaps + RMSE).
This generator produces the NEW data panels:
  (d) SVD spectra overlay — singular-value decay from the primary H matrix
      across frequency sub-bands, demonstrating consistent low-rank structure.
      (When per-material H matrices become available, overlay all 5 materials.)
  (e) Band-resolved routing consistency — 5-band summary showing routing
      maintains diagonal structure across frequency bands.

Data: h_matrix (SVD), band decomposition (fig5_b3), dictionary.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    DOUBLE_COL_MM,
    SEMANTIC_PALETTE,
)
from figures.naming import get_bound_label
from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)


# ---------------------------------------------------------------------------
# Band definitions (matching fig06_routing_mechanism.py)
# ---------------------------------------------------------------------------
BAND_CONFIGS = (
    {"min_hz": 300, "max_hz": 3000, "label": "Full\n300-3k", "include_upper": True},
    {"min_hz": 300, "max_hz": 500,  "label": "300-500",      "include_upper": False},
    {"min_hz": 500, "max_hz": 1000, "label": "500-1k",       "include_upper": False},
    {"min_hz": 1000, "max_hz": 2000, "label": "1k-2k",       "include_upper": False},
    {"min_hz": 2000, "max_hz": 3000, "label": "2k-3k",       "include_upper": True},
)

FIG06_GENERATOR = figure_section("fig06", "generator")
FIG06_SPLIT = dict(FIG06_GENERATOR["split"])
FIG06_GRID = dict(FIG06_GENERATOR["composite_grid"])
FIG06_STANDALONE = dict(FIG06_SPLIT["standalone_subplots"])
FIG06_PANEL_SLOT_WIDTH_MM = float(FIG06_SPLIT["panel_slot_width_mm"])
FIG06_PANEL_SLOT_HEIGHT_MM = float(FIG06_SPLIT["panel_slot_height_mm"])


def _load_h_matrix(h_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch
    h_data = torch.load(str(h_path), map_location="cpu", weights_only=False)
    return h_data["H"].cpu().numpy(), np.array(h_data["angles"], dtype=float)


def _build_freq_axis(F: int, fs: float = 16000.0, n_fft: float = 2048.0,
                     f_min: float = 300.0) -> np.ndarray:
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    return (k_start + np.arange(F)) * df


def _band_mask(freqs: np.ndarray, cfg: dict) -> np.ndarray:
    if cfg["include_upper"]:
        return (freqs >= cfg["min_hz"]) & (freqs <= cfg["max_hz"])
    return (freqs >= cfg["min_hz"]) & (freqs < cfg["max_hz"])


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
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 6 universality panels (d, e)."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    title_pt = typography["title"]
    annotation_pt = typography["annotation"]

    paths_cfg = load_paths()
    h_path = data_root / paths_cfg["h_matrix"]
    run_dir = data_root / paths_cfg["primary_run"]

    if not h_path.exists():
        print(f"[fig06] SKIP: H matrix not found at {h_path}")
        return []

    H_np, angles_deg = _load_h_matrix(h_path)
    F, E = H_np.shape
    freqs = _build_freq_axis(F)

    # -----------------------------------------------------------------------
    # Build composite figure (panels d and e side by side)
    # -----------------------------------------------------------------------
    fig = make_figure(
        width_mm=FIG06_GENERATOR["composite_width_mm"],
        height_mm=FIG06_GENERATOR["composite_height_mm"],
    )
    width_ratios = FIG06_GRID["width_ratios"]
    gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=width_ratios,
        wspace=FIG06_GRID["wspace"],
        left=FIG06_GRID["left"],
        right=FIG06_GRID["right"],
        bottom=FIG06_GRID["bottom"],
        top=FIG06_GRID["top"],
    )

    # --- Panel (d): SVD spectra per frequency band ---
    ax_d = fig.add_subplot(gs[0, 0])

    band_colors = ["#000000", "#0072B2", "#009E73", "#D55E00", "#CC79A7"]
    eps = 1e-8

    for band_idx, (band_cfg, color) in enumerate(zip(BAND_CONFIGS, band_colors)):
        mask = _band_mask(freqs, band_cfg)
        H_band = np.abs(H_np[mask, :])

        if H_band.shape[0] < 2:
            continue

        H_log_band = np.log(np.clip(H_band, eps, None))
        H_centered = H_log_band - H_log_band.mean(axis=1, keepdims=True)
        _, S_band, _ = np.linalg.svd(H_centered, full_matrices=False)

        # Normalize singular values
        S_norm = S_band / (S_band[0] + 1e-12)
        n_show = min(15, len(S_norm))
        r_idx = np.arange(1, n_show + 1)

        ax_d.semilogy(r_idx, S_norm[:n_show], "o-", color=color,
                      markersize=2, linewidth=0.8, alpha=0.8,
                      label=band_cfg["label"].replace("\n", " "))

    ax_d.set_xlabel("Mode index r", fontsize=axis_label_pt)
    ax_d.set_ylabel(r"$\sigma_r / \sigma_1$ (normalized)", fontsize=axis_label_pt)
    ax_d.set_title("SVD decay per band", fontsize=title_pt)
    ax_d.legend(fontsize=legend_pt, frameon=False, loc="upper right", ncol=2)
    ax_d.tick_params(axis="both", labelsize=tick_label_pt)
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_d, "d", x=-0.15, y=1.06)

    # --- Panel (e): Band-resolved routing consistency ---
    # For each band, compute the diagonal concentration of the selection
    # probability matrix (how much mass is on the correct expert)
    ax_e = fig.add_subplot(gs[0, 1])

    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"

    if routing_path.exists() and dict_path.exists():
        routing_data = dict(np.load(routing_path, allow_pickle=True))
        dict_data = dict(np.load(dict_path, allow_pickle=True))
        omp_label = get_bound_label("fig06", "e", "omp_bandwise", label_type="short")
        learned_label = get_bound_label(
            "fig06", "e", "learned_bandwise", label_type="short"
        )
        D = dict_data["D"]
        labels = routing_data["labels"]
        Y_val = routing_data["Y_val"]
        scores_atoms = routing_data["scores_atoms"]
        n_angles = 37

        band_labels = []
        omp_diag = []
        qk_diag = []

        for band_cfg in BAND_CONFIGS:
            mask = _band_mask(freqs, band_cfg)
            D_band = D[mask, :]

            # Compute per-angle selection for this band
            omp_correct = 0
            qk_correct = 0
            total = 0

            for true_angle in range(n_angles):
                angle_mask = labels == true_angle
                if angle_mask.sum() == 0:
                    continue

                for trial_idx in np.where(angle_mask)[0]:
                    Y_band = Y_val[trial_idx, mask]
                    atom_proj = D_band.T @ Y_band
                    physics_scores = np.abs(atom_proj).reshape(n_angles, -1).sum(axis=1)

                    # QK-weighted scores
                    qk_atoms = scores_atoms[trial_idx]
                    qk_weighted = np.abs(
                        (atom_proj.reshape(n_angles, -1) *
                         qk_atoms[:, :atom_proj.reshape(n_angles, -1).shape[1]]).sum(axis=1)
                    )

                    if np.argmax(physics_scores) == true_angle:
                        omp_correct += 1
                    if np.argmax(qk_weighted) == true_angle:
                        qk_correct += 1
                    total += 1

            band_labels.append(band_cfg["label"].replace("\n", " "))
            omp_diag.append(omp_correct / total if total > 0 else 0)
            qk_diag.append(qk_correct / total if total > 0 else 0)

        x = np.arange(len(band_labels))
        width = 0.35
        ax_e.bar(x - width / 2, omp_diag, width, color=SEMANTIC_PALETTE["ablation"],
                 alpha=0.7, label=omp_label)
        ax_e.bar(x + width / 2, qk_diag, width, color=SEMANTIC_PALETTE["learned"],
                 alpha=0.7, label=learned_label)

        ax_e.set_xticks(x)
        ax_e.set_xticklabels(band_labels, fontsize=tick_label_pt, rotation=30, ha="right")
        ax_e.set_ylabel("Diagonal accuracy", fontsize=axis_label_pt)
        ax_e.set_title("Band-resolved routing", fontsize=title_pt)
        ax_e.set_ylim(0, 1.05)
        ax_e.legend(fontsize=legend_pt, frameon=False, loc="upper right")
        ax_e.tick_params(axis="y", labelsize=tick_label_pt)
        ax_e.grid(axis="y", linestyle="--", alpha=0.3)
    else:
        ax_e.text(0.5, 0.5, "Routing data\nunavailable",
                  transform=ax_e.transAxes, ha="center", va="center", fontsize=annotation_pt)
        ax_e.set_axis_off()

    add_panel_label(ax_e, "e", x=-0.12, y=1.06)

    # Save composite
    all_paths = save_outputs(
        fig,
        output_dir / "fig06_universality",
        typography=typography,
    )
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig06_universality_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel d standalone
    fig_d = make_figure(
        width_mm=FIG06_PANEL_SLOT_WIDTH_MM,
        height_mm=FIG06_PANEL_SLOT_HEIGHT_MM,
    )
    ax = fig_d.add_subplot(111)
    for band_idx, (band_cfg, color) in enumerate(zip(BAND_CONFIGS, band_colors)):
        mask = _band_mask(freqs, band_cfg)
        H_band = np.abs(H_np[mask, :])
        if H_band.shape[0] < 2:
            continue
        H_log_band = np.log(np.clip(H_band, eps, None))
        H_centered = H_log_band - H_log_band.mean(axis=1, keepdims=True)
        _, S_band, _ = np.linalg.svd(H_centered, full_matrices=False)
        S_norm = S_band / (S_band[0] + 1e-12)
        n_show = min(15, len(S_norm))
        ax.semilogy(np.arange(1, n_show + 1), S_norm[:n_show], "o-",
                    color=color, markersize=3, linewidth=0.9,
                    label=band_cfg["label"].replace("\n", " "))
    ax.set_xlabel("Mode index r", fontsize=axis_label_pt)
    ax.set_ylabel(r"$\sigma_r / \sigma_1$", fontsize=axis_label_pt)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.legend(fontsize=legend_pt, frameon=False, ncol=2, loc="upper right")
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(**FIG06_STANDALONE["d"])
    all_paths.extend(
        save_outputs(
            fig_d,
            panel_dir / "fig06_panel_d_svd",
            typography=typography,
        )
    )
    plt.close(fig_d)

    # Panel e standalone
    if routing_path.exists() and dict_path.exists():
        fig_e = make_figure(
            width_mm=FIG06_PANEL_SLOT_WIDTH_MM,
            height_mm=FIG06_PANEL_SLOT_HEIGHT_MM,
        )
        ax = fig_e.add_subplot(111)
        x = np.arange(len(band_labels))
        ax.bar(x - width / 2, omp_diag, width, color=SEMANTIC_PALETTE["ablation"],
               alpha=0.7, label=omp_label)
        ax.bar(x + width / 2, qk_diag, width, color=SEMANTIC_PALETTE["learned"],
               alpha=0.7, label=learned_label)
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, fontsize=tick_label_pt, rotation=30, ha="right")
        ax.set_ylabel("Diagonal accuracy", fontsize=axis_label_pt)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="y", labelsize=tick_label_pt)
        ax.legend(fontsize=legend_pt, frameon=False, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        add_panel_label(ax, "e")
        fig_e.subplots_adjust(**FIG06_STANDALONE["e"])
        all_paths.extend(
            save_outputs(
                fig_e,
                panel_dir / "fig06_panel_e_band_routing",
                typography=typography,
            )
        )
        plt.close(fig_e)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "d",
                "title": "Per-band SVD spectra",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_d_svd.pdf",
                "provenance_mode": "data_backed",
                "description": "Singular-value decay across 5 frequency bands showing consistent low-rank structure.",
            },
            {
                "panel_id": "e",
                "title": "Band-resolved routing consistency",
                "asset_path": "figures/output/fig06_universality_panels/fig06_panel_e_band_routing.pdf",
                "provenance_mode": "data_backed",
                "description": "Diagonal accuracy per band comparing the analytical OMP baseline versus the full physics-aware solver.",
            },
        ],
        typography=typography,
    )
    all_paths.append(manifest)

    print(f"[fig06] Generated {len(all_paths)} files")
    return all_paths
