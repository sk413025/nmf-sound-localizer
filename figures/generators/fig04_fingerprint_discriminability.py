"""Figure 4 — Fingerprint Discriminability.

Panel (a): Angle–frequency heatmap of the physical dictionary H.
Panel (b): Within-angle vs between-angle Pearson r (violin/box) + statistical
           annotation (Mann–Whitney U, Cohen's d).
Panel (c): Per-angle fingerprint repeatability (line + ±SEM band).

Data: dictionary.npz (H, angles) + modal_routing_val.npz (Y_val, labels).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    DOUBLE_COL_MM,
    SEMANTIC_PALETTE,
)


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def _build_freq_axis(F: int, fs: float = 16000.0, n_fft: float = 2048.0,
                     f_min: float = 300.0) -> np.ndarray:
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    return (k_start + np.arange(F)) * df


def _compute_pairwise_corr(Y_val: np.ndarray, labels: np.ndarray,
                           n_angles: int) -> tuple[list[float], list[float],
                                                    np.ndarray, np.ndarray,
                                                    np.ndarray]:
    """Compute within-angle and between-angle Pearson r distributions.

    Returns (within_all, between_all, per_angle_mean, per_angle_std,
             per_angle_n_pairs).
    """
    within_all: list[float] = []
    between_all: list[float] = []
    per_angle_mean = np.full(n_angles, np.nan)
    per_angle_std = np.full(n_angles, np.nan)
    per_angle_n_pairs = np.zeros(n_angles, dtype=int)

    # Pre-compute per-angle indices
    angle_indices: dict[int, np.ndarray] = {}
    for a in range(n_angles):
        idx = np.where(labels == a)[0]
        if len(idx) > 1:
            angle_indices[a] = idx

    # Within-angle correlations
    for a, idx in angle_indices.items():
        clips = Y_val[idx]
        corr_mat = np.corrcoef(clips)
        n = len(idx)
        triu_idx = np.triu_indices(n, k=1)
        within_r = corr_mat[triu_idx].tolist()
        within_all.extend(within_r)
        per_angle_mean[a] = np.mean(within_r)
        per_angle_std[a] = np.std(within_r)
        per_angle_n_pairs[a] = len(within_r)

    # Between-angle correlations (subsample for speed)
    rng = np.random.default_rng(42)
    angle_keys = sorted(angle_indices.keys())
    max_between = 5000
    count = 0
    for i, a1 in enumerate(angle_keys):
        for a2 in angle_keys[i + 1:]:
            idx1 = angle_indices[a1]
            idx2 = angle_indices[a2]
            n_pairs = min(3, len(idx1), len(idx2))
            sel1 = rng.choice(idx1, size=n_pairs, replace=False)
            sel2 = rng.choice(idx2, size=n_pairs, replace=False)
            for s1, s2 in zip(sel1, sel2):
                r = np.corrcoef(Y_val[s1], Y_val[s2])[0, 1]
                between_all.append(r)
                count += 1
                if count >= max_between:
                    break
            if count >= max_between:
                break
        if count >= max_between:
            break

    return within_all, between_all, per_angle_mean, per_angle_std, per_angle_n_pairs


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d (pooled std)."""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                          (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def _p_to_stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig04_panel_manifest.json"
    payload = {
        "figure_id": "fig04",
        "composite_asset": "figures/output/fig04_fingerprint_discriminability.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 4 — Fingerprint Discriminability."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"

    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig04] SKIP: data not found at {run_dir}")
        return []

    # Load data
    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    Y_val = routing_data["Y_val"]
    labels = routing_data["labels"].astype(int)
    H = dict_data["H"]
    angles = dict_data["angles"]
    n_angles = len(angles)

    F = H.shape[0]
    freqs = _build_freq_axis(F)

    # Compute similarity distributions
    within_all, between_all, per_angle_mean, per_angle_std, per_angle_n = \
        _compute_pairwise_corr(Y_val, labels, n_angles)

    # Statistical test for panel (b)
    within_arr = np.array(within_all)
    between_arr = np.array(between_all)
    U_stat, p_value = sp_stats.mannwhitneyu(
        within_arr, between_arr, alternative="greater"
    )
    d_value = _cohens_d(within_arr, between_arr)

    # Per-angle SEM for panel (c)
    per_angle_sem = np.where(
        per_angle_n > 0,
        per_angle_std / np.sqrt(per_angle_n),
        0.0,
    )

    # -----------------------------------------------------------------------
    # Build composite figure  (P0-2: increased wspace 0.38 → 0.50)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=100)
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.2, 0.8, 1.0],
        wspace=0.50, left=0.06, right=0.97, bottom=0.15, top=0.88,
    )

    # --- Panel (a): Angle–frequency heatmap ---
    ax_a = fig.add_subplot(gs[0, 0])
    im = ax_a.imshow(
        H.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles[0], angles[-1]],
    )
    ax_a.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_a.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_a.set_title("Direction-dependent fingerprints", fontsize=6.5)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude", fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_a, "a", x=-0.12, y=1.06)

    # --- Panel (b): Within vs between violin + stats (P1-4) ---
    ax_b = fig.add_subplot(gs[0, 1])
    data_violin = [within_all, between_all]
    parts = ax_b.violinplot(data_violin, positions=[0, 1], showmeans=True,
                            showmedians=False, showextrema=False)
    colors = [SEMANTIC_PALETTE["physics"], SEMANTIC_PALETTE["ablation"]]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.0)

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Within\nangle", "Between\nangles"], fontsize=5.5)
    ax_b.set_ylabel("Pearson r", fontsize=6)
    ax_b.set_title("Fingerprint similarity", fontsize=6.5)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_b, "b", x=-0.25, y=1.06)

    # Statistical annotation bracket
    y_max = max(np.max(within_all), np.max(between_all))
    bracket_y = y_max + 0.02
    bar_y = bracket_y + 0.01
    ax_b.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y],
              lw=0.8, c="black")
    stars = _p_to_stars(p_value)
    ax_b.text(0.5, bar_y + 0.005, f"{stars}\nd = {d_value:.2f}",
              ha="center", va="bottom", fontsize=5, linespacing=1.2)

    # --- Panel (c): Per-angle repeatability with ±SEM (P1-5) ---
    ax_c = fig.add_subplot(gs[0, 2])
    valid = ~np.isnan(per_angle_mean)
    ax_c.plot(angles[valid], per_angle_mean[valid], "-o", markersize=2,
              linewidth=0.9, color=SEMANTIC_PALETTE["physics"])
    ax_c.fill_between(
        angles[valid],
        per_angle_mean[valid] - per_angle_sem[valid],
        per_angle_mean[valid] + per_angle_sem[valid],
        alpha=0.25, color=SEMANTIC_PALETTE["physics"],
        label="\u00b1 1 SEM",
    )
    ax_c.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_c.set_ylabel("Mean within-angle r", fontsize=6)
    ax_c.set_title("Fingerprint repeatability", fontsize=6.5)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)
    ax_c.legend(fontsize=4.5, frameon=False, loc="lower right")
    add_panel_label(ax_c, "c", x=-0.12, y=1.06)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig04_fingerprint_discriminability")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig04_fingerprint_discriminability_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel a standalone
    fig_a = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_a.add_subplot(111)
    im = ax.imshow(
        H.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles[0], angles[-1]],
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Amplitude", fontsize=5)
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig04_panel_a_heatmap"))
    plt.close(fig_a)

    # Panel b standalone
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_b.add_subplot(111)
    parts = ax.violinplot(data_violin, positions=[0, 1], showmeans=True,
                          showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within angle", "Between angles"])
    ax.set_ylabel("Pearson r")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    # Stats annotation on standalone
    ax.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y],
            lw=0.8, c="black")
    ax.text(0.5, bar_y + 0.005, f"{stars}\nd = {d_value:.2f}",
            ha="center", va="bottom", fontsize=5.5, linespacing=1.2)
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig04_panel_b_similarity"))
    plt.close(fig_b)

    # Panel c standalone
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_c.add_subplot(111)
    ax.plot(angles[valid], per_angle_mean[valid], "-o", markersize=2,
            linewidth=0.9, color=SEMANTIC_PALETTE["physics"])
    ax.fill_between(
        angles[valid],
        per_angle_mean[valid] - per_angle_sem[valid],
        per_angle_mean[valid] + per_angle_sem[valid],
        alpha=0.25, color=SEMANTIC_PALETTE["physics"],
        label="\u00b1 1 SEM",
    )
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Mean within-angle Pearson r")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig04_panel_c_repeatability"))
    plt.close(fig_c)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Direction-dependent spectral fingerprints",
                "asset_path": "figures/output/fig04_fingerprint_discriminability_panels/fig04_panel_a_heatmap.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle\u2013frequency heatmap of dictionary H (37 angles \u00d7 346 freq bins).",
            },
            {
                "panel_id": "b",
                "title": "Within vs between-angle similarity",
                "asset_path": "figures/output/fig04_fingerprint_discriminability_panels/fig04_panel_b_similarity.pdf",
                "provenance_mode": "data_backed",
                "description": "Violin plot with Mann\u2013Whitney U test and Cohen\u2019s d annotation.",
            },
            {
                "panel_id": "c",
                "title": "Per-angle fingerprint repeatability",
                "asset_path": "figures/output/fig04_fingerprint_discriminability_panels/fig04_panel_c_repeatability.pdf",
                "provenance_mode": "data_backed",
                "description": "Line plot of mean within-angle Pearson r \u00b1 SEM across the angle grid.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig04] Generated {len(all_paths)} files "
          f"(Mann\u2013Whitney p={p_value:.2e}, Cohen d={d_value:.2f})")
    return all_paths
