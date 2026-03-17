"""Figure 3 — Fingerprint Discriminability (5 panels).

Panel (a): Angle-frequency heatmap of the physical dictionary H.
Panel (b): Within-angle vs between-angle Pearson r (violin/box) + statistical
           annotation (Mann-Whitney U, Cohen's d).
Panel (c): Per-angle fingerprint repeatability (line + SEM band).
Panel (d): Full pairwise similarity matrix (37x37) from trial-level data.
Panel (e): Prediction confidence histogram (correct vs incorrect trials).

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
    """Compute within-angle and between-angle Pearson r distributions."""
    within_all: list[float] = []
    between_all: list[float] = []
    per_angle_mean = np.full(n_angles, np.nan)
    per_angle_std = np.full(n_angles, np.nan)
    per_angle_n_pairs = np.zeros(n_angles, dtype=int)

    angle_indices: dict[int, np.ndarray] = {}
    for a in range(n_angles):
        idx = np.where(labels == a)[0]
        if len(idx) > 1:
            angle_indices[a] = idx

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


def _compute_trial_pairwise_matrix(Y_val: np.ndarray, labels: np.ndarray,
                                    n_angles: int) -> np.ndarray:
    """Compute mean Pearson r between all angle pairs from trial-level data."""
    sim_matrix = np.full((n_angles, n_angles), np.nan)
    rng = np.random.default_rng(42)

    angle_indices: dict[int, np.ndarray] = {}
    for a in range(n_angles):
        idx = np.where(labels == a)[0]
        if len(idx) > 0:
            angle_indices[a] = idx

    for a1 in range(n_angles):
        if a1 not in angle_indices:
            continue
        for a2 in range(a1, n_angles):
            if a2 not in angle_indices:
                continue
            idx1 = angle_indices[a1]
            idx2 = angle_indices[a2]
            if a1 == a2:
                if len(idx1) < 2:
                    continue
                clips = Y_val[idx1]
                corr_mat = np.corrcoef(clips)
                triu = np.triu_indices(len(idx1), k=1)
                sim_matrix[a1, a2] = np.mean(corr_mat[triu])
            else:
                n_sample = min(10, len(idx1), len(idx2))
                sel1 = rng.choice(idx1, size=n_sample, replace=False)
                sel2 = rng.choice(idx2, size=n_sample, replace=False)
                r_vals = []
                for s1, s2 in zip(sel1, sel2):
                    r_vals.append(np.corrcoef(Y_val[s1], Y_val[s2])[0, 1])
                sim_matrix[a1, a2] = np.mean(r_vals)
                sim_matrix[a2, a1] = sim_matrix[a1, a2]

    return sim_matrix


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
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
    manifest_path = panel_dir / "fig03_panel_manifest.json"
    payload = {
        "figure_id": "fig03",
        "composite_asset": "figures/output/fig03_fingerprint_discriminability.pdf",
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
    """Generate Figure 3 — Fingerprint Discriminability (5 panels)."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"

    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig03] SKIP: data not found at {run_dir}")
        return []

    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    Y_val = routing_data["Y_val"]
    labels = routing_data["labels"].astype(int)
    scores_expert = routing_data["scores_expert"]
    H = dict_data["H"]
    angles = dict_data["angles"]
    n_angles = len(angles)

    F = H.shape[0]
    freqs = _build_freq_axis(F)

    # Compute similarity distributions
    within_all, between_all, per_angle_mean, per_angle_std, per_angle_n = \
        _compute_pairwise_corr(Y_val, labels, n_angles)

    # Statistical test
    within_arr = np.array(within_all)
    between_arr = np.array(between_all)
    U_stat, p_value = sp_stats.mannwhitneyu(
        within_arr, between_arr, alternative="greater"
    )
    d_value = _cohens_d(within_arr, between_arr)

    per_angle_sem = np.where(
        per_angle_n > 0,
        per_angle_std / np.sqrt(per_angle_n),
        0.0,
    )

    # Compute trial-level pairwise similarity matrix (panel d)
    sim_matrix = _compute_trial_pairwise_matrix(Y_val, labels, n_angles)

    # Compute prediction confidence (panel e)
    pred_labels = np.argmax(scores_expert, axis=1)
    correct_mask = pred_labels == labels
    # Confidence = max softmax score
    scores_exp = np.exp(scores_expert - scores_expert.max(axis=1, keepdims=True))
    scores_softmax = scores_exp / scores_exp.sum(axis=1, keepdims=True)
    confidence = np.max(scores_softmax, axis=1)

    # -----------------------------------------------------------------------
    # Build composite figure (2 rows: top 3 panels, bottom 2 panels)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=145)
    gs = gridspec.GridSpec(
        2, 6, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.55, wspace=0.80,
        left=0.06, right=0.97, bottom=0.08, top=0.92,
    )

    # --- Row 1 ---

    # Panel (a): Angle-frequency heatmap
    ax_a = fig.add_subplot(gs[0, 0:2])
    im = ax_a.imshow(
        H.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles[0], angles[-1]],
    )
    ax_a.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_a.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_a.set_title("Direction-dependent fingerprints", fontsize=6.5)
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_a, "a", x=-0.12, y=1.06)

    # Panel (b): Within vs between violin + stats
    ax_b = fig.add_subplot(gs[0, 2:4])
    data_violin = [within_all, between_all]
    parts = ax_b.violinplot(data_violin, positions=[0, 1], showmeans=True,
                            showmedians=False, showextrema=False)
    violin_colors = [SEMANTIC_PALETTE["physics"], SEMANTIC_PALETTE["ablation"]]
    for pc, color in zip(parts["bodies"], violin_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.0)

    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Within\nangle", "Between\nangles"], fontsize=5.5)
    ax_b.set_ylabel("Pearson r", fontsize=6)
    ax_b.set_title("Fingerprint similarity", fontsize=6.5)
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_b, "b", x=-0.20, y=1.06)

    y_max = max(np.max(within_all), np.max(between_all))
    bracket_y = y_max + 0.02
    bar_y = bracket_y + 0.01
    ax_b.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y],
              lw=0.8, c="black")
    stars = _p_to_stars(p_value)
    ax_b.text(0.5, bar_y + 0.005, f"{stars}\nd = {d_value:.2f}",
              ha="center", va="bottom", fontsize=6.5, linespacing=1.2)

    # Panel (c): Per-angle repeatability
    ax_c = fig.add_subplot(gs[0, 4:6])
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
    ax_c.legend(fontsize=6, frameon=False, loc="lower right")
    add_panel_label(ax_c, "c", x=-0.12, y=1.06)

    # --- Row 2 ---

    # Panel (d): Full pairwise similarity matrix (37x37)
    ax_d = fig.add_subplot(gs[1, 0:3])
    im_d = ax_d.imshow(sim_matrix, cmap="RdBu_r", aspect="equal",
                        vmin=-0.2, vmax=1.0)
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]
    ax_d.set_xticks(tick_positions)
    ax_d.set_xticklabels(tick_labels, fontsize=5)
    ax_d.set_yticks(tick_positions)
    ax_d.set_yticklabels(tick_labels, fontsize=5)
    ax_d.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_title("Trial-level pairwise similarity", fontsize=6.5)
    cbar = plt.colorbar(im_d, ax=ax_d, fraction=0.03, pad=0.01)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_d, "d", x=-0.10, y=1.06)

    # Panel (e): Prediction confidence histogram
    ax_e = fig.add_subplot(gs[1, 3:6])
    bins = np.linspace(0, 1, 30)
    ax_e.hist(confidence[correct_mask], bins=bins, alpha=0.7,
              color=SEMANTIC_PALETTE["learned"], label="Correct",
              density=True, edgecolor="none")
    ax_e.hist(confidence[~correct_mask], bins=bins, alpha=0.7,
              color=SEMANTIC_PALETTE["ablation"], label="Incorrect",
              density=True, edgecolor="none")
    ax_e.set_xlabel("Prediction confidence", fontsize=6)
    ax_e.set_ylabel("Density", fontsize=6)
    ax_e.set_title("Confidence distribution", fontsize=6.5)
    ax_e.legend(fontsize=5, frameon=False, loc="upper left")
    ax_e.grid(axis="y", linestyle="--", alpha=0.3)
    n_correct = correct_mask.sum()
    n_total = len(correct_mask)
    ax_e.text(0.95, 0.95, f"Acc: {n_correct/n_total:.1%}",
              transform=ax_e.transAxes, ha="right", va="top", fontsize=5.5,
              style="italic")
    add_panel_label(ax_e, "e", x=-0.10, y=1.06)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig03_fingerprint_discriminability")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig03_fingerprint_discriminability_panels"
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
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig03_panel_a_heatmap"))
    plt.close(fig_a)

    # Panel b standalone
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_b.add_subplot(111)
    parts = ax.violinplot(data_violin, positions=[0, 1], showmeans=True,
                          showmedians=False, showextrema=False)
    for pc, color in zip(parts["bodies"], violin_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within angle", "Between angles"])
    ax.set_ylabel("Pearson r")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y], lw=0.8, c="black")
    ax.text(0.5, bar_y + 0.005, f"{stars}\nd = {d_value:.2f}",
            ha="center", va="bottom", fontsize=5.5, linespacing=1.2)
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig03_panel_b_similarity"))
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
        alpha=0.25, color=SEMANTIC_PALETTE["physics"], label="\u00b1 1 SEM",
    )
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Mean within-angle Pearson r")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig03_panel_c_repeatability"))
    plt.close(fig_c)

    # Panel d standalone
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_d.add_subplot(111)
    im = ax.imshow(sim_matrix, cmap="RdBu_r", aspect="equal", vmin=-0.2, vmax=1.0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Mean Pearson r", fontsize=6)
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig03_panel_d_pairwise"))
    plt.close(fig_d)

    # Panel e standalone
    fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_e.add_subplot(111)
    ax.hist(confidence[correct_mask], bins=bins, alpha=0.7,
            color=SEMANTIC_PALETTE["learned"], label="Correct", density=True,
            edgecolor="none")
    ax.hist(confidence[~correct_mask], bins=bins, alpha=0.7,
            color=SEMANTIC_PALETTE["ablation"], label="Incorrect", density=True,
            edgecolor="none")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Density")
    ax.legend(fontsize=5, frameon=False)
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig03_panel_e_confidence"))
    plt.close(fig_e)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Direction-dependent spectral fingerprints",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_a_heatmap.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle-frequency heatmap of dictionary H (37 angles x 346 freq bins).",
            },
            {
                "panel_id": "b",
                "title": "Within vs between-angle similarity",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_b_similarity.pdf",
                "provenance_mode": "data_backed",
                "description": "Violin plot with Mann-Whitney U test and Cohen's d annotation.",
            },
            {
                "panel_id": "c",
                "title": "Per-angle fingerprint repeatability",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_c_repeatability.pdf",
                "provenance_mode": "data_backed",
                "description": "Line plot of mean within-angle Pearson r +/- SEM across the angle grid.",
            },
            {
                "panel_id": "d",
                "title": "Full pairwise similarity matrix",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_d_pairwise.pdf",
                "provenance_mode": "data_backed",
                "description": "37x37 mean Pearson r between all angle pairs from trial-level data.",
            },
            {
                "panel_id": "e",
                "title": "Prediction confidence distribution",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_e_confidence.pdf",
                "provenance_mode": "data_backed",
                "description": "Histograms of prediction confidence for correct vs incorrect trials.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig03] Generated {len(all_paths)} files "
          f"(Mann-Whitney p={p_value:.2e}, Cohen d={d_value:.2f})")
    return all_paths
