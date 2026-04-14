"""Figure 2 — compactness, component structure, and local neighborhood of H.

Double-column width (183 mm), 6 panels in a 2×3 grid:
  Row 1: (a) SVD singular-value spectrum + cumulative energy
         (b) Representative component frequency loadings (overlaid)
         (c) Representative component half-plane angle loadings (overlaid)
  Row 2: (d) Local-ordering decay in centered-|H|
         (e) All-angle reconstruction fidelity under rank-r truncation
         (f) Positive-similarity graph embedding of the centered-|H| neighborhood
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from figures.layout_contract import contract_version, font_pt, source_layout_spec
from figures.style import (
    add_panel_label,
    DOUBLE_COL_MM,
    SEMANTIC_PALETTE,
    STYLE_COLORS,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)

FIG02_TYPOGRAPHY = {
    "panel_label": font_pt("panel_label"),
    "title": font_pt("title"),
    "axis_label": font_pt("axis_label"),
    "tick_label": font_pt("tick_label"),
    "legend": font_pt("legend"),
    "annotation": font_pt("annotation"),
    "colorbar_tick": font_pt("colorbar_tick"),
    "colorbar_label": font_pt("colorbar_label"),
}
FIG02_COMPONENT_COLORS = [
    SEMANTIC_PALETTE["physics"],
    SEMANTIC_PALETTE["ablation"],
    SEMANTIC_PALETTE["learned"],
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_dictionary_h(dict_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dict_data = np.load(dict_path, allow_pickle=True)
    return np.asarray(dict_data["H"], dtype=float), np.asarray(dict_data["angles"], dtype=float)


def _build_freq_axis(F: int, fs: float = 16000.0, n_fft: float = 2048.0, f_min: float = 300.0) -> np.ndarray:
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    return (k_start + np.arange(F)) * df


def _process_frequency_mode(
    u_r: np.ndarray, freqs: np.ndarray, smooth: bool = True, n_interp: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    u_abs = np.abs(u_r)
    if not smooth:
        return u_abs, freqs

    target_coverage = 0.30
    window_length = max(15, int(len(u_abs) * target_coverage))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(u_abs) - 2)

    u_smoothed = savgol_filter(u_abs, window_length=window_length, polyorder=3, mode="nearest")

    if n_interp is not None and n_interp > len(freqs):
        cs = CubicSpline(freqs, u_smoothed)
        freqs_interp = np.linspace(freqs[0], freqs[-1], n_interp)
        u_interp = np.maximum(cs(freqs_interp), 0)
        return u_interp, freqs_interp

    return u_smoothed, freqs



def _process_angular_mode(
    v_half: np.ndarray, angles_half: np.ndarray, n_interp: int = 181
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a smooth 0–180° display curve for one angular mode."""
    v_abs = np.abs(v_half)
    v_max = np.max(v_abs)
    v_norm = v_abs / v_max if v_max > 1e-10 else v_abs

    if len(angles_half) < 2:
        return v_norm, angles_half

    spline = PchipInterpolator(angles_half, v_norm)
    angles_smooth = np.linspace(float(angles_half[0]), float(angles_half[-1]), n_interp)
    v_smooth = np.clip(spline(angles_smooth), 0, None)
    v_smooth_max = np.max(v_smooth)
    if v_smooth_max > 1e-10:
        v_smooth = v_smooth / v_smooth_max
    return v_smooth, angles_smooth


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig02_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig02",
        "composite_asset": "figures/output/fig02_svd_spectrum.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": FIG02_TYPOGRAPHY,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest_path


def _reconstruction_rmse_by_angle(
    h_centered: np.ndarray,
    u: np.ndarray,
    s: np.ndarray,
    vt: np.ndarray,
    rank: int,
) -> np.ndarray:
    """Return per-angle RMSE after rank-r reconstruction in the centered-magnitude domain."""
    reconstructed = u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :]
    return np.sqrt(np.mean((h_centered - reconstructed) ** 2, axis=0))


def _smooth_angle_series(
    angles_deg: np.ndarray,
    values: np.ndarray,
    avg_window: int = 5,
    n_interp: int = 181,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average first, then interpolate a smoother display curve through the averaged samples."""
    if len(values) < 2:
        clipped = np.clip(values, 0, None)
        return clipped, angles_deg, clipped

    clipped = np.clip(values, 0, None)
    avg_window = min(avg_window, len(clipped))
    if avg_window % 2 == 0:
        avg_window -= 1
    avg_window = max(avg_window, 1)

    if avg_window == 1:
        averaged = clipped
    else:
        pad = avg_window // 2
        padded = np.pad(clipped, pad_width=pad, mode="edge")
        kernel = np.ones(avg_window, dtype=float) / float(avg_window)
        averaged = np.convolve(padded, kernel, mode="valid")

    spline = PchipInterpolator(angles_deg, averaged)
    angles_interp = np.linspace(float(angles_deg[0]), float(angles_deg[-1]), n_interp)
    values_interp = np.clip(spline(angles_interp), 0, None)
    return averaged, angles_interp, values_interp


def _positive_similarity_graph_embedding(
    x_rows: np.ndarray,
    angles_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-dimensional embedding of the positive centered-neighborhood graph."""
    similarity = np.corrcoef(x_rows)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=0.0, neginf=0.0)
    affinity = np.clip(similarity, 0.0, None)
    np.fill_diagonal(affinity, 0.0)

    n_components, _ = connected_components(csr_matrix(affinity > 0.0), directed=False)
    if n_components != 1:
        raise RuntimeError(
            f"[fig02] positive similarity graph is disconnected ({n_components} components); "
            "panel f requires one connected neighborhood graph."
        )

    degree = affinity.sum(axis=1)
    if np.any(degree <= 0.0):
        raise RuntimeError("[fig02] positive similarity graph has an isolated node; panel f is undefined.")

    inv_sqrt_degree = np.diag(1.0 / np.sqrt(degree))
    laplacian_sym = np.eye(len(degree)) - inv_sqrt_degree @ affinity @ inv_sqrt_degree
    eigvals, eigvecs = np.linalg.eigh(laplacian_sym)
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    nontrivial = np.flatnonzero(eigvals > 1e-9)
    if len(nontrivial) < 2:
        raise RuntimeError("[fig02] graph embedding needs at least two nontrivial Laplacian eigenvectors.")

    coords = eigvecs[:, nontrivial[:2]].copy()

    corr_axis1 = np.corrcoef(coords[:, 0], angles_deg)[0, 1]
    if np.isfinite(corr_axis1) and corr_axis1 < 0:
        coords[:, 0] *= -1.0

    mid_idx = int(np.argmin(np.abs(angles_deg - 90.0)))
    end_mean = 0.5 * (coords[0, 1] + coords[-1, 1])
    if coords[mid_idx, 1] < end_mean:
        coords[:, 1] *= -1.0

    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(coords), axis=0)
    coords = coords / np.where(scale > 1e-10, scale, 1.0)
    return coords, affinity


def _centered_neighborhood_decay(
    centered_mag: np.ndarray,
    angle_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Mean and SEM of centered-|H| correlation versus angular separation."""
    angle_vectors = centered_mag.T.astype(float)
    angle_vectors = angle_vectors - angle_vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(angle_vectors, axis=1, keepdims=True) + 1e-12
    normalized = angle_vectors / norms
    corr = normalized @ normalized.T

    n_angles = corr.shape[0]
    separations = np.arange(n_angles, dtype=float) * angle_step_deg
    mean_corr = np.empty(n_angles, dtype=float)
    sem_corr = np.empty(n_angles, dtype=float)
    for lag in range(n_angles):
        vals = [corr[i, i + lag] for i in range(n_angles - lag)]
        if lag > 0:
            vals.extend(corr[i + lag, i] for i in range(n_angles - lag))
        vals_np = np.asarray(vals, dtype=float)
        mean_corr[lag] = float(vals_np.mean())
        if len(vals_np) > 1:
            sem_corr[lag] = float(vals_np.std(ddof=1) / np.sqrt(len(vals_np)))
        else:
            sem_corr[lag] = 0.0

    nonzero = mean_corr[1:]
    sep_nonzero = separations[1:]
    crossing = np.where(nonzero <= 0.0)[0]
    zero_cross_deg = float(sep_nonzero[crossing[0]]) if crossing.size else float(sep_nonzero[-1])
    return separations, mean_corr, sem_corr, zero_cross_deg


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 2 as a single composite figure at 183 mm width."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]
    dict_path = run_dir / "dictionary.npz"

    if not dict_path.exists():
        print(f"[fig02] SKIP: dictionary.npz not found at {dict_path}")
        return []

    H_np, angles_deg = _load_dictionary_h(dict_path)
    F, _E = H_np.shape
    freqs = _build_freq_axis(F)

    H_mag = np.abs(H_np)
    H_centered = H_mag - H_mag.mean(axis=1, keepdims=True)
    embedding_coords, _similarity_affinity = _positive_similarity_graph_embedding(
        H_centered.T,
        angles_deg,
    )
    angle_step_deg = float(np.median(np.diff(angles_deg))) if len(angles_deg) > 1 else 1.0
    decay_deg, mean_corr_decay, sem_corr_decay, zero_cross_deg = _centered_neighborhood_decay(
        H_centered,
        angle_step_deg=angle_step_deg,
    )

    U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
    V = Vt.T

    sigma_sq = S**2
    energy_r = sigma_sq / (sigma_sq.sum() + 1e-12)
    cum_energy_r = np.cumsum(energy_r)
    var_v_r = V.var(axis=0)
    doa_cap_r = energy_r * var_v_r
    doa_cap_r_norm = doa_cap_r / (doa_cap_r.sum() + 1e-12)
    cum_doa_cap_r = np.cumsum(doa_cap_r_norm)
    r80 = int(np.argmax(cum_energy_r >= 0.80) + 1)
    r85 = int(np.argmax(cum_energy_r >= 0.85) + 1)

    full_stop = len(S)
    panel_a_stop = min(full_stop, 10)

    selected_mode_indices = [0, 1, 5]
    n_modes = len(selected_mode_indices)
    colors = FIG02_COMPONENT_COLORS
    mode_labels = ["Comp. 1", "Comp. 2", "Comp. 6"]

    # Process modes
    modes_data = []
    for mode_idx in selected_mode_indices:
        u_smooth, freqs_smooth = _process_frequency_mode(
            U[:, mode_idx], freqs, smooth=True, n_interp=len(freqs) * 2
        )
        v_norm, angles_smooth = _process_angular_mode(V[:, mode_idx], angles_deg)
        modes_data.append((u_smooth, freqs_smooth, v_norm, angles_smooth))

    # -----------------------------------------------------------------------
    # Build composite figure: 2×3 grid
    # Row 1: (a) SVD spectrum, (b) overlaid freq profiles, (c) overlaid polar
    # Row 2: (d) local-ordering decay, (e) reconstruction quality, (f) graph embedding
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=130)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.35, wspace=0.40,
        left=0.07, right=0.96, bottom=0.07, top=0.95,
    )

    # --- Panel (a): Cumulative fraction + singular values ---
    ax_sv = fig.add_subplot(gs[0, 0])
    r_idx = np.arange(1, full_stop + 1)
    ax_sv.axvspan(1, 10, color=STYLE_COLORS["chance_fill"], alpha=0.9, zorder=0)
    line2, = ax_sv.plot(
        r_idx,
        cum_energy_r[:full_stop],
        color=SEMANTIC_PALETTE["learned"],
        marker="s",
        markersize=1.9,
        label="Cum Energy",
        linewidth=1.1,
        zorder=2,
    )
    line3, = ax_sv.plot(
        r_idx,
        cum_doa_cap_r[:full_stop],
        color=SEMANTIC_PALETTE["ablation"],
        marker="^",
        markersize=2.2,
        markerfacecolor="white",
        markeredgewidth=0.7,
        linestyle="--",
        dashes=(3.0, 1.4),
        label="Cum DoA",
        linewidth=0.9,
    )
    ax_sv.set_xlim(0.7, full_stop + 0.3)
    ax_sv.set_xticks([1, 6, 12, 18, 24, 30, 37])
    ax_sv.set_xlabel("Component index r", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_sv.set_ylabel("Cum. fraction", fontsize=FIG02_TYPOGRAPHY["axis_label"], labelpad=1)
    ax_sv.set_ylim(0.25, 1.01)
    ax_sv.set_yticks([0.25, 0.50, 0.75, 0.90, 1.00])
    ax_sv.tick_params(axis="both", labelsize=FIG02_TYPOGRAPHY["colorbar_tick"], pad=1)
    ax_sv.grid(True, axis="y", alpha=0.20, linewidth=0.5)

    ax_sv2 = ax_sv.twinx()
    ax_sv2.vlines(r_idx, 0.0, S[:full_stop], color=SEMANTIC_PALETTE["physics"], alpha=0.20, linewidth=0.9, zorder=1)
    line1, = ax_sv2.plot(
        r_idx,
        S[:full_stop],
        marker="o",
        markersize=2.0,
        label="σ_r",
        color=SEMANTIC_PALETTE["physics"],
        linewidth=1.0,
        zorder=2,
    )
    ax_sv2.set_ylabel("Singular value σ_r", fontsize=FIG02_TYPOGRAPHY["axis_label"], labelpad=1)
    ax_sv2.set_ylim(0.0, 0.76)
    ax_sv2.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_sv2.tick_params(axis="y", labelsize=FIG02_TYPOGRAPHY["colorbar_tick"], pad=1)

    for rank, text_xy in [
        (r80, (21.0, 0.855)),
        (r85, (21.0, 0.905)),
    ]:
        value = float(cum_energy_r[rank - 1])
        ax_sv.axvline(rank, color=STYLE_COLORS["guide_line"], linestyle=":", linewidth=0.6, zorder=1)
        ax_sv.scatter(rank, value, s=12, color=STYLE_COLORS["neutral_text"], zorder=4)
        ax_sv.annotate(
            f"r={rank}, {value * 100:.1f}%",
            xy=(rank, value),
            xytext=text_xy,
            textcoords="data",
            fontsize=FIG02_TYPOGRAPHY["tick_label"],
            color=STYLE_COLORS["neutral_text"],
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": STYLE_COLORS["dense_routing"]},
            va="bottom",
            ha="left",
        )

    lines = [line2, line3, line1]
    ax_sv.legend(
        lines,
        [l.get_label() for l in lines],
        loc="center",
        bbox_to_anchor=(0.73, 0.50),
        ncol=1,
        fontsize=FIG02_TYPOGRAPHY["legend"],
        frameon=True,
        framealpha=0.92,
        edgecolor=STYLE_COLORS["chance_line"],
        facecolor="white",
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    add_panel_label(ax_sv, "a", x=-0.15)

    # --- Panel (b): Overlaid frequency profiles ---
    ax_b = fig.add_subplot(gs[0, 1])
    for r in range(n_modes):
        u_smooth, freqs_smooth, _, _ = modes_data[r]
        u_abs = np.abs(u_smooth)
        ax_b.plot(freqs_smooth, u_abs, color=colors[r], linewidth=0.9, label=mode_labels[r])
    ax_b.set_title("Reusable spectral patterns", fontsize=FIG02_TYPOGRAPHY["title"], pad=1.5)
    ax_b.set_xlabel("Frequency (kHz)")
    ax_b.set_ylabel("Rel. loading", labelpad=1)
    ax_b.tick_params(axis="both", which="major", pad=1, labelsize=FIG02_TYPOGRAPHY["tick_label"])
    ax_b.set_xticks([500, 1500, 2500])
    ax_b.set_xticklabels(["0.5k", "1.5k", "2.5k"])
    ax_b.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.legend(fontsize=FIG02_TYPOGRAPHY["legend"], frameon=False, loc="upper right")
    add_panel_label(ax_b, "b", x=-0.15)

    # --- Panel (c): Overlaid polar patterns (0–180° half-plane) ---
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    for r in range(n_modes):
        _, _, v_norm, angles_smooth = modes_data[r]
        angles_rad = np.deg2rad(angles_smooth)
        ax_c.plot(angles_rad, v_norm, color=colors[r], linewidth=0.9, label=mode_labels[r])
    ax_c.set_title("Angle-selective loadings", fontsize=FIG02_TYPOGRAPHY["title"], pad=2.0)
    ax_c.grid(True, alpha=0.3, linewidth=0.5)
    ax_c.set_theta_zero_location("N")
    ax_c.set_theta_direction(-1)
    ax_c.set_thetalim(0, np.pi)
    ax_c.set_yticklabels([])
    ax_c.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax_c.set_xticklabels(["0\u00b0", "45\u00b0", "90\u00b0", "135\u00b0", "180\u00b0"], fontsize=FIG02_TYPOGRAPHY["tick_label"])
    ax_c.tick_params(pad=1)
    ax_c.legend(fontsize=FIG02_TYPOGRAPHY["legend"], frameon=False, loc="upper left", bbox_to_anchor=(-0.20, 1.10))
    add_panel_label(ax_c, "c", x=-0.15, y=1.09)

    # --- Panel (d): Local-ordering decay in centered-|H| ---
    ax_d = fig.add_subplot(gs[1, 0])
    focus_mask = decay_deg <= 90.0
    decay_focus = decay_deg[focus_mask]
    mean_focus = mean_corr_decay[focus_mask]
    sem_focus = sem_corr_decay[focus_mask]
    ax_d.axvspan(
        0.0,
        15.0,
        color=STYLE_COLORS["highlight_fill"],
        alpha=0.55,
        zorder=-1,
    )
    ax_d.fill_between(
        decay_focus,
        mean_focus - sem_focus,
        mean_focus + sem_focus,
        color=SEMANTIC_PALETTE["physics"],
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax_d.plot(
        decay_focus,
        mean_focus,
        color=SEMANTIC_PALETTE["physics"],
        linewidth=1.1,
        marker="o",
        markersize=2.4,
        zorder=2,
    )
    ax_d.axhline(
        0.0,
        color=STYLE_COLORS["guide_line"],
        linestyle="--",
        linewidth=0.6,
        zorder=0,
    )
    if zero_cross_deg <= float(decay_focus.max()):
        idx = int(np.argmin(np.abs(decay_focus - zero_cross_deg)))
        ax_d.scatter(
            [zero_cross_deg],
            [mean_focus[idx]],
            s=26,
            marker="o",
            facecolors="white",
            edgecolors=SEMANTIC_PALETTE["physics"],
            linewidths=0.9,
            zorder=3,
        )
    ax_d.text(
        0.04,
        0.08,
        "shaded = nearest-angle regime",
        transform=ax_d.transAxes,
        ha="left",
        va="bottom",
        fontsize=max(FIG02_TYPOGRAPHY["tick_label"] - 0.4, 5.0),
        color=STYLE_COLORS["muted_text"],
    )
    ax_d.set_xlim(0.0, float(decay_focus.max()))
    ax_d.set_ylim(-0.42, 1.02)
    ax_d.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax_d.set_xlabel("Angular separation |Δθ| (deg)", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_d.set_ylabel("Mean centered-|H| corr.", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_d.set_title("Local-ordering decay", fontsize=FIG02_TYPOGRAPHY["title"])
    ax_d.tick_params(axis="both", labelsize=FIG02_TYPOGRAPHY["tick_label"], pad=1)
    ax_d.grid(True, linestyle="--", alpha=0.22, linewidth=0.5)
    add_panel_label(ax_d, "d", x=-0.15)

    # --- Panel (e): All-angle reconstruction fidelity ---
    ax_e = fig.add_subplot(gs[1, 1])
    rank_styles = [
        (3, "o", FIG02_COMPONENT_COLORS[0]),
        (5, "s", FIG02_COMPONENT_COLORS[1]),
        (6, "^", FIG02_COMPONENT_COLORS[2]),
    ]
    for rank, marker, color in rank_styles:
        rmse_by_angle = _reconstruction_rmse_by_angle(H_centered, U, S, Vt, rank)
        rmse_avg, angles_interp, rmse_interp = _smooth_angle_series(
            angles_deg, rmse_by_angle, avg_window=7
        )
        ax_e.plot(
            angles_interp,
            rmse_interp,
            linewidth=1.0,
            color=color,
            label=f"r={rank} (mean={rmse_by_angle.mean():.4f})",
        )
        ax_e.plot(
            angles_deg,
            rmse_avg,
            marker=marker,
            markersize=2.5,
            linestyle="None",
            color=color,
        )
    ax_e.set_xlabel("Angle (\u00b0)", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_e.set_ylabel("RMSE", fontsize=FIG02_TYPOGRAPHY["axis_label"], labelpad=1)
    ax_e.set_title("Centered-|H| reconstruction fidelity", fontsize=FIG02_TYPOGRAPHY["title"])
    ax_e.set_xticks([0, 45, 90, 135, 180])
    ax_e.legend(fontsize=FIG02_TYPOGRAPHY["legend"], frameon=False, loc="upper right")
    ax_e.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    add_panel_label(ax_e, "e", x=-0.15)

    # --- Panel (f): Positive-similarity graph embedding ---
    ax_f = fig.add_subplot(gs[1, 2])
    angle_cmap = plt.cm.viridis
    angle_norm = plt.Normalize(float(angles_deg.min()), float(angles_deg.max()))
    ax_f.plot(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        color=STYLE_COLORS["chance_line"],
        linewidth=0.9,
        alpha=0.9,
        zorder=1,
    )
    scatter_f = ax_f.scatter(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        c=angles_deg,
        cmap=angle_cmap,
        norm=angle_norm,
        s=22,
        linewidths=0.0,
        zorder=2,
    )
    for target_angle in (0, 90, 180):
        idx = int(np.argmin(np.abs(angles_deg - target_angle)))
        ax_f.text(
            embedding_coords[idx, 0],
            embedding_coords[idx, 1],
            f"{int(angles_deg[idx])}°",
            fontsize=FIG02_TYPOGRAPHY["tick_label"],
            ha="left",
            va="bottom",
            color=STYLE_COLORS["neutral_text"],
        )
    ax_f.set_xlabel("Graph axis 1", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_f.set_ylabel("Graph axis 2", fontsize=FIG02_TYPOGRAPHY["axis_label"])
    ax_f.set_title("Centered-neighborhood graph embedding", fontsize=FIG02_TYPOGRAPHY["title"])
    ax_f.tick_params(axis="both", labelsize=FIG02_TYPOGRAPHY["tick_label"], pad=1)
    ax_f.grid(True, linestyle="--", alpha=0.18, linewidth=0.5)
    ax_f.set_aspect("equal", adjustable="box")
    cax_f = ax_f.inset_axes([1.04, 0.0, 0.045, 1.0])
    cbar = plt.colorbar(scatter_f, cax=cax_f)
    cbar.set_label("Angle (°)", fontsize=FIG02_TYPOGRAPHY["colorbar_label"])
    cbar.ax.tick_params(labelsize=FIG02_TYPOGRAPHY["colorbar_tick"])
    add_panel_label(ax_f, "f", x=-0.15, y=1.09)

    all_paths = save_outputs(fig, output_dir / "fig02_svd_spectrum")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Split panel assets for standalone panels (d, e, f)
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig02_svd_spectrum_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel d standalone
    fig_d_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_d_s.add_subplot(111)
    ax.axvspan(
        0.0,
        15.0,
        color=STYLE_COLORS["highlight_fill"],
        alpha=0.55,
        zorder=-1,
    )
    ax.fill_between(
        decay_focus,
        mean_focus - sem_focus,
        mean_focus + sem_focus,
        color=SEMANTIC_PALETTE["physics"],
        alpha=0.14,
        linewidth=0.0,
        zorder=1,
    )
    ax.plot(
        decay_focus,
        mean_focus,
        color=SEMANTIC_PALETTE["physics"],
        linewidth=1.1,
        marker="o",
        markersize=2.4,
        zorder=2,
    )
    ax.axhline(0.0, color=STYLE_COLORS["guide_line"], linestyle="--", linewidth=0.6, zorder=0)
    if zero_cross_deg <= float(decay_focus.max()):
        idx = int(np.argmin(np.abs(decay_focus - zero_cross_deg)))
        ax.scatter(
            [zero_cross_deg],
            [mean_focus[idx]],
            s=26,
            marker="o",
            facecolors="white",
            edgecolors=SEMANTIC_PALETTE["physics"],
            linewidths=0.9,
            zorder=3,
        )
    ax.text(
        0.04,
        0.08,
        "shaded = nearest-angle regime",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=max(FIG02_TYPOGRAPHY["tick_label"] - 0.4, 5.0),
        color=STYLE_COLORS["muted_text"],
    )
    ax.set_xlim(0.0, float(decay_focus.max()))
    ax.set_ylim(-0.42, 1.02)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Angular separation |Δθ| (deg)")
    ax.set_ylabel("Mean centered-|H| corr.")
    ax.grid(True, linestyle="--", alpha=0.22, linewidth=0.5)
    add_panel_label(ax, "d")
    fig_d_s.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d_s, panel_dir / "fig02_panel_d_local_order_decay"))
    plt.close(fig_d_s)

    # Panel e standalone
    fig_e_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_e_s.add_subplot(111)
    for rank, marker, color in rank_styles:
        rmse_by_angle = _reconstruction_rmse_by_angle(H_centered, U, S, Vt, rank)
        rmse_avg, angles_interp, rmse_interp = _smooth_angle_series(
            angles_deg, rmse_by_angle, avg_window=7
        )
        ax.plot(
            angles_interp,
            rmse_interp,
            linewidth=1.0,
            color=color,
            label=f"r={rank} (mean={rmse_by_angle.mean():.4f})",
        )
        ax.plot(
            angles_deg,
            rmse_avg,
            marker=marker,
            markersize=2.5,
            linestyle="None",
            color=color,
        )
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("RMSE")
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.legend(fontsize=FIG02_TYPOGRAPHY["legend"], frameon=False)
    add_panel_label(ax, "e")
    fig_e_s.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e_s, panel_dir / "fig02_panel_e_reconstruction"))
    plt.close(fig_e_s)

    # Panel f standalone
    fig_f_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_f_s.add_subplot(111)
    ax.plot(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        color=STYLE_COLORS["chance_line"],
        linewidth=0.9,
        alpha=0.9,
        zorder=1,
    )
    scatter = ax.scatter(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        c=angles_deg,
        cmap=angle_cmap,
        norm=angle_norm,
        s=20,
        linewidths=0.0,
        zorder=2,
    )
    for target_angle in (0, 90, 180):
        idx = int(np.argmin(np.abs(angles_deg - target_angle)))
        ax.text(
            embedding_coords[idx, 0],
            embedding_coords[idx, 1],
            f"{int(angles_deg[idx])}°",
            fontsize=FIG02_TYPOGRAPHY["tick_label"],
            ha="left",
            va="bottom",
            color=STYLE_COLORS["neutral_text"],
        )
    ax.set_xlabel("Graph axis 1")
    ax.set_ylabel("Graph axis 2")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.18, linewidth=0.5)
    cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0])
    plt.colorbar(scatter, cax=cax).set_label("Angle (°)", fontsize=FIG02_TYPOGRAPHY["colorbar_label"])
    add_panel_label(ax, "f")
    fig_f_s.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.92)
    all_paths.extend(save_outputs(fig_f_s, panel_dir / "fig02_panel_f_geometry"))
    plt.close(fig_f_s)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Singular-value spectrum",
                "asset_path": "figures/output/fig02_svd_spectrum.pdf",
                "provenance_mode": "data_backed",
                "description": "Full centered-|H| SVD spectrum with cumulative fraction on the left axis and singular values on the right, highlighting early saturation at r=6 (80.3%) and r=8 (85.1%).",
            },
            {
                "panel_id": "b",
                "title": "Overlaid frequency loadings (components 1, 2, 6)",
                "asset_path": "figures/output/fig02_svd_spectrum.pdf",
                "provenance_mode": "data_backed",
                "description": "Frequency loadings for representative centered-|H| components 1, 2, and 6 overlaid.",
            },
            {
                "panel_id": "c",
                "title": "Overlaid angle loadings (components 1, 2, 6)",
                "asset_path": "figures/output/fig02_svd_spectrum.pdf",
                "provenance_mode": "data_backed",
                "description": "Half-plane angle loadings for representative centered-|H| components 1, 2, and 6 overlaid across 0-180 degrees.",
            },
            {
                "panel_id": "d",
                "title": "Local-ordering decay",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_d_local_order_decay.pdf",
                "provenance_mode": "data_backed",
                "description": "Mean centered-|H| correlation versus angular separation for the acrylic reference object, showing a finite positive local neighborhood that decays toward a first nonpositive mean near 25 degrees.",
            },
            {
                "panel_id": "e",
                "title": "Centered-|H| reconstruction fidelity",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_e_reconstruction.pdf",
                "provenance_mode": "data_backed",
                "description": "Per-angle centered-magnitude RMSE under rank-3, rank-5, and rank-6 truncation across all 37 angles.",
            },
            {
                "panel_id": "f",
                "title": "Centered-neighborhood graph embedding",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_f_geometry.pdf",
                "provenance_mode": "data_backed",
                "description": "Two-dimensional spectral embedding of the positive centered-neighborhood graph derived from centered-|H|, showing a curved and continuous angle trajectory in graph coordinates.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig02] Generated {len(all_paths)} files (H={H_np.shape}, angles={len(angles_deg)})")
    return all_paths
