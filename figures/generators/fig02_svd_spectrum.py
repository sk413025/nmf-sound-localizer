"""Figure 2 — SVD spectrum + centered-|H| component patterns + dictionary + manifold.

Double-column width (183 mm), 6 panels in a 2×3 grid:
  Row 1: (a) SVD singular-value spectrum + cumulative energy
         (b) Representative component frequency loadings (overlaid)
         (c) Representative component half-plane angle loadings (overlaid)
  Row 2: (d) Full dictionary H heatmap (angle x freq)
         (e) All-angle reconstruction fidelity under rank-r truncation
         (f) Inter-angle correlation matrix of H
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, PchipInterpolator
from sklearn.manifold import MDS

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    DOUBLE_COL_MM,
)


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
        "figure_id": "fig02",
        "composite_asset": "figures/output/fig02_svd_spectrum.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
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


def _metric_mds_embedding(x_rows: np.ndarray) -> np.ndarray:
    """Two-dimensional metric-MDS embedding for the angle rows."""
    diffs = x_rows[:, None, :] - x_rows[None, :, :]
    distances = np.sqrt(np.maximum(np.sum(diffs * diffs, axis=2), 0.0))
    mds = MDS(
        n_components=2,
        metric=True,
        dissimilarity="precomputed",
        random_state=0,
        n_init=8,
        max_iter=500,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(distances)
    coords = coords - coords.mean(axis=0, keepdims=True)
    scale = np.max(np.abs(coords), axis=0)
    coords = coords / np.where(scale > 1e-10, scale, 1.0)
    return coords


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
    embedding_coords = _metric_mds_embedding(H_centered.T)

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
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
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
    # Row 2: (d) full H heatmap, (e) reconstruction quality, (f) correlation
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
    ax_sv.axvspan(1, 10, color="0.97", alpha=0.9, zorder=0)
    line2, = ax_sv.plot(
        r_idx,
        cum_energy_r[:full_stop],
        color="g",
        marker="s",
        markersize=1.9,
        label="Cum Energy",
        linewidth=1.1,
        zorder=2,
    )
    line3, = ax_sv.plot(
        r_idx,
        cum_doa_cap_r[:full_stop],
        color="#d62728",
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
    ax_sv.set_xlabel("Component index r", fontsize=6)
    ax_sv.set_ylabel("Cum. fraction", fontsize=6, labelpad=1)
    ax_sv.set_ylim(0.25, 1.01)
    ax_sv.set_yticks([0.25, 0.50, 0.75, 0.90, 1.00])
    ax_sv.tick_params(axis="both", labelsize=5.5, pad=1)
    ax_sv.grid(True, axis="y", alpha=0.20, linewidth=0.5)

    ax_sv2 = ax_sv.twinx()
    ax_sv2.vlines(r_idx, 0.0, S[:full_stop], color="C0", alpha=0.20, linewidth=0.9, zorder=1)
    line1, = ax_sv2.plot(
        r_idx,
        S[:full_stop],
        marker="o",
        markersize=2.0,
        label=r"$\sigma_r$",
        color="C0",
        linewidth=1.0,
        zorder=2,
    )
    ax_sv2.set_ylabel(r"Singular value $\sigma_r$", fontsize=6, labelpad=1)
    ax_sv2.set_ylim(0.0, 0.76)
    ax_sv2.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax_sv2.tick_params(axis="y", labelsize=5.5, pad=1)

    for rank, text_xy in [
        (r80, (21.0, 0.855)),
        (r85, (21.0, 0.905)),
    ]:
        value = float(cum_energy_r[rank - 1])
        ax_sv.axvline(rank, color="0.55", linestyle=":", linewidth=0.6, zorder=1)
        ax_sv.scatter(rank, value, s=12, color="#2f2f2f", zorder=4)
        ax_sv.annotate(
            f"r={rank}, {value * 100:.1f}%",
            xy=(rank, value),
            xytext=text_xy,
            textcoords="data",
            fontsize=4.8,
            color="#2f2f2f",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
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
        fontsize=4.9,
        frameon=True,
        framealpha=0.92,
        edgecolor="0.85",
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
    ax_b.set_title("Reusable spectral patterns", fontsize=6.2, pad=1.5)
    ax_b.set_xlabel("Frequency (kHz)")
    ax_b.set_ylabel("Rel. loading", labelpad=1)
    ax_b.tick_params(axis="both", which="major", pad=1, labelsize=5)
    ax_b.set_xticks([500, 1500, 2500])
    ax_b.set_xticklabels(["0.5k", "1.5k", "2.5k"])
    ax_b.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.legend(fontsize=5.5, frameon=False, loc="upper right")
    add_panel_label(ax_b, "b", x=-0.15)

    # --- Panel (c): Overlaid polar patterns (0–180° half-plane) ---
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    for r in range(n_modes):
        _, _, v_norm, angles_smooth = modes_data[r]
        angles_rad = np.deg2rad(angles_smooth)
        ax_c.plot(angles_rad, v_norm, color=colors[r], linewidth=0.9, label=mode_labels[r])
    ax_c.set_title("Angle-selective loadings", fontsize=6.2, pad=2.0)
    ax_c.grid(True, alpha=0.3, linewidth=0.5)
    ax_c.set_theta_zero_location("N")
    ax_c.set_theta_direction(-1)
    ax_c.set_thetalim(0, np.pi)
    ax_c.set_yticklabels([])
    ax_c.set_xticks(np.deg2rad([0, 45, 90, 135, 180]))
    ax_c.set_xticklabels(["0\u00b0", "45\u00b0", "90\u00b0", "135\u00b0", "180\u00b0"], fontsize=6)
    ax_c.tick_params(pad=1)
    ax_c.legend(fontsize=5.5, frameon=False, loc="upper left", bbox_to_anchor=(-0.20, 1.10))
    add_panel_label(ax_c, "c", x=-0.15, y=1.09)

    # --- Panel (d): Full dictionary H heatmap ---
    ax_d = fig.add_subplot(gs[1, 0])
    im_d = ax_d.imshow(
        H_mag.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles_deg[0], angles_deg[-1]],
    )
    ax_d.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_d.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_title("Angle-frequency dictionary H", fontsize=6.5)
    cbar = plt.colorbar(im_d, ax=ax_d, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_d, "d", x=-0.15)

    # --- Panel (e): All-angle reconstruction fidelity ---
    ax_e = fig.add_subplot(gs[1, 1])
    rank_styles = [
        (3, "o", "#1f77b4"),
        (5, "s", "#ff7f0e"),
        (6, "^", "#2ca02c"),
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
    ax_e.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_e.set_ylabel("RMSE", fontsize=6, labelpad=1)
    ax_e.set_title(r"Centered-$|H|$ reconstruction fidelity", fontsize=6.5)
    ax_e.set_xticks([0, 45, 90, 135, 180])
    ax_e.legend(fontsize=5, frameon=False, loc="upper right")
    ax_e.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    add_panel_label(ax_e, "e", x=-0.15)

    # --- Panel (f): Inter-angle fingerprint similarity ---
    ax_f = fig.add_subplot(gs[1, 2])
    H_corr = np.corrcoef(H_np.T)  # E x E correlation, matching Fig. 5 structure analysis
    # Positive-only similarity matrix: use a sequential colormap rather than a zero-centered diverging map.
    im_f = ax_f.imshow(H_corr, cmap="viridis", aspect="equal",
                        vmin=float(H_corr.min()), vmax=1.0)
    tick_pos = np.linspace(0, len(angles_deg) - 1, 5, dtype=int).tolist()
    tick_lab = [f"{int(angles_deg[i])}" for i in tick_pos]
    ax_f.set_xticks(tick_pos)
    ax_f.set_xticklabels(tick_lab, fontsize=5)
    ax_f.set_yticks(tick_pos)
    ax_f.set_yticklabels(tick_lab, fontsize=5)
    ax_f.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_f.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_f.set_title("Inter-angle fingerprint similarity", fontsize=6.5)
    cax_f = ax_f.inset_axes([1.04, 0.0, 0.045, 1.0])
    cbar = plt.colorbar(im_f, cax=cax_f)
    cbar.set_label("Pearson r", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    ax_f_in = ax_f.inset_axes([0.54, 0.53, 0.38, 0.38])
    angle_cmap = plt.cm.viridis
    angle_norm = plt.Normalize(float(angles_deg.min()), float(angles_deg.max()))
    ax_f_in.plot(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        color="0.75",
        linewidth=0.7,
        alpha=0.9,
        zorder=1,
    )
    ax_f_in.scatter(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        c=angles_deg,
        cmap=angle_cmap,
        norm=angle_norm,
        s=9,
        linewidths=0.0,
        zorder=2,
    )
    for target_angle in (0, 90, 180):
        idx = int(np.argmin(np.abs(angles_deg - target_angle)))
        ax_f_in.text(
            embedding_coords[idx, 0],
            embedding_coords[idx, 1],
            f"{int(angles_deg[idx])}°",
            fontsize=4.3,
            ha="left",
            va="bottom",
            color="0.2",
        )
    ax_f_in.set_xticks([])
    ax_f_in.set_yticks([])
    ax_f_in.set_title("2D geometry", fontsize=4.8, pad=1.2)
    for spine in ax_f_in.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.5)
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
    im = ax.imshow(
        H_mag.T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles_deg[0], angles_deg[-1]],
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(r"$|H|$", fontsize=6)
    add_panel_label(ax, "d")
    fig_d_s.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d_s, panel_dir / "fig02_panel_d_full_H"))
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
    ax.legend(fontsize=5, frameon=False)
    add_panel_label(ax, "e")
    fig_e_s.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e_s, panel_dir / "fig02_panel_e_reconstruction"))
    plt.close(fig_e_s)

    # Panel f standalone
    fig_f_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_f_s.add_subplot(111)
    im = ax.imshow(H_corr, cmap="viridis", aspect="equal",
                   vmin=float(H_corr.min()), vmax=1.0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=6)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=6)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0])
    plt.colorbar(im, cax=cax).set_label("Pearson r", fontsize=6)
    ax_in = ax.inset_axes([0.54, 0.53, 0.38, 0.38])
    ax_in.plot(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        color="0.75",
        linewidth=0.7,
        alpha=0.9,
        zorder=1,
    )
    ax_in.scatter(
        embedding_coords[:, 0],
        embedding_coords[:, 1],
        c=angles_deg,
        cmap=plt.cm.viridis,
        norm=plt.Normalize(float(angles_deg.min()), float(angles_deg.max())),
        s=9,
        linewidths=0.0,
        zorder=2,
    )
    for target_angle in (0, 90, 180):
        idx = int(np.argmin(np.abs(angles_deg - target_angle)))
        ax_in.text(
            embedding_coords[idx, 0],
            embedding_coords[idx, 1],
            f"{int(angles_deg[idx])}°",
            fontsize=4.3,
            ha="left",
            va="bottom",
            color="0.2",
        )
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_title("2D geometry", fontsize=4.8, pad=1.2)
    for spine in ax_in.spines.values():
        spine.set_color("0.55")
        spine.set_linewidth(0.5)
    add_panel_label(ax, "f")
    fig_f_s.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.92)
    all_paths.extend(save_outputs(fig_f_s, panel_dir / "fig02_panel_f_correlation"))
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
                "title": "Angle-frequency dictionary H",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_d_full_H.pdf",
                "provenance_mode": "data_backed",
                "description": "Complete angle-frequency heatmap of H (37 angles x 346 freq bins).",
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
                "title": "Inter-angle fingerprint similarity",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_f_correlation.pdf",
                "provenance_mode": "data_backed",
                "description": "37x37 fingerprint-similarity matrix whose near-diagonal band shows smooth local ordering, with a 2D embedding inset of the same centered-|H| geometry.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig02] Generated {len(all_paths)} files (H={H_np.shape}, angles={len(angles_deg)})")
    return all_paths
