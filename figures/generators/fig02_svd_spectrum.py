"""Figure 2 — SVD Spectrum + Modal Decomposition + Dictionary.

Single composite figure at double-column width (183 mm):
  Row 1: (a) SVD singular-value spectrum
  Row 2: Mode 1–3 frequency profiles (left) + polar angular modes (right)
  Row 3: Mode 1–3 dictionary heatmaps
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

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

def _load_h_matrix(h_path: Path) -> tuple[np.ndarray, np.ndarray]:
    import torch

    h_data = torch.load(str(h_path), map_location="cpu", weights_only=False)
    return h_data["H"].cpu().numpy(), np.array(h_data["angles"], dtype=float)


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


def _process_angular_mode(v_half: np.ndarray, angles_half: np.ndarray, n_interp: int = 360) -> tuple[np.ndarray, np.ndarray]:
    angles_mirror = 360.0 - angles_half
    angles_combined = np.concatenate([angles_half, angles_mirror])
    v_combined = np.concatenate([v_half, v_half])

    sort_idx = np.argsort(angles_combined)
    angles_sorted = angles_combined[sort_idx]
    v_sorted = v_combined[sort_idx]

    angles_ext = np.concatenate([angles_sorted - 360, angles_sorted, angles_sorted + 360])
    v_ext = np.concatenate([v_sorted, v_sorted, v_sorted])
    cs = CubicSpline(angles_ext, v_ext)

    angles_smooth = np.linspace(0, 360, n_interp + 1, endpoint=True)
    v_dense = cs(angles_smooth)

    window_length = 31
    if window_length > len(v_dense):
        window_length = len(v_dense) // 2 * 2 + 1
    v_smooth = savgol_filter(v_dense, window_length=window_length, polyorder=3, mode="wrap")

    v_abs = np.abs(v_smooth)
    v_max = np.max(v_abs)
    v_norm = v_abs / v_max if v_max > 1e-10 else v_abs
    return v_norm, angles_smooth


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 2 as a single composite figure at 183 mm width."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    h_path = data_root / paths_cfg["h_matrix"]

    if not h_path.exists():
        print(f"[fig02] SKIP: H matrix not found at {h_path}")
        return []

    H_np, angles_deg = _load_h_matrix(h_path)
    F, _E = H_np.shape
    freqs = _build_freq_axis(F)

    eps = 1e-8
    H_log = np.log(np.clip(np.abs(H_np), eps, None))
    H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)

    U, S, Vt = np.linalg.svd(H_log_centered, full_matrices=False)
    V = Vt.T

    sigma_sq = S**2
    energy_r = sigma_sq / (sigma_sq.sum() + 1e-12)
    var_v_r = V.var(axis=0)
    doa_cap_r = energy_r * var_v_r
    doa_cap_r_norm = doa_cap_r / (doa_cap_r.sum() + 1e-12)

    n_modes = 3
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Process modes
    modes_data = []
    for r in range(n_modes):
        u_smooth, freqs_smooth = _process_frequency_mode(U[:, r], freqs, smooth=True, n_interp=len(freqs) * 2)
        v_norm, angles_smooth = _process_angular_mode(V[:, r], angles_deg)
        modes_data.append((u_smooth, freqs_smooth, v_norm, angles_smooth))

    # Build composite figure: 3 rows
    # Row 1: singular values (spans full width)
    # Row 2: 3 x (freq + polar) = 6 columns
    # Row 3: 3 dictionary heatmaps
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=165)
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        height_ratios=[1.1, 0.95, 0.8],
        hspace=0.38, wspace=0.45,
        left=0.07, right=0.96, bottom=0.06, top=0.95,
    )

    # --- Row 1: Singular values (a) ---
    ax_sv = fig.add_subplot(gs[0, 1:5])
    r_idx = np.arange(1, len(S) + 1)
    line1, = ax_sv.semilogy(r_idx, S, marker="o", markersize=2, label=r"$\sigma_r$", color="C0", linewidth=0.9)
    ax_sv.set_xlabel("Mode index r")
    ax_sv.set_ylabel(r"Singular value $\sigma_r$")

    ax_sv2 = ax_sv.twinx()
    line2, = ax_sv2.plot(r_idx, np.cumsum(energy_r), "g-", marker="s", markersize=2, label="Cum Energy", linewidth=0.9)
    line3, = ax_sv2.plot(r_idx, np.cumsum(doa_cap_r_norm), "r-", marker="^", markersize=2, label="Cum DOA Cap", linewidth=0.9)
    ax_sv2.set_ylabel("Cumulative fraction")
    lines = [line1, line2, line3]
    ax_sv.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=6, frameon=False)
    add_panel_label(ax_sv, "a", x=-0.08)

    # --- Row 2: Freq + Polar for each mode ---
    panel_labels = ["b", "c", "d"]
    for r in range(n_modes):
        u_smooth, freqs_smooth, v_norm, angles_smooth = modes_data[r]
        color = colors[r]

        # Frequency mode (left sub-column)
        ax_f = fig.add_subplot(gs[1, r * 2])
        u_abs = np.abs(u_smooth)
        ax_f.plot(freqs_smooth, u_abs, color=color, linewidth=0.9)
        ax_f.fill_between(freqs_smooth, u_abs, color=color, alpha=0.15)
        ax_f.set_ylabel(f"$|u_{r+1}(f)|$", labelpad=1)
        ax_f.tick_params(axis="both", which="major", pad=1, labelsize=5)
        ax_f.set_xticks([500, 1500, 2500])
        ax_f.set_xticklabels(["0.5k", "1.5k", "2.5k"])
        ax_f.grid(True, alpha=0.3, linewidth=0.5)
        ax_f.set_title(f"Mode {r+1}", fontsize=7)
        if r == 0:
            add_panel_label(ax_f, panel_labels[r], x=-0.3)
        else:
            add_panel_label(ax_f, panel_labels[r], x=-0.15)

        # Angular mode (right sub-column, polar)
        ax_p = fig.add_subplot(gs[1, r * 2 + 1], projection="polar")
        angles_rad = np.deg2rad(angles_smooth)
        ax_p.plot(angles_rad, v_norm, color=color, linewidth=0.9)
        ax_p.fill(angles_rad, v_norm, color=color, alpha=0.15)
        ax_p.grid(True, alpha=0.3, linewidth=0.5)
        ax_p.set_yticklabels([])
        ax_p.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax_p.set_xticklabels(["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"], fontsize=5)
        ax_p.tick_params(pad=-2)

    # --- Row 3: Dictionary heatmaps ---
    panel_labels_d = ["e", "f", "g"]
    for r in range(n_modes):
        u_smooth, freqs_smooth, v_norm, angles_smooth = modes_data[r]
        mask = angles_smooth <= 180
        v_half = v_norm[mask]
        D_r = np.outer(np.abs(u_smooth), v_half)

        ax_d = fig.add_subplot(gs[2, r * 2:r * 2 + 2])
        extent = [0, D_r.shape[1], freqs_smooth[0], freqs_smooth[-1]]
        ax_d.imshow(D_r, aspect="auto", origin="lower", cmap="viridis", extent=extent)
        ax_d.set_title(f"Dict Mode {r+1}", fontsize=7)
        ax_d.set_xlabel("Atom index", fontsize=6, labelpad=1)
        if r == 0:
            ax_d.set_ylabel("Freq (Hz)", fontsize=6, labelpad=1)
        ax_d.tick_params(labelsize=5, pad=1)
        ax_d.grid(False)
        add_panel_label(ax_d, panel_labels_d[r], x=-0.08)

    all_paths = save_outputs(fig, output_dir / "fig02_svd_spectrum")
    plt.close(fig)

    print(f"[fig02] Generated {len(all_paths)} files")
    return all_paths
