"""Figure 2 — SVD Spectrum + Modal Decomposition + Dictionary + Manifold.

Double-column width (183 mm), 6 panels in a 2×3 grid:
  Row 1: (a) SVD singular-value spectrum + cumulative energy
         (b) Mode 1-3 frequency profiles (overlaid)
         (c) Mode 1-3 polar patterns (overlaid)
  Row 2: (d) Full dictionary H heatmap (angle x freq)
         (e) Rank-r reconstruction quality (original vs reconstructed)
         (f) Inter-angle correlation matrix of H
"""

from __future__ import annotations

import json
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
    SEMANTIC_PALETTE,
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
    """Mirror 0–180° to 0–360°, interpolate and smooth for polar plot."""
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
    mode_labels = ["Mode 1", "Mode 2", "Mode 3"]

    # Process modes
    modes_data = []
    for r in range(n_modes):
        u_smooth, freqs_smooth = _process_frequency_mode(U[:, r], freqs, smooth=True, n_interp=len(freqs) * 2)
        v_norm, angles_smooth = _process_angular_mode(V[:, r], angles_deg)
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

    # --- Panel (a): Singular values ---
    ax_sv = fig.add_subplot(gs[0, 0])
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
    add_panel_label(ax_sv, "a", x=-0.15)

    # --- Panel (b): Overlaid frequency profiles ---
    ax_b = fig.add_subplot(gs[0, 1])
    for r in range(n_modes):
        u_smooth, freqs_smooth, _, _ = modes_data[r]
        u_abs = np.abs(u_smooth)
        ax_b.plot(freqs_smooth, u_abs, color=colors[r], linewidth=0.9, label=mode_labels[r])
        ax_b.fill_between(freqs_smooth, u_abs, color=colors[r], alpha=0.15)
    ax_b.set_xlabel("Frequency (kHz)")
    ax_b.set_ylabel(r"$|u_r(f)|$", labelpad=1)
    ax_b.tick_params(axis="both", which="major", pad=1, labelsize=5)
    ax_b.set_xticks([500, 1500, 2500])
    ax_b.set_xticklabels(["0.5k", "1.5k", "2.5k"])
    ax_b.grid(True, alpha=0.3, linewidth=0.5)
    ax_b.legend(fontsize=5.5, frameon=False, loc="upper right")
    add_panel_label(ax_b, "b", x=-0.15)

    # --- Panel (c): Overlaid polar patterns (0–360°, mirrored) ---
    ax_c = fig.add_subplot(gs[0, 2], projection="polar")
    for r in range(n_modes):
        _, _, v_norm, angles_smooth = modes_data[r]
        angles_rad = np.deg2rad(angles_smooth)
        ax_c.plot(angles_rad, v_norm, color=colors[r], linewidth=0.9, label=mode_labels[r])
        ax_c.fill(angles_rad, v_norm, color=colors[r], alpha=0.2)
    ax_c.grid(True, alpha=0.3, linewidth=0.5)
    ax_c.set_yticklabels([])
    ax_c.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax_c.set_xticklabels(["0\u00b0", "90\u00b0", "180\u00b0", "270\u00b0"], fontsize=6)
    ax_c.tick_params(pad=-2)
    ax_c.legend(fontsize=5.5, frameon=False, loc="upper right", bbox_to_anchor=(1.25, 1.15))
    add_panel_label(ax_c, "c", x=-0.15)

    # --- Panel (d): Full dictionary H heatmap ---
    ax_d = fig.add_subplot(gs[1, 0])
    im_d = ax_d.imshow(
        np.abs(H_np).T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles_deg[0], angles_deg[-1]],
    )
    ax_d.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_d.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_title("Full dictionary H", fontsize=6.5)
    cbar = plt.colorbar(im_d, ax=ax_d, fraction=0.035, pad=0.02)
    cbar.set_label("Amplitude", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_d, "d", x=-0.15)

    # --- Panel (e): Rank-r reconstruction quality ---
    ax_e = fig.add_subplot(gs[1, 1])
    rep_angle_idx = int(np.argmin(np.abs(angles_deg - 90)))
    original = H_log_centered[:, rep_angle_idx]

    for rank, ls, alpha_val in [(3, "--", 0.7), (5, "-.", 0.5), (10, ":", 0.4)]:
        H_reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        reconstructed = H_reconstructed[:, rep_angle_idx]
        residual = np.sqrt(np.mean((original - reconstructed) ** 2))
        ax_e.plot(freqs / 1000, reconstructed, ls, linewidth=0.8,
                  alpha=alpha_val, label=f"r={rank} (RMSE={residual:.2f})")

    ax_e.plot(freqs / 1000, original, "-", linewidth=1.0,
              color="black", alpha=0.9, label="Original")
    ax_e.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_e.set_ylabel("Log amplitude (centered)", fontsize=6)
    ax_e.set_title(f"Reconstruction @ {angles_deg[rep_angle_idx]:.0f}\u00b0", fontsize=6.5)
    ax_e.legend(fontsize=5, frameon=False, loc="upper right")
    ax_e.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_e, "e", x=-0.15)

    # --- Panel (f): Inter-angle correlation matrix ---
    ax_f = fig.add_subplot(gs[1, 2])
    H_corr = np.corrcoef(np.abs(H_np).T)  # E x E correlation
    im_f = ax_f.imshow(H_corr, cmap="RdBu_r", aspect="auto",
                        vmin=-1.0, vmax=1.0)
    tick_pos = [0, 9, 18, 27, 36]
    tick_lab = [f"{int(angles_deg[i])}" for i in tick_pos if i < len(angles_deg)]
    ax_f.set_xticks(tick_pos[:len(tick_lab)])
    ax_f.set_xticklabels(tick_lab, fontsize=5)
    ax_f.set_yticks(tick_pos[:len(tick_lab)])
    ax_f.set_yticklabels(tick_lab, fontsize=5)
    ax_f.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_f.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_f.set_title("Inter-angle correlation", fontsize=6.5)
    cbar = plt.colorbar(im_f, ax=ax_f, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    add_panel_label(ax_f, "f", x=-0.15)

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
        np.abs(H_np).T, aspect="auto", origin="lower", cmap="viridis",
        extent=[freqs[0] / 1000, freqs[-1] / 1000, angles_deg[0], angles_deg[-1]],
    )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Amplitude", fontsize=6)
    add_panel_label(ax, "d")
    fig_d_s.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d_s, panel_dir / "fig02_panel_d_full_H"))
    plt.close(fig_d_s)

    # Panel e standalone
    fig_e_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_e_s.add_subplot(111)
    ax.plot(freqs / 1000, original, "-", linewidth=1.0, color="black", label="Original")
    for rank, ls, alpha_val in [(3, "--", 0.7), (5, "-.", 0.5), (10, ":", 0.4)]:
        H_reconstructed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
        reconstructed = H_reconstructed[:, rep_angle_idx]
        residual = np.sqrt(np.mean((original - reconstructed) ** 2))
        ax.plot(freqs / 1000, reconstructed, ls, linewidth=0.8,
                alpha=alpha_val, label=f"r={rank} (RMSE={residual:.2f})")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Log amplitude (centered)")
    ax.legend(fontsize=5, frameon=False)
    add_panel_label(ax, "e")
    fig_e_s.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e_s, panel_dir / "fig02_panel_e_reconstruction"))
    plt.close(fig_e_s)

    # Panel f standalone
    fig_f_s = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_f_s.add_subplot(111)
    im = ax.imshow(H_corr, cmap="RdBu_r", aspect="equal", vmin=-1.0, vmax=1.0)
    ax.set_xticks(tick_pos[:len(tick_lab)])
    ax.set_xticklabels(tick_lab, fontsize=6)
    ax.set_yticks(tick_pos[:len(tick_lab)])
    ax.set_yticklabels(tick_lab, fontsize=6)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=6)
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
                "description": "SVD spectrum with cumulative energy and DOA capacity.",
            },
            {
                "panel_id": "b",
                "title": "Overlaid frequency profiles (modes 1-3)",
                "asset_path": "figures/output/fig02_svd_spectrum.pdf",
                "provenance_mode": "data_backed",
                "description": "Frequency-selective spectra |u_r(f)| for modes 1-3 overlaid.",
            },
            {
                "panel_id": "c",
                "title": "Overlaid polar patterns (modes 1-3)",
                "asset_path": "figures/output/fig02_svd_spectrum.pdf",
                "provenance_mode": "data_backed",
                "description": "Direction-selective polar patterns v_r(theta) for modes 1-3 overlaid.",
            },
            {
                "panel_id": "d",
                "title": "Full dictionary H heatmap",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_d_full_H.pdf",
                "provenance_mode": "data_backed",
                "description": "Complete angle-frequency heatmap of H (37 angles x 346 freq bins).",
            },
            {
                "panel_id": "e",
                "title": "Rank-r reconstruction quality",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_e_reconstruction.pdf",
                "provenance_mode": "data_backed",
                "description": "Original vs reconstructed fingerprint at 90 deg for rank 3, 5, 10.",
            },
            {
                "panel_id": "f",
                "title": "Inter-angle correlation matrix",
                "asset_path": "figures/output/fig02_svd_spectrum_panels/fig02_panel_f_correlation.pdf",
                "provenance_mode": "data_backed",
                "description": "37x37 Pearson correlation matrix showing smooth angle-manifold structure.",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig02] Generated {len(all_paths)} files")
    return all_paths
