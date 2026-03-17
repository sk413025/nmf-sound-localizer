"""Figure 1 — Paradigm + First Data Evidence.

Panels (a) and (b) are external JPG assets (setup photo + conceptual schematic).
This generator produces the data panels tied to panel (b)'s physical principle:
  (c) Real time-domain WN waveforms at 5 representative angles
  (d) Spectral fingerprints H(theta,f) for the same 5 angles (smoothed)
  (e) Directivity polar plot H(theta) at 3 representative frequency bands

All panels demonstrate the direction-dependent transfer function H(theta,f)
predicted by the physical model in panel (b).
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
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_freq_axis(F: int, fs: float = 16000.0, n_fft: float = 2048.0,
                     f_min: float = 300.0) -> np.ndarray:
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    return (k_start + np.arange(F)) * df


def _smooth_spectrum(y: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay filter for smooth spectral curves."""
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig01_panel_manifest.json"
    payload = {
        "figure_id": "fig01",
        "composite_asset": "figures/output/fig01_paradigm_data.pdf",
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
    """Generate Figure 1 data panels (c, d, e)."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()

    # Use 37-angle H from dictionary.npz (0-180 deg, 5 deg steps)
    run_dir = data_root / paths_cfg["primary_run"]
    dict_path = run_dir / "dictionary.npz"

    if not dict_path.exists():
        print(f"[fig01] SKIP: dictionary.npz not found at {dict_path}")
        return []

    dict_data = dict(np.load(dict_path, allow_pickle=True))
    H_np = dict_data["H"]         # (346, 37)
    angles_deg = dict_data["angles"]  # (37,)
    F, E = H_np.shape
    freqs = _build_freq_axis(F)

    # 5 representative angles for panels (c) and (d)
    representative = [0, 45, 90, 135, 180]
    angle_indices = []
    for target in representative:
        idx = int(np.argmin(np.abs(angles_deg - target)))
        angle_indices.append(idx)

    colors_5 = ["#0072B2", "#56B4E9", "#D55E00", "#E69F00", "#009E73"]
    angle_labels = [f"{angles_deg[i]:.0f}\u00b0" for i in angle_indices]

    # Load real WN waveforms for panel (c), bandpassed to H's physical freq range
    from scipy.signal import butter, sosfiltfilt

    wn_dataset = paths_cfg.get("white_noise_dataset", "")
    wn_base = Path(wn_dataset)
    real_waveforms: list[tuple[np.ndarray, np.ndarray]] = []
    source_fs = 48000.0
    display_duration_ms = 5.0
    display_samples = int(source_fs * display_duration_ms / 1000)

    # Bandpass to H's effective physical frequency range (900-9000 Hz).
    bp_lo = 900.0
    bp_hi = min(9000.0, source_fs / 2 - 1)
    sos_bp = butter(4, [bp_lo, bp_hi], btype="bandpass", fs=source_fs, output="sos")

    for aidx in angle_indices:
        angle_val = int(angles_deg[aidx])
        clip_path = wn_base / f"angle_{angle_val}" / "clip_000.npy"
        if clip_path.exists():
            wav = np.load(clip_path).astype(np.float64)
            wav_bp = sosfiltfilt(sos_bp, wav)
            start = len(wav_bp) // 4
            segment = wav_bp[start:start + display_samples]
            segment = segment / (np.max(np.abs(segment)) + 1e-10)
            t_ms = np.arange(len(segment)) / source_fs * 1000
            real_waveforms.append((t_ms, segment))
        else:
            real_waveforms.append((None, None))

    has_real_waveforms = all(t is not None for t, _ in real_waveforms)

    # Frequency band intervals for panel (e) directivity
    # Matches the band decomposition used in the analysis pipeline
    band_defs = [
        ("0.3\u20130.5 kHz", 300.0, 500.0, False),
        ("0.5\u20131 kHz",   500.0, 1000.0, False),
        ("1\u20132 kHz",     1000.0, 2000.0, False),
        ("2\u20133 kHz",     2000.0, 3000.0, True),
    ]
    freq_band_masks = []
    freq_labels_e = []
    colors_4e = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for name, lo, hi, inc_upper in band_defs:
        if inc_upper:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        freq_band_masks.append(mask)
        freq_labels_e.append(name)

    # Smoothing parameters
    sg_window = 121  # extremely wide for clean overview envelope
    sg_poly = 2

    # -----------------------------------------------------------------------
    # Build composite figure (panels c, d, e in a row)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=65)
    gs = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.0],
        wspace=0.40, left=0.06, right=0.97, bottom=0.18, top=0.88,
    )

    # --- Panel (c): Real time-domain waveforms (5 angles) ---
    ax_c = fig.add_subplot(gs[0, 0])
    for k, (aidx, color, label) in enumerate(
        zip(angle_indices, colors_5, angle_labels)
    ):
        offset = k * 2.2
        if has_real_waveforms:
            t_ms, wav = real_waveforms[k]
            ax_c.plot(t_ms, wav + offset, color=color, linewidth=0.4,
                      label=label)

    ax_c.set_xlabel("Time (ms)", fontsize=6)
    ax_c.set_ylabel("Amplitude (offset)", fontsize=6)
    ax_c.set_title("Surface vibration", fontsize=6.5)
    ax_c.legend(fontsize=4.5, frameon=False, loc="upper right", ncol=1)
    ax_c.set_yticks([])
    ax_c.grid(axis="x", linestyle="--", alpha=0.3)
    add_panel_label(ax_c, "c", x=-0.10, y=1.06)

    # --- Panel (d): Spectral fingerprints H(theta,f), smoothed (5 angles) ---
    ax_d = fig.add_subplot(gs[0, 1])
    for aidx, color, label in zip(angle_indices, colors_5, angle_labels):
        spectrum = np.abs(H_np[:, aidx])
        spectrum_smooth = _smooth_spectrum(spectrum, sg_window, sg_poly)
        spectrum_norm = spectrum_smooth / (spectrum_smooth.max() + 1e-10)
        ax_d.plot(freqs / 1000, spectrum_norm, color=color, linewidth=0.8,
                  label=label, alpha=0.85)

    ax_d.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_d.set_ylabel("Normalized |H(\u03b8, f)|", fontsize=6)
    ax_d.set_title("Spectral fingerprints", fontsize=6.5)
    ax_d.legend(fontsize=4.5, frameon=False, loc="upper right", ncol=1)
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_d, "d", x=-0.12, y=1.06)

    # --- Panel (e): Directivity polar plot (3 frequency bands, 37 angles) ---
    ax_e = fig.add_subplot(gs[0, 2], polar=True)

    # Convert angles to radians for polar plot
    angles_rad = np.deg2rad(angles_deg)
    angles_fine_rad = np.linspace(angles_rad[0], angles_rad[-1], 300)

    for mask, color, flabel in zip(freq_band_masks, colors_4e, freq_labels_e):
        directivity = np.abs(H_np[mask, :]).mean(axis=0)
        directivity_norm = directivity / (directivity.max() + 1e-10)
        # Savgol smooth over angles, then cubic spline for display
        if len(directivity_norm) >= 9:
            directivity_smooth = savgol_filter(directivity_norm, 9, 3)
            directivity_smooth = np.clip(directivity_smooth, 0, None)
        else:
            directivity_smooth = directivity_norm
        cs = CubicSpline(angles_rad, directivity_smooth)
        ax_e.plot(angles_fine_rad, cs(angles_fine_rad), linewidth=1.0,
                  color=color, label=flabel)

    ax_e.set_theta_zero_location("N")
    ax_e.set_theta_direction(-1)
    ax_e.set_thetalim(0, np.pi)
    ax_e.set_rlabel_position(90)
    ax_e.tick_params(labelsize=5)
    ax_e.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_e.set_title("Directivity", fontsize=6.5, pad=12)
    ax_e.legend(fontsize=4.5, frameon=False, loc="lower left",
                bbox_to_anchor=(-0.20, -0.15))
    add_panel_label(ax_e, "e", x=-0.15, y=1.10)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig01_paradigm_data")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Standalone panels
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig01_paradigm_data_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel c standalone
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_c.add_subplot(111)
    if has_real_waveforms:
        for k, (aidx, color, label) in enumerate(
            zip(angle_indices, colors_5, angle_labels)
        ):
            t_ms, wav = real_waveforms[k]
            ax.plot(t_ms, wav + k * 2.2, color=color, linewidth=0.5, label=label)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (offset)")
    ax.set_yticks([])
    ax.legend(fontsize=6, frameon=False)
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig01_panel_c_waveforms"))
    plt.close(fig_c)

    # Panel d standalone
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_d.add_subplot(111)
    for aidx, color, label in zip(angle_indices, colors_5, angle_labels):
        spectrum = np.abs(H_np[:, aidx])
        spectrum_smooth = _smooth_spectrum(spectrum, sg_window, sg_poly)
        spectrum_norm = spectrum_smooth / (spectrum_smooth.max() + 1e-10)
        ax.plot(freqs / 1000, spectrum_norm, color=color, linewidth=0.9,
                label=label, alpha=0.85)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized |H(\u03b8, f)|")
    ax.legend(fontsize=6, frameon=False)
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig01_panel_d_spectra"))
    plt.close(fig_d)

    # Panel e standalone (polar)
    fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_e.add_subplot(111, polar=True)
    for mask, color, flabel in zip(freq_band_masks, colors_4e, freq_labels_e):
        directivity = np.abs(H_np[mask, :]).mean(axis=0)
        directivity_norm = directivity / (directivity.max() + 1e-10)
        if len(directivity_norm) >= 9:
            directivity_smooth = savgol_filter(directivity_norm, 9, 3)
            directivity_smooth = np.clip(directivity_smooth, 0, None)
        else:
            directivity_smooth = directivity_norm
        cs = CubicSpline(angles_rad, directivity_smooth)
        ax.plot(angles_fine_rad, cs(angles_fine_rad), linewidth=1.0,
                color=color, label=flabel)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, np.pi)
    ax.set_rlabel_position(90)
    ax.legend(fontsize=6, frameon=False, loc="lower right",
              bbox_to_anchor=(1.3, -0.05))
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.92)
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig01_panel_e_directivity"))
    plt.close(fig_e)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "c",
                "title": "Time-domain vibration at 5 angles",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_c_waveforms.pdf",
                "provenance_mode": "data_backed",
                "description": "Real WN waveforms (bandpassed 900-9000 Hz) at 0, 45, 90, 135, 180 degrees.",
            },
            {
                "panel_id": "d",
                "title": "Spectral fingerprints H(theta,f)",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_d_spectra.pdf",
                "provenance_mode": "data_backed",
                "description": "Savgol-smoothed transfer function at 5 angles from dictionary H (37 angles).",
            },
            {
                "panel_id": "e",
                "title": "Directivity polar plot",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_e_directivity.pdf",
                "provenance_mode": "data_backed",
                "description": "Polar directivity H(theta) at 0.7, 1.5, 2.5 kHz bands (37 angles, cubic spline).",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig01] Generated {len(all_paths)} files "
          f"(H={H_np.shape}, angles={len(angles_deg)}, "
          f"real_waveforms={'yes' if has_real_waveforms else 'no'})")
    return all_paths
