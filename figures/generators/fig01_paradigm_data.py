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

    # Load real WN spectra for panel (c): source (original) vs output (box)
    from scipy.signal import stft as scipy_stft

    wn_dataset = paths_cfg.get("white_noise_dataset", "")
    wn_base = Path(wn_dataset)
    # Source (original) WN — same dir structure but "original" instead of "box"
    src_base = Path(str(wn_base).replace(
        "white_noise_box_snrInf_sync_vad_normalized",
        "white_noise_original_sync_vad"))

    def _compute_mean_spectrum(wav_path: Path) -> np.ndarray | None:
        if not wav_path.exists():
            return None
        wav = np.load(wav_path).astype(np.float64)
        f, _, Zxx = scipy_stft(wav, fs=16000, window="hann", nperseg=2048,
                                noverlap=1024, nfft=2048, detrend="constant",
                                return_onesided=True, boundary=None, padded=False)
        mag = np.abs(Zxx)
        mask = (f >= 300) & (f <= 3000)
        return mag[mask].mean(axis=1)

    # Source spectrum (average over all 3 clips at one angle — should be flat)
    src_spec = _compute_mean_spectrum(src_base / "angle_90" / "clip_000.npy")

    # Output spectra at each representative angle
    output_spectra: list[np.ndarray | None] = []
    for aidx in angle_indices:
        angle_val = int(angles_deg[aidx])
        spec = _compute_mean_spectrum(wn_base / f"angle_{angle_val}" / "clip_000.npy")
        output_spectra.append(spec)

    has_io_data = src_spec is not None and all(s is not None for s in output_spectra)

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

    # --- Panel (c): Input vs Output spectral comparison ---
    ax_c = fig.add_subplot(gs[0, 0])
    if has_io_data:
        # Source spectrum (smoothed, normalized)
        src_smooth = _smooth_spectrum(src_spec, sg_window, sg_poly)
        src_norm = src_smooth / (src_smooth.max() + 1e-10)
        ax_c.plot(freqs / 1000, src_norm, color="gray", linewidth=1.5,
                  alpha=0.7, label="Source (WN)", linestyle="--")

        # Output spectra at each angle (smoothed, normalized to source max)
        for aidx, color, label, spec in zip(
            angle_indices, colors_5, angle_labels, output_spectra
        ):
            spec_smooth = _smooth_spectrum(spec, sg_window, sg_poly)
            spec_norm = spec_smooth / (src_smooth.max() + 1e-10)
            ax_c.plot(freqs / 1000, spec_norm, color=color, linewidth=0.8,
                      label=label, alpha=0.85)

    ax_c.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_c.set_ylabel("Normalized amplitude", fontsize=6)
    ax_c.set_title("Input \u2192 output spectral shaping", fontsize=6.5)
    ax_c.legend(fontsize=4, frameon=False, loc="center right", ncol=1)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_c, "c", x=-0.10, y=1.06)

    # --- Panel (d): Multi-trial repeatability (5 angles, mean ± std band) ---
    ax_d = fig.add_subplot(gs[0, 1])

    for aidx, color, label in zip(angle_indices, colors_5, angle_labels):
        angle_val = int(angles_deg[aidx])
        clip_dir = wn_base / f"angle_{angle_val}"
        clips = sorted(clip_dir.glob("*.npy"))[:3]
        trial_spectra = []
        for clip_path in clips:
            spec = _compute_mean_spectrum(clip_path)
            if spec is not None:
                spec_smooth = _smooth_spectrum(spec, sg_window, sg_poly)
                trial_spectra.append(spec_smooth)
        if trial_spectra:
            stacked = np.stack(trial_spectra)
            # Normalize all trials by the global max of this angle's mean
            mean_spec = stacked.mean(axis=0)
            norm_factor = mean_spec.max() + 1e-10
            mean_norm = mean_spec / norm_factor
            std_norm = stacked.std(axis=0) / norm_factor
            ax_d.plot(freqs / 1000, mean_norm, color=color, linewidth=0.9,
                      label=label)
            ax_d.fill_between(freqs / 1000,
                              mean_norm - std_norm, mean_norm + std_norm,
                              color=color, alpha=0.25)

    ax_d.set_xlabel("Frequency (kHz)", fontsize=6)
    ax_d.set_ylabel("Normalized amplitude", fontsize=6)
    ax_d.set_title("Trial repeatability (\u00b11 s.d.)", fontsize=6.5)
    ax_d.legend(fontsize=5, frameon=False, loc="upper right")
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
    ax_e.legend(fontsize=4.5, frameon=False, loc="upper left",
                bbox_to_anchor=(-0.25, 0.45))
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
    if has_io_data:
        src_smooth = _smooth_spectrum(src_spec, sg_window, sg_poly)
        src_norm = src_smooth / (src_smooth.max() + 1e-10)
        ax.plot(freqs / 1000, src_norm, color="gray", linewidth=1.5,
                alpha=0.7, label="Source (WN)", linestyle="--")
        for aidx, color, label, spec in zip(
            angle_indices, colors_5, angle_labels, output_spectra
        ):
            spec_smooth = _smooth_spectrum(spec, sg_window, sg_poly)
            spec_norm = spec_smooth / (src_smooth.max() + 1e-10)
            ax.plot(freqs / 1000, spec_norm, color=color, linewidth=0.9,
                    label=label, alpha=0.85)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=5, frameon=False, loc="center right", ncol=1)
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig01_panel_c_input_output"))
    plt.close(fig_c)

    # Panel d standalone
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_d.add_subplot(111)
    for aidx, color, label in zip(angle_indices, colors_5, angle_labels):
        angle_val = int(angles_deg[aidx])
        clip_dir = wn_base / f"angle_{angle_val}"
        clips = sorted(clip_dir.glob("*.npy"))[:3]
        trial_spectra = []
        for clip_path in clips:
            spec = _compute_mean_spectrum(clip_path)
            if spec is not None:
                trial_spectra.append(_smooth_spectrum(spec, sg_window, sg_poly))
        if trial_spectra:
            stacked = np.stack(trial_spectra)
            mean_spec = stacked.mean(axis=0)
            norm_factor = mean_spec.max() + 1e-10
            mean_norm = mean_spec / norm_factor
            std_norm = stacked.std(axis=0) / norm_factor
            ax.plot(freqs / 1000, mean_norm, color=color, linewidth=1.0, label=label)
            ax.fill_between(freqs / 1000,
                            mean_norm - std_norm, mean_norm + std_norm,
                            color=color, alpha=0.25)
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=6, frameon=False)
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig01_panel_d_repeatability"))
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
                "title": "Input-output spectral shaping",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_c_input_output.pdf",
                "provenance_mode": "data_backed",
                "description": "Flat WN source vs direction-shaped output spectra at 5 angles, demonstrating structural filtering.",
            },
            {
                "panel_id": "d",
                "title": "Trial repeatability",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_d_repeatability.pdf",
                "provenance_mode": "data_backed",
                "description": "3 trials × 3 angles showing stable fingerprints (within-angle overlap) and distinct profiles (between-angle separation).",
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
          f"io_data={'yes' if has_io_data else 'no'})")
    return all_paths
