"""Figure 1 — phenomenon-first evidence panels under the SM1 schematic.

Panel (a) remains the committed setup photo support asset.
Panel (b) is the legacy conceptual schematic composed separately.
This generator produces the bottom-row data-backed panels:
  (c) input-to-output spectral shaping with overlaid calibration-trial repeatability
  (d) frequency-dependent directivity
  (e) inter-angle fingerprint similarity

Together these panels establish that a fixed LDV point records angle-dependent,
repeatable spectral fingerprints, then show the first local-geometry signal
before Fig. 2 takes over the compactness argument.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from figures.layout_contract import contract_version, figure_section, font_pt, source_layout_spec, stroke_pt
from figures.style import (
    DOUBLE_COL_MM,
    PALETTE_VIRIDIS_8,
    STYLE_COLORS,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


FIG01_TYPOGRAPHY = {
    "panel_label": font_pt("panel_label"),
    "title": font_pt("title"),
    "axis_label": font_pt("axis_label"),
    "tick_label": font_pt("tick_label"),
    "legend": font_pt("legend"),
    "annotation": font_pt("annotation"),
    "colorbar_tick": font_pt("colorbar_tick"),
    "colorbar_label": font_pt("colorbar_label"),
}
FIG01_STROKES = {
    "annotation": stroke_pt("annotation"),
    "base": stroke_pt("base"),
    "data": stroke_pt("data"),
    "emphasis": stroke_pt("emphasis"),
    "grid": stroke_pt("grid"),
}
FIG01_GENERATOR = figure_section("fig01", "generator")
FIG01_COMPOSITE_GRID = dict(FIG01_GENERATOR["composite_grid"])
FIG01_ANGLE_COLORS = [
    PALETTE_VIRIDIS_8[1],
    PALETTE_VIRIDIS_8[2],
    PALETTE_VIRIDIS_8[3],
    PALETTE_VIRIDIS_8[4],
    PALETTE_VIRIDIS_8[5],
]
FIG01_BAND_COLORS = [
    PALETTE_VIRIDIS_8[1],
    PALETTE_VIRIDIS_8[2],
    PALETTE_VIRIDIS_8[4],
    PALETTE_VIRIDIS_8[5],
]


def _build_freq_axis(
    F: int,
    fs: float = 16000.0,
    n_fft: float = 2048.0,
    f_min: float = 300.0,
) -> np.ndarray:
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    return (k_start + np.arange(F)) * df


def _smooth_spectrum(y: np.ndarray, window: int = 31, polyorder: int = 3) -> np.ndarray:
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig01_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig01",
        "composite_asset": "figures/output/fig01_paradigm_data.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": FIG01_TYPOGRAPHY,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 1 data-backed panels (c, d, e)."""
    set_nature_rcparams(base_fontsize=7)

    from scipy.signal import stft as scipy_stft

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]
    dict_path = run_dir / "dictionary.npz"

    if not dict_path.exists():
        print(f"[fig01] SKIP: dictionary.npz not found at {dict_path}")
        return []

    dict_data = dict(np.load(dict_path, allow_pickle=True))
    H_np = np.asarray(dict_data["H"], dtype=float)
    angles_deg = np.asarray(dict_data["angles"], dtype=float)
    F, _E = H_np.shape
    freqs = _build_freq_axis(F)

    representative = [0, 45, 90, 135, 180]
    angle_indices = [int(np.argmin(np.abs(angles_deg - target))) for target in representative]
    angle_labels = [f"{angles_deg[idx]:.0f}°" for idx in angle_indices]
    tick_pos = np.linspace(0, len(angles_deg) - 1, 5, dtype=int).tolist()
    tick_lab = [f"{int(angles_deg[i])}" for i in tick_pos]

    wn_dataset = paths_cfg.get("white_noise_dataset", "")
    wn_base = Path(wn_dataset)
    src_base = Path(
        str(wn_base).replace(
            "white_noise_box_snrInf_sync_vad_normalized",
            "white_noise_original_sync_vad",
        )
    )

    def _compute_mean_spectrum(wav_path: Path) -> np.ndarray | None:
        if not wav_path.exists():
            return None
        wav = np.load(wav_path).astype(np.float64)
        f, _, Zxx = scipy_stft(
            wav,
            fs=16000,
            window="hann",
            nperseg=2048,
            noverlap=1024,
            nfft=2048,
            detrend="constant",
            return_onesided=True,
            boundary=None,
            padded=False,
        )
        mag = np.abs(Zxx)
        mask = (f >= 300) & (f <= 3000)
        return mag[mask].mean(axis=1)

    def _load_trial_spectra(angle_dir: Path, max_clips: int = 3) -> list[np.ndarray]:
        trial_spectra: list[np.ndarray] = []
        for clip_path in sorted(angle_dir.glob("*.npy"))[:max_clips]:
            spec = _compute_mean_spectrum(clip_path)
            if spec is not None:
                trial_spectra.append(_smooth_spectrum(spec, window=121, polyorder=2))
        return trial_spectra

    source_trials = _load_trial_spectra(src_base / "angle_90")
    output_trials_by_angle: list[list[np.ndarray]] = []
    for angle_idx in angle_indices:
        angle_val = int(angles_deg[angle_idx])
        output_trials_by_angle.append(_load_trial_spectra(wn_base / f"angle_{angle_val}"))

    src_spec = np.stack(source_trials).mean(axis=0) if source_trials else None
    has_io_data = src_spec is not None and all(trials for trials in output_trials_by_angle)

    band_defs = [
        ("0.3–0.5 kHz", 300.0, 500.0, False),
        ("0.5–1 kHz", 500.0, 1000.0, False),
        ("1–2 kHz", 1000.0, 2000.0, False),
        ("2–3 kHz", 2000.0, 3000.0, True),
    ]
    freq_band_masks = []
    freq_labels_d = []
    for name, lo, hi, inc_upper in band_defs:
        if inc_upper:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        freq_band_masks.append(mask)
        freq_labels_d.append(name)

    H_corr = np.corrcoef(H_np.T)

    fig = make_figure(
        width_mm=float(FIG01_GENERATOR["composite_width_mm"]),
        height_mm=float(FIG01_GENERATOR["composite_height_mm"]),
    )
    gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        width_ratios=list(FIG01_COMPOSITE_GRID["width_ratios"]),
        wspace=float(FIG01_COMPOSITE_GRID["wspace"]),
        left=float(FIG01_COMPOSITE_GRID["left"]),
        right=float(FIG01_COMPOSITE_GRID["right"]),
        bottom=float(FIG01_COMPOSITE_GRID["bottom"]),
        top=float(FIG01_COMPOSITE_GRID["top"]),
    )

    ax_c = fig.add_subplot(gs[0, 0])
    if has_io_data:
        src_smooth = _smooth_spectrum(src_spec, window=121, polyorder=2)
        src_norm = src_smooth / (src_smooth.max() + 1e-10)
        ax_c.plot(
            freqs / 1000.0,
            src_norm,
            color=STYLE_COLORS["guide_line"],
            linewidth=FIG01_STROKES["emphasis"],
            linestyle="--",
            alpha=0.8,
            label="Source (WN)",
            zorder=2,
        )
        norm_factor = src_smooth.max() + 1e-10
        for color, label, trial_spectra in zip(
            FIG01_ANGLE_COLORS,
            angle_labels,
            output_trials_by_angle,
            strict=True,
        ):
            stacked = np.stack(trial_spectra)
            for trial in stacked:
                ax_c.plot(
                    freqs / 1000.0,
                    trial / norm_factor,
                    color=color,
                    linewidth=FIG01_STROKES["base"] * 0.75,
                    alpha=0.22,
                    zorder=1,
                )
            mean_spec = stacked.mean(axis=0) / norm_factor
            ax_c.plot(
                freqs / 1000.0,
                mean_spec,
                color=color,
                linewidth=FIG01_STROKES["data"],
                alpha=0.95,
                label=label,
                zorder=3,
            )
    ax_c.set_xlabel("Frequency (kHz)", fontsize=FIG01_TYPOGRAPHY["axis_label"])
    ax_c.set_ylabel("Normalized amplitude", fontsize=FIG01_TYPOGRAPHY["axis_label"])
    ax_c.set_title("Spectral shaping with repeatability", fontsize=FIG01_TYPOGRAPHY["title"])
    ax_c.tick_params(axis="both", labelsize=FIG01_TYPOGRAPHY["tick_label"], pad=1)
    ax_c.grid(axis="y", linestyle="--", linewidth=FIG01_STROKES["grid"], alpha=0.3)
    ax_c.legend(fontsize=FIG01_TYPOGRAPHY["legend"], frameon=False, loc="center right", ncol=1)
    ax_c.text(
        0.02,
        0.98,
        "Thin = trials, thick = mean",
        transform=ax_c.transAxes,
        fontsize=FIG01_TYPOGRAPHY["tick_label"],
        ha="left",
        va="top",
        color=STYLE_COLORS["neutral_text"],
    )
    add_panel_label(ax_c, "c", x=-0.12, y=1.06)

    ax_d = fig.add_subplot(gs[0, 1], polar=True)
    angles_rad = np.deg2rad(angles_deg)
    angles_fine_rad = np.linspace(angles_rad[0], angles_rad[-1], 300)
    for mask, color, label in zip(freq_band_masks, FIG01_BAND_COLORS, freq_labels_d, strict=True):
        directivity = np.abs(H_np[mask, :]).mean(axis=0)
        directivity_norm = directivity / (directivity.max() + 1e-10)
        if len(directivity_norm) >= 9:
            directivity_smooth = savgol_filter(directivity_norm, 9, 3)
            directivity_smooth = np.clip(directivity_smooth, 0, None)
        else:
            directivity_smooth = directivity_norm
        cs = CubicSpline(angles_rad, directivity_smooth)
        ax_d.plot(
            angles_fine_rad,
            cs(angles_fine_rad),
            linewidth=FIG01_STROKES["data"],
            color=color,
            label=label,
        )
    ax_d.set_theta_zero_location("N")
    ax_d.set_theta_direction(-1)
    ax_d.set_thetalim(0, np.pi)
    ax_d.set_rlabel_position(90)
    ax_d.tick_params(labelsize=FIG01_TYPOGRAPHY["tick_label"], pad=1)
    ax_d.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_d.set_title("Frequency-dependent directivity", fontsize=FIG01_TYPOGRAPHY["title"], pad=12)
    ax_d.legend(
        fontsize=FIG01_TYPOGRAPHY["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.26, 0.48),
    )
    add_panel_label(ax_d, "d", x=-0.15, y=1.10)

    ax_e = fig.add_subplot(gs[0, 2])
    im_e = ax_e.imshow(H_corr, cmap="viridis", aspect="equal", vmin=float(H_corr.min()), vmax=1.0)
    ax_e.set_xticks(tick_pos)
    ax_e.set_xticklabels(tick_lab, fontsize=FIG01_TYPOGRAPHY["tick_label"])
    ax_e.set_yticks(tick_pos)
    ax_e.set_yticklabels(tick_lab, fontsize=FIG01_TYPOGRAPHY["tick_label"])
    ax_e.set_xlabel("Angle (°)", fontsize=FIG01_TYPOGRAPHY["axis_label"])
    ax_e.set_ylabel("Angle (°)", fontsize=FIG01_TYPOGRAPHY["axis_label"])
    ax_e.set_title("Inter-angle fingerprint similarity", fontsize=FIG01_TYPOGRAPHY["title"])
    cax_e = ax_e.inset_axes([1.04, 0.0, 0.045, 1.0])
    cbar_e = plt.colorbar(im_e, cax=cax_e)
    cbar_e.set_label("Pearson r", fontsize=FIG01_TYPOGRAPHY["colorbar_label"])
    cbar_e.ax.tick_params(labelsize=FIG01_TYPOGRAPHY["colorbar_tick"])
    add_panel_label(ax_e, "e", x=-0.15, y=1.06)

    all_paths = save_outputs(fig, output_dir / "fig01_paradigm_data")
    plt.close(fig)

    panel_dir = output_dir / "fig01_paradigm_data_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    ax = fig_c.add_subplot(111)
    if has_io_data:
        src_smooth = _smooth_spectrum(src_spec, window=121, polyorder=2)
        src_norm = src_smooth / (src_smooth.max() + 1e-10)
        ax.plot(
            freqs / 1000.0,
            src_norm,
            color=STYLE_COLORS["guide_line"],
            linewidth=FIG01_STROKES["emphasis"],
            linestyle="--",
            alpha=0.8,
            label="Source (WN)",
        )
        norm_factor = src_smooth.max() + 1e-10
        for color, label, trial_spectra in zip(
            FIG01_ANGLE_COLORS,
            angle_labels,
            output_trials_by_angle,
            strict=True,
        ):
            stacked = np.stack(trial_spectra)
            for trial in stacked:
                ax.plot(
                    freqs / 1000.0,
                    trial / norm_factor,
                    color=color,
                    linewidth=FIG01_STROKES["base"] * 0.75,
                    alpha=0.22,
                )
            ax.plot(
                freqs / 1000.0,
                stacked.mean(axis=0) / norm_factor,
                color=color,
                linewidth=FIG01_STROKES["data"],
                alpha=0.95,
                label=label,
            )
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Normalized amplitude")
    ax.tick_params(axis="both", labelsize=FIG01_TYPOGRAPHY["tick_label"], pad=1)
    ax.grid(axis="y", linestyle="--", linewidth=FIG01_STROKES["grid"], alpha=0.3)
    ax.legend(fontsize=FIG01_TYPOGRAPHY["legend"], frameon=False, loc="center right", ncol=1)
    ax.text(
        0.02,
        0.98,
        "Thin = trials, thick = mean",
        transform=ax.transAxes,
        fontsize=FIG01_TYPOGRAPHY["tick_label"],
        ha="left",
        va="top",
        color=STYLE_COLORS["neutral_text"],
    )
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig01_panel_c_spectral_shaping_repeatability"))
    plt.close(fig_c)

    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_d.add_subplot(111, polar=True)
    for mask, color, label in zip(freq_band_masks, FIG01_BAND_COLORS, freq_labels_d, strict=True):
        directivity = np.abs(H_np[mask, :]).mean(axis=0)
        directivity_norm = directivity / (directivity.max() + 1e-10)
        if len(directivity_norm) >= 9:
            directivity_smooth = savgol_filter(directivity_norm, 9, 3)
            directivity_smooth = np.clip(directivity_smooth, 0, None)
        else:
            directivity_smooth = directivity_norm
        cs = CubicSpline(angles_rad, directivity_smooth)
        ax.plot(
            angles_fine_rad,
            cs(angles_fine_rad),
            linewidth=FIG01_STROKES["data"],
            color=color,
            label=label,
        )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetalim(0, np.pi)
    ax.set_rlabel_position(90)
    ax.legend(
        fontsize=FIG01_TYPOGRAPHY["legend"],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(-0.26, 0.48),
    )
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.05, right=0.85, bottom=0.05, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig01_panel_d_directivity"))
    plt.close(fig_d)

    fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_e.add_subplot(111)
    im = ax.imshow(H_corr, cmap="viridis", aspect="equal", vmin=float(H_corr.min()), vmax=1.0)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=FIG01_TYPOGRAPHY["tick_label"])
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=FIG01_TYPOGRAPHY["tick_label"])
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Angle (°)")
    cax = ax.inset_axes([1.04, 0.0, 0.045, 1.0])
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Pearson r", fontsize=FIG01_TYPOGRAPHY["colorbar_label"])
    cbar.ax.tick_params(labelsize=FIG01_TYPOGRAPHY["colorbar_tick"])
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig01_panel_e_similarity"))
    plt.close(fig_e)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "c",
                "title": "Spectral shaping with repeatability",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_c_spectral_shaping_repeatability.pdf",
                "provenance_mode": "data_backed",
                "description": "Flat white-noise source baseline plus five representative angle bundles, with thin trial traces and thick means showing both spectral reshaping and calibration repeatability.",
            },
            {
                "panel_id": "d",
                "title": "Frequency-dependent directivity",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_d_directivity.pdf",
                "provenance_mode": "data_backed",
                "description": "Polar directivity across the 0°–180° half-plane for four frequency bands spanning 0.3–3 kHz.",
            },
            {
                "panel_id": "e",
                "title": "Inter-angle fingerprint similarity",
                "asset_path": "figures/output/fig01_paradigm_data_panels/fig01_panel_e_similarity.pdf",
                "provenance_mode": "data_backed",
                "description": "Inter-angle similarity matrix of the angle-indexed prototype fingerprints, showing a near-diagonal local-geometry band.",
            },
        ],
    )
    all_paths.append(manifest)

    print(
        f"[fig01] Generated {len(all_paths)} files "
        f"(H={H_np.shape}, angles={len(angles_deg)}, io_data={'yes' if has_io_data else 'no'})"
    )
    return all_paths
