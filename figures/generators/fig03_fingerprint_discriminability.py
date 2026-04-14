"""Figure 3 — speech-side compactness, neighborhood broadening, and local separability.

Panel (a): Mirrored compactness (calibration vs speech).
Panel (b): Stimulus-conditioned local-ordering decay.
Panel (c): Neighborhood-coherence map (calibration lower-left, speech upper-right).
Panel (d): Stage-0 grouped-match mass within radius.
Panel (e): Angle-resolved exact first-choice failure.
Panel (f): Speech exact vs local tolerance on the same stage-0 diagnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft as scipy_stft

from figures.layout_contract import contract_version, font_pt, source_layout_spec
from figures.style import (
    DOUBLE_COL_MM,
    SEMANTIC_PALETTE,
    STYLE_COLORS,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


FIG03_LOCAL_RADII_DEG = np.arange(0.0, 31.0, 5.0, dtype=np.float32)
FIG03_LOCAL_TOLERANCE_DEG = 10.0
FIG03_TYPOGRAPHY = {
    "panel_label": font_pt("panel_label"),
    "title": font_pt("title"),
    "axis_label": font_pt("axis_label"),
    "tick_label": font_pt("tick_label"),
    "legend": font_pt("legend"),
    "annotation": font_pt("annotation"),
    "colorbar_tick": font_pt("colorbar_tick"),
    "colorbar_label": font_pt("colorbar_label"),
}


def _load_white_noise_features(
    dataset_root: str | Path,
    *,
    fs: float = 16000.0,
    n_fft: int = 2048,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load white-noise waveforms and reduce them to normalized spectral features."""
    dataset_root = Path(dataset_root)
    angle_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith("angle_")],
        key=lambda d: int(d.name.split("_")[1]),
    )

    features_list: list[np.ndarray] = []
    labels_list: list[int] = []
    angles_deg: list[float] = []

    for angle_idx, adir in enumerate(angle_dirs):
        angle_val = int(adir.name.split("_")[1])
        angles_deg.append(float(angle_val))
        for fpath in sorted(adir.glob("*.npy")):
            wav = np.load(fpath).astype(np.float64)
            freqs, _times, zxx = scipy_stft(
                wav,
                fs=fs,
                window="hann",
                nperseg=n_fft,
                noverlap=n_fft // 2,
                nfft=n_fft,
                detrend="constant",
                return_onesided=True,
                boundary=None,
                padded=False,
            )
            mag = np.abs(zxx)
            mask = (freqs >= freq_min) & (freqs <= freq_max)
            feat = mag[mask].mean(axis=1).astype(np.float32)
            norm = np.linalg.norm(feat)
            if norm > 1e-12:
                feat = feat / norm
            features_list.append(feat)
            labels_list.append(angle_idx)

    return (
        np.stack(features_list).astype(np.float32),
        np.asarray(labels_list, dtype=np.int64),
        np.asarray(angles_deg, dtype=np.float32),
    )


def _centered_abs_dictionary(H: np.ndarray) -> np.ndarray:
    arr = np.abs(np.asarray(H, dtype=np.float64))
    return (arr - arr.mean(axis=1, keepdims=True)).astype(np.float32)


def _angle_mean_centered_representation(
    Y: np.ndarray,
    labels: np.ndarray,
    n_angles: int,
) -> np.ndarray:
    """Angle-mean centered magnitude summary used for speech-side neighborhood plots."""
    means = np.zeros((Y.shape[1], n_angles), dtype=np.float32)
    for angle_idx in range(n_angles):
        mask = labels == angle_idx
        if not np.any(mask):
            continue
        means[:, angle_idx] = np.mean(np.abs(Y[mask]), axis=0).astype(np.float32)
    means = means - means.mean(axis=1, keepdims=True)
    return means.astype(np.float32)


def _cumulative_energy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, svals, _ = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    energy = np.cumsum(svals**2)
    energy /= max(float(energy[-1]), 1e-12)
    ranks = np.arange(1, len(svals) + 1, dtype=np.int32)
    return ranks, energy.astype(np.float32)


def _corrcoef_columns(matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(np.asarray(matrix, dtype=np.float64).T).astype(np.float32)


def _local_ordering_decay(
    similarity: np.ndarray,
    angle_step_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    n_angles = int(similarity.shape[0])
    seps_deg = np.arange(1, n_angles, dtype=np.float32) * float(angle_step_deg)
    means = np.zeros_like(seps_deg, dtype=np.float32)
    sems = np.zeros_like(seps_deg, dtype=np.float32)
    first_nonpositive: float | None = None

    for lag in range(1, n_angles):
        vals = np.asarray([similarity[idx, idx + lag] for idx in range(n_angles - lag)], dtype=np.float32)
        means[lag - 1] = float(np.mean(vals))
        if vals.size > 1:
            sems[lag - 1] = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        if first_nonpositive is None and means[lag - 1] <= 0.0:
            first_nonpositive = float(seps_deg[lag - 1])

    return seps_deg, means, sems, first_nonpositive


def _grouped_match_surface(
    Y: np.ndarray,
    D: np.ndarray,
    n_experts: int,
) -> np.ndarray:
    n_atoms = int(D.shape[1] // n_experts)
    grouped = np.abs(np.asarray(Y, dtype=np.float32) @ np.asarray(D, dtype=np.float32))
    return grouped.reshape(len(Y), n_experts, n_atoms).sum(axis=2).astype(np.float32)


def _radius_mass_curves(
    grouped_match: np.ndarray,
    labels: np.ndarray,
    angle_step_deg: float,
    radii_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probs = grouped_match / np.maximum(grouped_match.sum(axis=1, keepdims=True), 1e-8)
    expert_idx = np.arange(grouped_match.shape[1], dtype=np.float32)
    curves = np.zeros((grouped_match.shape[0], len(radii_deg)), dtype=np.float32)
    for clip_idx, label_idx in enumerate(labels.astype(int)):
        distance = np.abs(expert_idx - float(label_idx)) * float(angle_step_deg)
        for ridx, radius in enumerate(radii_deg):
            curves[clip_idx, ridx] = float(probs[clip_idx, distance <= radius].sum())
    means = curves.mean(axis=0, dtype=np.float64).astype(np.float32)
    sems = np.zeros_like(means)
    if curves.shape[0] > 1:
        sems = (curves.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(curves.shape[0])).astype(np.float32)
    return means, sems


def _per_angle_accuracy(
    grouped_match: np.ndarray,
    labels: np.ndarray,
    n_angles: int,
    *,
    tolerance_deg: float,
    angle_step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.argmax(grouped_match, axis=1)
    angle_error = np.abs(pred - labels.astype(int)) * float(angle_step_deg)
    acc = np.zeros(n_angles, dtype=np.float32)
    sem = np.zeros(n_angles, dtype=np.float32)
    for angle_idx in range(n_angles):
        mask = labels == angle_idx
        if not np.any(mask):
            continue
        correct = (angle_error[mask] <= tolerance_deg + 1e-9).astype(np.float32)
        acc[angle_idx] = float(np.mean(correct))
        if correct.size > 1:
            sem[angle_idx] = float(np.std(correct, ddof=1) / np.sqrt(correct.size))
    return acc, sem


def _build_split_similarity_matrix(lower_left: np.ndarray, upper_right: np.ndarray) -> np.ndarray:
    split = np.full(lower_left.shape, np.nan, dtype=np.float32)
    lower = np.tril_indices_from(split, k=-1)
    upper = np.triu_indices_from(split, k=1)
    split[lower] = lower_left[lower]
    split[upper] = upper_right[upper]
    return split


def _positive_similarity_cmap():
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    return cmap


def _fill_curve(ax: plt.Axes, x: np.ndarray, mean: np.ndarray, sem: np.ndarray, *, color: str, alpha: float) -> None:
    ax.fill_between(
        x,
        mean - sem,
        mean + sem,
        color=color,
        alpha=alpha,
        linewidth=0.0,
        zorder=1,
    )


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict]) -> Path:
    manifest_path = panel_dir / "fig03_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig03",
        "composite_asset": "figures/output/fig03_fingerprint_discriminability.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": FIG03_TYPOGRAPHY,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 3 paper-facing panels."""
    set_nature_rcparams(base_fontsize=int(round(FIG03_TYPOGRAPHY["title"])))
    paths = load_paths()
    run_dir = data_root / paths["primary_run"]
    dictionary = np.load(run_dir / "dictionary.npz", allow_pickle=True)
    modal = np.load(run_dir / "modal_routing_val.npz", allow_pickle=True)

    H = np.asarray(dictionary["H"], dtype=np.float32)
    D = np.asarray(dictionary["D"], dtype=np.float32)
    angles = np.asarray(dictionary["angles"], dtype=np.float32)
    labels_sp = np.asarray(modal["labels"], dtype=np.int64)
    Y_sp = np.asarray(modal["Y_val"], dtype=np.float32)

    Y_wn, labels_wn, angles_wn = _load_white_noise_features(paths["white_noise_dataset"])
    if not np.allclose(angles, angles_wn):
        raise ValueError("White-noise angle grid does not match the active dictionary angle grid")

    n_angles = int(len(angles))
    angle_step_deg = float(np.median(np.diff(angles)))

    H_fig = _centered_abs_dictionary(H)
    Y_sp_fig = _angle_mean_centered_representation(Y_sp, labels_sp, n_angles)

    ranks_h, cum_h = _cumulative_energy(H_fig)
    ranks_sp, cum_sp = _cumulative_energy(Y_sp_fig)
    r80_h = int(np.searchsorted(cum_h, 0.80) + 1)
    r80_sp = int(np.searchsorted(cum_sp, 0.80) + 1)

    S_h = _corrcoef_columns(H_fig)
    S_sp = _corrcoef_columns(Y_sp_fig)
    decay_x_h, decay_mean_h, decay_sem_h, zero_h = _local_ordering_decay(S_h, angle_step_deg)
    decay_x_sp, decay_mean_sp, decay_sem_sp, zero_sp = _local_ordering_decay(S_sp, angle_step_deg)

    G_wn = _grouped_match_surface(Y_wn, D, n_angles)
    G_sp = _grouped_match_surface(Y_sp, D, n_angles)
    local_mass_wn, local_mass_sem_wn = _radius_mass_curves(G_wn, labels_wn, angle_step_deg, FIG03_LOCAL_RADII_DEG)
    local_mass_sp, local_mass_sem_sp = _radius_mass_curves(G_sp, labels_sp, angle_step_deg, FIG03_LOCAL_RADII_DEG)

    exact_wn, exact_sem_wn = _per_angle_accuracy(G_wn, labels_wn, n_angles, tolerance_deg=0.0, angle_step_deg=angle_step_deg)
    exact_sp, exact_sem_sp = _per_angle_accuracy(G_sp, labels_sp, n_angles, tolerance_deg=0.0, angle_step_deg=angle_step_deg)
    local_sp, local_sem_sp = _per_angle_accuracy(
        G_sp,
        labels_sp,
        n_angles,
        tolerance_deg=FIG03_LOCAL_TOLERANCE_DEG,
        angle_step_deg=angle_step_deg,
    )

    split_matrix = _build_split_similarity_matrix(S_h, S_sp)
    sim_vmin = float(min(np.nanmin(S_h), np.nanmin(S_sp)))
    sim_cmap = _positive_similarity_cmap()

    output_dir.mkdir(parents=True, exist_ok=True)
    all_paths: list[Path] = []
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=130.0)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        left=0.07,
        right=0.96,
        bottom=0.07,
        top=0.95,
        hspace=0.35,
        wspace=0.40,
    )

    # Panel a
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(ranks_h, cum_h, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=2.4, label="Calibration")
    ax_a.plot(ranks_sp, cum_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=2.4, label="Speech")
    ax_a.axhline(0.80, color=STYLE_COLORS["chance_line"], linestyle="--", linewidth=0.6, alpha=0.8)
    ax_a.axvline(r80_h, color=SEMANTIC_PALETTE["physics"], linestyle=":", linewidth=0.7, alpha=0.8)
    ax_a.axvline(r80_sp, color=SEMANTIC_PALETTE["ablation"], linestyle=":", linewidth=0.7, alpha=0.8)
    ax_a.text(r80_h + 0.2, 0.825, f"r80={r80_h}", fontsize=FIG03_TYPOGRAPHY["annotation"], color=SEMANTIC_PALETTE["physics"])
    ax_a.text(r80_sp + 0.2, 0.745, f"r80={r80_sp}", fontsize=FIG03_TYPOGRAPHY["annotation"], color=SEMANTIC_PALETTE["ablation"])
    ax_a.set_xlim(1, min(16, len(ranks_h)))
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel("Component index r", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_a.set_ylabel("Cum. energy fraction", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_a.set_title("Mirrored compactness", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_a.grid(axis="y", linestyle="--", alpha=0.3)
    ax_a.tick_params(labelsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_a.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="lower right")
    add_panel_label(ax_a, "a", x=-0.14, y=1.06)

    # Panel b
    ax_b = fig.add_subplot(gs[0, 1])
    _fill_curve(ax_b, decay_x_h, decay_mean_h, decay_sem_h, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax_b, decay_x_sp, decay_mean_sp, decay_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax_b.plot(decay_x_h, decay_mean_h, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=2.4, label="Calibration")
    ax_b.plot(decay_x_sp, decay_mean_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=2.4, label="Speech")
    ax_b.axhline(0.0, color=STYLE_COLORS["chance_line"], linestyle="--", linewidth=0.6, alpha=0.8)
    ax_b.axvspan(0.0, 15.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.12, linewidth=0.0)
    if zero_h is not None:
        idx_h = int(np.argmin(np.abs(decay_x_h - zero_h)))
        ax_b.plot(decay_x_h[idx_h], decay_mean_h[idx_h], marker="o", markersize=5.0, markerfacecolor="white", markeredgecolor=SEMANTIC_PALETTE["physics"])
    if zero_sp is not None:
        idx_sp = int(np.argmin(np.abs(decay_x_sp - zero_sp)))
        ax_b.plot(decay_x_sp[idx_sp], decay_mean_sp[idx_sp], marker="o", markersize=5.0, markerfacecolor="white", markeredgecolor=SEMANTIC_PALETTE["ablation"])
    ax_b.set_xlim(5.0, 90.0)
    ax_b.set_xlabel("Angular separation |Δθ| (deg)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_b.set_ylabel("Mean centered corr.", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_b.set_title("Speech-side local-ordering decay", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_b.grid(axis="y", linestyle="--", alpha=0.3)
    ax_b.tick_params(labelsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_b.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax_b, "b", x=-0.14, y=1.06)

    # Panel c
    ax_c = fig.add_subplot(gs[0, 2])
    im_c = ax_c.imshow(split_matrix, cmap=sim_cmap, aspect="auto", vmin=sim_vmin, vmax=1.0)
    ax_c.plot([-0.5, n_angles - 0.5], [-0.5, n_angles - 0.5], color=STYLE_COLORS["neutral_text"], linewidth=0.8, alpha=0.7)
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]
    ax_c.set_xticks(tick_positions)
    ax_c.set_xticklabels(tick_labels, fontsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_c.set_yticks(tick_positions)
    ax_c.set_yticklabels(tick_labels, fontsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_c.set_xlabel("Angle (°)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_c.set_ylabel("Angle (°)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_c.set_title("Neighborhood-coherence map", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_c.text(n_angles * 0.76, n_angles * 0.24, "Speech", ha="center", va="center", fontsize=FIG03_TYPOGRAPHY["annotation"], fontstyle="italic", color=STYLE_COLORS["neutral_text"], alpha=0.7)
    ax_c.text(n_angles * 0.24, n_angles * 0.76, "Calib", ha="center", va="center", fontsize=FIG03_TYPOGRAPHY["annotation"], fontstyle="italic", color=STYLE_COLORS["neutral_text"], alpha=0.7)
    cbar = plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=FIG03_TYPOGRAPHY["colorbar_tick"])
    cbar.set_label("Pearson r", fontsize=FIG03_TYPOGRAPHY["colorbar_label"])
    add_panel_label(ax_c, "c", x=-0.14, y=1.06)

    # Panel d
    ax_d = fig.add_subplot(gs[1, 0])
    _fill_curve(ax_d, FIG03_LOCAL_RADII_DEG, local_mass_wn, local_mass_sem_wn, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax_d, FIG03_LOCAL_RADII_DEG, local_mass_sp, local_mass_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax_d.plot(FIG03_LOCAL_RADII_DEG, local_mass_wn, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=2.8, label="Calibration")
    ax_d.plot(FIG03_LOCAL_RADII_DEG, local_mass_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=2.8, label="Speech")
    ax_d.axvspan(0.0, 15.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.12, linewidth=0.0)
    ax_d.set_xlim(float(FIG03_LOCAL_RADII_DEG[0]), float(FIG03_LOCAL_RADII_DEG[-1]))
    ax_d.set_ylim(0.0, 1.02)
    ax_d.set_xlabel("Neighborhood radius (°)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_d.set_ylabel("Stage-0 mass within radius", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_d.set_title("Stage-0 local separability", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    ax_d.tick_params(labelsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_d.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="lower right")
    add_panel_label(ax_d, "d", x=-0.14, y=1.06)

    # Panel e
    ax_e = fig.add_subplot(gs[1, 1])
    _fill_curve(ax_e, angles, exact_wn, exact_sem_wn, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax_e, angles, exact_sp, exact_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax_e.plot(angles, exact_wn, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=0.95, markersize=2.2, label=f"Calibration ({np.mean(exact_wn):.1%})")
    ax_e.plot(angles, exact_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=0.95, markersize=2.2, label=f"Speech ({np.mean(exact_sp):.1%})")
    ax_e.set_xlim(float(angles[0]), float(angles[-1]))
    ax_e.set_ylim(-0.02, 1.02)
    ax_e.set_xticks([0, 45, 90, 135, 180])
    ax_e.set_xlabel("Angle (°)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_e.set_ylabel("Exact Top-1 rate", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_e.set_title("Exact first-choice collapse", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_e.grid(axis="y", linestyle="--", alpha=0.3)
    ax_e.tick_params(labelsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_e.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax_e, "e", x=-0.14, y=1.06)

    # Panel f
    ax_f = fig.add_subplot(gs[1, 2])
    _fill_curve(ax_f, angles, exact_sp, exact_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.10)
    _fill_curve(ax_f, angles, local_sp, local_sem_sp, color=SEMANTIC_PALETTE["learned"], alpha=0.12)
    ax_f.plot(angles, exact_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=0.95, markersize=2.2, label=f"Exact ({np.mean(exact_sp):.1%})")
    ax_f.plot(angles, local_sp, "-^", color=SEMANTIC_PALETTE["learned"], linewidth=0.95, markersize=2.4, label=f"Within {int(FIG03_LOCAL_TOLERANCE_DEG)}° ({np.mean(local_sp):.1%})")
    ax_f.set_xlim(float(angles[0]), float(angles[-1]))
    ax_f.set_ylim(-0.02, 1.02)
    ax_f.set_xticks([0, 45, 90, 135, 180])
    ax_f.set_xlabel("Angle (°)", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_f.set_ylabel("Speech match rate", fontsize=FIG03_TYPOGRAPHY["axis_label"])
    ax_f.set_title("Speech exact vs local tolerance", fontsize=FIG03_TYPOGRAPHY["title"])
    ax_f.grid(axis="y", linestyle="--", alpha=0.3)
    ax_f.tick_params(labelsize=FIG03_TYPOGRAPHY["tick_label"])
    ax_f.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax_f, "f", x=-0.14, y=1.06)

    all_paths.extend(save_outputs(fig, output_dir / "fig03_fingerprint_discriminability"))
    plt.close(fig)

    panel_dir = output_dir / "fig03_fingerprint_discriminability_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Standalone a
    fig_a = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_a.add_subplot(111)
    ax.plot(ranks_h, cum_h, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=3.0, label="Calibration")
    ax.plot(ranks_sp, cum_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=3.0, label="Speech")
    ax.axhline(0.80, color=STYLE_COLORS["chance_line"], linestyle="--", linewidth=0.6, alpha=0.8)
    ax.axvline(r80_h, color=SEMANTIC_PALETTE["physics"], linestyle=":", linewidth=0.7, alpha=0.8)
    ax.axvline(r80_sp, color=SEMANTIC_PALETTE["ablation"], linestyle=":", linewidth=0.7, alpha=0.8)
    ax.set_xlim(1, min(16, len(ranks_h)))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Component index r")
    ax.set_ylabel("Cum. energy fraction")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="lower right")
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig03_panel_a_mirrored_compactness"))
    plt.close(fig_a)

    # Standalone b
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_b.add_subplot(111)
    _fill_curve(ax, decay_x_h, decay_mean_h, decay_sem_h, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax, decay_x_sp, decay_mean_sp, decay_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax.plot(decay_x_h, decay_mean_h, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=3.0, label="Calibration")
    ax.plot(decay_x_sp, decay_mean_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=3.0, label="Speech")
    ax.axhline(0.0, color=STYLE_COLORS["chance_line"], linestyle="--", linewidth=0.6, alpha=0.8)
    ax.axvspan(0.0, 15.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.12, linewidth=0.0)
    ax.set_xlim(5.0, 90.0)
    ax.set_xlabel("Angular separation |Δθ| (deg)")
    ax.set_ylabel("Mean centered corr.")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig03_panel_b_local_ordering_decay"))
    plt.close(fig_b)

    # Standalone c
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_c.add_subplot(111)
    im = ax.imshow(split_matrix, cmap=sim_cmap, aspect="auto", vmin=sim_vmin, vmax=1.0)
    ax.plot([-0.5, n_angles - 0.5], [-0.5, n_angles - 0.5], color=STYLE_COLORS["neutral_text"], linewidth=0.8, alpha=0.7)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Angle (°)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Pearson r", fontsize=FIG03_TYPOGRAPHY["colorbar_label"])
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig03_panel_c_neighborhood_coherence"))
    plt.close(fig_c)

    # Standalone d
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_d.add_subplot(111)
    _fill_curve(ax, FIG03_LOCAL_RADII_DEG, local_mass_wn, local_mass_sem_wn, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax, FIG03_LOCAL_RADII_DEG, local_mass_sp, local_mass_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax.plot(FIG03_LOCAL_RADII_DEG, local_mass_wn, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=1.0, markersize=3.0, label="Calibration")
    ax.plot(FIG03_LOCAL_RADII_DEG, local_mass_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=1.0, markersize=3.0, label="Speech")
    ax.axvspan(0.0, 15.0, color=SEMANTIC_PALETTE["highlight"], alpha=0.12, linewidth=0.0)
    ax.set_xlim(float(FIG03_LOCAL_RADII_DEG[0]), float(FIG03_LOCAL_RADII_DEG[-1]))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Neighborhood radius (°)")
    ax.set_ylabel("Stage-0 mass within radius")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="lower right")
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig03_panel_d_local_separability"))
    plt.close(fig_d)

    # Standalone e
    fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_e.add_subplot(111)
    _fill_curve(ax, angles, exact_wn, exact_sem_wn, color=SEMANTIC_PALETTE["physics"], alpha=0.10)
    _fill_curve(ax, angles, exact_sp, exact_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.12)
    ax.plot(angles, exact_wn, "-o", color=SEMANTIC_PALETTE["physics"], linewidth=0.95, markersize=2.6, label="Calibration")
    ax.plot(angles, exact_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=0.95, markersize=2.6, label="Speech")
    ax.set_xlim(float(angles[0]), float(angles[-1]))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Exact Top-1 rate")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig03_panel_e_exact_first_choice"))
    plt.close(fig_e)

    # Standalone f
    fig_f = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80.0)
    ax = fig_f.add_subplot(111)
    _fill_curve(ax, angles, exact_sp, exact_sem_sp, color=SEMANTIC_PALETTE["ablation"], alpha=0.10)
    _fill_curve(ax, angles, local_sp, local_sem_sp, color=SEMANTIC_PALETTE["learned"], alpha=0.12)
    ax.plot(angles, exact_sp, "-s", color=SEMANTIC_PALETTE["ablation"], linewidth=0.95, markersize=2.6, label="Exact")
    ax.plot(angles, local_sp, "-^", color=SEMANTIC_PALETTE["learned"], linewidth=0.95, markersize=2.6, label=f"Within {int(FIG03_LOCAL_TOLERANCE_DEG)}°")
    ax.set_xlim(float(angles[0]), float(angles[-1]))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Speech match rate")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=FIG03_TYPOGRAPHY["legend"], loc="upper right")
    add_panel_label(ax, "f")
    fig_f.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_f, panel_dir / "fig03_panel_f_local_tolerance"))
    plt.close(fig_f)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Mirrored compactness",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_a_mirrored_compactness.pdf",
                "provenance_mode": "data_backed",
                "description": "Cumulative energy saturation for the calibration centered dictionary and the speech angle-mean centered summary, showing that compact shared structure survives speech.",
            },
            {
                "panel_id": "b",
                "title": "Speech-side local-ordering decay",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_b_local_ordering_decay.pdf",
                "provenance_mode": "data_backed",
                "description": "Mean centered correlation versus angular separation for calibration and speech, showing that the finite neighborhood survives but broadens under speech.",
            },
            {
                "panel_id": "c",
                "title": "Neighborhood-coherence map",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_c_neighborhood_coherence.pdf",
                "provenance_mode": "data_backed",
                "description": "Split-triangle angle-angle similarity map with calibration in the lower-left and speech in the upper-right, showing retained coarse ordering under speech.",
            },
            {
                "panel_id": "d",
                "title": "Stage-0 local separability",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_d_local_separability.pdf",
                "provenance_mode": "data_backed",
                "description": "Mean grouped-match mass within neighborhood radius for calibration and speech on the frozen stage-0 diagnostic surface.",
            },
            {
                "panel_id": "e",
                "title": "Exact first-choice collapse",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_e_exact_first_choice.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle-resolved exact stage-0 first-choice rate for calibration and speech, showing collapse of immediate one-angle commitment under speech.",
            },
            {
                "panel_id": "f",
                "title": "Speech exact vs local tolerance",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_f_local_tolerance.pdf",
                "provenance_mode": "data_backed",
                "description": "Speech stage-0 exact versus within-10-degree match rates over angle, showing that local support remains even when exact commitment fails.",
            },
        ],
    )
    all_paths.append(manifest)
    print(f"[fig03] Generated {len(all_paths)} files")
    return all_paths

