"""Figure 3 — White Noise vs Speech Discriminability (6 panels).

Panel (a): White noise within vs between Pearson r (violin + stats).
Panel (b): Speech within vs between Pearson r (violin + stats).
Panel (c): Per-angle repeatability comparison (WN vs Speech lines +/- SEM).
Panel (d): OMP per-angle accuracy comparison (WN vs Speech grouped bars).
Panel (e): 37x37 pairwise similarity matrix (speech trial-level).
Panel (f): OMP accuracy dose-response curve across SNR levels.

Data: dictionary.npz (D, angles) + modal_routing_val.npz (Y_val, labels,
      g_energy_expert) + white noise raw waveforms (.npy -> STFT inline).
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

def _load_white_noise_features(
    dataset_root: str | Path,
    fs: float = 16000.0,
    n_fft: int = 2048,
    freq_min: float = 300.0,
    freq_max: float = 3000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load white noise waveforms and compute STFT-based features.

    Applies scipy.signal.stft with fs=16000 directly on the raw waveforms
    (no resampling), matching the training pipeline in omp-transformer-ldv.py.
    Time-averages the magnitude spectrogram and L2-normalises each sample.

    Returns (Y_wn, labels_wn, angles_deg) where Y_wn has shape (N, F).
    """
    from scipy.signal import stft as scipy_stft

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
            f, _t, Zxx = scipy_stft(
                wav, fs=fs, window="hann", nperseg=n_fft,
                noverlap=n_fft // 2, nfft=n_fft, detrend="constant",
                return_onesided=True, boundary=None, padded=False,
            )
            mag = np.abs(Zxx)
            mask = (f >= freq_min) & (f <= freq_max)
            y = mag[mask].mean(axis=1).astype(np.float32)
            norm = np.linalg.norm(y)
            if norm > 1e-12:
                y = y / norm
            features_list.append(y)
            labels_list.append(angle_idx)

    Y_wn = np.stack(features_list)  # (N, F)
    labels_wn = np.array(labels_list, dtype=int)
    angles_arr = np.array(angles_deg, dtype=np.float32)
    return Y_wn, labels_wn, angles_arr


def _compute_omp_accuracy(
    Y: np.ndarray, labels: np.ndarray,
    D: np.ndarray, n_experts: int, n_atoms: int,
) -> np.ndarray:
    """Compute OMP per-angle accuracy: g = |Y @ D|.reshape(N,E,M).sum(2)."""
    g = np.abs(Y @ D).reshape(len(Y), n_experts, n_atoms).sum(axis=2)
    pred = np.argmax(g, axis=1)
    per_angle_acc = np.array([
        np.mean(pred[labels == a] == a) if np.sum(labels == a) > 0 else 0.0
        for a in range(n_experts)
    ])
    return per_angle_acc


def _compute_omp_mean_accuracy(
    Y: np.ndarray, labels: np.ndarray,
    D: np.ndarray, n_experts: int, n_atoms: int,
) -> float:
    """Compute OMP mean accuracy across all angles."""
    return float(np.mean(_compute_omp_accuracy(Y, labels, D, n_experts, n_atoms)))


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


def _draw_violin_panel(
    ax, within_all, between_all, title: str, panel_label: str,
    *, ylim: tuple[float, float] | None = None,
):
    """Draw within vs between violin + stats annotation on given axes."""
    within_arr = np.array(within_all)
    between_arr = np.array(between_all)

    data_violin = [within_all, between_all]
    parts = ax.violinplot(data_violin, positions=[0, 1], showmeans=True,
                          showmedians=False, showextrema=False)
    violin_colors = [SEMANTIC_PALETTE["physics"], SEMANTIC_PALETTE["ablation"]]
    for pc, color in zip(parts["bodies"], violin_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(1.0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Within\nangle", "Between\nangles"], fontsize=5.5)
    ax.set_ylabel("Pearson r", fontsize=6)
    ax.set_title(title, fontsize=6.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    # Stats
    U_stat, p_value = sp_stats.mannwhitneyu(
        within_arr, between_arr, alternative="greater"
    )
    d_value = _cohens_d(within_arr, between_arr)
    stars = _p_to_stars(p_value)

    y_max = max(np.max(within_all), np.max(between_all))
    bracket_y = y_max + 0.02
    bar_y = bracket_y + 0.01
    ax.plot([0, 0, 1, 1], [bracket_y, bar_y, bar_y, bracket_y],
            lw=0.8, c="black")
    ax.text(0.5, bar_y + 0.005, f"{stars}\nd = {d_value:.2f}",
            ha="center", va="bottom", fontsize=6.5, linespacing=1.2)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Annotate mean values near bottom
    y_annot = ylim[0] + 0.02 if ylim else 0.35
    ax.text(0, y_annot, f"r\u0304={np.mean(within_arr):.3f}",
            ha="center", va="bottom", fontsize=5, color=SEMANTIC_PALETTE["physics"])
    ax.text(1, y_annot, f"r\u0304={np.mean(between_arr):.3f}",
            ha="center", va="bottom", fontsize=5, color=SEMANTIC_PALETTE["ablation"])

    add_panel_label(ax, panel_label, x=-0.20, y=1.06)
    return p_value, d_value, np.mean(within_arr)


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
    """Generate Figure 3 — WN vs Speech Discriminability (6 panels)."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]

    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"

    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig03] SKIP: data not found at {run_dir}")
        return []

    # --- Load speech data ---
    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    Y_val = routing_data["Y_val"]
    labels = routing_data["labels"].astype(int)
    D = dict_data["D"]           # (346, 296)
    angles = dict_data["angles"]  # (37,)
    n_angles = len(angles)
    n_atoms = D.shape[1] // n_angles  # 8

    # --- Load white noise data (pure WN = snrInf) ---
    wn_dataset = paths_cfg.get("white_noise_dataset", "")
    if not Path(wn_dataset).exists():
        print(f"[fig03] SKIP: white noise dataset not found at {wn_dataset}")
        return []

    print("[fig03] Loading white noise features (STFT inline)...")
    Y_wn, labels_wn, _ = _load_white_noise_features(wn_dataset)
    print(f"[fig03] White noise features: {Y_wn.shape}")

    # --- Compute discriminability for both ---
    within_wn, between_wn, pa_mean_wn, pa_std_wn, pa_n_wn = \
        _compute_pairwise_corr(Y_wn, labels_wn, n_angles)
    within_sp, between_sp, pa_mean_sp, pa_std_sp, pa_n_sp = \
        _compute_pairwise_corr(Y_val, labels, n_angles)

    pa_sem_wn = np.where(pa_n_wn > 0, pa_std_wn / np.sqrt(pa_n_wn), 0.0)
    pa_sem_sp = np.where(pa_n_sp > 0, pa_std_sp / np.sqrt(pa_n_sp), 0.0)

    # --- OMP accuracy (WN and Speech) ---
    omp_acc_wn = _compute_omp_accuracy(Y_wn, labels_wn, D, n_angles, n_atoms)
    omp_acc_sp = _compute_omp_accuracy(Y_val, labels, D, n_angles, n_atoms)
    omp_mean_wn = float(np.mean(omp_acc_wn))
    omp_mean_sp = float(np.mean(omp_acc_sp))
    print(f"[fig03] OMP accuracy — WN: {omp_mean_wn:.3f}, Speech: {omp_mean_sp:.3f}")

    # --- Pairwise similarity matrix (speech) ---
    sim_matrix = _compute_trial_pairwise_matrix(Y_val, labels, n_angles)

    # --- SNR sweep dose-response (panel f) ---
    snr_base = paths_cfg.get("snr_sweep_base", "")
    snr_levels_cfg = paths_cfg.get("snr_sweep_levels", [])

    snr_labels: list[str] = []
    snr_x_positions: list[float] = []
    snr_omp_accs: list[float] = []

    print("[fig03] Computing SNR dose-response curve...")
    for item in snr_levels_cfg:
        snr_db = str(item["snr_db"])
        suffix = item["dir_suffix"]
        ds_path = Path(snr_base) / f"white_noise_box_{suffix}_sync_vad_normalized"
        if not ds_path.exists():
            print(f"[fig03]   SKIP SNR={snr_db}: {ds_path} not found")
            continue
        Y_snr, lab_snr, _ = _load_white_noise_features(str(ds_path))
        acc = _compute_omp_mean_accuracy(Y_snr, lab_snr, D, n_angles, n_atoms)
        snr_labels.append(snr_db)
        # Map SNR to x position: Inf -> 7, 30 -> 6, 20 -> 5, ... 0 -> 1
        snr_omp_accs.append(acc)
        print(f"[fig03]   SNR={snr_db:>3s} dB: OMP acc = {acc:.3f}")

    # Append pure speech as rightmost point
    snr_labels.append("Speech")
    snr_omp_accs.append(omp_mean_sp)

    # X positions: evenly spaced categorical axis
    n_snr_points = len(snr_labels)
    snr_x_positions = list(range(n_snr_points))

    # -----------------------------------------------------------------------
    # Build composite figure (2x3 grid, 6 panels)
    # -----------------------------------------------------------------------
    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=130)
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.35, wspace=0.40,
        left=0.07, right=0.96, bottom=0.07, top=0.95,
    )

    # Shared y-axis range for both violin panels
    all_r = within_wn + between_wn + within_sp + between_sp
    violin_ylim = (min(all_r) - 0.05, max(all_r) + 0.10)

    # --- Panel (a): White noise violin ---
    ax_a = fig.add_subplot(gs[0, 0])
    p_wn, d_wn, mean_within_wn = _draw_violin_panel(
        ax_a, within_wn, between_wn, "White noise", "a",
        ylim=violin_ylim,
    )

    # --- Panel (b): Speech violin ---
    ax_b = fig.add_subplot(gs[0, 1])
    p_sp, d_sp, mean_within_sp = _draw_violin_panel(
        ax_b, within_sp, between_sp, "Speech", "b",
        ylim=violin_ylim,
    )

    # --- Panel (c): Per-angle repeatability comparison ---
    ax_c = fig.add_subplot(gs[0, 2])
    valid_wn = ~np.isnan(pa_mean_wn)
    valid_sp = ~np.isnan(pa_mean_sp)

    ax_c.plot(angles[valid_wn], pa_mean_wn[valid_wn], "-o", markersize=2,
              linewidth=0.9, color=SEMANTIC_PALETTE["physics"], label="White noise")
    ax_c.fill_between(
        angles[valid_wn],
        pa_mean_wn[valid_wn] - pa_sem_wn[valid_wn],
        pa_mean_wn[valid_wn] + pa_sem_wn[valid_wn],
        alpha=0.20, color=SEMANTIC_PALETTE["physics"],
    )
    ax_c.plot(angles[valid_sp], pa_mean_sp[valid_sp], "-s", markersize=2,
              linewidth=0.9, color=SEMANTIC_PALETTE["ablation"], label="Speech")
    ax_c.fill_between(
        angles[valid_sp],
        pa_mean_sp[valid_sp] - pa_sem_sp[valid_sp],
        pa_mean_sp[valid_sp] + pa_sem_sp[valid_sp],
        alpha=0.20, color=SEMANTIC_PALETTE["ablation"],
    )
    ax_c.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_c.set_ylabel("Mean within-angle r", fontsize=6)
    ax_c.set_title("Fingerprint repeatability", fontsize=6.5)
    ax_c.grid(axis="y", linestyle="--", alpha=0.3)
    ax_c.legend(fontsize=5.5, frameon=False, loc="center left")
    add_panel_label(ax_c, "c", x=-0.12, y=1.06)

    # --- Panel (d): OMP per-angle accuracy comparison ---
    ax_d = fig.add_subplot(gs[1, 0])
    bar_width = 2.2
    ax_d.bar(angles - bar_width / 2, omp_acc_wn, width=bar_width,
             color=SEMANTIC_PALETTE["physics"], alpha=0.85, label=f"WN ({omp_mean_wn:.1%})")
    ax_d.bar(angles + bar_width / 2, omp_acc_sp, width=bar_width,
             color=SEMANTIC_PALETTE["ablation"], alpha=0.85, label=f"Speech ({omp_mean_sp:.1%})")
    ax_d.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_d.set_ylabel("OMP accuracy", fontsize=6)
    ax_d.set_title("Classical OMP baseline", fontsize=6.5)
    ax_d.set_ylim(0, 1.12)
    ax_d.legend(fontsize=5.5, frameon=False, loc="upper right",
                bbox_to_anchor=(1.0, 0.55))
    ax_d.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax_d, "d", x=-0.15, y=1.06)

    # --- Panel (e): Pairwise similarity matrix (speech) ---
    ax_e = fig.add_subplot(gs[1, 1])
    im_e = ax_e.imshow(sim_matrix, cmap="RdBu_r", aspect="auto",
                        vmin=-0.2, vmax=1.0)
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels_e = [f"{int(angles[i])}" for i in tick_positions]
    ax_e.set_xticks(tick_positions)
    ax_e.set_xticklabels(tick_labels_e, fontsize=5)
    ax_e.set_yticks(tick_positions)
    ax_e.set_yticklabels(tick_labels_e, fontsize=5)
    ax_e.set_xlabel("Angle (\u00b0)", fontsize=6)
    ax_e.set_ylabel("Angle (\u00b0)", fontsize=6)
    ax_e.set_title("Pairwise similarity (speech)", fontsize=6.5)
    cbar = plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5)
    add_panel_label(ax_e, "e", x=-0.12, y=1.06)

    # --- Panel (f): SNR dose-response curve ---
    ax_f = fig.add_subplot(gs[1, 2])
    # Color gradient: blue (WN) -> red (speech)
    n_pts = len(snr_x_positions)
    cmap_dr = plt.cm.RdYlBu_r
    colors_dr = [cmap_dr(i / max(n_pts - 1, 1)) for i in range(n_pts)]

    ax_f.plot(snr_x_positions, snr_omp_accs, "-", color="gray",
              linewidth=1.0, alpha=0.5, zorder=1)
    for xi, yi, ci in zip(snr_x_positions, snr_omp_accs, colors_dr):
        ax_f.scatter(xi, yi, color=ci, s=25, zorder=2, edgecolors="black",
                     linewidths=0.3)

    ax_f.set_xticks(snr_x_positions)
    # X-tick labels: "Inf", "30", ..., "0", "Speech" (add dB to numeric)
    xtick_labels_f = []
    for lbl in snr_labels:
        if lbl == "Inf":
            xtick_labels_f.append("\u221e")
        elif lbl == "Speech":
            xtick_labels_f.append("Speech")
        else:
            xtick_labels_f.append(f"{lbl}")
    ax_f.set_xticklabels(xtick_labels_f, fontsize=5, rotation=45, ha="right")
    ax_f.set_xlabel("SNR (dB)  \u2192  content variation", fontsize=6)
    ax_f.set_ylabel("OMP accuracy", fontsize=6)
    ax_f.set_title("Dose-response curve", fontsize=6.5)
    ax_f.set_ylim(-0.02, 1.05)
    ax_f.grid(axis="y", linestyle="--", alpha=0.3)

    # Annotate endpoints
    ax_f.annotate(f"{snr_omp_accs[0]:.1%}", (snr_x_positions[0], snr_omp_accs[0]),
                  textcoords="offset points", xytext=(8, -4), fontsize=5.5,
                  color=SEMANTIC_PALETTE["physics"])
    ax_f.annotate(f"{snr_omp_accs[-1]:.1%}", (snr_x_positions[-1], snr_omp_accs[-1]),
                  textcoords="offset points", xytext=(-20, 8), fontsize=5.5,
                  color=SEMANTIC_PALETTE["ablation"])

    add_panel_label(ax_f, "f", x=-0.12, y=1.06)

    # Save composite
    all_paths = save_outputs(fig, output_dir / "fig03_fingerprint_discriminability")
    plt.close(fig)

    # -----------------------------------------------------------------------
    # Standalone panels
    # -----------------------------------------------------------------------
    panel_dir = output_dir / "fig03_fingerprint_discriminability_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    # Panel a standalone
    fig_a = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_a.add_subplot(111)
    _draw_violin_panel(ax, within_wn, between_wn, "White noise", "a",
                       ylim=violin_ylim)
    fig_a.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig03_panel_a_wn_violin"))
    plt.close(fig_a)

    # Panel b standalone
    fig_b = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_b.add_subplot(111)
    _draw_violin_panel(ax, within_sp, between_sp, "Speech", "b",
                       ylim=violin_ylim)
    fig_b.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig03_panel_b_speech_violin"))
    plt.close(fig_b)

    # Panel c standalone
    fig_c = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_c.add_subplot(111)
    ax.plot(angles[valid_wn], pa_mean_wn[valid_wn], "-o", markersize=2,
            linewidth=0.9, color=SEMANTIC_PALETTE["physics"], label="White noise")
    ax.fill_between(angles[valid_wn],
                    pa_mean_wn[valid_wn] - pa_sem_wn[valid_wn],
                    pa_mean_wn[valid_wn] + pa_sem_wn[valid_wn],
                    alpha=0.20, color=SEMANTIC_PALETTE["physics"])
    ax.plot(angles[valid_sp], pa_mean_sp[valid_sp], "-s", markersize=2,
            linewidth=0.9, color=SEMANTIC_PALETTE["ablation"], label="Speech")
    ax.fill_between(angles[valid_sp],
                    pa_mean_sp[valid_sp] - pa_sem_sp[valid_sp],
                    pa_mean_sp[valid_sp] + pa_sem_sp[valid_sp],
                    alpha=0.20, color=SEMANTIC_PALETTE["ablation"])
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Mean within-angle Pearson r")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(fontsize=5, frameon=False, loc="center left")
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(left=0.10, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig03_panel_c_repeatability"))
    plt.close(fig_c)

    # Panel d standalone
    fig_d = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_d.add_subplot(111)
    ax.bar(angles - bar_width / 2, omp_acc_wn, width=bar_width,
           color=SEMANTIC_PALETTE["physics"], alpha=0.85, label=f"WN ({omp_mean_wn:.1%})")
    ax.bar(angles + bar_width / 2, omp_acc_sp, width=bar_width,
           color=SEMANTIC_PALETTE["ablation"], alpha=0.85, label=f"Speech ({omp_mean_sp:.1%})")
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("OMP accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 0.55))
    add_panel_label(ax, "d")
    fig_d.subplots_adjust(left=0.08, right=0.95, bottom=0.15, top=0.92)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig03_panel_d_omp_comparison"))
    plt.close(fig_d)

    # Panel e standalone
    fig_e = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_e.add_subplot(111)
    im = ax.imshow(sim_matrix, cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=1.0)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_e)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels_e)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Mean Pearson r", fontsize=6)
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.92)
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig03_panel_e_pairwise"))
    plt.close(fig_e)

    # Panel f standalone
    fig_f = make_figure(width_mm=DOUBLE_COL_MM, height_mm=80)
    ax = fig_f.add_subplot(111)
    ax.plot(snr_x_positions, snr_omp_accs, "-", color="gray", linewidth=1.0,
            alpha=0.5, zorder=1)
    for xi, yi, ci in zip(snr_x_positions, snr_omp_accs, colors_dr):
        ax.scatter(xi, yi, color=ci, s=30, zorder=2, edgecolors="black",
                   linewidths=0.3)
    ax.set_xticks(snr_x_positions)
    ax.set_xticklabels(xtick_labels_f, fontsize=6, rotation=45, ha="right")
    ax.set_xlabel("SNR (dB)  \u2192  content variation")
    ax.set_ylabel("OMP accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.annotate(f"{snr_omp_accs[0]:.1%}", (snr_x_positions[0], snr_omp_accs[0]),
                textcoords="offset points", xytext=(8, -4), fontsize=6,
                color=SEMANTIC_PALETTE["physics"])
    ax.annotate(f"{snr_omp_accs[-1]:.1%}", (snr_x_positions[-1], snr_omp_accs[-1]),
                textcoords="offset points", xytext=(-20, 8), fontsize=6,
                color=SEMANTIC_PALETTE["ablation"])
    add_panel_label(ax, "f")
    fig_f.subplots_adjust(left=0.08, right=0.95, bottom=0.20, top=0.92)
    all_paths.extend(save_outputs(fig_f, panel_dir / "fig03_panel_f_dose_response"))
    plt.close(fig_f)

    # Panel manifest
    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "White noise within vs between similarity",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_a_wn_violin.pdf",
                "provenance_mode": "data_backed",
                "description": "Violin plot of within vs between-angle Pearson r for white noise stimuli.",
            },
            {
                "panel_id": "b",
                "title": "Speech within vs between similarity",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_b_speech_violin.pdf",
                "provenance_mode": "data_backed",
                "description": "Violin plot of within vs between-angle Pearson r for speech stimuli.",
            },
            {
                "panel_id": "c",
                "title": "Per-angle repeatability comparison",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_c_repeatability.pdf",
                "provenance_mode": "data_backed",
                "description": "WN vs Speech mean within-angle r per angle with SEM bands.",
            },
            {
                "panel_id": "d",
                "title": "OMP accuracy comparison",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_d_omp_comparison.pdf",
                "provenance_mode": "data_backed",
                "description": "Side-by-side OMP per-angle accuracy: WN vs Speech.",
            },
            {
                "panel_id": "e",
                "title": "Trial-level pairwise similarity matrix",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_e_pairwise.pdf",
                "provenance_mode": "data_backed",
                "description": "37x37 mean Pearson r between all angle pairs (speech trial-level).",
            },
            {
                "panel_id": "f",
                "title": "OMP dose-response curve",
                "asset_path": "figures/output/fig03_fingerprint_discriminability_panels/fig03_panel_f_dose_response.pdf",
                "provenance_mode": "data_backed",
                "description": "OMP accuracy vs content variation (SNR sweep from pure WN to speech).",
            },
        ],
    )
    all_paths.append(manifest)

    print(f"[fig03] Generated {len(all_paths)} files")
    print(f"[fig03] WN: p={p_wn:.2e}, d={d_wn:.2f}, mean_within_r={mean_within_wn:.3f}")
    print(f"[fig03] Speech: p={p_sp:.2e}, d={d_sp:.2f}, mean_within_r={mean_within_sp:.3f}")
    print(f"[fig03] OMP — WN: {omp_mean_wn:.3f}, Speech: {omp_mean_sp:.3f}")
    print(f"[fig03] Dose-response: {' -> '.join(f'{a:.1%}' for a in snr_omp_accs)}")
    return all_paths
