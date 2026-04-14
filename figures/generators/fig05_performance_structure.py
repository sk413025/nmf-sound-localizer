"""Figure 5 — coherent family preservation through final prediction (7 panels).

Panel (a): Neighborhood-preservation cascade on one shared radius axis.
Panel (b): Coherent family neighborhood preservation.
Panel (c): Coherent family exact-vs-local scorecard.
Panel (d): Final-prediction locality by family.
Panel (e): Measured neighborhood geometry.
Panel (f): Family-to-measured neighborhood alignment.
Panel (g): Noise-robustness consequence.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from figures.layout_contract import (
    contract_version,
    figure_section,
    font_tokens,
    source_layout_spec,
)
from figures.style import (
    SEMANTIC_PALETTE,
    STYLE_COLORS,
    add_panel_label,
    load_paths,
    make_figure,
    save_outputs,
    set_nature_rcparams,
)


SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "\u221e"]
DENSE_COLOR = "#4A4A4A"
LOCAL_RADII_DEG = np.asarray([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0], dtype=np.float32)
LOCAL_CALLOUT_DEG = 15.0
FIG05_CORR_NORM = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=1.0)
FIG05_CORR_TICKS = [-0.1, 0.0, 0.5, 1.0]
LEGACY_SNR_TO_TOKEN = {
    "0dB": "snr0dB",
    "5dB": "snr5dB",
    "10dB": "snr10dB",
    "15dB": "snr15dB",
    "20dB": "snr20dB",
    "30dB": "snr30dB",
    "Clean": "snrInf",
}
FAMILY_CONFIG = {
    "guided": {
        "display": "guided solver",
        "legacy_label": "No Type Bias",
        "prefix": "ablate_speech260_no_type_bias",
        "color": SEMANTIC_PALETTE["learned"],
    },
    "router_bypass": {
        "display": "router-bypass",
        "legacy_label": "No Transformer",
        "prefix": "ablate_speech260_no_transformer",
        "color": SEMANTIC_PALETTE["ablation"],
    },
    "omp": {
        "display": "OMP baseline",
        "legacy_label": "Fixed Heuristic",
        "prefix": "ablate_speech260_g_routing",
        "color": SEMANTIC_PALETTE["classical"],
    },
    "dense": {
        "display": "dense routing",
        "legacy_label": "Dense Routing",
        "prefix": "ablate_speech260_disable_sparsity",
        "color": DENSE_COLOR,
    },
}

FIG05_GENERATOR = figure_section("fig05", "generator")
FIG05_COMPOSITE_GRID = dict(FIG05_GENERATOR["composite_grid"])
FIG05_TOP_ROW = dict(FIG05_GENERATOR["top_row"])
FIG05_MIDDLE_ROW = dict(FIG05_GENERATOR["middle_row"])
FIG05_BOTTOM_ROW = dict(FIG05_GENERATOR["bottom_row"])
FIG05_CONFUSION_PAIR = dict(FIG05_GENERATOR["confusion_pair"])
FIG05_STANDALONE = dict(FIG05_GENERATOR["standalone"])


def _set_gid(ax, gid: str):
    ax.set_gid(gid)
    return ax


def _row_normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def _grouped_match_surface(Y: np.ndarray, D: np.ndarray, n_experts: int) -> np.ndarray:
    n_atoms = int(D.shape[1] // n_experts)
    grouped = np.abs(np.asarray(Y, dtype=np.float32) @ np.asarray(D, dtype=np.float32))
    return grouped.reshape(len(Y), n_experts, n_atoms).sum(axis=2).astype(np.float32)


def _mass_within_radius_rows(
    support_rows: np.ndarray,
    angles: np.ndarray,
    truth_angles: np.ndarray,
    radii_deg: np.ndarray,
    *,
    normalize_rows: bool,
) -> np.ndarray:
    rows = np.asarray(support_rows, dtype=float)
    if normalize_rows:
        rows = _row_normalize_matrix(rows)
    angles = np.asarray(angles, dtype=float)
    truth_angles = np.asarray(truth_angles, dtype=float)
    radii_deg = np.asarray(radii_deg, dtype=float)
    out = np.zeros((rows.shape[0], radii_deg.size), dtype=np.float32)
    for idx, target_angle in enumerate(truth_angles):
        distance = np.abs(angles - float(target_angle))
        for ridx, radius in enumerate(radii_deg):
            out[idx, ridx] = float(rows[idx, distance <= radius].sum())
    return out


def _curve_mean_sem(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rows, dtype=np.float32)
    mean = arr.mean(axis=0, dtype=np.float64).astype(np.float32)
    sem = np.zeros_like(mean)
    if arr.shape[0] > 1:
        sem = (arr.std(axis=0, ddof=1, dtype=np.float64) / np.sqrt(arr.shape[0])).astype(np.float32)
    return mean, sem


def _mass_value_at_radius(curve: np.ndarray, radii_deg: np.ndarray, radius_deg: float) -> float:
    ridx = int(np.argmin(np.abs(np.asarray(radii_deg, dtype=float) - float(radius_deg))))
    return float(np.asarray(curve, dtype=float)[ridx])


def _compute_global_correlation(
    routing_data: dict,
    dict_data: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores_expert = np.asarray(routing_data["scores_expert"], dtype=float)
    H = np.asarray(dict_data["H"], dtype=float)
    expert_corr = np.corrcoef(scores_expert.T)
    H_corr = np.corrcoef(H.T)
    structure_corr = float(np.corrcoef(H_corr.flatten(), expert_corr.flatten())[0, 1])
    return expert_corr, H_corr, structure_corr


def _local_band_scores(matrix: np.ndarray, angles: np.ndarray, radius_deg: float = 15.0) -> np.ndarray:
    scores: list[float] = []
    for idx, angle in enumerate(np.asarray(angles, dtype=float)):
        dist = np.abs(angles - angle)
        mask = (dist <= radius_deg) & (dist > 0)
        scores.append(float(np.asarray(matrix[idx, mask], dtype=float).mean()))
    return np.asarray(scores, dtype=float)


def _minmax_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 0:
        raise ValueError("min-max normalization requires non-constant values")
    return (values - lo) / span


def _plot_confusion_matrix(
    ax,
    matrix_norm: np.ndarray,
    title: str,
    *,
    title_pt: float,
    axis_label_pt: float,
    tick_label_pt: float,
    tick_positions: list[int],
    tick_labels: list[str],
    xlabel: str | None = None,
    ylabel: str | None = None,
    show_xticklabels: bool = True,
    show_yticklabels: bool = True,
    fontweight: str | None = None,
):
    im = ax.imshow(
        matrix_norm,
        cmap="viridis",
        aspect="equal",
        extent=[0, 37, 37, 0],
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=title_pt, fontweight=fontweight)
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    if show_xticklabels:
        ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
        ax.tick_params(axis="x", labelbottom=True, labelsize=tick_label_pt, pad=1.0)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", labelbottom=False)
    if show_yticklabels:
        ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
        ax.tick_params(axis="y", labelleft=True, labelsize=tick_label_pt, pad=1.0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", labelleft=False)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=axis_label_pt)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=axis_label_pt)
    ax.tick_params(axis="both", length=2)
    return im


def _plot_support_curves(
    ax,
    *,
    radii_deg: np.ndarray,
    curves: list[tuple[str, np.ndarray, np.ndarray | None, str, str, float]],
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    title: str,
    annotation_pt: float,
    show_ylabel: bool = True,
    callout_radius_deg: float | None = LOCAL_CALLOUT_DEG,
    shade_local_regime: bool = True,
) -> None:
    if shade_local_regime:
        ax.axvspan(
            0.0,
            LOCAL_CALLOUT_DEG,
            color=SEMANTIC_PALETTE["highlight"],
            alpha=0.12,
            linewidth=0.0,
            zorder=0,
        )
    ax.axvline(
        LOCAL_CALLOUT_DEG,
        color=STYLE_COLORS["guide_line"],
        linewidth=0.8,
        linestyle=":",
        alpha=0.9,
        zorder=1,
    )
    for label, mean, sem, color, linestyle, linewidth in curves:
        mean = np.asarray(mean, dtype=float)
        if sem is not None:
            sem = np.asarray(sem, dtype=float)
            ax.fill_between(
                radii_deg,
                np.clip(mean - sem, 0.0, 1.0),
                np.clip(mean + sem, 0.0, 1.0),
                color=color,
                alpha=0.10,
                linewidth=0.0,
                zorder=1,
            )
        ax.plot(
            radii_deg,
            mean,
            marker="o",
            markersize=2.8,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            label=label,
            zorder=3,
        )
    ax.set_title(title, fontsize=annotation_pt + 0.5)
    ax.set_xlim(float(radii_deg[0]), float(radii_deg[-1]))
    ax.set_ylim(0.0, 1.04)
    ax.set_xticks(radii_deg)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Neighborhood radius (\u00b0)", fontsize=axis_label_pt)
    if show_ylabel:
        ax.set_ylabel("Mass within radius", fontsize=axis_label_pt)
    ax.tick_params(axis="both", labelsize=tick_label_pt, length=2)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(
        fontsize=max(legend_pt - 0.35, 5.8),
        frameon=False,
        loc="lower right",
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    if callout_radius_deg is not None:
        ax.text(
            0.02,
            0.94,
            f"within {int(callout_radius_deg)}\u00b0",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(annotation_pt - 0.2, 5.8),
            color=STYLE_COLORS["muted_text"],
        )


def _plot_snr_panel(
    ax,
    snr_curves: dict[str, dict[str, list[float]]],
    *,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
) -> None:
    x_snr = np.arange(len(SNR_ORDER))
    for family_key, cfg in FAMILY_CONFIG.items():
        family_levels = snr_curves[family_key]
        means = np.asarray([np.mean(family_levels[snr_key]) for snr_key in SNR_ORDER], dtype=float)
        stds = np.asarray([np.std(family_levels[snr_key]) for snr_key in SNR_ORDER], dtype=float)
        ax.plot(
            x_snr,
            means,
            "-o",
            color=cfg["color"],
            markersize=3.0,
            linewidth=1.0,
            label=cfg["display"],
            zorder=3,
        )
        ax.fill_between(x_snr, means - stds, means + stds, color=cfg["color"], alpha=0.14, zorder=1)
    ax.set_xticks(x_snr)
    ax.set_xticklabels(SNR_DISPLAY)
    ax.set_xlabel("SNR (dB)", fontsize=axis_label_pt)
    ax.set_ylabel("Accuracy", fontsize=axis_label_pt)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        fontsize=max(legend_pt - 0.4, 5.8),
        frameon=False,
        loc="lower left",
        ncol=1,
        handlelength=1.4,
        handletextpad=0.35,
        borderaxespad=0.2,
    )


def _plot_exact_vs_local_scorecard(
    ax,
    *,
    exact_accuracy_mean: dict[str, float],
    exact_accuracy_sem: dict[str, float],
    within15_mean: dict[str, float],
    within15_sem: dict[str, float],
    title_pt: float,
    axis_label_pt: float,
    tick_label_pt: float,
    annotation_pt: float,
) -> None:
    ax.set_title("Coherent family exact-vs-local scorecard", fontsize=title_pt)
    for family_key, cfg in FAMILY_CONFIG.items():
        x = exact_accuracy_mean[family_key]
        y = within15_mean[family_key]
        ax.errorbar(
            x,
            y,
            xerr=exact_accuracy_sem[family_key],
            yerr=within15_sem[family_key],
            fmt="o",
            color=cfg["color"],
            markersize=5.0,
            capsize=2.0,
        )
        ax.text(
            x + 0.015,
            y + 0.01,
            cfg["display"],
            fontsize=annotation_pt,
            color=cfg["color"],
            ha="left",
            va="bottom",
        )
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color=STYLE_COLORS["guide_line"], linewidth=0.9)
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Exact clean accuracy", fontsize=axis_label_pt)
    ax.set_ylabel("Within-15\u00b0 local mass", fontsize=axis_label_pt)
    ax.tick_params(axis="both", labelsize=tick_label_pt)
    ax.grid(True, linestyle="--", alpha=0.25)


def _profile_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.allclose(a.std(ddof=0), 0.0) or np.allclose(b.std(ddof=0), 0.0):
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _build_family_alignment_bundle(
    *,
    H_corr: np.ndarray,
    angles: np.ndarray,
    clean_confusion_rows: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, float]]:
    measured_profile = _local_band_scores(H_corr, angles)
    local_agreement_mean: dict[str, float] = {}
    local_agreement_sem: dict[str, float] = {}
    global_r_mean: dict[str, float] = {}
    global_r_sem: dict[str, float] = {}

    for family_key, matrices in clean_confusion_rows.items():
        local_scores = []
        global_scores = []
        for matrix in matrices:
            matrix_norm = _row_normalize_matrix(matrix)
            family_profile = _local_band_scores(matrix_norm, angles)
            profile_corr = _profile_corr(measured_profile, family_profile)
            within15 = _within15_from_confusion_rows(matrix_norm)
            # Keep retained local mass as the primary term and use profile agreement
            # only to modulate how well that mass follows the measured neighborhood.
            local_scores.append(within15 * (0.75 + 0.25 * (0.5 * (profile_corr + 1.0))))
            global_scores.append(float(np.corrcoef(H_corr.flatten(), matrix_norm.flatten())[0, 1]))
        local_arr = np.asarray(local_scores, dtype=float)
        global_arr = np.asarray(global_scores, dtype=float)
        local_agreement_mean[family_key] = float(local_arr.mean())
        local_agreement_sem[family_key] = float(local_arr.std(ddof=0) / np.sqrt(local_arr.size))
        global_r_mean[family_key] = float(global_arr.mean())
        global_r_sem[family_key] = float(global_arr.std(ddof=0) / np.sqrt(global_arr.size))

    return {
        "local_agreement_mean": local_agreement_mean,
        "local_agreement_sem": local_agreement_sem,
        "global_r_mean": global_r_mean,
        "global_r_sem": global_r_sem,
    }


def _plot_family_alignment_panel(
    ax,
    *,
    local_agreement_mean: dict[str, float],
    local_agreement_sem: dict[str, float],
    global_r_mean: dict[str, float],
    global_r_sem: dict[str, float],
    title_pt: float,
    axis_label_pt: float,
    tick_label_pt: float,
    legend_pt: float,
    annotation_pt: float,
) -> None:
    families = list(FAMILY_CONFIG.keys())
    x = np.arange(len(families), dtype=float)
    labels = [FAMILY_CONFIG[key]["display"] for key in families]
    colors = [FAMILY_CONFIG[key]["color"] for key in families]
    local_vals = [local_agreement_mean[key] for key in families]
    local_err = [local_agreement_sem[key] for key in families]
    global_vals = [global_r_mean[key] for key in families]
    global_err = [global_r_sem[key] for key in families]
    ax.bar(
        x,
        local_vals,
        yerr=local_err,
        color=colors,
        alpha=0.82,
        width=0.65,
        capsize=2.0,
        linewidth=0.0,
        zorder=2,
    )
    ax.errorbar(
        x,
        global_vals,
        yerr=global_err,
        fmt="o",
        color=STYLE_COLORS["neutral_text"],
        markerfacecolor="white",
        markeredgewidth=1.0,
        markersize=4.8,
        capsize=2.0,
        linewidth=0.9,
        zorder=4,
    )
    ax.axhline(0.0, color=STYLE_COLORS["guide_line"], linewidth=0.8, linestyle="--", zorder=1)
    ax.set_title("Family-to-measured neighborhood alignment", fontsize=title_pt)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=tick_label_pt)
    ax.set_ylabel("Alignment score", fontsize=axis_label_pt)
    ax.set_ylim(-0.18, 1.02)
    ax.tick_params(axis="y", labelsize=tick_label_pt)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.text(
        0.02,
        0.97,
        "bars: local-band agreement\ncircles: global matrix r",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=max(annotation_pt - 0.15, 5.8),
        color=STYLE_COLORS["muted_text"],
    )


def _save_panel_manifest(panel_dir: Path, panel_specs: list[dict], typography: dict[str, float]) -> Path:
    manifest_path = panel_dir / "fig05_panel_manifest.json"
    payload = {
        "contract_version": contract_version(),
        "figure_id": "fig05",
        "composite_asset": "figures/output/fig05_performance_structure.pdf",
        "storage_mode": "direct_generator_outputs",
        "panel_order": [item["panel_id"] for item in panel_specs],
        "panels": panel_specs,
        "source_layout_spec": source_layout_spec(),
        "typography_pt": typography,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _legacy_metric_paths_sorted(legacy_runs_root: Path, legacy_suffix: str, family_key: str, snr_key: str) -> list[Path]:
    prefix = FAMILY_CONFIG[family_key]["prefix"]
    token = LEGACY_SNR_TO_TOKEN[snr_key]
    pattern = f"{prefix}_{token}_seed*_ep20_lr1e-3_{legacy_suffix}/metrics.npz"
    return sorted(legacy_runs_root.glob(pattern))


def _legacy_metrics_path(legacy_runs_root: Path, legacy_suffix: str, family_key: str, snr_key: str, seed: int) -> Path:
    prefix = FAMILY_CONFIG[family_key]["prefix"]
    token = LEGACY_SNR_TO_TOKEN[snr_key]
    run_name = f"{prefix}_{token}_seed{seed}_ep20_lr1e-3_{legacy_suffix}"
    return legacy_runs_root / run_name / "metrics.npz"


def _within15_from_confusion_rows(matrix: np.ndarray) -> float:
    matrix_norm = _row_normalize_matrix(matrix)
    vals = []
    for idx in range(matrix_norm.shape[0]):
        lo = max(0, idx - 3)
        hi = min(matrix_norm.shape[1], idx + 4)
        vals.append(float(matrix_norm[idx, lo:hi].sum()))
    return float(np.mean(vals))


def _build_legacy_family_bundle(
    *,
    legacy_runs_root: Path,
    legacy_suffix: str,
    angles: np.ndarray,
) -> dict[str, object]:
    snr_summary: dict[str, dict[str, list[float]]] = {}
    clean_curves_mean: dict[str, np.ndarray] = {}
    clean_curves_sem: dict[str, np.ndarray] = {}
    clean_confusion_seed42: dict[str, np.ndarray] = {}
    clean_confusion_rows: dict[str, list[np.ndarray]] = {}
    exact_accuracy_mean: dict[str, float] = {}
    exact_accuracy_sem: dict[str, float] = {}
    within15_mean: dict[str, float] = {}
    within15_sem: dict[str, float] = {}

    for family_key in FAMILY_CONFIG:
        snr_summary[family_key] = {}
        clean_confusion_rows[family_key] = []
        curve_rows = []
        exact_rows = []
        within_rows = []
        for snr_key in SNR_ORDER:
            run_paths = _legacy_metric_paths_sorted(legacy_runs_root, legacy_suffix, family_key, snr_key)
            if not run_paths:
                raise FileNotFoundError(
                    f"Required legacy Fig. 5 family runs not found for family={family_key}, snr={snr_key}, root={legacy_runs_root}"
                )
            accs = []
            for run_path in run_paths:
                metrics = dict(np.load(run_path, allow_pickle=True))
                accs.append(float(metrics["best_accuracy"]))
            snr_summary[family_key][snr_key] = accs

            if snr_key == "Clean":
                seed42 = dict(np.load(_legacy_metrics_path(legacy_runs_root, legacy_suffix, family_key, "Clean", 42), allow_pickle=True))
                clean_confusion_seed42[family_key] = np.asarray(seed42["confusion_matrix"], dtype=float)
                for run_path in run_paths:
                    metrics = dict(np.load(run_path, allow_pickle=True))
                    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
                    clean_confusion_rows[family_key].append(cm)
                    rows = _mass_within_radius_rows(
                        _row_normalize_matrix(cm),
                        angles,
                        angles,
                        LOCAL_RADII_DEG,
                        normalize_rows=False,
                    )
                    curve_rows.append(rows.mean(axis=0))
                    exact_rows.append(float(metrics["best_accuracy"]))
                    within_rows.append(_within15_from_confusion_rows(cm))

        curve_rows_arr = np.stack(curve_rows, axis=0)
        clean_curves_mean[family_key], clean_curves_sem[family_key] = _curve_mean_sem(curve_rows_arr)
        exact_accuracy_mean[family_key] = float(np.mean(exact_rows))
        exact_accuracy_sem[family_key] = float(np.std(exact_rows, ddof=0) / np.sqrt(len(exact_rows)))
        within15_mean[family_key] = float(np.mean(within_rows))
        within15_sem[family_key] = float(np.std(within_rows, ddof=0) / np.sqrt(len(within_rows)))

    return {
        "snr_summary": snr_summary,
        "clean_curves_mean": clean_curves_mean,
        "clean_curves_sem": clean_curves_sem,
        "clean_confusion_seed42": clean_confusion_seed42,
        "clean_confusion_rows": clean_confusion_rows,
        "exact_accuracy_mean": exact_accuracy_mean,
        "exact_accuracy_sem": exact_accuracy_sem,
        "within15_mean": within15_mean,
        "within15_sem": within15_sem,
    }


def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 5 — coherent family bridge, locality, and consequences."""
    typography = font_tokens()
    set_nature_rcparams(base_fontsize=int(round(typography["title"])))
    title_pt = typography["title"]
    axis_label_pt = typography["axis_label"]
    tick_label_pt = typography["tick_label"]
    legend_pt = typography["legend"]
    annotation_pt = typography["annotation"]
    colorbar_tick_pt = typography["colorbar_tick"]
    colorbar_label_pt = typography["colorbar_label"]

    paths_cfg = load_paths()
    run_dir = data_root / paths_cfg["primary_run"]
    legacy_runs_root = Path(paths_cfg["fig05_legacy_runs_root"])
    legacy_suffix = str(paths_cfg["fig05_legacy_sweep_suffix"])

    routing_path = run_dir / "modal_routing_val.npz"
    dict_path = run_dir / "dictionary.npz"
    if not routing_path.exists() or not dict_path.exists():
        print(f"[fig05] SKIP: routing data not found at {run_dir}")
        return []

    routing_data = dict(np.load(routing_path, allow_pickle=True))
    dict_data = dict(np.load(dict_path, allow_pickle=True))
    angles = np.asarray(dict_data["angles"], dtype=float)
    labels = np.asarray(routing_data["labels"], dtype=int)
    Y_val = np.asarray(routing_data["Y_val"], dtype=np.float32)
    D = np.asarray(dict_data["D"], dtype=np.float32)

    _expert_corr, H_corr, _pearson_r = _compute_global_correlation(routing_data, dict_data)

    speech_grouped = _grouped_match_surface(Y_val, D, len(angles))
    speech_stage0_rows = _mass_within_radius_rows(
        speech_grouped,
        angles,
        angles[labels],
        LOCAL_RADII_DEG,
        normalize_rows=True,
    )
    speech_stage0_mean, speech_stage0_sem = _curve_mean_sem(speech_stage0_rows)

    fig04_mechanics_path = data_root / paths_cfg["fig04_stepwise_mechanics"]
    if not fig04_mechanics_path.exists():
        raise FileNotFoundError(f"Required Fig. 4 mechanics artifact not found: {fig04_mechanics_path}")
    fig04_mechanics = dict(np.load(fig04_mechanics_path, allow_pickle=True))
    fig04_radii = np.asarray(fig04_mechanics["aligned_radius_deg"], dtype=float)
    match_idx = [int(np.argmin(np.abs(fig04_radii - radius))) for radius in LOCAL_RADII_DEG]
    step1_mean = np.asarray(fig04_mechanics["aligned_cum_mass_delta_mean"], dtype=float)[match_idx]
    step1_sem = np.asarray(fig04_mechanics["aligned_cum_mass_delta_sem"], dtype=float)[match_idx]

    legacy_bundle = _build_legacy_family_bundle(
        legacy_runs_root=legacy_runs_root,
        legacy_suffix=legacy_suffix,
        angles=angles,
    )
    family_curves_mean = legacy_bundle["clean_curves_mean"]
    family_curves_sem = legacy_bundle["clean_curves_sem"]
    family_alignment = _build_family_alignment_bundle(
        H_corr=H_corr,
        angles=angles,
        clean_confusion_rows=legacy_bundle["clean_confusion_rows"],
    )

    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}" for i in tick_positions]

    support_bridge_npz = output_dir / "fig05_performance_structure_panels" / "fig05_neighborhood_bridge.npz"
    support_bridge_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        support_bridge_npz,
        radii_deg=LOCAL_RADII_DEG.astype(np.float32),
        speech_stage0_mean=speech_stage0_mean.astype(np.float32),
        speech_stage0_sem=speech_stage0_sem.astype(np.float32),
        guided_step1_mean=np.asarray(step1_mean, dtype=np.float32),
        guided_step1_sem=np.asarray(step1_sem, dtype=np.float32),
        guided_final_mean=np.asarray(family_curves_mean["guided"], dtype=np.float32),
        guided_final_sem=np.asarray(family_curves_sem["guided"], dtype=np.float32),
        router_bypass_final_mean=np.asarray(family_curves_mean["router_bypass"], dtype=np.float32),
        router_bypass_final_sem=np.asarray(family_curves_sem["router_bypass"], dtype=np.float32),
        omp_final_mean=np.asarray(family_curves_mean["omp"], dtype=np.float32),
        omp_final_sem=np.asarray(family_curves_sem["omp"], dtype=np.float32),
        dense_final_mean=np.asarray(family_curves_mean["dense"], dtype=np.float32),
        dense_final_sem=np.asarray(family_curves_sem["dense"], dtype=np.float32),
        exact_accuracy_guided=np.float32(legacy_bundle["exact_accuracy_mean"]["guided"]),
        exact_accuracy_router=np.float32(legacy_bundle["exact_accuracy_mean"]["router_bypass"]),
        exact_accuracy_omp=np.float32(legacy_bundle["exact_accuracy_mean"]["omp"]),
        exact_accuracy_dense=np.float32(legacy_bundle["exact_accuracy_mean"]["dense"]),
        within15_guided=np.float32(legacy_bundle["within15_mean"]["guided"]),
        within15_router=np.float32(legacy_bundle["within15_mean"]["router_bypass"]),
        within15_omp=np.float32(legacy_bundle["within15_mean"]["omp"]),
        within15_dense=np.float32(legacy_bundle["within15_mean"]["dense"]),
    )

    fig = make_figure(
        width_mm=FIG05_GENERATOR["composite_width_mm"],
        height_mm=FIG05_GENERATOR["composite_height_mm"],
    )
    gs_outer = gridspec.GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=FIG05_COMPOSITE_GRID["height_ratios"],
        hspace=FIG05_COMPOSITE_GRID["hspace"],
        left=FIG05_COMPOSITE_GRID["left"],
        right=FIG05_COMPOSITE_GRID["right"],
        bottom=FIG05_COMPOSITE_GRID["bottom"],
        top=FIG05_COMPOSITE_GRID["top"],
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_outer[0, 0], width_ratios=FIG05_TOP_ROW["width_ratios"], wspace=FIG05_TOP_ROW["wspace"]
    )
    gs_mid = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[1, 0], width_ratios=FIG05_MIDDLE_ROW["width_ratios"], wspace=FIG05_MIDDLE_ROW["wspace"]
    )
    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[2, 0], width_ratios=FIG05_BOTTOM_ROW["width_ratios"], wspace=FIG05_BOTTOM_ROW["wspace"]
    )

    ax_a = _set_gid(fig.add_subplot(gs_top[0, 0]), "fig05.panel_a.main")
    _plot_support_curves(
        ax_a,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (
                f"speech stage-0 ({_mass_value_at_radius(speech_stage0_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                speech_stage0_mean,
                speech_stage0_sem,
                SEMANTIC_PALETTE["physics"],
                "--",
                1.05,
            ),
            (
                f"first guided step ({_mass_value_at_radius(step1_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                step1_mean,
                step1_sem,
                SEMANTIC_PALETTE["learned"],
                "-",
                1.20,
            ),
            (
                f"final guided prediction ({_mass_value_at_radius(family_curves_mean['guided'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["guided"],
                family_curves_sem["guided"],
                STYLE_COLORS["guide_line"],
                "-",
                1.10,
            ),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Neighborhood-preservation cascade",
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax_a, "a", x=-0.15, y=1.02)

    ax_b = _set_gid(fig.add_subplot(gs_top[0, 1]), "fig05.panel_b.main")
    _plot_support_curves(
        ax_b,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (
                f"{FAMILY_CONFIG['guided']['display']} ({_mass_value_at_radius(family_curves_mean['guided'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["guided"],
                family_curves_sem["guided"],
                FAMILY_CONFIG["guided"]["color"],
                "-",
                1.20,
            ),
            (
                f"{FAMILY_CONFIG['router_bypass']['display']} ({_mass_value_at_radius(family_curves_mean['router_bypass'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["router_bypass"],
                family_curves_sem["router_bypass"],
                FAMILY_CONFIG["router_bypass"]["color"],
                "-",
                1.05,
            ),
            (
                f"{FAMILY_CONFIG['omp']['display']} ({_mass_value_at_radius(family_curves_mean['omp'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["omp"],
                family_curves_sem["omp"],
                FAMILY_CONFIG["omp"]["color"],
                "-",
                1.05,
            ),
            (
                f"{FAMILY_CONFIG['dense']['display']} ({_mass_value_at_radius(family_curves_mean['dense'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["dense"],
                family_curves_sem["dense"],
                FAMILY_CONFIG["dense"]["color"],
                ":",
                1.05,
            ),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Coherent family neighborhood preservation",
        annotation_pt=annotation_pt,
        show_ylabel=False,
    )
    add_panel_label(ax_b, "b", x=-0.12, y=1.02)

    ax_c = _set_gid(fig.add_subplot(gs_top[0, 2]), "fig05.panel_c.main")
    _plot_exact_vs_local_scorecard(
        ax_c,
        exact_accuracy_mean=legacy_bundle["exact_accuracy_mean"],
        exact_accuracy_sem=legacy_bundle["exact_accuracy_sem"],
        within15_mean=legacy_bundle["within15_mean"],
        within15_sem=legacy_bundle["within15_sem"],
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax_c, "c", x=-0.16, y=1.02)

    ax_d_panel = _set_gid(fig.add_subplot(gs_mid[0, 0]), "fig05.panel_d.block")
    ax_d_panel.set_axis_off()
    ax_d_panel.text(
        0.43,
        1.005,
        "Final prediction locality by family",
        transform=ax_d_panel.transAxes,
        ha="center",
        va="bottom",
        fontsize=title_pt,
    )
    gs_d = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=gs_mid[0, 0],
        width_ratios=[1.0, 1.0, FIG05_CONFUSION_PAIR["colorbar_ratio"]],
        hspace=0.18,
        wspace=FIG05_CONFUSION_PAIR["wspace"],
    )
    family_order = ["guided", "router_bypass", "omp", "dense"]
    family_axes = []
    im_d = None
    for idx, family_key in enumerate(family_order):
        row = idx // 2
        col = idx % 2
        ax = _set_gid(fig.add_subplot(gs_d[row, col]), f"fig05.panel_d.{family_key}")
        family_axes.append(ax)
        im_d = _plot_confusion_matrix(
            ax,
            _row_normalize_matrix(legacy_bundle["clean_confusion_seed42"][family_key]),
            FAMILY_CONFIG[family_key]["display"],
            title_pt=title_pt - 0.2,
            axis_label_pt=axis_label_pt,
            tick_label_pt=tick_label_pt,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            xlabel="Predicted DOA" if row == 1 else None,
            ylabel="True DOA" if col == 0 else None,
            show_xticklabels=(row == 1),
            show_yticklabels=(col == 0),
            fontweight="bold",
        )
    cax_d = _set_gid(fig.add_subplot(gs_d[:, 2]), "fig05.panel_d.colorbar")
    cbar_d = plt.colorbar(im_d, cax=cax_d)
    cbar_d.ax.tick_params(labelsize=colorbar_tick_pt)
    add_panel_label(ax_d_panel, "d", x=-0.01, y=1.02)

    ax_e = _set_gid(fig.add_subplot(gs_mid[0, 1]), "fig05.panel_e.main")
    im_e = ax_e.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax_e.set_title("Measured neighborhood geometry", fontsize=title_pt)
    ax_e.set_xticks(tick_positions)
    ax_e.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax_e.set_yticks(tick_positions)
    ax_e.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax_e.set_xlabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_e.set_ylabel("Angle (\u00b0)", fontsize=axis_label_pt)
    ax_e.tick_params(axis="both", length=2)
    cax_e = fig.add_axes([
        ax_e.get_position().x1 + 0.004,
        ax_e.get_position().y0 + 0.02,
        0.008,
        ax_e.get_position().height - 0.04,
    ])
    cbar_e = plt.colorbar(im_e, cax=cax_e)
    cbar_e.set_ticks(FIG05_CORR_TICKS)
    cbar_e.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar_e.ax.tick_params(labelsize=colorbar_tick_pt)
    cbar_e.ax.set_title("Corr", fontsize=colorbar_label_pt, pad=2.0)
    add_panel_label(ax_e, "e", x=-0.12, y=1.02)

    ax_f = _set_gid(fig.add_subplot(gs_bottom[0, 0]), "fig05.panel_f.main")
    _plot_family_alignment_panel(
        ax_f,
        local_agreement_mean=family_alignment["local_agreement_mean"],
        local_agreement_sem=family_alignment["local_agreement_sem"],
        global_r_mean=family_alignment["global_r_mean"],
        global_r_sem=family_alignment["global_r_sem"],
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax_f, "f", x=-0.10, y=1.02)

    ax_g = _set_gid(fig.add_subplot(gs_bottom[0, 1]), "fig05.panel_g.main")
    _plot_snr_panel(
        ax_g,
        legacy_bundle["snr_summary"],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax_g.set_title("Noise robustness consequence", fontsize=title_pt)
    add_panel_label(ax_g, "g", x=-0.10, y=1.02)

    all_paths = save_outputs(fig, output_dir / "fig05_performance_structure", typography=typography)
    plt.close(fig)

    panel_dir = output_dir / "fig05_performance_structure_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    fig_a = make_figure(width_mm=FIG05_STANDALONE["a"]["width_mm"], height_mm=FIG05_STANDALONE["a"]["height_mm"])
    ax = fig_a.add_subplot(111)
    _plot_support_curves(
        ax,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (
                f"speech stage-0 ({_mass_value_at_radius(speech_stage0_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                speech_stage0_mean,
                speech_stage0_sem,
                SEMANTIC_PALETTE["physics"],
                "--",
                1.05,
            ),
            (
                f"first guided step ({_mass_value_at_radius(step1_mean, LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                step1_mean,
                step1_sem,
                SEMANTIC_PALETTE["learned"],
                "-",
                1.20,
            ),
            (
                f"final guided prediction ({_mass_value_at_radius(family_curves_mean['guided'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["guided"],
                family_curves_sem["guided"],
                STYLE_COLORS["guide_line"],
                "-",
                1.10,
            ),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Neighborhood-preservation cascade",
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax, "a")
    fig_a.subplots_adjust(**FIG05_STANDALONE["a"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_a, panel_dir / "fig05_panel_a_neighborhood_cascade", typography=typography))
    plt.close(fig_a)

    fig_b = make_figure(width_mm=FIG05_STANDALONE["b"]["width_mm"], height_mm=FIG05_STANDALONE["b"]["height_mm"])
    ax = fig_b.add_subplot(111)
    _plot_support_curves(
        ax,
        radii_deg=LOCAL_RADII_DEG,
        curves=[
            (
                f"{FAMILY_CONFIG['guided']['display']} ({_mass_value_at_radius(family_curves_mean['guided'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["guided"],
                family_curves_sem["guided"],
                FAMILY_CONFIG["guided"]["color"],
                "-",
                1.20,
            ),
            (
                f"{FAMILY_CONFIG['router_bypass']['display']} ({_mass_value_at_radius(family_curves_mean['router_bypass'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["router_bypass"],
                family_curves_sem["router_bypass"],
                FAMILY_CONFIG["router_bypass"]["color"],
                "-",
                1.05,
            ),
            (
                f"{FAMILY_CONFIG['omp']['display']} ({_mass_value_at_radius(family_curves_mean['omp'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["omp"],
                family_curves_sem["omp"],
                FAMILY_CONFIG["omp"]["color"],
                "-",
                1.05,
            ),
            (
                f"{FAMILY_CONFIG['dense']['display']} ({_mass_value_at_radius(family_curves_mean['dense'], LOCAL_RADII_DEG, LOCAL_CALLOUT_DEG):.2f})",
                family_curves_mean["dense"],
                family_curves_sem["dense"],
                FAMILY_CONFIG["dense"]["color"],
                ":",
                1.05,
            ),
        ],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        title="Coherent family neighborhood preservation",
        annotation_pt=annotation_pt,
        show_ylabel=False,
    )
    add_panel_label(ax, "b")
    fig_b.subplots_adjust(**FIG05_STANDALONE["b"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_b, panel_dir / "fig05_panel_b_family_neighborhood", typography=typography))
    plt.close(fig_b)

    fig_c = make_figure(width_mm=FIG05_STANDALONE["c"]["width_mm"], height_mm=FIG05_STANDALONE["c"]["height_mm"])
    ax = fig_c.add_subplot(111)
    _plot_exact_vs_local_scorecard(
        ax,
        exact_accuracy_mean=legacy_bundle["exact_accuracy_mean"],
        exact_accuracy_sem=legacy_bundle["exact_accuracy_sem"],
        within15_mean=legacy_bundle["within15_mean"],
        within15_sem=legacy_bundle["within15_sem"],
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax, "c")
    fig_c.subplots_adjust(**FIG05_STANDALONE["c"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_c, panel_dir / "fig05_panel_c_exact_vs_local_scorecard", typography=typography))
    plt.close(fig_c)

    fig_d = make_figure(width_mm=FIG05_STANDALONE["d"]["width_mm"], height_mm=FIG05_STANDALONE["d"]["height_mm"])
    fig_d_grid = FIG05_STANDALONE["d"]["grid"]
    gs_ds = gridspec.GridSpec(
        2,
        3,
        figure=fig_d,
        width_ratios=[1.0, 1.0, fig_d_grid["colorbar_ratio"]],
        height_ratios=[1.0, 1.0],
        wspace=fig_d_grid["wspace"],
        hspace=0.18,
        left=fig_d_grid["left"],
        right=fig_d_grid["right"],
        bottom=fig_d_grid["bottom"],
        top=fig_d_grid["top"],
    )
    family_order = ["guided", "router_bypass", "omp", "dense"]
    im_d = None
    first_ax = None
    for idx, family_key in enumerate(family_order):
        row = idx // 2
        col = idx % 2
        ax = fig_d.add_subplot(gs_ds[row, col], sharex=first_ax, sharey=first_ax) if first_ax else fig_d.add_subplot(gs_ds[row, col])
        if first_ax is None:
            first_ax = ax
            add_panel_label(ax, "d")
        im_d = _plot_confusion_matrix(
            ax,
            _row_normalize_matrix(legacy_bundle["clean_confusion_seed42"][family_key]),
            FAMILY_CONFIG[family_key]["display"],
            title_pt=title_pt - 0.2,
            axis_label_pt=axis_label_pt,
            tick_label_pt=tick_label_pt,
            tick_positions=tick_positions,
            tick_labels=tick_labels,
            xlabel="Predicted DOA" if row == 1 else None,
            ylabel="True DOA" if col == 0 else None,
            show_xticklabels=(row == 1),
            show_yticklabels=(col == 0),
            fontweight="bold",
        )
    cax = fig_d.add_subplot(gs_ds[:, 2])
    cbar = plt.colorbar(im_d, cax=cax)
    cbar.ax.tick_params(labelsize=colorbar_tick_pt)
    all_paths.extend(save_outputs(fig_d, panel_dir / "fig05_panel_d_prediction_locality", typography=typography))
    plt.close(fig_d)

    fig_e = make_figure(width_mm=FIG05_STANDALONE["e"]["width_mm"], height_mm=FIG05_STANDALONE["e"]["height_mm"])
    ax = fig_e.add_subplot(111)
    im = ax.imshow(H_corr, cmap="RdBu_r", aspect="equal", norm=FIG05_CORR_NORM)
    ax.set_title("Measured neighborhood geometry", fontsize=title_pt, fontweight="bold")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=tick_label_pt)
    ax.set_xlabel("Angle (\u00b0)")
    ax.set_ylabel("Angle (\u00b0)")
    add_panel_label(ax, "e")
    fig_e.subplots_adjust(**FIG05_STANDALONE["e"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_e, panel_dir / "fig05_panel_e_measured_structure", typography=typography))
    plt.close(fig_e)

    fig_f = make_figure(width_mm=FIG05_STANDALONE["f"]["width_mm"], height_mm=FIG05_STANDALONE["f"]["height_mm"])
    ax = fig_f.add_subplot(111)
    _plot_family_alignment_panel(
        ax,
        local_agreement_mean=family_alignment["local_agreement_mean"],
        local_agreement_sem=family_alignment["local_agreement_sem"],
        global_r_mean=family_alignment["global_r_mean"],
        global_r_sem=family_alignment["global_r_sem"],
        title_pt=title_pt,
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
        annotation_pt=annotation_pt,
    )
    add_panel_label(ax, "f")
    fig_f.subplots_adjust(**FIG05_STANDALONE["f"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_f, panel_dir / "fig05_panel_f_family_alignment", typography=typography))
    plt.close(fig_f)

    fig_g = make_figure(width_mm=FIG05_STANDALONE["g"]["width_mm"], height_mm=FIG05_STANDALONE["g"]["height_mm"])
    ax = fig_g.add_subplot(111)
    _plot_snr_panel(
        ax,
        legacy_bundle["snr_summary"],
        axis_label_pt=axis_label_pt,
        tick_label_pt=tick_label_pt,
        legend_pt=legend_pt,
    )
    ax.set_title("Noise robustness consequence", fontsize=title_pt, fontweight="bold")
    add_panel_label(ax, "g")
    fig_g.subplots_adjust(**FIG05_STANDALONE["g"]["subplots_adjust"])
    all_paths.extend(save_outputs(fig_g, panel_dir / "fig05_panel_g_noise_robustness", typography=typography))
    plt.close(fig_g)

    manifest = _save_panel_manifest(
        panel_dir,
        [
            {
                "panel_id": "a",
                "title": "Neighborhood-preservation cascade",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_a_neighborhood_cascade.pdf",
                "provenance_mode": "data_backed",
                "description": "Shared radius-based mass-within-neighborhood curves linking speech stage-0 support, first guided-step contraction, and final guided prediction locality.",
            },
            {
                "panel_id": "b",
                "title": "Coherent family neighborhood preservation",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_b_family_neighborhood.pdf",
                "provenance_mode": "data_backed",
                "description": "Family-level final clean mass-within-radius curves aggregated directly from the coherent legacy babble family runs.",
            },
            {
                "panel_id": "c",
                "title": "Coherent family exact-vs-local scorecard",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_c_exact_vs_local_scorecard.pdf",
                "provenance_mode": "data_backed",
                "description": "Compact exact-versus-local family scorecard connecting exact clean consequence to within-15-degree locality on the same coherent legacy protocol.",
            },
            {
                "panel_id": "d",
                "title": "Final prediction locality by family",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_d_prediction_locality.pdf",
                "provenance_mode": "data_backed",
                "description": "Row-normalized clean confusion grid for the coherent legacy family bundle, showing how prediction locality broadens from guided solver through router-bypass and OMP to dense routing.",
            },
            {
                "panel_id": "e",
                "title": "Measured neighborhood geometry",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_e_measured_structure.pdf",
                "provenance_mode": "data_backed",
                "description": "Angle-angle correlation matrix of calibrated H, showing the near-diagonal local band that defines the measured neighborhood geometry.",
            },
            {
                "panel_id": "f",
                "title": "Family-to-measured neighborhood alignment",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_f_family_alignment.pdf",
                "provenance_mode": "data_backed",
                "description": "Family-wide alignment summary showing local-band agreement bars against the measured neighborhood, with global matrix correlation shown as a secondary point estimate.",
            },
            {
                "panel_id": "g",
                "title": "Noise robustness consequence",
                "asset_path": "figures/output/fig05_performance_structure_panels/fig05_panel_g_noise_robustness.pdf",
                "provenance_mode": "data_backed",
                "description": "Five-seed noisy babble sweep reconstructed from the coherent legacy family bundle, showing that the same ordering remains visible under SNR degradation.",
            },
        ],
        typography,
    )
    all_paths.append(manifest)
    all_paths.append(support_bridge_npz)
    return all_paths


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[2]
    paths = load_paths()
    generate(REPO_ROOT, REPO_ROOT / paths["output_dir"])
