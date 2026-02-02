#!/usr/bin/env python3
"""
Compute multi-sample averaged angle-index distributions for Figure 5 B3 frequency panels.
Outputs raw OMP/QK means and derived QK-OMP, QK/OMP, plus line plots.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import NA_matplotlib_guild as na_style

RUN_DIR = Path(
    "/Users/jnrle/Documents/LDVReorientation/worktrees/nature-comm-repro/results/"
    "omp_transformer_speech260_trainval_split_full_20251115_082341"
)
OUT_DIR = Path(__file__).parent / "results" / "results_figure5_v9" / "freq_angle_avg"

FREQ_TARGETS_HZ = (500, 1000, 2000)
FREQ_BANDWIDTH_HZ = 100
ANGLE_TARGET_DEG = 145

# Style setup
na_style.set_nature_rcparams(base_fontsize=7)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

PLOT_W_MM = 90
PLOT_H_MM = 60


def mm_to_inch(mm):
    return mm / 25.4


def save_plot(fig, out_prefix, dpi=300):
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png", dpi=dpi)


def to_serializable(arr):
    return [None if not np.isfinite(x) else float(x) for x in arr]


def build_freq_masks(freqs):
    masks = {}
    half_bw = FREQ_BANDWIDTH_HZ / 2.0
    for target in FREQ_TARGETS_HZ:
        mask = (freqs >= target - half_bw) & (freqs <= target + half_bw)
        if not np.any(mask):
            raise ValueError(f"No bins for {target} Hz within {FREQ_BANDWIDTH_HZ} Hz bandwidth.")
        masks[target] = mask
    return masks


def compute_group_means(indices, Y_all, scores_atoms_all, D, masks, n_experts):
    Y_sel = Y_all[indices].astype(np.float64, copy=False)
    scores_sel = scores_atoms_all[indices].reshape(len(indices), -1).astype(np.float64, copy=False)
    group_data = {}
    for target, mask in masks.items():
        D_band = D[mask, :].astype(np.float64, copy=False)
        Y_band = Y_sel[:, mask]
        omp_all = np.abs(Y_band @ D_band)
        qk_all = np.abs(omp_all * scores_sel)
        omp_all = omp_all.reshape(len(indices), n_experts, 8).sum(axis=2)
        qk_all = qk_all.reshape(len(indices), n_experts, 8).sum(axis=2)
        omp_mean = omp_all.mean(axis=0)
        qk_mean = qk_all.mean(axis=0)
        diff = qk_mean - omp_mean
        ratio = np.where(omp_mean > 0, qk_mean / omp_mean, np.nan)
        group_data[str(target)] = {
            "omp_raw_mean": omp_mean,
            "qk_raw_mean": qk_mean,
            "qk_minus_omp": diff,
            "qk_over_omp": ratio,
        }
    return group_data


def setup_axes():
    fig, ax = plt.subplots(figsize=(mm_to_inch(PLOT_W_MM), mm_to_inch(PLOT_H_MM)))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.9)
    return fig, ax


def set_angle_ticks(ax, angles_deg):
    tick_indices = [0, 9, 18, 27, 36]
    tick_indices = [idx for idx in tick_indices if idx < len(angles_deg)]
    if not tick_indices:
        return
    tick_vals = [angles_deg[idx] for idx in tick_indices]
    tick_labels = [f"{int(round(angles_deg[idx]))}°" for idx in tick_indices]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)


def add_ground_truth_marker(ax, ground_truth_angle):
    if ground_truth_angle is None:
        return
    ax.axvline(ground_truth_angle, color="lime", linewidth=1.2, alpha=0.5, zorder=0)


def plot_raw(ax, angles_deg, omp, qk, title, ground_truth_angle):
    add_ground_truth_marker(ax, ground_truth_angle)
    ax.plot(angles_deg, omp, color="coral", linewidth=1.0, alpha=0.7, label="Traditional OMP")
    ax.plot(angles_deg, qk, color="darkgreen", linewidth=1.2, alpha=0.9, label="Physics-Aware AI")
    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Raw Score")
    ax.set_xlim(angles_deg[0], angles_deg[-1])
    set_angle_ticks(ax, angles_deg)
    ax.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


def plot_diff(ax, angles_deg, diff, title, ground_truth_angle):
    add_ground_truth_marker(ax, ground_truth_angle)
    ax.plot(angles_deg, diff, color="black", linewidth=1.0, alpha=0.9, label="QK - OMP")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("QK - OMP")
    ax.set_xlim(angles_deg[0], angles_deg[-1])
    set_angle_ticks(ax, angles_deg)
    ax.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


def plot_ratio(ax, angles_deg, ratio, title, ground_truth_angle):
    add_ground_truth_marker(ax, ground_truth_angle)
    ax.plot(angles_deg, ratio, color="slategray", linewidth=1.0, alpha=0.9, label="QK / OMP")
    ax.axhline(1.0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("QK / OMP")
    ax.set_xlim(angles_deg[0], angles_deg[-1])
    set_angle_ticks(ax, angles_deg)
    ax.grid(True, alpha=0.2, linestyle="--", color="gray")
    ax.legend(frameon=False, loc="upper right", handlelength=1.2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    routing = np.load(RUN_DIR / "modal_routing_val.npz")
    dict_data = np.load(RUN_DIR / "dictionary.npz", allow_pickle=True)

    D = dict_data["D"]
    angles = dict_data["angles"]
    labels = routing["labels"]
    Y_all = routing["Y_val"]
    scores_atoms_all = routing["scores_atoms"]

    angle_idx = int(np.argmin(np.abs(angles - ANGLE_TARGET_DEG)))
    if float(angles[angle_idx]) != float(ANGLE_TARGET_DEG):
        raise ValueError(
            f"ANGLE_TARGET_DEG {ANGLE_TARGET_DEG} not found in angles. "
            f"Nearest is {angles[angle_idx]:.1f}."
        )

    all_indices = np.arange(labels.shape[0])
    angle_indices = np.where(labels == angle_idx)[0]
    if angle_indices.size == 0:
        raise ValueError(f"No samples found for angle index {angle_idx} ({ANGLE_TARGET_DEG} deg).")

    freqs = np.linspace(300, 3000, D.shape[0])
    masks = build_freq_masks(freqs)

    n_experts = D.shape[1] // 8
    groups = {
        "all": compute_group_means(all_indices, Y_all, scores_atoms_all, D, masks, n_experts),
        f"angle_{int(ANGLE_TARGET_DEG)}": compute_group_means(
            angle_indices, Y_all, scores_atoms_all, D, masks, n_experts
        ),
    }

    angle_values = angles.astype(float)
    gt_angle = float(angles[angle_idx])

    # Save JSON
    payload = {
        "run_dir": str(RUN_DIR),
        "angle_target_deg": float(ANGLE_TARGET_DEG),
        "angle_target_idx": int(angle_idx),
        "angle_target_value_deg": float(angles[angle_idx]),
        "freq_bandwidth_hz": FREQ_BANDWIDTH_HZ,
        "freq_targets_hz": list(FREQ_TARGETS_HZ),
        "angle_count": int(n_experts),
        "angles_deg": to_serializable(angle_values),
        "ground_truth_angle_deg": gt_angle,
        "groups": {},
    }

    for group_name, group_data in groups.items():
        payload_group = {
            "n_samples": int(all_indices.size if group_name == "all" else angle_indices.size),
            "panels": {},
        }
        for target, stats in group_data.items():
            payload_group["panels"][target] = {
                "omp_raw_mean": to_serializable(stats["omp_raw_mean"]),
                "qk_raw_mean": to_serializable(stats["qk_raw_mean"]),
                "qk_minus_omp": to_serializable(stats["qk_minus_omp"]),
                "qk_over_omp": to_serializable(stats["qk_over_omp"]),
            }
        payload["groups"][group_name] = payload_group

    json_path = OUT_DIR / "Fig5_B3_freq_angle_avg_raw_values.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    # Plots
    for group_name, group_data in groups.items():
        group_label = "All samples" if group_name == "all" else f"Angle {ANGLE_TARGET_DEG}°"
        group_gt_angle = gt_angle
        for target, stats in group_data.items():
            freq_label = f"{target} Hz"

            fig, ax = setup_axes()
            plot_raw(
                ax,
                angle_values,
                stats["omp_raw_mean"],
                stats["qk_raw_mean"],
                f"{group_label} | {freq_label} | Raw",
                group_gt_angle,
            )
            save_plot(fig, OUT_DIR / f"Fig5_B3_FREQ_{target}_AVG_{group_name}_RAW")
            plt.close(fig)

            fig, ax = setup_axes()
            plot_diff(
                ax,
                angle_values,
                stats["qk_minus_omp"],
                f"{group_label} | {freq_label} | QK - OMP",
                group_gt_angle,
            )
            save_plot(fig, OUT_DIR / f"Fig5_B3_FREQ_{target}_AVG_{group_name}_QK_MINUS_OMP")
            plt.close(fig)

            fig, ax = setup_axes()
            plot_ratio(
                ax,
                angle_values,
                stats["qk_over_omp"],
                f"{group_label} | {freq_label} | QK / OMP",
                group_gt_angle,
            )
            save_plot(fig, OUT_DIR / f"Fig5_B3_FREQ_{target}_AVG_{group_name}_QK_OVER_OMP")
            plt.close(fig)

    print(f"Wrote JSON: {json_path}")
    print(f"Plots saved under: {OUT_DIR}")


if __name__ == "__main__":
    main()
