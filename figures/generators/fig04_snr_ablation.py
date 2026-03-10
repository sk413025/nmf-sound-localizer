"""Figure 4 — Component Ablation + SNR Robustness.

Panel (a): Ablation strip chart (SNR=Inf slice) — jittered dots + mean bar.
Panel (b): Multi-variant SNR degradation curves — mean +/- 1 std shading.

Data source: babble Speech260 full sweep (245 runs: 7 variants x 7 SNR x 5 seeds).
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from figures.style import (
    set_nature_rcparams,
    make_figure,
    save_outputs,
    add_panel_label,
    load_paths,
    DOUBLE_COL_MM,
    PALETTE_WONG,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VARIANT_ORDER = [
    "Baseline",
    "No Transformer",
    "Fixed Heuristic",
    "No Type Bias",
    "Dense Routing",
    "G-Teacher",
    "G-Fixed",
]

VARIANT_LABELS_SHORT = [
    "Baseline",
    "No Transf.",
    "Fixed Heur.",
    "No Type Bias",
    "Dense Rout.",
    "G-Teacher",
    "G-Fixed",
]

SNR_ORDER = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
SNR_DISPLAY = ["0", "5", "10", "15", "20", "30", "Inf"]

VARIANT_MAP = {
    "baseline": "Baseline",
    "no_transformer": "No Transformer",
    "g_routing": "Fixed Heuristic",
    "no_type_bias": "No Type Bias",
    "disable_sparsity": "Dense Routing",
    "g_teacher": "G-Teacher",
    "g_fixed_F": "G-Fixed",
}

DIR_RE = re.compile(
    r"ablate_speech260_(?P<variant>.+?)_snr(?P<snr>[^_]+)_seed(?P<seed>\d+)_ep20_"
)


# ---------------------------------------------------------------------------
# Data gathering (merged from gather_fig4_data.py)
# ---------------------------------------------------------------------------

def _gather_data(results_dir: Path, sweep_suffix: str) -> dict:
    pattern = str(results_dir / f"ablate_speech260_*_{sweep_suffix}")
    run_dirs = sorted(glob.glob(pattern))

    ablation: dict[str, list[float]] = {}
    snr: dict[str, dict[str, list[float]]] = {}

    for run_dir in run_dirs:
        dirname = Path(run_dir).name
        m = DIR_RE.search(dirname)
        if not m:
            continue

        variant_code = m.group("variant")
        snr_level = m.group("snr")
        display_name = VARIANT_MAP.get(variant_code)
        if display_name is None:
            continue

        metrics_path = Path(run_dir) / "metrics.npz"
        if not metrics_path.exists():
            continue
        try:
            data = np.load(metrics_path, allow_pickle=True)
            acc = float(data["best_accuracy"])
        except Exception:
            continue

        snr_key = "Clean" if snr_level == "Inf" else snr_level

        if snr_level == "Inf":
            ablation.setdefault(display_name, []).append(acc)
        snr.setdefault(display_name, {}).setdefault(snr_key, []).append(acc)

    return {"ablation": ablation, "snr": snr}


def _load_or_gather(data_root: Path, paths_cfg: dict) -> dict:
    """Load pre-aggregated JSON if available, otherwise gather from sweep dirs."""
    agg_path = data_root / paths_cfg["ablation_sweep"]["aggregated_json"]
    if agg_path.exists():
        with open(agg_path) as f:
            return json.load(f)

    results_dir = data_root / paths_cfg["ablation_sweep"]["results_dir"]
    sweep_suffix = paths_cfg["ablation_sweep"]["sweep_suffix"]
    if results_dir.exists():
        return _gather_data(results_dir, sweep_suffix)

    return {"ablation": {}, "snr": {}}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(data_root: Path, output_dir: Path) -> list[Path]:
    """Generate Figure 4 panels. Returns list of output file paths."""
    set_nature_rcparams(base_fontsize=7)

    paths_cfg = load_paths()
    raw_data = _load_or_gather(data_root, paths_cfg)

    if not raw_data.get("ablation") and not raw_data.get("snr"):
        print("[fig04] SKIP: No ablation/SNR data found")
        return []

    ablation_vals = [raw_data["ablation"].get(k, []) for k in VARIANT_ORDER]
    snr_data = raw_data.get("snr", {})

    fig = make_figure(width_mm=DOUBLE_COL_MM, height_mm=70)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Panel A: Ablation strip chart
    x_pos = np.arange(len(VARIANT_ORDER))
    rng = np.random.default_rng(42)

    for i, vals in enumerate(ablation_vals):
        y = np.array(vals)
        x = rng.normal(i, 0.04, size=len(y))
        ax1.scatter(x, y, s=10, alpha=0.7, color="#377eb8", edgecolors="none", zorder=3)
        if len(y) > 0:
            ax1.hlines(np.mean(y), i - 0.2, i + 0.2, colors="k", linewidth=1.2, zorder=4)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(VARIANT_LABELS_SHORT, rotation=45, ha="right")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    add_panel_label(ax1, "a")

    # Panel B: Multi-variant SNR degradation curves
    x_snr = np.arange(len(SNR_ORDER))

    for idx, variant in enumerate(VARIANT_ORDER):
        variant_levels = snr_data.get(variant, {})
        means, stds = [], []
        for snr_key in SNR_ORDER:
            vals = variant_levels.get(snr_key, [])
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals) if vals else 0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        color = PALETTE_WONG[idx % len(PALETTE_WONG)]
        ax2.plot(x_snr, means_arr, "-o", color=color, markersize=3, linewidth=1, label=variant, zorder=3)
        ax2.fill_between(x_snr, means_arr - stds_arr, means_arr + stds_arr, color=color, alpha=0.15, zorder=1)

    ax2.set_xticks(x_snr)
    ax2.set_xticklabels(SNR_DISPLAY)
    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=5, frameon=False, loc="lower left")
    add_panel_label(ax2, "b")

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.22, top=0.95, wspace=0.30)
    all_paths = save_outputs(fig, output_dir / "fig04_snr_ablation")
    plt.close(fig)

    print(f"[fig04] Generated {len(all_paths)} files")
    return all_paths
