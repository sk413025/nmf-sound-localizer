#!/usr/bin/env python3
"""Build self-contained assets for the Fig. 6 factor-importance note."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


REPORT_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPORT_DIR.parents[2]
ASSETS_DIR = REPORT_DIR / "assets"

FACTOR_AUDIT_DIR = REPO_ROOT / "results" / "fig06_universal_equation_factor_audit_20260327_173647"
GEOMETRY_DIR = REPO_ROOT / "results" / "fig06_cross_material_geometry_20260327_171243"
FIG06_PREVIEWS = REPO_ROOT / "figures" / "review_artifacts" / "fig06" / "panel_previews"

COPY_ASSETS = {
    FIG06_PREVIEWS / "c.png": ASSETS_DIR / "fig06_panel_c_low_rank_context.png",
    FIG06_PREVIEWS / "d.png": ASSETS_DIR / "fig06_panel_d_screening_context.png",
    FIG06_PREVIEWS / "e.png": ASSETS_DIR / "fig06_panel_e_frequency_structure_context.png",
    FACTOR_AUDIT_DIR / "mae_factor_importance.png": ASSETS_DIR / "mae_factor_importance.png",
    FACTOR_AUDIT_DIR / "top1_factor_importance.png": ASSETS_DIR / "top1_factor_importance_caution.png",
    GEOMETRY_DIR / "top3_subspace_overlap_heatmap.png": ASSETS_DIR / "top3_subspace_overlap_heatmap.png",
}

FACTOR_TIERS = [
    {
        "family": "Geometry separability",
        "tier": "Tier 1",
        "representative_metrics": "full_far_mean; band_local_far_gap; rank95_far_collision_gt_0_8",
        "mae_stability": "Strong",
        "descriptive_support": "Strong",
        "family_level_support": "Mixed",
        "manuscript_safety": "High",
        "internal_note": "Current leading candidate driver. The main signal comes from non-local overlap, collision, and local-vs-far separation rather than rank alone.",
    },
    {
        "family": "Cross-material subspace overlap",
        "tier": "Tier 2",
        "representative_metrics": "mean_top3_subspace_overlap_to_others",
        "mae_stability": "Strong",
        "descriptive_support": "Moderate",
        "family_level_support": "Weak",
        "manuscript_safety": "Medium",
        "internal_note": "Stable in elastic-net ranking, but family-level held-out support is still mixed. Good internal hypothesis, not yet a main-paper law.",
    },
    {
        "family": "Dictionary conditioning",
        "tier": "Tier 2",
        "representative_metrics": "raw_condition_number",
        "mae_stability": "Strong",
        "descriptive_support": "Weak",
        "family_level_support": "Moderate",
        "manuscript_safety": "Medium",
        "internal_note": "Conditioning still matters, but less than separability. It behaves more like a secondary structural constraint than the main driver.",
    },
    {
        "family": "Low-rankness itself",
        "tier": "Tier 3",
        "representative_metrics": "effective_rank_centered_mag",
        "mae_stability": "Weak",
        "descriptive_support": "Weak",
        "family_level_support": "Weak",
        "manuscript_safety": "High",
        "internal_note": "Necessary background substrate, but weak as a standalone predictor of encoding quality.",
    },
    {
        "family": "Raw coherence",
        "tier": "Tier 4",
        "representative_metrics": "raw_mean_coherence",
        "mae_stability": "Unsupported",
        "descriptive_support": "Unsupported",
        "family_level_support": "Unsupported",
        "manuscript_safety": "Low",
        "internal_note": "Should not be used as the main summary variable for Fig. 6. It is inconsistent with downstream outcome and was dropped by collinearity filtering.",
    },
]

VALUE_COLOR = {
    "Tier 1": "#1b8a5a",
    "Tier 2": "#5b8def",
    "Tier 3": "#d8a44d",
    "Tier 4": "#c85c5c",
    "Strong": "#2f9e44",
    "Moderate": "#4c6ef5",
    "Mixed": "#f08c00",
    "Weak": "#adb5bd",
    "Unsupported": "#868e96",
    "High": "#2b8a3e",
    "Medium": "#4263eb",
    "Low": "#868e96",
}


def copy_existing_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in COPY_ASSETS.items():
        shutil.copy2(src, dst)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_baseline_vs_model_summary() -> None:
    top1 = load_json(FACTOR_AUDIT_DIR / "top1_model_summary.json")
    mae = load_json(FACTOR_AUDIT_DIR / "mae_model_summary.json")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    colors = ["#2b8a3e", "#c92a2a"]

    top1_values = [top1["pooled_oof_log_loss"], top1["baseline_oof_log_loss"]]
    axes[0].bar(["Model", "Intercept"], top1_values, color=colors, width=0.62)
    axes[0].set_ylabel("Pooled Top-1 log-loss")
    axes[0].set_title("Top-1: no gain over intercept")
    axes[0].grid(True, axis="y", alpha=0.2)
    for idx, value in enumerate(top1_values):
        axes[0].text(idx, value + 0.008, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    axes[0].text(
        0.5,
        max(top1_values) + 0.045,
        f"Accuracy tie: {top1['pooled_oof_accuracy']:.4f}",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#495057",
    )

    mae_values = [mae["pooled_oof_mae"], mae["baseline_oof_mae"]]
    axes[1].bar(["Model", "Intercept"], mae_values, color=colors, width=0.62)
    axes[1].set_ylabel("Pooled MAE (deg)")
    axes[1].set_title("MAE: clear gain over intercept")
    axes[1].grid(True, axis="y", alpha=0.2)
    for idx, value in enumerate(mae_values):
        axes[1].text(idx, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    delta = mae["baseline_oof_mae"] - mae["pooled_oof_mae"]
    axes[1].text(
        0.5,
        max(mae_values) + 0.55,
        f"Gain vs intercept: {delta:.2f} deg",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#2b8a3e",
    )

    fig.suptitle("Outcome choice for factor ranking", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "baseline_vs_model_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_factor_tier_summary() -> None:
    csv_path = REPORT_DIR / "factor_tier_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "tier",
                "representative_metrics",
                "mae_stability",
                "descriptive_support",
                "family_level_support",
                "manuscript_safety",
                "internal_note",
            ],
        )
        writer.writeheader()
        writer.writerows(FACTOR_TIERS)


def plot_factor_tier_matrix() -> None:
    columns = [
        ("Tier", "tier"),
        ("MAE stability", "mae_stability"),
        ("Descriptive support", "descriptive_support"),
        ("Family support", "family_level_support"),
        ("Manuscript safety", "manuscript_safety"),
    ]
    rows = FACTOR_TIERS

    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    ax.set_xlim(0, len(columns) + 1)
    ax.set_ylim(0, len(rows) + 1)
    ax.axis("off")

    ax.text(0.05, len(rows) + 0.65, "Current factor families for direction-encoding capacity", fontsize=13, weight="bold")
    ax.text(
        0.05,
        len(rows) + 0.35,
        "Qualitative evidence matrix. This is a tiered evidence view, not a causal percentage decomposition.",
        fontsize=9,
        color="#495057",
    )

    for col_idx, (label, _) in enumerate(columns, start=1):
        ax.text(col_idx + 0.5, len(rows), label, ha="center", va="center", fontsize=10, weight="bold")

    for row_idx, row in enumerate(rows):
        y = len(rows) - 1 - row_idx
        ax.text(0.05, y + 0.5, row["family"], ha="left", va="center", fontsize=10, weight="bold")
        for col_idx, (_, key) in enumerate(columns, start=1):
            value = row[key]
            rect = Rectangle((col_idx, y), 1.0, 1.0, facecolor=VALUE_COLOR[value], edgecolor="white", linewidth=2.0)
            ax.add_patch(rect)
            ax.text(
                col_idx + 0.5,
                y + 0.5,
                value,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value not in {"Weak", "Unsupported"} else "#212529",
                weight="bold",
            )

    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "factor_tier_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report_numbers() -> None:
    top1 = load_json(FACTOR_AUDIT_DIR / "top1_model_summary.json")
    mae = load_json(FACTOR_AUDIT_DIR / "mae_model_summary.json")
    payload = {
        "top1": {
            "pooled_oof_log_loss": top1["pooled_oof_log_loss"],
            "baseline_oof_log_loss": top1["baseline_oof_log_loss"],
            "pooled_oof_accuracy": top1["pooled_oof_accuracy"],
            "baseline_oof_accuracy": top1["baseline_oof_accuracy"],
        },
        "mae": {
            "pooled_oof_mae": mae["pooled_oof_mae"],
            "baseline_oof_mae": mae["baseline_oof_mae"],
        },
        "factor_tiers": FACTOR_TIERS,
    }
    (REPORT_DIR / "report_numbers.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    copy_existing_assets()
    plot_baseline_vs_model_summary()
    write_factor_tier_summary()
    plot_factor_tier_matrix()
    write_report_numbers()
    print("[fig06-factor-report] assets ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
