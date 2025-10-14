#!/usr/bin/env python3
"""
Evaluate and Compare Multi-Modal ICL Experiments
================================================

Analyzes results from comparison experiments:
- Baseline vs Multi-Modal performance
- Token ordering ablations
- ICL effectiveness
- Token budget impact

Generates:
- Accuracy comparison tables
- Attention weight visualizations
- Loss curve plots
- Statistical significance tests
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# ============================================================================
# Canonical Palette (from AGENTS.md)
# ============================================================================
NATURE_ACCENT_PALETTE = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#56B4E9",  # Sky blue
    "#CC79A7",  # Reddish purple
    "#009E73",  # Bluish green
    "#F0E442",  # Yellow
    "#D55E00",  # Vermillion (warnings)
    "#999999",  # Neutral grey
]

sns.set_theme(context="talk", style="white")
sns.set_palette(NATURE_ACCENT_PALETTE)

mpl.rcParams.update({
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.dpi": 150,
    "text.color": "#000000",
    "axes.edgecolor": "#4D4D4D",
    "axes.labelcolor": "#000000",
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.grid": False,
    "axes.linewidth": 1.0,
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
})


def enable_light_ygrid():
    """Enable light y-grid as per AGENTS.md standards."""
    ax = plt.gca()
    ax.yaxis.grid(True, color="#E6E6E6", linewidth=0.6)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


# ============================================================================
# Metrics Extraction
# ============================================================================

def load_experiment_results(exp_dir: Path) -> Dict:
    """Load all metrics from an experiment directory."""
    results = {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "metrics": {},
        "checkpoints": [],
    }
    
    # Load training logs
    log_file = exp_dir / "training_log.json"
    if log_file.exists():
        with open(log_file) as f:
            results["metrics"] = json.load(f)
    
    # Find checkpoints
    results["checkpoints"] = sorted(exp_dir.glob("*.pth"))
    
    return results


def compute_top_k_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Compute Top-K accuracy."""
    _, topk_indices = torch.topk(logits, k=k, dim=-1)
    correct = (topk_indices == targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()


def evaluate_model(model_path: Path, data_loader, device: str = "cpu") -> Dict:
    """Evaluate a trained model on test data."""
    # Load model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # TODO: Instantiate model and load state_dict
    # This requires access to model architecture
    
    metrics = {
        "top1_acc": 0.0,
        "top3_acc": 0.0,
        "top5_acc": 0.0,
        "mean_loss": 0.0,
    }
    
    # Placeholder for actual evaluation logic
    print(f"⚠️  Evaluation logic not yet implemented for {model_path}")
    
    return metrics


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_accuracy_comparison(results: List[Dict], save_path: Path):
    """Bar chart comparing Top-1/3/5 accuracy across experiments."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exp_names = [r["name"] for r in results]
    top1 = [r["metrics"].get("top1_acc", 0) for r in results]
    top3 = [r["metrics"].get("top3_acc", 0) for r in results]
    top5 = [r["metrics"].get("top5_acc", 0) for r in results]
    
    x = np.arange(len(exp_names))
    width = 0.25
    
    ax.bar(x - width, top1, width, label="Top-1", color=NATURE_ACCENT_PALETTE[0])
    ax.bar(x, top3, width, label="Top-3", color=NATURE_ACCENT_PALETTE[1])
    ax.bar(x + width, top5, width, label="Top-5", color=NATURE_ACCENT_PALETTE[2])
    
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison Across Experiments")
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=45, ha="right")
    ax.legend()
    
    enable_light_ygrid()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved accuracy comparison to {save_path}")
    plt.close()


def plot_loss_curves(results: List[Dict], save_path: Path):
    """Line plot of training loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, res in enumerate(results):
        if "loss_history" in res["metrics"]:
            losses = res["metrics"]["loss_history"]
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, label=res["name"], 
                   color=NATURE_ACCENT_PALETTE[i % len(NATURE_ACCENT_PALETTE)],
                   linewidth=2)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    
    enable_light_ygrid()
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved loss curves to {save_path}")
    plt.close()


def plot_attention_heatmap(attention_weights: np.ndarray, token_labels: List[str], 
                          save_path: Path):
    """Heatmap of attention weights across token types."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(attention_weights, annot=True, fmt=".3f", 
                xticklabels=token_labels, cmap="Blues",
                cbar_kws={"label": "Attention Weight"}, ax=ax)
    
    ax.set_title("Attention Weights: Multi-Modal Tokens")
    ax.set_xlabel("Token Type")
    ax.set_ylabel("Example Index")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✓ Saved attention heatmap to {save_path}")
    plt.close()


def create_summary_table(results: List[Dict], save_path: Path):
    """Generate LaTeX/Markdown summary table."""
    table_lines = [
        "| Experiment | Top-1 Acc | Top-3 Acc | Top-5 Acc | Final Loss |",
        "|------------|-----------|-----------|-----------|------------|",
    ]
    
    for res in results:
        metrics = res["metrics"]
        row = (
            f"| {res['name']:20s} | "
            f"{metrics.get('top1_acc', 0):.3f} | "
            f"{metrics.get('top3_acc', 0):.3f} | "
            f"{metrics.get('top5_acc', 0):.3f} | "
            f"{metrics.get('final_loss', 0):.4f} |"
        )
        table_lines.append(row)
    
    table_md = "\n".join(table_lines)
    save_path.write_text(table_md)
    print(f"✓ Saved summary table to {save_path}")
    print("\n" + table_md + "\n")


# ============================================================================
# Statistical Tests
# ============================================================================

def paired_t_test(baseline_scores: np.ndarray, treatment_scores: np.ndarray) -> Tuple[float, float]:
    """Paired t-test for statistical significance."""
    from scipy import stats
    
    t_stat, p_value = stats.ttest_rel(baseline_scores, treatment_scores)
    return t_stat, p_value


def compute_effect_size(baseline_scores: np.ndarray, treatment_scores: np.ndarray) -> float:
    """Cohen's d effect size."""
    diff = treatment_scores - baseline_scores
    return diff.mean() / diff.std()


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate comparison experiments")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/comparison"),
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/analysis"),
        help="Directory to save analysis outputs",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all experiment results
    print("Loading experiment results...")
    experiments = []
    for exp_dir in sorted(args.results_dir.glob("*_rm")):
        if exp_dir.is_dir():
            print(f"  - {exp_dir.name}")
            experiments.append(load_experiment_results(exp_dir))
    
    print(f"\n✓ Loaded {len(experiments)} experiments\n")
    
    # Generate visualizations
    print("Generating visualizations...\n")
    
    plot_accuracy_comparison(experiments, args.output_dir / "accuracy_comparison.pdf")
    plot_loss_curves(experiments, args.output_dir / "loss_curves.pdf")
    create_summary_table(experiments, args.output_dir / "summary_table.md")
    
    # TODO: Extract and plot attention weights
    # This requires inference on test data with attention output
    
    print("\n" + "="*70)
    print("✅ Evaluation Complete!")
    print("="*70)
    print(f"\nAnalysis outputs saved to: {args.output_dir}/")
    print("\nKey Findings to Report:")
    print("  1. Does multi-modal improve over baseline? (accuracy delta)")
    print("  2. Best token ordering strategy? (physics_first vs structure_first)")
    print("  3. ICL benefit? (3-shot nearest vs no ICL)")
    print("  4. Token budget trade-off? (50 vs 200 tokens)")
    print("\nNext: Document results in experiment report!\n")


if __name__ == "__main__":
    main()
