import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import NA_matplotlib_guild as na_style

# Set Nature style
na_style.set_nature_rcparams()

# Absolute paths to data
BASELINE_PATH = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz"
NO_TRANS_PATH = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz"

def load_metrics(path):
    if not Path(path).exists():
        print(f"Error: File not found at {path}")
        return None
    return np.load(path, allow_pickle=True)

def plot_heatmaps(baseline_cm, no_trans_cm, angles, out_path):
    # Figure size: Full width (180mm) x Height (e.g. 60mm)
    fig = na_style.make_figure(width_mm=180, height_mm=70)
    
    # 2 subplots
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25, left=0.08, right=0.95, bottom=0.15, top=0.85)
    
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    cms = [baseline_cm, no_trans_cm]
    titles = ['Baseline (Transformer)', 'No Transformer']
    
    # Determine common vmin/vmax for colorbar consistency?
    # Confusion matrices are counts. They might have different total counts if validation set size differs?
    # Usually validation set is fixed.
    # Let's use independent scaling or common max?
    # Independent is safer for visibility if one is much worse.
    
    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation='nearest', cmap='viridis', origin='upper')
        ax.set_title(title, fontsize=8, fontweight='bold')
        
        # Ticks
        tick_indices = np.arange(0, len(angles), 5)
        tick_labels = [f"{angles[i]:.0f}" for i in tick_indices]
        
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=6)
        ax.set_yticks(tick_indices)
        ax.set_yticklabels(tick_labels, fontsize=6)
        
        ax.set_xlabel('Predicted Atom Index', fontsize=7)
        ax.set_ylabel('True DOA (Angle)', fontsize=7)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=6)

    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_distribution_comparison(baseline_row, no_trans_row, angles, target_angle, out_path):
    # Figure size: Full width (180mm) x Height (e.g. 60mm)
    fig = na_style.make_figure(width_mm=180, height_mm=60)
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2, left=0.08, right=0.95, bottom=0.2, top=0.85)
    
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    rows = [baseline_row, no_trans_row]
    titles = [f'Baseline @ {target_angle:.0f}°', f'No Transformer @ {target_angle:.0f}°']
    
    # Find target index
    target_idx = np.where(np.isclose(angles, target_angle))[0][0]
    
    for ax, row, title in zip(axes, rows, titles):
        # Normalize
        total = row.sum()
        probs = row / total if total > 0 else row
        
        atom_indices = np.arange(len(angles))
        
        # Colors
        colors = np.array(['#7f7f7f'] * len(angles)) # Gray
        colors[target_idx] = '#d62728' # Red
        
        # Stem plot
        ax.vlines(x=atom_indices, ymin=0, ymax=probs, colors=colors, linewidth=1.0, alpha=0.8)
        ax.scatter(atom_indices, probs, color=colors, s=10, zorder=3, edgecolor='none')
        ax.axhline(0, color='black', linewidth=0.5)
        
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.set_xlabel('Predicted Atom Index', fontsize=7)
        ax.set_ylabel('Probability', fontsize=7)
        ax.set_ylim(0, 1.05)
        
        # Ticks
        tick_indices = np.arange(0, len(angles), 5)
        tick_labels = [f"{angles[i]:.0f}°" for i in tick_indices]
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(tick_labels, rotation=45, fontsize=6)
        ax.tick_params(axis='y', labelsize=6)
        
        # Grid
        ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Annotate correct prob
        correct_prob = probs[target_idx]
        ax.text(target_idx, correct_prob + 0.05, f"{correct_prob:.1%}", 
                ha='center', va='bottom', color='#d62728', fontsize=6, fontweight='bold')

    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='results_figure8')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("Loading data...")
    baseline_data = load_metrics(BASELINE_PATH)
    no_trans_data = load_metrics(NO_TRANS_PATH)
    
    if baseline_data is None or no_trans_data is None:
        return
        
    baseline_cm = baseline_data['confusion_matrix']
    no_trans_cm = no_trans_data['confusion_matrix']
    angles = baseline_data['angles']
    
    # 1. Heatmaps
    plot_heatmaps(baseline_cm, no_trans_cm, angles, out_dir / "Fig8a_heatmaps.pdf")
    
    # 2. Angle 55 Distribution
    target_angle = 55.0
    target_idx = np.where(np.isclose(angles, target_angle))[0]
    if len(target_idx) > 0:
        idx = target_idx[0]
        plot_distribution_comparison(baseline_cm[idx], no_trans_cm[idx], angles, target_angle, out_dir / "Fig8b_angle55.pdf")
        
    # 3. Angle 100 Distribution
    target_angle = 100.0
    target_idx = np.where(np.isclose(angles, target_angle))[0]
    if len(target_idx) > 0:
        idx = target_idx[0]
        plot_distribution_comparison(baseline_cm[idx], no_trans_cm[idx], angles, target_angle, out_dir / "Fig8c_angle100.pdf")

if __name__ == "__main__":
    main()
