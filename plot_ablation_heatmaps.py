import numpy as np
import matplotlib.pyplot as plt
import os

def load_metrics(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return None
    return np.load(path, allow_pickle=True)

def plot_heatmap(ax, cm, title, angles):
    # Normalize confusion matrix to be between 0 and 1 (optional, but good for heatmap)
    # Or just plot counts. Let's plot counts but use a colorbar.
    # Actually, row-normalized is often better to see recall per class.
    # But raw counts show the distribution of data too.
    # Let's stick to raw counts for now, or maybe log scale if there's huge imbalance.
    # Given the "val split" is likely balanced, raw counts are fine.
    
    im = ax.imshow(cm, interpolation='nearest', cmap='viridis', origin='upper')
    ax.set_title(title)
    
    # We have 37 angles. Showing all ticks might be crowded.
    # Let's show every 5th angle or so.
    tick_indices = np.arange(0, len(angles), 5)
    tick_labels = [f"{angles[i]:.0f}" for i in tick_indices]
    
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel('Select atom index (Predicted Angle)')
    ax.set_ylabel('True DOA (Angle)')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    baseline_path = os.path.join(base_dir, 'results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz')
    no_trans_path = os.path.join(base_dir, 'results/ablate_identity_speech260_seed42_20251210_134919/metrics.npz')
    
    print(f"Loading Baseline from: {baseline_path}")
    baseline_data = load_metrics(baseline_path)
    
    print(f"Loading No Transformer from: {no_trans_path}")
    no_trans_data = load_metrics(no_trans_path)
    
    if baseline_data is None or no_trans_data is None:
        return

    baseline_cm = baseline_data['confusion_matrix']
    no_trans_cm = no_trans_data['confusion_matrix']
    angles = baseline_data['angles'] # Assuming angles are same for both
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    plot_heatmap(axes[0], baseline_cm, 'Baseline (Trans-Routed Soft-OMP)', angles)
    plot_heatmap(axes[1], no_trans_cm, 'No Transformer (Identity Encoder)', angles)
    
    plt.tight_layout()
    output_dir = os.path.join(base_dir, 'results/ablation_analysis_20251219')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'ablation_heatmaps.png')
    plt.savefig(output_path)
    print(f"Saved heatmaps to {output_path}")

    output_path_pdf = os.path.join(output_dir, 'ablation_heatmaps.pdf')
    plt.savefig(output_path_pdf)
    print(f"Saved heatmaps to {output_path_pdf}")

if __name__ == "__main__":
    main()
