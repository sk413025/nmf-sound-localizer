import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

def plot_training_history(out_dir):
    # Data extracted from experiment logs
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    train_loss = [2.2219, 2.1602, 2.1380, 2.1178, 2.0983, 2.0750, 2.0482, 2.0308, 2.0110, 1.9933, 
                  1.9716, 1.9409, 1.9078, 1.8801, 1.8574, 1.8367, 1.8133, 1.7928, 1.7765, 1.7582, 
                  1.7505, 1.7354, 1.7295, 1.7267, 1.7307, 1.7281, 1.7397, 1.7337, 1.7213, 1.7234]
    val_loss = [2.1780, 2.1568, 2.1434, 2.1201, 2.1013, 2.0689, 2.0552, 2.0371, 2.0231, 2.0652,
                1.9911, 1.9549, 1.9676, 1.9289, 1.9622, 1.8563, 1.8557, 1.9932, 1.8632, 1.9533,
                1.7891, 1.7886, 1.7983, 1.8784, 1.7846, 1.8174, 1.8589, 1.8136, 2.0243, 1.8879]
    acc = [0.2614, 0.2731, 0.2793, 0.2876, 0.2895, 0.3019, 0.3082, 0.3138, 0.3199, 0.3028,
           0.3269, 0.3373, 0.3296, 0.3440, 0.3326, 0.3655, 0.3676, 0.3315, 0.3621, 0.3473,
           0.3884, 0.3886, 0.3829, 0.3691, 0.3942, 0.3832, 0.3715, 0.3838, 0.3403, 0.3624]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (CrossEntropy)', color=color)
    ax1.plot(epochs, train_loss, label='Train Loss', color=color, linestyle='-')
    ax1.plot(epochs, val_loss, label='Val Loss', color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  
    ax2.plot(epochs, acc, label='Val Accuracy', color=color, linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Freq-Aware DTmin Training Trajectory (Full Spectrum)')
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig(out_dir / 'training_curve.png')
    print(f"Saved training_curve.png to {out_dir}")

def plot_reduction_comparison(out_dir):
    # Data from evaluation
    categories = ['Low Freq (5-300)', 'Full Spectrum (5-1024)']
    omp_scores = [97.88, 78.99]
    dt_scores = [np.nan, 65.46] # Low freq DT score estimated or separate run? Use what we have.
    # Note: We only have Full Spectrum DT score from recent run.
    # Let's just plot Full Spectrum comparison OMP vs DT
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['OMP (Oracle)', 'DTmin (Model)']
    values = [78.99, 65.46]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(labels, values, color=colors, width=0.5)
    
    # Add values
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
    ax.set_ylabel('Energy Reduction (%)')
    ax.set_title('Performance Comparison on Full Spectrum (5-1024 Hz)')
    ax.set_ylim(0, 100)
    
    # Add info about recovery
    ax.text(0.5, 90, f"Recovery Ratio: {65.46/78.99:.1%}", 
            ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_dir / 'reduction_comparison.png')
    print(f"Saved reduction_comparison.png to {out_dir}")

if __name__ == "__main__":
    out_dir = Path("results/dt_freq_aware_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(out_dir)
    plot_reduction_comparison(out_dir)
