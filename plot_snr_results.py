import numpy as np
import matplotlib.pyplot as plt
import os
import glob

results_dir = "results"
folds = [0, 1, 2]
snrs = ["Inf", "30dB", "20dB", "15dB", "10dB", "5dB", "0dB"]

data = {snr: [] for snr in snrs}

print("Gathering data...")
for snr in snrs:
    for fold in folds:
        folder_pattern = f"qk_snr{snr}_30epochs_fold{fold}_*"
        folders = glob.glob(os.path.join(results_dir, folder_pattern))
        
        if not folders:
            continue
            
        folder = sorted(folders)[-1]
        metrics_path = os.path.join(folder, "metrics.npz")
        
        if os.path.exists(metrics_path):
            try:
                d = np.load(metrics_path, allow_pickle=True)
                acc = float(d['best_accuracy'])
                data[snr].append(acc * 100)
            except Exception as e:
                print(f"Error reading {metrics_path}: {e}")

# Prepare data for boxplot
plot_data = [data[snr] for snr in snrs]

plt.figure(figsize=(10, 6))
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Create boxplot
bp = plt.boxplot(plot_data, labels=snrs, patch_artist=True, showmeans=True)

# Style
for box in bp['boxes']:
    box.set(facecolor='lightblue', linewidth=1)
for median in bp['medians']:
    median.set(color='red', linewidth=2)

# Add scatter points
for i, snr in enumerate(snrs):
    y = data[snr]
    if not y:
        continue
    x = np.random.normal(i + 1, 0.04, size=len(y))
    plt.plot(x, y, 'k.', alpha=0.6)
    
    # Add mean text
    mean_val = np.mean(y)
    # Position text above if high, below if low, or just above box
    text_y = min(102, max(y) + 2) if y else 0
    plt.text(i + 1, text_y, f"{mean_val:.1f}%", 
             ha='center', va='bottom', fontweight='bold', color='blue', fontsize=9)

plt.title("SNR Sweep Results (3-Fold CV) - QK Mode 30 Epochs")
plt.ylabel("Accuracy (%)")
plt.xlabel("SNR Level")
plt.ylim(-5, 110)

plt.tight_layout()
output_path = "snr_5fold_results_boxplot.png"
plt.savefig(output_path, dpi=300)
print(f"Plot saved to {output_path}")
