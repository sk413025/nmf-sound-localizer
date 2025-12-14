import numpy as np
import os
import glob

results_dir = "results"
folds = [0, 1, 2]
snrs = ["Inf", "30dB", "20dB", "15dB", "10dB", "5dB", "0dB"]
date_str = "20251214"

print(f"--- Results for SNR Sweep (3-fold CV) ---")

for snr in snrs:
    print(f"\n=== SNR {snr} ===")
    accuracies = []
    best_epochs = []

    for fold in folds:
        folder_pattern = f"qk_snr{snr}_30epochs_fold{fold}_*"
        folders = glob.glob(os.path.join(results_dir, folder_pattern))
        
        if not folders:
            print(f"  Fold {fold}: No results found.")
            continue
            
        # Take the most recent one
        folder = sorted(folders)[-1]
        metrics_path = os.path.join(folder, "metrics.npz")
        
        if not os.path.exists(metrics_path):
            print(f"  Fold {fold}: metrics.npz not found in {folder}")
            continue
            
        try:
            data = np.load(metrics_path, allow_pickle=True)
            acc = float(data['best_accuracy'])
            epoch = int(data['best_epoch'])
            accuracies.append(acc)
            best_epochs.append(epoch)
            print(f"  Fold {fold}: Accuracy = {acc*100:.2f}%, Best Epoch = {epoch}")
        except Exception as e:
            print(f"  Fold {fold}: Error loading metrics - {e}")

    if accuracies:
        avg_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        print(f"  >> Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
        print(f"  >> Average Best Epoch: {np.mean(best_epochs):.1f}")
    else:
        print("  >> No valid results found.")
