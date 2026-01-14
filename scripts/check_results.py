import numpy as np
import os

base_dir = "/Users/sbplab/LDV-data-experiments/snr-synthetic-2025-12/results/white_noise_transformer"
snr_levels = ["Inf", "30dB", "20dB", "15dB", "10dB", "5dB", "0dB"]

print("| SNR | Accuracy |")
print("|---|---|")

for snr in snr_levels:
    path = os.path.join(base_dir, f"snr_{snr}", "metrics.npz")
    if os.path.exists(path):
        data = np.load(path)
        # Assuming 'val_acc' is stored. Let's check keys if this fails.
        # Based on typical scripts, it might be 'val_acc_history' or similar.
        # Let's try to list keys first if I can't guess.
        # But usually 'val_acc' is the key for best accuracy or final accuracy.
        # Let's just print all keys for the first one to debug.
        if snr == "Inf":
            print(f"Keys for {snr}: {list(data.keys())}")
        
        if 'best_accuracy' in data:
            print(f"| {snr} | {data['best_accuracy']:.4f} |")
        elif 'val_acc_history' in data:
            acc = np.max(data['val_acc_history'])
            print(f"| {snr} | {acc:.4f} |")
        elif 'val_acc' in data:
             print(f"| {snr} | {data['val_acc']:.4f} |")
        else:
            print(f"| {snr} | N/A |")
    else:
        print(f"| {snr} | Missing |")
