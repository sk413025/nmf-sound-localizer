import numpy as np
import os

# Ablation path
ablation_path = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_20251202_192153/metrics.npz"

# SNR path
snr_path = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-snr-5fold-repro/results/qk_snr0dB_30epochs_fold0_20251214/metrics.npz"

def inspect_npz(path):
    print(f"--- Inspecting {path} ---")
    try:
        data = np.load(path)
        print("Keys:", data.files)
        for k in data.files:
            val = data[k]
            if val.size < 10:
                print(f"{k}: {val}")
            else:
                print(f"{k}: shape {val.shape}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

if os.path.exists(ablation_path):
    inspect_npz(ablation_path)
else:
    print(f"File not found: {ablation_path}")

if os.path.exists(snr_path):
    inspect_npz(snr_path)
else:
    print(f"File not found: {snr_path}")
