import numpy as np
import os
import glob
import json

metrics_file = "metrics.npz"

def get_accuracy(path):
    full_path = os.path.join(path, metrics_file)
    if not os.path.exists(full_path):
        print(f"Warning: {full_path} not found")
        return None
    try:
        data = np.load(full_path, allow_pickle=True)
        # best_accuracy might be a 0-d array
        return float(data['best_accuracy'])
    except Exception as e:
        print(f"Error reading {full_path}: {e}")
        return None

data = {
    "ablation": {},
    "snr": {}
}

# --- Ablation Data ---
ablation_base = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results"

# Manually mapped/globbed based on IMPORTANT_ARTIFACTS.md
ablation_map = {
    "Baseline": [
        "omp_transformer_speech260_trainval_split_full_20251202_192153", # Seed 42
        "omp_transformer_speech260_trainval_split_full_seed1_*",
        "omp_transformer_speech260_trainval_split_full_seed2_*",
        "omp_transformer_speech260_trainval_split_full_seed3_*",
        "omp_transformer_speech260_trainval_split_full_seed4_*",
        "omp_transformer_speech260_trainval_split_full_seed5_*" 
    ],
    "No Transformer": ["ablate_identity_speech260_seed*"],
    "Fixed Heuristic": ["ablate_g_routing_speech260_seed*"],
    "No Type Bias": ["ablate_no_type_bias_speech260_seed*"],
    "Dense Routing": ["ablate_disable_omp_sparsity_speech260_seed*"],
    "G-Teacher": ["ablate_g_teacher_F_seed*"],
    "G-Fixed": ["ablate_g_fixed_F_seed*"]
}

print("Gathering Ablation Data...")
for label, patterns in ablation_map.items():
    accuracies = []
    for pattern in patterns:
        full_pattern = os.path.join(ablation_base, pattern)
        matches = glob.glob(full_pattern)
        for m in matches:
            acc = get_accuracy(m)
            if acc is not None:
                accuracies.append(acc)
    data["ablation"][label] = accuracies
    print(f"  {label}: {len(accuracies)} runs")

# --- SNR Data ---
snr_base = "/Users/jnrle/Documents/LDVReorientation/worktrees/exp-snr-5fold-repro/results"
snr_levels = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Inf"]

print("Gathering SNR Data...")
for level in snr_levels:
    pattern = f"qk_snr{level}_30epochs_fold*"
    full_pattern = os.path.join(snr_base, pattern)
    matches = glob.glob(full_pattern)
    accuracies = []
    for m in matches:
        acc = get_accuracy(m)
        if acc is not None:
            accuracies.append(acc)
    
    # Store with 'Inf' mapped to 'Clean' or kept as Inf
    key = "Clean" if level == "Inf" else level
    data["snr"][key] = accuracies
    print(f"  {key}: {len(accuracies)} runs")

# Save
output_file = "/Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/figure4_data.json"
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"Data saved to {output_file}")
