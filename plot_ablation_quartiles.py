import matplotlib.pyplot as plt
import numpy as np

# Data from README_run_notes.md
data = {
    "No Transformer": [0.631, 0.623, 0.727, 0.640, 0.678],
    "Fixed Heuristic": [0.017, 0.014, 0.019, 0.027, 0.030],
    "No Type Bias": [0.912, 0.940, 0.920, 0.929, 0.922],
    "Dense Routing": [0.027, 0.027, 0.027, 0.027, 0.027],
    "Pure G (g-teacher F)": [0.017, 0.014, 0.019, 0.027, 0.030]
}

# Calculate mean for sorting
mean_scores = {k: np.mean(v) for k, v in data.items()}
# Sort keys by mean score descending
sorted_keys = sorted(data.keys(), key=lambda k: mean_scores[k], reverse=True)

# Prepare sorted data
labels = sorted_keys
values = [data[k] for k in sorted_keys]

# Create boxplot
plt.figure(figsize=(12, 8))
# Use tick_labels for newer matplotlib, but fallback to labels if needed or just pass as second arg
# To be safe across versions, we can use plt.xticks
bp = plt.boxplot(values, patch_artist=True, showmeans=True)
plt.xticks(range(1, len(labels) + 1), labels, rotation=15)

plt.title("Ablation Study: Validation Accuracy Distribution (5 seeds)\nSorted by Performance", fontsize=16)
plt.ylabel("Validation Accuracy", fontsize=14)
plt.xlabel("Method", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, 1.05)

# Add individual data points
for i, method_values in enumerate(values):
    x = np.random.normal(i + 1, 0.04, size=len(method_values))
    plt.plot(x, method_values, 'r.', alpha=0.5)

# Save as PDF
output_path = "results/ablation_accuracy_quartiles_updated.pdf"
plt.savefig(output_path, format='pdf', bbox_inches='tight')
print(f"Plot saved to {output_path}")
