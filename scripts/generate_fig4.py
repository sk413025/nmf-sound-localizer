import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Add current directory to path to import NA_matplotlib_guild
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import NA_matplotlib_guild as na_style

# Load data
data_path = "figure4_data.json"
if not os.path.exists(data_path):
    print("Run gather_figure4_data.py first!")
    sys.exit(1)

with open(data_path, 'r') as f:
    raw_data = json.load(f)

# Prep Data
# A: Ablation
ablation_order = ["Baseline", "No Transformer", "Fixed Heuristic", "No Type Bias", "Dense Routing", "G-Teacher", "G-Fixed"]
ablation_labels_short = ["Baseline", "No Transf.", "Fixed Heur.", "No Type Bias", "Dense Rout.", "G-Teacher", "G-Fixed"]
ablation_vals = [raw_data["ablation"].get(k, []) for k in ablation_order]

# B: SNR
snr_order = ["0dB", "5dB", "10dB", "15dB", "20dB", "30dB", "Clean"]
snr_vals = [raw_data["snr"].get(k, []) for k in snr_order]
# Map 'Clean' to 'Inf' for display if preferred, or keep 'Clean'
snr_display = ["0", "5", "10", "15", "20", "30", "Inf"]

# Setup Figure
na_style.set_nature_rcparams(base_fontsize=7)
# Double column: 183 mm wide. Height: let's pick something reasonable, e.g. 60-70mm
fig = na_style.make_figure(width_mm=183, height_mm=70)

# Subplots
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1]) # Ablation needs more width for labels
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# --- Panel A: Ablation ---
x_pos = np.arange(len(ablation_order))
np.random.seed(42) # Consistent jitter

# Plot points
for i, vals in enumerate(ablation_vals):
    # Jitter
    y = np.array(vals)
    x = np.random.normal(i, 0.04, size=len(y))
    ax1.scatter(x, y, s=10, alpha=0.7, color='#377eb8', edgecolors='none', zorder=3)
    
    # Mean bar
    if len(y) > 0:
        mean_val = np.mean(y)
        ax1.hlines(mean_val, i-0.2, i+0.2, colors='k', linewidth=1.2, zorder=4)

ax1.set_xticks(x_pos)
ax1.set_xticklabels(ablation_labels_short, rotation=45, ha='right')
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.05) # Assume accuracy is 0-1
ax1.grid(axis='y', linestyle='--', alpha=0.3)
na_style.add_panel_label(ax1, "a")

# --- Panel B: SNR ---
x_pos_snr = np.arange(len(snr_order))

# Calculate means and stds for line plot
means = [np.mean(v) if len(v)>0 else np.nan for v in snr_vals]
stds = [np.std(v) if len(v)>0 else 0 for v in snr_vals]

# Plot Line
ax2.plot(x_pos_snr, means, '-', color='gray', alpha=0.5, linewidth=1, zorder=1)

# Plot Points
for i, vals in enumerate(snr_vals):
    y = np.array(vals)
    x = np.random.normal(i, 0.04, size=len(y))
    ax2.scatter(x, y, s=10, alpha=0.7, color='#e41a1c', edgecolors='none', zorder=3)
    
    # Error bars or mean marker?
    # Let's add mean marker
    if len(y) > 0:
        ax2.errorbar(i, np.mean(y), yerr=np.std(y), fmt='none', color='k', capsize=3, zorder=2)

ax2.set_xticks(x_pos_snr)
ax2.set_xticklabels(snr_display)
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
na_style.add_panel_label(ax2, "b")

# Adjust layout to handle rotated labels
plt.tight_layout()

# Save
output_pdf = "Figure4.pdf"
output_png = "Figure4.png"
fig.savefig(output_pdf, dpi=300)
fig.savefig(output_png, dpi=300)
print(f"Saved {output_pdf} and {output_png}")
