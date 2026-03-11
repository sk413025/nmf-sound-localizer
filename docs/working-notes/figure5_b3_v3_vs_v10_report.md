# Figure 5 B3: v3 vs v10 Full-Band and Banded Comparison Report

## Scope
This report compares the **Figure 5 B3** outputs from **v3** (original polar plot using model-native scores) and **v10** (banded line plots using frequency-band projection). It documents the visual differences, numerical discrepancies, and root causes.

## Data Source
- Run directory: `/Users/jnrle/Documents/LDVReorientation/worktrees/nature-comm-repro/results/omp_transformer_speech260_trainval_split_full_20251115_082341`
- Inputs:
  - `modal_routing_val.npz` (contains `scores_expert`, `g_energy_expert`, `scores_atoms`, `Y_val`)
  - `dictionary.npz` (contains `D`, `angles`)
- Representative sample selection logic (same as the figure scripts):
  - Prefer a case where **true angle=145°**, **OMP predicts 60°**, **QK predicts 145°**.
  - If missing, fall back to the first sample where QK corrects OMP.
  - Result: `sample_idx=1509`, `true_angle=145°`.

## Method Summary

### v3 (Original B3)
- **Scores:**
  - OMP: `routing_data['g_energy_expert']` (full-band, model-native)
  - QK: `routing_data['scores_expert']` (full-band, model-native)
- **Plot:** Polar (radar-style), min-max normalized per curve.
- **Ground Truth:** star at radius 1.05.

### v10 (Banded / Full Band via Projection)
- **Scores (full band 300–3000 Hz):**
  - OMP: `abs(D.T @ Y)` → reshape `(37,8)` → sum over atoms.
  - QK: `abs(sum(atom_proj * scores_atoms))` per expert.
- **Plot:** Square line plot (angle on x-axis), **raw values (no normalization)**.
- **Ground Truth:** star at `1.05 * max(curve)`.

This v10 method follows the **frequency-decomposed** logic described in `docs/working-notes/polar_plot_architecture_analysis.md` (section 6.2), but uses fixed frequency **bands** instead of center-frequency windows.

## Figures

### v3: Full-Band Polar Plot (Model-Native Scores)
![v3 B3 Polar](results/figure5_v3_v10_report_assets/Fig5_B3_POLAR_ESTIMATION.png)

### v10: Full-Band Line Plot (Projection-Based Scores)
![v10 B3 Full Band](results/figure5_v3_v10_report_assets/Fig5_B3_BAND_300_3000.png)

### v10: Banded Line Plots (Projection-Based Scores)
![v10 B3 Band 0-500](results/figure5_v3_v10_report_assets/Fig5_B3_BAND_0_500.png)
![v10 B3 Band 500-1000](results/figure5_v3_v10_report_assets/Fig5_B3_BAND_500_1000.png)
![v10 B3 Band 1000-2000](results/figure5_v3_v10_report_assets/Fig5_B3_BAND_1000_2000.png)

## Numerical Comparison (Full Band)

### Representative Sample (idx=1509, true angle=145°)
**OMP (v3 vs v10):**
- Correlation: **0.99999999999**
- Relative L2 error: **2.13e-07**
- MAE: **3.25e-07**
- Top-1 angle: **both 60°**

**QK (v3 vs v10):**
- Correlation: **-0.2067**
- Relative L2 error: **1.0356**
- MAE: **0.5292**
- Top-1 angle: **v3=145°**, **v10=90°**

**Scale difference (QK):**
- v3 QK min/max: **-0.841 / 5.600**
- v10 QK min/max: **0.00123 / 0.383**

### Dataset-Wide Agreement
- **OMP correlation mean:** 0.99999999999
- **OMP top-1 match rate:** 1.000
- **QK correlation mean:** 0.192
- **QK top-1 match rate:** 0.319

## Interpretation

### Why OMP Matches
- Both v3 and v10 compute OMP using `|D.T @ Y|` aggregated by expert.
- The nearly perfect correlation confirms that the full-band OMP calculation is consistent.

### Why QK Differs
- **v3 QK** is the **model-native attention score** (`scores_expert`) derived from Transformer internal representations.
- **v10 QK** is a **projection-weighted score**: `abs(sum(atom_proj * scores_atoms))`.
- These are **different physical quantities**: one is attention magnitude, the other is an energy-weighted response.
- v3 allows negative scores (pre-normalization), while v10 is non-negative by construction.
- This explains the low correlation and mismatched top-1 predictions.

## Key Takeaways
- **OMP is consistent** between v3 and v10 full-band methods.
- **QK is not consistent** because the definitions differ, not due to a calculation error.
- To match v3 behavior in a line plot, the v10 plot must use `scores_expert` directly.
- The v10 banded plots are valid for **frequency analysis**, but should not be interpreted as the model's native scoring.

## Reproduction Notes

### Generate v3 B3
```bash
python /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/generate_figure5_atomic_v3.py
```

### Generate v10 B3 (banded + full band)
```bash
python /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/generate_figure5_atomic_v10.py
```

### Convert PDFs to PNGs (for this report)
```bash
mkdir -p /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets
sips -s format png \
  /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/results_figure5_v3/Fig5_B3_POLAR_ESTIMATION.pdf \
  --out /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets/Fig5_B3_POLAR_ESTIMATION.png
sips -s format png \
  /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/results_figure5_v10/Fig5_B3_BAND_0_500.pdf \
  --out /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets/Fig5_B3_BAND_0_500.png
sips -s format png \
  /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/results_figure5_v10/Fig5_B3_BAND_500_1000.pdf \
  --out /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets/Fig5_B3_BAND_500_1000.png
sips -s format png \
  /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/results_figure5_v10/Fig5_B3_BAND_1000_2000.pdf \
  --out /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets/Fig5_B3_BAND_1000_2000.png
sips -s format png \
  /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/results_figure5_v10/Fig5_B3_BAND_300_3000.pdf \
  --out /Users/jnrle/Documents/LDVReorientation/worktrees/nature-figures/results/figure5_v3_v10_report_assets/Fig5_B3_BAND_300_3000.png
```

### Numeric Comparison Script (full band)
```bash
python3 - <<'PY'
import numpy as np
from pathlib import Path

run_dir = Path('/Users/jnrle/Documents/LDVReorientation/worktrees/nature-comm-repro/results/omp_transformer_speech260_trainval_split_full_20251115_082341')
routing = np.load(run_dir / 'modal_routing_val.npz')
dict_data = np.load(run_dir / 'dictionary.npz', allow_pickle=True)

scores_expert = routing['scores_expert']
g_energy_expert = routing['g_energy_expert']
labels = routing['labels']
angles = dict_data['angles']
D = dict_data['D']
Y_all = routing['Y_val']
scores_atoms_all = routing['scores_atoms']

qk_pred = np.argmax(scores_expert, axis=1)
g_pred = np.argmax(g_energy_expert, axis=1)
angle_145_idx = np.argmin(np.abs(angles - 145))
angle_60_idx = np.argmin(np.abs(angles - 60))
rep_idx = None
for idx in range(len(labels)):
    if labels[idx] == angle_145_idx and g_pred[idx] == angle_60_idx and qk_pred[idx] == angle_145_idx:
        rep_idx = idx
        break
if rep_idx is None:
    qk_correct = (qk_pred == labels)
    g_wrong = (g_pred != labels)
    qk_fix = qk_correct & g_wrong
    rep_idx = int(np.where(qk_fix)[0][0]) if qk_fix.sum() > 0 else 0

atom_proj = Y_all @ D
v10_omp_all = np.abs(atom_proj).reshape(len(Y_all), 37, 8).sum(axis=2)
v10_qk_all = np.abs((atom_proj.reshape(len(Y_all), 37, 8) * scores_atoms_all).sum(axis=2))

print('rep_idx:', rep_idx)
print('true_angle:', angles[labels[rep_idx]])
print('OMP corr (rep):', np.corrcoef(g_energy_expert[rep_idx], v10_omp_all[rep_idx])[0,1])
print('QK corr (rep):', np.corrcoef(scores_expert[rep_idx], v10_qk_all[rep_idx])[0,1])
print('OMP top1 match:', np.mean(np.argmax(g_energy_expert, axis=1) == np.argmax(v10_omp_all, axis=1)))
print('QK top1 match:', np.mean(np.argmax(scores_expert, axis=1) == np.argmax(v10_qk_all, axis=1)))
PY
```

## Recommendations
- If the goal is **visual consistency with v3**, plot v10 full-band using **`scores_expert`** and **`g_energy_expert`** directly.
- If the goal is **frequency interpretation**, keep the projection-based method but explicitly label it as a **derived proxy** rather than a model-native score.
