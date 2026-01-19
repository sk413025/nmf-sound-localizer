import json
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

results_dir = Path("/Users/jnrle/Documents/LDVReorientation/worktrees/exp-interspeech-GRU2/results/exp_interspeech_gru2_tw32_lag50_k16_ep50")
report_dir = results_dir / "report"
fig_dir = report_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

pipeline_log = results_dir / "pipeline.log"
text = pipeline_log.read_text()

pattern = re.compile(r"Epoch (\d+): Train Loss=([0-9.]+), Val Loss=([0-9.]+), Acc=([0-9.]+)")
epochs = []
train_loss = []
val_loss = []
acc = []
for match in pattern.finditer(text):
    epochs.append(int(match.group(1)))
    train_loss.append(float(match.group(2)))
    val_loss.append(float(match.group(3)))
    acc.append(float(match.group(4)))

blocks_match = re.search(r"Saved (\d+) blocks to .*\(Variants: (\d+)\)", text)
blocks = int(blocks_match.group(1)) if blocks_match else None
variants = int(blocks_match.group(2)) if blocks_match else None
pairs_match = re.search(r"Found (\d+) WAV pairs", text)
pairs = int(pairs_match.group(1)) if pairs_match else None

k_sweep_path = results_dir / "omp_vs_random_k_sweep" / "omp_vs_random_k_sweep.json"
k_sweep = json.loads(k_sweep_path.read_text())
ks = [d["k"] for d in k_sweep["per_k"]]
omp_k = [d["omp"] for d in k_sweep["per_k"]]
rand_k = [d["random"] for d in k_sweep["per_k"]]

stats = torch.load(results_dir / "eval" / "eval_stats.pt", map_location="cpu")
DT = stats["DT"]
OMP = stats["OMP"]

DT_mean = DT.mean(dim=0).numpy()
OMP_mean = OMP.mean(dim=0).numpy()
K_full = np.arange(1, DT_mean.shape[0] + 1)
dt_sparse = [DT_mean[k - 1] for k in ks]

fs = 16000
n_fft = 2048
freq_bins = np.linspace(0, fs / 2, n_fft // 2 + 1)
BANDS = {
    "Sub-Bass (0-60Hz)": (0, 60),
    "Bass (60-250Hz)": (60, 250),
    "Low Mids (250-500Hz)": (250, 500),
    "Mids (500-2k Hz)": (500, 2000),
    "Upper Mids (2k-4k Hz)": (2000, 4000),
    "Presence (4k-6k Hz)": (4000, 6000),
    "Highs (6k-8k Hz)": (6000, 8000),
}

band_names = list(BANDS.keys())
band_eff = []
for name in band_names:
    f0, f1 = BANDS[name]
    mask = (freq_bins >= f0) & (freq_bins < f1)
    idx = np.where(mask)[0]
    idx = idx[idx < DT.shape[0]]
    if len(idx) == 0:
        band_eff.append(np.full(DT.shape[1], np.nan))
        continue
    dt_band = DT[idx, :].mean(dim=0).numpy()
    omp_band = OMP[idx, :].mean(dim=0).numpy()
    eff = dt_band / (omp_band + 1e-6) * 100
    band_eff.append(eff)

band_eff = np.array(band_eff)

rtg_grid_path = results_dir / "rtg_grid" / "rtg0_rtg1_override_grid.json"
rtg_grid = json.loads(rtg_grid_path.read_text())
rtg0_vals = rtg_grid["config"]["rtg0_values"]
rtg1_vals = rtg_grid["config"]["rtg1_values"]
rtg_map = {(g["rtg0"], g["rtg1"]): g["dt_final_mean"] for g in rtg_grid["grid"]}
rtg_matrix = np.array([[rtg_map[(r0, r1)] for r1 in rtg1_vals] for r0 in rtg0_vals])

if epochs:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, train_loss, label="Train Loss")
    ax[0].plot(epochs, val_loss, label="Val Loss")
    ax[0].set_title("Training/Validation Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    ax[1].plot(epochs, acc, label="Accuracy", color="#2a9d8f")
    ax[1].set_title("Validation Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=200)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, omp_k, marker="o", label="OMP")
ax.plot(ks, rand_k, marker="o", label="Random")
ax.plot(ks, dt_sparse, marker="o", label="DT (overall mean)")
ax.set_title("OMP vs Random (K-sweep) + DT")
ax.set_xlabel("K")
ax.set_ylabel("Energy Capture")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "omp_vs_random_k_sweep.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(K_full, DT_mean, label="DT")
ax.plot(K_full, OMP_mean, label="OMP")
ax.set_title("DT vs OMP (Overall Mean)")
ax.set_xlabel("K")
ax.set_ylabel("Energy Capture")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "dt_vs_omp_overall.png", dpi=200)
plt.close(fig)

k_to_idx = {k: k - 1 for k in ks}
DT_sparse = [DT_mean[k_to_idx[k]] for k in ks]
OMP_sparse = [OMP_mean[k_to_idx[k]] for k in ks]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, DT_sparse, marker="o", label="DT")
ax.plot(ks, OMP_sparse, marker="o", label="OMP")
ax.plot(ks, rand_k, marker="o", label="Random")
ax.set_title("DT/OMP/Random at Sparse K")
ax.set_xlabel("K")
ax.set_ylabel("Energy Capture")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(fig_dir / "dt_omp_random_sparse.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(band_eff, aspect="auto", cmap="viridis")
ax.set_yticks(range(len(band_names)))
ax.set_yticklabels(band_names)
ax.set_xticks(range(0, DT.shape[1], 2))
ax.set_xticklabels([str(k) for k in range(1, DT.shape[1] + 1, 2)])
ax.set_xlabel("K")
ax.set_title("DT/OMP Efficiency by Band (%)")
fig.colorbar(im, ax=ax, label="Efficiency (%)")
fig.tight_layout()
fig.savefig(fig_dir / "band_efficiency_heatmap.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(rtg_matrix, cmap="magma", aspect="auto")
ax.set_xticks(range(len(rtg1_vals)))
ax.set_xticklabels([str(v) for v in rtg1_vals])
ax.set_yticks(range(len(rtg0_vals)))
ax.set_yticklabels([str(v) for v in rtg0_vals])
ax.set_xlabel("RTG1 override")
ax.set_ylabel("RTG0 override")
ax.set_title("RTG0/RTG1 Override Grid (DT Final Mean)")
fig.colorbar(im, ax=ax, label="DT final mean")
fig.tight_layout()
fig.savefig(fig_dir / "rtg0_rtg1_heatmap.png", dpi=200)
plt.close(fig)

report_path = report_dir / "REPORT.md"

k_table_lines = ["| K | OMP | Random | Gap | Ratio (O/R) |", "|---:|---:|---:|---:|---:|"]
for d in k_sweep["per_k"]:
    k_table_lines.append(
        f"| {d['k']} | {d['omp']:.4f} | {d['random']:.4f} | {d['gap']:.4f} | {d['ratio']:.2f}x |"
    )

cmp_lines = ["| K | DT (overall) | OMP (overall) | Random (k-sweep) |", "|---:|---:|---:|---:|"]
for k, r in zip(ks, rand_k):
    dtv = DT_mean[k - 1]
    ompv = OMP_mean[k - 1]
    cmp_lines.append(f"| {k} | {dtv:.4f} | {ompv:.4f} | {r:.4f} |")

report = f"""# Experiment Report: exp_interspeech_gru2_tw32_lag50_k16_ep50

## Summary
- Dataset: boy1 MIC/LDV WAV pairs (no angle filtering). Found {pairs} pairs.
- Trajectory generation: {blocks} blocks, variants_per_clip={variants}.
- OMP diagnostic: OMP ≫ Random in sparse regime (K=1,2,4,8); gap shrinks near K=16.
- DTmin training completed for 50 epochs; validation loss decreased steadily.
- RTG0/RTG1 override grid shows near-constant DT final mean across overrides.

## Data Sources
- MIC root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- LDV root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing: direct MIC/LDV pairs, no angle filtering.

## Parameters
- STFT: n_fft=2048, hop_length=160, fs=16000, freq bins=1025
- Window: Tw=32 frames
- Lag range: [-50, 50] (M=101 lags)
- K (selection budget): 16
- Gain: 100.0
- Variants per clip: {variants}
- Max items: 1000

## OMP Teacher Setup
- OMP acts as the oracle teacher per frequency bin for each block.
- Total OMP teacher blocks: {blocks}
- Each block produces a per-frequency selection of K atoms over M=101 lags.

## DTmin Student Setup
- Single DTmin student trained across all OMP teacher blocks.
- Model: SeqDT_FreqAware (GRU-based), rtg_dim=2.
- RTG inputs: remaining energy (rtg0) + remaining steps (rtg1).
- Student traverses all frequencies and K steps to match OMP teacher behavior.

## Training Curves
![Training Curves](figures/training_curves.png)

## OMP vs Random (K-sweep)
![OMP vs Random](figures/omp_vs_random_k_sweep.png)

{chr(10).join(k_table_lines)}

## DT vs OMP (Overall Mean Across Frequency Bins)
![DT vs OMP Overall](figures/dt_vs_omp_overall.png)

## DT/OMP/Random at Sparse K
![DT/OMP/Random Sparse](figures/dt_omp_random_sparse.png)

{chr(10).join(cmp_lines)}

**Note:** DT/OMP values are from eval_stats.pt (10 clips). Random values come from the K-sweep diagnostic (3 clips, 5 random trials). The sparse regime still shows OMP ≫ Random.

## Band Efficiency Heatmap (DT/OMP)
![Band Efficiency Heatmap](figures/band_efficiency_heatmap.png)

## RTG0/RTG1 Override Grid
![RTG0/RTG1 Heatmap](figures/rtg0_rtg1_heatmap.png)

## Observations
- OMP strongly dominates Random in sparse K (K=1,2,4), consistent with a meaningful dictionary and sparse selection budget.
- DTmin approaches OMP at higher K; DT/OMP efficiency converges toward ~100% across bands.
- RTG0/RTG1 overrides barely change DT final mean, indicating limited sensitivity of RTG conditioning in this run.

## Current Issues
- **RTG meaning remains unclear**: both RTG0 and RTG1 overrides yield near-identical DT performance, suggesting the RTG inputs are not materially influencing policy decisions.

## Artifacts
- Pipeline log: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/pipeline.log
- Eval stats: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/eval/eval_stats.pt
- OMP vs Random: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/omp_vs_random_k_sweep/omp_vs_random_k_sweep.json
- RTG grid: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/rtg_grid/rtg0_rtg1_override_grid.json
"""

report_path.write_text(report)
print(f"Wrote report to {report_path}")
