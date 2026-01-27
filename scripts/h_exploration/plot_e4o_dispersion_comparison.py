#!/usr/bin/env python3
"""
E4o Dispersion Comparison: Visualize how DTmin per-freq lags reduce dispersion.

Uses existing E4o experiment data to compare:
- none: no phase equalization (baseline)
- agg: DTmin per-freq aggregated phase equalization
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_per_band_data(jsonl_path: Path) -> dict:
    """Load per-band tau and r2 from JSONL."""
    tau_by_band_all = []
    r2_by_band_all = []
    spread_all = []

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get('coarse_source') == 'dt':  # Use DT's coarse delay
                tau = d.get('phase_slope_tau_hat_ms_by_band')
                r2 = d.get('phase_slope_r2_by_band')
                spread = d.get('tau_band_spread_ms')
                if tau and all(t is not None for t in tau):
                    tau_by_band_all.append(tau)
                    r2_by_band_all.append(r2)
                    if spread is not None:
                        spread_all.append(spread)

    return {
        'tau_by_band': np.array(tau_by_band_all),  # (N, 3)
        'r2_by_band': np.array(r2_by_band_all),
        'spread': np.array(spread_all),
    }


def main():
    # Load data from three conditions
    conditions = {
        'none': 'results/e4o_speech_scale_none/subsample_delay_diagnostics.jsonl',
        'agg': 'results/e4o_speech_scale_agg/subsample_delay_diagnostics.jsonl',
    }

    data = {}
    for name, path in conditions.items():
        if Path(path).exists():
            data[name] = load_per_band_data(Path(path))
            print(f"Loaded {name}: {len(data[name]['tau_by_band'])} windows")
        else:
            print(f"Missing: {path}")

    if len(data) < 2:
        print("Not enough data!")
        return

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Band labels
    bands = ['300-900 Hz', '900-1800 Hz', '1800-3000 Hz']
    band_centers = [600, 1350, 2400]

    # Plot 1: Per-band τ distribution comparison
    ax1 = axes[0, 0]
    positions = np.arange(3)
    width = 0.35

    for i, (name, d) in enumerate(data.items()):
        tau = d['tau_by_band']
        means = np.mean(tau, axis=0)
        stds = np.std(tau, axis=0)
        offset = (i - 0.5) * width
        bars = ax1.bar(positions + offset, means, width, label=name, alpha=0.8)
        ax1.errorbar(positions + offset, means, yerr=stds, fmt='none', color='black', capsize=3)

    ax1.set_xticks(positions)
    ax1.set_xticklabels(bands)
    ax1.set_ylabel('τ (ms)')
    ax1.set_title('Per-Band Delay Estimate (Mean ± Std)')
    ax1.legend()
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spread distribution
    ax2 = axes[0, 1]
    for name, d in data.items():
        ax2.hist(d['spread'], bins=30, alpha=0.5, label=f"{name} (median={np.median(d['spread']):.1f})")
    ax2.set_xlabel('τ spread (max - min) (ms)')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Per-Window Dispersion')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-band R² comparison
    ax3 = axes[1, 0]
    for i, (name, d) in enumerate(data.items()):
        r2 = d['r2_by_band']
        means = np.mean(r2, axis=0)
        stds = np.std(r2, axis=0)
        offset = (i - 0.5) * width
        ax3.bar(positions + offset, means, width, label=name, alpha=0.8)
        ax3.errorbar(positions + offset, means, yerr=stds, fmt='none', color='black', capsize=3)

    ax3.set_xticks(positions)
    ax3.set_xticklabels(bands)
    ax3.set_ylabel('R²')
    ax3.set_title('Per-Band Phase-Slope Fit Quality')
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
    for name, d in data.items():
        spread_median = np.median(d['spread'])
        spread_p90 = np.percentile(d['spread'], 90)
        tau_overall_std = np.std(d['tau_by_band'])
        summary_text += f"Condition: {name}\n"
        summary_text += f"  τ_spread median: {spread_median:.2f} ms\n"
        summary_text += f"  τ_spread p90:    {spread_p90:.2f} ms\n"
        summary_text += f"  τ overall std:   {tau_overall_std:.2f} ms\n\n"

    # Improvement
    if 'none' in data and 'agg' in data:
        baseline = np.median(data['none']['spread'])
        improved = np.median(data['agg']['spread'])
        pct = (baseline - improved) / baseline * 100
        summary_text += f"Improvement (agg vs none):\n"
        summary_text += f"  τ_spread: {baseline:.2f} → {improved:.2f} ms\n"
        summary_text += f"  Reduction: {pct:.1f}%\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('E4o: DTmin Per-Freq Lags Reduce Dispersion', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = 'results/e4o_dispersion_viz/e4o_dispersion_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
