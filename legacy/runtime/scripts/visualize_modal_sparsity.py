#!/usr/bin/env python3
"""
Visualize Modal Sparsity: Linear-scale emphasis on few dominant modes

This script creates a dual Y-axis visualization that emphasizes the sparsity
of the SVD decomposition, showing that only a few modes capture most of the
transfer function information.

Key Features:
- Linear scale for singular values (emphasizes sparsity vs log scale)
- Dual Y-axis: singular values (left) and cumulative metrics (right)
- Visual indicators for effective rank (90%, 95%, 99% thresholds)
- Automatic determination of significant mode cutoff

Processing:
1. Load singular values and cumulative metrics from transfer_modes.npz
2. Identify effective rank (e.g., modes capturing 90% energy)
3. Plot singular values with linear scale
4. Overlay cumulative energy and DOA capacity curves
5. Add vertical lines marking reconstruction quality thresholds

Purpose:
Demonstrate that the transfer function W(f,θ) has a highly sparse representation,
requiring only R_eff ≈ 10-15 modes (out of 37 total) for accurate reconstruction.

Author: Claude Code
Date: 2025-12-04
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from pathlib import Path


def find_effective_rank(cumulative_values, thresholds=[0.90, 0.95, 0.99]):
    """
    Find mode indices where cumulative metric reaches specified thresholds.

    Parameters
    ----------
    cumulative_values : ndarray
        Cumulative metric values (monotonically increasing to 1.0)
    thresholds : list of float
        Threshold values to find (e.g., [0.90, 0.95, 0.99])

    Returns
    -------
    ranks : dict
        Dictionary mapping threshold to mode index (1-indexed)
    """
    ranks = {}
    for thresh in thresholds:
        # Find first index where cumulative value exceeds threshold
        idx = np.searchsorted(cumulative_values, thresh)
        if idx < len(cumulative_values):
            ranks[thresh] = idx + 1  # 1-indexed mode number
        else:
            ranks[thresh] = len(cumulative_values)
    return ranks


def plot_modal_sparsity(npz_path, output_path, dpi=300,
                         energy_thresh=0.90, doa_thresh=0.90,
                         show_all_modes=False):
    """
    Create linear-scale visualization emphasizing modal sparsity.

    Parameters
    ----------
    npz_path : str or Path
        Path to transfer_modes.npz file
    output_path : str or Path
        Output figure path
    dpi : int, default=300
        Output resolution
    energy_thresh : float, default=0.90
        Energy threshold for effective rank visualization
    doa_thresh : float, default=0.90
        DOA capacity threshold for effective rank visualization
    show_all_modes : bool, default=False
        If True, show all modes; if False, auto-truncate display at effective rank

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """

    # === Load Data ===
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path)
    S = data['S']                    # Singular values
    cum_energy = data['cum_energy']  # Cumulative energy fraction
    cum_doa_cap = data['cum_doa_cap']  # Cumulative DOA capacity

    n_modes = len(S)
    r_idx = np.arange(1, n_modes + 1)  # 1-indexed mode numbers

    print(f"\nData loaded:")
    print(f"  Total modes: {n_modes}")
    print(f"  Singular value range: [{S.min():.4f}, {S.max():.4f}]")

    # Calculate dynamic range (handle near-zero values)
    S_nonzero = S[S > 1e-10]
    if len(S_nonzero) > 0:
        dynamic_range_linear = S.max() / S_nonzero.min()
        dynamic_range_db = 20 * np.log10(dynamic_range_linear)
        print(f"  Dynamic range (linear): {dynamic_range_linear:.1f}x")
        print(f"  Dynamic range (dB): {dynamic_range_db:.1f} dB")
    else:
        print(f"  Dynamic range: Unable to compute (all values near zero)")

    # === Find Effective Ranks ===
    # Include custom thresholds along with standard ones
    standard_thresholds = [0.90, 0.95, 0.99]
    all_thresholds = sorted(set(standard_thresholds + [energy_thresh, doa_thresh]))
    energy_ranks = find_effective_rank(cum_energy, all_thresholds)
    doa_ranks = find_effective_rank(cum_doa_cap, all_thresholds)

    print(f"\nEffective ranks (energy-based):")
    for thresh, rank in energy_ranks.items():
        print(f"  {thresh*100:.0f}% energy: {rank} modes")

    print(f"\nEffective ranks (DOA capacity-based):")
    for thresh, rank in doa_ranks.items():
        print(f"  {thresh*100:.0f}% DOA capacity: {rank} modes")

    # Determine display cutoff
    if not show_all_modes:
        # Show up to 99% energy threshold + 5 modes for context
        display_cutoff = min(energy_ranks[0.99] + 5, n_modes)
    else:
        display_cutoff = n_modes

    print(f"\nDisplay settings:")
    print(f"  Showing first {display_cutoff} modes (out of {n_modes} total)")

    # === Create Figure ===
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # === Left Y-axis: Singular Values (Linear Scale) ===
    color_sv = '#2E86AB'  # Deep blue
    ax1.plot(r_idx[:display_cutoff], S[:display_cutoff],
             color=color_sv, marker='o', markersize=8,
             linewidth=2.5, label='Singular values σᵣ')

    # Highlight dominant modes with fill
    dominant_cutoff = energy_ranks[energy_thresh]
    ax1.fill_between(r_idx[:dominant_cutoff], 0, S[:dominant_cutoff],
                      color=color_sv, alpha=0.15)

    ax1.set_xlabel('Mode Index r', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Singular Value σᵣ (linear scale)',
                   fontsize=14, fontweight='bold', color=color_sv)
    ax1.tick_params(axis='y', labelcolor=color_sv, labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlim(0.5, display_cutoff + 0.5)
    ax1.set_ylim(0, S[:display_cutoff].max() * 1.1)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # === Right Y-axis: Cumulative Metrics ===
    ax2 = ax1.twinx()

    color_energy = '#06A77D'  # Green
    color_doa = '#D62828'     # Red

    ax2.plot(r_idx[:display_cutoff], cum_energy[:display_cutoff],
             color=color_energy, marker='s', markersize=6,
             linewidth=2, linestyle='-', label='Cumulative energy')
    ax2.plot(r_idx[:display_cutoff], cum_doa_cap[:display_cutoff],
             color=color_doa, marker='^', markersize=6,
             linewidth=2, linestyle='-', label='Cumulative DOA capacity')

    ax2.set_ylabel('Cumulative Fraction', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set_ylim(0, 1.05)

    # === Add Threshold Markers ===
    # Energy threshold
    energy_rank = energy_ranks[energy_thresh]
    ax1.axvline(x=energy_rank, color=color_energy,
                linestyle='--', linewidth=2.5, alpha=0.7)
    ax1.text(energy_rank, S[:display_cutoff].max() * 0.95,
             f'{energy_thresh*100:.0f}% energy\n(r={energy_rank})',
             fontsize=11, fontweight='bold', color=color_energy,
             ha='left', va='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=color_energy, alpha=0.8))

    # DOA capacity threshold
    doa_rank = doa_ranks[doa_thresh]
    if doa_rank != energy_rank and abs(doa_rank - energy_rank) > 2:
        ax1.axvline(x=doa_rank, color=color_doa,
                    linestyle='--', linewidth=2.5, alpha=0.7)
        ax1.text(doa_rank, S[:display_cutoff].max() * 0.85,
                 f'{doa_thresh*100:.0f}% DOA\n(r={doa_rank})',
                 fontsize=11, fontweight='bold', color=color_doa,
                 ha='left', va='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                          edgecolor=color_doa, alpha=0.8))

    # === Title and Legends ===
    ax1.set_title('Modal Decomposition Sparsity: Few Modes Dominate Reconstruction',
                  fontsize=16, fontweight='bold', pad=15)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=12, framealpha=0.95,
               edgecolor='black', fancybox=True)

    # === Summary Text Box ===
    summary_text = (
        f"Sparsity Summary:\n"
        f"• Total modes: {n_modes}\n"
        f"• {energy_thresh*100:.0f}% energy: {energy_rank} modes "
        f"({energy_rank/n_modes*100:.1f}%)\n"
        f"• {doa_thresh*100:.0f}% DOA capacity: {doa_rank} modes "
        f"({doa_rank/n_modes*100:.1f}%)\n"
        f"• Compression: {n_modes/energy_rank:.1f}× fewer modes"
    )

    ax1.text(0.02, 0.98, summary_text,
             transform=ax1.transAxes,
             fontsize=11, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                      edgecolor='black', alpha=0.9, linewidth=2))

    # === Save ===
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')

    print(f"\n✅ Saved: {output_path}")
    print(f"   Resolution: {dpi} DPI")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    # Print sparsity analysis
    print(f"\n=== Sparsity Analysis ===")
    print(f"Only {energy_rank} out of {n_modes} modes ({energy_rank/n_modes*100:.1f}%) "
          f"capture {energy_thresh*100:.0f}% of energy")
    print(f"This represents a {n_modes/energy_rank:.1f}× compression in modal representation")
    print(f"Linear scale visualization clearly shows dominance of first {energy_rank} modes")

    return fig


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Visualize modal sparsity with linear-scale emphasis on dominant modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default thresholds (90% energy, 90% DOA)
  python scripts/visualize_modal_sparsity.py \\
    --npz_path results/omp_transformer_speech260_trainval_split_full_20251115_082341/transfer_modes.npz \\
    --output modal_sparsity_linear.png

  # Show all modes (no auto-truncation)
  python scripts/visualize_modal_sparsity.py \\
    --npz_path results/.../transfer_modes.npz \\
    --output modal_sparsity_all_modes.png \\
    --show_all_modes

  # Custom thresholds for effective rank
  python scripts/visualize_modal_sparsity.py \\
    --npz_path results/.../transfer_modes.npz \\
    --output modal_sparsity_95pct.png \\
    --energy_thresh 0.95 \\
    --doa_thresh 0.95 \\
    --dpi 600
        """
    )

    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to transfer_modes.npz file')
    parser.add_argument('--output', type=str,
                        default='modal_sparsity_linear.png',
                        help='Output figure path (default: modal_sparsity_linear.png)')
    parser.add_argument('--energy_thresh', type=float, default=0.90,
                        help='Energy threshold for effective rank (default: 0.90)')
    parser.add_argument('--doa_thresh', type=float, default=0.90,
                        help='DOA capacity threshold for effective rank (default: 0.90)')
    parser.add_argument('--show_all_modes', action='store_true',
                        help='Show all modes instead of auto-truncating at effective rank')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Validate inputs
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Run visualization
    plot_modal_sparsity(
        npz_path=args.npz_path,
        output_path=args.output,
        dpi=args.dpi,
        energy_thresh=args.energy_thresh,
        doa_thresh=args.doa_thresh,
        show_all_modes=args.show_all_modes
    )


if __name__ == "__main__":
    main()
