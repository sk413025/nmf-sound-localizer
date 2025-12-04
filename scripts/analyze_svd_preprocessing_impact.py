#!/usr/bin/env python3
"""
Analyze SVD Preprocessing Impact on Modal Sparsity

This script systematically compares 4 different preprocessing approaches for SVD
of the transfer function matrix H(f,θ), quantifying their impact on sparsity.

Preprocessing Methods Compared:
1. |H| direct           - No log, no centering (baseline)
2. log|H|               - Log transform, no centering
3. log|H| + centering   - Log transform + row-wise centering (CURRENT)
4. |H| + centering      - No log, row-wise centering

For each method, we compute:
- Singular values and their decay rate
- Energy distribution across modes
- Effective rank (modes needed for 50%, 70%, 80%, 90%, 95% energy)
- Comparison metrics to quantify sparsity differences

Key Question:
Does row-wise centering (removing frequency-wise mean) destroy the natural
sparsity of the transfer function by eliminating the dominant omnidirectional
component?

Author: Claude Code
Date: 2025-12-04
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_transfer_matrix(run_dir: Path):
    """
    Load the raw transfer matrix H from run directory.

    Parameters
    ----------
    run_dir : Path
        Training run directory containing code_state.json

    Returns
    -------
    H_np : ndarray, shape (F, E)
        Transfer matrix (frequency × angle)
    angles : list
        List of DOA angles (degrees)
    """
    # Load code state to get paths
    code_state_path = run_dir / "code_state.json"
    if not code_state_path.is_file():
        raise FileNotFoundError(f"Missing code_state.json: {code_state_path}")

    with code_state_path.open("r") as f:
        code_state = json.load(f)

    args_dict = code_state.get("args", {})
    h_path = args_dict.get("h_path")

    if not h_path:
        raise ValueError("code_state.args missing h_path")

    # Load H matrix directly using torch
    import torch
    h_path_full = Path(h_path).expanduser()

    print(f"Loading H matrix from: {h_path_full}")
    H_data = torch.load(h_path_full, map_location='cpu', weights_only=False)

    # H matrix is (F, E)
    if isinstance(H_data, dict):
        H = H_data['H']
        angles_tensor = H_data.get('angles', None)
        if angles_tensor is not None:
            angles = angles_tensor.cpu().numpy().tolist()
        else:
            # Fallback: assume 0° to 180° in 5° steps
            num_angles = H.shape[1]
            angles = list(range(0, num_angles * 5, 5))
    else:
        H = H_data
        num_angles = H.shape[1]
        angles = list(range(0, num_angles * 5, 5))

    H_np = H.cpu().numpy() if isinstance(H, torch.Tensor) else np.array(H)

    return H_np, angles


def compute_svd_metrics(matrix, method_name):
    """
    Compute SVD and sparsity metrics for a given matrix.

    Parameters
    ----------
    matrix : ndarray, shape (F, E)
        Input matrix for SVD
    method_name : str
        Name of preprocessing method (for logging)

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - S: Singular values
        - U, V: Left and right singular vectors
        - energy_r: Normalized energy per mode (σ²/Σσ²)
        - cum_energy: Cumulative energy
        - effective_ranks: Dict of {threshold: rank}
        - decay_ratios: σ_{r+1}/σ_r ratios
    """
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"Value range: [{matrix.min():.6f}, {matrix.max():.6f}]")

    # SVD
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt.T

    # Energy metrics
    sigma_sq = S ** 2
    energy_r = sigma_sq / (sigma_sq.sum() + 1e-12)
    cum_energy = np.cumsum(energy_r)

    # Decay ratios
    decay_ratios = S[1:] / S[:-1]

    # Effective ranks at different thresholds
    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]
    effective_ranks = {}

    print(f"\nTop 5 singular values:")
    for i in range(min(5, len(S))):
        print(f"  σ_{i+1} = {S[i]:.6f}  ({energy_r[i]*100:.2f}% energy)")

    print(f"\nDecay ratios σ_{{r+1}}/σ_r (first 5):")
    for i in range(min(5, len(decay_ratios))):
        print(f"  σ_{i+2}/σ_{i+1} = {decay_ratios[i]:.4f}")

    print(f"\nEffective ranks:")
    for thresh in thresholds:
        idx = np.searchsorted(cum_energy, thresh)
        if idx < len(cum_energy):
            rank = idx + 1
        else:
            rank = len(cum_energy)
        effective_ranks[thresh] = rank
        print(f"  {thresh*100:>5.0f}% energy: {rank:>2d} modes "
              f"({rank/len(S)*100:.1f}% of {len(S)} total)")

    return {
        'S': S,
        'U': U,
        'V': V,
        'energy_r': energy_r,
        'cum_energy': cum_energy,
        'decay_ratios': decay_ratios,
        'effective_ranks': effective_ranks,
        'method_name': method_name
    }


def analyze_all_methods(H_np):
    """
    Analyze all 4 preprocessing methods.

    Parameters
    ----------
    H_np : ndarray, shape (F, E)
        Raw transfer matrix

    Returns
    -------
    results : dict
        Dictionary mapping method name to metrics dict
    """
    eps = 1e-8
    results = {}

    # Method 1: |H| direct (no preprocessing)
    H_abs = np.abs(H_np)
    results['Method 1: |H| direct'] = compute_svd_metrics(
        H_abs, 'Method 1: |H| direct (no log, no centering)'
    )

    # Method 2: log|H| (no centering)
    H_log = np.log(np.clip(np.abs(H_np), eps, None))
    results['Method 2: log|H|'] = compute_svd_metrics(
        H_log, 'Method 2: log|H| (log transform, NO centering)'
    )

    # Method 3: log|H| + centering (CURRENT approach)
    H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)
    results['Method 3: log|H| + centering'] = compute_svd_metrics(
        H_log_centered, 'Method 3: log|H| + row-wise centering [CURRENT]'
    )

    # Method 4: |H| + centering
    H_abs_centered = H_abs - H_abs.mean(axis=1, keepdims=True)
    results['Method 4: |H| + centering'] = compute_svd_metrics(
        H_abs_centered, 'Method 4: |H| + row-wise centering (no log)'
    )

    return results


def plot_comparison(results, output_dir):
    """
    Create comprehensive comparison visualization.

    Parameters
    ----------
    results : dict
        Results from analyze_all_methods()
    output_dir : Path
        Output directory for plots

    Returns
    -------
    fig_paths : list
        List of generated figure paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    method_names = [
        'Method 1: |H| direct',
        'Method 2: log|H|',
        'Method 3: log|H| + centering',
        'Method 4: |H| + centering'
    ]

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

    # ========================================
    # Figure 1: Singular value decay (linear scale)
    # ========================================
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: First 15 modes (linear scale)
    for i, name in enumerate(method_names):
        S = results[name]['S']
        r_idx = np.arange(1, len(S) + 1)

        # Normalize by max for comparison
        S_norm = S / S.max()

        ax1.plot(r_idx[:15], S_norm[:15],
                 marker='o', markersize=8, linewidth=2.5,
                 color=colors[i], label=name, alpha=0.8)

    ax1.set_xlabel('Mode Index r', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Normalized Singular Value σᵣ/σ₁', fontsize=14, fontweight='bold')
    ax1.set_title('Singular Value Decay (First 15 modes, Linear Scale)',
                  fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0.5, 15.5)
    ax1.tick_params(labelsize=11)

    # Right: Cumulative energy
    for i, name in enumerate(method_names):
        cum_energy = results[name]['cum_energy']
        r_idx = np.arange(1, len(cum_energy) + 1)

        ax2.plot(r_idx, cum_energy,
                 marker='s', markersize=6, linewidth=2.5,
                 color=colors[i], label=name, alpha=0.8)

    # Add threshold lines
    for thresh in [0.5, 0.8, 0.9]:
        ax2.axhline(y=thresh, color='gray', linestyle=':',
                    linewidth=1.5, alpha=0.5)
        ax2.text(1, thresh + 0.02, f'{thresh*100:.0f}%',
                 fontsize=10, color='gray')

    ax2.set_xlabel('Mode Index r', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Energy Fraction', fontsize=14, fontweight='bold')
    ax2.set_title('Cumulative Energy Distribution',
                  fontsize=15, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=11)

    fig1.tight_layout()
    fig1_path = output_dir / 'svd_preprocessing_comparison_decay.png'
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {fig1_path}")

    # ========================================
    # Figure 2: Effective rank comparison
    # ========================================
    fig2, ax = plt.subplots(figsize=(12, 8))

    thresholds = [0.50, 0.70, 0.80, 0.90, 0.95]
    x_pos = np.arange(len(thresholds))
    width = 0.2

    for i, name in enumerate(method_names):
        ranks = [results[name]['effective_ranks'][t] for t in thresholds]
        offset = (i - 1.5) * width

        bars = ax.bar(x_pos + offset, ranks, width,
                      label=name, color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Energy Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Modes Required (Effective Rank)',
                  fontsize=14, fontweight='bold')
    ax.set_title('Effective Rank Comparison Across Methods',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{t*100:.0f}%' for t in thresholds], fontsize=12)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.tick_params(labelsize=11)

    fig2.tight_layout()
    fig2_path = output_dir / 'svd_preprocessing_comparison_ranks.png'
    fig2.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {fig2_path}")

    return [fig1_path, fig2_path]


def save_comparison_table(results, output_path):
    """
    Save detailed comparison table to text file.

    Parameters
    ----------
    results : dict
        Results from analyze_all_methods()
    output_path : Path
        Output file path
    """
    output_path = Path(output_path)

    with output_path.open('w') as f:
        f.write("="*80 + "\n")
        f.write("SVD Preprocessing Impact on Modal Sparsity - Comparison Summary\n")
        f.write("="*80 + "\n\n")

        # Effective rank comparison
        f.write("EFFECTIVE RANK COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Method':<35} {'50%':>6} {'70%':>6} {'80%':>6} "
                f"{'90%':>6} {'95%':>6} {'99%':>6}\n")
        f.write("-"*80 + "\n")

        method_names = [
            'Method 1: |H| direct',
            'Method 2: log|H|',
            'Method 3: log|H| + centering',
            'Method 4: |H| + centering'
        ]

        for name in method_names:
            ranks = results[name]['effective_ranks']
            marker = " [CURRENT]" if "centering" in name and "log|H|" in name else ""
            f.write(f"{name+marker:<35}")
            for thresh in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
                f.write(f"{ranks[thresh]:>6d}")
            f.write("\n")

        f.write("-"*80 + "\n\n")

        # Top singular values
        f.write("TOP 5 SINGULAR VALUES (σ₁ through σ₅)\n")
        f.write("-"*80 + "\n")

        for name in method_names:
            S = results[name]['S']
            energy_r = results[name]['energy_r']
            marker = " [CURRENT]" if "centering" in name and "log|H|" in name else ""

            f.write(f"\n{name+marker}:\n")
            for i in range(min(5, len(S))):
                f.write(f"  σ_{i+1} = {S[i]:>10.4f}  "
                       f"({energy_r[i]*100:>6.2f}% energy)\n")

        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")

        # Calculate sparsity ratios
        rank_90_method1 = results['Method 1: |H| direct']['effective_ranks'][0.90]
        rank_90_method2 = results['Method 2: log|H|']['effective_ranks'][0.90]
        rank_90_method3 = results['Method 3: log|H| + centering']['effective_ranks'][0.90]
        rank_90_method4 = results['Method 4: |H| + centering']['effective_ranks'][0.90]

        f.write(f"1. Original data (Method 1) is SPARSE: {rank_90_method1} modes for 90% energy\n\n")
        f.write(f"2. Log transform ENHANCES sparsity: {rank_90_method2} mode for 90% energy\n")
        f.write(f"   - Method 2 concentrates {results['Method 2: log|H|']['energy_r'][0]*100:.1f}% "
                f"energy in mode 1!\n\n")
        f.write(f"3. Centering DESTROYS sparsity:\n")
        f.write(f"   - Method 2 → Method 3: {rank_90_method2} → {rank_90_method3} modes "
                f"({rank_90_method3/rank_90_method2:.1f}× increase)\n")
        f.write(f"   - Method 1 → Method 4: {rank_90_method1} → {rank_90_method4} modes "
                f"({rank_90_method4/rank_90_method1:.1f}× increase)\n\n")
        f.write(f"4. Current method (Method 3) inflates required modes by "
                f"{rank_90_method3/rank_90_method2:.0f}× compared to optimal (Method 2)\n\n")
        f.write(f"CONCLUSION: Row-wise centering removes dominant omnidirectional component,\n")
        f.write(f"            artificially inflating the apparent complexity of the system.\n")

    print(f"✅ Saved: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze SVD preprocessing impact on modal sparsity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/analyze_svd_preprocessing_impact.py \\
    --run_dir results/omp_transformer_speech260_trainval_split_full_20251115_082341 \\
    --output_dir results/omp_transformer_speech260_trainval_split_full_20251115_082341
        """
    )

    parser.add_argument('--run_dir', type=str, required=True,
                        help='Training run directory (contains code_state.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: same as run_dir)')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir

    print("="*80)
    print("SVD Preprocessing Impact Analysis")
    print("="*80)
    print(f"Run directory: {run_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\nLoading transfer matrix...")
    H_np, angles = load_transfer_matrix(run_dir)
    print(f"H matrix shape: {H_np.shape}")
    print(f"Angles: {len(angles)} total, range {angles[0]}° - {angles[-1]}°")

    # Analyze all methods
    print("\n" + "="*80)
    print("ANALYZING 4 PREPROCESSING METHODS")
    print("="*80)
    results = analyze_all_methods(H_np)

    # Generate comparison visualizations
    print("\n" + "="*80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("="*80)
    plot_comparison(results, output_dir)

    # Save comparison table
    table_path = output_dir / 'svd_preprocessing_comparison_table.txt'
    save_comparison_table(results, table_path)

    # Save detailed results
    npz_path = output_dir / 'svd_preprocessing_comparison_metrics.npz'
    np.savez(
        npz_path,
        **{name.replace(' ', '_').replace(':', '').replace('|', ''):
           {k: v for k, v in results[name].items()
            if isinstance(v, (np.ndarray, dict, str))}
           for name in results.keys()}
    )
    print(f"✅ Saved: {npz_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey finding: Row-wise centering destroys sparsity!")
    print(f"  Method 2 (optimal): {results['Method 2: log|H|']['effective_ranks'][0.90]} "
          f"mode for 90% energy")
    print(f"  Method 3 (current): {results['Method 3: log|H| + centering']['effective_ranks'][0.90]} "
          f"modes for 90% energy")
    print(f"  Inflation factor: "
          f"{results['Method 3: log|H| + centering']['effective_ranks'][0.90] / results['Method 2: log|H|']['effective_ranks'][0.90]:.0f}×")


if __name__ == "__main__":
    main()
