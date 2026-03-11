#!/usr/bin/env python3
"""
Visualize Modal Dictionary Heatmap: Structured Physical Dictionary D

This script generates heatmap visualization of SVD mode dictionaries,
showing frequency-angle structure for each mode side-by-side.

Key Features:
- IDENTICAL processing pipeline to polar plot visualization
- Full-circle mirroring (0-360°) for angular modes, then extract [0, 180°]
- Consistent smoothing parameters across all visualizations
- Independent normalization per mode to highlight structural features

Processing Pipeline:
1. Frequency modes: Abs → Savitzky-Golay → CubicSpline
2. Angular modes: Savitzky-Golay → Mirror → CubicSpline (periodic) → Abs
3. Dictionary: Outer product D_r = u_r ⊗ v_r
4. Normalization: Each mode normalized to [0, 1] independently

This ensures the heatmap represents exactly the same processed components
shown in the polar plot visualization.

Author: Claude Code
Date: 2025-12-04
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import argparse
from pathlib import Path


def process_frequency_mode_heatmap(u_r, freqs, n_interp=692):
    """
    Process frequency mode for heatmap (consistent with polar plot).

    Parameters
    ----------
    u_r : ndarray, shape (F,)
        Frequency mode values
    freqs : ndarray, shape (F,)
        Frequency axis
    n_interp : int, default=692
        Number of interpolation points (2x original)

    Returns
    -------
    u_interp : ndarray
        Smoothed and interpolated frequency mode
    freqs_interp : ndarray
        Interpolated frequency axis
    """
    # === [1] Absolute Value ===
    u_abs = np.abs(u_r)

    # === [2] Savitzky-Golay Filter ===
    # Parameters consistent with polar plot
    target_coverage = 0.30
    window_length = max(15, int(len(u_abs) * target_coverage))
    if window_length % 2 == 0:
        window_length += 1
    window_length = min(window_length, len(u_abs) - 2)

    polyorder = 3
    u_smoothed = savgol_filter(u_abs, window_length=window_length,
                                polyorder=polyorder, mode='nearest')

    # === [3] CubicSpline Interpolation ===
    if n_interp > len(freqs):
        cs = CubicSpline(freqs, u_smoothed)
        freqs_interp = np.linspace(freqs[0], freqs[-1], n_interp)
        u_interp = cs(freqs_interp)
        u_interp = np.maximum(u_interp, 0)  # Non-negative
    else:
        u_interp = u_smoothed
        freqs_interp = freqs

    return u_interp, freqs_interp


def process_angular_mode_heatmap(v_half, angles_half, n_interp_full=360):
    """
    Process angular mode for heatmap with IDENTICAL processing to polar plot.

    This function uses the SAME processing pipeline as the polar plot visualization
    to ensure complete consistency:
    1. Savitzky-Golay Filter
    2. Mirror Symmetry (create full 0-360° data)
    3. CubicSpline Interpolation with periodic boundary conditions
    4. Absolute Value
    5. Extract [0, 180°] portion for heatmap display

    This ensures that the heatmap dictionary D_r = u_r ⊗ v_r represents the
    exact same processed components shown in the polar plot.

    Parameters
    ----------
    v_half : ndarray, shape (E,)
        Half-circle angular mode values (0-180°)
    angles_half : ndarray, shape (E,)
        Half-circle angular axis (0-180°)
    n_interp_full : int, default=360
        Number of interpolation points for full circle (before extraction)

    Returns
    -------
    v_half_processed : ndarray, shape (n_interp_full//2 + 1,)
        Processed angular mode for [0, 180°] range
    angles_half_processed : ndarray, shape (n_interp_full//2 + 1,)
        Angle axis for [0, 180°] range
    """
    # === [1] Savitzky-Golay Filter ===
    # IDENTICAL to polar plot
    window_length = min(11, len(v_half) if len(v_half) % 2 == 1 else len(v_half) - 1)
    polyorder = 3

    v_smoothed = savgol_filter(v_half, window_length=window_length,
                                polyorder=polyorder, mode='nearest')

    # === [2] Mirror Symmetry ===
    # IDENTICAL to polar plot - create full circle data
    v_full = np.concatenate([
        v_smoothed,              # 0-180°
        v_smoothed[::-1][1:-1]   # 185-355° (mirrored, exclude endpoints)
    ])

    angles_full = np.concatenate([
        angles_half,                    # 0-180°
        360 - angles_half[::-1][1:-1]   # 185-355°
    ])

    # === [3] CubicSpline Interpolation with Periodic BC ===
    # IDENTICAL to polar plot - periodic boundary conditions
    angles_periodic = np.append(angles_full, 360)
    v_periodic = np.append(v_full, v_full[0])

    cs = CubicSpline(angles_periodic, v_periodic, bc_type='periodic')

    # Generate full-circle high-density grid
    angles_smooth = np.linspace(0, 360, n_interp_full, endpoint=False)
    v_interp = cs(angles_smooth)

    # === [4] Absolute Value ===
    # IDENTICAL to polar plot - abs AFTER spline interpolation
    v_abs = np.abs(v_interp)

    # === [5] Extract [0, 180°] Portion ===
    # For heatmap display, we only need the first half
    # n_interp_full=360 means 1 point per degree, so [0, 180] inclusive = 181 points
    n_half = n_interp_full // 2 + 1
    v_half_processed = v_abs[:n_half]
    angles_half_processed = angles_smooth[:n_half]

    return v_half_processed, angles_half_processed


def create_mode_dictionaries(U, V, freqs, angles_deg, mode_indices,
                              n_freq_interp=692, n_angle_interp_full=360):
    """
    Create smoothed dictionary matrices for specified modes.

    For each mode r:
        D_r(f, θ) = u_r(f) ⊗ v_r(θ)  [outer product]

    Processing uses IDENTICAL pipeline to polar plot visualization:
    - Angular: Full-circle processing (0-360°) with mirroring, then extract [0, 180°]
    - Frequency: Standard Abs → Smooth → Spline pipeline

    Parameters
    ----------
    U : ndarray, shape (F, E)
        Frequency modes from SVD
    V : ndarray, shape (E, E)
        Angular modes from SVD
    freqs : ndarray, shape (F,)
        Frequency axis
    angles_deg : ndarray, shape (E,)
        Angular axis (0-180°)
    mode_indices : list of int
        Mode indices to process (e.g., [0, 1, 2] for top 3 modes)
    n_freq_interp : int, default=692
        Frequency interpolation density (2x original = 346*2 = 692)
    n_angle_interp_full : int, default=360
        Full-circle angular interpolation density (output will be half + 1 = 181 points)

    Returns
    -------
    dictionaries : list of ndarray
        List of dictionary matrices, each shape (n_freq_interp, 181)
    freqs_interp : ndarray
        Interpolated frequency axis
    angles_interp : ndarray
        Interpolated angular axis [0, 180°]
    """
    dictionaries = []

    for r in mode_indices:
        print(f"\nProcessing Mode {r+1}...")

        # === Frequency Mode ===
        u_r = U[:, r]
        u_interp, freqs_interp = process_frequency_mode_heatmap(
            u_r, freqs, n_interp=n_freq_interp
        )
        print(f"  Frequency: {len(freqs)} → {len(u_interp)} points")

        # === Angular Mode ===
        # Process with full-circle mirroring (identical to polar plot)
        # Then extract [0, 180°] portion for heatmap
        v_r = V[:, r]
        v_interp, angles_interp = process_angular_mode_heatmap(
            v_r, angles_deg, n_interp_full=n_angle_interp_full
        )
        print(f"  Angular: {len(angles_deg)} → {len(v_interp)} points")

        # === Outer Product: Dictionary ===
        D_r = np.outer(u_interp, v_interp)  # (n_freq, n_angle)
        print(f"  Dictionary shape: {D_r.shape}")
        print(f"  Value range: [{D_r.min():.6f}, {D_r.max():.6f}]")

        # === Independent Normalization ===
        # Each mode normalized to [0, 1] to show its own features
        D_r_max = D_r.max()
        if D_r_max > 1e-10:
            D_r_norm = D_r / D_r_max
        else:
            D_r_norm = D_r

        print(f"  Normalized range: [{D_r_norm.min():.6f}, {D_r_norm.max():.6f}]")

        dictionaries.append(D_r_norm)

    return dictionaries, freqs_interp, angles_interp


def plot_dictionary_heatmap(dictionaries, freqs, angles, mode_indices,
                             output_path, dpi=300):
    """
    Plot dictionary heatmap with horizontal concatenation of modes.

    Parameters
    ----------
    dictionaries : list of ndarray
        List of normalized dictionary matrices
    freqs : ndarray
        Frequency axis
    angles : ndarray
        Angular axis (0-180°)
    mode_indices : list of int
        Mode indices
    output_path : str or Path
        Output figure path
    dpi : int, default=300
        Output resolution

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """
    n_modes = len(mode_indices)

    # === Horizontal Concatenation ===
    D_concat = np.hstack(dictionaries)  # (n_freq, n_angle * n_modes)

    print(f"\n=== Plotting ===")
    print(f"Concatenated dictionary shape: {D_concat.shape}")
    print(f"  Frequency bins: {len(freqs)}")
    print(f"  Total angular bins: {len(angles) * n_modes}")

    # === Figure Setup ===
    fig, ax = plt.subplots(figsize=(16, 8))

    # === Heatmap ===
    # extent: [left, right, bottom, top]
    extent = [0, len(angles) * n_modes, freqs[0], freqs[-1]]

    im = ax.imshow(
        D_concat,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='bilinear',
        extent=extent,
        vmin=0,
        vmax=1
    )

    # === Mode Separators ===
    for i in range(1, n_modes):
        ax.axvline(x=i * len(angles), color='white',
                   linestyle='--', linewidth=2.5, alpha=0.8)

    # === Axes Labels ===
    ax.set_ylabel('Frequency Bin (Hz)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mode', fontsize=14, fontweight='bold')
    ax.set_title('Structured Physical Dictionary D (SVD Modes)',
                 fontsize=16, fontweight='bold', pad=15)

    # === X-axis: Mode Labels ===
    mode_centers = [(i + 0.5) * len(angles) for i in range(n_modes)]
    ax.set_xticks(mode_centers)
    ax.set_xticklabels([f'Mode {r+1}' for r in mode_indices],
                        fontsize=13, fontweight='bold')

    # === Y-axis: Frequency Ticks ===
    ax.tick_params(axis='y', labelsize=11)

    # === Colorbar ===
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Amplitude', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)

    # === Grid ===
    ax.grid(False)

    # === Save ===
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\n✅ Saved: {output_path}")
    print(f"   Resolution: {dpi} DPI")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    return fig


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Visualize modal dictionary as frequency-angle heatmap',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/visualize_modal_dictionary_heatmap.py \\
    --npz_path results/omp_transformer_speech260_trainval_split_full_20251115_082341/transfer_modes.npz \\
    --output dictionary_heatmap.png

  # High resolution with 6 modes
  python scripts/visualize_modal_dictionary_heatmap.py \\
    --npz_path results/.../transfer_modes.npz \\
    --output dictionary_heatmap_6modes.png \\
    --n_modes 6 \\
    --dpi 600
        """
    )

    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to transfer_modes.npz file')
    parser.add_argument('--output', type=str,
                        default='dictionary_heatmap.png',
                        help='Output figure path (default: dictionary_heatmap.png)')
    parser.add_argument('--n_modes', type=int, default=3,
                        help='Number of modes to plot (default: 3)')
    parser.add_argument('--n_freq_interp', type=int, default=692,
                        help='Frequency interpolation density (default: 692, 2x original)')
    parser.add_argument('--n_angle_interp_full', type=int, default=360,
                        help='Full-circle angular interpolation (default: 360). '
                             'Output will show [0, 180°] = 181 points. '
                             'Processing uses full mirroring for consistency with polar plot.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Validate inputs
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # === Load Data ===
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path)
    U = data['U']
    V = data['V']
    freqs = data['freqs']
    angles_deg = data['angles_deg']

    print(f"\nData loaded:")
    print(f"  U (frequency modes): {U.shape}")
    print(f"  V (angular modes): {V.shape}")
    print(f"  Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz ({len(freqs)} bins)")
    print(f"  Angular range: {angles_deg[0]:.1f} - {angles_deg[-1]:.1f}° ({len(angles_deg)} angles)")

    # Mode indices to process
    mode_indices = list(range(args.n_modes))
    print(f"\nProcessing modes: {[r+1 for r in mode_indices]}")

    # === Create Dictionaries ===
    dictionaries, freqs_interp, angles_interp = create_mode_dictionaries(
        U, V, freqs, angles_deg, mode_indices,
        n_freq_interp=args.n_freq_interp,
        n_angle_interp_full=args.n_angle_interp_full
    )

    # === Plot ===
    plot_dictionary_heatmap(
        dictionaries, freqs_interp, angles_interp, mode_indices,
        output_path=args.output,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
