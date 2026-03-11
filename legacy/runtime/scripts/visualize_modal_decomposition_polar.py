#!/usr/bin/env python3
"""
Visualize Modal Decomposition: Spectrum & Directivity

This script generates publication-quality visualizations of transfer function modes
from SVD analysis, showing both frequency spectrum and directivity patterns.

Features:
- Savitzky-Golay filtering for noise reduction
- Mirror symmetry for full-circle polar plots
- Cubic spline interpolation for smooth curves
- Side-by-side frequency spectrum and polar directivity plots

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


def process_angular_mode(v_half, angles_half, n_interp=360):
    """
    Process half-circle angular mode to generate smooth full-circle data.

    Processing pipeline:
    1. Savitzky-Golay Filter: Smooth raw data, remove high-frequency noise
    2. Mirror Symmetry: Create full circle assuming left-right symmetry
    3. CubicSpline Interpolation: Generate high-density smooth curve
    4. Absolute Value: Ensure non-negative values
    5. Normalization: Scale to [0, 1] for comparison across modes

    Parameters
    ----------
    v_half : ndarray, shape (N,)
        Half-circle angular mode values (0-180°)
    angles_half : ndarray, shape (N,)
        Half-circle angles (0-180°)
    n_interp : int, default=360
        Number of interpolation points (one point per degree)

    Returns
    -------
    v_norm : ndarray, shape (n_interp,)
        Normalized smooth full-circle mode values
    angles_smooth : ndarray, shape (n_interp,)
        Smooth full-circle angles
    """

    # === [1] Savitzky-Golay Filter ===
    # Smooth original data to remove high-frequency noise
    # window_length must be odd and < len(v_half)
    window_length = min(11, len(v_half) if len(v_half) % 2 == 1 else len(v_half) - 1)
    polyorder = 3  # Polynomial order

    v_smoothed = savgol_filter(v_half, window_length=window_length,
                                polyorder=polyorder, mode='nearest')

    # === [2] Mirror Symmetry ===
    # Create full circle data (mirror symmetry)
    # Assumption: Acoustic field is left-right symmetric (physically reasonable)
    v_full = np.concatenate([
        v_smoothed,              # 0-180°
        v_smoothed[::-1][1:-1]   # 185-175° (mirrored, remove duplicate endpoints)
    ])

    angles_full = np.concatenate([
        angles_half,                    # 0-180°
        360 - angles_half[::-1][1:-1]   # 185-355°
    ])

    # === [3] CubicSpline Interpolation ===
    # Use cubic spline interpolation for high-density smooth curve
    # Add periodic boundary condition (connect end to start)

    # For periodicity, append first point to end (360° = 0°)
    angles_periodic = np.append(angles_full, 360)
    v_periodic = np.append(v_full, v_full[0])

    # Create cubic spline with periodic boundary conditions
    cs = CubicSpline(angles_periodic, v_periodic, bc_type='periodic')

    # Generate high-density angle grid
    angles_smooth = np.linspace(0, 360, n_interp, endpoint=False)
    v_interp = cs(angles_smooth)

    # === [4] Absolute Value ===
    v_abs = np.abs(v_interp)

    # === [5] Normalization ===
    # Normalize to [0, 1] for comparison across different modes
    v_max = np.max(v_abs)
    if v_max > 1e-10:
        v_norm = v_abs / v_max
    else:
        v_norm = v_abs

    return v_norm, angles_smooth


def process_frequency_mode(u_r, freqs, smooth=True, n_interp=None):
    """
    Process frequency mode with full smoothing pipeline.

    Processing pipeline (when smooth=True):
    1. Absolute Value
    2. Savitzky-Golay Filter: Smooth raw data
    3. CubicSpline Interpolation: Generate smooth curve (optional)

    Parameters
    ----------
    u_r : ndarray, shape (F,)
        Frequency mode values
    freqs : ndarray, shape (F,)
        Frequency axis
    smooth : bool, default=True
        Whether to apply smoothing
    n_interp : int, optional
        Number of interpolation points (if None, use original points)

    Returns
    -------
    u_processed : ndarray
        Processed frequency mode
    freqs_out : ndarray
        Frequency axis (interpolated if n_interp is specified)
    """
    # === [1] Absolute Value ===
    u_abs = np.abs(u_r)

    if not smooth:
        if n_interp is None:
            return u_abs, freqs
        else:
            # Simple linear interpolation without smoothing
            freqs_interp = np.linspace(freqs[0], freqs[-1], n_interp)
            u_interp = np.interp(freqs_interp, freqs, u_abs)
            return u_interp, freqs_interp

    # === [2] Savitzky-Golay Filter ===
    # Use wider window for frequency (more data points)
    # Aim for similar relative coverage as angular mode (~30%)
    target_coverage = 0.30
    window_length = max(15, int(len(u_abs) * target_coverage))
    # Ensure odd
    if window_length % 2 == 0:
        window_length += 1
    # Ensure not too large
    window_length = min(window_length, len(u_abs) - 2)

    polyorder = 3  # Cubic polynomial
    u_smoothed = savgol_filter(u_abs, window_length=window_length,
                                 polyorder=polyorder, mode='nearest')

    # === [3] CubicSpline Interpolation (optional) ===
    if n_interp is not None and n_interp > len(freqs):
        # Create cubic spline
        cs = CubicSpline(freqs, u_smoothed)

        # Generate high-density frequency grid
        freqs_interp = np.linspace(freqs[0], freqs[-1], n_interp)
        u_interp = cs(freqs_interp)

        # Ensure non-negative (physical constraint)
        u_interp = np.maximum(u_interp, 0)

        return u_interp, freqs_interp
    else:
        return u_smoothed, freqs


def plot_modal_decomposition(npz_path, output_path, n_modes=3,
                              n_interp=360, smooth_freq=True, dpi=300):
    """
    Plot modal decomposition: Spectrum & Directivity.

    Parameters
    ----------
    npz_path : str or Path
        Path to transfer_modes.npz file
    output_path : str or Path
        Output figure path
    n_modes : int, default=3
        Number of modes to plot
    n_interp : int, default=360
        Number of interpolation points for polar plot
    smooth_freq : bool, default=True
        Whether to smooth frequency spectrum
    dpi : int, default=300
        Output resolution (dots per inch)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Generated figure
    """

    # === Load Data ===
    print(f"Loading data from: {npz_path}")
    data = np.load(npz_path)
    U = data['U']              # (F, E): Frequency modes
    V = data['V']              # (E, E): Angular modes
    freqs = data['freqs']      # (F,): Frequency axis
    angles_deg = data['angles_deg']  # (E,): Angle axis

    print(f"Data shapes: U={U.shape}, V={V.shape}")
    print(f"Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
    print(f"Angle range: {angles_deg[0]:.1f} - {angles_deg[-1]:.1f}°")
    print(f"Plotting top {n_modes} modes...")

    # === Setup Figure ===
    fig = plt.figure(figsize=(14, 3.5 * n_modes))
    gs = GridSpec(n_modes, 2, figure=fig,
                  width_ratios=[1.3, 1],
                  hspace=0.35, wspace=0.4,
                  left=0.08, right=0.95, top=0.94, bottom=0.06)

    # Professional color scheme
    colors = ['#4A90E2', '#F5A623', '#7ED321']  # Blue, Orange, Green

    # === Plot Each Mode ===
    for idx in range(n_modes):
        r = idx  # Mode index
        color = colors[idx % len(colors)]

        print(f"\nProcessing Mode {r+1}/{n_modes}...")

        # --- Left: Frequency Spectrum |u_r(f)| ---
        ax_freq = fig.add_subplot(gs[idx, 0])

        u_r = U[:, r]

        # Apply same interpolation density as angular mode for consistency
        n_freq_interp = len(freqs) * 2 if smooth_freq else None
        u_processed, freqs_plot = process_frequency_mode(u_r, freqs,
                                                         smooth=smooth_freq,
                                                         n_interp=n_freq_interp)

        print(f"  Frequency spectrum: {len(u_processed)} points (from {len(freqs)})")
        print(f"  Amplitude range: {np.min(u_processed):.4f} - {np.max(u_processed):.4f}")

        # Plot spectrum
        ax_freq.plot(freqs_plot, u_processed, color=color, linewidth=2,
                     label=f'Mode {r+1}')
        ax_freq.fill_between(freqs_plot, 0, u_processed,
                              color=color, alpha=0.25)

        # Styling
        ax_freq.set_ylabel(f'$|u_{{{r+1}}}(f)|$', fontsize=14,
                           fontweight='bold', color=color)
        ax_freq.set_xlim(freqs[0], freqs[-1])
        ax_freq.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax_freq.tick_params(labelsize=11)

        # Only show x-axis label at bottom
        if idx == n_modes - 1:
            ax_freq.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        else:
            ax_freq.set_xticklabels([])

        # --- Right: Directivity Polar Plot |v_r(θ)| ---
        ax_polar = fig.add_subplot(gs[idx, 1], projection='polar')

        v_r = V[:, r]
        v_smooth, angles_smooth = process_angular_mode(
            v_r, angles_deg, n_interp=n_interp
        )

        print(f"  Directivity pattern: {len(v_smooth)} points (interpolated from {len(v_r)})")
        print(f"  Normalized amplitude range: {np.min(v_smooth):.4f} - {np.max(v_smooth):.4f}")

        # Convert to radians
        theta = np.deg2rad(angles_smooth)

        # Plot polar
        ax_polar.plot(theta, v_smooth, color=color, linewidth=2.5)
        ax_polar.fill(theta, v_smooth, color=color, alpha=0.25)

        # Polar plot settings
        ax_polar.set_theta_zero_location('N')    # 0° at top
        ax_polar.set_theta_direction(-1)         # Clockwise

        # Direction labels
        ax_polar.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax_polar.set_xticklabels(['0°', '90°', '180°', '270°'],
                                  fontsize=11, fontweight='bold')

        # Radial labels
        ax_polar.set_rlabel_position(45)
        ax_polar.tick_params(labelsize=10)

        # Title
        ax_polar.set_title(f'$|v_{{{r+1}}}(\\theta)|$',
                           fontsize=14, fontweight='bold',
                           color=color, pad=15)

        # Grid
        ax_polar.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)

    # === Overall Title ===
    fig.suptitle('Modal Decomposition: Spectrum & Directivity',
                 fontsize=18, fontweight='bold', y=0.995)

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
        description='Visualize modal decomposition with smooth polar plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/visualize_modal_decomposition_polar.py \\
    --npz_path results/omp_transformer_speech260_trainval_split_full_20251115_082341/transfer_modes.npz \\
    --output modal_decomposition_smooth.png

  # High resolution with frequency smoothing
  python scripts/visualize_modal_decomposition_polar.py \\
    --npz_path results/.../transfer_modes.npz \\
    --output modal_decomposition_ultra_smooth.png \\
    --n_modes 3 \\
    --n_interp 720 \\
    --smooth_freq \\
    --dpi 600
        """
    )

    parser.add_argument('--npz_path', type=str, required=True,
                        help='Path to transfer_modes.npz file')
    parser.add_argument('--output', type=str,
                        default='modal_decomposition_smooth.png',
                        help='Output figure path (default: modal_decomposition_smooth.png)')
    parser.add_argument('--n_modes', type=int, default=3,
                        help='Number of modes to plot (default: 3)')
    parser.add_argument('--n_interp', type=int, default=360,
                        help='Number of interpolation points for polar plot (default: 360)')
    parser.add_argument('--smooth_freq', action='store_true',
                        help='Apply smoothing to frequency spectrum')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution in DPI (default: 300)')

    args = parser.parse_args()

    # Validate inputs
    npz_path = Path(args.npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    # Run visualization
    plot_modal_decomposition(
        npz_path=args.npz_path,
        output_path=args.output,
        n_modes=args.n_modes,
        n_interp=args.n_interp,
        smooth_freq=args.smooth_freq,
        dpi=args.dpi
    )


if __name__ == "__main__":
    main()
