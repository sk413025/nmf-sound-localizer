import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
import NA_matplotlib_guild as na_style

# Set Nature style
na_style.set_nature_rcparams()

def load_h_matrix(h_path, device='cpu'):
    print(f"Loading H matrix from: {h_path}")
    h_data = torch.load(h_path, map_location=device, weights_only=False)
    H = h_data['H']  # (F, E)
    angles = h_data['angles']
    return H, angles

def build_freq_axis(F, fs=16000.0, n_fft=2048.0, f_min=300.0, f_max=3000.0):
    df = fs / n_fft
    k_start = int(np.ceil(f_min / df))
    freqs = (k_start + np.arange(F)) * df
    return freqs

def process_frequency_mode(u_r, freqs, smooth=True, n_interp=None):
    """
    Process frequency mode with full smoothing pipeline (Savitzky-Golay).
    Replicates logic from visualize_modal_decomposition_polar.py (commit e3d8462).
    """
    # === [1] Absolute Value ===
    u_abs = np.abs(u_r)

    if not smooth:
        return u_abs, freqs

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

def process_angular_mode(v_half, angles_half, n_interp=360):
    """
    Process angular mode to generate smooth full-circle data.
    Handles non-uniform angles and enforces mirror symmetry and periodicity.
    """
    # 1. Construct Full Circle Data via Mirroring
    # Original angles (e.g., 30..150)
    angles_orig = angles_half
    v_orig = v_half
    
    # Mirrored angles (e.g., 330..210)
    angles_mirror = 360.0 - angles_orig
    v_mirror = v_orig # Symmetric magnitude
    
    # Combine
    angles_combined = np.concatenate([angles_orig, angles_mirror])
    v_combined = np.concatenate([v_orig, v_mirror])
    
    # Sort by angle
    sort_idx = np.argsort(angles_combined)
    angles_sorted = angles_combined[sort_idx]
    v_sorted = v_combined[sort_idx]
    
    # 2. Cubic Spline Interpolation (Periodic)
    # We don't need to manually append 360 if we use bc_type='periodic' correctly,
    # but CubicSpline expects strictly increasing x.
    # Also, if 0 and 360 are missing, periodic spline will solve for them.
    
    # However, scipy's CubicSpline with bc_type='periodic' requires y[0] == y[-1] if x[0] and x[-1] are the period boundaries.
    # Here our data doesn't touch the boundaries (starts at 30, ends at 330).
    # So we can't strictly use bc_type='periodic' on the raw data if it doesn't span the full period.
    
    # Instead, we can wrap the data for interpolation:
    # Append (angles + 360) to the end and (angles - 360) to the beginning to guide the spline.
    
    angles_ext = np.concatenate([angles_sorted - 360, angles_sorted, angles_sorted + 360])
    v_ext = np.concatenate([v_sorted, v_sorted, v_sorted])
    
    # Create Spline on extended data
    cs = CubicSpline(angles_ext, v_ext)
    
    # 3. Evaluate on Dense Grid (0 to 360)
    angles_smooth = np.linspace(0, 360, n_interp + 1, endpoint=True)
    v_dense = cs(angles_smooth)
    
    # 4. Smooth the result (Savitzky-Golay)
    # Now we have uniform data, so SavGol is valid and helps remove noise/wiggles
    window_length = 31 # ~30 degrees window
    if window_length > len(v_dense): window_length = len(v_dense) // 2 * 2 + 1
    v_smooth = savgol_filter(v_dense, window_length=window_length, polyorder=3, mode='wrap')
    
    # 5. Absolute Value & Normalization
    v_abs = np.abs(v_smooth)
    v_max = np.max(v_abs)
    if v_max > 1e-10:
        v_norm = v_abs / v_max
    else:
        v_norm = v_abs
        
    return v_norm, angles_smooth

def plot_singular_values(S, energy_r, doa_cap_r_norm, out_path):
    # Block 1: 62.307 x 115.17 mm
    fig = na_style.make_figure(width_mm=62.307, height_mm=115.17)
    ax = fig.add_subplot(1, 1, 1)
    
    r_idx = np.arange(1, len(S) + 1)
    
    # Plot Singular Values
    line1, = ax.semilogy(r_idx, S, marker="o", markersize=2, label="$\sigma_r$", color='C0')
    ax.set_xlabel("Mode index r")
    ax.set_ylabel("Singular value $\sigma_r$")
    
    # Twin axis for capacity
    ax2 = ax.twinx()
    cum_energy = np.cumsum(energy_r)
    cum_doa_cap = np.cumsum(doa_cap_r_norm)
    
    line2, = ax2.plot(r_idx, cum_energy, "g-", marker="s", markersize=2, label="Cum Energy")
    line3, = ax2.plot(r_idx, cum_doa_cap, "r-", marker="^", markersize=2, label="Cum DOA Cap")
    
    ax2.set_ylabel("Cumulative fraction")
    
    # Legend
    lines = [line1, line2, line3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center right", fontsize=5, frameon=False)
    
    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_modal_mode(u_vec, freqs, v_norm, angles_smooth, mode_idx, out_path):
    # Block 2-4: 46.544 x 31.36 mm
    fig = na_style.make_figure(width_mm=46.544, height_mm=31.36)
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3, left=0.15, right=0.95, bottom=0.2, top=0.85)
    
    # === Left: Frequency Mode ===
    ax1 = fig.add_subplot(gs[0])
    u_abs = np.abs(u_vec) # u_vec is already smooth mean from bootstrap
    
    # Color cycle (Blue, Orange, Green)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color = colors[(mode_idx - 1) % 3]
    
    ax1.plot(freqs, u_abs, color=color, linewidth=0.8)
    ax1.fill_between(freqs, u_abs, color=color, alpha=0.15)
    
    ax1.set_ylabel(f"$|u_{mode_idx}(f)|$", labelpad=1)
    
    # Simplify ticks
    ax1.tick_params(axis='both', which='major', pad=1, labelsize=5)
    ax1.set_xticks([500, 1500, 2500])
    ax1.set_xticklabels(['0.5k', '1.5k', '2.5k'])
    ax1.grid(True, alpha=0.3, linewidth=0.3)

    # === Right: Angular Mode (Polar) ===
    ax2 = fig.add_subplot(gs[1], projection='polar')
    angles_rad = np.deg2rad(angles_smooth)
    
    ax2.plot(angles_rad, v_norm, color=color, linewidth=0.8)
    ax2.fill(angles_rad, v_norm, color=color, alpha=0.15)
    
    # Title
    fig.suptitle(f"Mode {mode_idx}", fontsize=7, y=0.98)
    
    # Polar Grid
    ax2.grid(True, alpha=0.3, linewidth=0.3)
    ax2.set_yticklabels([]) # No radial labels
    
    # Angular ticks
    ax2.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax2.set_xticklabels(['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'], fontsize=4)
    ax2.tick_params(pad=-2) # Move labels closer
    
    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def plot_dictionary_heatmap(u, v_norm, freqs, angles_smooth, mode_idx, out_path):
    # Block 5-7: 52.437 x 31.36 mm
    
    # Slice to 0-180 degrees (remove mirror)
    mask = angles_smooth <= 180
    v_half = v_norm[mask]
    
    D_r = np.outer(u, v_half) # (F, N_half)
    
    fig = na_style.make_figure(width_mm=52.437, height_mm=31.36)
    
    # Enforce same margins as plot_modal_mode to ensure alignment
    # plot_modal_mode uses: bottom=0.2, top=0.85
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.2, top=0.85)
    
    ax = fig.add_subplot(1, 1, 1)
    
    # Extent: x is atom index (0..N_half), y is freq
    extent = [0, D_r.shape[1], freqs[0], freqs[-1]]
    
    im = ax.imshow(D_r, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    
    ax.set_title(f"Dict Mode {mode_idx}", fontsize=7)
    ax.set_xlabel("atom index", fontsize=6, labelpad=1)
    ax.set_ylabel("Freq (Hz)", fontsize=6, labelpad=1)
    
    ax.tick_params(labelsize=5, pad=1)
    
    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_path', type=str, default='h_matrix_normalized_original_to_box.pth')
    parser.add_argument('--out_dir', type=str, default='results_nature_repro')
    parser.add_argument('--n_modes', type=int, default=10)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Load Data
    H, angles = load_h_matrix(args.h_path)
    H_np = H.cpu().numpy()
    F, E = H_np.shape
    freqs = build_freq_axis(F)
    angles_deg = np.array(angles, dtype=float)
    
    # Preprocess
    eps = 1e-8
    H_log = np.log(np.clip(np.abs(H_np), eps, None))
    H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)
    
    # Single SVD (No Bootstrap)
    U, S, Vt = np.linalg.svd(H_log_centered, full_matrices=False)
    V = Vt.T
    
    # Capacity Metrics
    sigma_sq = S**2
    energy_r = sigma_sq / (sigma_sq.sum() + 1e-12)
    var_v_r = V.var(axis=0)
    doa_cap_r = energy_r * var_v_r
    doa_cap_r_norm = doa_cap_r / (doa_cap_r.sum() + 1e-12)
    
    # 1. Singular Values
    plot_singular_values(S, energy_r, doa_cap_r_norm, out_dir / "Fig1_singular_values.pdf")
    
    # Process Modes
    for r in range(args.n_modes):
        u_vec = U[:, r]
        v_vec = V[:, r] # 0-180
        
        # Process Frequency Mode (SavGol + CubicSpline)
        u_smooth, freqs_smooth = process_frequency_mode(u_vec, freqs, smooth=True, n_interp=len(freqs)*2)
        
        # Process Angular Mode (SavGol + Mirror + CubicSpline)
        v_norm, angles_smooth = process_angular_mode(v_vec, angles_deg, n_interp=360)
        
        # 2-4. Modal Decomposition (Freq + Polar)
        plot_modal_mode(u_smooth, freqs_smooth, v_norm, angles_smooth, r+1, out_dir / f"Fig{r+2}_mode{r+1}.pdf")
        
        # 5-7. Dictionary Heatmap
        plot_dictionary_heatmap(u_smooth, v_norm, freqs_smooth, angles_smooth, r+1, out_dir / f"Fig{r+5}_dict_mode{r+1}.pdf")

if __name__ == "__main__":
    main()
