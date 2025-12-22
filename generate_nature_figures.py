import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
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

def process_angular_mode(v_half, angles_half, n_interp=360):
    """
    Process half-circle angular mode to generate smooth full-circle data.
    Implements Mirror Symmetry: v(360 - theta) = v(theta).
    """
    # === [1] Savitzky-Golay Filter ===
    window_length = min(11, len(v_half) if len(v_half) % 2 == 1 else len(v_half) - 1)
    polyorder = 3
    v_smoothed = savgol_filter(v_half, window_length=window_length, polyorder=polyorder, mode='nearest')

    # === [2] Mirror Symmetry (1 deg to 359 deg) ===
    # v_smoothed corresponds to 0..180.
    # We want to mirror it such that 360-theta has same value.
    # 0 is axis of symmetry? Or 0-180 line?
    # If 1 maps to 359, then 0 is the axis.
    # Sequence: 0, 1, ..., 180, 179, ..., 1
    # Total length: 181 + 179 = 360 points.
    
    v_full = np.concatenate([
        v_smoothed,              # 0-180
        v_smoothed[::-1][1:-1]   # 179-1
    ])
    
    angles_full = np.concatenate([
        angles_half,                    # 0-180
        360 - angles_half[::-1][1:-1]   # 181-359 (approx)
    ])

    # === [3] CubicSpline Interpolation ===
    # Periodic boundary condition
    angles_periodic = np.append(angles_full, 360)
    v_periodic = np.append(v_full, v_full[0])
    
    cs = CubicSpline(angles_periodic, v_periodic, bc_type='periodic')
    angles_smooth = np.linspace(0, 360, n_interp, endpoint=False)
    v_interp = cs(angles_smooth)

    # === [4] Absolute Value ===
    v_abs = np.abs(v_interp)

    # === [5] Normalization ===
    v_max = np.max(v_abs)
    if v_max > 1e-10:
        v_norm = v_abs / v_max
    else:
        v_norm = v_abs
    
    return v_norm, angles_smooth

def plot_singular_values(S, energy_r, doa_cap_r_norm, out_path):
    # Block 1: 45.113 x 94.08 mm
    fig = na_style.make_figure(width_mm=45.113, height_mm=94.08)
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
    # Layout: Left (Freq), Right (Polar)
    fig = na_style.make_figure(width_mm=46.544, height_mm=31.36)
    
    # Adjust layout manually to fit
    # Left: [0.12, 0.2, 0.35, 0.6] (x, y, w, h)
    # Right: [0.55, 0.1, 0.4, 0.8]
    
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3, left=0.15, right=0.95, bottom=0.2, top=0.85)
    
    # === Left: Frequency Mode ===
    ax1 = fig.add_subplot(gs[0])
    u_abs = np.abs(u_vec)
    
    # Smooth u_abs
    window_length = min(21, len(u_abs) if len(u_abs) % 2 == 1 else len(u_abs) - 1)
    u_smooth = savgol_filter(u_abs, window_length=window_length, polyorder=3)
    # Ensure non-negative after smoothing
    u_smooth = np.maximum(u_smooth, 0)
    
    # Color cycle (Blue, Orange, Green)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    color = colors[(mode_idx - 1) % 3]
    
    ax1.plot(freqs, u_smooth, color=color, linewidth=0.8)
    ax1.fill_between(freqs, u_smooth, color=color, alpha=0.15)
    
    ax1.set_ylabel(f"$|u_{mode_idx}(f)|$", labelpad=1)
    # ax1.set_xlabel("Freq (Hz)", labelpad=1) # Might be too crowded
    
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
    D_r = np.outer(u, v_norm) # (F, N_angles)
    
    fig = na_style.make_figure(width_mm=52.437, height_mm=31.36)
    ax = fig.add_subplot(1, 1, 1)
    
    # Extent: angles 0-360, freqs min-max
    extent = [0, 360, freqs[0], freqs[-1]]
    
    im = ax.imshow(D_r, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    
    ax.set_title(f"Dict Mode {mode_idx}", fontsize=7)
    ax.set_xlabel("Angle (°)", fontsize=6, labelpad=1)
    ax.set_ylabel("Freq (Hz)", fontsize=6, labelpad=1)
    
    ax.tick_params(labelsize=5, pad=1)
    
    na_style.save_outputs(fig, str(out_path).replace('.pdf', ''))
    plt.close(fig)
    print(f"Saved {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h_path', type=str, default='h_matrix_normalized_original_to_box.pth')
    parser.add_argument('--out_dir', type=str, default='results_nature')
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Load Data
    H, angles = load_h_matrix(args.h_path)
    H_np = H.cpu().numpy()
    F, E = H_np.shape
    freqs = build_freq_axis(F)
    angles_deg = np.array(angles, dtype=float)
    
    # SVD
    eps = 1e-8
    H_log = np.log(np.clip(np.abs(H_np), eps, None))
    H_log_centered = H_log - H_log.mean(axis=1, keepdims=True)
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
    for r in range(3): # First 3 modes
        u_vec = U[:, r]
        v_vec = V[:, r]
        
        # Process Angular Mode (Smooth & Normalize)
        if not np.all(np.diff(angles_deg) > 0):
            sort_idx = np.argsort(angles_deg)
            angles_sorted = angles_deg[sort_idx]
            v_sorted = v_vec[sort_idx]
        else:
            angles_sorted = angles_deg
            v_sorted = v_vec
            
        v_norm, angles_smooth = process_angular_mode(v_sorted, angles_sorted)
        
        # 2-4. Modal Decomposition (Freq + Polar)
        plot_modal_mode(u_vec, freqs, v_norm, angles_smooth, r+1, out_dir / f"Fig{r+2}_mode{r+1}.pdf")
        
        # 5-7. Dictionary Heatmap
        plot_dictionary_heatmap(u_vec, v_norm, freqs, angles_smooth, r+1, out_dir / f"Fig{r+5}_dict_mode{r+1}.pdf")

if __name__ == "__main__":
    main()
