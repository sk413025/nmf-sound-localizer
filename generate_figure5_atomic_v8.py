#!/usr/bin/env python3
"""
Generate Figure 5 Atomic Panels (Nature Style) - v8
Changes from v7:
1. Fixed Colorbar Ticks: Increased `cbar_bottom_m` to 6.0mm to ensure ticks are visible.
2. B-Series Proportional Adjustment:
   - Height is set to match A/C side length (28.482 mm).
   - Width is reduced to 20.302 mm to maintain the aspect ratio from v5 (H/W ~ 1.4).
   - This prevents the "square" look while respecting the height constraint.
3. Output Directory: Fixed to `nature-figures/results/results_figure5_v8`.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import NA_matplotlib_guild as na_style

# Set Nature style
na_style.set_nature_rcparams(base_fontsize=7)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
})

# Dimensions (mm)
W_STD = 35.482
H_STD = 50.658
W_POLAR = 49.314
H_POLAR = 102.055

def mm_to_inch(mm):
    return mm / 25.4

def save_outputs_fixed_size(fig, out_prefix, dpi_tiff=300):
    """
    Save figure without bbox_inches='tight' to strictly respect the defined figsize.
    """
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.tiff", dpi=dpi_tiff)

def setup_figure_and_axes(width_mm, height_mm, is_square=False, is_polar=False, custom_w_ax=None):
    """
    Create figure and axes with precise layout.
    """
    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))
    
    if is_polar:
        # Polar plot layout
        top_m = 4.0
        bottom_m = 1.0
        left_m = 1.0
        right_m = 1.0
        
        w_ax = width_mm - left_m - right_m
        h_ax = height_mm - top_m - bottom_m
        
        rect = [left_m/width_mm, bottom_m/height_mm, w_ax/width_mm, h_ax/height_mm]
        ax = fig.add_axes(rect, projection='polar')
        return fig, ax, None

    # Standard Plot Layout
    top_m = 3.5
    left_m = 6.5
    right_m = 0.5 # Default right margin (minimized)
    
    # Colorbar settings
    cbar_h = 1.2
    cbar_bottom_m = 6.0 # Increased to show ticks (was 2.5)
    cbar_space = 3.5
    
    # Calculate Axes Dimensions
    # Default max width
    w_ax_max = width_mm - left_m - right_m
    
    if custom_w_ax is not None:
        w_ax = custom_w_ax
        # If custom width is used, right margin increases
    else:
        w_ax = w_ax_max
    
    if is_square:
        h_ax = w_ax 
    else:
        # For B-series, we want Height = A's side length (28.482)
        # If custom_w_ax is provided (for B), we still want h_ax to be 28.482?
        # Wait, if is_square=False, we usually fill vertical space.
        # But for v8, we want specific height.
        # Let's assume if custom_w_ax is passed, we calculate h_ax based on ratio?
        # No, the caller should control this.
        # Let's just use the logic:
        # If is_square=True, h_ax = w_ax.
        # If is_square=False, h_ax = height_mm - top_m - ... (Standard fill)
        # BUT for B, we want specific height.
        # So I will add `custom_h_ax` argument?
        # Or just use `is_square` logic carefully.
        h_ax = height_mm - top_m - (cbar_bottom_m + cbar_h + cbar_space)

    # Override for B-series logic (handled by caller passing custom_w_ax and is_square=False?)
    # Actually, for B, we want h_ax = 28.482.
    # If I pass is_square=False, h_ax will be calculated as ~39mm (too tall).
    # So I need a way to force h_ax.
    pass

    # Let's refactor slightly to allow explicit height control
    return fig, None, None # Placeholder, logic moved to specific functions or below

def setup_figure_and_axes_v8(width_mm, height_mm, is_square=False, is_polar=False, force_h_ax=None, force_w_ax=None):
    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))
    
    if is_polar:
        top_m = 4.0
        bottom_m = 1.0
        left_m = 1.0
        right_m = 1.0
        w_ax = width_mm - left_m - right_m
        h_ax = height_mm - top_m - bottom_m
        rect = [left_m/width_mm, bottom_m/height_mm, w_ax/width_mm, h_ax/height_mm]
        ax = fig.add_axes(rect, projection='polar')
        return fig, ax, None

    # Standard Layout
    top_m = 3.5
    left_m = 6.5
    right_m_min = 0.5
    
    cbar_h = 1.2
    cbar_bottom_m = 6.0 # Increased for ticks
    cbar_space = 3.5
    
    w_ax_max = width_mm - left_m - right_m_min
    
    if force_w_ax is not None:
        w_ax = force_w_ax
    else:
        w_ax = w_ax_max
        
    if force_h_ax is not None:
        h_ax = force_h_ax
    elif is_square:
        h_ax = w_ax
    else:
        h_ax = height_mm - top_m - (cbar_bottom_m + cbar_h + cbar_space)
        
    # Calculate positions
    # Top aligned
    bottom_ax = height_mm - top_m - h_ax
    
    rect_ax = [left_m/width_mm, bottom_ax/height_mm, w_ax/width_mm, h_ax/height_mm]
    ax = fig.add_axes(rect_ax)
    
    # Colorbar aligned with axes width
    rect_cbar = [left_m/width_mm, cbar_bottom_m/height_mm, w_ax/width_mm, cbar_h/height_mm]
    cax = fig.add_axes(rect_cbar)
    
    return fig, ax, cax

def load_data(run_dir):
    run_dir = Path(run_dir)
    routing_data = np.load(run_dir / "modal_routing_val.npz")
    dict_data = np.load(run_dir / "dictionary.npz", allow_pickle=True)
    return routing_data, dict_data

def compute_global_correlation(routing_data, dict_data):
    scores_expert = routing_data['scores_expert']
    H = dict_data['H']
    expert_corr = np.corrcoef(scores_expert.T)
    H_corr = np.corrcoef(H.T)
    return expert_corr, H_corr

def find_representative_case(routing_data, dict_data):
    scores_expert = routing_data['scores_expert']
    g_energy = routing_data['g_energy_expert']
    labels = routing_data['labels']
    angles = dict_data['angles']
    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)
    angle_145_idx = np.argmin(np.abs(angles - 145))
    angle_60_idx = np.argmin(np.abs(angles - 60))
    for idx in range(len(labels)):
        if (labels[idx] == angle_145_idx and
            g_pred[idx] == angle_60_idx and
            qk_pred[idx] == angle_145_idx):
            return idx, angle_145_idx, angle_60_idx
    qk_correct = (qk_pred == labels)
    g_wrong = (g_pred != labels)
    qk_fix = qk_correct & g_wrong
    if qk_fix.sum() > 0:
        idx = np.where(qk_fix)[0][0]
        return idx, labels[idx], g_pred[idx]
    return 0, labels[0], np.argmax(g_energy[0])

def compute_selection_probability(routing_data):
    scores_expert = routing_data['scores_expert']
    g_energy_expert = routing_data['g_energy_expert']
    labels = routing_data['labels']
    n_angles = 37
    physics_prob = np.zeros((n_angles, n_angles))
    qk_prob = np.zeros((n_angles, n_angles))
    for true_angle in range(n_angles):
        samples_mask = (labels == true_angle)
        n_samples_angle = samples_mask.sum()
        if n_samples_angle == 0: continue
        for expert_idx in range(n_angles):
            physics_selected = (np.argmax(g_energy_expert[samples_mask], axis=1) == expert_idx)
            physics_prob[true_angle, expert_idx] = physics_selected.mean()
            qk_selected = (np.argmax(scores_expert[samples_mask], axis=1) == expert_idx)
            qk_prob[true_angle, expert_idx] = qk_selected.mean()
    return physics_prob, qk_prob

def create_atomic_panels_a(routing_data, dict_data, out_dir):
    print("Creating Atomic Panels A...")
    expert_corr, H_corr = compute_global_correlation(routing_data, dict_data)
    angles = dict_data['angles']
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}°" for i in tick_positions]
    cmap_corr = 'RdBu_r'

    # A1: Physical Structure
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, is_square=True)
    im = ax.imshow(H_corr, cmap=cmap_corr, aspect='equal', vmin=-1.0, vmax=1.0)
    ax.set_title('H Matrix Physical Structure')
    ax.set_xlabel('Angle Index j')
    ax.set_ylabel('Angle Index i')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Correlation', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_A1_PHYSICAL_STRUCTURE"))
    plt.close(fig)

    # A2: QK Learned Structure
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, is_square=True)
    im = ax.imshow(expert_corr, cmap=cmap_corr, aspect='equal', vmin=-1.0, vmax=1.0)
    ax.set_title('QK Learned Structure')
    ax.set_xlabel('Angle Index j')
    ax.set_ylabel('Angle Index i')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Correlation', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_A2_QK_STRUCTURE"))
    plt.close(fig)

def create_atomic_panels_b(routing_data, dict_data, out_dir):
    print("Creating Atomic Panels B...")
    sample_idx, true_expert, wrong_expert = find_representative_case(routing_data, dict_data)
    angles = dict_data['angles']
    D = dict_data['D']
    Y = routing_data['Y_val'][sample_idx]
    scores_atoms_sample = routing_data['scores_atoms'][sample_idx]
    g_energy_atoms = np.abs(D.T @ Y).reshape(37, 8)

    D_true = D[:, true_expert*8:(true_expert+1)*8]
    D_wrong = D[:, wrong_expert*8:(wrong_expert+1)*8]
    g_true_atoms = g_energy_atoms[true_expert, :]
    g_wrong_atoms = g_energy_atoms[wrong_expert, :]

    freqs_all = np.linspace(300, 3000, D_true.shape[0])
    freq_mask = freqs_all <= 2000

    physics_true_heatmap = np.abs(D_true[freq_mask, :]) * g_true_atoms[np.newaxis, :]
    physics_wrong_heatmap = np.abs(D_wrong[freq_mask, :]) * g_wrong_atoms[np.newaxis, :]
    
    qk_true_atoms = scores_atoms_sample[true_expert, :]
    qk_wrong_atoms = scores_atoms_sample[wrong_expert, :]
    qk_true_heatmap = D_true[freq_mask, :] * qk_true_atoms[np.newaxis, :]
    qk_wrong_heatmap = D_wrong[freq_mask, :] * qk_wrong_atoms[np.newaxis, :]

    vmin_phys = min(physics_true_heatmap.min(), physics_wrong_heatmap.min())
    vmax_phys = max(physics_true_heatmap.max(), physics_wrong_heatmap.max())
    cmap_phys = 'viridis'

    # Y-axis settings
    y_ticks = [0, 500, 1000, 1500, 2000]
    y_extent = [0, 8, 0, 2000] # Left, Right, Bottom, Top

    # B Dimensions Logic
    # A/C Side Length = 28.482 mm
    # We want Height = 28.482 mm
    # We want Width = 20.302 mm (to maintain v5 aspect ratio)
    h_ax_b = 28.482
    w_ax_b = 20.302

    # B1 Left: Physics True
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
    im = ax.imshow(physics_true_heatmap, aspect='auto', cmap=cmap_phys,
                   extent=y_extent, interpolation='nearest', origin='lower',
                   vmin=vmin_phys, vmax=vmax_phys)
    ax.set_title(f'Physics: True ({angles[true_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    ax.set_yticks(y_ticks)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Energy', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_B1_PHYSICS_TRUE"))
    plt.close(fig)

    # B1 Right: Physics Wrong
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
    im = ax.imshow(physics_wrong_heatmap, aspect='auto', cmap=cmap_phys,
                   extent=y_extent, interpolation='nearest', origin='lower',
                   vmin=vmin_phys, vmax=vmax_phys)
    ax.set_title(f'Physics: Wrong ({angles[wrong_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    ax.set_yticks(y_ticks)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('Energy', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_B1_PHYSICS_WRONG"))
    plt.close(fig)

    vmax_qk = max(np.abs(qk_true_heatmap).max(), np.abs(qk_wrong_heatmap).max())
    vmin_qk = -vmax_qk
    cmap_qk = 'RdBu_r'

    # B2 Left: QK True
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
    im = ax.imshow(qk_true_heatmap, aspect='auto', cmap=cmap_qk,
                   extent=y_extent, interpolation='nearest', origin='lower',
                   vmin=vmin_qk, vmax=vmax_qk)
    ax.set_title(f'QK: True ({angles[true_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    ax.set_yticks(y_ticks)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('QK Score', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_B2_QK_TRUE"))
    plt.close(fig)

    # B2 Right: QK Wrong
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
    im = ax.imshow(qk_wrong_heatmap, aspect='auto', cmap=cmap_qk,
                   extent=y_extent, interpolation='nearest', origin='lower',
                   vmin=vmin_qk, vmax=vmax_qk)
    ax.set_title(f'QK: Wrong ({angles[wrong_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    ax.set_yticks(y_ticks)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('QK Score', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_B2_QK_WRONG"))
    plt.close(fig)

    # B3: Polar Plot (Unchanged)
    fig, ax, _ = setup_figure_and_axes_v8(W_POLAR, H_POLAR, is_polar=True)
    angles_rad = np.deg2rad(angles)
    true_angle_rad = np.deg2rad(angles[true_expert])
    
    physics_scores = routing_data['g_energy_expert'][sample_idx, :]
    qk_scores = routing_data['scores_expert'][sample_idx, :]
    
    physics_scores_norm = (physics_scores - physics_scores.min()) / \
                         (physics_scores.max() - physics_scores.min() + 1e-12)
    qk_scores_norm = (qk_scores - qk_scores.min()) / \
                    (qk_scores.max() - qk_scores.min() + 1e-12)

    ax.grid(True, alpha=0.2, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    ax.plot(angles_rad, physics_scores_norm, 'o-', color='coral',
            linewidth=0.8, markersize=2, alpha=0.6, 
            label='Traditional OMP', zorder=1)
    ax.fill_between(angles_rad, 0, physics_scores_norm, color='coral', alpha=0.1, zorder=1)
    
    ax.plot(angles_rad, qk_scores_norm, '-', color='darkgreen',
            linewidth=1.5, alpha=0.9, 
            label='Physics-Aware AI', zorder=2)
    ax.fill_between(angles_rad, 0, qk_scores_norm, color='darkgreen', alpha=0.1, zorder=2)

    peak_idx = np.argmax(qk_scores_norm)
    ax.scatter([angles_rad[peak_idx]], [qk_scores_norm[peak_idx]], 
               color='darkgreen', s=15, zorder=3)

    ax.plot([true_angle_rad], [1.05], '*', color='lime', markersize=8,
            markeredgecolor='black', markeredgewidth=0.5,
            label='Ground Truth', zorder=4)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    ax.set_title('Final Estimation', pad=10)
    
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.05), 
              fontsize=5, handlelength=1.0)
    
    ax.set_ylim(0, 1.1)
    ax.set_yticklabels([])
    
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_B3_POLAR_ESTIMATION"))
    plt.close(fig)

def create_atomic_panels_c(routing_data, out_dir):
    print("Creating Atomic Panels C...")
    physics_prob, qk_prob = compute_selection_probability(routing_data)
    vmax_unified = max(physics_prob.max(), qk_prob.max())
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(i*5)}°" for i in tick_positions]
    cmap_prob = 'viridis'

    # C1: Physics
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, is_square=True)
    im = ax.imshow(physics_prob, cmap=cmap_prob, aspect='equal',
                   extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                   interpolation='nearest')
    ax.set_ylabel('True DOA (θ°)')
    ax.set_xlabel('Selected Expert')
    ax.set_title('Traditional OMP')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('P(select)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_C1_PHYSICS_SELECTION"))
    plt.close(fig)

    # C2: QK
    fig, ax, cax = setup_figure_and_axes_v8(W_STD, H_STD, is_square=True)
    im = ax.imshow(qk_prob, cmap=cmap_prob, aspect='equal',
                   extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                   interpolation='nearest')
    ax.set_ylabel('True DOA (θ°)')
    ax.set_xlabel('Selected Expert')
    ax.set_title('Physics-Aware AI')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label('P(select)', fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    save_outputs_fixed_size(fig, str(out_dir / "Fig5_C2_QK_SELECTION"))
    plt.close(fig)

def main():
    run_dir = "/Users/jnrle/Documents/LDVReorientation/worktrees/nature-comm-repro/results/omp_transformer_speech260_trainval_split_full_20251115_082341"
    # Output relative to the script location
    script_dir = Path(__file__).parent
    out_dir = script_dir / "results" / "results_figure5_v8"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Generating Figure 5 Atomic Panels (Nature Style) - v8")
    print("="*60)

    routing_data, dict_data = load_data(run_dir)

    create_atomic_panels_a(routing_data, dict_data, out_dir)
    create_atomic_panels_b(routing_data, dict_data, out_dir)
    create_atomic_panels_c(routing_data, out_dir)

    print("\nCOMPLETE")

if __name__ == "__main__":
    main()
