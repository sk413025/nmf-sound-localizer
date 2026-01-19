#!/usr/bin/env python3
"""
Generate Figure 5 Atomic Panels (Nature Style) - v9
Changes from v8:
1. Replace B3 polar plot with three frequency-resolved atom-index panels (500/1000/2000 Hz).
2. Each B3 panel overlays Traditional OMP, Physics-Aware AI, and Ground Truth atom band.
3. Output Directory: Fixed to `nature-figures/results/results_figure5_v9`.
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
FREQ_PANEL_TARGETS_HZ = (500, 1000, 2000)
FREQ_PANEL_BANDWIDTH_HZ = 100

def mm_to_inch(mm):
    return mm / 25.4

def save_outputs_fixed_size(fig, out_prefix, dpi_tiff=300):
    """
    Save figure without bbox_inches='tight' to strictly respect the defined figsize.
    """
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.tiff", dpi=dpi_tiff)

def setup_figure_and_axes(width_mm, height_mm, is_square=False, force_h_ax=None, force_w_ax=None):
    fig = plt.figure(figsize=(mm_to_inch(width_mm), mm_to_inch(height_mm)))

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

def compute_frequency_atom_distributions(Y, D, scores_atoms, target_freq_hz, bandwidth_hz):
    freqs = np.linspace(300, 3000, D.shape[0])
    half_bw = bandwidth_hz / 2.0
    freq_mask = (freqs >= target_freq_hz - half_bw) & (freqs <= target_freq_hz + half_bw)
    if not np.any(freq_mask):
        raise ValueError(f"No bins for {target_freq_hz} Hz within {bandwidth_hz} Hz bandwidth.")
    D_band = D[freq_mask, :]
    Y_band = Y[freq_mask]
    omp_atom = np.abs(D_band.T @ Y_band)
    qk_weights = scores_atoms.reshape(-1)
    qk_atom = np.abs(omp_atom * qk_weights)
    return omp_atom, qk_atom

def normalize_to_unit(*arrays):
    max_val = max(arr.max() for arr in arrays)
    if max_val <= 0:
        return [np.zeros_like(arr) for arr in arrays]
    return [arr / max_val for arr in arrays]

def plot_frequency_panel(out_dir, freq_hz, Y, D, scores_atoms_sample, true_expert):
    omp_atom, qk_atom = compute_frequency_atom_distributions(
        Y, D, scores_atoms_sample, freq_hz, FREQ_PANEL_BANDWIDTH_HZ
    )
    omp_norm, qk_norm = normalize_to_unit(omp_atom, qk_atom)

    atom_idx = np.arange(omp_atom.shape[0])
    gt_start = true_expert * 8
    gt_end = (true_expert + 1) * 8 - 1
    gt_left = max(-0.5, gt_start - 0.5)
    gt_right = min(omp_atom.shape[0] - 0.5, gt_end + 0.5)

    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
    cax.remove()

    ax.axvspan(gt_left, gt_right, color='lime', alpha=0.15, label='Ground Truth', zorder=0)
    ax.plot(atom_idx, omp_norm, color='coral', linewidth=1.0, alpha=0.6,
            label='Traditional OMP', zorder=2)
    ax.plot(atom_idx, qk_norm, color='darkgreen', linewidth=1.2, alpha=0.9,
            label='Physics-Aware AI', zorder=3)

    ax.set_title(f"Frequency {freq_hz} Hz")
    ax.set_xlabel('Atom Index')
    ax.set_ylabel('Normalized Score')
    ax.set_xlim(-0.5, omp_atom.shape[0] - 0.5)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2, linestyle='--', color='gray')

    tick_positions = [idx for idx in (0, 72, 144, 216, 288) if idx < omp_atom.shape[0]]
    ax.set_xticks(tick_positions)

    ax.legend(frameon=False, loc='upper right', fontsize=5, handlelength=1.2)

    save_outputs_fixed_size(fig, str(out_dir / f"Fig5_B3_FREQ_{freq_hz}"))
    plt.close(fig)

def create_atomic_panels_a(routing_data, dict_data, out_dir):
    print("Creating Atomic Panels A...")
    expert_corr, H_corr = compute_global_correlation(routing_data, dict_data)
    angles = dict_data['angles']
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(angles[i])}°" for i in tick_positions]
    cmap_corr = 'RdBu_r'

    # A1: Physical Structure
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, force_h_ax=h_ax_b, force_w_ax=w_ax_b)
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

    for freq_hz in FREQ_PANEL_TARGETS_HZ:
        plot_frequency_panel(out_dir, freq_hz, Y, D, scores_atoms_sample, true_expert)

def create_atomic_panels_c(routing_data, out_dir):
    print("Creating Atomic Panels C...")
    physics_prob, qk_prob = compute_selection_probability(routing_data)
    vmax_unified = max(physics_prob.max(), qk_prob.max())
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(i*5)}°" for i in tick_positions]
    cmap_prob = 'viridis'

    # C1: Physics
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
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
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
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
    out_dir = script_dir / "results" / "results_figure5_v9"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Generating Figure 5 Atomic Panels (Nature Style) - v9")
    print("="*60)

    routing_data, dict_data = load_data(run_dir)

    create_atomic_panels_a(routing_data, dict_data, out_dir)
    create_atomic_panels_b(routing_data, dict_data, out_dir)
    create_atomic_panels_c(routing_data, out_dir)

    print("\nCOMPLETE")

if __name__ == "__main__":
    main()
