#!/usr/bin/env python3
"""
Generate Figure 5 Atomic Panels (Nature Style) - v10
Changes from v9:
1. Replace B3 atom-index panels with three banded line plots (0-500, 500-1000, 1000-2000 Hz).
2. Each B3 plot overlays Traditional OMP, Physics-Aware AI, and Ground Truth angle marker.
3. Output Directory: Fixed to `nature-figures/results/results_figure5_v10`.
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
BAND_CONFIGS = (
    {"min_hz": 0, "max_hz": 500, "label": "0-500 Hz", "suffix": "0_500", "include_upper": False},
    {"min_hz": 500, "max_hz": 1000, "label": "500-1000 Hz", "suffix": "500_1000", "include_upper": False},
    {"min_hz": 1000, "max_hz": 2000, "label": "1000-2000 Hz", "suffix": "1000_2000", "include_upper": True},
    {"min_hz": 300, "max_hz": 3000, "label": "300-3000 Hz (Full Band)", "suffix": "300_3000", "include_upper": True},
)

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

def band_mask(freqs, band_min_hz, band_max_hz, include_upper=False):
    if include_upper:
        return (freqs >= band_min_hz) & (freqs <= band_max_hz)
    return (freqs >= band_min_hz) & (freqs < band_max_hz)

def compute_band_scores(Y, D, scores_atoms, freqs, band_config):
    mask = band_mask(
        freqs,
        band_config["min_hz"],
        band_config["max_hz"],
        include_upper=band_config["include_upper"],
    )
    if not np.any(mask):
        raise ValueError(
            f"No bins for band {band_config['min_hz']}-{band_config['max_hz']} Hz."
        )
    D_band = D[mask, :]
    Y_band = Y[mask]
    atom_proj = D_band.T @ Y_band
    physics_scores = np.abs(atom_proj).reshape(37, 8).sum(axis=1)
    qk_scores = np.abs((atom_proj.reshape(37, 8) * scores_atoms).sum(axis=1))
    return physics_scores, qk_scores

def plot_band_line(out_dir, band_config, angles, physics_scores, qk_scores, true_expert):
    fig, ax, cax = setup_figure_and_axes(W_STD, H_STD, is_square=True)
    cax.remove()

    physics_plot = physics_scores
    qk_plot = qk_scores
    true_angle = angles[true_expert]
    y_top = max(physics_plot.max(), qk_plot.max())
    if y_top <= 0:
        y_top = 1.0

    ax.grid(True, alpha=0.2, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    ax.plot(angles, physics_plot, 'o-', color='coral',
            linewidth=0.8, markersize=2, alpha=0.6,
            label='Traditional OMP', zorder=1)

    ax.plot(angles, qk_plot, '-', color='darkgreen',
            linewidth=1.5, alpha=0.9,
            label='Physics-Aware AI', zorder=2)

    peak_idx = np.argmax(qk_plot)
    ax.scatter([angles[peak_idx]], [qk_plot[peak_idx]],
               color='darkgreen', s=15, zorder=3)

    ax.scatter([true_angle], [y_top * 1.05], marker='*', color='lime', s=45,
               edgecolors='black', linewidths=0.5, label='Ground Truth', zorder=4)

    ax.set_title(band_config["label"])
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Raw Score')
    ax.set_xlim(angles[0], angles[-1])
    ax.set_ylim(0, y_top * 1.1)

    tick_positions = [0, 9, 18, 27, 36]
    tick_positions = [pos for pos in tick_positions if pos < len(angles)]
    ax.set_xticks([angles[pos] for pos in tick_positions])
    ax.set_xticklabels([f"{int(angles[pos])}°" for pos in tick_positions])

    ax.legend(frameon=False, loc='upper right', fontsize=5, handlelength=1.0)

    save_outputs_fixed_size(fig, str(out_dir / f"Fig5_B3_BAND_{band_config['suffix']}"))
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

    freqs = np.linspace(300, 3000, D.shape[0])
    for band_config in BAND_CONFIGS:
        physics_scores, qk_scores = compute_band_scores(
            Y, D, scores_atoms_sample, freqs, band_config
        )
        plot_band_line(out_dir, band_config, angles, physics_scores, qk_scores, true_expert)

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
    out_dir = script_dir / "results" / "results_figure5_v10"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Generating Figure 5 Atomic Panels (Nature Style) - v10")
    print("="*60)

    routing_data, dict_data = load_data(run_dir)

    create_atomic_panels_a(routing_data, dict_data, out_dir)
    create_atomic_panels_b(routing_data, dict_data, out_dir)
    create_atomic_panels_c(routing_data, out_dir)

    print("\nCOMPLETE")

if __name__ == "__main__":
    main()
