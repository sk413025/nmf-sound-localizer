#!/usr/bin/env python3
"""
Generate Figure 5 Atomic Panels (Nature Style) - v3
Refined Aesthetics: Serif fonts, Professional Color Palette, Improved Polar Plot
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import NA_matplotlib_guild as na_style

# Set Nature style but override for Serif fonts as requested
na_style.set_nature_rcparams(base_fontsize=8)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
    "mathtext.fontset": "stix", # LaTeX-like math font
})

def load_data(run_dir):
    """Load routing data and dictionary"""
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

    # Use RdBu_r for correlation (Diverging: Blue=-1, White=0, Red=1)
    cmap_corr = 'RdBu_r'

    # A1: Physical Structure
    fig = na_style.make_figure(width_mm=45, height_mm=40)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(H_corr, cmap=cmap_corr, aspect='auto', vmin=-1.0, vmax=1.0)
    ax.set_title('H Matrix Physical Structure')
    ax.set_xlabel('Angle Index j')
    ax.set_ylabel('Angle Index i')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation')
    na_style.save_outputs(fig, str(out_dir / "Fig5_A1_PHYSICAL_STRUCTURE"))
    plt.close(fig)

    # A2: QK Learned Structure
    fig = na_style.make_figure(width_mm=45, height_mm=40)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(expert_corr, cmap=cmap_corr, aspect='auto', vmin=-1.0, vmax=1.0)
    ax.set_title('QK Learned Structure')
    ax.set_xlabel('Angle Index j')
    ax.set_ylabel('Angle Index i')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Correlation')
    na_style.save_outputs(fig, str(out_dir / "Fig5_A2_QK_STRUCTURE"))
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

    # Shared limits for Physics (Energy - Sequential)
    vmin_phys = min(physics_true_heatmap.min(), physics_wrong_heatmap.min())
    vmax_phys = max(physics_true_heatmap.max(), physics_wrong_heatmap.max())
    cmap_phys = 'viridis'

    # B1 Left: Physics True
    fig = na_style.make_figure(width_mm=45, height_mm=35)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(physics_true_heatmap, aspect='auto', cmap=cmap_phys,
                   extent=[0, 8, 2000, 300], interpolation='nearest',
                   vmin=vmin_phys, vmax=vmax_phys)
    ax.set_title(f'Physics: True ({angles[true_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Energy')
    na_style.save_outputs(fig, str(out_dir / "Fig5_B1_PHYSICS_TRUE"))
    plt.close(fig)

    # B1 Right: Physics Wrong
    fig = na_style.make_figure(width_mm=45, height_mm=35)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(physics_wrong_heatmap, aspect='auto', cmap=cmap_phys,
                   extent=[0, 8, 2000, 300], interpolation='nearest',
                   vmin=vmin_phys, vmax=vmax_phys)
    ax.set_title(f'Physics: Wrong ({angles[wrong_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Energy')
    na_style.save_outputs(fig, str(out_dir / "Fig5_B1_PHYSICS_WRONG"))
    plt.close(fig)

    # Shared limits for QK (Score - Diverging)
    vmax_qk = max(np.abs(qk_true_heatmap).max(), np.abs(qk_wrong_heatmap).max())
    vmin_qk = -vmax_qk
    cmap_qk = 'RdBu_r'

    # B2 Left: QK True
    fig = na_style.make_figure(width_mm=45, height_mm=35)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(qk_true_heatmap, aspect='auto', cmap=cmap_qk,
                   extent=[0, 8, 2000, 300], interpolation='nearest',
                   vmin=vmin_qk, vmax=vmax_qk)
    ax.set_title(f'QK: True ({angles[true_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('QK Score')
    na_style.save_outputs(fig, str(out_dir / "Fig5_B2_QK_TRUE"))
    plt.close(fig)

    # B2 Right: QK Wrong
    fig = na_style.make_figure(width_mm=45, height_mm=35)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(qk_wrong_heatmap, aspect='auto', cmap=cmap_qk,
                   extent=[0, 8, 2000, 300], interpolation='nearest',
                   vmin=vmin_qk, vmax=vmax_qk)
    ax.set_title(f'QK: Wrong ({angles[wrong_expert]:.0f}°)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Atom Index')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('QK Score')
    na_style.save_outputs(fig, str(out_dir / "Fig5_B2_QK_WRONG"))
    plt.close(fig)

    # B3: Polar Plot - REFINED V3
    fig = na_style.make_figure(width_mm=50, height_mm=50)
    ax = fig.add_subplot(111, projection='polar')
    angles_rad = np.deg2rad(angles)
    true_angle_rad = np.deg2rad(angles[true_expert])
    
    physics_scores = routing_data['g_energy_expert'][sample_idx, :]
    qk_scores = routing_data['scores_expert'][sample_idx, :]
    
    physics_scores_norm = (physics_scores - physics_scores.min()) / \
                         (physics_scores.max() - physics_scores.min() + 1e-12)
    qk_scores_norm = (qk_scores - qk_scores.min()) / \
                    (qk_scores.max() - qk_scores.min() + 1e-12)

    # 1. Grid Customization
    ax.grid(True, alpha=0.2, linestyle='--', color='gray')
    ax.set_axisbelow(True) # Grid behind data

    # 2. Traditional OMP: Coral (Nature style), Thin Line
    ax.plot(angles_rad, physics_scores_norm, 'o-', color='coral',
            linewidth=0.8, markersize=2, alpha=0.6, 
            label='Traditional OMP', zorder=1)
    ax.fill_between(angles_rad, 0, physics_scores_norm, color='coral', alpha=0.1, zorder=1)
    
    # 3. Physics-Aware AI: Dark Green (Nature style), Strong Line
    ax.plot(angles_rad, qk_scores_norm, '-', color='darkgreen',
            linewidth=1.5, alpha=0.9, 
            label='Physics-Aware AI', zorder=2)
    ax.fill_between(angles_rad, 0, qk_scores_norm, color='darkgreen', alpha=0.1, zorder=2)

    # Add a scatter point only at the peak to highlight the prediction
    peak_idx = np.argmax(qk_scores_norm)
    ax.scatter([angles_rad[peak_idx]], [qk_scores_norm[peak_idx]], 
               color='darkgreen', s=15, zorder=3)

    # 4. Ground Truth: Star marker only (No line)
    ax.plot([true_angle_rad], [1.05], '*', color='lime', markersize=8,
            markeredgecolor='black', markeredgewidth=0.5,
            label='Ground Truth', zorder=4)

    # 5. Aesthetics
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title('Final Estimation', pad=10, fontsize=9)
    
    # Legend: Frameless, outside, raised higher to avoid overlap
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.45, 1.35), 
              fontsize=6, handlelength=1.5)
    
    ax.set_ylim(0, 1.1)
    # Remove radial tick labels to reduce clutter
    ax.set_yticklabels([])
    
    na_style.save_outputs(fig, str(out_dir / "Fig5_B3_POLAR_ESTIMATION"))
    plt.close(fig)

def create_atomic_panels_c(routing_data, out_dir):
    print("Creating Atomic Panels C...")
    physics_prob, qk_prob = compute_selection_probability(routing_data)
    vmax_unified = max(physics_prob.max(), qk_prob.max())
    tick_positions = [0, 9, 18, 27, 36]
    tick_labels = [f"{int(i*5)}°" for i in tick_positions]

    # Use viridis for probability (Sequential: 0 to 1)
    cmap_prob = 'viridis'

    # C1: Physics
    fig = na_style.make_figure(width_mm=45, height_mm=40)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(physics_prob, cmap=cmap_prob, aspect='auto',
                   extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                   interpolation='nearest')
    ax.set_ylabel('True DOA (θ°)')
    ax.set_xlabel('Selected Expert')
    ax.set_title('Traditional OMP')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(select)')
    na_style.save_outputs(fig, str(out_dir / "Fig5_C1_PHYSICS_SELECTION"))
    plt.close(fig)

    # C2: QK
    fig = na_style.make_figure(width_mm=45, height_mm=40)
    ax = fig.add_subplot(1,1,1)
    im = ax.imshow(qk_prob, cmap=cmap_prob, aspect='auto',
                   extent=[0, 37, 37, 0], vmin=0, vmax=vmax_unified,
                   interpolation='nearest')
    ax.set_ylabel('True DOA (θ°)')
    ax.set_xlabel('Selected Expert')
    ax.set_title('Physics-Aware AI')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('P(select)')
    na_style.save_outputs(fig, str(out_dir / "Fig5_C2_QK_SELECTION"))
    plt.close(fig)

def main():
    # Use absolute path to data in nature-comm-repro worktree
    run_dir = "/Users/jnrle/Documents/LDVReorientation/worktrees/nature-comm-repro/results/omp_transformer_speech260_trainval_split_full_20251115_082341"
    
    # New output directory structure: nature-figures/results/results_figure5_v3
    out_dir = Path("results/results_figure5_v3")
    out_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*60)
    print("Generating Figure 5 Atomic Panels (Nature Style) - v3")
    print("="*60)

    routing_data, dict_data = load_data(run_dir)
    run_path = Path(run_dir)

    create_atomic_panels_a(routing_data, dict_data, out_dir)
    create_atomic_panels_b(routing_data, dict_data, out_dir)
    create_atomic_panels_c(routing_data, out_dir)

    print("\nCOMPLETE")

if __name__ == "__main__":
    main()
