#!/usr/bin/env python3
"""
Frequency-Angle Joint Analysis (SVD/Green's Function Approach)

Analogy to PDE/Green's function decomposition:
    Response(freq, angle) = Σ Mode_i(freq) × Mode_i(angle)

Analysis Strategy:
1. Frequency-Angle Joint Confusion: confusion(f, θ_i, θ_j)
2. Modal Decomposition (SVD of H matrix): frequency modes × spatial modes
3. QK Selection in Joint Space: which frequencies for which angles
4. Physical Interpretation: Green's function perspective

Author: Claude
Date: 2025-12-25
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from scipy.linalg import svd
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'DejaVu Sans', 'sans-serif']


def load_data(run_dir):
    """Load all necessary data."""
    run_dir = Path(run_dir)
    routing_data = np.load(run_dir / "modal_routing_val.npz")
    dict_data = np.load(run_dir / "dictionary.npz", allow_pickle=True)

    print("Data loaded:")
    print(f"  Dictionary D: {dict_data['D'].shape}")
    print(f"  H matrix: {dict_data['H'].shape}")
    print(f"  Routing samples: {routing_data['scores_expert'].shape}")

    return routing_data, dict_data


def compute_frequency_angle_confusion(D, routing_data, dict_data):
    """
    Compute 3D confusion tensor: confusion(freq, angle_i, angle_j)

    Question: Which frequencies cause confusion between which angle pairs?
    """
    print("\n=== Phase 1: Frequency-Angle Joint Confusion Analysis ===")

    scores_expert = routing_data['scores_expert']
    g_energy = routing_data['g_energy_expert']
    labels = routing_data['labels']
    angles = dict_data['angles']

    n_samples, n_experts = scores_expert.shape
    n_freqs = D.shape[0]

    # Predictions
    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)

    # Initialize 3D confusion tensor (freq, angle_i, angle_j)
    # This will store: for each frequency, which angle pairs are similar
    freq_angle_similarity = np.zeros((n_freqs, n_experts, n_experts))

    # For each frequency, compute angle pair similarity
    for f in range(n_freqs):
        for i in range(n_experts):
            for j in range(n_experts):
                # Extract frequency f response for angles i and j
                D_i_f = D[f, i*8:(i+1)*8]  # 8 atoms for angle i at freq f
                D_j_f = D[f, j*8:(j+1)*8]  # 8 atoms for angle j at freq f

                # Compute similarity (correlation)
                if D_i_f.std() > 1e-6 and D_j_f.std() > 1e-6:
                    corr, _ = pearsonr(D_i_f, D_j_f)
                    freq_angle_similarity[f, i, j] = corr
                else:
                    freq_angle_similarity[f, i, j] = 1.0 if i == j else 0.0

    # Compute confusion matrices (angle space)
    physics_confusion = np.zeros((n_experts, n_experts))
    qk_confusion = np.zeros((n_experts, n_experts))

    for sample_idx in range(n_samples):
        true = labels[sample_idx]
        g_wrong = g_pred[sample_idx]
        qk_wrong = qk_pred[sample_idx]
        physics_confusion[true, g_wrong] += 1
        qk_confusion[true, qk_wrong] += 1

    # Find top confused pairs
    confused_pairs = []
    for i in range(n_experts):
        for j in range(n_experts):
            if i != j and physics_confusion[i, j] > 10:  # Threshold
                confused_pairs.append((i, j, physics_confusion[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_confused = confused_pairs[:5]

    print(f"\nTop 5 confused angle pairs (physics):")
    for i, j, count in top_confused:
        print(f"  {angles[i]:.0f}° → {angles[j]:.0f}°: {count:.0f} errors")
        # Average frequency similarity for this pair
        avg_sim = freq_angle_similarity[:, i, j].mean()
        print(f"    Average freq similarity: {avg_sim:.3f}")

    result = {
        'freq_angle_similarity': freq_angle_similarity,
        'physics_confusion': physics_confusion,
        'qk_confusion': qk_confusion,
        'top_confused': top_confused,
    }

    return result


def modal_decomposition_svd(H, dict_data):
    """
    Phase 2: Modal Decomposition (analogous to SVD)

    H matrix (346 freq × 37 angles) decomposition:
        H = U @ S @ V^T
        U: frequency modes (346, 346)
        S: singular values (37,)
        V^T: angle modes (37, 37)
    """
    print("\n=== Phase 2: Modal Decomposition (SVD of H matrix) ===")

    H = dict_data['H']  # (346, 37)
    angles = dict_data['angles']

    # SVD decomposition
    U, S, Vt = svd(H, full_matrices=False)

    print(f"H matrix shape: {H.shape}")
    print(f"U (freq modes): {U.shape}")
    print(f"S (singular values): {S.shape}")
    print(f"Vt (angle modes): {Vt.shape}")

    # Analyze singular values
    total_energy = (S**2).sum()
    cumulative_energy = np.cumsum(S**2) / total_energy

    print(f"\nSingular value energy distribution:")
    print(f"  Top 5 singular values: {S[:5]}")
    print(f"  Cumulative energy (top 5): {cumulative_energy[4]:.3f}")
    print(f"  Cumulative energy (top 10): {cumulative_energy[9]:.3f}")

    # Dominant modes
    n_dominant = np.where(cumulative_energy > 0.95)[0][0] + 1
    print(f"  Modes for 95% energy: {n_dominant}")

    result = {
        'U': U,
        'S': S,
        'Vt': Vt,
        'cumulative_energy': cumulative_energy,
        'n_dominant': n_dominant,
    }

    return result


def analyze_qk_joint_space_selection(routing_data, dict_data, confusion_result):
    """
    Phase 3: QK Selection in Frequency-Angle Joint Space

    Question: Not just "which frequencies", but "which frequencies for which angles"
    """
    print("\n=== Phase 3: QK Selection in Frequency-Angle Joint Space ===")

    D = dict_data['D']
    angles = dict_data['angles']
    scores_atoms = routing_data['scores_atoms']
    scores_expert = routing_data['scores_expert']
    g_energy = routing_data['g_energy_expert']
    labels = routing_data['labels']

    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)
    qk_correct = (qk_pred == labels)
    g_correct = (g_pred == labels)

    # QK-corrected cases
    qk_fix_indices = np.where(qk_correct & (~g_correct))[0]

    print(f"QK-corrected cases: {len(qk_fix_indices)}")

    # For top confused pairs, analyze QK selection pattern
    top_confused = confusion_result['top_confused']
    freq_angle_similarity = confusion_result['freq_angle_similarity']

    joint_patterns = []

    for angle_i, angle_j, _ in top_confused[:3]:  # Top 3 pairs
        # Find samples where true=angle_i, physics=angle_j, QK=correct
        cases = [idx for idx in qk_fix_indices
                 if labels[idx] == angle_i and g_pred[idx] == angle_j]

        if len(cases) == 0:
            continue

        print(f"\n  Analyzing pair: {angles[angle_i]:.0f}° vs {angles[angle_j]:.0f}° ({len(cases)} cases)")

        # For these cases, which atoms does QK select?
        avg_qk_scores = scores_atoms[cases].mean(axis=0)  # (37, 8)

        # Atoms selected for true angle (positive scores)
        true_selected = avg_qk_scores[angle_i] > 0
        true_atoms = np.where(true_selected)[0]

        # Atoms suppressed for wrong angle (negative scores)
        wrong_suppressed = avg_qk_scores[angle_j] < 0
        wrong_atoms = np.where(wrong_suppressed)[0]

        print(f"    True angle ({angles[angle_i]:.0f}°): {true_selected.sum()} atoms selected")
        print(f"    Wrong angle ({angles[angle_j]:.0f}°): {wrong_suppressed.sum()} atoms suppressed")

        # Extract 2D frequency-atom structure for BOTH angles
        # Dictionary atoms (346 freq, 8 atoms per angle)
        D_true = D[:, angle_i*8:(angle_i+1)*8]  # (346, 8)
        D_wrong = D[:, angle_j*8:(angle_j+1)*8]  # (346, 8)

        # Concatenate for side-by-side comparison
        D_pair = np.concatenate([D_true, D_wrong], axis=1)  # (346, 16)

        # QK selection scores for both angles (weighted by QK scores)
        # avg_qk_scores has shape (37, 8), we want the two angles
        qk_scores_true = avg_qk_scores[angle_i, :]  # (8,)
        qk_scores_wrong = avg_qk_scores[angle_j, :]  # (8,)
        qk_scores_pair = np.concatenate([qk_scores_true, qk_scores_wrong])  # (16,)

        # Create 2D QK selection pattern: dictionary atoms WEIGHTED by QK selection
        # This shows which frequency components are emphasized/suppressed by QK attention
        # Broadcasting: (346, 16) * (16,) -> (346, 16)
        qk_weighted_atoms = D_pair * qk_scores_pair  # Element-wise multiplication

        # Frequency similarity for this angle pair (can be expanded to 2D for visualization)
        freq_sim = freq_angle_similarity[:, angle_i, angle_j]  # (346,)

        # For 2D visualization: create multiple columns showing the same freq_sim
        # This shows which frequencies are confused for this angle pair
        freq_sim_2d = np.tile(freq_sim[:, np.newaxis], (1, 3))  # (346, 3)

        joint_patterns.append({
            'angle_i': angle_i,
            'angle_j': angle_j,
            'D_pair': D_pair,  # (346, 16) - dictionary atoms for both angles
            'qk_weighted_atoms': qk_weighted_atoms,  # (346, 16) - atoms weighted by QK selection
            'freq_sim_2d': freq_sim_2d,  # (346, 3) - frequency similarity
            'qk_scores_pair': qk_scores_pair,  # (16,) - QK scores for atoms
            'n_cases': len(cases),
        })

    return joint_patterns


def visualize_joint_analysis(confusion_result, svd_result, joint_patterns, dict_data, routing_data, output_dir):
    """
    Phase 4: Joint Visualization

    Create comprehensive visualizations showing frequency-angle coupling
    """
    print("\n=== Phase 4: Joint Visualization ===")

    output_dir = Path(output_dir)
    angles = dict_data['angles']
    freqs = np.linspace(300, 3000, dict_data['D'].shape[0])

    # ============================================================
    # Visualization 1: Confusion Matrices + Similarity Matrix + Global QK Selection
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Panel 1: Physics confusion
    ax = axes[0, 0]
    im = ax.imshow(confusion_result['physics_confusion'], cmap='hot', aspect='auto')
    ax.set_title('Physics Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Angle Index')
    ax.set_ylabel('True Angle Index')
    plt.colorbar(im, ax=ax, label='Error Count')

    # Mark top confused pairs
    for angle_i, angle_j, _ in confusion_result['top_confused'][:5]:
        ax.plot(angle_j, angle_i, 'cx', markersize=10, markeredgewidth=2)

    # Panel 2: QK confusion
    ax = axes[0, 1]
    im = ax.imshow(confusion_result['qk_confusion'], cmap='hot', aspect='auto')
    ax.set_title('QK Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Angle Index')
    ax.set_ylabel('True Angle Index')
    plt.colorbar(im, ax=ax, label='Error Count')

    # Panel 3: Difference (correction effect)
    ax = axes[0, 2]
    diff = confusion_result['physics_confusion'] - confusion_result['qk_confusion']
    im = ax.imshow(diff, cmap='RdBu_r', aspect='auto', vmin=-diff.max(), vmax=diff.max())
    ax.set_title('Correction Effect\n(Physics - QK)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Angle Index')
    ax.set_ylabel('True Angle Index')
    plt.colorbar(im, ax=ax, label='Errors Corrected')

    # Panel 4: Average frequency similarity
    ax = axes[1, 0]
    avg_freq_sim = confusion_result['freq_angle_similarity'].mean(axis=0)
    im = ax.imshow(avg_freq_sim, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Avg Frequency Similarity\n(Across All Frequencies)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Angle Index j')
    ax.set_ylabel('Angle Index i')
    plt.colorbar(im, ax=ax, label='Correlation')

    # Panel 5: Global QK Selection Pattern
    ax = axes[1, 1]
    # Compute average QK expert scores across all QK-corrected cases
    scores_expert = routing_data['scores_expert']
    g_energy = routing_data['g_energy_expert']
    labels = routing_data['labels']

    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)
    qk_correct = (qk_pred == labels)
    g_correct = (g_pred == labels)
    qk_fix_indices = np.where(qk_correct & (~g_correct))[0]

    # Average QK scores across all QK-corrected cases
    avg_qk_selection = scores_expert[qk_fix_indices, :].mean(axis=0)  # (37,)

    # Plot as line plot with markers
    ax.plot(angles, avg_qk_selection, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Highlight top confused angle pairs
    for angle_i, angle_j, _ in confusion_result['top_confused'][:3]:
        ax.axvline(x=angles[angle_i], color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvline(x=angles[angle_j], color='red', linestyle=':', linewidth=2, alpha=0.7)

    ax.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average QK Score', fontsize=12, fontweight='bold')
    ax.set_title('Panel 5: Global QK Selection Pattern\n(Averaged over QK-corrected cases)',
                fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xlim(angles[0], angles[-1])

    # Panel 6: Leave empty or add text summary
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = f"""
Global QK Selection Analysis

Total samples: {len(labels)}
QK correct: {qk_correct.sum()} ({qk_correct.sum()/len(labels)*100:.1f}%)
Physics correct: {g_correct.sum()} ({g_correct.sum()/len(labels)*100:.1f}%)
QK-corrected: {len(qk_fix_indices)} ({len(qk_fix_indices)/len(labels)*100:.1f}%)

Top confused pairs (Physics):
"""
    for i, (angle_i, angle_j, count) in enumerate(confusion_result['top_confused'][:3], 1):
        summary_text += f"  {i}. {angles[angle_i]:.0f}° → {angles[angle_j]:.0f}° ({count:.0f} errors)\n"

    ax.text(0.05, 0.5, summary_text, fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

    fig.suptitle('Global Analysis: Confusion Matrices, Similarity & QK Selection',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "angle_confusion_matrices_global.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: angle_confusion_matrices_global.png")

    # ============================================================
    # Visualization 2: SVD Modal Decomposition
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Panel 1: Singular values
    ax = axes[0, 0]
    ax.plot(svd_result['S'], 'o-', markersize=5)
    ax.set_yscale('log')
    ax.set_xlabel('Mode Index')
    ax.set_ylabel('Singular Value')
    ax.set_title('Singular Values (Log Scale)', fontweight='bold')
    ax.grid(alpha=0.3)

    # Panel 2: Cumulative energy
    ax = axes[0, 1]
    ax.plot(svd_result['cumulative_energy'], 'o-', markersize=4)
    ax.axhline(y=0.95, color='red', linestyle='--', label='95% energy')
    ax.axvline(x=svd_result['n_dominant'], color='red', linestyle='--',
              label=f'{svd_result["n_dominant"]} modes')
    ax.set_xlabel('Number of Modes')
    ax.set_ylabel('Cumulative Energy Ratio')
    ax.set_title('Cumulative Energy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: Top 3 frequency modes (U)
    ax = axes[0, 2]
    for i in range(3):
        ax.plot(freqs, svd_result['U'][:, i], label=f'Mode {i+1}', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Top 3 Frequency Modes (U)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 4: Top 3 angle modes (Vt)
    ax = axes[1, 0]
    for i in range(3):
        ax.plot(angles, svd_result['Vt'][i, :], 'o-', label=f'Mode {i+1}', alpha=0.7)
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Top 3 Angle Modes (V^T)', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 5: Reconstructed H matrix (top 5 modes)
    ax = axes[1, 1]
    H_recon = (svd_result['U'][:, :5] @ np.diag(svd_result['S'][:5]) @ svd_result['Vt'][:5, :])
    im = ax.imshow(H_recon, cmap='viridis', aspect='auto')
    ax.set_title(f'Reconstructed H (top {5} modes)', fontweight='bold')
    ax.set_xlabel('Angle Index')
    ax.set_ylabel('Frequency Bin')
    plt.colorbar(im, ax=ax)

    # Panel 6: Full H matrix
    ax = axes[1, 2]
    im = ax.imshow(dict_data['H'], cmap='viridis', aspect='auto')
    ax.set_title('Original H Matrix (346×37)', fontweight='bold')
    ax.set_xlabel('Angle Index')
    ax.set_ylabel('Frequency Bin')
    plt.colorbar(im, ax=ax)

    fig.suptitle('SVD Modal Decomposition of H Matrix (Frequency × Angle)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_dir / "svd_modal_decomposition.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: svd_modal_decomposition.png")

    # ============================================================
    # Visualization 3: Frequency-Angle Joint Patterns
    # ============================================================
    n_patterns = len(joint_patterns)
    if n_patterns > 0:
        fig, axes = plt.subplots(n_patterns, 3, figsize=(18, 6*n_patterns))

        if n_patterns == 1:
            axes = axes.reshape(1, -1)

        for idx, pattern in enumerate(joint_patterns):
            angle_i = pattern['angle_i']
            angle_j = pattern['angle_j']

            # Panel 1: 2D Dictionary Atoms Comparison (freq × atoms)
            ax = axes[idx, 0]
            D_pair = pattern['D_pair']  # (346, 16)

            # Plot 2D heatmap with both frequency and atom axes
            im = ax.imshow(D_pair, aspect='auto', cmap='viridis',
                          extent=[0, 16, freqs[-1], freqs[0]],  # [left, right, bottom, top]
                          interpolation='nearest')

            # Add vertical line separating the two angles
            ax.axvline(x=8, color='white', linestyle='--', linewidth=2, label='Angle boundary')

            # Labels
            ax.set_xlabel('Atom Index (0-7: True angle, 8-15: Wrong angle)', fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontweight='bold')
            ax.set_title(f'Panel 1: Dictionary Atoms\n{angles[angle_i]:.0f}° (left) vs {angles[angle_j]:.0f}° (right)',
                        fontweight='bold', fontsize=11)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Atom Amplitude')

            # Set x-ticks to show atom indices
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.set_xticklabels(['0', '4', '8', '12', '16'])

            # Panel 2: 2D Frequency-Angle Similarity Map (freq × columns)
            ax = axes[idx, 1]
            freq_sim_2d = pattern['freq_sim_2d']  # (346, 3)

            # Plot 2D heatmap showing frequency similarity
            im = ax.imshow(freq_sim_2d, aspect='auto', cmap='RdBu_r',
                          extent=[0, 3, freqs[-1], freqs[0]],
                          vmin=-1, vmax=1,
                          interpolation='nearest')

            # Labels
            ax.set_xlabel('Similarity Metric', fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontweight='bold')
            ax.set_title(f'Panel 2: Freq Similarity Map\n{angles[angle_i]:.0f}° vs {angles[angle_j]:.0f}° (N={pattern["n_cases"]})',
                        fontweight='bold', fontsize=11)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Correlation (-1 to 1)')

            # Hide x-ticks
            ax.set_xticks([])

            # Panel 3: 2D QK Weighted Atoms (freq × atoms)
            ax = axes[idx, 2]
            qk_weighted = pattern['qk_weighted_atoms']  # (346, 16)

            # Plot 2D heatmap showing atoms weighted by QK selection
            # This shows which frequency components are emphasized (positive QK score)
            # or suppressed (negative QK score) by the attention mechanism
            im = ax.imshow(qk_weighted, aspect='auto', cmap='RdYlGn',
                          extent=[0, 16, freqs[-1], freqs[0]],
                          vmin=-np.abs(qk_weighted).max(),
                          vmax=np.abs(qk_weighted).max(),
                          interpolation='nearest')

            # Add vertical line separating the two angles
            ax.axvline(x=8, color='black', linestyle='--', linewidth=2, alpha=0.7)

            # Labels
            ax.set_xlabel('Atom Index (0-7: True angle, 8-15: Wrong angle)', fontweight='bold')
            ax.set_ylabel('Frequency (Hz)', fontweight='bold')
            ax.set_title(f'Panel 3: QK Weighted Atoms\nAtom Freq Response × QK Selection',
                        fontweight='bold', fontsize=11)

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, label='Weighted Amplitude')

            # Set x-ticks
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.set_xticklabels(['0', '4', '8', '12', '16'])

        fig.suptitle('Frequency-Angle Joint Patterns (Top Confused Pairs)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_dir / "frequency_angle_joint_patterns.png", dpi=150, bbox_inches='tight')
        print(f"✓ Saved: frequency_angle_joint_patterns.png")


def visualize_per_pair_detailed(angle_i, angle_j, D, routing_data, dict_data, confusion_result, output_path):
    """
    Create detailed 6-panel (3×2) analysis for a specific confused angle pair.

    Row 1 (Frequency Domain - Local perspective, 2 angles):
        - Panel 1A: Dictionary Atoms Comparison
        - Panel 1B: QK Weighted Atoms
        - Panel 1C: Frequency Similarity Map

    Row 2 (Angle Domain - Global perspective, all 37 experts):
        - Panel 2A: Expert (37) × Frequency Bands (12 bands) - KEY INNOVATION
        - Panel 2B: QK Selection in Angle Space
        - Panel 2C: Statistics Summary
    """
    print(f"\n  Creating detailed analysis for {dict_data['angles'][angle_i]:.0f}° vs {dict_data['angles'][angle_j]:.0f}°")

    angles = dict_data['angles']
    freqs = np.linspace(300, 3000, D.shape[0])
    scores_expert = routing_data['scores_expert']
    scores_atoms = routing_data['scores_atoms']
    g_energy = routing_data['g_energy_expert']
    labels = routing_data['labels']

    # Get QK-corrected cases for this specific pair
    qk_pred = np.argmax(scores_expert, axis=1)
    g_pred = np.argmax(g_energy, axis=1)
    qk_correct = (qk_pred == labels)
    g_correct = (g_pred == labels)

    # Find cases where true=angle_i, physics=angle_j, QK=correct
    cases = [idx for idx in range(len(labels))
             if labels[idx] == angle_i and g_pred[idx] == angle_j and qk_correct[idx]]

    n_cases = len(cases)
    if n_cases == 0:
        print(f"    No QK-corrected cases found for this pair!")
        return

    # Average QK scores for this pair
    avg_qk_scores_atoms = scores_atoms[cases].mean(axis=0)  # (37, 8)

    # Create figure with 3×2 grid
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1, 1],
                          width_ratios=[1, 1, 1])

    # ========== ROW 1: Frequency Domain (Local - 2 angles) ==========

    # Panel 1A: Dictionary Atoms Comparison (freq × 16 atoms)
    ax = fig.add_subplot(gs[0, 0])
    D_true = D[:, angle_i*8:(angle_i+1)*8]  # (346, 8)
    D_wrong = D[:, angle_j*8:(angle_j+1)*8]  # (346, 8)
    D_pair = np.concatenate([D_true, D_wrong], axis=1)  # (346, 16)

    im = ax.imshow(D_pair, aspect='auto', cmap='viridis',
                  extent=[0, 16, freqs[-1], freqs[0]],
                  interpolation='nearest')
    ax.axvline(x=8, color='white', linestyle='--', linewidth=2, label='Angle boundary')
    ax.set_xlabel('Atom Index (0-7: True, 8-15: Wrong)', fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax.set_title(f'1A: Dictionary Atoms\n{angles[angle_i]:.0f}° vs {angles[angle_j]:.0f}°',
                fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Atom Amplitude')
    ax.set_xticks([0, 4, 8, 12, 16])

    # Panel 1B: QK Weighted Atoms (freq × 16 atoms)
    ax = fig.add_subplot(gs[0, 1])
    qk_scores_true = avg_qk_scores_atoms[angle_i, :]  # (8,)
    qk_scores_wrong = avg_qk_scores_atoms[angle_j, :]  # (8,)
    qk_scores_pair = np.concatenate([qk_scores_true, qk_scores_wrong])  # (16,)
    qk_weighted = D_pair * qk_scores_pair  # (346, 16)

    im = ax.imshow(qk_weighted, aspect='auto', cmap='RdYlGn',
                  extent=[0, 16, freqs[-1], freqs[0]],
                  vmin=-np.abs(qk_weighted).max(),
                  vmax=np.abs(qk_weighted).max(),
                  interpolation='nearest')
    ax.axvline(x=8, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Atom Index (0-7: True, 8-15: Wrong)', fontweight='bold')
    ax.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax.set_title(f'1B: QK Weighted Atoms\nGreen=Selected, Red=Suppressed',
                fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Weighted Amplitude')
    ax.set_xticks([0, 4, 8, 12, 16])

    # Panel 1C: Frequency Similarity Map (freq × 3 columns for display)
    ax = fig.add_subplot(gs[0, 2])
    freq_sim = confusion_result['freq_angle_similarity'][:, angle_i, angle_j]  # (346,)
    freq_sim_2d = np.tile(freq_sim[:, np.newaxis], (1, 3))  # (346, 3)

    im = ax.imshow(freq_sim_2d, aspect='auto', cmap='RdBu_r',
                  extent=[0, 3, freqs[-1], freqs[0]],
                  vmin=-1, vmax=1,
                  interpolation='nearest')
    ax.set_ylabel('Frequency (Hz)', fontweight='bold')
    ax.set_title(f'1C: Frequency Similarity\nCorr={freq_sim.mean():.3f}',
                fontweight='bold', fontsize=12)
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_xticks([])

    # ========== ROW 2: Angle Domain (Global - all 37 experts) ==========

    # Panel 2A: Expert (37) × Frequency Bands (12 bands) - KEY VISUALIZATION
    ax = fig.add_subplot(gs[1, :])  # Span all 3 columns for better visibility
    n_bands = 12
    freq_edges = np.linspace(300, 3000, n_bands+1)
    expert_freq_bands = np.zeros((37, n_bands))

    for expert in range(37):
        D_expert = D[:, expert*8:(expert+1)*8]  # (346, 8)
        for band_idx in range(n_bands):
            freq_mask = (freqs >= freq_edges[band_idx]) & (freqs < freq_edges[band_idx+1])
            # Average across atoms and frequencies in this band
            expert_freq_bands[expert, band_idx] = D_expert[freq_mask, :].mean()

    im = ax.imshow(expert_freq_bands, aspect='auto', cmap='viridis',
                  extent=[freq_edges[0], freq_edges[-1], 37, 0],
                  interpolation='nearest')

    # Highlight the confused pair
    ax.axhline(y=angle_i, color='lime', linewidth=3, label=f'True {angles[angle_i]:.0f}°')
    ax.axhline(y=angle_j, color='red', linewidth=3, label=f'Wrong {angles[angle_j]:.0f}°')

    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expert Index (Angle 0°-180°)', fontsize=12, fontweight='bold')
    ax.set_title('2A: Expert (37) × Frequency Bands (12) - Global Angle-Frequency Structure',
                fontweight='bold', fontsize=13)
    plt.colorbar(im, ax=ax, label='Dict Response')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_yticks(np.arange(0, 37, 5))
    ax.set_yticklabels([f'{angles[i]:.0f}°' for i in range(0, 37, 5)])

    # Panel 2B: QK Selection in Angle Space (1D plot)
    ax = fig.add_subplot(gs[2, 0:2])
    avg_qk_expert_scores = scores_expert[cases, :].mean(axis=0)  # (37,)

    ax.plot(angles, avg_qk_expert_scores, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.axvline(x=angles[angle_i], color='green', linewidth=3, linestyle='--',
              label=f'True {angles[angle_i]:.0f}°', alpha=0.8)
    ax.axvline(x=angles[angle_j], color='red', linewidth=3, linestyle='--',
              label=f'Wrong {angles[angle_j]:.0f}° (Physics)', alpha=0.8)
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_ylabel('QK Expert Score', fontsize=12, fontweight='bold')
    ax.set_title(f'2B: QK Selection in Angle Space (N={n_cases} cases)',
                fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(angles[0], angles[-1])

    # Panel 2C: Statistics Summary
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    # Compute statistics
    physics_score_true = g_energy[cases, angle_i].mean()
    physics_score_wrong = g_energy[cases, angle_j].mean()
    qk_score_true = scores_expert[cases, angle_i].mean()
    qk_score_wrong = scores_expert[cases, angle_j].mean()

    summary_text = f"""
STATISTICS SUMMARY

Confused Pair:
  True:  {angles[angle_i]:.0f}°
  Wrong: {angles[angle_j]:.0f}° (Physics)

Cases Analyzed: {n_cases}
  (QK-corrected samples)

Frequency Correlation:
  Mean: {freq_sim.mean():.3f}
  Max:  {freq_sim.max():.3f}
  Min:  {freq_sim.min():.3f}
  Std:  {freq_sim.std():.3f}

Physics Scores (avg):
  True:  {physics_score_true:.3f}
  Wrong: {physics_score_wrong:.3f}
  → Physics chose WRONG!

QK Scores (avg):
  True:  {qk_score_true:.3f}
  Wrong: {qk_score_wrong:.3f}
  → QK chose CORRECT!

Atoms Selected (QK):
  True:  {(qk_scores_true > 0).sum()}/8 positive
  Wrong: {(qk_scores_wrong < 0).sum()}/8 negative
"""

    ax.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=1))

    # Overall title
    fig.suptitle(f'Detailed Analysis: {angles[angle_i]:.0f}° vs {angles[angle_j]:.0f}° (N={n_cases} QK-corrected cases)',
                fontsize=16, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path.name}")


def main():
    run_dir = "results/omp_transformer_speech260_trainval_split_full_20251115_082341"

    print("="*80)
    print("FREQUENCY-ANGLE JOINT ANALYSIS")
    print("(Analogous to SVD/Green's Function Decomposition)")
    print("="*80)

    # Load data
    routing_data, dict_data = load_data(run_dir)

    # Phase 1: Frequency-Angle Joint Confusion
    confusion_result = compute_frequency_angle_confusion(
        dict_data['D'], routing_data, dict_data
    )

    # Phase 2: Modal Decomposition (SVD)
    svd_result = modal_decomposition_svd(dict_data['H'], dict_data)

    # Phase 3: QK Joint Space Selection
    joint_patterns = analyze_qk_joint_space_selection(
        routing_data, dict_data, confusion_result
    )

    # Phase 4: Visualizations (Global Overview)
    visualize_joint_analysis(
        confusion_result, svd_result, joint_patterns, dict_data, routing_data, run_dir
    )

    # Phase 5: Detailed Per-Pair Analysis
    print("\n=== Phase 5: Detailed Per-Pair Analysis ===")
    print("Generating 6-panel detailed figures for top 3 confused pairs...")

    from pathlib import Path
    output_dir = Path(run_dir)

    # Get top 3 confused pairs
    top_3_pairs = confusion_result['top_confused'][:3]

    for angle_i, angle_j, count in top_3_pairs:
        angle_i_deg = dict_data['angles'][angle_i]
        angle_j_deg = dict_data['angles'][angle_j]

        output_filename = f"detailed_pair_{int(angle_i_deg)}deg_vs_{int(angle_j_deg)}deg.png"
        output_path = output_dir / output_filename

        visualize_per_pair_detailed(
            angle_i, angle_j,
            dict_data['D'],
            routing_data,
            dict_data,
            confusion_result,
            output_path
        )

    print("\n" + "="*80)
    print("✓ Analysis Complete!")
    print("="*80)
    print(f"\nGenerated visualizations in: {run_dir}/")
    print("\nGlobal Overview:")
    print("  1. angle_confusion_matrices_global.png - Confusion matrices, similarity & QK selection")
    print("  2. svd_modal_decomposition.png - SVD modal analysis")
    print("  3. frequency_angle_joint_patterns.png - Joint frequency-angle patterns (legacy)")
    print("\nDetailed Per-Pair Analysis (6-panel figures):")
    for i, (angle_i, angle_j, count) in enumerate(top_3_pairs, 1):
        angle_i_deg = int(dict_data['angles'][angle_i])
        angle_j_deg = int(dict_data['angles'][angle_j])
        print(f"  {i+3}. detailed_pair_{angle_i_deg}deg_vs_{angle_j_deg}deg.png - {angle_i_deg}° vs {angle_j_deg}° ({int(count)} errors)")


if __name__ == '__main__':
    main()
