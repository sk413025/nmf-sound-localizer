#!/usr/bin/env python
"""
Debug atom masking behavior - inspect step-by-step atom selection.
"""
import torch
import numpy as np
import importlib.util

# Load omp-transformer-ldv.py module
spec = importlib.util.spec_from_file_location("omp_transformer_ldv", "scripts/omp-transformer-ldv.py")
omp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omp_module)

load_raw_ldv_matrices = omp_module.load_raw_ldv_matrices
reduce_atoms_kmeans = omp_module.reduce_atoms_kmeans
build_dictionary = omp_module.build_dictionary
FullTransformerRoutedSoftOMP = omp_module.FullTransformerRoutedSoftOMP

from doa_rl.data import DoADataset

def debug_masking():
    """Debug atom selection and masking behavior."""

    device = 'cpu'

    # Load matrices
    print("Loading H and W matrices...")
    H_path = "/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
    W_path = "doa_normalized_config_c_corrected/models/usm.pth"
    H_raw, W_raw, angles = load_raw_ldv_matrices(H_path, W_path, device='cpu')

    # Reduce atoms with K-means
    print("Reducing atoms with K-means...")
    W_reduced, _, _ = reduce_atoms_kmeans(W_raw, n_clusters=8)

    # Build dictionary
    D, idx2angle = build_dictionary(H_raw, W_reduced, angles)

    F, E = H_raw.shape
    _, M = W_reduced.shape
    P = E * M
    print(f"Dictionary: F={F}, E={E} experts, M={M} atoms/expert, P={P} total\n")

    # Load model
    model_path = "results/qk_phase2_hard_gumbel_masking_20251102_231358/model_best.pth"
    print(f"Loading model from: {model_path}")
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=128,
        nhead=2, nlayers=1,
        steps=6,  # Test with 6 steps to see masking behavior
        routing_mode='qk',
        score_norm_mode='std',
        expert_agg='l2'
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set configuration flags
    model.no_type_bias = True
    model.score_center_atoms = True
    model.score_center_expert = True
    model.use_hard_gumbel = True

    model.eval()

    # Load ONE test sample
    print("\nLoading test sample...")
    dataset_root = "/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
    dataset = DoADataset(dataset_root, angles)

    data = dataset[0]
    Y = data['Y']  # (F, N) STFT data
    y = Y.mean(dim=1)  # (F,)
    y = y / (y.norm() + 1e-12)
    y = y.to(device)

    print(f"\n{'='*80}")
    print(f"Debugging Step-by-Step Atom Selection")
    print(f"{'='*80}\n")

    # Manually step through the forward pass to inspect masking
    # We'll need to modify the forward pass to add debug prints
    # For now, let's just see what support_indices looks like

    with torch.no_grad():
        x_pred, res_curve = model.forward(y, D, train_mode=False)

    if hasattr(model, 'last_outputs') and 'support_indices' in model.last_outputs:
        S = model.last_outputs['support_indices']

        print(f"Selected atoms (global indices): {S}")
        print(f"Total atoms selected: {len(S)}")
        print(f"Unique atoms: {len(set(S))}")
        print(f"Diversity: {len(set(S))/len(S)*100:.1f}%\n")

        # Analyze by expert
        print("Breakdown by expert:")
        for e in range(E):
            expert_atoms = [idx for idx in S if idx // M == e]
            if expert_atoms:
                local_atoms = [idx % M for idx in expert_atoms]
                print(f"  Expert {e:2d} (angle {angles[e]:5.1f}°): selected {len(expert_atoms)} atoms, local indices: {local_atoms}")

        # Check for repetitions
        print("\nRepetition analysis:")
        from collections import Counter
        counts = Counter(S)
        repeated = [(idx, count) for idx, count in counts.items() if count > 1]
        if repeated:
            print("Repeated atoms:")
            for idx, count in repeated:
                e = idx // M
                m = idx % M
                print(f"  Atom {idx} (expert {e}, local atom {m}): selected {count} times")
        else:
            print("No repeated atoms (perfect diversity!)")

        # Show selection order
        print(f"\nSelection order (steps with top_e={model.top_e}, L={model.L}):")
        step_size = model.top_e * model.L
        for step in range(len(S) // step_size):
            start = step * step_size
            end = start + step_size
            step_atoms = S[start:end]
            step_experts = [idx // M for idx in step_atoms]
            step_local = [idx % M for idx in step_atoms]
            print(f"  Step {step}: atoms {step_atoms}")
            print(f"          experts {step_experts}, local atoms {step_local}")

if __name__ == '__main__':
    debug_masking()
