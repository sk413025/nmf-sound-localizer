#!/usr/bin/env python
"""
Test atom diversity in trajectory generation with hard Gumbel-trained QK model.
"""
import torch
import numpy as np
import sys
import importlib.util

# Load omp-transformer-ldv.py module (dash in filename requires special handling)
spec = importlib.util.spec_from_file_location("omp_transformer_ldv", "scripts/omp-transformer-ldv.py")
omp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omp_module)

load_raw_ldv_matrices = omp_module.load_raw_ldv_matrices
reduce_atoms_kmeans = omp_module.reduce_atoms_kmeans
build_dictionary = omp_module.build_dictionary
FullTransformerRoutedSoftOMP = omp_module.FullTransformerRoutedSoftOMP

from doa_rl.data import DoADataset

def test_atom_diversity(model_path, num_test_samples=5):
    """Test atom diversity using the hard Gumbel-trained model."""

    device = 'cpu'

    # Load matrices
    print("Loading H and W matrices...")
    H_path = "/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
    W_path = "doa_normalized_config_c_corrected/models/usm.pth"
    H_raw, W_raw, angles = load_raw_ldv_matrices(H_path, W_path, device='cpu')

    # Reduce atoms with K-means
    print("Reducing atoms with K-means...")
    W_reduced, _, _ = reduce_atoms_kmeans(W_raw, n_clusters=8)

    # Build dictionary using the standard function
    D, idx2angle = build_dictionary(H_raw, W_reduced, angles)

    F, E = H_raw.shape
    _, M = W_reduced.shape
    P = E * M
    print(f"Dictionary: F={F}, E={E}, M={M}, P={P}")

    # Load model
    print(f"\nLoading model from: {model_path}")
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=128,
        nhead=2, nlayers=1,
        steps=6,  # Use 6 steps for trajectory generation
        routing_mode='qk',
        score_norm_mode='std',
        expert_agg='l2'
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set training configuration flags (CRITICAL: Must match training!)
    model.no_type_bias = True
    model.score_center_atoms = True
    model.score_center_expert = True
    model.use_hard_gumbel = True
    print("Enabled all training flags (no_type_bias, score_center_*, use_hard_gumbel)")

    model.eval()

    # Load test dataset
    print("\nLoading test samples...")
    dataset_root = "/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
    dataset = DoADataset(dataset_root, angles)

    # Test on first num_test_samples
    print(f"\n{'='*80}")
    print(f"Testing Atom Diversity on {num_test_samples} samples")
    print(f"{'='*80}\n")

    atom_diversity_results = []

    for i in range(min(num_test_samples, len(dataset))):
        data = dataset[i]
        Y = data['Y']  # (F, N) STFT data
        y = Y.mean(dim=1)  # Time-average to get single spectrum (F,)
        y = y / (y.norm() + 1e-12)  # Normalize
        y = y.to(device)

        # Run model forward pass with train_mode=False (inference)
        with torch.no_grad():
            x_pred, res_curve = model.forward(y, D, train_mode=False)

        # Extract selected atoms from model's last_outputs
        if hasattr(model, 'last_outputs') and 'support_indices' in model.last_outputs:
            S = model.last_outputs['support_indices']  # List of selected atom indices
            selected_atoms = [int(j) for j in S]
            unique_atoms = len(set(selected_atoms))
            diversity = unique_atoms / len(selected_atoms) * 100

            print(f"Sample {i}: Atoms={selected_atoms}, Unique={unique_atoms}/{len(selected_atoms)} ({diversity:.1f}%)")
            atom_diversity_results.append(diversity)
        else:
            print(f"Sample {i}: Model did not save support_indices!")

    # Summary
    if atom_diversity_results:
        avg_diversity = np.mean(atom_diversity_results)
        print(f"\n{'='*80}")
        print(f"Average Atom Diversity: {avg_diversity:.1f}%")
        print(f"Expected (hard Gumbel training): >80%")
        print(f"Baseline (soft training): ~17%")
        print(f"{'='*80}")

        if avg_diversity > 80:
            print("✅ SUCCESS: Hard Gumbel training improved atom diversity!")
        elif avg_diversity > 50:
            print("⚠️  PARTIAL: Some improvement, but not as much as expected")
        else:
            print("❌ FAILURE: Hard Gumbel training did not improve atom diversity")
    else:
        print("\n❌ ERROR: Could not extract atom diversity from model outputs")
        print("Model may need modification to save support_indices during inference")

if __name__ == '__main__':
    model_path = "results/qk_phase2_hard_gumbel_masking_20251102_234521/model_best.pth"
    test_atom_diversity(model_path, num_test_samples=10)
