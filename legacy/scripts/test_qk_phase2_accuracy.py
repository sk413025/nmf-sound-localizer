#!/usr/bin/env python
"""
Test classification accuracy of Phase 2 QK model (hard Gumbel + masking) during inference.
"""
import torch
import numpy as np
import importlib.util
from collections import defaultdict

# Load omp-transformer-ldv.py module
spec = importlib.util.spec_from_file_location("omp_transformer_ldv", "scripts/omp-transformer-ldv.py")
omp_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(omp_module)

load_raw_ldv_matrices = omp_module.load_raw_ldv_matrices
reduce_atoms_kmeans = omp_module.reduce_atoms_kmeans
build_dictionary = omp_module.build_dictionary
FullTransformerRoutedSoftOMP = omp_module.FullTransformerRoutedSoftOMP

from doa_rl.data import DoADataset

def test_accuracy(model_path):
    """Test classification accuracy during inference."""

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
    print(f"Dictionary: F={F}, E={E}, M={M}, P={P}")

    # Load model (steps=2, same as training)
    # IMPORTANT: Must match ALL training parameters!
    print(f"\nLoading model from: {model_path}")
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=128,
        nhead=2, nlayers=1,
        steps=2,  # Use 2 steps (same as training!)
        routing_mode='qk',
        score_norm_mode='std',
        expert_agg='l2'
    ).to(device)

    # Set training configuration flags (CRITICAL: Must match training!)
    model.no_type_bias = True
    model.score_center_atoms = True
    model.score_center_expert = True

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Enable hard Gumbel mode (same as training)
    model.use_hard_gumbel = True
    print(f"ENABLED use_hard_gumbel (using hard Gumbel routing as in training)")
    print(f"Model steps={model.steps}, top_e={model.top_e}, L={model.L}")

    model.eval()

    # Load test dataset
    print("\nLoading test dataset...")
    dataset_root = "/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
    dataset = DoADataset(dataset_root, angles)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Angles: {len(angles)} total")

    # Test classification accuracy
    correct = 0
    total = 0
    angle_correct = defaultdict(int)
    angle_total = defaultdict(int)

    print(f"\n{'='*80}")
    print(f"Testing Classification Accuracy (Inference Mode)")
    print(f"{'='*80}\n")

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            Y = data['Y']  # (F, N) STFT data
            gt_angle = data['angle_deg']

            # Time-average to get single spectrum
            y = Y.mean(dim=1)  # (F,)
            y = y / (y.norm() + 1e-12)
            y = y.to(device)

            # Run inference
            x_pred, res_curve = model.forward(y, D, train_mode=False)

            # Predict angle from routing scores (CRITICAL: same method as training evaluation)
            # Use scores_expert from forward pass, not sparse coefficients
            if model.last_diag is not None and 'scores_expert' in model.last_diag:
                scores = model.last_diag['scores_expert'].cpu()
            else:
                # Fallback to g-based scoring
                g_vec = (D.T @ y)
                scores = g_vec.abs().view(E, M).sum(dim=1)

            # Predict expert index (which corresponds to angle index)
            pred_idx = scores.argmax().item()
            pred_angle = angles[pred_idx]

            # Check accuracy
            is_correct = (pred_angle == gt_angle)
            if is_correct:
                correct += 1
                angle_correct[gt_angle] += 1
            total += 1
            angle_total[gt_angle] += 1

            if i < 5:  # Print first 5 samples
                print(f"Sample {i}: GT={gt_angle:.1f}°, Pred={pred_angle:.1f}°, {'✓' if is_correct else '✗'}")

    # Print results
    accuracy = 100.0 * correct / total
    print(f"\n{'='*80}")
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"{'='*80}")

    # Print per-angle accuracy
    print(f"\nPer-Angle Accuracy:")
    failed_angles = []
    for angle in sorted(angles):
        if angle_total[angle] > 0:
            acc = 100.0 * angle_correct[angle] / angle_total[angle]
            status = '✓' if acc == 100.0 else '✗'
            print(f"  {angle:5.1f}°: {angle_correct[angle]}/{angle_total[angle]} = {acc:5.1f}% {status}")
            if acc < 100.0:
                failed_angles.append(angle)

    if failed_angles:
        print(f"\nFailed angles: {failed_angles}")

    print(f"\nExpected (training): 97.3% (36/37 angles, only 95° failed)")

    if accuracy >= 97.0:
        print("✅ SUCCESS: Inference accuracy matches training!")
    elif accuracy >= 90.0:
        print("⚠️  PARTIAL: Good accuracy but below training")
    else:
        print("❌ FAILURE: Inference accuracy much lower than training!")

if __name__ == '__main__':
    model_path = "results/qk_phase2_hard_gumbel_masking_20251102_234521/model_best.pth"
    test_accuracy(model_path)
