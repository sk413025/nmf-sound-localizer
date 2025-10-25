#!/usr/bin/env python3
"""
Evaluate Greedy Routed Soft-OMP using DoADataset (matching b573aa6 exactly).

This script uses the SAME data loading as b573aa6 to ensure STFT processing is identical.
Key difference from eval_greedy_soft_omp_ldv.py: Uses DoADataset instead of custom torch.stft.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.omp.soft_omp import TrainableRoutedSoftOMP, build_dictionary


def main():
    parser = argparse.ArgumentParser(description='Evaluate Greedy Soft-OMP with DoADataset')
    
    # Data paths
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of LDV dataset')
    parser.add_argument('--h_path', type=str, required=True, help='Path to H matrix')
    parser.add_argument('--w_path', type=str, required=True, help='Path to W matrix')
    
    # STFT parameters (must match b573aa6)
    parser.add_argument('--freq_min', type=float, default=300.0, help='Min frequency (Hz)')
    parser.add_argument('--freq_max', type=float, default=3000.0, help='Max frequency (Hz)')
    parser.add_argument('--fs', type=int, default=16000, help='Sample rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT size')
    
    # Model parameters
    parser.add_argument('--steps', type=int, default=6, help='Number of OMP steps')
    parser.add_argument('--top_e', type=int, default=2, help='Top-K experts')
    parser.add_argument('--L', type=int, default=2, help='Top-L atoms per expert')
    
    # Evaluation
    parser.add_argument('--n_eval', type=int, default=200, help='Max samples to evaluate')
    
    # Output
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/mps/cuda)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/greedy_doadataset_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {device}")
    
    # Load H matrix
    print(f"\nLoading H from: {args.h_path}")
    h_data = torch.load(args.h_path, weights_only=False)
    H = h_data['H'].float().to(device)  # (F, E)
    print(f"  H shape: {H.shape}")
    
    # Load W matrix
    print(f"Loading W from: {args.w_path}")
    w_data = torch.load(args.w_path, weights_only=False)
    if isinstance(w_data, dict) and 'W' in w_data:
        W = w_data['W'].float()
    elif isinstance(w_data, torch.Tensor):
        W = w_data.float()
    else:
        raise ValueError(f"Unrecognized W format in {args.w_path}")
    W = W.to(device)
    print(f"  W shape: {W.shape}")
    
    F, E = H.shape
    _, M = W.shape
    print(f"\nDimensions: F={F}, E={E}, M={M}, P={E*M}")
    
    # Build dictionary (using doa_rl.omp.soft_omp.build_dictionary)
    print("\nBuilding dictionary...")
    D, idx2 = build_dictionary(W, H)
    D = D.to(device)
    
    # Compute coherence
    G = (D.T @ D).cpu().numpy()
    np.fill_diagonal(G, 0)
    mu_max = float(np.max(np.abs(G)))
    mu_mean = float(np.mean(np.abs(G)))
    print(f"  D shape: {D.shape}")
    print(f"  Mutual coherence: μ_max={mu_max:.4f}, μ_mean={mu_mean:.4f}")
    
    # Create DoADataset (SAME as b573aa6)
    print(f"\nCreating DoADataset from: {args.dataset_root}")
    angles = list(range(0, 181, 5))  # 0°, 5°, ..., 180° (37 angles)
    dataset = DoADataset(
        root=args.dataset_root,
        angles=angles,
        fs=args.fs,
        n_fft=args.n_fft,
        window='hann',
        freq_min=args.freq_min,
        freq_max=args.freq_max
    )
    dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
    print(f"  Dataset size: {len(dataset)} samples")
    
    # Create model (no training needed for greedy)
    print(f"\nCreating TrainableRoutedSoftOMP...")
    model = TrainableRoutedSoftOMP(
        F=F, E=E, M=M,
        steps=args.steps,
        top_e=args.top_e,
        L=args.L,
    ).to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Angle mapping
    angle_to_idx = {int(a): i for i, a in enumerate(angles)}
    
    # Evaluation loop
    print(f"\n{'='*60}")
    print(f"Evaluating with greedy selection (train_mode=False)...")
    print(f"{'='*60}")
    
    n_samples = 0
    n_correct = 0
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in dataloader:
            if n_samples >= args.n_eval:
                break
            
            Y = batch['Y'][0].to(device)  # (F, N) - STFT frames
            if Y.shape[0] != F:
                raise ValueError(f"STFT grid mismatch: Y.F={Y.shape[0]} vs H/W.F={F}")
            
            # Average over time (same as b573aa6)
            y = Y.mean(dim=1)  # (F,)
            
            # Greedy selection
            x_hat, r_curve = model(y, D, train_mode=False)
            
            # Vote by expert (sum atom coefficients per expert)
            A_group = x_hat.view(E, M).abs().sum(dim=1)
            pred_idx = int(torch.argmax(A_group).item())
            
            # Ground truth
            gt_angle = int(batch['angle_deg'][0].item())
            gt_idx = angle_to_idx[gt_angle]
            
            # Check correctness
            correct = int(pred_idx == gt_idx)
            n_correct += correct
            n_samples += 1
            
            predictions.append(pred_idx)
            ground_truths.append(gt_idx)
            
            if n_samples % 20 == 0:
                print(f"  Processed {n_samples}/{min(args.n_eval, len(dataset))} samples, acc={n_correct/n_samples*100:.1f}%")
    
    # Compute overall accuracy
    overall_acc = n_correct / n_samples * 100
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_acc:.1f}% ({n_correct}/{n_samples})")
    print(f"Expected (b573aa6): 83.8%")
    
    # Per-angle accuracy
    per_angle_acc = []
    for angle_idx in range(E):
        mask = np.array(ground_truths) == angle_idx
        if mask.sum() > 0:
            angle_correct = np.sum(np.array(predictions)[mask] == angle_idx)
            angle_total = mask.sum()
            angle_acc = angle_correct / angle_total * 100
        else:
            angle_acc = 0.0
        per_angle_acc.append(angle_acc)
    
    # Show problematic angles
    problematic = [i*5 for i, acc in enumerate(per_angle_acc) if acc < 50]
    if problematic:
        print(f"\nProblematic angles (<50% acc): {problematic}")
    
    perfect = [i*5 for i, acc in enumerate(per_angle_acc) if acc == 100]
    if perfect:
        print(f"Perfect angles (100% acc): {perfect}")
    
    # Save results
    results = {
        'overall_accuracy': float(overall_acc),
        'correct': int(n_correct),
        'total': int(n_samples),
        'per_angle_accuracy': per_angle_acc,
        'predictions': predictions,
        'ground_truths': ground_truths,
    }
    
    results_path = os.path.join(args.out_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save metadata
    metadata = {
        'args': vars(args),
        'dimensions': {'F': F, 'E': E, 'M': M, 'P': E*M},
        'coherence': {'mu_max': mu_max, 'mu_mean': mu_mean},
        'dataset_size': len(dataset),
    }
    
    meta_path = os.path.join(args.out_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"{'='*60}")
    print(f"SUCCESS: Greedy evaluation complete!")
    if abs(overall_acc - 83.8) < 5:
        print(f"✅ Accuracy matches b573aa6 baseline (~83.8%)")
    else:
        print(f"❌ Accuracy differs from b573aa6 baseline (expected ~83.8%)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
