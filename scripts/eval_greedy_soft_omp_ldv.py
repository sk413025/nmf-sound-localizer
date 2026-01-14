#!/usr/bin/env python3
"""
Evaluate Greedy Routed Soft-OMP (TrainableRoutedSoftOMP train_mode=False) on real LDV data.

This script validates that greedy correlation-based selection (g=D^T@r) achieves
high accuracy (~83.8% like b573aa6) on 37-angle LDV dataset, confirming that:
1. The data and dictionary are correct
2. Greedy works when Transformer fails
3. High coherence (μ=0.9995) doesn't block greedy selection

Purpose: Option 1 verification to establish baseline before improving Transformer.
"""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import TrainableRoutedSoftOMP
from doa_rl.omp.soft_omp import TrainableRoutedSoftOMP


# ============================================================================
# PART 1: Data Loading and Preprocessing (Same as omp-transformer-ldv.py)
# ============================================================================

def load_raw_ldv_matrices(h_path: str, w_path: str) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Load H and W matrices for LDV data."""
    print(f"Loading H from: {h_path}")
    h_data = torch.load(h_path, weights_only=False)
    H = h_data['H']  # (F, E) where E=37 angles
    print(f"  H shape: {H.shape}, range: [{H.min():.3f}, {H.max():.3f}]")
    
    print(f"Loading W from: {w_path}")
    w_data = torch.load(w_path, weights_only=False)
    W = w_data['W'] if isinstance(w_data, dict) else w_data
    print(f"  W shape: {W.shape}, range: [{W.min():.3f}, {W.max():.3f}]")
    
    metadata = {
        'H_shape': list(H.shape),
        'W_shape': list(W.shape),
        'H_range': [float(H.min()), float(H.max())],
        'W_range': [float(W.min()), float(W.max())],
    }
    
    return H, W, metadata


def load_ldv_samples(dataset_root: str, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """Load LDV audio samples and convert to STFT observations."""
    dataset_root = Path(dataset_root)
    Y_list = []
    labels = []
    metadata_list = []
    
    # STFT parameters (must match H matrix metadata)
    n_fft = 2048
    hop_length = 512
    fs = 16000
    freq_min, freq_max = 300, 3000
    
    # Get frequency bins (rfftfreq returns Hz when d=1/fs)
    freq_bins = torch.fft.rfftfreq(n_fft, d=1/fs)  # Already in Hz
    freq_mask = (freq_bins >= freq_min) & (freq_bins <= freq_max)
    F_expected = freq_mask.sum().item()
    F_h = H.shape[0]
    
    if F_expected != F_h:
        raise ValueError(f"STFT freq bins ({F_expected}) != H.F ({F_h}). Check n_fft/fs/freq_min/freq_max.")
    
    print(f"\nLoading dataset from: {dataset_root}")
    print(f"STFT: n_fft={n_fft}, hop_length={hop_length}, fs={fs}")
    print(f"Band: [{freq_min}, {freq_max}]Hz -> {F_h} freq bins")
    
    # Load samples from all angle_XX folders
    angle_folders = sorted([d for d in dataset_root.iterdir() if d.is_dir() and d.name.startswith('angle_')])
    
    for angle_idx, angle_dir in enumerate(angle_folders):
        angle_deg = int(angle_dir.name.split('_')[1]) * 5  # angle_00 -> 0°, angle_01 -> 5°, etc.
        npy_files = sorted(angle_dir.glob('*.npy'))
        
        for clip_file in npy_files:
            # Load audio
            audio = np.load(clip_file)
            audio_t = torch.from_numpy(audio).float()
            
            # STFT
            stft = torch.stft(audio_t, n_fft=n_fft, hop_length=hop_length, 
                            return_complex=True, window=torch.hann_window(n_fft))
            Y_full = stft.abs().mean(dim=1)  # Average over time frames
            Y = Y_full[freq_mask]
            
            Y_list.append(Y)
            labels.append(angle_idx)
            metadata_list.append({
                'angle_deg': angle_deg,
                'angle_idx': angle_idx,
                'clip_file': str(clip_file),
            })
    
    Y_samples = torch.stack(Y_list, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    
    print(f"Loaded {len(Y_list)} samples from {len(angle_folders)} angles")
    print(f"Y_samples shape: {Y_samples.shape}, labels shape: {labels.shape}")
    
    return Y_samples, labels, metadata_list


def reduce_atoms_kmeans(W: torch.Tensor, n_atoms: int, seed: int = 42) -> Tuple[torch.Tensor, np.ndarray, float]:
    """Reduce W atoms using K-means clustering."""
    M_original = W.shape[1]
    if n_atoms >= M_original:
        print(f"n_atoms ({n_atoms}) >= M ({M_original}), skipping K-means")
        return W, np.arange(M_original), 0.0
    
    print(f"\nK-means clustering: {M_original} -> {n_atoms} atoms")
    W_np = W.T.cpu().numpy()  # (M, F)
    
    kmeans = KMeans(n_clusters=n_atoms, random_state=seed, n_init=10)
    kmeans.fit(W_np)
    
    W_reduced_np = kmeans.cluster_centers_.T  # (F, n_atoms)
    W_reduced = torch.from_numpy(W_reduced_np).float()
    
    # Compute reconstruction error
    reconstructed = kmeans.predict(W_np)
    centers = kmeans.cluster_centers_[reconstructed]
    reconstruction_error = np.linalg.norm(W_np - centers) / np.linalg.norm(W_np) * 100
    
    print(f"  Reconstruction error: {reconstruction_error:.2f}%")
    print(f"  Cluster sizes: {np.bincount(kmeans.labels_)}")
    
    return W_reduced, kmeans.labels_, reconstruction_error


def reduce_frequency_pca(H: torch.Tensor, W: torch.Tensor, n_components: int, 
                        seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor, object]:
    """Reduce frequency dimension using PCA."""
    F_original = H.shape[0]
    
    # Adjust n_components if needed
    n_samples = H.shape[1] + W.shape[1]
    max_components = min(F_original, n_samples)
    if n_components > max_components:
        print(f"WARNING: n_components ({n_components}) > max ({max_components}), adjusting to {max_components}")
        n_components = max_components
    
    print(f"\nPCA reduction: F={F_original} -> {n_components}")
    
    # Combine H and W for PCA fitting
    combined = torch.cat([H, W], dim=1).T.cpu().numpy()  # (E+M, F)
    
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(combined)
    
    # Transform H and W
    H_pca_np = pca.transform(H.T.cpu().numpy()).T  # (n_components, E)
    W_pca_np = pca.transform(W.T.cpu().numpy()).T  # (n_components, M)
    
    H_pca = torch.from_numpy(H_pca_np).float()
    W_pca = torch.from_numpy(W_pca_np).float()
    
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  Variance explained: {variance_explained:.2f}%")
    print(f"  H_pca shape: {H_pca.shape}, W_pca shape: {W_pca.shape}")
    
    return H_pca, W_pca, pca


def build_dictionary(H: torch.Tensor, W: torch.Tensor) -> Tuple[torch.Tensor, List, List, float, float]:
    """Build dictionary D where atoms are H[:,e] ⊙ W[:,k]."""
    F, E = H.shape
    _, M = W.shape
    P = E * M
    
    D = torch.zeros((F, P), dtype=H.dtype)
    idx_to_expert = []
    idx_to_atom = []
    
    j = 0
    for e in range(E):
        h = H[:, e]
        for k in range(M):
            atom = h * W[:, k]
            norm = torch.norm(atom) + 1e-12
            D[:, j] = atom / norm
            idx_to_expert.append(e)
            idx_to_atom.append(k)
            j += 1
    
    # Compute mutual coherence
    G = (D.T @ D).cpu().numpy()
    np.fill_diagonal(G, 0)
    mu_max = float(np.max(np.abs(G)))
    mu_mean = float(np.mean(np.abs(G)))
    
    print(f"\nDictionary: D shape {D.shape}")
    print(f"  Mutual coherence: μ_max={mu_max:.4f}, μ_mean={mu_mean:.4f}")
    
    return D, idx_to_expert, idx_to_atom, mu_max, mu_mean


# ============================================================================
# PART 2: Evaluation
# ============================================================================

@torch.no_grad()
def evaluate_greedy(model: TrainableRoutedSoftOMP, D: torch.Tensor, 
                   Y_samples: torch.Tensor, labels: torch.Tensor,
                   idx_to_expert: List[int], device='cpu') -> Dict:
    """Evaluate greedy Soft-OMP (train_mode=False)."""
    model.eval()
    D = D.to(device)
    Y_samples = Y_samples.to(device)
    labels = labels.to(device)
    
    N = Y_samples.size(0)
    predictions = []
    residual_curves = []
    
    print(f"\nEvaluating {N} samples with greedy selection...")
    
    for i in range(N):
        y = Y_samples[i]
        
        # Greedy selection (train_mode=False)
        x_hat, res_curve = model(y, D, train_mode=False)
        
        # Vote by expert (angle)
        votes = torch.zeros(model.E, device=device)
        for j, coef in enumerate(x_hat):
            expert_idx = idx_to_expert[j]
            votes[expert_idx] += coef.abs()
        
        pred = votes.argmax().item()
        predictions.append(pred)
        residual_curves.append(res_curve)
    
    predictions = torch.tensor(predictions, dtype=torch.long, device=device)
    correct = (predictions == labels).sum().item()
    accuracy = correct / N * 100
    
    # Per-angle accuracy
    E = model.E
    per_angle_acc = []
    for angle_idx in range(E):
        mask = (labels == angle_idx)
        if mask.sum() > 0:
            angle_correct = (predictions[mask] == labels[mask]).sum().item()
            angle_total = mask.sum().item()
            angle_acc = angle_correct / angle_total * 100
        else:
            angle_acc = 0.0
        per_angle_acc.append(angle_acc)
    
    results = {
        'overall_accuracy': accuracy,
        'correct': correct,
        'total': N,
        'per_angle_accuracy': per_angle_acc,
        'predictions': predictions.cpu().numpy().tolist(),
        'residual_curves': residual_curves,
    }
    
    print(f"\nGreedy Evaluation Results:")
    print(f"  Overall Accuracy: {accuracy:.1f}% ({correct}/{N})")
    print(f"  Per-angle accuracy range: [{min(per_angle_acc):.1f}%, {max(per_angle_acc):.1f}%]")
    
    # Show problematic angles
    problematic = [i for i, acc in enumerate(per_angle_acc) if acc < 50]
    if problematic:
        print(f"  Problematic angles (<50%): {problematic}")
    
    return results


# ============================================================================
# PART 3: Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate Greedy Routed Soft-OMP on LDV data')
    
    # Data paths
    parser.add_argument('--h_path', type=str, required=True, help='Path to H matrix')
    parser.add_argument('--w_path', type=str, required=True, help='Path to W matrix')
    parser.add_argument('--dataset_root', type=str, required=True, help='Root directory of LDV dataset')
    
    # Preprocessing
    parser.add_argument('--n_atoms', type=int, default=8, help='Number of atoms after K-means (0=skip)')
    parser.add_argument('--pca_dim', type=int, default=45, help='PCA dimension (0=skip)')
    
    # Model parameters
    parser.add_argument('--steps', type=int, default=6, help='Number of OMP steps')
    parser.add_argument('--top_e', type=int, default=2, help='Top-K experts for hard routing')
    parser.add_argument('--L', type=int, default=2, help='Top-L atoms per expert')
    
    # Output
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/mps/cuda)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/greedy_soft_omp_ldv_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {args.device}")
    
    # Load data
    H, W, meta_hw = load_raw_ldv_matrices(args.h_path, args.w_path)
    Y_samples, labels, meta_samples = load_ldv_samples(args.dataset_root, H)
    
    # Preprocessing
    if args.n_atoms > 0:
        W, kmeans_labels, recon_error = reduce_atoms_kmeans(W, args.n_atoms)
        meta_hw['kmeans_reconstruction_error'] = float(recon_error)
    
    if args.pca_dim > 0:
        H, W, pca_model = reduce_frequency_pca(H, W, args.pca_dim)
        # Transform Y_samples
        Y_pca_np = pca_model.transform(Y_samples.cpu().numpy())
        Y_samples = torch.from_numpy(Y_pca_np).float()
        print(f"Transformed Y_samples to PCA space: {Y_samples.shape}")
    
    # Build dictionary
    D, idx_to_expert, idx_to_atom, mu_max, mu_mean = build_dictionary(H, W)
    
    F, P = D.shape
    E = H.shape[1]
    M = W.shape[1]
    
    print(f"\nModel configuration:")
    print(f"  F={F}, E={E}, M={M}, P={P}")
    print(f"  steps={args.steps}, top_e={args.top_e}, L={args.L}")
    
    # Create model (no training, just for evaluation structure)
    model = TrainableRoutedSoftOMP(
        F=F, E=E, M=M,
        steps=args.steps,
        top_e=args.top_e,
        L=args.L,
    )
    
    print(f"\nModel: TrainableRoutedSoftOMP")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Evaluate with greedy selection
    results = evaluate_greedy(model, D, Y_samples, labels, idx_to_expert, device=args.device)
    
    # Save results
    results_path = os.path.join(args.out_dir, 'greedy_results.json')
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_save = {
            'overall_accuracy': results['overall_accuracy'],
            'correct': results['correct'],
            'total': results['total'],
            'per_angle_accuracy': results['per_angle_accuracy'],
        }
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save metadata
    meta_path = os.path.join(args.out_dir, 'metadata.json')
    metadata = {
        'args': vars(args),
        'hw_metadata': meta_hw,
        'dictionary': {
            'F': F, 'E': E, 'M': M, 'P': P,
            'mu_max': mu_max,
            'mu_mean': mu_mean,
        },
        'num_samples': len(labels),
        'num_angles': E,
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Greedy Soft-OMP Evaluation Complete!")
    print(f"Overall Accuracy: {results['overall_accuracy']:.1f}%")
    print(f"Expected (b573aa6 baseline): ~83.8%")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
