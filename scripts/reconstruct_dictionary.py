#!/usr/bin/env python3
"""
Reconstruct Dictionary D from H matrix and W atoms

This script loads the H matrix and W atoms used in the experiment
and reconstructs the dictionary D = H ⊙ W (outer product).
"""
import numpy as np
import torch
from pathlib import Path
import json

def reduce_atoms_kmeans(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42):
    """
    Reduce W atoms from 50 → n_clusters using K-means clustering.
    (Copied from omp-transformer-ldv.py for consistency)
    """
    print(f"\n=== Atom Reduction via K-means ===")
    print(f"Input: W shape {W.shape}")
    print(f"Target: {n_clusters} clusters")

    # Transpose for clustering: cluster atoms (rows), not frequencies
    X = W.cpu().numpy().T  # (M_full, F)
    rng = np.random.RandomState(random_state)
    M_full, Fdim = X.shape

    # k-means++ initialization
    centers = []
    idx0 = int(rng.randint(M_full))
    centers.append(X[idx0].copy())
    for _ in range(1, n_clusters):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(M_full, p=probs))
        centers.append(X[idx].copy())
    C = np.stack(centers, axis=0)  # (k, F)
    labels = np.zeros(M_full, dtype=int)

    # Lloyd's algorithm (30 iterations)
    for _ in range(30):
        # Assign to nearest centroid
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        # Update centroids
        newC = np.zeros_like(C)
        for k in range(n_clusters):
            mask = (labels == k)
            if np.any(mask):
                newC[k] = X[mask].mean(axis=0)
            else:
                # Reinit empty cluster to farthest point
                far_idx = int(np.argmax(np.min(dists, axis=1)))
                newC[k] = X[far_idx]
        if np.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC

    # Convert back to torch
    W_reduced = torch.from_numpy(C.T).float()  # (F, k)
    W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)

    # Compute reconstruction error
    W_reconstructed = W_reduced[:, labels]
    recon_error = (W - W_reconstructed).norm().item() / W.norm().item()

    print(f"K-means completed:")
    print(f"  Centroids shape: {W_reduced.shape}")
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")
    print(f"  Reconstruction error: {recon_error:.4f}")

    return W_reduced, labels


def build_dictionary(H: torch.Tensor, W: torch.Tensor, angles: np.ndarray):
    """
    Build dictionary D = H ⊙ W (outer product over all combinations).
    (Copied from omp-transformer-ldv.py)
    """
    print(f"\n=== Building Dictionary ===")
    F, E = H.shape
    _, M = W.shape
    P = E * M

    D = torch.zeros(F, P)
    idx2angle = []

    j = 0
    for e in range(E):
        for m in range(M):
            atom = H[:, e] * W[:, m]
            atom = atom / (atom.norm() + 1e-12)
            D[:, j] = atom
            idx2angle.append((float(angles[e]), m))
            j += 1

    # Compute mutual coherence
    G = D.T @ D
    G.fill_diagonal_(0)
    mu = G.abs().max().item()
    mu_mean = G.abs().mean().item()

    print(f"Dictionary built:")
    print(f"  Shape: {D.shape} (F={F}, P={P})")
    print(f"  Mutual coherence μ_max: {mu:.4f}")
    print(f"  Mutual coherence μ_mean: {mu_mean:.4f}")

    return D, idx2angle


def main():
    # Load configuration
    config_path = Path("results/omp_transformer_speech260_trainval_split_full_20251115_082341/code_state.json")
    with open(config_path) as f:
        config = json.load(f)

    h_path = config['args']['h_path']
    w_path = config['args']['w_path']
    n_atoms = config['args']['n_atoms']
    atom_reduce_mode = config['args']['atom_reduce_mode']

    print(f"Loading matrices from experiment configuration:")
    print(f"  H path: {h_path}")
    print(f"  W path: {w_path}")
    print(f"  n_atoms: {n_atoms}")
    print(f"  atom_reduce_mode: {atom_reduce_mode}")

    # Load H matrix
    print(f"\n=== Loading H Matrix ===")
    h_data = torch.load(h_path, map_location='cpu', weights_only=False)
    H = h_data['H']  # (346, 37)
    angles = h_data['angles']  # 37 angles
    print(f"  H shape: {H.shape}")
    print(f"  Angles: {len(angles)} total")
    print(f"  Angle range: {angles.min():.1f}° to {angles.max():.1f}°")

    # Load W matrix (50 atoms)
    print(f"\n=== Loading W Matrix ===")
    w_data = torch.load(w_path, map_location='cpu', weights_only=False)
    if isinstance(w_data, dict):
        W_full = w_data['W']
    else:
        W_full = w_data
    print(f"  W shape: {W_full.shape}")

    # Reduce atoms using k-means
    if atom_reduce_mode == 'kmeans':
        W_reduced, labels = reduce_atoms_kmeans(W_full, n_clusters=n_atoms)
    else:
        raise ValueError(f"Unknown atom_reduce_mode: {atom_reduce_mode}")

    # Build dictionary
    D, idx2angle = build_dictionary(H, W_reduced, angles)

    # Save dictionary
    output_path = Path("results/omp_transformer_speech260_trainval_split_full_20251115_082341/dictionary.npz")
    np.savez(
        output_path,
        D=D.numpy(),
        H=H.numpy(),
        W_reduced=W_reduced.numpy(),
        W_full=W_full.numpy(),
        angles=angles.numpy(),
        idx2angle=idx2angle,
        atom_labels=labels
    )
    print(f"\n✓ Dictionary saved to: {output_path}")
    print(f"  D shape: {D.shape}")
    print(f"  Available keys: D, H, W_reduced, W_full, angles, idx2angle, atom_labels")

    # Quick verification
    print(f"\n=== Verification ===")
    print(f"  Expert 0 (angle {angles[0]:.1f}°) atoms: D[:, 0:8]")
    print(f"  Expert 1 (angle {angles[1]:.1f}°) atoms: D[:, 8:16]")
    print(f"  Expert 16 (angle {angles[16]:.1f}°) atoms: D[:, 128:136]")
    print(f"  ...")
    print(f"  Expert 36 (angle {angles[36]:.1f}°) atoms: D[:, 288:296]")


if __name__ == '__main__':
    main()
