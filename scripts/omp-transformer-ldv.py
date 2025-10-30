#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Routed Soft-OMP with Real LDV Data
- Uses real H matrix (37 angles, Original→Box transfer functions)
- Uses real W matrix (50-atom USM from 111 speakers)
- Preprocessing: K-means atom reduction (50→8), NO PCA (use full F=346)
- Parameter reduction: d_model < F to reduce Transformer parameters
- Training: Reconstruction + monotonicity + angle classification
- Evaluation: Compare with greedy Soft-OMP baseline (83.8% accuracy)

Migration from commit 4d9bb81 (synthetic VQ codebook) to real LDV data (commits b573aa6 & dd1e20d)
"""

import math, os, json, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import datetime
from sklearn.cluster import KMeans
from scipy import signal as scipy_signal

# Import DoADataset for proper STFT processing (matches b573aa6)
from doa_rl.data import DoADataset, create_dataloader

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# PART 0: Data Loading and Preprocessing
# ============================================================================

def load_raw_ldv_matrices(h_path: str, w_path: str, device='cpu'):
    """Load raw H and W matrices from LDV data (commits dd1e20d & b573aa6)."""
    print(f"Loading H matrix from: {h_path}")
    h_data = torch.load(h_path, map_location=device, weights_only=False)
    H = h_data['H']  # (346, 37)
    angles = h_data['angles']  # 37 angles in degrees
    
    print(f"Loading W matrix from: {w_path}")
    w_data = torch.load(w_path, map_location=device, weights_only=False)
    # USM file is a dict with key 'W'
    if isinstance(w_data, dict):
        W = w_data['W']  # (346, 50)
    else:
        W = w_data
    
    print(f"Raw data loaded:")
    print(f"  H shape: {H.shape}, range: [{H.min():.6f}, {H.max():.6f}]")
    print(f"  W shape: {W.shape}, range: [{W.min():.6f}, {W.max():.6f}]")
    print(f"  Angles: {len(angles)} total, {angles.tolist()}")
    
    return H, W, angles


def reduce_atoms_kmeans(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42):
    """
    Reduce W atoms from 50 → n_clusters using K-means clustering.
    
    Args:
        W: (F, M) where F=346 freq bins, M=50 atoms
        n_clusters: Target number of atoms (default: 8)
    
    Returns:
        W_reduced: (F, n_clusters) centroids
        labels: (M,) cluster assignment for each original atom
        kmeans: fitted KMeans object
    """
    print(f"\n=== Atom Reduction via K-means ===")
    print(f"Input: W shape {W.shape} (346 freq bins × 50 atoms)")
    print(f"Target: {n_clusters} clusters")
    
    # Transpose for clustering: cluster atoms (rows), not frequencies
    W_np = W.cpu().numpy().T  # (50, 346)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(W_np)
    
    # Get centroids and transpose back to (F, M')
    centroids = kmeans.cluster_centers_  # (n_clusters, 346)
    W_reduced = torch.from_numpy(centroids.T).float()  # (346, n_clusters)
    
    # Normalize centroids
    W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)
    
    # Compute reconstruction error
    W_reconstructed = W_reduced[:, labels]  # (346, 50)
    recon_error = (W - W_reconstructed).norm().item() / W.norm().item()
    
    print(f"K-means completed:")
    print(f"  Centroids shape: {W_reduced.shape}")
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}")
    print(f"  Reconstruction error: {recon_error:.4f} ({recon_error*100:.2f}%)")
    
    return W_reduced, labels, kmeans


def reduce_atoms_kcenter(W: torch.Tensor, n_clusters: int = 8):
    """
    Reduce W atoms using greedy k-center (farthest-first) selection under cosine distance.

    Args:
        W: (F, M_full)
        n_clusters: number of atoms to select

    Returns:
        W_reduced: (F, n_clusters) selected atoms (from original W), L2-normalized per column
        labels: (M_full,) assignment to nearest selected center (cosine similarity)
        info: dict with 'selected_indices'
    """
    F, M_full = W.shape
    X = W.cpu().numpy().T  # (M_full, F)
    # Normalize rows for cosine computations
    eps = 1e-12
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    # pick first center as farthest from mean
    mean = Xn.mean(axis=0, keepdims=False)
    d0 = 1.0 - (Xn @ (mean / (np.linalg.norm(mean) + eps)))
    first = int(np.argmax(d0))
    centers_idx = [first]
    # iterative farthest-first
    dmin = 1.0 - (Xn @ Xn[first].T)
    for _ in range(1, n_clusters):
        idx = int(np.argmax(dmin))
        centers_idx.append(idx)
        dnew = 1.0 - (Xn @ Xn[idx].T)
        dmin = np.minimum(dmin, dnew)
    centers_idx = sorted(list(dict.fromkeys(centers_idx)))
    # Build reduced W from original columns and normalize
    W_reduced = W[:, centers_idx].clone()
    W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)
    # Assign labels to nearest center (cosine sim)
    C = Xn[centers_idx]  # (k, F)
    sims = Xn @ C.T  # (M_full, k)
    labels = np.argmax(sims, axis=1)
    # Diagnostics: reconstruction using nearest selected atom (not linear combo)
    W_reconstructed = W_reduced[:, labels]
    recon_error = (W - W_reconstructed).norm().item() / (W.norm().item() + 1e-12)
    print(f"K-center completed:")
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = [0]*len(centers_idx)
    for u, c in zip(unique.tolist(), counts.tolist()):
        if 0 <= u < len(cluster_sizes):
            cluster_sizes[u] = c
    print(f"  Selected indices: {centers_idx}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Reconstruction (NN) error: {recon_error:.4f} ({recon_error*100:.2f}%)")
    return W_reduced, labels, {'selected_indices': centers_idx}


def reduce_atoms_diverse(W: torch.Tensor, n_clusters: int = 8, min_cos: float = 0.98):
    """
    Diversity selection with cosine max-min and separation threshold.

    Picks centers greedily (farthest-first) but requires every new center to have
    cosine similarity < min_cos to all previously selected centers. If no candidate
    satisfies the threshold, picks the farthest anyway to ensure we reach n_clusters.
    """
    import numpy as np
    F, M_full = W.shape
    X = W.cpu().numpy().T  # (M_full, F)
    eps = 1e-12
    norms = np.linalg.norm(X, axis=1, keepdims=True) + eps
    Xn = X / norms
    # start from farthest from mean
    mean = Xn.mean(axis=0, keepdims=False)
    mean /= (np.linalg.norm(mean) + eps)
    d_to_mean = 1.0 - (Xn @ mean)
    first = int(np.argmax(d_to_mean))
    centers_idx = [first]
    # precompute cosine to speed up checks
    def cos(a, b):
        return float((a @ b))
    # distances to current set (cosine distance = 1 - cos)
    dmin = 1.0 - (Xn @ Xn[first].T)
    for _ in range(1, n_clusters):
        # candidate order by farthest-first
        order = np.argsort(-dmin)
        chosen = None
        for idx in order:
            ok = True
            xi = Xn[idx]
            for c in centers_idx:
                if cos(xi, Xn[c]) >= min_cos:
                    ok = False
                    break
            if ok:
                chosen = int(idx)
                break
        if chosen is None:
            # fallback: take the farthest to progress
            chosen = int(order[0])
        centers_idx.append(chosen)
        # update dmin
        dnew = 1.0 - (Xn @ Xn[chosen].T)
        dmin = np.minimum(dmin, dnew)
    centers_idx = sorted(list(dict.fromkeys(centers_idx)))
    W_reduced = W[:, centers_idx].clone()
    W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)
    # assign labels to nearest center by cosine similarity
    C = Xn[centers_idx]
    sims = Xn @ C.T
    labels = np.argmax(sims, axis=1)
    # diagnostics
    W_reconstructed = W_reduced[:, labels]
    recon_error = (W - W_reconstructed).norm().item() / (W.norm().item() + 1e-12)
    # cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    sizes = [0] * len(centers_idx)
    for u, c in zip(unique.tolist(), counts.tolist()):
        if 0 <= u < len(sizes):
            sizes[u] = c
    print("Diverse selection completed:")
    print(f"  min_cos threshold: {min_cos}")
    print(f"  Selected indices: {centers_idx}")
    print(f"  Cluster sizes: {sizes}")
    print(f"  Reconstruction (NN) error: {recon_error:.4f} ({recon_error*100:.2f}%)")
    return W_reduced, labels, {'selected_indices': centers_idx}


def reduce_atoms(W: torch.Tensor, mode: str = 'kmeans', n_clusters: int = 8, random_state: int = 42, min_cos: float = 0.98):
    if mode == 'kmeans':
        return reduce_atoms_kmeans(W, n_clusters=n_clusters, random_state=random_state)
    elif mode == 'kcenter':
        return reduce_atoms_kcenter(W, n_clusters=n_clusters)
    elif mode == 'diverse':
        return reduce_atoms_diverse(W, n_clusters=n_clusters, min_cos=float(min_cos))
    else:
        raise ValueError(f"Invalid atom reduce mode: {mode}")


def build_dictionary(H: torch.Tensor, W: torch.Tensor, angles: np.ndarray):
    """
    Build dictionary D = H ⊙ W (outer product over all combinations).
    
    Args:
        H: (F, E) where E=37 angles
        W: (F, M) where M=8 atoms
        angles: (E,) angle values in degrees
    
    Returns:
        D: (F, P) where P=E×M=296
        idx2angle: List of (angle_deg, atom_idx) for each dictionary column
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
    print(f"  Index mapping: {len(idx2angle)} entries")
    
    return D, idx2angle


def load_ldv_samples(dataset_root: str, H: torch.Tensor, W: torch.Tensor, 
                     angles_target: np.ndarray, device='cpu'):
    """
    Load all LDV samples using DoADataset (matching b573aa6 STFT processing).
    
    CRITICAL: Uses DoADataset with scipy.signal.stft (DC offset removal, detrend='constant')
    instead of custom torch.stft to ensure 83.8% greedy baseline accuracy.
    
    Args:
        dataset_root: Path to white_noise_box_data_no_edge_sync_vad_normalized/
        H: (F, E) where F=346, E=37 angles (NO PCA)
        W: (F, M) where F=346, M=8 atoms (NO PCA)
        angles_target: (37,) Expected angles array
        
    Returns:
        Y_samples: (N, F) where N=111, F=346 (full frequency resolution)
        labels: (N,) angle indices (0-36)
        metadata: List of dicts with angle_deg, clip_id, file_path
    """
    print(f"\n=== Loading LDV Samples with DoADataset ===")
    print(f"Dataset root: {dataset_root}")
    
    # Create DoADataset (SAME as b573aa6 greedy evaluation)
    angles_list = angles_target.tolist()
    dataset = DoADataset(
        root=dataset_root,
        angles=angles_list,
        fs=16000,
        n_fft=2048,
        window='hann',
        freq_min=300.0,
        freq_max=3000.0
    )
    
    print(f"DoADataset created:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Angles: {angles_list}")
    print(f"  STFT: fs=16000, n_fft=2048, freq_band=[300, 3000]Hz")
    print(f"  Using scipy.signal.stft with DC offset removal")
    
    # Load all samples
    Y_list = []
    labels = []
    metadata = []
    
    dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
    
    for batch in dataloader:
        Y = batch['Y'][0]  # (F, N) where F=346, N=time frames
        angle_deg = float(batch['angle_deg'][0])
        angle_idx = int(batch['angle_index'][0])
        path = batch['path'][0]
        
        # Time-average to get single spectrum (matching b573aa6)
        y = Y.mean(dim=1)  # (F,) where F=346
        
        # Normalize
        y = y / (y.norm() + 1e-12)
        
        # Use full frequency resolution (NO PCA projection)
        y_final = y.to(device)
        
        Y_list.append(y_final)
        labels.append(angle_idx)
        metadata.append({
            'angle_deg': angle_deg,
            'angle_idx': angle_idx,
            'path': path
        })
    
    if len(Y_list) == 0:
        raise ValueError("No samples loaded! Check dataset_root and angles.")
    
    Y_samples = torch.stack(Y_list, dim=0)  # (N, F_pca)
    labels = torch.tensor(labels, dtype=torch.long).to(device)  # (N,)
    
    print(f"\nLoaded samples:")
    print(f"  Y_samples: {Y_samples.shape}")
    print(f"  Labels: {labels.shape}, unique angles: {labels.unique().numel()}")
    print(f"  Metadata: {len(metadata)} entries")
    
    return Y_samples, labels, metadata


def compute_dataset_fingerprint(dataset_root: str):
    """Compute MD5 fingerprint of all .npy files in dataset."""
    root = Path(dataset_root)
    npy_files = sorted(root.rglob('*.npy'))
    
    hasher = hashlib.md5()
    for npy_file in npy_files:
        with open(npy_file, 'rb') as f:
            hasher.update(f.read())
    
    fingerprint = hasher.hexdigest()
    print(f"Dataset fingerprint (MD5): {fingerprint}")
    print(f"  Computed from {len(npy_files)} .npy files")
    return fingerprint


# ============================================================================
# PART 1: Soft Routing Operations (from original omp-transformer.py)
# ============================================================================

def gumbel_softmax(logits: torch.Tensor, tau: torch.Tensor | float, hard: bool = False) -> torch.Tensor:
    """Gumbel-Softmax sampling for differentiable routing.

    Keeps `tau` as a tensor when provided to preserve gradient flow.
    """
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    if isinstance(tau, torch.Tensor):
        tau_eff = tau.clamp_min(1e-8)
    else:
        tau_eff = max(float(tau), 1e-8)
    y = F.softmax((logits + g) / tau_eff, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax activation for sparse routing."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim) + 1, device=logits.device).view((1,) * (z.dim() - 1) + (-1,))
    cumsum = z_sorted.cumsum(dim=dim)
    rho = torch.sum(z_sorted > (cumsum - 1) / k, dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, rho - 1) - 1) / rho
    out = torch.clamp(z - tau, min=0.0)
    return out


def entmax15(logits: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """Entmax-1.5 activation for intermediate sparsity."""
    eps = 1e-12
    x = logits / 2
    for _ in range(n_iter):
        p = torch.clamp(x, min=0) ** 2
        Z = p.sum(dim=dim, keepdim=True) + eps
        x = ((logits + 2 * p.sum(dim=dim, keepdim=True) / p.size(dim)) / 2).clamp(min=0)
    p = torch.clamp(x, min=0) ** 2
    return p / (p.sum(dim=dim, keepdim=True) + eps)


# ============================================================================
# PART 2: Transformer Routed Soft-OMP Model
# ============================================================================

class FullTransformerRoutedSoftOMP(nn.Module):
    """
    Full-Transformer Routed Soft-OMP with real LDV data.
    
    Architecture:
    - Explicit tokens: [Residual token; Dictionary tokens]
    - TransformerEncoder with custom attention mask
    - Training: Soft routing (Gumbel-Softmax)
    - Inference: Hard routing (Top-K experts, Top-L atoms)
    
    Changes from synthetic version:
    - F=64 (PCA-reduced frequency)
    - E=37 (all LDV angles)
    - M=8 (K-means reduced atoms)
    - P=296 (total dictionary size)
    """
    
    def __init__(self, F: int, E: int, M: int, d: int = None, nhead: int = 8, nlayers: int = 1,
                 steps: int = 6, top_e: int = 2, L: int = 2,
                 tau_e: float = 0.5, tau_a: float = 0.2, eta: float = 0.5,
                 routing: str = 'gumbel', routing_mode: str = 'qk', hybrid_alpha: float = 0.5,
                 routing_e: str | None = None, routing_a: str | None = None,
                 score_norm_mode: str = 'none', score_reg_weight: float = 0.0,
                 expert_agg: str = 'l2'):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.P = E * M
        self.d = d if d is not None else F
        self.steps, self.top_e, self.L = steps, top_e, L
        
        # Learnable routing parameters
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        # Routing activations
        self.routing = routing  # backward-compat (unused after split)
        self.routing_e = routing_e if routing_e is not None else routing
        self.routing_a = routing_a if routing_a is not None else routing
        # Routing score source: 'qk' (Transformer QK), 'g' (physics correlation), 'hybrid' (blend)
        assert routing_mode in ('qk', 'g', 'hybrid')
        self.routing_mode = routing_mode
        self.hybrid_alpha = float(hybrid_alpha)
        # Score normalization and regularization controls
        assert score_norm_mode in ('none', 'std')
        self.score_norm_mode = score_norm_mode
        self.score_reg_weight = float(score_reg_weight)
        # Expert aggregation mode for QK → expert scores
        assert expert_agg in ('l2', 'max', 'mean_relu')
        self.expert_agg = expert_agg

        # Token projections + type embeddings
        self.P_R = nn.Linear(F, self.d, bias=False)
        self.P_D = nn.Linear(F, self.d, bias=False)
        nn.init.eye_(self.P_R.weight) if self.d == F else nn.init.xavier_uniform_(self.P_R.weight)
        nn.init.eye_(self.P_D.weight) if self.d == F else nn.init.xavier_uniform_(self.P_D.weight)
        self.type_R = nn.Parameter(torch.randn(self.d))
        self.type_D = nn.Parameter(torch.randn(self.d))
        # Signal-preserving toggles (configured from CLI in main)
        self.no_type_bias: bool = False
        self.encoder_identity: bool = False
        self.single_gate_expert: bool = False
        self.score_center_atoms: bool = False
        self.score_center_expert: bool = False
        self.d_can_attend_r: bool = False
        
        # TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d, nhead=nhead, dim_feedforward=4 * self.d, 
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        
        # Query/Key projections
        self.Wq = nn.Linear(self.d, self.d, bias=False)
        self.Wk = nn.Linear(self.d, self.d, bias=False)

        # Diagnostics controls (non-breaking): when enabled, forward() stores step-0 signals
        self.enable_diag: bool = False
        self.last_diag = None
    
    def _build_tokens(self, r: torch.Tensor, D: torch.Tensor):
        """Build token sequence: [Residual; Dictionary atoms]."""
        if self.no_type_bias:
            t_R = self.P_R(r)
            T_D = self.P_D(D.T)
        else:
            t_R = self.P_R(r) + self.type_R  # (d,)
            T_D = self.P_D(D.T) + self.type_D  # (P, d)
        T = torch.cat([t_R[None, :], T_D], dim=0)  # (1+P, d)
        return T
    
    def _make_mask(self, S: int):
        """
        Build attention mask:
        - R (index 0) attends to all tokens
        - Each D_j (index j≥1) attends only to itself
        """
        mask = torch.full((S, S), float('-inf'))
        mask[0, :] = 0.0  # R attends to all
        for j in range(1, S):
            mask[j, j] = 0.0  # D_j attends to itself
            if self.d_can_attend_r:
                mask[j, 0] = 0.0  # allow D_j to attend R
        return mask
    
    def _soft_picker(self, logits: torch.Tensor, tau: torch.Tensor, mode: str, hard: bool):
        """Apply soft routing function."""
        if mode == 'gumbel':
            return gumbel_softmax(logits, tau=tau, hard=hard)
        elif mode == 'entmax':
            return entmax15(logits, dim=-1)
        elif mode == 'sparsemax':
            return sparsemax(logits, dim=-1)
        else:
            return F.softmax(logits / tau.clamp_min(1e-8), dim=-1)
    
    def forward(self, y: torch.Tensor, D: torch.Tensor, train_mode: bool = True):
        """
        Forward pass: iterative residual reduction with routing.
        
        Args:
            y: (F,) observation
            D: (F, P) dictionary
            train_mode: If True, use soft routing; if False, use hard routing
        
        Returns:
            x: (P,) sparse coefficients
            res_curve: List of residual norms at each step
        """
        F, P = D.shape
        x = torch.zeros(P, device=D.device)
        r = y.clone()
        res_curve = []
        
        # reset diagnostics snapshot each call
        if self.enable_diag:
            self.last_diag = None
        # expose non-detached tensors for training supervision
        self.last_outputs = None
        _res_norms_t: list[torch.Tensor] = []
        _scores_expert_steps: list[torch.Tensor] = []

        for step in range(self.steps):
            # Build tokens and apply Transformer
            T = self._build_tokens(r, D)  # (1+P, d)
            S = T.size(0)
            mask = self._make_mask(S).to(T.device)
            if self.encoder_identity:
                H = T
            else:
                H = self.encoder(T, mask=mask)  # (1+P, d)
            
            h_R = H[0]  # (d,)
            H_D = H[1:]  # (P, d)
            
            # Compute QK scores from Transformer
            qk_atoms = (self.Wk(H_D) @ self.Wq(h_R)) / math.sqrt(self.d)  # (P,)
            qk_atoms = qk_atoms.reshape(self.E, self.M)  # (E, M)
            if self.expert_agg == 'l2':
                qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)  # (E,)
            elif self.expert_agg == 'max':
                qk_expert = qk_atoms.abs().max(dim=1).values
            else:  # 'mean_relu'
                qk_expert = F.relu(qk_atoms).mean(dim=1)

            # Compute physics correlation scores from g = D^T r
            g_vec_all = (D.T @ r)  # (P,)
            g_atoms = g_vec_all.reshape(self.E, self.M)
            g_expert = torch.sqrt((g_atoms ** 2).sum(dim=1) + 1e-12)  # (E,)

            # Choose routing score source
            if self.routing_mode == 'qk':
                scores_atoms = qk_atoms
                scores_expert = qk_expert
            elif self.routing_mode == 'g':
                scores_atoms = g_atoms
                scores_expert = g_expert
            else:  # hybrid
                # Normalize both to unit norm before blending
                def _norm(x, dim=None):
                    n = x.norm(dim=dim, keepdim=True) if dim is not None else x.norm()
                    return x / (n + 1e-12)
                scores_atoms = self.hybrid_alpha * _norm(g_atoms, dim=None) + (1.0 - self.hybrid_alpha) * _norm(qk_atoms, dim=None)
                scores_expert = self.hybrid_alpha * _norm(g_expert) + (1.0 - self.hybrid_alpha) * _norm(qk_expert)
            # Optional: score normalization to increase contrast and stability
            if self.score_norm_mode == 'std':
                # standardize expert and atom scores separately (global stats)
                se_mean, se_std = scores_expert.mean(), scores_expert.std()
                scores_expert = (scores_expert - se_mean) / (se_std + 1e-8)
                sa_mean, sa_std = scores_atoms.mean(), scores_atoms.std()
                scores_atoms = (scores_atoms - sa_mean) / (sa_std + 1e-8)
            # Optional: explicit centering to reduce common-mode terms
            if self.score_center_expert:
                scores_expert = scores_expert - scores_expert.mean()
            if self.score_center_atoms:
                scores_atoms = scores_atoms - scores_atoms.mean(dim=1, keepdim=True)
            
            if train_mode:
                # Soft routing
                # NOTE: use learnable temperatures directly (no .item()) so they can get gradients / schedules
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing_e, hard=False)  # (E,)
                w_all = torch.zeros(self.E, self.M, device=D.device)
                w_a_list = []  # Collect w_a for all experts
                for e in range(self.E):
                    if self.single_gate_expert:
                        w_a_e = torch.ones(self.M, device=D.device)
                    else:
                        w_a_e = self._soft_picker(scores_atoms[e].abs(), self.tau_a, self.routing_a, hard=False)
                    w_all[e] = w_e[e] * w_a_e
                    w_a_list.append(w_a_e.detach())
                w_all = w_all.reshape(-1)  # (P,)
                # Save diagnostics (detached) and training-useful tensors (non-detached)
                if self.last_diag is None:
                    self.last_diag = {}
                self.last_diag['scores_expert'] = scores_expert.detach()
                if self.enable_diag and step == 0:
                    self.last_diag.update({
                        'scores_atoms': scores_atoms.detach(),
                        'w_e': w_e.detach(),
                        'w_a_list': w_a_list,
                        'g_vec': g_vec_all.detach(),
                    })
                if step == 0:
                    self.last_outputs = {
                        'scores_expert': scores_expert,
                        'scores_atoms': scores_atoms,
                        'w_e': w_e,
                    }
                _scores_expert_steps.append(scores_expert)
                g = g_vec_all  # (P,) gradient-like signal from above
                # Remove .item() on eta to keep it learnable / schedulable
                x = x + self.eta * (w_all * g)
            else:
                # Hard routing
                kE = min(self.top_e, self.E)
                chosen_e = torch.topk(scores_expert, k=kE).indices.tolist()
                chosen_idx = []
                for e in chosen_e:
                    kL = min(self.L, self.M)
                    chosen_a = torch.topk(scores_atoms[e].abs(), k=kL).indices.tolist()
                    chosen_idx += [int(e) * self.M + int(a) for a in chosen_a]
                chosen_idx = list(dict.fromkeys(chosen_idx))
                if step == 0:
                    if self.last_diag is None:
                        self.last_diag = {}
                    self.last_diag['scores_expert'] = scores_expert.detach()
                if len(chosen_idx) > 0:
                    g = (D.T @ r)
                    x[chosen_idx] = x[chosen_idx] + self.eta * g[chosen_idx]
            
            # Update residual and track norms for mono loss (keep tensor for gradients)
            r = y - D @ x
            res_norm = torch.norm(r)
            _res_norms_t.append(res_norm)
            res_curve.append(float(res_norm.detach()))
        
        # Attach residual norms tensor sequence for training use
        if self.last_outputs is None:
            self.last_outputs = {}
        self.last_outputs['res_norms_t'] = torch.stack(_res_norms_t, dim=0) if _res_norms_t else torch.tensor([], device=D.device)
        if _scores_expert_steps:
            try:
                self.last_outputs['scores_expert_steps'] = torch.stack(_scores_expert_steps, dim=0)
            except Exception:
                self.last_outputs['scores_expert_steps'] = None

        return x, res_curve


# ============================================================================
# PART 3: Training and Evaluation
# ============================================================================

def train_epoch(model: FullTransformerRoutedSoftOMP, D: torch.Tensor, 
                Y_samples: torch.Tensor, labels: torch.Tensor,
                opt: torch.optim.Optimizer, idx2angle: List[Tuple[float, int]],
                batch_size: int = 16, device='cpu',
                alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.5,
                align_weight: float = 0.0,
                distill_T: float = 1.0, distill_weight: float = 0.0,
                probe_grad_split: bool = False,
                supervise_steps: str = 'first', supervise_k: int = 1,
                w_e_entropy_penalty: float = 0.0,
                nce_weight: float = 0.0, nce_T: float = 1.0,
                epoch: int | None = None, diag_path: str | None = None, diag_subset: int = 16,
                teacher_warmup_epochs: int = 0, teacher_weight: float = 0.0):
    """
    Train for one epoch on real LDV data.
    
    Loss = α * reconstruction + β * monotonicity + γ * classification
    
    Args:
        model: Transformer Routed Soft-OMP model
        D: (F, P) dictionary
        Y_samples: (N, F) observations
        labels: (N,) angle indices
        opt: Optimizer
        idx2angle: Mapping from dictionary column to (angle_deg, atom_idx)
        batch_size: Batch size
        alpha, beta, gamma: Loss weights
    
    Returns:
        metrics: Dict of average losses
    """
    model.train()
    N = Y_samples.size(0)
    indices = torch.randperm(N)
    
    rec_losses = []
    mono_losses = []
    class_losses = []
    teacher_losses = []
    align_losses = []
    reg_losses = []
    ce_only_terms = []
    align_terms = []
    reg_terms = []
    logits_margins = []
    distill_terms = []
    w_e_entropy_terms = []
    nce_terms = []
    total_losses = []
    
    # Diagnostics accumulators (subset)
    diag_seen = 0
    teacher_correct = 0
    teacher_margins = []
    qk_g_corrs = []
    qk_top1_matches = 0
    w_e_entropies = []
    # NEW: Enhanced diagnostics for hypothesis verification
    g_stats_list = []  # g distribution statistics
    scores_stats_list = []  # scores distribution statistics
    w_a_entropies = []  # atom-level routing entropy

    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_indices = indices[start_idx:end_idx]
        
        yb = Y_samples[batch_indices].to(device)
        lb = labels[batch_indices].to(device)
        # Negatives pool for InfoNCE: collect from previous samples in the same batch (detached)
        neg_pool: list[torch.Tensor] = []
        
        rec_loss = 0.0
        mono_loss = 0.0
        class_loss = 0.0
        teacher_loss = 0.0
        
        for b in range(yb.size(0)):
            # Forward pass
            # enable one-shot diagnostics capture for first few samples
            model.enable_diag = (diag_seen < diag_subset)
            x_hat, r_curve = model(yb[b], D.to(device), train_mode=True)
            y_hat = D.to(device) @ x_hat
            
            # Reconstruction loss
            rec_loss = rec_loss + F.mse_loss(y_hat, yb[b])
            
            # Monotonicity loss (use tensor norms from forward to keep gradients)
            if getattr(model, 'last_outputs', None) is not None and 'res_norms_t' in model.last_outputs:
                rc = model.last_outputs['res_norms_t']  # (steps,)
                if rc.numel() > 1:
                    diffs = rc[1:] - rc[:-1]
                    mono_loss = mono_loss + torch.relu(diffs).sum()
                else:
                    mono_loss = mono_loss + torch.tensor(0.0, device=device)
            else:
                # Fallback (detached; should not happen)
                rc = torch.tensor(r_curve, device=device)
                if rc.numel() > 1:
                    diffs = rc[1:] - rc[:-1]
                    mono_loss = mono_loss + torch.relu(diffs).sum()
                else:
                    mono_loss = mono_loss + torch.tensor(0.0, device=device)
            
            # Classification loss: predict angle from routing scores (NOT from x_hat)
            # Directly supervise routing mechanism; prefer non-detached tensors from forward
            if model.routing_mode == 'g':
                # For g-routing, use |g| energy as ground truth scores
                g_vec = (D.to(device).T @ yb[b])
                logits = g_vec.abs().view(model.E, model.M).sum(dim=1)
            else:
                # For qk/hybrid, supervise first K steps or all steps if requested
                logits = None
                if getattr(model, 'last_outputs', None) is not None and 'scores_expert_steps' in model.last_outputs and model.last_outputs['scores_expert_steps'] is not None and supervise_steps in ('first', 'all'):
                    se_steps = model.last_outputs['scores_expert_steps']  # (steps, E)
                    if se_steps is not None and se_steps.dim() == 2 and se_steps.size(0) > 0:
                        if supervise_steps == 'first':
                            k = min(supervise_k, se_steps.size(0))
                            logits = se_steps[:k].mean(dim=0)
                        else:  # all
                            logits = se_steps.mean(dim=0)
                if logits is None:
                    if getattr(model, 'last_outputs', None) is not None and 'scores_expert' in model.last_outputs:
                        logits = model.last_outputs['scores_expert']
                    elif model.last_diag is not None and 'scores_expert' in model.last_diag:
                        logits = model.last_diag['scores_expert'].to(device)
                    else:
                        # Fallback: recompute g for safety (should not happen in normal flow)
                        g_vec = (D.to(device).T @ yb[b])
                        logits = g_vec.abs().view(model.E, model.M).sum(dim=1)

            ce_term = F.cross_entropy(logits.unsqueeze(0), lb[b].unsqueeze(0))
            class_loss = class_loss + ce_term
            ce_only_terms.append(float(ce_term.item()))
            # record margin
            if logits.numel() >= 2:
                top2 = torch.topk(logits, k=2).values
                margin = float((top2[0] - top2[1]).item())
                logits_margins.append(margin)

            # Optional: score L2 regularization (controls drift/saturation)
            if model.score_reg_weight > 0.0 and getattr(model, 'last_outputs', None) is not None:
                se = model.last_outputs.get('scores_expert', None)
                sa = model.last_outputs.get('scores_atoms', None)
                if se is not None and sa is not None:
                    reg = model.score_reg_weight * ((se ** 2).mean() + (sa ** 2).mean())
                    reg_losses.append(float(reg.item()))
                    class_loss = class_loss + reg
                    reg_terms.append(float(reg.item()))

            # Optional teacher warm-up: supervise per-angle logits with |g|-based teacher
            if (epoch is not None) and (teacher_weight > 0.0) and (epoch < teacher_warmup_epochs):
                g_vec_tw = (D.to(device).T @ yb[b])
                g_energy_tw = g_vec_tw.abs().view(model.E, model.M).sum(dim=1)
                teacher_label = torch.argmax(g_energy_tw)
                teacher_loss = teacher_loss + F.cross_entropy(logits.unsqueeze(0), teacher_label.unsqueeze(0))

            # Optional: alignment loss between QK scores and |g| expert energy (physics anchoring)
            if align_weight > 0.0 and model.routing_mode in ('qk', 'hybrid'):
                g_vec_al = (D.to(device).T @ yb[b])
                g_energy_al = g_vec_al.abs().view(model.E, model.M).sum(dim=1)
                se = logits
                se_n = se / (se.norm() + 1e-8)
                ge_n = g_energy_al / (g_energy_al.norm() + 1e-8)
                align = 1.0 - torch.dot(se_n, ge_n)
                align_losses.append(float(align.item()))
                class_loss = class_loss + align_weight * align
                align_terms.append(float((align_weight * align).item()))

            # Distillation (soft targets from |g|), only for qk/hybrid
            if distill_weight > 0.0 and model.routing_mode in ('qk', 'hybrid'):
                # teacher: softmax(|g|/T); student: softmax(scores_expert/T)
                with torch.no_grad():
                    g_vec_dist = (D.to(device).T @ yb[b])
                    g_energy_dist = g_vec_dist.abs().view(model.E, model.M).sum(dim=1)
                    p_teacher = F.softmax(g_energy_dist / max(distill_T, 1e-8), dim=-1)
                p_student_log = F.log_softmax(logits / max(distill_T, 1e-8), dim=-1)
                # batchmean on single sample equals mean over classes
                distill = (distill_T ** 2) * F.kl_div(p_student_log, p_teacher, reduction='batchmean')
                distill_terms.append(float(distill.item()))
                class_loss = class_loss + distill_weight * distill

            # Expert entropy penalty (encourage peaky expert routing)
            if w_e_entropy_penalty > 0.0 and getattr(model, 'last_outputs', None) is not None and 'w_e' in model.last_outputs:
                pe = torch.clamp(model.last_outputs['w_e'], min=1e-12)
                H = -(pe * pe.log()).sum()
                pen = w_e_entropy_penalty * H
                w_e_entropy_terms.append(float(pen.item()))
                class_loss = class_loss + pen

            # Step-wise InfoNCE (anchor=step0, positive=step1), negatives from previous samples in batch
            if nce_weight > 0.0 and getattr(model, 'last_outputs', None) is not None and 'scores_expert_steps' in model.last_outputs:
                se_steps = model.last_outputs['scores_expert_steps']
                if se_steps is not None and se_steps.dim() == 2 and se_steps.size(0) >= 2:
                    eps = 1e-8
                    a = se_steps[0]
                    p = se_steps[1]
                    a = a / (a.norm() + eps)
                    p = p / (p.norm() + eps)
                    sims = []
                    sims.append(torch.dot(a, p) / max(nce_T, eps))
                    # negatives: use up to 8 most recent vectors in pool
                    if len(neg_pool) > 0:
                        k = min(8, len(neg_pool))
                        for z in neg_pool[-k:]:
                            sims.append(torch.dot(a, z) / max(nce_T, eps))
                        sims_t = torch.stack(sims, dim=0)
                        # InfoNCE loss: -log softmax at index 0
                        nce_loss = -F.log_softmax(sims_t, dim=0)[0]
                        class_loss = class_loss + nce_weight * nce_loss
                        nce_terms.append(float((nce_weight * nce_loss).item()))
                    # add current anchor to pool for future negatives (detach to avoid graph across samples)
                    neg_pool.append(a.detach())
                else:
                    # fallback: add available step0 as potential negative for later samples
                    if se_steps is not None and se_steps.dim() == 2 and se_steps.size(0) >= 1:
                        z = se_steps[0]
                        z = z / (z.norm() + 1e-8)
                        neg_pool.append(z.detach())
            else:
                # if not using nce, still populate pool for consistency
                if getattr(model, 'last_outputs', None) is not None and 'scores_expert_steps' in model.last_outputs:
                    se_steps = model.last_outputs['scores_expert_steps']
                    if se_steps is not None and se_steps.dim() == 2 and se_steps.size(0) >= 1:
                        z = se_steps[0]
                        z = z / (z.norm() + 1e-8)
                        neg_pool.append(z.detach())

            # Gradient split probe (first sample only)
            # NOTE: skip for routing_mode='g' because logits are non-learned (no grad path)
            if probe_grad_split and start_idx == 0 and b == 0 and model.routing_mode != 'g':
                # CE-only
                opt.zero_grad()
                # Recompute forward for clean graph
                x_probe, _ = model(yb[b], D.to(device), train_mode=True)
                y_probe = D.to(device) @ x_probe
                # logits for probe
                if model.routing_mode == 'g':
                    g_vec_p = (D.to(device).T @ yb[b])
                    logits_p = g_vec_p.abs().view(model.E, model.M).sum(dim=1)
                else:
                    logits_p = model.last_outputs.get('scores_expert', logits)
                ce_only = F.cross_entropy(logits_p.unsqueeze(0), lb[b].unsqueeze(0))
                ce_only.backward(retain_graph=True)
                def _gnv():
                    def _gn(param):
                        try:
                            return float(param.grad.norm().item()) if (param is not None and getattr(param, 'grad', None) is not None) else 0.0
                        except Exception:
                            return 0.0
                    return {
                        'Wq': _gn(model.Wq.weight),
                        'Wk': _gn(model.Wk.weight),
                        'encoder': float(sum((p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None), 0.0)),
                    }
                grad_norms_ce = _gnv()
                opt.zero_grad()
                # REC-only
                x_probe2, _ = model(yb[b], D.to(device), train_mode=True)
                y_probe2 = D.to(device) @ x_probe2
                rec_only = F.mse_loss(y_probe2, yb[b])
                rec_only.backward()
                grad_norms_rec = _gnv()
                opt.zero_grad()

            # Diagnostics: teacher (|g|) vs QK alignment on a small subset
            if model.enable_diag:
                diag_seen += 1
                g_vec = (D.to(device).T @ yb[b])
                g_energy = g_vec.abs().view(model.E, model.M).sum(dim=1)
                # teacher stats
                top_vals, _ = torch.topk(g_energy, k=min(2, g_energy.numel()))
                margin = float((top_vals[0] - (top_vals[1] if top_vals.numel() > 1 else 0.0)).item())
                teacher_margins.append(margin)
                teacher_pred = int(torch.argmax(g_energy).item())
                teacher_correct += int(teacher_pred == int(lb[b].item()))
                # alignment with scores_expert
                if model.last_diag is not None and 'scores_expert' in model.last_diag:
                    se = model.last_diag['scores_expert'].to(device)
                    se_c = se - se.mean()
                    ge_c = g_energy - g_energy.mean()
                    denom = (se_c.norm() * ge_c.norm()).item()
                    pearson = float((se_c @ ge_c).item() / denom) if denom > 0 else 0.0
                    qk_g_corrs.append(pearson)
                    qk_pred = int(torch.argmax(se).item())
                    if qk_pred == teacher_pred:
                        qk_top1_matches += 1
                if model.last_diag is not None and 'w_e' in model.last_diag:
                    w_e = model.last_diag['w_e'].to(device)
                    pe = torch.clamp(w_e, min=1e-12)
                    ent = float((-pe * pe.log()).sum().item())
                    w_e_entropies.append(ent)
                
                # NEW: Collect g statistics
                if model.last_diag is not None and 'g_vec' in model.last_diag:
                    g_vec_diag = model.last_diag['g_vec'].to(device)
                    g_abs = g_vec_diag.abs()
                    g_stats = {
                        'g_min': float(g_abs.min().item()),
                        'g_max': float(g_abs.max().item()),
                        'g_mean': float(g_abs.mean().item()),
                        'g_std': float(g_abs.std().item()),
                        'g_p50': float(g_abs.median().item()),
                        'g_p95': float(torch.quantile(g_abs, 0.95).item()),
                        'g_p99': float(torch.quantile(g_abs, 0.99).item()),
                        'g_near_zero_ratio': float((g_abs < 0.01).float().mean().item()),
                        'g_top10_mean': float(torch.topk(g_abs, k=min(10, g_abs.numel())).values.mean().item()),
                    }
                    g_stats_list.append(g_stats)
                
                # NEW: Collect scores statistics
                if model.last_diag is not None and 'scores_atoms' in model.last_diag and 'scores_expert' in model.last_diag:
                    scores_atoms_diag = model.last_diag['scores_atoms'].to(device)
                    scores_expert_diag = model.last_diag['scores_expert'].to(device)
                    scores_stats = {
                        'scores_atoms_min': float(scores_atoms_diag.min().item()),
                        'scores_atoms_max': float(scores_atoms_diag.max().item()),
                        'scores_atoms_mean': float(scores_atoms_diag.mean().item()),
                        'scores_atoms_std': float(scores_atoms_diag.std().item()),
                        'scores_expert_min': float(scores_expert_diag.min().item()),
                        'scores_expert_max': float(scores_expert_diag.max().item()),
                        'scores_expert_mean': float(scores_expert_diag.mean().item()),
                        'scores_expert_std': float(scores_expert_diag.std().item()),
                    }
                    scores_stats_list.append(scores_stats)
                
                # NEW: Collect w_a entropy (atom-level routing)
                if model.last_diag is not None and 'w_a_list' in model.last_diag:
                    w_a_list_diag = model.last_diag['w_a_list']
                    entropies = []
                    for w_a in w_a_list_diag:
                        pe = torch.clamp(w_a, min=1e-12)
                        ent = float((-pe * pe.log()).sum().item())
                        entropies.append(ent)
                    w_a_entropy_mean = float(np.mean(entropies)) if entropies else 0.0
                    w_a_entropies.append(w_a_entropy_mean)
        
        # Average over batch
        rec_loss /= yb.size(0)
        mono_loss /= yb.size(0)
        class_loss /= yb.size(0)
        teacher_loss = (teacher_loss / yb.size(0)) if (teacher_weight > 0.0 and epoch is not None and epoch < teacher_warmup_epochs) else 0.0
        
        # Total loss
        loss = alpha * rec_loss + beta * mono_loss + gamma * class_loss + (teacher_weight * teacher_loss if isinstance(teacher_loss, torch.Tensor) else 0.0)
        
        # Backward
        opt.zero_grad()
        loss.backward()
        # Gradient norms (last batch only)
        def _gn(param):
            try:
                return float(param.grad.norm().item()) if (param is not None and getattr(param, 'grad', None) is not None) else 0.0
            except Exception:
                return 0.0
        grad_norms = {
            'P_R': _gn(model.P_R.weight),
            'P_D': _gn(model.P_D.weight),
            'Wq': _gn(model.Wq.weight),
            'Wk': _gn(model.Wk.weight),
            'type_R': _gn(model.type_R),
            'type_D': _gn(model.type_D),
            'tau_e': _gn(model.tau_e),
            'tau_a': _gn(model.tau_a),
            'eta': _gn(model.eta),
            'encoder': float(sum((p.grad.norm().item() for p in model.encoder.parameters() if p.grad is not None), 0.0)),
        }
        # NEW: Direct parameter norms for H2 verification
        param_norms = {
            'Wq': float(model.Wq.weight.norm().item()),
            'Wk': float(model.Wk.weight.norm().item()),
            'encoder': float(sum(p.norm().item() for p in model.encoder.parameters())),
        }
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Record
        rec_losses.append(rec_loss.item())
        mono_losses.append(mono_loss.item())
        class_losses.append(class_loss.item())
        total_losses.append(loss.item())
        teacher_losses.append(float(teacher_loss.item()) if isinstance(teacher_loss, torch.Tensor) else 0.0)
    
    # Persist per-epoch diagnostics JSONL
    if diag_path is not None and epoch is not None:
        try:
            rec = {
                'epoch': int(epoch) + 1,
                'rec_loss': float(np.mean(rec_losses)) if rec_losses else None,
                'mono_loss': float(np.mean(mono_losses)) if mono_losses else None,
                'class_loss': float(np.mean(class_losses)) if class_losses else None,
                'teacher_loss': float(np.mean(teacher_losses)) if teacher_losses else None,
                'total_loss': float(np.mean(total_losses)) if total_losses else None,
                'class_ce_only': float(np.mean(ce_only_terms)) if ce_only_terms else None,
                'class_align_term': float(np.mean(align_terms)) if align_terms else None,
                'class_reg_term': float(np.mean(reg_terms)) if reg_terms else None,
                'class_distill_term': float(np.mean(distill_terms)) if distill_terms else None,
                'w_e_entropy_term': float(np.mean(w_e_entropy_terms)) if w_e_entropy_terms else None,
                'teacher_samples': int(diag_seen),
                'teacher_acc_subset': float(teacher_correct / max(1, diag_seen)),
                'teacher_margin_p50': float(np.median(teacher_margins)) if teacher_margins else None,
                'teacher_margin_p95': float(np.percentile(teacher_margins, 95)) if teacher_margins else None,
                'logits_margin_p50': float(np.median(logits_margins)) if logits_margins else None,
                'logits_margin_p95': float(np.percentile(logits_margins, 95)) if logits_margins else None,
                'qk_g_corr_pearson_mean': float(np.mean(qk_g_corrs)) if qk_g_corrs else None,
                'qk_top1_match_rate': float(qk_top1_matches / max(1, diag_seen)),
                'w_e_entropy_mean': float(np.mean(w_e_entropies)) if w_e_entropies else None,
                'tau_e': float(model.tau_e.item()) if isinstance(model.tau_e, torch.Tensor) else None,
                'tau_a': float(model.tau_a.item()) if isinstance(model.tau_a, torch.Tensor) else None,
                'eta': float(model.eta.item()) if isinstance(model.eta, torch.Tensor) else None,
                'grad_norms': grad_norms,
                'param_norms': param_norms,  # NEW: Direct parameter norms
                # NEW: Enhanced diagnostics for hypothesis verification
                'w_a_entropy_mean': float(np.mean(w_a_entropies)) if w_a_entropies else None,
                'g_stats': {k: float(np.mean([d[k] for d in g_stats_list])) for k in g_stats_list[0].keys()} if g_stats_list else None,
                'scores_stats': {k: float(np.mean([d[k] for d in scores_stats_list])) for k in scores_stats_list[0].keys()} if scores_stats_list else None,
                'align_loss': float(np.mean(align_losses)) if align_losses else None,
                'score_reg_mean': float(np.mean(reg_losses)) if reg_losses else None,
                'nce_term': float(np.mean(nce_terms)) if nce_terms else None,
            }
            # Attach gradient split probe if computed
            if probe_grad_split:
                try:
                    rec['grad_norms_ce'] = grad_norms_ce
                    rec['grad_norms_rec'] = grad_norms_rec
                    # simple ratio on Wq+Wk
                    eps = 1e-12
                    ce_sum = (grad_norms_ce.get('Wq', 0.0) + grad_norms_ce.get('Wk', 0.0))
                    rec_sum = (grad_norms_rec.get('Wq', 0.0) + grad_norms_rec.get('Wk', 0.0))
                    rec['grad_ce_rec_ratio'] = float(ce_sum / (rec_sum + eps))
                except Exception:
                    pass
            with open(diag_path, 'a') as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    return {
        'rec_loss': np.mean(rec_losses),
        'mono_loss': np.mean(mono_losses),
        'class_loss': np.mean(class_losses),
        'total_loss': np.mean(total_losses),
        'teacher_loss': float(np.mean(teacher_losses)) if teacher_losses else 0.0,
        'teacher_acc_subset': (teacher_correct / max(1, diag_seen)) if diag_seen > 0 else None,
        'qk_g_corr_pearson_mean': (float(np.mean(qk_g_corrs)) if qk_g_corrs else None)
    }


@torch.no_grad()
def evaluate(model: FullTransformerRoutedSoftOMP, D: torch.Tensor,
             Y_samples: torch.Tensor, labels: torch.Tensor,
             idx2angle: List[Tuple[float, int]], device='cpu'):
    """
    Evaluate model on samples.
    
    Returns:
        metrics: Dict with accuracy, confusion matrix, per-angle metrics
    """
    model.eval()
    N = Y_samples.size(0)
    E = model.E
    
    predictions = []
    residuals = []
    
    for i in range(N):
        y = Y_samples[i].to(device)
        x_hat, r_curve = model(y, D.to(device), train_mode=False)
        
        # Predict angle from routing scores (consistent with training)
        # CRITICAL FIX: Use scores_expert (routing decision) not x_hat (cumulative coefficients)
        if model.routing_mode == 'g':
            # For g-routing, use |g| energy
            g_vec = (D.to(device).T @ y)
            scores = g_vec.abs().view(E, model.M).sum(dim=1)
        else:
            # For qk/hybrid, use scores_expert from forward pass
            if model.last_diag is not None and 'scores_expert' in model.last_diag:
                scores = model.last_diag['scores_expert'].cpu()
            else:
                # Fallback to g for safety
                g_vec = (D.to(device).T @ y)
                scores = g_vec.abs().view(E, model.M).sum(dim=1)
        
        pred_idx = scores.argmax().item()
        predictions.append(pred_idx)
        residuals.append(r_curve)
    
    predictions = np.array(predictions)
    labels_np = labels.cpu().numpy()
    
    # Accuracy
    accuracy = (predictions == labels_np).mean()
    
    # Confusion matrix
    confusion = np.zeros((E, E), dtype=int)
    for true, pred in zip(labels_np, predictions):
        confusion[true, pred] += 1
    
    # Per-angle accuracy
    per_angle_acc = []
    for e in range(E):
        mask = (labels_np == e)
        if mask.sum() > 0:
            acc = (predictions[mask] == e).mean()
            per_angle_acc.append(acc)
        else:
            per_angle_acc.append(0.0)
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'confusion_matrix': confusion,
        'per_angle_accuracy': per_angle_acc,
        'residuals': residuals
    }


# ============================================================================
# PART 4: Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Transformer Routed Soft-OMP with Real LDV Data')
    
    # Data paths
    parser.add_argument('--h_path', type=str, 
                        default='/Users/sbplab/jiawei/LDV-data-processed/h_matrix_box_ldv_correct.pth',
                        help='Path to H matrix (commit dd1e20d)')
    parser.add_argument('--w_path', type=str,
                        default='doa_normalized_config_c_corrected/models/usm.pth',
                        help='Path to W matrix (commit b573aa6)')
    parser.add_argument('--dataset_root', type=str,
                        default='/Users/sbplab/jiawei/LDV-data-processed/white_noise_box_data_no_edge_sync_vad',
                        help='Path to LDV samples (37 angles × 3 clips)')
    
    # Preprocessing
    parser.add_argument('--n_atoms', type=int, default=8,
                        help='Number of atoms per expert after reduction (default: 8)')
    parser.add_argument('--atom_reduce_mode', type=str, default='kmeans', choices=['kmeans', 'kcenter', 'diverse'],
                        help='Atom reduction: kmeans (centroids), kcenter (farthest-first), or diverse (cos-sep)')
    parser.add_argument('--atom_min_cos', type=float, default=0.98,
                        help='Minimum cosine separation for --atom_reduce_mode diverse (default: 0.98)')
    
    # Model architecture
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer embedding dimension (default: 64, must be < F=346 to reduce parameters)')
    parser.add_argument('--nhead', type=int, default=2,
                        help='Number of attention heads (default: 2, must divide d_model)')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of Transformer layers (default: 1)')
    parser.add_argument('--steps', type=int, default=2,
                        help='Number of OMP steps (default: 2, reduced from 6 to lower task difficulty)')
    parser.add_argument('--top_e', type=int, default=2,
                        help='Top-K experts during inference (default: 2)')
    parser.add_argument('--top_l', type=int, default=2,
                        help='Top-L atoms per expert during inference (default: 2)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Reconstruction loss weight (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.2,
                        help='Monotonicity loss weight (default: 0.2)')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Classification loss weight (default: 0.5)')
    # Routing source selection
    parser.add_argument('--routing_mode', type=str, default='qk', choices=['qk', 'g', 'hybrid'],
                        help="Routing score source: 'qk' (Transformer QK), 'g' (physics correlation), 'hybrid' (blend)")
    parser.add_argument('--hybrid_alpha', type=float, default=0.5,
                        help='Hybrid blend weight for g (0..1); 1.0 = pure g, 0.0 = pure qk')
    # Routing activation per level
    parser.add_argument('--routing_activation_e', type=str, default='gumbel', choices=['gumbel', 'softmax', 'entmax', 'sparsemax'],
                        help='Activation for expert-level routing (default: gumbel)')
    parser.add_argument('--routing_activation_a', type=str, default='gumbel', choices=['gumbel', 'softmax', 'entmax', 'sparsemax'],
                        help='Activation for atom-level routing (default: gumbel)')
    # Score normalization and regularization
    parser.add_argument('--score_norm', type=str, default='none', choices=['none', 'std'],
                        help='Normalization for routing scores before softmax (default: none)')
    parser.add_argument('--score_reg_weight', type=float, default=0.0,
                        help='L2 regularization weight for routing scores (default: 0.0)')
    # Expert aggregation and cross-attention toggles
    parser.add_argument('--expert_agg', type=str, default='l2', choices=['l2', 'max', 'mean_relu'],
                        help='Aggregate atom scores into expert logits: l2, max, or mean_relu (default: l2)')
    parser.add_argument('--d_can_attend_r', action='store_true',
                        help='Allow dictionary tokens to attend the residual token (adds R to D attention)')
    parser.add_argument('--align_weight', type=float, default=0.0,
                        help='Weight for alignment loss between QK scores and |g| energy (default: 0.0)')
    parser.add_argument('--probe_grad_split', action='store_true',
                        help='If set, run CE-only and REC-only gradient probes on the first sample to log grad norms.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='If set, freeze Transformer encoder parameters (no gradient)')
    parser.add_argument('--init_qk_identity', action='store_true',
                        help='If set and d_model==F, initialize Wq/Wk to identity for g-like scoring')
    parser.add_argument('--distill_T', type=float, default=1.0,
                        help='Distillation temperature T for soft targets from |g| (default: 1.0)')
    parser.add_argument('--distill_weight', type=float, default=0.0,
                        help='Weight for distillation KL loss (student=QK, teacher=|g|)')
    # Signal-preserving toggles
    parser.add_argument('--no_type_bias', action='store_true',
                        help='Remove type_R/type_D biases from token construction')
    parser.add_argument('--encoder_identity', action='store_true',
                        help='Bypass Transformer encoder (identity mapping)')
    parser.add_argument('--single_gate_expert', action='store_true',
                        help='Use only expert-level gating (atoms uniformly weighted)')
    parser.add_argument('--score_center_atoms', action='store_true',
                        help='Center atom-level scores per expert before routing')
    parser.add_argument('--score_center_expert', action='store_true',
                        help='Center expert-level scores before routing')
    parser.add_argument('--supervise_steps', type=str, default='first', choices=['first', 'all'],
                        help='Supervise routing scores from first K steps or all steps (default: first)')
    parser.add_argument('--supervise_k', type=int, default=1,
                        help='Number of initial steps to supervise when supervise_steps=first (default: 1)')
    parser.add_argument('--w_e_entropy_penalty', type=float, default=0.0,
                        help='Entropy penalty weight on expert routing distribution w_e (default: 0.0)')
    parser.add_argument('--nce_weight', type=float, default=0.0,
                        help='Weight for step-wise InfoNCE loss (anchor=step0, positive=step1)')
    parser.add_argument('--nce_T', type=float, default=1.0,
                        help='Temperature for InfoNCE (default: 1.0)')
    # Routing temperature annealing
    parser.add_argument('--tau_e_start', type=float, default=1.0,
                        help='Initial expert temperature for routing (default: 1.0)')
    parser.add_argument('--tau_e_end', type=float, default=0.2,
                        help='Final expert temperature for routing (default: 0.2)')
    parser.add_argument('--tau_a_start', type=float, default=1.0,
                        help='Initial atom temperature for routing (default: 1.0)')
    parser.add_argument('--tau_a_end', type=float, default=0.2,
                        help='Final atom temperature for routing (default: 0.2)')
    parser.add_argument('--tau_anneal_epochs', type=int, default=30,
                        help='Epochs over which to anneal temperatures linearly (default: 30)')
    # Teacher warm-up
    parser.add_argument('--teacher_warmup_epochs', type=int, default=10,
                        help='Number of initial epochs to include teacher CE from |g| per-angle (default: 10)')
    parser.add_argument('--teacher_weight', type=float, default=0.5,
                        help='Weight of teacher CE loss during warm-up (default: 0.5)')
    
    # Output
    parser.add_argument('--out_dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device: cpu|mps|cuda')
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"results/omp_transformer_ldv_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("Transformer Routed Soft-OMP with Real LDV Data")
    print("=" * 80)
    print(f"Output directory: {args.out_dir}")
    
    # ========================================================================
    # STEP 1: Load and preprocess data
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 1: Data Loading and Preprocessing")
    print("=" * 80)
    
    # Load raw matrices
    H_raw, W_raw, angles = load_raw_ldv_matrices(args.h_path, args.w_path, device='cpu')
    
    # Reduce atoms (50 → n_atoms)
    if args.atom_reduce_mode == 'kmeans':
        W_reduced, kmeans_labels, kmeans_model = reduce_atoms_kmeans(W_raw, n_clusters=args.n_atoms)
    elif args.atom_reduce_mode == 'kcenter':
        W_reduced, kmeans_labels, kmeans_model = reduce_atoms(W_raw, mode='kcenter', n_clusters=args.n_atoms)
    else:  # 'diverse'
        W_reduced, kmeans_labels, kmeans_model = reduce_atoms(W_raw, mode='diverse', n_clusters=args.n_atoms, min_cos=args.atom_min_cos)
    
    # NO PCA - use full frequency resolution (F=346)
    print(f"\n=== Using Full Frequency Resolution (NO PCA) ===")
    print(f"H shape: {H_raw.shape}")  # (346, 37)
    print(f"W shape: {W_reduced.shape}")  # (346, 8)
    print(f"Rationale: PCA may discard low-variance but high-discriminability features")
    print(f"           Greedy achieves 83.8% with F=346; test if Transformer needs full info too")
    
    H_final = H_raw
    W_final = W_reduced
    
    # Build dictionary
    D, idx2angle = build_dictionary(H_final, W_final, angles)
    
    # Load samples
    Y_samples, labels, metadata = load_ldv_samples(
        args.dataset_root, H_final, W_final, angles, device='cpu'
    )
    
    # Compute dataset fingerprint
    fingerprint = compute_dataset_fingerprint(args.dataset_root)
    
    # ========================================================================
    # STEP 2: Build model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: Model Initialization")
    print("=" * 80)
    
    F = H_final.shape[0]  # Full frequency dimension: 346
    E = len(angles)  # 37
    M = args.n_atoms  # 8
    d = args.d_model  # 64 (default)
    
    # Adjust nhead to divide d_model (not F)
    # d=64, divisors: 1, 2, 4, 8, 16, 32, 64
    # Use nhead=2 by default
    if args.d_model % args.nhead != 0:
        # Find a suitable divisor of d_model
        for candidate in [2, 4, 8, 1]:
            if args.d_model % candidate == 0:
                nhead = candidate
                break
        else:
            nhead = 1
        print(f"  Auto-adjusted nhead to {nhead} (must divide d_model={args.d_model})")
    else:
        nhead = args.nhead
    
    print(f"\n=== Parameter Reduction Strategy ===")
    print(f"  Input frequency dimension F: {F}")
    print(f"  Transformer embedding dimension d: {d}")
    print(f"  Ratio d/F: {d/F:.3f}")
    print(f"  Token projection size: 2 × ({F} × {d}) = {2*F*d:,} parameters")
    print(f"  (vs naive F×F: 2 × ({F} × {F}) = {2*F*F:,} parameters)")
    print(f"  Parameter reduction: {100*(1 - d/F):.1f}%")
    
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=d, nhead=nhead, nlayers=args.nlayers,
        steps=args.steps, top_e=args.top_e, L=args.top_l,
        tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel',
        routing_mode=args.routing_mode, hybrid_alpha=args.hybrid_alpha,
        routing_e=args.routing_activation_e, routing_a=args.routing_activation_a,
        score_norm_mode=args.score_norm, score_reg_weight=args.score_reg_weight
    ).to(args.device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print('Freeze: encoder parameters frozen (no gradient)')

    # Optional: initialize Wq/Wk to identity when d_model matches input F
    if args.init_qk_identity and d == F:
        with torch.no_grad():
            if model.Wq.weight.shape[0] == model.Wq.weight.shape[1] == d:
                model.Wq.weight.copy_(torch.eye(d, device=model.Wq.weight.device))
            if model.Wk.weight.shape[0] == model.Wk.weight.shape[1] == d:
                model.Wk.weight.copy_(torch.eye(d, device=model.Wk.weight.device))
        print('Init: Wq/Wk initialized to identity (g-like start)')

    # Apply signal-preserving toggles
    model.no_type_bias = bool(getattr(args, 'no_type_bias', False))
    model.encoder_identity = bool(getattr(args, 'encoder_identity', False))
    model.single_gate_expert = bool(getattr(args, 'single_gate_expert', False))
    model.score_center_atoms = bool(getattr(args, 'score_center_atoms', False))
    model.score_center_expert = bool(getattr(args, 'score_center_expert', False))
    model.expert_agg = getattr(args, 'expert_agg', 'l2')
    model.d_can_attend_r = bool(getattr(args, 'd_can_attend_r', False))
    
    print(f"\nModel architecture:")
    print(f"  F (input freq dim): {F}")
    print(f"  d_model (embedding dim): {d}")
    print(f"  E (experts/angles): {E}")
    print(f"  M (atoms/expert): {M}")
    print(f"  P (total atoms): {E * M}")
    print(f"  nhead: {nhead}")
    print(f"  nlayers: {args.nlayers}")
    print(f"  routing_mode: {args.routing_mode} (hybrid_alpha={args.hybrid_alpha if args.routing_mode=='hybrid' else 'n/a'})")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
    
    # ========================================================================
    # STEP 3: Training
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 3: Training")
    print("=" * 80)
    
    train_history = []
    best_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Snapshot parameters to compute per-epoch parameter deltas after updates
        prev_Wq = model.Wq.weight.detach().clone()
        prev_Wk = model.Wk.weight.detach().clone()
        prev_encoder = [p.detach().clone() for p in model.encoder.parameters() if p.requires_grad]

        # Temperature annealing (linear)
        if args.tau_anneal_epochs > 0:
            t = min(1.0, epoch / float(max(1, args.tau_anneal_epochs)))
            tau_e_cur = args.tau_e_start + t * (args.tau_e_end - args.tau_e_start)
            tau_a_cur = args.tau_a_start + t * (args.tau_a_end - args.tau_a_start)
            with torch.no_grad():
                model.tau_e.data.fill_(float(tau_e_cur))
                model.tau_a.data.fill_(float(tau_a_cur))
        metrics = train_epoch(
            model, D, Y_samples, labels, opt, idx2angle,
            batch_size=args.batch_size, device=args.device,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma, align_weight=args.align_weight,
            distill_T=args.distill_T, distill_weight=args.distill_weight,
            probe_grad_split=args.probe_grad_split,
            supervise_steps=args.supervise_steps, supervise_k=args.supervise_k,
            w_e_entropy_penalty=args.w_e_entropy_penalty,
            nce_weight=args.nce_weight, nce_T=args.nce_T,
            epoch=epoch, diag_path=os.path.join(args.out_dir, 'diagnostics.jsonl'), diag_subset=10**9,
            teacher_warmup_epochs=args.teacher_warmup_epochs, teacher_weight=args.teacher_weight
        )

        train_history.append(metrics)

        # Compute parameter deltas (L2) for Wq/Wk/encoder and append to diagnostics
        try:
            wq_delta = float((model.Wq.weight.detach() - prev_Wq).norm().item())
            wk_delta = float((model.Wk.weight.detach() - prev_Wk).norm().item())
            enc_delta_sq = 0.0
            for p, p_prev in zip([p for p in model.encoder.parameters() if p.requires_grad], prev_encoder):
                d = p.detach() - p_prev
                enc_delta_sq += float((d.norm().item()) ** 2)
            enc_delta = float(enc_delta_sq ** 0.5)
            with open(os.path.join(args.out_dir, 'diagnostics.jsonl'), 'a') as f:
                f.write(json.dumps({'epoch': int(epoch) + 1,
                                    'param_delta': {'Wq': wq_delta, 'Wk': wk_delta, 'encoder': enc_delta}}) + "\n")
        except Exception:
            pass
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_metrics = evaluate(model, D, Y_samples, labels, idx2angle, device=args.device)
            accuracy = eval_metrics['accuracy']
            
            parts = [
                f"Epoch {epoch+1:3d}/{args.epochs}:",
                f"loss={metrics['total_loss']:.4f}",
                f"rec={metrics['rec_loss']:.4f}",
                f"mono={metrics['mono_loss']:.4f}",
                f"class={metrics['class_loss']:.4f}",
                f"acc={accuracy:.3f}",
            ]
            if metrics.get('teacher_acc_subset') is not None:
                parts.append(f"teach={metrics['teacher_acc_subset']:.3f}")
            if metrics.get('qk_g_corr_pearson_mean') is not None:
                parts.append(f"align={metrics['qk_g_corr_pearson_mean']:.3f}")
            parts += [
                f"tau_e={float(model.tau_e.item()):.3f}",
                f"tau_a={float(model.tau_a.item()):.3f}",
                f"eta={float(model.eta.item()):.3f}",
            ]
            print(" ".join(parts))
        
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = epoch + 1
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'accuracy': accuracy,
                    'metrics': eval_metrics
                }, os.path.join(args.out_dir, 'model_best.pth'))
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={metrics['total_loss']:.4f} "
                  f"rec={metrics['rec_loss']:.4f} "
                  f"mono={metrics['mono_loss']:.4f} "
                  f"class={metrics['class_loss']:.4f}")
    
    # ========================================================================
    # STEP 4: Final Evaluation
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 4: Final Evaluation")
    print("=" * 80)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.out_dir, 'model_best.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics = evaluate(model, D, Y_samples, labels, idx2angle, device=args.device)
    
    print(f"\nFinal Results (Best model from epoch {best_epoch}):")
    print(f"  Overall accuracy: {final_metrics['accuracy']:.3f} ({final_metrics['accuracy']*100:.1f}%)")
    print(f"  Baseline (commit b573aa6): 0.838 (83.8%)")
    
    # Per-angle breakdown
    print("\nPer-angle accuracy:")
    for e, acc in enumerate(final_metrics['per_angle_accuracy']):
        angle_deg = int(angles[e].item()) if hasattr(angles[e], 'item') else int(angles[e])
        n_samples = (labels == e).sum().item()
        print(f"  Angle {angle_deg:3d}°: {acc:.3f} ({acc*100:5.1f}%) - {n_samples} samples")
    
    # Identify problematic angles (from b573aa6: 15°, 35°, 50°, 70°, 105°, 120°)
    problematic_angles = [15, 35, 50, 70, 105, 120]
    print("\nProblematic angles from baseline:")
    for angle_deg in problematic_angles:
        if angle_deg in angles:
            e = np.where(angles == angle_deg)[0][0]
            acc = final_metrics['per_angle_accuracy'][e]
            print(f"  {angle_deg:3d}°: {acc:.3f} ({'IMPROVED' if acc > 0.5 else 'STILL PROBLEMATIC'})")
    
    # ========================================================================
    # STEP 5: Visualizations
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 5: Visualization")
    print("=" * 80)
    
    # Training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs_arr = np.arange(1, len(train_history) + 1)
    
    axes[0, 0].plot(epochs_arr, [m['total_loss'] for m in train_history])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(epochs_arr, [m['rec_loss'] for m in train_history], label='Reconstruction')
    axes[0, 1].plot(epochs_arr, [m['mono_loss'] for m in train_history], label='Monotonicity')
    axes[0, 1].plot(epochs_arr, [m['class_loss'] for m in train_history], label='Classification')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = final_metrics['confusion_matrix']
    im = axes[1, 0].imshow(cm, cmap='Blues', aspect='auto')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted Angle Index')
    axes[1, 0].set_ylabel('True Angle Index')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Per-angle accuracy
    axes[1, 1].bar(range(len(angles)), final_metrics['per_angle_accuracy'])
    axes[1, 1].axhline(y=final_metrics['accuracy'], color='r', linestyle='--', label=f'Overall: {final_metrics["accuracy"]:.3f}')
    axes[1, 1].axhline(y=0.838, color='g', linestyle='--', label='Baseline: 0.838')
    axes[1, 1].set_title('Per-Angle Accuracy')
    axes[1, 1].set_xlabel('Angle Index')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, 'results.png')
    plt.savefig(fig_path, dpi=150)
    print(f"Saved figure: {fig_path}")
    
    # ========================================================================
    # STEP 6: Save results
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 6: Saving Results")
    print("=" * 80)
    
    # Save metrics
    np.savez(
        os.path.join(args.out_dir, 'metrics.npz'),
        train_history=[{k: float(v) for k, v in m.items()} for m in train_history],
        confusion_matrix=final_metrics['confusion_matrix'],
        per_angle_accuracy=final_metrics['per_angle_accuracy'],
        predictions=final_metrics['predictions'],
        labels=labels.cpu().numpy(),
        angles=angles,
        best_epoch=best_epoch,
        best_accuracy=best_accuracy
    )
    
    # Save code state
    code_state = {
        'git_head': os.popen('git rev-parse HEAD').read().strip(),
        'git_dirty': bool(os.popen('git diff --quiet').close()),
        'script': __file__,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args),
        'dataset_fingerprint': fingerprint
    }
    
    with open(os.path.join(args.out_dir, 'code_state.json'), 'w') as f:
        json.dump(code_state, f, indent=2)
    
    # Save preprocessing artifacts (NO PCA)
    torch.save({
        'H': H_final,
        'W': W_final,
        'D': D,
        'idx2angle': idx2angle,
        'angles': angles,
        'kmeans_labels': kmeans_labels
    }, os.path.join(args.out_dir, 'preprocessing.pth'))
    
    print(f"All results saved to: {args.out_dir}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
