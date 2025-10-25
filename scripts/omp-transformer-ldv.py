#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer Routed Soft-OMP with Real LDV Data
- Uses real H matrix (37 angles, Original→Box transfer functions)
- Uses real W matrix (50-atom USM from 111 speakers)
- Preprocessing: K-means atom reduction (50→8) + PCA frequency reduction (346→64)
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
from sklearn.decomposition import PCA
from scipy import signal as scipy_signal

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


def reduce_frequency_pca(H: torch.Tensor, W: torch.Tensor, n_components: int = 64, random_state: int = 42):
    """
    Reduce frequency dimension from 346 → n_components using PCA on [H | W].
    
    Args:
        H: (F, E) where F=346, E=37 angles
        W: (F, M) where F=346, M=8 atoms (after K-means)
        n_components: Target frequency dimension (default: 64)
    
    Returns:
        H_pca: (n_components, E)
        W_pca: (n_components, M)
        pca: fitted PCA object
    """
    print(f"\n=== Frequency Reduction via PCA ===")
    print(f"Input: H {H.shape}, W {W.shape}")
    
    # Concatenate for joint PCA
    HW = torch.cat([H, W], dim=1).cpu().numpy()  # (346, 37+8=45)
    print(f"Joint matrix [H|W]: {HW.shape}")
    
    # Adjust n_components if necessary (must be <= min(n_samples, n_features))
    n_samples_pca = HW.shape[1]  # 45 (angles + atoms)
    n_features_pca = HW.shape[0]  # 346 (freq bins)
    max_components = min(n_samples_pca, n_features_pca)
    
    if n_components > max_components:
        print(f"  WARNING: n_components={n_components} > max_components={max_components}")
        print(f"  Adjusting to n_components={max_components}")
        n_components = max_components
    
    print(f"Target: {n_components} frequency components")
    
    # PCA on transposed data (samples are angles+atoms)
    pca = PCA(n_components=n_components, random_state=random_state)
    HW_pca = pca.fit_transform(HW.T).T  # (n_components, 45)
    
    # Split back
    E = H.shape[1]  # 37
    H_pca = HW_pca[:, :E]  # (n_components, 37)
    W_pca = HW_pca[:, E:]  # (n_components, 8)
    
    # Convert to tensors and normalize
    H_pca = torch.from_numpy(H_pca).float()
    W_pca = torch.from_numpy(W_pca).float()
    H_pca = H_pca / (H_pca.norm(dim=0, keepdim=True) + 1e-12)
    W_pca = W_pca / (W_pca.norm(dim=0, keepdim=True) + 1e-12)
    
    # Compute explained variance
    var_explained = pca.explained_variance_ratio_.sum()
    
    print(f"PCA completed:")
    print(f"  H_pca shape: {H_pca.shape}")
    print(f"  W_pca shape: {W_pca.shape}")
    print(f"  Variance explained: {var_explained:.4f} ({var_explained*100:.2f}%)")
    print(f"  Top 5 components: {pca.explained_variance_ratio_[:5]}")
    
    return H_pca, W_pca, pca


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


def load_ldv_samples(dataset_root: str, H_pca: torch.Tensor, W_pca: torch.Tensor, 
                     pca: PCA, angles_target: np.ndarray, device='cpu'):
    """
    Load all LDV samples (37 angles × 3 clips = 111 samples).
    Apply same STFT processing as H estimation, then PCA projection.
    
    Args:
        dataset_root: Path to white_noise_box_data_no_edge_sync_vad/
        H_pca: (F_pca, E) PCA-reduced H for reference
        W_pca: (F_pca, M) PCA-reduced W for reference
        pca: Fitted PCA object from preprocessing
        angles_target: (37,) Expected angles array
        
    Returns:
        Y_samples: (N, F_pca) where N=111, F_pca=45
        labels: (N,) angle indices (0-36)
        metadata: List of dicts with angle_deg, clip_id, file_path
    """
    print(f"\n=== Loading LDV Samples ===")
    print(f"Dataset root: {dataset_root}")
    
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # STFT parameters (matching H estimation from commit dd1e20d)
    # From H matrix metadata: nperseg=2048, noverlap=1536, fs=16000
    fs = 16000
    n_fft = 2048
    hop_length = 512  # noverlap = 2048 - 512 = 1536
    freq_min = 300
    freq_max = 3000
    
    print(f"STFT parameters (matching H matrix):")
    print(f"  fs: {fs}, n_fft: {n_fft}, hop_length: {hop_length}")
    print(f"  noverlap: {n_fft - hop_length}")
    print(f"  Frequency band: [{freq_min}, {freq_max}] Hz")
    
    Y_list = []
    labels = []
    metadata = []
    
    angle_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith('angle_')])
    print(f"Found {len(angle_dirs)} angle directories")
    
    for angle_dir in angle_dirs:
        # Parse angle from directory name
        angle_str = angle_dir.name.split('_')[1]
        angle_deg = int(angle_str)
        
        # Find corresponding index in angles_target
        angle_idx = np.where(angles_target == angle_deg)[0]
        if len(angle_idx) == 0:
            print(f"  WARNING: Angle {angle_deg}° not in target angles, skipping")
            continue
        angle_idx = angle_idx[0]
        
        # Load all .npy files in this directory
        npy_files = sorted(angle_dir.glob('*.npy'))
        
        for npy_file in npy_files:
            # Load audio waveform
            audio = np.load(npy_file)  # (T,) time-domain signal
            
            # Compute STFT
            freqs, times, Zxx = scipy_signal.stft(
                audio,
                fs=fs,
                nperseg=n_fft,
                noverlap=n_fft - hop_length,
                window='hann'
            )
            
            # Get magnitude
            magnitude = np.abs(Zxx)  # (freq_bins, time_frames)
            
            # Apply frequency band mask [300, 3000] Hz
            freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
            magnitude_band = magnitude[freq_mask, :]  # Should be (346, time_frames)
            
            if magnitude_band.shape[0] != 346:
                print(f"  WARNING: Expected 346 freq bins, got {magnitude_band.shape[0]} for {npy_file}")
                print(f"  Frequency range: {freqs[freq_mask].min():.1f} - {freqs[freq_mask].max():.1f} Hz")
                continue
            
            # Time-average to get single spectrum per sample
            y_avg = magnitude_band.mean(axis=1)  # (346,)
            
            # Normalize
            y_torch = torch.from_numpy(y_avg).float()
            y_torch = y_torch / (y_torch.norm() + 1e-12)
            
            # Apply PCA projection (same as H, W)
            y_pca = pca.transform(y_torch.cpu().numpy().reshape(1, -1))[0]  # (n_components,)
            y_pca = torch.from_numpy(y_pca).float()
            y_pca = y_pca / (y_pca.norm() + 1e-12)
            
            Y_list.append(y_pca)
            labels.append(angle_idx)
            metadata.append({
                'angle_deg': angle_deg,
                'angle_idx': angle_idx,
                'clip_id': npy_file.stem,
                'file_path': str(npy_file)
            })
        
        if len(npy_files) > 0:
            print(f"  Angle {angle_deg:3d}° (index {angle_idx:2d}): {len(npy_files)} clips loaded")
    
    if len(Y_list) == 0:
        raise ValueError("No samples loaded! Check dataset_root and frequency band parameters.")
    
    Y_samples = torch.stack(Y_list).to(device)  # (N, F_pca)
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

def gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    """Gumbel-Softmax sampling for differentiable routing."""
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    y = F.softmax((logits + g) / max(tau, 1e-8), dim=-1)
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
                 routing: str = 'gumbel'):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.P = E * M
        self.d = d if d is not None else F
        self.steps, self.top_e, self.L = steps, top_e, L
        
        # Learnable routing parameters
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.routing = routing
        
        # Token projections + type embeddings
        self.P_R = nn.Linear(F, self.d, bias=False)
        self.P_D = nn.Linear(F, self.d, bias=False)
        nn.init.eye_(self.P_R.weight) if self.d == F else nn.init.xavier_uniform_(self.P_R.weight)
        nn.init.eye_(self.P_D.weight) if self.d == F else nn.init.xavier_uniform_(self.P_D.weight)
        self.type_R = nn.Parameter(torch.randn(self.d))
        self.type_D = nn.Parameter(torch.randn(self.d))
        
        # TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d, nhead=nhead, dim_feedforward=4 * self.d, 
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        
        # Query/Key projections
        self.Wq = nn.Linear(self.d, self.d, bias=False)
        self.Wk = nn.Linear(self.d, self.d, bias=False)
    
    def _build_tokens(self, r: torch.Tensor, D: torch.Tensor):
        """Build token sequence: [Residual; Dictionary atoms]."""
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
        return mask
    
    def _soft_picker(self, logits: torch.Tensor, tau: torch.Tensor, mode: str, hard: bool):
        """Apply soft routing function."""
        if mode == 'gumbel':
            return gumbel_softmax(logits, tau=float(tau.item()), hard=hard)
        elif mode == 'entmax':
            return entmax15(logits, dim=-1)
        else:
            return F.softmax(logits / max(float(tau.item()), 1e-8), dim=-1)
    
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
        
        for step in range(self.steps):
            # Build tokens and apply Transformer
            T = self._build_tokens(r, D)  # (1+P, d)
            S = T.size(0)
            mask = self._make_mask(S).to(T.device)
            H = self.encoder(T, mask=mask)  # (1+P, d)
            
            h_R = H[0]  # (d,)
            H_D = H[1:]  # (P, d)
            
            # Compute scores
            scores_atoms = (self.Wk(H_D) @ self.Wq(h_R)) / math.sqrt(self.d)  # (P,)
            scores_atoms = scores_atoms.reshape(self.E, self.M)  # (E, M)
            
            # Expert-level scores (L2 pooling over atoms)
            scores_expert = torch.sqrt((scores_atoms.abs() ** 2).sum(dim=1) + 1e-12)  # (E,)
            
            if train_mode:
                # Soft routing
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)  # (E,)
                w_all = torch.zeros(self.E, self.M, device=D.device)
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_atoms[e].abs(), self.tau_a, self.routing, hard=False)
                    w_all[e] = w_e[e] * w_a_e
                w_all = w_all.reshape(-1)  # (P,)
                g = (D.T @ r)  # (P,) gradient-like signal
                x = x + float(self.eta.item()) * (w_all * g)
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
                if len(chosen_idx) > 0:
                    g = (D.T @ r)
                    x[chosen_idx] = x[chosen_idx] + float(self.eta.item()) * g[chosen_idx]
            
            # Update residual
            r = y - D @ x
            res_curve.append(float(torch.norm(r)))
        
        return x, res_curve


# ============================================================================
# PART 3: Training and Evaluation
# ============================================================================

def train_epoch(model: FullTransformerRoutedSoftOMP, D: torch.Tensor, 
                Y_samples: torch.Tensor, labels: torch.Tensor,
                opt: torch.optim.Optimizer, idx2angle: List[Tuple[float, int]],
                batch_size: int = 16, device='cpu',
                alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.5):
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
    total_losses = []
    
    for start_idx in range(0, N, batch_size):
        end_idx = min(start_idx + batch_size, N)
        batch_indices = indices[start_idx:end_idx]
        
        yb = Y_samples[batch_indices].to(device)
        lb = labels[batch_indices].to(device)
        
        rec_loss = 0.0
        mono_loss = 0.0
        class_loss = 0.0
        
        for b in range(yb.size(0)):
            # Forward pass
            x_hat, r_curve = model(yb[b], D.to(device), train_mode=True)
            y_hat = D.to(device) @ x_hat
            
            # Reconstruction loss
            rec_loss = rec_loss + F.mse_loss(y_hat, yb[b])
            
            # Monotonicity loss
            rc = torch.tensor(r_curve, device=device)
            diffs = rc[1:] - rc[:-1]
            mono_loss = mono_loss + torch.relu(diffs).sum()
            
            # Classification loss: predict angle from x_hat
            # Aggregate x by expert (angle)
            x_by_expert = x_hat.reshape(model.E, model.M).sum(dim=1)  # (E,)
            logits = x_by_expert  # Use expert activations as logits
            class_loss = class_loss + F.cross_entropy(logits.unsqueeze(0), lb[b].unsqueeze(0))
        
        # Average over batch
        rec_loss /= yb.size(0)
        mono_loss /= yb.size(0)
        class_loss /= yb.size(0)
        
        # Total loss
        loss = alpha * rec_loss + beta * mono_loss + gamma * class_loss
        
        # Backward
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Record
        rec_losses.append(rec_loss.item())
        mono_losses.append(mono_loss.item())
        class_losses.append(class_loss.item())
        total_losses.append(loss.item())
    
    return {
        'rec_loss': np.mean(rec_losses),
        'mono_loss': np.mean(mono_losses),
        'class_loss': np.mean(class_losses),
        'total_loss': np.mean(total_losses)
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
        
        # Predict angle from x_hat
        x_by_expert = x_hat.reshape(E, model.M).sum(dim=1)  # (E,)
        pred_idx = x_by_expert.argmax().item()
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
                        help='Number of atoms after K-means clustering (default: 8)')
    parser.add_argument('--pca_dim', type=int, default=45,
                        help='Frequency dimension after PCA (default: 45, max=37+n_atoms)')
    
    # Model architecture
    parser.add_argument('--nhead', type=int, default=3,
                        help='Number of attention heads (default: 3, must divide pca_dim)')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='Number of Transformer layers (default: 1)')
    parser.add_argument('--steps', type=int, default=6,
                        help='Number of OMP steps (default: 6)')
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
    
    # Reduce atoms (50 → 8)
    W_reduced, kmeans_labels, kmeans_model = reduce_atoms_kmeans(W_raw, n_clusters=args.n_atoms)
    
    # Reduce frequency (346 → 64)
    H_pca, W_pca, pca_model = reduce_frequency_pca(H_raw, W_reduced, n_components=args.pca_dim)
    
    # Build dictionary
    D, idx2angle = build_dictionary(H_pca, W_pca, angles)
    
    # Load samples
    Y_samples, labels, metadata = load_ldv_samples(
        args.dataset_root, H_pca, W_pca, pca_model, angles, device='cpu'
    )
    
    # Compute dataset fingerprint
    fingerprint = compute_dataset_fingerprint(args.dataset_root)
    
    # ========================================================================
    # STEP 2: Build model
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("STEP 2: Model Initialization")
    print("=" * 80)
    
    F = args.pca_dim
    E = len(angles)  # 37
    M = args.n_atoms  # 8
    
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=F, nhead=args.nhead, nlayers=args.nlayers,
        steps=args.steps, top_e=args.top_e, L=args.top_l,
        tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel'
    ).to(args.device)
    
    print(f"Model architecture:")
    print(f"  F (freq dim): {F}")
    print(f"  E (experts/angles): {E}")
    print(f"  M (atoms/expert): {M}")
    print(f"  P (total atoms): {E * M}")
    print(f"  d_model: {F}")
    print(f"  nhead: {args.nhead}")
    print(f"  nlayers: {args.nlayers}")
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
        metrics = train_epoch(
            model, D, Y_samples, labels, opt, idx2angle,
            batch_size=args.batch_size, device=args.device,
            alpha=args.alpha, beta=args.beta, gamma=args.gamma
        )
        
        train_history.append(metrics)
        
        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            eval_metrics = evaluate(model, D, Y_samples, labels, idx2angle, device=args.device)
            accuracy = eval_metrics['accuracy']
            
            print(f"Epoch {epoch+1:3d}/{args.epochs}: "
                  f"loss={metrics['total_loss']:.4f} "
                  f"rec={metrics['rec_loss']:.4f} "
                  f"mono={metrics['mono_loss']:.4f} "
                  f"class={metrics['class_loss']:.4f} "
                  f"acc={accuracy:.3f} "
                  f"tau_e={float(model.tau_e.item()):.3f} "
                  f"tau_a={float(model.tau_a.item()):.3f} "
                  f"eta={float(model.eta.item()):.3f}")
            
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
    
    # Save preprocessing artifacts
    torch.save({
        'H_pca': H_pca,
        'W_pca': W_pca,
        'D': D,
        'idx2angle': idx2angle,
        'angles': angles,
        'kmeans_labels': kmeans_labels,
        'pca': pca_model
    }, os.path.join(args.out_dir, 'preprocessing.pth'))
    
    print(f"All results saved to: {args.out_dir}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
