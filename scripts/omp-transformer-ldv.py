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

        # Diagnostics controls (non-breaking): when enabled, forward() stores step-0 signals
        self.enable_diag: bool = False
        self.last_diag = None
    
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
        
        # reset diagnostics snapshot each call
        if self.enable_diag:
            self.last_diag = None

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
                # NOTE: use learnable temperatures directly (no .item()) so they can get gradients / schedules
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)  # (E,)
                w_all = torch.zeros(self.E, self.M, device=D.device)
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_atoms[e].abs(), self.tau_a, self.routing, hard=False)
                    w_all[e] = w_e[e] * w_a_e
                w_all = w_all.reshape(-1)  # (P,)
                # Snapshot diagnostics at step 0
                if self.enable_diag and step == 0:
                    self.last_diag = {
                        'scores_expert': scores_expert.detach(),
                        'w_e': w_e.detach(),
                    }
                g = (D.T @ r)  # (P,) gradient-like signal
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
                if self.enable_diag and step == 0:
                    self.last_diag = {
                        'scores_expert': scores_expert.detach(),
                    }
                if len(chosen_idx) > 0:
                    g = (D.T @ r)
                    x[chosen_idx] = x[chosen_idx] + self.eta * g[chosen_idx]
            
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
                alpha: float = 1.0, beta: float = 0.2, gamma: float = 0.5,
                epoch: int | None = None, diag_path: str | None = None, diag_subset: int = 16):
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
    
    # Diagnostics accumulators (subset)
    diag_seen = 0
    teacher_correct = 0
    teacher_margins = []
    qk_g_corrs = []
    qk_top1_matches = 0
    w_e_entropies = []

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
            # enable one-shot diagnostics capture for first few samples
            model.enable_diag = (diag_seen < diag_subset)
            x_hat, r_curve = model(yb[b], D.to(device), train_mode=True)
            y_hat = D.to(device) @ x_hat
            
            # Reconstruction loss
            rec_loss = rec_loss + F.mse_loss(y_hat, yb[b])
            
            # Monotonicity loss
            rc = torch.tensor(r_curve, device=device)
            diffs = rc[1:] - rc[:-1]
            mono_loss = mono_loss + torch.relu(diffs).sum()
            
            # Classification loss: predict angle from x_hat
            # IMPORTANT: avoid sign cancellation — use magnitude aggregation like the greedy baseline
            x_by_expert = x_hat.reshape(model.E, model.M).abs().sum(dim=1)  # (E,)
            logits = x_by_expert
            class_loss = class_loss + F.cross_entropy(logits.unsqueeze(0), lb[b].unsqueeze(0))

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
        
        # Average over batch
        rec_loss /= yb.size(0)
        mono_loss /= yb.size(0)
        class_loss /= yb.size(0)
        
        # Total loss
        loss = alpha * rec_loss + beta * mono_loss + gamma * class_loss
        
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        
        # Record
        rec_losses.append(rec_loss.item())
        mono_losses.append(mono_loss.item())
        class_losses.append(class_loss.item())
        total_losses.append(loss.item())
    
    # Persist per-epoch diagnostics JSONL
    if diag_path is not None and epoch is not None:
        try:
            rec = {
                'epoch': int(epoch) + 1,
                'rec_loss': float(np.mean(rec_losses)) if rec_losses else None,
                'mono_loss': float(np.mean(mono_losses)) if mono_losses else None,
                'class_loss': float(np.mean(class_losses)) if class_losses else None,
                'total_loss': float(np.mean(total_losses)) if total_losses else None,
                'teacher_samples': int(diag_seen),
                'teacher_acc_subset': float(teacher_correct / max(1, diag_seen)),
                'teacher_margin_p50': float(np.median(teacher_margins)) if teacher_margins else None,
                'teacher_margin_p95': float(np.percentile(teacher_margins, 95)) if teacher_margins else None,
                'qk_g_corr_pearson_mean': float(np.mean(qk_g_corrs)) if qk_g_corrs else None,
                'qk_top1_match_rate': float(qk_top1_matches / max(1, diag_seen)),
                'w_e_entropy_mean': float(np.mean(w_e_entropies)) if w_e_entropies else None,
                'tau_e': float(model.tau_e.item()) if isinstance(model.tau_e, torch.Tensor) else None,
                'tau_a': float(model.tau_a.item()) if isinstance(model.tau_a, torch.Tensor) else None,
                'eta': float(model.eta.item()) if isinstance(model.eta, torch.Tensor) else None,
                'grad_norms': grad_norms,
            }
            with open(diag_path, 'a') as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    return {
        'rec_loss': np.mean(rec_losses),
        'mono_loss': np.mean(mono_losses),
        'class_loss': np.mean(class_losses),
        'total_loss': np.mean(total_losses),
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
        
        # Predict angle from x_hat (magnitude aggregation to avoid sign cancellation)
        x_by_expert = x_hat.reshape(E, model.M).abs().sum(dim=1)  # (E,)
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
        tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel'
    ).to(args.device)
    
    print(f"\nModel architecture:")
    print(f"  F (input freq dim): {F}")
    print(f"  d_model (embedding dim): {d}")
    print(f"  E (experts/angles): {E}")
    print(f"  M (atoms/expert): {M}")
    print(f"  P (total atoms): {E * M}")
    print(f"  nhead: {nhead}")
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
            alpha=args.alpha, beta=args.beta, gamma=args.gamma,
            epoch=epoch, diag_path=os.path.join(args.out_dir, 'diagnostics.jsonl'), diag_subset=16
        )
        
        train_history.append(metrics)
        
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
