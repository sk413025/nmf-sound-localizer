#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DTMinPointerV2: Atom-Sequence Input for Decision Transformer

Extends DTMinPointer by providing all P dictionary atoms as explicit state tokens.
Per time step t, input sequence: [RTG_tok, R_tok, Atom_0, ..., Atom_{P-1}]
Total sequence length: K × (2 + P) tokens

Based on:
- Design doc: docs/dt_atom_sequence_design.md (commit 0fca4de)
- Baseline: scripts/dt_pointer_ldv.py (DTMinPointer)

Key changes:
1. Forward(): Build sequence [RTG, R, Atoms] × K steps, flatten to (K×seq_len,)
2. Causal mask: Within-step full attention, cross-step causal
3. Pointer head: Query from H_R (residual position), keys from H_atoms (atom positions)
4. Unroll policy: Incremental sequence construction per step
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

# Reuse helper functions from dt_pointer_ldv.py
from dt_pointer_ldv import (
    load_manifest,
    load_trajectories,
    torch_sha256,
    load_H_W,
    build_D,
    reduce_atoms_kmeans,
    reduce_atoms_kmeans_numpy,
    recompute_r_t,
)

from doa_rl.data import DoADataset


# ========== Atom-Sequence Specific Functions ==========

def generate_causal_mask_atom_seq(K: int, seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate causal mask for atom-sequence input.
    
    Args:
        K: Number of time steps
        seq_len: Tokens per step (2 + P)
        device: torch device
    
    Returns:
        mask: (K×seq_len, K×seq_len) with -inf for masked positions
    
    Mask structure:
        - Diagonal blocks (within-step): Full attention (0)
        - Lower triangle blocks (past steps): Full attention (0)
        - Upper triangle blocks (future steps): Masked (-inf)
        
    Example for K=2, seq_len=3:
        Step 0: tokens [0,1,2] can attend to themselves
        Step 1: tokens [3,4,5] can attend to [0,1,2,3,4,5]
        
        Mask (0=attend, -inf=mask):
        [[0, 0, 0, -inf, -inf, -inf],   # token 0
         [0, 0, 0, -inf, -inf, -inf],   # token 1
         [0, 0, 0, -inf, -inf, -inf],   # token 2
         [0, 0, 0,    0,    0,    0],   # token 3
         [0, 0, 0,    0,    0,    0],   # token 4
         [0, 0, 0,    0,    0,    0]]   # token 5
    """
    total_len = K * seq_len
    mask = torch.zeros(total_len, total_len, device=device)
    
    for t in range(K):
        # Current step block: [t×seq_len : (t+1)×seq_len]
        start_t = t * seq_len
        end_t = (t + 1) * seq_len
        
        # Mask future steps (t' > t)
        for t_future in range(t + 1, K):
            start_future = t_future * seq_len
            end_future = (t_future + 1) * seq_len
            mask[start_t:end_t, start_future:end_future] = float('-inf')
    
    return mask


# ========== DTMinPointerV2 Model ==========

class DTMinPointerV2(nn.Module):
    """
    Decision Transformer with atom-sequence input.
    
    Architecture:
        1. Per-step sequence: [RTG_tok, R_tok, Atom_0, ..., Atom_{P-1}]
        2. Full episode: Stack K steps and flatten to (B, K×(2+P), d)
        3. Transformer encoder with causal mask
        4. Pointer head: Query from R position, keys from atom positions
    
    Input:
        R_seq: (B, K, F) - Residual per step
        RTG_seq: (B, K, 2) - RTG targets [resid, acc]
        D: (F, P) - Dictionary atoms
        STEP_seq: (B, K, 2) - Step encoding [t/K, (K-t)/K]
        causal_mask: (K×seq_len, K×seq_len) - Causal mask
    
    Output:
        scores_e: (B, K, E) - Expert scores
        scores_em: (B, K, E, M) - Atom scores per expert
    """
    
    def __init__(
        self,
        F: int,
        E: int,
        M: int,
        d_model: int = 128,
        nhead: int = 2,
        nlayers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.F = F  # Frequency bins
        self.E = E  # Number of experts (angles)
        self.M = M  # Atoms per expert
        self.P = E * M  # Total dictionary atoms
        self.d = d_model
        self.seq_len = 2 + self.P  # RTG + R + P atoms
        
        # Token projections
        self.proj_rtg = nn.Linear(2, d_model)  # RTG [resid, acc] → d
        self.P_R = nn.Linear(F, d_model, bias=False)  # Residual F → d
        self.P_D = nn.Linear(F, d_model, bias=False)  # Dictionary atoms F → d
        
        # Type embeddings (distinguish token types)
        self.type_RTG = nn.Parameter(torch.randn(d_model))
        self.type_R = nn.Parameter(torch.randn(d_model))
        self.type_D = nn.Parameter(torch.randn(d_model))
        
        # LayerNorm before encoder
        self.ln = nn.LayerNorm(d_model)
        
        # Transformer encoder (processes full sequence)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        
        # Pointer head: Query and Key projections
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        
    def forward(
        self,
        R_seq: torch.Tensor,
        RTG_seq: torch.Tensor,
        D: torch.Tensor,
        STEP_seq: torch.Tensor,
        causal_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with atom-sequence input.
        
        Args:
            R_seq: (B, K, F) - Residual vectors per step
            RTG_seq: (B, K, 2) - RTG targets
            D: (F, P) - Dictionary atoms
            STEP_seq: (B, K, 2) - Step encoding (currently unused but kept for compatibility)
            causal_mask: (K×seq_len, K×seq_len) - Causal attention mask
        
        Returns:
            scores_e: (B, K, E) - Expert selection scores
            scores_em: (B, K, E, M) - Atom selection scores per expert
        """
        B, K, F = R_seq.shape
        
        # 1. Build per-step sequences
        # RTG tokens: (B, K, d)
        rtg_tok = self.proj_rtg(RTG_seq) + self.type_RTG
        
        # Residual tokens: (B, K, d)
        r_tok = self.P_R(R_seq) + self.type_R
        
        # Atom tokens: (B, K, P, d)
        # D: (F, P) → D.T: (P, F)
        # Project each atom independently: P_D(D.T) → (P, d)
        # Expand to (B, K, P, d)
        atom_tokens = self.P_D(D.T) + self.type_D  # (P, d)
        atom_tokens = atom_tokens.unsqueeze(0).unsqueeze(0)  # (1, 1, P, d)
        atom_tokens = atom_tokens.expand(B, K, -1, -1)  # (B, K, P, d)
        
        # 2. Concatenate tokens per step: [RTG, R, Atoms]
        # Shape: (B, K, 2+P, d)
        tokens_per_step = torch.cat([
            rtg_tok.unsqueeze(2),      # (B, K, 1, d)
            r_tok.unsqueeze(2),         # (B, K, 1, d)
            atom_tokens                 # (B, K, P, d)
        ], dim=2)  # (B, K, 2+P, d)
        
        # 3. Flatten to sequence: (B, K×(2+P), d)
        seq_flat = tokens_per_step.view(B, K * self.seq_len, self.d)
        
        # 4. Apply LayerNorm and Transformer encoder
        seq_normalized = self.ln(seq_flat)  # (B, K×(2+P), d)
        H = self.encoder(seq_normalized, mask=causal_mask)  # (B, K×(2+P), d)
        
        # 5. Reshape back to per-step structure
        H_steps = H.view(B, K, self.seq_len, self.d)  # (B, K, 2+P, d)
        
        # 6. Extract position-specific hidden states
        H_RTG = H_steps[:, :, 0, :]      # (B, K, d) - RTG position
        H_R = H_steps[:, :, 1, :]        # (B, K, d) - Residual position
        H_atoms = H_steps[:, :, 2:, :]   # (B, K, P, d) - Atom positions
        
        # 7. Pointer head: Compute attention scores
        # Query from residual position (analogous to r in r^T·d_j)
        Q = self.Wq(H_R)  # (B, K, d)
        
        # Keys from atom positions (analogous to d_j)
        K_atoms = self.Wk(H_atoms)  # (B, K, P, d)
        
        # Attention scores: Q·K^T / sqrt(d)
        # Expand Q to match K_atoms: (B, K, 1, d) · (B, K, P, d) → (B, K, P)
        scores = torch.einsum('bkd,bkpd->bkp', Q, K_atoms) / math.sqrt(self.d)  # (B, K, P)
        
        # 8. Reshape to hierarchical structure (E, M)
        scores_em = scores.view(B, K, self.E, self.M)  # (B, K, E, M)
        
        # 9. Expert scores: L2 aggregation over atoms (like teacher qk-model)
        scores_e = torch.sqrt((scores_em.abs() ** 2).sum(dim=3) + 1e-12)  # (B, K, E)
        
        return scores_e, scores_em


# ========== Inference (Unroll Policy) ==========

@torch.no_grad()
def unroll_policy_v2(
    model: DTMinPointerV2,
    D: torch.Tensor,
    y: torch.Tensor,
    K: int,
    rtg_resid_target: float,
    rtg_acc_target: float,
    device: torch.device
) -> Tuple[List[Tuple[int, int, int]], int, torch.Tensor]:
    """
    Greedy unroll with atom-sequence input.
    
    At each step t:
    1. Recompute r_t from y and actions[:t]
    2. Build sequences up to step t (incremental history)
    3. Forward pass with current history
    4. Select action from step t predictions
    5. Update actions list
    
    Args:
        model: DTMinPointerV2 instance
        D: (F, P) dictionary
        y: (F,) observation
        K: Number of steps
        rtg_resid_target: RTG target for residual
        rtg_acc_target: RTG target for accuracy
        device: torch device
    
    Returns:
        em_list: List of (expert, atom, dict_index) per step
        e_pred: Final predicted angle (expert index)
        y_hat: Final reconstruction
    """
    E = model.E
    M = model.M
    F = model.F
    
    actions: List[int] = []
    em_list: List[Tuple[int, int, int]] = []
    
    y = y.to(device)
    D = D.to(device)
    
    # Helper to estimate p_true from y_hat
    def p_true_from_yhat(y_hat: torch.Tensor) -> float:
        g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)
        p = torch.softmax(g_hat, dim=0)
        return float(p.max().item())
    
    # Incremental unroll
    for t in range(K):
        # Recompute residual from current actions
        r_t = recompute_r_t(y, D, actions)
        
        # Build sequences for steps 0..t
        R_history = []
        RTG_history = []
        
        for tau in range(t + 1):
            r_tau = recompute_r_t(y, D, actions[:tau])
            
            # Estimate current y_hat and p_true for RTG computation
            if actions[:tau]:
                y_hat_tau = D[:, actions[:tau]] @ torch.linalg.lstsq(
                    D[:, actions[:tau]], y
                ).solution
            else:
                y_hat_tau = torch.zeros_like(y)
            
            p_est = p_true_from_yhat(y_hat_tau)
            
            # Compute RTG targets
            rtg_r = max(0.0, float((r_tau @ r_tau).item()) - rtg_resid_target)
            rtg_a = max(0.0, rtg_acc_target - p_est)
            
            R_history.append(r_tau)
            RTG_history.append(torch.tensor([rtg_r, rtg_a], dtype=torch.float32, device=device))
        
        # Stack into batch dimension 1 (single episode)
        R_seq = torch.stack(R_history).unsqueeze(0)       # (1, t+1, F)
        RTG_seq = torch.stack(RTG_history).unsqueeze(0)   # (1, t+1, 2)
        STEP_seq = torch.zeros(1, t + 1, 2, device=device)  # Placeholder (not used in v2)
        
        # Generate causal mask for current history length
        cmask = generate_causal_mask_atom_seq(t + 1, model.seq_len, device)
        
        # Forward pass
        scores_e, scores_em = model(R_seq, RTG_seq, D, STEP_seq, cmask)
        
        # Extract predictions for step t (last step in history)
        e = int(scores_e[0, t].argmax().item())
        m = int(scores_em[0, t, e].argmax().item())
        j = e * M + m
        
        # Avoid duplicates (pick next best atom in same expert)
        if j in actions:
            scores_em_filtered = scores_em[0, t, e].clone()
            scores_em_filtered[m] = -1e9
            m = int(scores_em_filtered.argmax().item())
            j = e * M + m
        
        actions.append(j)
        em_list.append((e, m, j))
    
    # Final reconstruction and angle prediction
    if actions:
        x_S = torch.linalg.lstsq(D[:, actions], y).solution
        y_hat = D[:, actions] @ x_S
    else:
        y_hat = torch.zeros_like(y)
    
    g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)
    e_pred = int(g_hat.argmax().item())
    
    return em_list, e_pred, y_hat


# ========== Main Training Loop ==========

def main():
    ap = argparse.ArgumentParser(description='DTMinPointerV2: Atom-sequence input DT trainer')
    
    # Data and output
    ap.add_argument('--traj_dir', type=str, default='results/dt_traj_g_full',
                    help='Directory containing trajectories.jsonl and manifest.json')
    ap.add_argument('--out_dir', type=str, default='results/dt_v2_g_full',
                    help='Output directory for checkpoints and logs')
    
    # Training hyperparameters
    ap.add_argument('--epochs', type=int, default=480,
                    help='Number of training epochs')
    ap.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training')
    ap.add_argument('--lr', type=float, default=3e-3,
                    help='Learning rate')
    
    # Model architecture
    ap.add_argument('--d_model', type=int, default=128,
                    help='Model embedding dimension')
    ap.add_argument('--nhead', type=int, default=2,
                    help='Number of attention heads')
    ap.add_argument('--nlayers', type=int, default=1,
                    help='Number of Transformer encoder layers')
    ap.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate')
    
    # Device
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'],
                    help='Device for training')
    
    # Train/test split
    ap.add_argument('--test_split', type=float, default=0.15,
                    help='Fraction of samples for test set')
    ap.add_argument('--split_seed', type=int, default=42,
                    help='Random seed for train/test split')
    
    # Subset for smoke test
    ap.add_argument('--subset_angles', type=str, default='',
                    help='Comma-separated angle indices for subset (e.g., "0,5,10")')
    
    args = ap.parse_args()
    
    # Setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    print('=' * 80)
    print('DTMinPointerV2: Atom-Sequence Input Decision Transformer')
    print('=' * 80)
    print(f"Trajectory dir: {args.traj_dir}")
    print(f"Output dir: {out_dir}")
    print(f"Device: {device}")
    print()
    
    # Load manifest and trajectories
    manifest = load_manifest(args.traj_dir)
    trajs = load_trajectories(args.traj_dir)
    
    # Subset filtering (for smoke tests)
    if args.subset_angles:
        subset_indices = [int(x) for x in args.subset_angles.split(',')]
        trajs = [t for t in trajs if int(t['angle_index']) in subset_indices]
        print(f"Subset filter: Using {len(trajs)} samples from angles {subset_indices}")
    
    # Rebuild dictionary from manifest
    angles = manifest['angles']
    H_raw = torch.load(manifest['h_path'], map_location='cpu', weights_only=False)['H'].float()
    w_data = torch.load(manifest['w_path'], map_location='cpu', weights_only=False)
    W_full = w_data['W'].float() if isinstance(w_data, dict) and 'W' in w_data else w_data.float()
    
    # Reconstruct reduced W from manifest
    ar = manifest.get('atom_reduce', {})
    sel = ar.get('selected_indices', None)
    mode = ar.get('mode', 'kcenter')
    M = int(manifest.get('M', 8))
    
    if sel is not None:
        W = W_full[:, sel].clone()
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    else:
        if mode == 'kmeans':
            W = reduce_atoms_kmeans_numpy(W_full, n_clusters=M, random_state=manifest.get('seed', 42))
        else:
            W = W_full[:, :M].clone()
            W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    
    H, W = H_raw, W
    D, E, M = build_D(H, W, angles)
    F_bins = D.shape[0]
    P = E * M
    
    print(f"Dictionary: F={F_bins}, E={E}, M={M}, P={P}")
    print(f"Sequence length per step: {2 + P} (RTG + R + {P} atoms)")
    print(f"Total sequence per episode: K×(2+P) = {manifest['K']}×{2+P} = {manifest['K'] * (2 + P)} tokens")
    print()
    
    # Load dataset for y reconstruction
    ds = DoADataset(
        manifest['dataset_root'],
        angles,
        fs=manifest['fs'],
        n_fft=manifest['n_fft'],
        window='hann',
        freq_min=manifest['freq_min'],
        freq_max=manifest['freq_max']
    )
    
    index_by_path = {str(p): i for i, (p, _, _) in enumerate(ds.index)}
    
    def load_y_for_path(path: str) -> torch.Tensor:
        i = index_by_path[path]
        sample = ds[i]
        Y = sample['Y']
        if Y.shape[0] != F_bins:
            raise RuntimeError(f"Y.F mismatch for {path}: {Y.shape[0]} vs D.F={F_bins}")
        y = Y.mean(dim=1).float()
        y = y / (y.norm() + 1e-12)
        return y
    
    # Build training samples
    K = manifest['K']
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    
    for obj in trajs:
        path = obj['path']
        angle_idx = int(obj['angle_index'])
        steps = obj['steps']
        
        y = load_y_for_path(path)
        actions_prev: List[int] = []
        
        R_list = []
        RTG_list = []
        STEP_list = []
        lab_e = []
        lab_m = []
        
        for t, s in enumerate(steps):
            r_t = recompute_r_t(y, D, actions_prev)
            
            # Inputs
            R_list.append(r_t.unsqueeze(0))
            RTG_list.append(torch.tensor([[float(s['rtg_resid']), float(s['rtg_acc'])]], dtype=torch.float32))
            STEP_list.append(torch.tensor([[t / K, (K - t) / K]], dtype=torch.float32))
            
            # Labels
            e = int(s['expert'])
            m = int(s['atom'])
            j = int(s['dict_index'])
            lab_e.append(torch.tensor([e], dtype=torch.long))
            lab_m.append(torch.tensor([m], dtype=torch.long))
            
            actions_prev.append(j)
        
        R_seq = torch.cat(R_list, dim=0)       # (K, F)
        RTG_seq = torch.cat(RTG_list, dim=0)   # (K, 2)
        STEP_seq = torch.cat(STEP_list, dim=0) # (K, 2)
        lab_e = torch.cat(lab_e, dim=0)        # (K,)
        lab_m = torch.cat(lab_m, dim=0)        # (K,)
        
        samples.append((R_seq, RTG_seq, STEP_seq, lab_e, lab_m))
    
    # Train/test split (stratified by angle)
    print(f"{'='*80}\nTrain/Test Split\n{'='*80}")
    n_samples = len(samples)
    
    from collections import defaultdict
    angle_groups = defaultdict(list)
    for i, obj in enumerate(trajs):
        angle_idx = int(obj['angle_index'])
        angle_groups[angle_idx].append(i)
    
    rng = np.random.RandomState(args.split_seed)
    train_indices = []
    test_indices = []
    
    for angle_idx, indices in sorted(angle_groups.items()):
        n_angle = len(indices)
        n_test_angle = max(1, int(n_angle * args.test_split))
        shuffled = rng.permutation(indices).tolist()
        test_indices.extend(shuffled[:n_test_angle])
        train_indices.extend(shuffled[n_test_angle:])
    
    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]
    
    print(f"Total samples: {n_samples}")
    print(f"Train samples: {len(train_samples)} ({len(train_samples)/n_samples*100:.1f}%)")
    print(f"Test samples:  {len(test_samples)} ({len(test_samples)/n_samples*100:.1f}%)")
    print(f"Train steps: {len(train_samples) * K}")
    print(f"Test steps:  {len(test_samples) * K}")
    print()
    
    # Data loader
    def batch_iter(sample_list: List, batch_size: int, shuffle: bool = True):
        if shuffle:
            idx = np.random.permutation(len(sample_list))
        else:
            idx = np.arange(len(sample_list))
        for i in range(0, len(idx), batch_size):
            batch = [sample_list[j] for j in idx[i:i+batch_size]]
            R_b = torch.stack([b[0] for b in batch], dim=0)      # (B, K, F)
            RTG_b = torch.stack([b[1] for b in batch], dim=0)    # (B, K, 2)
            STEP_b = torch.stack([b[2] for b in batch], dim=0)   # (B, K, 2)
            E_b = torch.stack([b[3] for b in batch], dim=0)      # (B, K)
            M_b = torch.stack([b[4] for b in batch], dim=0)      # (B, K)
            yield R_b, RTG_b, STEP_b, E_b, M_b
    
    # Initialize model
    model = DTMinPointerV2(
        F=F_bins,
        E=E,
        M=M,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        dropout=args.dropout
    ).to(device)
    
    D = D.to(device)
    
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Causal mask (same for all batches)
    causal = generate_causal_mask_atom_seq(K, model.seq_len, device)
    
    print(f"{'='*80}\nModel Architecture\n{'='*80}")
    print(f"Embedding dim: {args.d_model}")
    print(f"Attention heads: {args.nhead}")
    print(f"Encoder layers: {args.nlayers}")
    print(f"Sequence length: {K * model.seq_len} tokens")
    print(f"Causal mask shape: {causal.shape}")
    print()
    
    # Evaluation function
    def evaluate(sample_list: List, split_name: str = 'eval'):
        model.eval()
        total_loss = 0.0
        n_batches = 0
        correct_e = 0
        total_e = 0
        correct_a = 0
        total_a = 0
        
        n_experts = model.E
        n_atoms = model.M
        
        with torch.no_grad():
            for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(sample_list, batch_size=len(sample_list), shuffle=False):
                R_b = R_b.to(device)
                RTG_b = RTG_b.to(device)
                STEP_b = STEP_b.to(device)
                E_b = E_b.to(device)
                M_b = M_b.to(device)
                
                # Forward pass
                scores_e, scores_em = model(R_b, RTG_b, D, STEP_b, causal_mask=causal)
                
                # Expert loss
                loss_e = F.cross_entropy(scores_e.reshape(-1, n_experts), E_b.reshape(-1))
                
                # Atom loss (gather expert-specific atom scores)
                idx = E_b.reshape(-1)
                se_flat = scores_em.reshape(-1, n_experts, n_atoms)
                se_sel = se_flat.gather(1, idx.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)
                loss_a = F.cross_entropy(se_sel, M_b.reshape(-1))
                
                loss = loss_e + loss_a
                total_loss += float(loss.item())
                n_batches += 1
                
                # Accuracy
                pred_e = scores_e.argmax(dim=-1)
                correct_e += int((pred_e == E_b).sum().item())
                total_e += int(E_b.numel())
                
                pred_a = se_sel.argmax(dim=-1).view(E_b.shape)
                correct_a += int((pred_a == M_b).sum().item())
                total_a += int(M_b.numel())
        
        avg_loss = total_loss / max(n_batches, 1)
        acc_e = correct_e / total_e if total_e > 0 else 0.0
        acc_a = correct_a / total_a if total_a > 0 else 0.0
        
        return avg_loss, acc_e, acc_a
    
    # Training loop
    print(f"{'='*80}\nTraining Start\n{'='*80}")
    best_test_loss = float('inf')
    best_epoch = 0
    
    n_experts = model.E
    n_atoms = model.M
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        total_loss = 0.0
        n_batches = 0
        
        for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(train_samples, args.batch_size, shuffle=True):
            R_b = R_b.to(device)
            RTG_b = RTG_b.to(device)
            STEP_b = STEP_b.to(device)
            E_b = E_b.to(device)
            M_b = M_b.to(device)
            
            # Forward pass
            scores_e, scores_em = model(R_b, RTG_b, D, STEP_b, causal_mask=causal)
            
            # Loss: CE over steps
            loss_e = F.cross_entropy(scores_e.reshape(-1, n_experts), E_b.reshape(-1))
            
            # Atom CE: select expert row per sample/step
            idx = E_b.reshape(-1)
            se_flat = scores_em.reshape(-1, n_experts, n_atoms)
            se_sel = se_flat.gather(1, idx.view(-1, 1, 1).expand(-1, 1, n_atoms)).squeeze(1)
            loss_a = F.cross_entropy(se_sel, M_b.reshape(-1))
            
            loss = loss_e + loss_a
            
            # Optimization step
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            total_loss += float(loss.item())
            n_batches += 1
        
        train_loss = total_loss / max(n_batches, 1)
        
        # Evaluate on train and test sets
        train_loss_eval, train_acc_e, train_acc_a = evaluate(train_samples, 'train')
        test_loss, test_acc_e, test_acc_a = evaluate(test_samples, 'test')
        
        # Track best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            
            # Save best model
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'epoch': epoch + 1,
                    'test_loss': test_loss,
                    'test_acc_e': test_acc_e,
                    'test_acc_a': test_acc_a,
                    'manifest': manifest,
                    'args': vars(args)
                }, out_dir / 'ckpt_best.pth')
            except Exception as e:
                print(f"Warning: Failed to save checkpoint: {e}")
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"  Train loss: {train_loss:.4f} | Train E-acc: {train_acc_e:.3f} | Train A-acc: {train_acc_a:.3f}")
            print(f"  Test loss:  {test_loss:.4f} | Test E-acc:  {test_acc_e:.3f} | Test A-acc:  {test_acc_a:.3f}")
            print(f"  Best: epoch {best_epoch}, test loss {best_test_loss:.4f}")
            print()
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'epoch': args.epochs,
        'manifest': manifest,
        'args': vars(args)
    }, out_dir / 'ckpt_final.pth')
    
    print(f"{'='*80}\nTraining Complete\n{'='*80}")
    print(f"Best model: epoch {best_epoch}, test loss {best_test_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")
    print()


if __name__ == '__main__':
    main()
