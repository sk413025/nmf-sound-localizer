#!/usr/bin/env python3
"""
Evaluate OMP Transformer accuracy across all samples for a given SNR level.
Uses FullTransformerRoutedSoftOMP model (with Transformer layers) to match baseline.

Based on baseline commit 1f6b68c configuration:
- Model: FullTransformerRoutedSoftOMP
- F=346, E=37, M=8 (K-means reduced)
- d_model=64, nhead=2, nlayers=1
- routing_mode='g'
- epochs=10, batch_size=16, lr=3e-3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from nmf_localizer.core.transfer_functions import TransferFunctionProcessor
from nmf_localizer.config.defaults import NMFConfig
from doa_rl.data import DoADataset


# ============================================================================
# K-means Atom Reduction (from omp-transformer-ldv.py)
# ============================================================================

def reduce_atoms_kmeans(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42):
    """
    Reduce atoms using NumPy K-means (faster, deterministic).

    Args:
        W: (F, M) atom matrix
        n_clusters: Target number of atoms (default: 8)
        random_state: Random seed for reproducibility

    Returns:
        W_reduced: (F, n_clusters) reduced atom matrix
        labels: (M,) cluster assignment for each original atom
        info: Dictionary with reduction statistics
    """
    F, M = W.shape
    W_np = W.cpu().numpy().T  # (M, F)

    # Simple NumPy K-means
    rng = np.random.RandomState(random_state)
    centroids = W_np[rng.choice(M, n_clusters, replace=False)]

    for _ in range(100):  # Max 100 iterations
        # Assign to nearest centroid
        dists = np.linalg.norm(W_np[:, None, :] - centroids[None, :, :], axis=2)  # (M, n_clusters)
        labels = np.argmin(dists, axis=1)  # (M,)

        # Update centroids
        new_centroids = np.array([W_np[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
                                   for k in range(n_clusters)])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    W_reduced = torch.from_numpy(centroids.T).float()  # (F, n_clusters)

    info = {
        'original_atoms': M,
        'reduced_atoms': n_clusters,
        'reduction_ratio': n_clusters / M,
        'cluster_sizes': [int(np.sum(labels == k)) for k in range(n_clusters)]
    }

    return W_reduced, labels, info


# ============================================================================
# Dictionary Construction
# ============================================================================

def build_dictionary(W: torch.Tensor, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build dictionary D from atoms W and transfer functions H.

    Args:
        W: (F, M) atom matrix
        H: (F, E) transfer function matrix

    Returns:
        D: (F, E*M) dictionary matrix (column-normalized)
        idx2: (E*M, 2) mapping from column index to (expert_idx, atom_idx)
    """
    F, M = W.shape
    _, E = H.shape
    P = E * M

    # Create mapping: column index → (expert, atom)
    idx2 = torch.zeros((P, 2), dtype=torch.long)
    for e in range(E):
        for m in range(M):
            col_idx = e * M + m
            idx2[col_idx, 0] = e
            idx2[col_idx, 1] = m

    # Build dictionary: D[:, e*M+m] = W[:, m] * H[:, e]
    D = torch.zeros((F, P))
    for e in range(E):
        for m in range(M):
            col_idx = e * M + m
            D[:, col_idx] = W[:, m] * H[:, e]

    # Column-normalize
    D = D / (torch.norm(D, dim=0, keepdim=True) + 1e-10)

    return D, idx2


# ============================================================================
# Transformer Model (from omp-transformer-ldv.py)
# ============================================================================

class FullTransformerRoutedSoftOMP(nn.Module):
    """
    Full-Transformer Routed Soft-OMP for SNR experiments.
    Simplified version of the full model, focusing on inference.
    """

    def __init__(self, F: int, E: int, M: int, d: int = 64, nhead: int = 2, nlayers: int = 1,
                 steps: int = 6, top_e: int = 2, L: int = 2,
                 tau_e: float = 0.5, tau_a: float = 0.2, eta: float = 0.5,
                 routing_mode: str = 'g'):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.P = E * M
        self.d = d
        self.steps, self.top_e, self.L = steps, top_e, L
        self.routing_mode = routing_mode
        self.tau_e, self.tau_a, self.eta = tau_e, tau_a, eta

        # Learnable routing parameters (for QK mode)
        self.Wq = nn.Parameter(torch.randn(E, M, d) * 0.01)
        self.Wk = nn.Parameter(torch.randn(F, d) * 0.01)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=nhead, dim_feedforward=4 * d,
            dropout=0.0, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        # Projections
        self.fc_in = nn.Linear(F, d)
        self.fc_res = nn.Linear(F, d)

    def forward(self, y: torch.Tensor, D: torch.Tensor, train_mode: bool = True) -> tuple:
        """
        Forward pass with Transformer routing.

        Args:
            y: (F,) observation
            D: (F, P) dictionary
            train_mode: Whether in training mode (soft routing) or inference (hard routing)

        Returns:
            x_hat: (P,) sparse code
            r_curve: List of residuals at each step
        """
        device = y.device
        F, P = D.shape
        E, M = self.E, self.M

        # Initialize
        r = y.clone()
        x = torch.zeros(P, device=device)
        r_curve = [r.clone()]

        # Routing scores: g = D^T @ y (physics-based)
        if self.routing_mode == 'g':
            g = D.T @ y  # (P,)
            g_per_expert = g.view(E, M)  # (E, M)

        for step in range(self.steps):
            # Compute routing scores
            if self.routing_mode == 'g':
                # Physics-based routing
                logits_e = g_per_expert.abs().sum(dim=1)  # (E,) expert scores
                logits_a = g_per_expert  # (E, M) atom scores
            else:
                # QK routing (learned)
                r_emb = self.fc_in(r)  # (d,)
                scores = torch.einsum('emd,d->em', self.Wq, r_emb)  # (E, M)
                logits_e = scores.abs().sum(dim=1)  # (E,)
                logits_a = scores  # (E, M)

            # Select experts
            if train_mode:
                # Soft routing (softmax)
                w_e = torch.softmax(logits_e / self.tau_e, dim=0)  # (E,)
                top_e_indices = torch.topk(w_e, self.top_e).indices
            else:
                # Hard routing (top-K)
                top_e_indices = torch.topk(logits_e, self.top_e).indices
                w_e = torch.zeros(E, device=device)
                w_e[top_e_indices] = 1.0 / self.top_e

            # Select atoms per expert
            for e_idx in top_e_indices:
                e = e_idx.item()
                atom_logits = logits_a[e]  # (M,)

                if train_mode:
                    # Soft atom selection
                    w_a = torch.softmax(atom_logits / self.tau_a, dim=0)  # (M,)
                    top_a_indices = torch.topk(w_a, self.L).indices
                else:
                    # Hard atom selection
                    top_a_indices = torch.topk(atom_logits, self.L).indices
                    w_a = torch.zeros(M, device=device)
                    w_a[top_a_indices] = 1.0 / self.L

                # Update sparse code
                for a_idx in top_a_indices:
                    a = a_idx.item()
                    col_idx = e * M + a
                    d_col = D[:, col_idx]

                    # OMP update
                    alpha = (r @ d_col) / (torch.norm(d_col) ** 2 + 1e-10)
                    x[col_idx] += alpha * w_e[e] * w_a[a] if train_mode else alpha

            # Update residual
            r = y - D @ x
            r_curve.append(r.clone())

            # Early stopping if residual is small
            if torch.norm(r) < 1e-6:
                break

        return x, r_curve


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_and_evaluate(dataset_root: Path, H_path: Path, W_path: Path,
                       freq_min: float, freq_max: float,
                       steps: int, top_e: int, L: int, epochs: int,
                       batch_size: int, lr: float, device: torch.device,
                       d_model: int = 64, nhead: int = 2, nlayers: int = 1,
                       n_atoms: int = 8, routing_mode: str = 'g') -> Dict:
    """Train Transformer model and evaluate on all samples."""

    # Load H matrix
    tfp = TransferFunctionProcessor(NMFConfig(freq_min=freq_min, freq_max=freq_max, device=str(device)))
    H_all, angles_tensor, meta = tfp.load_transfer_functions(str(H_path))
    freqs = None
    try:
        d = torch.load(H_path, weights_only=False)
        if isinstance(d, dict) and 'freqs' in d:
            freqs = np.array(d['freqs'])
    except Exception:
        pass

    if freqs is not None:
        H_proc, _ = tfp.apply_frequency_limit(H_all, freqs, freq_min, freq_max)
    else:
        H_proc = H_all
    H = H_proc.float().to(device)
    angles_subset = angles_tensor
    E = H.shape[1]

    # Load W (USM)
    Wd = torch.load(W_path, map_location='cpu', weights_only=False)
    if isinstance(Wd, dict) and 'W' in Wd:
        W_raw = Wd['W'].float()
    elif isinstance(Wd, torch.Tensor):
        W_raw = Wd.float()
    else:
        raise ValueError(f"Unrecognized W format")

    F = W_raw.shape[0]
    M_raw = W_raw.shape[1]

    # Reduce atoms using K-means (50 → 8)
    print(f"  Reducing atoms: {M_raw} → {n_atoms} using K-means...")
    W_reduced, kmeans_labels, info = reduce_atoms_kmeans(W_raw, n_clusters=n_atoms)
    W = W_reduced.to(device)
    M = n_atoms

    print(f"  Loaded H: {tuple(H.shape)}, W: {tuple(W.shape)} (F={F}, E={E}, M={M})")
    print(f"  Atom reduction: {info}")

    # Build dictionary
    D, idx2 = build_dictionary(W, H)
    D = D.to(device)
    print(f"  Dictionary D: {tuple(D.shape)}")

    # Create dataset
    angles_list = [int(a) for a in angles_subset.tolist()]
    ds = DoADataset(str(dataset_root), angles=angles_list, fs=NMFConfig.sample_rate,
                    n_fft=NMFConfig.n_fft, freq_min=freq_min, freq_max=freq_max)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"  Dataset size: {len(ds)} samples across {len(angles_list)} angles")

    # Create model
    model = FullTransformerRoutedSoftOMP(
        F=F, E=E, M=M, d=d_model, nhead=nhead, nlayers=nlayers,
        steps=steps, top_e=top_e, L=L,
        tau_e=0.5, tau_a=0.2, eta=0.5,
        routing_mode=routing_mode
    ).to(device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    print(f"  Model: FullTransformerRoutedSoftOMP")
    print(f"  d_model={d_model}, nhead={nhead}, nlayers={nlayers}")
    print(f"  routing_mode={routing_mode}")
    print(f"  Training for {epochs} epochs...")

    # Training loop
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dl:
            Y = batch['Y'].to(device)  # (B, F, N)
            y_batch = Y.mean(dim=2)  # (B, F) - average over time

            batch_loss = 0.0
            for i in range(y_batch.size(0)):
                y = y_batch[i]  # (F,)
                x_hat, r_curve = model(y, D, train_mode=True)
                y_hat = D @ x_hat
                rec_loss = torch.nn.functional.mse_loss(y_hat, y)
                batch_loss += rec_loss

            batch_loss = batch_loss / y_batch.size(0)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            epoch_loss += batch_loss.item()
            losses.append(batch_loss.item())

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={epoch_loss/len(dl):.6f}")

    print(f"  Training complete. Final loss: {losses[-1]:.6f}")

    # Evaluation
    print(f"  Evaluating on all {len(ds)} samples...")
    model.eval()

    predictions = []
    ground_truths = []
    per_angle_results = defaultdict(lambda: {'correct': 0, 'total': 0, 'predictions': []})

    with torch.no_grad():
        eval_dl = DataLoader(ds, batch_size=1, shuffle=False)
        for batch in eval_dl:
            Y = batch['Y'][0].to(device)  # (F, N)
            y = Y.mean(dim=1)  # (F,)
            x_hat, _ = model(y, D, train_mode=False)

            # Get predicted angle from group energy
            A_group = x_hat.view(E, M).abs().sum(dim=1).cpu().numpy()
            pred_idx = int(np.argmax(A_group))
            pred_angle = float(angles_subset[pred_idx].item())

            # Ground truth
            gt_angle = float(batch['angle_deg'][0].item())
            gt_idx = angles_list.index(int(gt_angle))

            predictions.append(pred_angle)
            ground_truths.append(gt_angle)

            # Per-angle statistics
            is_correct = (pred_idx == gt_idx)
            per_angle_results[gt_angle]['correct'] += int(is_correct)
            per_angle_results[gt_angle]['total'] += 1
            per_angle_results[gt_angle]['predictions'].append({
                'predicted': pred_angle,
                'correct': is_correct
            })

    # Calculate metrics
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)
    overall_accuracy = np.mean(predictions == ground_truths)

    per_angle_accuracy = {}
    for angle, results in per_angle_results.items():
        accuracy = results['correct'] / results['total']
        per_angle_accuracy[angle] = {
            'accuracy': accuracy,
            'correct': results['correct'],
            'total': results['total']
        }

    return {
        'overall_accuracy': float(overall_accuracy),
        'total_samples': len(ds),
        'correct_predictions': int(np.sum(predictions == ground_truths)),
        'per_angle_accuracy': per_angle_accuracy,
        'training_final_loss': losses[-1],
        'angles_list': [float(a) for a in angles_list],
        'F': int(F),
        'E': int(E),
        'M': int(M),
        'd_model': int(d_model),
        'nhead': int(nhead),
        'nlayers': int(nlayers),
        'routing_mode': routing_mode
    }


def main():
    ap = argparse.ArgumentParser(description="Evaluate OMP Transformer accuracy on SNR dataset")
    ap.add_argument('--dataset_root', type=str, required=True, help='Root with angle_XX/*.npy')
    ap.add_argument('--H_path', type=str, required=True)
    ap.add_argument('--W_path', type=str, required=True)
    ap.add_argument('--freq_min', type=float, default=300.0)
    ap.add_argument('--freq_max', type=float, default=3000.0)
    ap.add_argument('--steps', type=int, default=6)
    ap.add_argument('--top_e', type=int, default=2)
    ap.add_argument('--L', type=int, default=2)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--nhead', type=int, default=2)
    ap.add_argument('--nlayers', type=int, default=1)
    ap.add_argument('--n_atoms', type=int, default=8)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--routing_mode', type=str, default='g', choices=['qk', 'g', 'hybrid'])
    ap.add_argument('--out_dir', type=str, required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Evaluation configuration:")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  Device: {device}")
    print(f"  Output: {out_dir}")

    # Train and evaluate
    results = train_and_evaluate(
        dataset_root=Path(args.dataset_root),
        H_path=Path(args.H_path),
        W_path=Path(args.W_path),
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        steps=args.steps,
        top_e=args.top_e,
        L=args.L,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        n_atoms=args.n_atoms,
        routing_mode=args.routing_mode
    )

    # Save results
    results_path = out_dir / 'accuracy_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Evaluation Results:")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {results['overall_accuracy']*100:.2f}%")
    print(f"Correct: {results['correct_predictions']}/{results['total_samples']}")
    print(f"Training Final Loss: {results['training_final_loss']:.6f}")
    print(f"\nPer-Angle Accuracy:")
    for angle in sorted(results['per_angle_accuracy'].keys()):
        acc_data = results['per_angle_accuracy'][angle]
        print(f"  {angle:6.1f}°: {acc_data['accuracy']*100:5.1f}% ({acc_data['correct']}/{acc_data['total']})")
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
