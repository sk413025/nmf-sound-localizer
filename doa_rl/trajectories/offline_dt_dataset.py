#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline trajectory builder for Decision Transformer (g-teacher, hierarchical pointer).

Generates K-step greedy trajectories using |g| = |D^T r| energies with:
- State: residual r_t (time-averaged, band-limited STFT magnitude, F=346),
         selected set S_t (stored as indices), step t and remaining budget K-t.
- Action: hierarchical pointer (expert e in [0,E), atom m in [0,M)), dict index j=e*M+m.
- Transition: deterministic least-squares re-fitting on selected atoms (orthogonal projection).
- Rewards/Diagnostics: per-step Δ||r||^2, classification confidence p_t estimated from y_hat_t,
  and RTGs (RTG_resid, RTG_acc) against targets.

Outputs under results/<run_name>/:
- trajectories.jsonl  (one JSON per sample; includes per-step fields: a_t, Δr2, p_t, RTG_t)
- numeric_diagnostics.jsonl  (per-sample summary + per-step stats)
- manifest.json  (subset list, angles, seeds, STFT grid)
- code_state.json (git_head, args, files sha256, dataset fingerprint)
- run.log         (recommend capturing via: python -u ... 2>&1 | tee results/.../run.log)

No-Fallback policy:
- Validates STFT band length F equals H.F equals W.F; mismatches raise with clear message.
- Angles in dataset must match H angles; duplicates and missing are reported.
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import argparse
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from doa_rl.data import DoADataset, create_dataloader


# -----------------------------
# Utilities (local, minimal)
# -----------------------------

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_dataset_fingerprint(dataset_root: str) -> Tuple[str, int]:
    root = Path(dataset_root)
    npy_files = sorted(root.rglob('*.npy'))
    h = hashlib.md5()
    for p in npy_files:
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest(), len(npy_files)


def load_raw_ldv_matrices(h_path: str, w_path: str, device: str = 'cpu'):
    print(f"Loading H matrix from: {h_path}")
    h_data = torch.load(h_path, map_location=device, weights_only=False)
    H = h_data['H']  # (F, E)
    angles = h_data['angles']  # (E,)

    print(f"Loading W matrix from: {w_path}")
    w_data = torch.load(w_path, map_location=device, weights_only=False)
    if isinstance(w_data, dict) and 'W' in w_data:
        W = w_data['W']  # (F, M_full)
    else:
        W = w_data

    print("Raw matrices:")
    print(f"  H shape: {tuple(H.shape)}")
    print(f"  W shape: {tuple(W.shape)}")
    print(f"  Angles: {len(angles)} total -> {angles.tolist()}")
    return H.float(), W.float(), np.array(angles, dtype=float)


def reduce_atoms_kcenter(W: torch.Tensor, n_clusters: int = 8) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """Greedy k-center (cosine) to pick diverse atoms without sklearn dependency."""
    Fdim, M_full = W.shape
    X = W.cpu().numpy().T  # (M_full, F)
    eps = 1e-12
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    mean = Xn.mean(axis=0)
    mean /= (np.linalg.norm(mean) + eps)
    d_to_mean = 1.0 - (Xn @ mean)
    centers = [int(np.argmax(d_to_mean))]
    dmin = 1.0 - (Xn @ Xn[centers[0]].T)
    for _ in range(1, n_clusters):
        idx = int(np.argmax(dmin))
        centers.append(idx)
        dmin = np.minimum(dmin, 1.0 - (Xn @ Xn[idx].T))
    centers = sorted(list(dict.fromkeys(centers)))
    W_red = W[:, centers].clone()
    W_red = W_red / (W_red.norm(dim=0, keepdim=True) + 1e-12)
    # label each original to nearest center
    sims = Xn @ Xn[centers].T
    labels = np.argmax(sims, axis=1)
    info = {"selected_indices": centers, "cluster_sizes": np.bincount(labels, minlength=len(centers)).tolist()}
    return W_red, labels, info


def reduce_atoms_kmeans(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """Lightweight NumPy KMeans (k-means++ init, 30 iters) over atom columns.
    Avoids sklearn overhead; fast for M_full≈50.
    """
    X = W.cpu().numpy().T  # (M_full, F)
    rng = np.random.RandomState(random_state)
    M_full, Fdim = X.shape
    # k-means++ init
    centers = []
    idx0 = int(rng.randint(M_full))
    centers.append(X[idx0].copy())
    for _ in range(1, n_clusters):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)  # (M_full,)
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(M_full, p=probs))
        centers.append(X[idx].copy())
    C = np.stack(centers, axis=0)  # (k, F)
    labels = np.zeros(M_full, dtype=int)
    for _ in range(30):
        # Assign
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)  # (M_full, k)
        labels = np.argmin(dists, axis=1)
        # Update
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
    W_red = torch.from_numpy(C.T).float()  # (F, k)
    W_red = W_red / (W_red.norm(dim=0, keepdim=True) + 1e-12)
    info = {}
    return W_red, labels, info




def build_dictionary(H: torch.Tensor, W: torch.Tensor, angles: np.ndarray) -> Tuple[torch.Tensor, List[Tuple[float, int]]]:
    """Build D = normalized (H[:,e] * W[:,m]) for all e,m; return D (F,P) and idx2angle list."""
    Fdim, E = H.shape
    _, M = W.shape
    P = E * M
    D = torch.zeros(Fdim, P, dtype=torch.float32)
    idx2angle: List[Tuple[float, int]] = []
    j = 0
    for e in range(E):
        h = H[:, e]
        for m in range(M):
            atom = h * W[:, m]
            atom = atom / (atom.norm() + 1e-12)
            D[:, j] = atom
            idx2angle.append((float(angles[e]), m))
            j += 1
    # diagnostics
    G = (D.T @ D)
    G.fill_diagonal_(0)
    mu_max = G.abs().max().item()
    mu_mean = G.abs().mean().item()
    print("Dictionary built:")
    print(f"  D shape: {tuple(D.shape)} (P = {P})")
    print(f"  Mutual coherence μ_max={mu_max:.4f}, μ_mean={mu_mean:.4f}")
    return D, idx2angle


# -----------------------------
# Greedy teacher (|g| energies)
# -----------------------------

@dataclass
class StepRecord:
    step: int
    expert: int
    atom: int
    dict_index: int
    resid_sq: float
    delta_resid_sq: float
    reward: float
    p_true: float
    rtg_resid: float
    rtg_acc: float


def hierarchical_pick_g(D: torch.Tensor, r: torch.Tensor, E: int, M: int) -> Tuple[int, int, int, float, float]:
    """
    Pick expert and atom using |g| energies from current residual.
    Returns (e, m, j, e_energy_max, a_score).
    """
    g = (D.T @ r)  # (P,)
    g_em = g.view(E, M)
    energy_e = g_em.abs().sum(dim=1)  # (E,)
    e = int(torch.argmax(energy_e).item())
    a_scores = g_em[e, :].abs()  # (M,)
    m = int(torch.argmax(a_scores).item())
    j = e * M + m
    return e, m, j, float(energy_e[e].item()), float(a_scores[m].item())


def solve_ls(D_sub: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Least-squares with torch.linalg.lstsq for stability (no NNLS)."""
    # y ~ D_sub x; returns x (min-norm solution)
    sol = torch.linalg.lstsq(D_sub, y)
    x = sol.solution
    return x


def angle_prob_from_yhat(D: torch.Tensor, y_hat: torch.Tensor, E: int, M: int, true_e: int, temp: float = 1.0) -> float:
    """Compute p_true using per-angle energies from D^T y_hat and softmax."""
    g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)  # (E,)
    logits = g_hat / max(temp, 1e-8)
    p = F.softmax(logits, dim=0)
    return float(p[true_e].item())


# -----------------------------
# Main routine
# -----------------------------

def _load_qk_model(d_model: int, nhead: int, nlayers: int, F: int, E: int, M: int, device: torch.device,
                   checkpoint_path: str):
    """Dynamically load FullTransformerRoutedSoftOMP from scripts/omp-transformer-ldv.py and restore weights.

    Returns an eval() model whose internals can be used to compute QK scores step-wise.
    """
    import importlib.util
    mod_path = Path('scripts/omp-transformer-ldv.py')
    if not mod_path.exists():
        raise FileNotFoundError(f"Cannot find {mod_path} to load QK teacher model")
    spec = importlib.util.spec_from_file_location('omp_transformer_ldv', str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, 'FullTransformerRoutedSoftOMP')
    model = ModelClass(F=F, E=E, M=M, d=d_model, nhead=nhead, nlayers=nlayers, steps=1).to(device)
    # Load weights
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = state.get('model_state_dict', state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[QK teacher] load_state_dict: missing={missing}, unexpected={unexpected}")
    # Restore critical config flags (from fe4ef60 run)
    model.routing_mode = 'qk'
    model.score_norm_mode = 'std'
    model.score_center_expert = True
    model.score_center_atoms = True
    model.no_type_bias = True
    model.expert_agg = 'l2'
    model.eval()
    return model


def _qk_scores_with_config(model, D: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (qk_expert, qk_atoms) with model configuration: std norm and centering if enabled.
    Returns:
        qk_expert: (E,)
        qk_atoms: (E,M)
    """
    E, M = model.E, model.M
    t_R = model.P_R(r) + (model.type_R if not model.no_type_bias else 0)
    T_D = model.P_D(D.T) + (model.type_D if not model.no_type_bias else 0)
    T = torch.cat([t_R[None, :], T_D], dim=0)
    mask = model._make_mask(T.size(0)).to(T.device)
    Henc = T if model.encoder_identity else model.encoder(T, mask=mask)
    h_R = Henc[0]
    H_D = Henc[1:]
    qk_atoms = (model.Wk(H_D) @ model.Wq(h_R)) / math.sqrt(model.d)  # (P,)
    qk_atoms = qk_atoms.reshape(E, M)
    # Aggregate expert scores
    if model.expert_agg == 'l2':
        qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)
    elif model.expert_agg == 'max':
        qk_expert = qk_atoms.abs().max(dim=1).values
    else:
        qk_expert = F.relu(qk_atoms).mean(dim=1)
    # Score normalization
    if getattr(model, 'score_norm_mode', 'none') == 'std':
        se_mean, se_std = qk_expert.mean(), qk_expert.std()
        qk_expert = (qk_expert - se_mean) / (se_std + 1e-8)
        sa_mean, sa_std = qk_atoms.mean(), qk_atoms.std()
        qk_atoms = (qk_atoms - sa_mean) / (sa_std + 1e-8)
    # Centering
    if getattr(model, 'score_center_expert', False):
        qk_expert = qk_expert - qk_expert.mean()
    if getattr(model, 'score_center_atoms', False):
        qk_atoms = qk_atoms - qk_atoms.mean(dim=1, keepdim=True)
    return qk_expert, qk_atoms


def main():
    ap = argparse.ArgumentParser(description='Offline DT trajectory builder (g-teacher)')
    # Data paths
    ap.add_argument('--h_path', type=str, default='/Users/sbplab/jiawei/LDV-data-processed/h_matrix_box_ldv_correct.pth')
    ap.add_argument('--w_path', type=str, default='doa_normalized_config_c_corrected/models/usm.pth')
    ap.add_argument('--dataset_root', type=str, default='/Users/sbplab/jiawei/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized')
    # STFT/grid
    ap.add_argument('--fs', type=int, default=16000)
    ap.add_argument('--n_fft', type=int, default=2048)
    ap.add_argument('--freq_min', type=float, default=300.0)
    ap.add_argument('--freq_max', type=float, default=3000.0)
    # Dictionary/atoms
    ap.add_argument('--n_atoms', type=int, default=8)
    ap.add_argument('--atom_reduce_mode', type=str, default='kcenter', choices=['kcenter','kmeans','diverse'])
    # Trajectory/targets
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--rtg_target_resid', type=float, default=0.02)
    ap.add_argument('--rtg_target_acc', type=float, default=0.95)
    ap.add_argument('--softmax_T', type=float, default=1.0)
    # Teacher selection
    ap.add_argument('--teacher', type=str, default='g', choices=['g','qk'], help='Trajectory teacher policy')
    ap.add_argument('--qk_ckpt', type=str, default='results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth', help='Checkpoint for qk teacher')
    ap.add_argument('--qk_d_model', type=int, default=128)
    ap.add_argument('--qk_nhead', type=int, default=2)
    ap.add_argument('--qk_nlayers', type=int, default=1)
    # Subset controls
    ap.add_argument('--subset_angles', type=str, default='', help='Comma-separated angles, e.g., "0,5,10"')
    ap.add_argument('--max_samples', type=int, default=0, help='Limit number of samples (0 = all)')
    ap.add_argument('--seed', type=int, default=42)
    # Output
    ap.add_argument('--out_dir', type=str, default='results/dt_traj_v1')
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu', 'mps', 'cuda'])

    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Offline DT Trajectory Builder (g-teacher)")
    print("=" * 80)
    print(f"Output directory: {out_dir}")

    # ------------------------------------------------------------------
    # Load H, W and reduce atoms; build dictionary D
    # ------------------------------------------------------------------
    H, W_full, angles = load_raw_ldv_matrices(args.h_path, args.w_path, device='cpu')
    Fdim_H, E = H.shape
    if args.atom_reduce_mode == 'kcenter':
        print("\n=== Atom reduction (k-center) ===")
        W, labels, info = reduce_atoms_kcenter(W_full, n_clusters=int(args.n_atoms))
        print(f"Selected indices: {info['selected_indices']}")
        print(f"Cluster sizes: {info['cluster_sizes']}")
    elif args.atom_reduce_mode == 'kmeans':
        print("\n=== Atom reduction (K-means) ===")
        print("  impl = numpy (k-means++ init, iters=30)")
        W, labels, _ = reduce_atoms_kmeans(W_full, n_clusters=int(args.n_atoms), random_state=args.seed)
        info = {'selected_indices': None, 'cluster_sizes': np.bincount(labels).tolist()}
        print(f"Cluster sizes: {info['cluster_sizes']}")
    else:
        print("\n=== Atom reduction (diverse cos-sep) ===")
        W, labels, info = reduce_atoms_diverse(W_full, n_clusters=int(args.n_atoms))
        print(f"Selected indices: {info['selected_indices']}")
        print(f"Cluster sizes: {info['cluster_sizes']}")
    Fdim_W, M = W.shape
    if Fdim_W != Fdim_H:
        raise RuntimeError(f"F mismatch after atom reduction: H.F={Fdim_H} vs W.F={Fdim_W}")

    print("\n=== Build dictionary D = H ⊙ W ===")
    D, idx2angle = build_dictionary(H, W, angles)
    Fdim_D, P = D.shape

    # ------------------------------------------------------------------
    # Load dataset, produce Y vectors with STFT band (F must match)
    # ------------------------------------------------------------------
    print("\n=== Loading dataset (DoADataset) ===")
    angles_list = angles.tolist()
    dataset = DoADataset(
        root=args.dataset_root,
        angles=angles_list,
        fs=args.fs,
        n_fft=args.n_fft,
        window='hann',
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(dataset, batch_size=1, shuffle=False)
    print(f"Dataset size: {len(dataset)}; angles covered: {angles_list}")

    # Dataset fingerprint
    fingerprint, n_files = compute_dataset_fingerprint(args.dataset_root)
    print(f"Dataset fingerprint (MD5): {fingerprint} (from {n_files} .npy files)")

    # Optional subset filter
    subset_angles: Optional[List[int]] = None
    if args.subset_angles.strip():
        subset_angles = [int(a.strip()) for a in args.subset_angles.split(',') if a.strip()]
        print(f"Subset angles requested: {subset_angles}")

    # ------------------------------------------------------------------
    # No-Fallback checks: F consistency
    # ------------------------------------------------------------------
    F_expected = Fdim_D
    if F_expected != Fdim_H or F_expected != Fdim_W:
        raise RuntimeError(f"Inconsistent F: D.F={Fdim_D}, H.F={Fdim_H}, W.F={Fdim_W}")

    # ------------------------------------------------------------------
    # Trajectory generation
    # ------------------------------------------------------------------
    traj_path = out_dir / 'trajectories.jsonl'
    diag_path = out_dir / 'numeric_diagnostics.jsonl'
    manifest_path = out_dir / 'manifest.json'

    n_written = 0
    manifest: Dict[str, Any] = {
        'dataset_root': args.dataset_root,
        'angles': angles_list,
        'fs': args.fs,
        'n_fft': args.n_fft,
        'freq_min': args.freq_min,
        'freq_max': args.freq_max,
        'F': F_expected,
        'E': E,
        'M': M,
        'P': P,
        'K': args.K,
        'rtg_targets': {'resid': args.rtg_target_resid, 'acc': args.rtg_target_acc},
        'softmax_T': args.softmax_T,
        'teacher': args.teacher,
        'qk': {
            'ckpt': args.qk_ckpt,
            'd_model': args.qk_d_model,
            'nhead': args.qk_nhead,
            'nlayers': args.qk_nlayers,
        } if args.teacher == 'qk' else None,
        'fingerprint_md5': fingerprint,
        'npy_file_count': n_files,
        'subset_angles': subset_angles,
        'seed': args.seed,
        'h_path': args.h_path,
        'w_path': args.w_path,
        'atom_reduce': {'mode': args.atom_reduce_mode, 'selected_indices': info['selected_indices'], 'cluster_sizes': info['cluster_sizes']},
        'samples': []
    }

    # Open outputs
    traj_f = open(traj_path, 'w')
    diag_f = open(diag_path, 'w')

    device = torch.device(args.device)
    D = D.to(device)
    qk_model = None
    if args.teacher == 'qk':
        print("\n=== Loading QK teacher model ===")
        qk_model = _load_qk_model(d_model=args.qk_d_model, nhead=args.qk_nhead, nlayers=args.qk_nlayers,
                                  F=F_expected, E=E, M=M, device=device, checkpoint_path=args.qk_ckpt)
        # Log teacher config and quick eval on current D
        print("QK teacher config:")
        print(f"  routing_mode={qk_model.routing_mode}")
        print(f"  score_norm_mode={qk_model.score_norm_mode}")
        print(f"  score_center_expert={qk_model.score_center_expert}")
        print(f"  score_center_atoms={qk_model.score_center_atoms}")
        print(f"  no_type_bias={qk_model.no_type_bias}")
        print(f"  expert_agg={qk_model.expert_agg}")
        print("\n=== Teacher t=0 evaluation on dataset (current D) ===")
        correct = 0
        expert_counts = [0]*E
        for batch in dl:
            Y = batch['Y'][0]
            angle_idx = int(batch['angle_index'][0])
            r0 = (Y.mean(dim=1).float().to(device))
            r0 = r0 / (r0.norm() + 1e-12)
            qk_exp, _ = _qk_scores_with_config(qk_model, D, r0)
            e_pred = int(qk_exp.argmax().item())
            expert_counts[e_pred] += 1
            if e_pred == angle_idx:
                correct += 1
        acc = correct / max(1, len(dataset))
        print(f"Teacher accuracy (t=0, current D): {acc:.3f}")
        print(f"Expert distribution: {[int(c) for c in expert_counts]}")
        # Save to manifest
        manifest['teacher_eval'] = {
            'accuracy_t0_currentD': acc,
            'expert_hist': expert_counts,
            'config': {
                'routing_mode': qk_model.routing_mode,
                'score_norm_mode': qk_model.score_norm_mode,
                'score_center_expert': qk_model.score_center_expert,
                'score_center_atoms': qk_model.score_center_atoms,
                'no_type_bias': qk_model.no_type_bias,
                'expert_agg': qk_model.expert_agg,
            }
        }

    t0 = time.time()
    for i, batch in enumerate(dl):
        angle_deg = float(batch['angle_deg'][0])
        angle_idx = int(batch['angle_index'][0])
        path = str(batch['path'][0])

        if subset_angles is not None and int(angle_deg) not in subset_angles:
            continue

        # Time-average spectrum and normalize (match existing pipeline)
        Y = batch['Y'][0]  # (F, N)
        if Y.shape[0] != F_expected:
            raise RuntimeError(f"Y.F mismatch at sample {path}: Y.F={Y.shape[0]} vs expected {F_expected}")
        y = Y.mean(dim=1).float()
        y = y / (y.norm() + 1e-12)
        y = y.to(device)

        # Initialize residual and selection set
        r = y.clone()
        S: List[int] = []
        steps: List[StepRecord] = []
        res_prev = float((r @ r).item())

        # Also track probability using current reconstruction (start with zeros)
        y_hat = torch.zeros_like(y)

        total_reward = 0.0

        for t in range(args.K):
            # Hierarchical pick
            if args.teacher == 'g':
                e, m, j, energy_e_max, a_score = hierarchical_pick_g(D, r, E=E, M=M)
            else:
                assert qk_model is not None
                with torch.no_grad():
                    qk_expert, qk_atoms = _qk_scores_with_config(qk_model, D, r)
                    e = int(torch.argmax(qk_expert).item())
                    m = int(torch.argmax(qk_atoms[e].abs()).item())
                    j = e * M + m
            S.append(j)

            # Orthogonal LS refit on selected atoms
            D_S = D[:, S]
            x_S = solve_ls(D_S, y)
            y_hat = D_S @ x_S
            r = y - y_hat
            res_now = float((r @ r).item())
            delta_res = res_prev - res_now
            res_prev = res_now
            # Step reward: residual energy reduction (positive when reconstruction improves).
            reward_t = delta_res

            # Classification confidence from y_hat via per-angle energy softmax
            p_true = angle_prob_from_yhat(D, y_hat, E=E, M=M, true_e=angle_idx, temp=args.softmax_T)

            # RTGs (targets remain as before; reward is logged separately)
            rtg_res = max(0.0, res_now - args.rtg_target_resid)
            rtg_acc = max(0.0, args.rtg_target_acc - p_true)

            steps.append(StepRecord(
                step=t,
                expert=e,
                atom=m,
                dict_index=j,
                resid_sq=res_now,
                delta_resid_sq=delta_res,
                 reward=reward_t,
                p_true=p_true,
                rtg_resid=rtg_res,
                rtg_acc=rtg_acc,
            ))
            total_reward += reward_t

        # Write trajectory line
        traj_obj = {
            'path': path,
            'angle_deg': angle_deg,
            'angle_index': angle_idx,
            'steps': [asdict(s) for s in steps],
            'final': {
                'resid_sq': steps[-1].resid_sq if steps else float((y @ y).item()),
                'p_true': steps[-1].p_true if steps else float('nan'),
                # Trajectory-level return: sum of per-step rewards (residual energy reductions).
                'total_reward': float(total_reward),
            }
        }
        traj_f.write(json.dumps(traj_obj) + "\n")

        # Diagnostics summary per sample
        deltas = [s.delta_resid_sq for s in steps]
        p_vals = [s.p_true for s in steps]
        diag_obj = {
            'path': path,
            'angle_deg': angle_deg,
            'angle_index': angle_idx,
            'F': F_expected,
            'K': args.K,
            'resid_sq_init': float((y @ y).item()),
            'resid_sq_final': steps[-1].resid_sq if steps else float((y @ y).item()),
            'delta_resid_sq_stats': {
                'min': float(np.min(deltas)) if deltas else 0.0,
                'median': float(np.median(deltas)) if deltas else 0.0,
                'mean': float(np.mean(deltas)) if deltas else 0.0,
                'p95': float(np.percentile(deltas, 95)) if deltas else 0.0,
                'p99': float(np.percentile(deltas, 99)) if deltas else 0.0,
                'max': float(np.max(deltas)) if deltas else 0.0,
            },
            'p_true_stats': {
                'min': float(np.min(p_vals)) if p_vals else 0.0,
                'median': float(np.median(p_vals)) if p_vals else 0.0,
                'mean': float(np.mean(p_vals)) if p_vals else 0.0,
                'p95': float(np.percentile(p_vals, 95)) if p_vals else 0.0,
                'p99': float(np.percentile(p_vals, 99)) if p_vals else 0.0,
                'max': float(np.max(p_vals)) if p_vals else 0.0,
            }
        }
        diag_f.write(json.dumps(diag_obj) + "\n")

        manifest['samples'].append({'path': path, 'angle_deg': angle_deg, 'angle_index': angle_idx})
        n_written += 1

        if args.max_samples and n_written >= args.max_samples:
            print(f"Reached max_samples={args.max_samples}; stopping early.")
            break

        if (i + 1) % 20 == 0:
            print(f"Processed {i+1} samples...")

    traj_f.close()
    diag_f.close()

    # Save manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Save code_state
    code_state = {
        'git_head': os.popen('git rev-parse HEAD').read().strip(),
        'git_dirty': bool(os.popen('git diff --quiet').close()),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'args': vars(args),
        'dataset_fingerprint': fingerprint,
        'files_sha256': {
            'doa_rl/trajectories/offline_dt_dataset.py': sha256_of_file(Path(__file__)),
            'doa_rl/data.py': sha256_of_file(Path('doa_rl/data.py')),
            'nmf_localizer/utils/audio_utils.py': sha256_of_file(Path('nmf_localizer/utils/audio_utils.py')),
        }
    }
    if qk_model is not None:
        code_state['teacher_model_config'] = {
            'routing_mode': qk_model.routing_mode,
            'score_norm_mode': qk_model.score_norm_mode,
            'score_center_expert': qk_model.score_center_expert,
            'score_center_atoms': qk_model.score_center_atoms,
            'no_type_bias': qk_model.no_type_bias,
            'expert_agg': qk_model.expert_agg,
        }
    with open(out_dir / 'code_state.json', 'w') as f:
        json.dump(code_state, f, indent=2)

    dt = time.time() - t0
    print("\n" + "=" * 80)
    print("DONE: trajectories built")
    print(f"Samples written: {n_written}")
    print(f"Artifacts: {traj_path}, {diag_path}, {manifest_path}")
    print(f"Elapsed: {dt:.2f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
