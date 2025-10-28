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
    ap.add_argument('--atom_reduce_mode', type=str, default='kcenter', choices=['kcenter'])
    # Trajectory/targets
    ap.add_argument('--K', type=int, default=6)
    ap.add_argument('--rtg_target_resid', type=float, default=0.02)
    ap.add_argument('--rtg_target_acc', type=float, default=0.95)
    ap.add_argument('--softmax_T', type=float, default=1.0)
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
    print("\n=== Atom reduction (k-center) ===")
    W, labels, info = reduce_atoms_kcenter(W_full, n_clusters=int(args.n_atoms))
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

        for t in range(args.K):
            # Hierarchical pick using |g| on residual
            e, m, j, energy_e_max, a_score = hierarchical_pick_g(D, r, E=E, M=M)
            S.append(j)

            # Orthogonal LS refit on selected atoms
            D_S = D[:, S]
            x_S = solve_ls(D_S, y)
            y_hat = D_S @ x_S
            r = y - y_hat
            res_now = float((r @ r).item())
            delta_res = res_prev - res_now
            res_prev = res_now

            # Classification confidence from y_hat via per-angle energy softmax
            p_true = angle_prob_from_yhat(D, y_hat, E=E, M=M, true_e=angle_idx, temp=args.softmax_T)

            # RTGs
            rtg_res = max(0.0, res_now - args.rtg_target_resid)
            rtg_acc = max(0.0, args.rtg_target_acc - p_true)

            steps.append(StepRecord(
                step=t,
                expert=e,
                atom=m,
                dict_index=j,
                resid_sq=res_now,
                delta_resid_sq=delta_res,
                p_true=p_true,
                rtg_resid=rtg_res,
                rtg_acc=rtg_acc,
            ))

        # Write trajectory line
        traj_obj = {
            'path': path,
            'angle_deg': angle_deg,
            'angle_index': angle_idx,
            'steps': [asdict(s) for s in steps],
            'final': {
                'resid_sq': steps[-1].resid_sq if steps else float((y @ y).item()),
                'p_true': steps[-1].p_true if steps else float('nan')
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

