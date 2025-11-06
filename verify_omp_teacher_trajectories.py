#!/usr/bin/env python3
"""
Verify Traditional OMP Teacher trajectory accuracy.
Compare against g-teacher to assess first-step accuracy differences.

Usage:
1. Generate OMP trajectories: python doa_rl/trajectories/offline_dt_dataset.py --teacher omp --out_dir results/dt_traj_omp_test
2. Run verification: python verify_omp_teacher_trajectories.py --traj_dir results/dt_traj_omp_test
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor


def load_band_spectrogram_from_npy(
    npy_path: Path,
    fs: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    """Load band-limited STFT spectrogram from .npy file"""
    wav = np.load(str(npy_path))
    if wav.ndim != 1:
        raise RuntimeError(f"Expected mono waveform in {npy_path}")
    freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=n_fft, window="hann"
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


def simple_kcenter_torch(X: torch.Tensor, n_centers: int, seed: int = 42) -> Tuple[torch.Tensor, torch.Tensor]:
    """K-center greedy selection"""
    torch.manual_seed(seed)
    N = X.shape[0]
    centers = []
    indices = []
    
    # Random first center
    first_idx = torch.randint(0, N, (1,)).item()
    centers.append(X[first_idx])
    indices.append(first_idx)
    
    for _ in range(1, n_centers):
        distances = torch.stack([torch.norm(X - c, dim=1) for c in centers])
        min_distances = distances.min(dim=0)[0]
        farthest_idx = min_distances.argmax().item()
        centers.append(X[farthest_idx])
        indices.append(farthest_idx)
    
    centers_tensor = torch.stack(centers)
    dist = torch.cdist(X, centers_tensor)
    labels = dist.argmin(dim=1)
    return centers_tensor, labels


def build_dictionary(
    H: torch.Tensor,
    W: torch.Tensor,
    n_atoms: int = 8,
    atom_reduce_mode: str = "kcenter",
    seed: int = 42
) -> Tuple[torch.Tensor, int, int]:
    """Build dictionary D = H ⊙ W_reduced"""
    E = H.shape[1]
    M_full = W.shape[1]
    
    if n_atoms < M_full:
        if atom_reduce_mode == "kcenter":
            W_T = W.T
            centroids, labels = simple_kcenter_torch(W_T, n_centers=n_atoms, seed=seed)
            W_reduced = centroids.T
        else:
            raise ValueError(f"Unknown atom_reduce_mode: {atom_reduce_mode}")
        W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)
    else:
        W_reduced = W
    
    M = W_reduced.shape[1]
    P = E * M
    D = torch.zeros(H.shape[0], P)
    for e in range(E):
        for m in range(M):
            D[:, e * M + m] = H[:, e] * W_reduced[:, m]
    D = D / (D.norm(dim=0, keepdim=True) + 1e-12)
    
    return D, E, M


def traditional_omp_forward(D: torch.Tensor, y: torch.Tensor, K: int, E: int, M: int) -> Tuple[List[int], List[int]]:
    """
    Traditional OMP: greedy selection based on max |D^T @ r|
    No hierarchical structure - purely correlation-based.
    
    Returns:
        expert_seq: List of selected expert indices [0, E-1]
        atom_seq: List of selected atom indices [0, P-1]
    """
    r = y.clone()
    selected_atoms = []
    expert_seq = []
    
    for t in range(K):
        # Compute gradient g = D^T @ r
        g = D.T @ r  # (P,)
        
        # Mask already selected atoms
        if len(selected_atoms) > 0:
            g_masked = g.clone()
            for prev_idx in selected_atoms:
                g_masked[prev_idx] = 0
            g = g_masked
        
        # Traditional OMP: select atom with max absolute correlation
        j = g.abs().argmax().item()
        
        # Derive expert and atom indices
        e = j // M
        m = j % M
        
        selected_atoms.append(j)
        expert_seq.append(e)
        
        # OMP update (least-squares)
        D_S = D[:, selected_atoms]
        result = torch.linalg.lstsq(D_S, y.unsqueeze(1))
        x = result.solution.squeeze(1)
        r = y - D_S @ x
    
    return expert_seq, selected_atoms


def g_teacher_forward(D: torch.Tensor, y: torch.Tensor, K: int, E: int, M: int) -> Tuple[List[int], List[int]]:
    """
    G-Teacher OMP: Hierarchical selection
    1. First select expert (based on total |g| energy per expert)
    2. Then select atom (based on max |g| within expert)
    
    Returns:
        expert_seq: List of selected expert indices [0, E-1]
        atom_seq: List of selected atom indices [0, P-1]
    """
    r = y.clone()
    selected_atoms = []
    expert_seq = []
    
    for t in range(K):
        # Compute gradient g = D^T @ r
        g = D.T @ r  # (P,)
        g_em = g.view(E, M)  # (E, M)
        
        # Mask already selected atoms
        if len(selected_atoms) > 0:
            for prev_idx in selected_atoms:
                e_prev = prev_idx // M
                m_prev = prev_idx % M
                g_em[e_prev, m_prev] = 0
        
        # Stage 1: Select expert (based on total energy)
        energy_e = g_em.abs().sum(dim=1)  # (E,)
        e = energy_e.argmax().item()
        
        # Stage 2: Select atom within expert
        a_scores = g_em[e, :].abs()  # (M,)
        m = a_scores.argmax().item()
        
        j = e * M + m
        selected_atoms.append(j)
        expert_seq.append(e)
        
        # OMP update (least-squares)
        D_S = D[:, selected_atoms]
        result = torch.linalg.lstsq(D_S, y.unsqueeze(1))
        x = result.solution.squeeze(1)
        r = y - D_S @ x
    
    return expert_seq, selected_atoms


def parse_gt_angle_from_path(path: str) -> int:
    """Parse ground-truth angle from file path"""
    p = Path(path)
    angle_dir = p.parent.name
    if angle_dir.startswith("angle_"):
        angle_str = angle_dir.replace("angle_", "")
        angle_int = int(angle_str)
        return angle_int * 5
    raise ValueError(f"Cannot parse angle from path: {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify OMP teacher trajectories")
    parser.add_argument("--traj_dir", type=str, required=True, help="Directory containing trajectories.jsonl and manifest.json")
    parser.add_argument("--h_path", type=str, default="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth")
    parser.add_argument("--w_path", type=str, default="doa_normalized_config_c_corrected/models/usm.pth")
    
    args = parser.parse_args()
    
    traj_dir = Path(args.traj_dir)
    manifest_path = traj_dir / "manifest.json"
    trajectories_path = traj_dir / "trajectories.jsonl"
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)
    
    if not trajectories_path.exists():
        print(f"❌ Trajectories not found: {trajectories_path}")
        sys.exit(1)
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print("=" * 80)
    print("OMP Teacher Verification")
    print("=" * 80)
    print(f"Trajectory directory: {traj_dir}")
    print(f"Teacher: {manifest.get('teacher', 'unknown')}")
    print(f"K: {manifest['K']}")
    print(f"Samples: {len(manifest['samples'])}")
    print()
    
    # Load H and W
    print("Loading matrices...")
    H, angles = load_H(args.h_path)
    W = load_W(args.w_path)
    
    # Build dictionary
    D, E, M = build_dictionary(
        H, W,
        n_atoms=manifest['M'],
        atom_reduce_mode=manifest['atom_reduce']['mode'],
        seed=manifest['seed']
    )
    
    print(f"Dictionary built: D.shape={D.shape}, E={E}, M={M}, P={E*M}")
    print()
    
    # Load trajectories
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    
    print(f"Loaded {len(trajectories)} trajectories")
    print()
    
    # Verify each trajectory
    stats = {
        'total': 0,
        'first_step_match_omp': 0,
        'first_step_match_g': 0,
        'gt_expert_first_omp': 0,
        'gt_expert_first_g': 0,
        'gt_expert_in_seq_omp': 0,
        'gt_expert_in_seq_g': 0,
        'full_seq_match_omp': 0,
        'full_seq_match_g': 0,
        'omp_vs_g_first_agree': 0,
        'omp_vs_g_seq_agree': 0,
    }
    
    K = manifest['K']
    fs = manifest['fs']
    n_fft = manifest['n_fft']
    freq_min = manifest['freq_min']
    freq_max = manifest['freq_max']
    
    print("Running verification for both OMP and G-teacher...")
    print()
    
    for i, traj in enumerate(trajectories):
        path = traj['path']
        angle_deg = traj['angle_deg']
        angle_idx = traj['angle_index']
        recorded_steps = traj['steps']
        
        # Load Y
        Y = load_band_spectrogram_from_npy(Path(path), fs, n_fft, freq_min, freq_max)
        y = Y.mean(dim=1)
        y = y / (y.norm() + 1e-12)
        
        # Run both teachers
        expert_seq_omp, atom_seq_omp = traditional_omp_forward(D, y, K, E, M)
        expert_seq_g, atom_seq_g = g_teacher_forward(D, y, K, E, M)
        
        # Recorded sequence
        expert_seq_recorded = [s['expert'] for s in recorded_steps]
        atom_seq_recorded = [s['dict_index'] for s in recorded_steps]
        
        stats['total'] += 1
        
        # OMP checks
        if expert_seq_omp[0] == expert_seq_recorded[0]:
            stats['first_step_match_omp'] += 1
        if expert_seq_omp[0] == angle_idx:
            stats['gt_expert_first_omp'] += 1
        if angle_idx in expert_seq_omp:
            stats['gt_expert_in_seq_omp'] += 1
        if expert_seq_omp == expert_seq_recorded:
            stats['full_seq_match_omp'] += 1
        
        # G-teacher checks
        if expert_seq_g[0] == expert_seq_recorded[0]:
            stats['first_step_match_g'] += 1
        if expert_seq_g[0] == angle_idx:
            stats['gt_expert_first_g'] += 1
        if angle_idx in expert_seq_g:
            stats['gt_expert_in_seq_g'] += 1
        if expert_seq_g == expert_seq_recorded:
            stats['full_seq_match_g'] += 1
        
        # OMP vs G comparison
        if expert_seq_omp[0] == expert_seq_g[0]:
            stats['omp_vs_g_first_agree'] += 1
        if expert_seq_omp == expert_seq_g:
            stats['omp_vs_g_seq_agree'] += 1
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1}/{len(trajectories)} samples...")
    
    print()
    print("=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    print()
    
    total = stats['total']
    
    print(f"Total samples: {total}")
    print()
    
    print("Traditional OMP Teacher:")
    print(f"  First-step match with recorded: {stats['first_step_match_omp']}/{total} = {100*stats['first_step_match_omp']/total:.1f}%")
    print(f"  GT expert at first step: {stats['gt_expert_first_omp']}/{total} = {100*stats['gt_expert_first_omp']/total:.1f}%")
    print(f"  GT expert appears in sequence: {stats['gt_expert_in_seq_omp']}/{total} = {100*stats['gt_expert_in_seq_omp']/total:.1f}%")
    print(f"  Full sequence match: {stats['full_seq_match_omp']}/{total} = {100*stats['full_seq_match_omp']/total:.1f}%")
    print()
    
    print("G-Teacher (for comparison):")
    print(f"  First-step match with recorded: {stats['first_step_match_g']}/{total} = {100*stats['first_step_match_g']/total:.1f}%")
    print(f"  GT expert at first step: {stats['gt_expert_first_g']}/{total} = {100*stats['gt_expert_first_g']/total:.1f}%")
    print(f"  GT expert appears in sequence: {stats['gt_expert_in_seq_g']}/{total} = {100*stats['gt_expert_in_seq_g']/total:.1f}%")
    print(f"  Full sequence match: {stats['full_seq_match_g']}/{total} = {100*stats['full_seq_match_g']/total:.1f}%")
    print()
    
    print("OMP vs G-Teacher:")
    print(f"  First-step agreement: {stats['omp_vs_g_first_agree']}/{total} = {100*stats['omp_vs_g_first_agree']/total:.1f}%")
    print(f"  Full sequence agreement: {stats['omp_vs_g_seq_agree']}/{total} = {100*stats['omp_vs_g_seq_agree']/total:.1f}%")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    omp_acc = 100*stats['gt_expert_first_omp']/total
    g_acc = 100*stats['gt_expert_first_g']/total
    
    if omp_acc >= 95:
        print(f"✅ Traditional OMP achieves excellent first-step accuracy: {omp_acc:.1f}%")
    elif omp_acc >= 80:
        print(f"⚠️  Traditional OMP achieves good first-step accuracy: {omp_acc:.1f}%")
    else:
        print(f"❌ Traditional OMP has low first-step accuracy: {omp_acc:.1f}%")
    
    print()
    
    diff = g_acc - omp_acc
    if abs(diff) < 1:
        print(f"📊 G-teacher and OMP have similar accuracy (diff: {diff:.1f}%)")
        print("   → Hierarchical structure provides minimal benefit for this task")
    elif diff > 0:
        print(f"📊 G-teacher outperforms OMP by {diff:.1f}%")
        print("   → Hierarchical expert-first selection improves accuracy")
    else:
        print(f"📊 OMP outperforms G-teacher by {-diff:.1f}%")
        print("   → Pure correlation-based selection is more effective")
    
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
