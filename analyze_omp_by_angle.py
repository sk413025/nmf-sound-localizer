#!/usr/bin/env python3
"""
Detailed analysis of OMP vs G-teacher accuracy by angle.
"""

import json
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor
from verify_omp_teacher_trajectories import (
    load_band_spectrogram_from_npy,
    build_dictionary,
    traditional_omp_forward,
    g_teacher_forward,
)


def main():
    traj_dir = Path("results/dt_traj_omp_verification_20251106_120030")
    manifest_path = traj_dir / "manifest.json"
    trajectories_path = traj_dir / "trajectories.jsonl"
    
    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Load matrices
    H, angles = load_H("/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth")
    W = load_W("doa_normalized_config_c_corrected/models/usm.pth")
    
    # Build dictionary
    D, E, M = build_dictionary(
        H, W,
        n_atoms=manifest['M'],
        atom_reduce_mode=manifest['atom_reduce']['mode'],
        seed=manifest['seed']
    )
    
    # Load trajectories
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            trajectories.append(json.loads(line))
    
    K = manifest['K']
    fs = manifest['fs']
    n_fft = manifest['n_fft']
    freq_min = manifest['freq_min']
    freq_max = manifest['freq_max']
    
    # Analyze by angle
    angle_stats = defaultdict(lambda: {'total': 0, 'omp_correct': 0, 'g_correct': 0})
    
    for traj in trajectories:
        path = traj['path']
        angle_deg = int(traj['angle_deg'])
        angle_idx = traj['angle_index']
        
        # Load Y
        Y = load_band_spectrogram_from_npy(Path(path), fs, n_fft, freq_min, freq_max)
        y = Y.mean(dim=1)
        y = y / (y.norm() + 1e-12)
        
        # Run both teachers
        expert_seq_omp, _ = traditional_omp_forward(D, y, K, E, M)
        expert_seq_g, _ = g_teacher_forward(D, y, K, E, M)
        
        angle_stats[angle_deg]['total'] += 1
        if expert_seq_omp[0] == angle_idx:
            angle_stats[angle_deg]['omp_correct'] += 1
        if expert_seq_g[0] == angle_idx:
            angle_stats[angle_deg]['g_correct'] += 1
    
    # Print results
    print("=" * 80)
    print("First-Step Accuracy by Angle")
    print("=" * 80)
    print(f"{'Angle (°)':<12} {'Samples':<10} {'OMP Acc':<12} {'G-Teacher Acc':<15} {'Diff':<10}")
    print("-" * 80)
    
    for angle in sorted(angle_stats.keys()):
        stats = angle_stats[angle]
        omp_acc = 100 * stats['omp_correct'] / stats['total']
        g_acc = 100 * stats['g_correct'] / stats['total']
        diff = g_acc - omp_acc
        status = "✓" if omp_acc >= 66 else "✗"
        print(f"{angle:<12} {stats['total']:<10} {omp_acc:>6.1f}% {status:<5} {g_acc:>6.1f}%         {diff:>+6.1f}%")
    
    # Overall statistics
    total = sum(s['total'] for s in angle_stats.values())
    omp_total = sum(s['omp_correct'] for s in angle_stats.values())
    g_total = sum(s['g_correct'] for s in angle_stats.values())
    
    print("-" * 80)
    print(f"{'OVERALL':<12} {total:<10} {100*omp_total/total:>6.1f}%       {100*g_total/total:>6.1f}%         {100*(g_total-omp_total)/total:>+6.1f}%")
    print("=" * 80)
    print()
    print("Key findings:")
    print(f"  - Traditional OMP: {omp_total}/{total} = {100*omp_total/total:.1f}% first-step accuracy")
    print(f"  - G-Teacher: {g_total}/{total} = {100*g_total/total:.1f}% first-step accuracy")
    print(f"  - Performance gap: {100*(g_total-omp_total)/total:.1f}%")
    print()
    
    # Count perfect vs imperfect angles
    perfect_omp = sum(1 for s in angle_stats.values() if s['omp_correct'] == s['total'])
    perfect_g = sum(1 for s in angle_stats.values() if s['g_correct'] == s['total'])
    
    print(f"Angles with 100% accuracy:")
    print(f"  - OMP: {perfect_omp}/{len(angle_stats)} angles")
    print(f"  - G-Teacher: {perfect_g}/{len(angle_stats)} angles")
    print()


if __name__ == "__main__":
    main()
