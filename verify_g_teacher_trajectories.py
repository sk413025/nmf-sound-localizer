#!/usr/bin/env python3
"""
驗證 G-Teacher 軌跡準確性
從生成的軌跡 JSONL 文件中讀取，並驗證每一步的選擇是否符合預期

使用方式：
1. 先用 REPRODUCE_dt_traj_g_full.sh 生成軌跡
2. 用此腳本驗證軌跡的正確性
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
    """從 .npy 檔案載入頻段限定的 STFT spectrogram"""
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


def g_teacher_forward(D: torch.Tensor, y: torch.Tensor, K: int, E: int, M: int) -> Tuple[List[int], List[int]]:
    """
    G-Teacher OMP: 階層式選擇
    1. 先選 expert (基於 total |g| energy per expert)
    2. 再選 atom (基於 max |g| within expert)
    
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
    """從路徑解析 ground-truth 角度"""
    p = Path(path)
    angle_dir = p.parent.name
    if not angle_dir.startswith("angle_"):
        raise RuntimeError(f"Cannot parse angle from path: {path}")
    try:
        angle = int(angle_dir.split('_')[1])
        return angle
    except (IndexError, ValueError) as e:
        raise RuntimeError(f"Invalid angle directory format: {angle_dir}") from e


def angle_to_expert_idx(angle_deg: int) -> int:
    """將角度轉換為 expert index (0-based)"""
    # 假設 0° -> expert 0, 5° -> expert 1, ..., 180° -> expert 36
    expert_idx = angle_deg // 5
    if expert_idx < 0 or expert_idx > 36:
        raise ValueError(f"Invalid angle: {angle_deg}°")
    return expert_idx


def main():
    ap = argparse.ArgumentParser(description="驗證 G-Teacher 軌跡準確性")
    ap.add_argument("--traj_jsonl", type=str, required=True,
                    help="Path to trajectories.jsonl")
    ap.add_argument("--h_path", type=str, required=True,
                    help="Path to H matrix")
    ap.add_argument("--w_path", type=str, required=True,
                    help="Path to W matrix")
    ap.add_argument("--n_atoms", type=int, default=8,
                    help="Number of atoms per expert (after reduction)")
    ap.add_argument("--atom_reduce_mode", type=str, default="kcenter",
                    choices=["kcenter", "kmeans"],
                    help="Atom reduction method")
    ap.add_argument("--fs", type=int, default=16000,
                    help="Sampling rate")
    ap.add_argument("--n_fft", type=int, default=2048,
                    help="FFT size")
    ap.add_argument("--freq_min", type=float, default=300.0,
                    help="Min frequency")
    ap.add_argument("--freq_max", type=float, default=3000.0,
                    help="Max frequency")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for atom reduction")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Max samples to verify (0=all)")
    args = ap.parse_args()
    
    print("=" * 80)
    print("G-Teacher 軌跡驗證")
    print("=" * 80)
    print(f"軌跡文件: {args.traj_jsonl}")
    print(f"H matrix: {args.h_path}")
    print(f"W matrix: {args.w_path}")
    print(f"Atoms per expert: {args.n_atoms} (via {args.atom_reduce_mode})")
    print(f"STFT: fs={args.fs}, n_fft={args.n_fft}, band=[{args.freq_min}, {args.freq_max}] Hz")
    print(f"Random seed: {args.seed}")
    print()
    
    # Load matrices
    print("Loading H and W matrices...")
    H, angles = load_H(args.h_path)
    W = load_W(args.w_path)
    print(f"  H shape: {H.shape}")
    print(f"  W shape: {W.shape}")
    print(f"  Angles: {angles.shape} (range: {angles.min():.0f}° - {angles.max():.0f}°)")
    
    # Build dictionary
    print(f"Building dictionary with {args.n_atoms} atoms per expert...")
    D, E, M = build_dictionary(H, W, n_atoms=args.n_atoms, atom_reduce_mode=args.atom_reduce_mode, seed=args.seed)
    P = D.shape[1]
    print(f"  Dictionary D shape: {D.shape}")
    print(f"  E={E} experts, M={M} atoms/expert, P={P} total atoms")
    print()
    
    # Load trajectories
    print("Loading trajectories...")
    trajectories = []
    with open(args.traj_jsonl, 'r') as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"  Total trajectories: {len(trajectories)}")
    
    if args.max_samples > 0:
        trajectories = trajectories[:args.max_samples]
        print(f"  Limiting to first {args.max_samples} samples")
    print()
    
    # Verification statistics
    total_samples = len(trajectories)
    first_step_matches = 0
    all_steps_match = 0
    any_step_correct = 0
    expert_sequence_matches = 0
    
    print("開始驗證每條軌跡...")
    print()
    
    for idx, traj in enumerate(trajectories):
        y_path = traj["path"]
        gt_angle = int(traj["angle_deg"])
        gt_expert = angle_to_expert_idx(gt_angle)
        
        # Extract expert sequence from trajectory steps
        recorded_experts = [step["expert"] for step in traj["steps"]]
        K = len(recorded_experts)
        
        # Load Y spectrogram
        Y_mag = load_band_spectrogram_from_npy(
            Path(y_path),
            fs=args.fs,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max
        )
        y = Y_mag.mean(dim=1)  # Average over time
        
        # Re-run g-teacher
        computed_experts, computed_atoms = g_teacher_forward(D, y, K, E, M)
        
        # Compare
        first_match = (recorded_experts[0] == computed_experts[0])
        sequence_match = (recorded_experts == computed_experts)
        gt_first_match = (recorded_experts[0] == gt_expert)
        gt_in_sequence = (gt_expert in recorded_experts)
        
        if first_match:
            first_step_matches += 1
        if sequence_match:
            all_steps_match += 1
        if gt_first_match:
            any_step_correct += 1
        if gt_in_sequence:
            expert_sequence_matches += 1
        
        # Print detailed info for failed cases
        if not sequence_match or not gt_first_match:
            print(f"[{idx+1}/{total_samples}] {Path(y_path).name} (GT angle={gt_angle}°, expert={gt_expert})")
            print(f"  Recorded:  {recorded_experts}")
            print(f"  Computed:  {computed_experts}")
            if not sequence_match:
                print(f"  ⚠️  序列不匹配!")
            if not gt_first_match:
                print(f"  ⚠️  第一步未選中 GT expert (選了 {recorded_experts[0]} 而非 {gt_expert})")
            print()
    
    # Summary
    print("=" * 80)
    print("驗證結果摘要")
    print("=" * 80)
    print(f"總樣本數: {total_samples}")
    print()
    print("軌跡一致性檢查 (recorded vs re-computed):")
    print(f"  第一步匹配: {first_step_matches}/{total_samples} = {100*first_step_matches/total_samples:.1f}%")
    print(f"  完整序列匹配: {all_steps_match}/{total_samples} = {100*all_steps_match/total_samples:.1f}%")
    print()
    print("G-Teacher 準確性 (vs ground truth):")
    print(f"  第一步選中 GT expert: {any_step_correct}/{total_samples} = {100*any_step_correct/total_samples:.1f}%")
    print(f"  GT expert 出現在序列中: {expert_sequence_matches}/{total_samples} = {100*expert_sequence_matches/total_samples:.1f}%")
    print()
    
    if all_steps_match == total_samples:
        print("✅ 所有軌跡序列完全匹配! G-teacher 實現正確。")
    else:
        print(f"⚠️  有 {total_samples - all_steps_match} 條軌跡序列不匹配!")
        print("   可能原因:")
        print("   1. STFT 參數不同")
        print("   2. Atom reduction seed 不同")
        print("   3. G-teacher 實現邏輯差異")
    
    if any_step_correct < total_samples * 0.5:
        print()
        print(f"⚠️  G-teacher 第一步準確率僅 {100*any_step_correct/total_samples:.1f}%")
        print("   這與 mdp-diverse-policies 聲稱的 100% 有明顯差距")
        print("   可能需要檢查:")
        print("   1. 數據預處理是否一致")
        print("   2. Dictionary 構建方式")
        print("   3. G-teacher 選擇邏輯")
    
    print("=" * 80)
    
    # Exit code based on results
    if all_steps_match < total_samples:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
