#!/usr/bin/env python3
"""最小化 G-teacher 測試 - 逐步驗證每個操作"""

import torch
import numpy as np
from pathlib import Path

# ===== 步驟 1: 加載矩陣 =====
print("=" * 60)
print("步驟 1: 加載 H 和 W 矩陣")
print("=" * 60)

h_path = "/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
w_path = "doa_normalized_config_c_corrected/models/usm.pth"

H_data = torch.load(h_path, map_location='cpu', weights_only=False)
W_data = torch.load(w_path, map_location='cpu', weights_only=False)

H = H_data['H'] if isinstance(H_data, dict) else H_data
W = W_data['W'] if isinstance(W_data, dict) else W_data

print(f"✓ H shape: {H.shape}")
print(f"✓ W shape: {W.shape}")
print()

# ===== 步驟 2: Atom Reduction =====
print("=" * 60)
print("步驟 2: K-center Atom Reduction (M=50 → M=8)")
print("=" * 60)

def simple_kcenter(X, n_centers, seed=42):
    torch.manual_seed(seed)
    N = X.shape[0]
    centers = []
    indices = []
    
    first_idx = torch.randint(0, N, (1,)).item()
    centers.append(X[first_idx])
    indices.append(first_idx)
    
    for _ in range(1, n_centers):
        distances = torch.stack([torch.norm(X - c, dim=1) for c in centers])
        min_distances = distances.min(dim=0)[0]
        farthest_idx = min_distances.argmax().item()
        centers.append(X[farthest_idx])
        indices.append(farthest_idx)
    
    return torch.stack(centers)

W_T = W.T  # (50, 346)
W_centers = simple_kcenter(W_T, n_centers=8, seed=42)
W_reduced = W_centers.T  # (346, 8)
W_reduced = W_reduced / (W_reduced.norm(dim=0, keepdim=True) + 1e-12)

print(f"✓ W_reduced shape: {W_reduced.shape}")
print(f"✓ Reduction: 50 atoms → 8 atoms")
print()

# ===== 步驟 3: 構建 Dictionary =====
print("=" * 60)
print("步驟 3: 構建 Dictionary D = H ⊙ W_reduced")
print("=" * 60)

E = H.shape[1]  # 37 experts
M = W_reduced.shape[1]  # 8 atoms
P = E * M  # 296 total atoms

D = torch.zeros(H.shape[0], P)
for e in range(E):
    for m in range(M):
        D[:, e * M + m] = H[:, e] * W_reduced[:, m]

D = D / (D.norm(dim=0, keepdim=True) + 1e-12)

print(f"✓ Dictionary D shape: {D.shape}")
print(f"✓ E={E} experts, M={M} atoms/expert, P={P} total")
print()

# ===== 步驟 4: 加載測試樣本 =====
print("=" * 60)
print("步驟 4: 加載測試樣本 (angle_0/clip_000.npy)")
print("=" * 60)

from nmf_localizer.utils.audio_utils import AudioProcessor

test_file = "/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/angle_0/clip_000.npy"
wav = np.load(test_file)

freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
    wav, fs=16000, nperseg=2048, window="hann"
)
mask = (freqs >= 300.0) & (freqs <= 3000.0)
Y_mag = magnitude[mask, :].astype(np.float32)
Y = torch.from_numpy(Y_mag)

y = Y.mean(dim=1)  # 時間平均
y = y / (y.norm() + 1e-12)  # 正規化

print(f"✓ Y shape: {Y.shape}")
print(f"✓ y shape: {y.shape}")
print(f"✓ Ground truth: angle 0° → expert 0")
print()

# ===== 步驟 5: G-Teacher 第一步選擇 =====
print("=" * 60)
print("步驟 5: G-Teacher 階層式選擇 (第一步)")
print("=" * 60)

r = y.clone()

# 計算梯度
g = D.T @ r  # (296,)
g_em = g.view(E, M)  # (37, 8)

print(f"✓ Gradient g shape: {g.shape}")
print(f"✓ Reshaped g_em shape: {g_em.shape}")
print()

# Stage 1: 選擇 expert
energy_e = g_em.abs().sum(dim=1)  # (37,)
e_selected = int(torch.argmax(energy_e).item())

print(f"Expert energies (top 5):")
top5_experts = torch.argsort(energy_e, descending=True)[:5]
for rank, e_idx in enumerate(top5_experts):
    angle = e_idx * 5
    energy = energy_e[e_idx].item()
    marker = " ← SELECTED" if e_idx == e_selected else ""
    print(f"  Rank {rank+1}: Expert {e_idx:2d} (angle {angle:3d}°) - energy={energy:.4f}{marker}")
print()

# Stage 2: 選擇 atom
a_scores = g_em[e_selected, :].abs()  # (8,)
m_selected = int(torch.argmax(a_scores).item())

print(f"Atom scores within expert {e_selected}:")
for m_idx in range(M):
    score = a_scores[m_idx].item()
    marker = " ← SELECTED" if m_idx == m_selected else ""
    print(f"  Atom {m_idx}: score={score:.4f}{marker}")
print()

j_selected = e_selected * M + m_selected

print(f"Final selection:")
print(f"  Expert: {e_selected} (angle {e_selected * 5}°)")
print(f"  Atom: {m_selected}")
print(f"  Global index: {j_selected}")
print()

# ===== 步驟 6: 驗證結果 =====
print("=" * 60)
print("步驟 6: 驗證結果")
print("=" * 60)

gt_expert = 0  # angle_0 → expert 0
if e_selected == gt_expert:
    print(f"✅ SUCCESS! G-teacher 正確選中 GT expert {gt_expert}")
else:
    print(f"❌ FAIL! 選中 expert {e_selected}，應該是 {gt_expert}")

print()

# ===== 額外測試：多個角度 =====
print("=" * 60)
print("額外測試：驗證其他角度")
print("=" * 60)

test_angles = [0, 45, 90, 135, 180]
all_correct = True

for angle_deg in test_angles:
    test_file = f"/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/angle_{angle_deg}/clip_000.npy"
    
    if not Path(test_file).exists():
        print(f"⚠️  文件不存在: {test_file}")
        continue
    
    wav = np.load(test_file)
    freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=16000, nperseg=2048, window="hann"
    )
    mask = (freqs >= 300.0) & (freqs <= 3000.0)
    Y_mag = magnitude[mask, :].astype(np.float32)
    Y = torch.from_numpy(Y_mag)
    
    y = Y.mean(dim=1)
    y = y / (y.norm() + 1e-12)
    
    r = y.clone()
    g = D.T @ r
    g_em = g.view(E, M)
    energy_e = g_em.abs().sum(dim=1)
    e_selected = int(torch.argmax(energy_e).item())
    
    gt_expert = angle_deg // 5
    
    status = "✅" if e_selected == gt_expert else "❌"
    print(f"{status} Angle {angle_deg:3d}° → Expected expert {gt_expert:2d}, Got {e_selected:2d}")
    
    if e_selected != gt_expert:
        all_correct = False

print()
print("=" * 60)
print("最終結果")
print("=" * 60)

if all_correct:
    print("✅ 所有測試角度都正確!")
    print("✅ G-Teacher 本地端實現完全正確")
else:
    print("⚠️  部分角度有誤，需要進一步檢查")

print()
print("=" * 60)
print("G-Teacher 本地端逐步驗證完成")
print("=" * 60)
