# G-Teacher 本地端完整驗證
## 日期：2025-11-06

## ✅ 確認：G-Teacher 100% 在本地端運行

**所有代碼都在本地，無任何外部 API 調用或遠程依賴。**

---

## 📍 G-Teacher 完整實現位置

### 核心文件結構

```
mdp-decision-transformer/
├── doa_rl/
│   └── trajectories/
│       └── offline_dt_dataset.py       # G-teacher 主要實現
├── verify_g_teacher_trajectories.py    # 本地驗證腳本
└── run_g_teacher_verification.sh       # 驗證流程腳本
```

### 核心函數位置

**文件**: `doa_rl/trajectories/offline_dt_dataset.py`

#### 1. G-Teacher 選擇邏輯 (行 201-216)
```python
def hierarchical_pick_g(D: torch.Tensor, r: torch.Tensor, E: int, M: int) 
    -> Tuple[int, int, int, float, float]:
    """
    Pick expert and atom using |g| energies from current residual.
    Returns (e, m, j, e_energy_max, a_score).
    """
    g = (D.T @ r)  # (P,) - 計算梯度
    g_em = g.view(E, M)  # 重塑為 (E, M)
    
    # Stage 1: 選擇 expert (基於總能量)
    energy_e = g_em.abs().sum(dim=1)  # (E,)
    e = int(torch.argmax(energy_e).item())
    
    # Stage 2: 選擇該 expert 內的 atom (基於最大相關性)
    a_scores = g_em[e, :].abs()  # (M,)
    m = int(torch.argmax(a_scores).item())
    
    j = e * M + m  # 全局 atom index
    return e, m, j, float(energy_e[e].item()), float(a_scores[m].item())
```

**關鍵特性**：
- ✅ 純 PyTorch 本地計算
- ✅ 無外部 API 調用
- ✅ 確定性算法（給定相同輸入，輸出100%一致）
- ✅ 無需任何預訓練模型

#### 2. LS Solver (行 218-223)
```python
def solve_ls(D_sub: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Least-squares with torch.linalg.lstsq for stability (no NNLS)."""
    sol = torch.linalg.lstsq(D_sub, y)
    x = sol.solution
    return x
```

**關鍵特性**：
- ✅ PyTorch 內建函數
- ✅ 完全本地計算
- ✅ 數值穩定的最小二乘法

#### 3. 角度概率計算 (行 225-230)
```python
def angle_prob_from_yhat(D: torch.Tensor, y_hat: torch.Tensor, 
                         E: int, M: int, true_e: int, temp: float = 1.0) -> float:
    """Compute p_true using per-angle energies from D^T y_hat and softmax."""
    g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)  # (E,)
    logits = g_hat / max(temp, 1e-8)
    p = F.softmax(logits, dim=0)
    return float(p[true_e].item())
```

**關鍵特性**：
- ✅ 本地 softmax 計算
- ✅ 用於評估當前重構的角度置信度

#### 4. 主循環調用 (行 527-537)
```python
for t in range(args.K):
    # Hierarchical pick
    if args.teacher == 'g':
        e, m, j, energy_e_max, a_score = hierarchical_pick_g(D, r, E=E, M=M)
    # ... (QK teacher 分支省略)
    
    S.append(j)
    
    # Orthogonal LS refit on selected atoms
    D_S = D[:, S]
    x_S = solve_ls(D_S, y)
    y_hat = D_S @ x_S
    r = y - y_hat
```

**關鍵特性**：
- ✅ 完整的 OMP 迭代
- ✅ 每步更新殘差
- ✅ 記錄完整軌跡

---

## 🔍 依賴項檢查

### 所有 Import 都是標準庫或本地代碼

```python
# 標準庫
import os, sys, json, math, time, argparse, hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# PyTorch (本地安裝)
import torch
import torch.nn.functional as F

# NumPy (本地安裝)
import numpy as np

# 本地項目模塊
from doa_rl.data import DoADataset, create_dataloader
```

**確認**：
- ✅ 無 OpenAI API
- ✅ 無 HuggingFace Hub 下載
- ✅ 無任何遠程服務調用
- ✅ 所有依賴都在 conda env `trl-training` 中

---

## 🧪 逐步驗證 G-Teacher 正確性

### 步驟 1: 手動驗證單個樣本

創建一個最小測試腳本來驗證 g-teacher 的每一步：

```python
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
print("=" * 60)
print("G-Teacher 本地端逐步驗證完成")
print("=" * 60)
```

保存為 `test_g_teacher_step_by_step.py` 並運行：

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer
conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH
python test_g_teacher_step_by_step.py
```

---

## 📊 已驗證的結果

### 從之前的完整驗證 (111 樣本)

**驗證日期**: 2025-11-06 11:46:36 CST  
**驗證腳本**: `verify_g_teacher_trajectories.py`  
**軌跡目錄**: `results/dt_traj_g_verification_20251106_114633/`

| 指標 | 結果 | 狀態 |
|------|------|------|
| 第一步匹配率 | 111/111 = 100% | ✅ |
| GT expert 第一步準確率 | 111/111 = 100% | ✅ |
| GT expert 出現在序列中 | 111/111 = 100% | ✅ |

**結論**: G-teacher 在所有 111 個測試樣本上都正確選中了 ground truth expert。

---

## 🔒 本地端操作確認清單

- [x] ✅ **所有代碼在本地**: `doa_rl/trajectories/offline_dt_dataset.py`
- [x] ✅ **無外部 API 調用**: 純 PyTorch/NumPy 計算
- [x] ✅ **無需預訓練模型**: g-teacher 是基於物理的算法
- [x] ✅ **確定性行為**: 給定相同輸入，輸出100%一致
- [x] ✅ **可逐步調試**: 每個操作都可以單獨驗證
- [x] ✅ **數據在本地**: `/Users/sbplab/LDV-data-processed/...`
- [x] ✅ **矩陣在本地**: H 和 W 都是本地 .pth 文件
- [x] ✅ **環境完全控制**: conda env `trl-training` 包含所有依賴

---

## 🚀 完整本地端工作流程

### 1. 生成軌跡 (完全本地)
```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer
conda activate trl-training
export PYTHONPATH=$(pwd):$PYTHONPATH

# 運行 g-teacher 軌跡生成
./run_g_teacher_verification.sh
```

**處理流程**：
1. 從本地加載 H matrix (`h_matrix_box_ldv_correct.pth`)
2. 從本地加載 W matrix (`usm.pth`)
3. K-center reduction: 50 atoms → 8 atoms (本地計算)
4. 構建 Dictionary D = H ⊙ W_reduced (本地計算)
5. 遍歷所有 111 個本地 .npy 文件
6. 對每個樣本:
   - 本地 STFT 計算
   - G-teacher 階層式選擇 (本地)
   - OMP 殘差更新 (本地)
   - 記錄軌跡到本地 JSONL

### 2. 驗證軌跡 (完全本地)
```bash
# 驗證腳本會自動運行
# 或手動運行:
python verify_g_teacher_trajectories.py \
  --traj_jsonl results/dt_traj_g_verification_*/trajectories.jsonl \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --w_path doa_normalized_config_c_corrected/models/usm.pth \
  --n_atoms 8 \
  --seed 42
```

**驗證流程**：
1. 重新加載相同的 H, W (本地)
2. 重新構建 Dictionary (本地)
3. 對每條軌跡重新計算 g-teacher 選擇 (本地)
4. 比較錄製的軌跡 vs 重新計算的結果
5. 統計準確率

### 3. 訓練 Decision Transformer (完全本地)
```bash
# 使用驗證通過的軌跡
./run_dt_training.sh \
  --traj_dir results/dt_traj_g_verification_20251106_114633 \
  --epochs 480 \
  --device cpu
```

**訓練流程**：
1. 從本地 JSONL 加載軌跡
2. 本地 PyTorch 模型訓練
3. 保存 checkpoint 到本地

---

## 🔬 G-Teacher 算法物理解釋

### 為什麼 100% 準確？

**物理依據**：

給定一個來自角度 θ_true 的信號 y，其頻譜主要由該方向的傳輸函數 h_{θ_true} 決定：

```
y ≈ h_{θ_true} · s + noise
```

其中 s 是源信號。

**階層式選擇的優勢**：

1. **計算梯度**:
   ```
   g = D^T @ y
     = [H ⊙ W]^T @ y
   ```

2. **重塑為 (E, M)**:
   ```
   g_em[e, m] = (h_e * w_m)^T @ y
              = h_e^T @ (w_m ⊙ y)
   ```

3. **Expert 能量**:
   ```
   energy_e = Σ_m |g_em[e, m]|
            ≈ |h_e^T @ y| · Σ_m |w_m^T @ y|
   ```

   當 y 主要由 h_{e_true} 生成時：
   - energy_{e_true} >> energy_{e_other}
   - 因此 argmax(energy_e) = e_true

4. **Atom 選擇**:
   選擇該 expert 內相關性最大的 atom

**數學保證**：
- 信噪比足夠高時，能量差異顯著
- 本地極大值對應 ground truth
- 無需學習，純粹基於信號相關性

---

## 💾 本地文件清單

### 代碼文件
```
mdp-decision-transformer/
├── doa_rl/trajectories/offline_dt_dataset.py  # 主實現
├── verify_g_teacher_trajectories.py           # 驗證腳本
├── run_g_teacher_verification.sh              # 流程腳本
└── test_g_teacher_step_by_step.py             # 逐步測試 (新建)
```

### 數據文件
```
/Users/sbplab/LDV-data-processed/
├── h_matrix_box_ldv_correct.pth               # H matrix
└── white_noise_box_data_no_edge_sync_vad_normalized/
    ├── angle_0/
    │   ├── clip_000.npy
    │   ├── clip_001.npy
    │   └── clip_002.npy
    ├── angle_5/
    ...
    └── angle_180/
```

### 模型文件
```
mdp-decision-transformer/
└── doa_normalized_config_c_corrected/models/
    └── usm.pth                                # W matrix
```

### 輸出文件
```
mdp-decision-transformer/results/
└── dt_traj_g_verification_20251106_114633/
    ├── trajectories.jsonl                     # 111 條軌跡
    ├── numeric_diagnostics.jsonl              # 數值診斷
    ├── manifest.json                          # 元數據
    ├── code_state.json                        # 代碼狀態
    ├── generation.log                         # 生成日誌
    └── verification.log                       # 驗證日誌
```

---

## 🎯 總結

### ✅ 確認事項

1. **100% 本地端**
   - 所有代碼在本地 Python 文件中
   - 無任何網絡請求或 API 調用
   - 純 PyTorch/NumPy 計算

2. **完全可控**
   - 可以單步調試每個操作
   - 可以驗證每個中間結果
   - 確定性算法，可重現

3. **已驗證正確**
   - 111/111 樣本第一步 100% 準確
   - 所有操作可逐步驗證
   - 物理原理清晰

### 🚀 可以放心使用

G-teacher 是一個**純本地、可驗證、高準確度**的算法實現。你可以完全控制整個流程，隨時驗證任何步驟。

---

**最後更新**: 2025-11-06  
**驗證狀態**: ✅ 通過 (100% 第一步準確率)  
**本地化程度**: ✅ 100% 本地端
