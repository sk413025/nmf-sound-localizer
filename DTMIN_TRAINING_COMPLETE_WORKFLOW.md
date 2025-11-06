# DTMin 完整訓練流程文件

**撰寫日期**: 2025-11-04
**版本**: 2.0
**專案**: DTMin Physics-Informed Decision Transformer

---

## 📋 目錄

1. [執行摘要](#執行摘要)
2. [系統架構總覽](#系統架構總覽)
3. [數據流程圖](#數據流程圖)
4. [OMP Trajectory 生成與儲存](#omp-trajectory-生成與儲存)
5. [DTMin 訓練流程](#dtmin-訓練流程)
6. [Token Embedding 架構](#token-embedding-架構)
7. [完整操作指南](#完整操作指南)
8. [數據格式規範](#數據格式規範)
9. [故障排除](#故障排除)

---

## 執行摘要

DTMin 使用 **OMP (Orthogonal Matching Pursuit) Trajectories** 進行訓練。

### 原始數據來源

**參考 Commit**: `abde66a` (mdp-decision-transformer worktree)

- **原始音頻數據集**:

  ```
  /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/
  ├── angle_0/
  │   ├── clip_000.npy  # (samples,) 原始波形 @ 16kHz
  │   ├── clip_001.npy
  │   └── ... (3 clips per angle)
  ├── angle_5/
  ├── ... (37 angles total: 0°, 5°, 10°, ..., 180°)
  └── angle_180/

  Total: 111 .npy files (37 angles × 3 clips)
  Data fingerprint (MD5): 713c0635878a04b32f4ee30208904d11
  ```
- **物理字典文件**:

  ```
  H matrix: /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
            Shape: (F=346, E=37) - Transfer functions for 37 angles

  W matrix: doa_normalized_config_c_corrected/models/usm.pth
            Shape: (F=346, M=50) - USM dictionary (未縮減)
  ```

### Trajectory 儲存位置

預先生成的 OMP trajectories 統一存放於：

```
/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-diverse-policies/results/
├── policy_library_phase1/              # Phase 1: 基礎配置（無加噪）
│   ├── M8_K6_kmeans_greedy/
│   │   ├── trajectories.jsonl          # OMP trajectory 記錄
│   │   ├── manifest.json               # 完整配置和數據指紋
│   │   ├── numeric_diagnostics.jsonl   # IS divergence 診斷
│   │   └── code_state.json             # Git hash + 文件指紋
│   ├── M8_K6_kcenter_greedy/
│   ├── M8_K6_random_greedy/
│   ├── M4_K3_kmeans_greedy/            # 不同 M 值配置
│   ├── M16_K6_kmeans_greedy/
│   └── ... (26 configurations total)
│
└── policy_library_phase1p5_perturbations/  # Phase 1.5: 擾動測試
    ├── w_noise_std0.01_M8/             # W matrix 加噪音
    ├── w_noise_std0.05_M8/
    ├── w_noise_std0.10_M8/
    ├── h_noise_std0.01_M8/             # H matrix 加噪音
    ├── h_noise_std0.05_M8/
    ├── h_noise_std0.10_M8/
    ├── w_freq_mask_1k-2k_M8/           # W 頻率遮罩
    ├── h_subset_0-90deg_M8/            # H 角度子集
    ├── h_subset_90-180deg_M8/
    ├── combined_w_noise0.05_h_noise0.05_M8/  # 組合擾動
    └── ... (17 perturbation configs)
```

**關鍵特性**:

- **Phase 1**: 26 種基礎配置（不同 M, K, 原子縮減方法）
- **Phase 1.5**: 17 種擾動配置（測試 robustness）
- 預先計算並儲存，提升訓練效率
- 標準化的 JSONL 格式，便於版本控制和重現
- ⚠️ **每個配置的字典維度 P=E×M 可能不同**

**重要**: 這些 trajectories 的原始數據來自 commit `abde66a` (mdp-decision-transformer worktree)，詳細記錄在該 commit 的 `results/dt_traj_kmeans_v3/` 目錄中。

### Phase 1.5 擾動配置詳解

**Phase 1.5** 探討字典擾動對訓練的影響，測試模型的 robustness。這對後續訓練很重要，因為：

1. **不同配置使用不同的 M 值** → 字典維度 P = E × M 會變化
2. **Phase 1.5 對 M 和 H 加噪音** → 需記錄加噪方法以確保一致性
3. **訓練時是否使用相同加噪可能影響 transformer 學習**

#### 擾動類型與實現

所有擾動都在 trajectory 生成時應用，記錄在各配置的 `manifest.json` 中：

```json
{
  "atom_reduce": {
    "mode": "kmeans",
    "selected_indices": null,
    "cluster_sizes": [1, 2, 6, 1, 35, 1, 1, 3]
  }
  // 注意：manifest.json 目前未明確記錄擾動參數
  // 擾動信息隱含在配置名稱中
}
```

**⚠️ 重要發現**:

- Manifest 中**沒有明確的 `perturbation` 或 `atom_perturb` 欄位**
- 擾動參數目前只能從**配置目錄名稱**推斷
- 例如：`w_noise_std0.05_M8` → W matrix 加噪音，std=0.05

**擾動實現位置**:

```
mdp-diverse-policies/doa_rl/trajectories/offline_dt_dataset.py
```

#### 1. W Matrix 噪音擾動 (Spectral Noise)

**生成命令範例**:

```bash
python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher g \
    --atom_reduce_mode kmeans \
    --n_atoms 8 \
    --K 6 \
    --atom_perturb noise \
    --atom_noise_std 0.05 \
    --out_dir results/policy_library_phase1p5_perturbations/w_noise_std0.05_M8
```

**實現細節**:

```python
# in offline_dt_dataset.py
def perturb_atoms_noise(W: torch.Tensor, n_clusters: int = 8, 
                        noise_std: float = 0.1, 
                        base_method: str = 'kmeans', 
                        random_state: int = 42):
    """
    對縮減後的原子字典添加頻譜噪音
  
    步驟:
    1. 使用 base_method (kmeans/kcenter) 縮減原子
    2. 對每個縮減後的原子添加高斯噪音
    3. 重新歸一化
  
    Args:
        W: (F, M_full) 原始字典
        n_clusters: 縮減後的原子數 M
        noise_std: 高斯噪音標準差
        base_method: 原子縮減方法
        random_state: 隨機種子
  
    Returns:
        W_reduced: (F, M) 加噪後的縮減字典
        labels: (M_full,) 原子分配標籤
        info: 包含 noise_std 等信息
    """
    # 1. 先縮減
    if base_method == 'kmeans':
        W_red, labels, info = reduce_atoms_kmeans(W, n_clusters, random_state)
    elif base_method == 'kcenter':
        W_red, labels, info = reduce_atoms_kcenter(W, n_clusters)
    else:
        W_red, labels, info = reduce_atoms_random(W, n_clusters, random_state)
  
    # 2. 添加高斯噪音
    rng = np.random.RandomState(random_state)
    noise = rng.randn(*W_red.shape) * noise_std
    W_noisy = W_red + torch.from_numpy(noise).float()
  
    # 3. 重新歸一化（重要！）
    W_noisy = W_noisy / (W_noisy.norm(dim=0, keepdim=True) + 1e-12)
  
    info['perturbation'] = {
        'type': 'noise',
        'noise_std': noise_std,
        'seed': random_state
    }
  
    return W_noisy, labels, info
```

**物理意義**:

- 模擬字典學習的誤差
- 測試模型對原子表示不精確的容忍度
- std=0.01: 小擾動（~1% 能量）
- std=0.05: 中等擾動（~5% 能量）
- std=0.10: 大擾動（~10% 能量）

#### 2. H Matrix 噪音擾動 (Calibration Error)

**生成命令範例**:

```bash
python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher g \
    --h_noise_std 0.05 \
    --out_dir results/policy_library_phase1p5_perturbations/h_noise_std0.05_M8
```

**實現細節**:

```python
def perturb_H_noise(H: torch.Tensor, angles: List[float], 
                    noise_std: float = 0.05, 
                    random_state: int = 42):
    """
    對 H matrix 添加校準誤差噪音
  
    Args:
        H: (F, E) transfer functions
        angles: E 個角度列表
        noise_std: 噪音標準差
        random_state: 隨機種子
  
    Returns:
        H_noisy: (F, E) 加噪後的 H
        angles: 相同的角度列表
        info: 擾動信息
    """
    rng = np.random.RandomState(random_state)
    noise = rng.randn(*H.shape) * noise_std
    H_noisy = H + torch.from_numpy(noise).float()
  
    # 重新歸一化每個 expert
    H_noisy = H_noisy / (H_noisy.norm(dim=0, keepdim=True) + 1e-12)
  
    info = {
        'perturbation': 'h_noise',
        'noise_std': noise_std,
        'seed': random_state
    }
  
    return H_noisy, angles, info
```

**物理意義**:

- 模擬麥克風陣列校準誤差
- 測試對 transfer function 測量不準確的容忍度

#### 3. 頻率遮罩擾動 (Frequency Masking)

**配置**: `w_freq_mask_1k-2k_M8`, `w_freq_mask_2k-3k_M8` 等

**實現**:

```python
def perturb_atoms_freq_mask(W: torch.Tensor, n_clusters: int = 8,
                            freq_min_mask: float = 1000.0,
                            freq_max_mask: float = 2000.0,
                            freqs: np.ndarray = None,
                            base_method: str = 'kmeans',
                            random_state: int = 42):
    """
    遮罩特定頻率範圍
  
    物理意義：模擬某些頻率範圍的資訊遺失
    """
    # 1. 縮減原子
    W_red, labels, info = reduce_atoms_kmeans(W, n_clusters, random_state)
  
    # 2. 找到需要遮罩的頻率 bin
    mask_idx = (freqs >= freq_min_mask) & (freqs <= freq_max_mask)
  
    # 3. 將對應頻率設為 0
    W_masked = W_red.clone()
    W_masked[mask_idx, :] = 0.0
  
    # 4. 重新歸一化
    W_masked = W_masked / (W_masked.norm(dim=0, keepdim=True) + 1e-12)
  
    info['perturbation'] = {
        'type': 'freq_mask',
        'freq_min_mask': freq_min_mask,
        'freq_max_mask': freq_max_mask
    }
  
    return W_masked, labels, info
```

#### 4. H 角度子集擾動

**配置**: `h_subset_0-90deg_M8`, `h_subset_90-180deg_M8` 等

**實現**: 只使用部分角度的 transfer functions

#### 5. 組合擾動

**配置**: `combined_w_noise0.05_h_noise0.05_M8` 等

同時應用 W 和 H 的噪音擾動。

### ⚠️ 訓練時的重要考量

#### 問題：是否需要在訓練時應用相同的噪音？

**答案：不需要，但要理解差異**

1. **Trajectory 生成時**：

   - 使用加噪的字典 D_noisy 運行 OMP
   - 記錄的 actions 是基於 D_noisy 的最優選擇
   - RTG 等 reward 也是基於 D_noisy 計算
2. **DTMin 訓練時**：

   - 使用**相同的 D_noisy** 初始化 frozen dictionary embeddings
   - 從 trajectories.jsonl 重建 r_seq 時，需要用**相同的 D_noisy**
   - 這保證了物理一致性

#### 如何確保一致性？

**當前實現**（需改進）：

```python
# 問題：manifest.json 沒有明確記錄擾動參數
# 只能從配置名稱推斷

# 建議改進：在 manifest.json 中添加
{
  "perturbation": {
    "w_perturbation": {
      "type": "noise",
      "noise_std": 0.05,
      "seed": 42
    },
    "h_perturbation": null
  }
}
```

**載入時重建**：

```python
# 在 MDPTrajectoryAdapter 或訓練腳本中
def load_perturbed_dictionary(manifest_path, h_path, w_path):
    """
    根據 manifest 重建與 trajectory 生成時相同的字典
    """
    manifest = json.load(open(manifest_path))
  
    # 載入原始字典
    H_orig = torch.load(h_path)
    W_orig = torch.load(w_path)
  
    # 應用與生成時相同的縮減
    atom_reduce = manifest['atom_reduce']
    if atom_reduce['mode'] == 'kmeans':
        W_reduced, _, _ = reduce_atoms_kmeans(
            W_orig, 
            n_clusters=manifest['M'],
            random_state=manifest['seed']
        )
    # ... 其他方法
  
    # ⚠️ 重要：應用與生成時相同的擾動
    # 目前需要從配置名稱推斷
    config_name = Path(manifest_path).parent.name
  
    if 'w_noise_std' in config_name:
        # 解析噪音標準差
        import re
        match = re.search(r'w_noise_std(\d+\.\d+)', config_name)
        if match:
            noise_std = float(match.group(1))
            W_reduced = apply_noise(W_reduced, noise_std, manifest['seed'])
  
    if 'h_noise_std' in config_name:
        match = re.search(r'h_noise_std(\d+\.\d+)', config_name)
        if match:
            noise_std = float(match.group(1))
            H_perturbed = apply_noise(H_orig, noise_std, manifest['seed'])
    else:
        H_perturbed = H_orig
  
    # 構建字典
    D = build_dictionary(H_perturbed, W_reduced)
  
    return D
```

### 建議改進事項

**對於 Phase 1.5（已完成的工作）**：

1. ✅ Trajectories 已生成
2. ⚠️ Manifest 缺少明確的擾動參數記錄
3. ⚠️ 需要從配置名稱解析擾動參數

**對於未來訓練**：

1. 從配置名稱解析擾動參數（臨時方案）
2. 使用相同的 random_seed 重建加噪字典
3. 驗證重建的 D 與生成 trajectory 時一致

**程式碼改進建議**：

```python
# 在 offline_dt_dataset.py 中改進 manifest 記錄
manifest = {
    # ... 現有欄位 ...
    "perturbation": {
        "w_perturbation": {
            "type": args.atom_perturb,  # 'noise', 'freq_mask', 'none'
            "noise_std": args.atom_noise_std if args.atom_perturb == 'noise' else None,
            "freq_range": [args.freq_min_mask, args.freq_max_mask] if args.atom_perturb == 'freq_mask' else None,
            "seed": args.seed
        },
        "h_perturbation": {
            "type": "noise" if args.h_noise_std > 0 else "none",
            "noise_std": args.h_noise_std if args.h_noise_std > 0 else None,
            "seed": args.seed
        } if hasattr(args, 'h_noise_std') else None
    }
}
```

---

## 系統架構總覽

```
┌─────────────────────────────────────────────────────────────────┐
│                        DTMin Training System                     │
└─────────────────────────────────────────────────────────────────┘
                                |
                    ┌───────────▼───────────┐
                    │ OMP Trajectory        │
                    │ Generation            │
                    │ (預先生成並儲存)      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ trajectories.jsonl    │
                    │ (儲存於 mdp-diverse-  │
                    │  policies/results/)   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Load & Collate        │
                    │ (DataLoader)          │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Token Embedding       │
                    │ (r_tok + rtg_tok +    │
                    │  step_tok)            │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ DTMin Transformer     │
                    │ (DTMinCore)           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Hierarchical Action   │
                    │ Prediction            │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │ Trained Model         │
                    │ (checkpoint)          │
                    └───────────────────────┘
```

---

## 數據流程圖

### 階段 1: OMP Trajectory 生成與儲存

```
┌──────────────────────────────────────────────────────────────────┐
│ OMP Trajectory Generation (預先生成)                             │
└──────────────────────────────────────────────────────────────────┘

原始音頻文件 (.npy)
    │
    ├─ Path: /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/
    │
    ▼
┌─────────────────────────────────────┐
│ 1. 加載字典與配置                   │
│                                     │
│ • H matrix (F=346, E=37 experts)   │
│ • W matrix (F=346, M atoms/expert) │
│ • 原子縮減 (kmeans/kcenter/random) │
│                                     │
│ Output: D = H ⊗ W (F, P=E×M)       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 2. 運行 OMP 策略                    │
│                                     │
│ For each audio file:                │
│   • 計算 STFT → Y (F, N)           │
│   • 時間平均 → y (F,)              │
│   • 執行 greedy OMP selection       │
│   • 記錄 actions + rewards          │
│                                     │
│ OMP Algorithm:                      │
│   r_0 = y / ||y||                  │
│   for t = 0 to K-1:                │
│     j_t = argmax |D^T @ r_t|      │
│     r_{t+1} = r_t - proj(r_t)     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 3. 儲存為 trajectories.jsonl        │
│                                     │
│ 儲存位置:                            │
│ mdp-diverse-policies/results/       │
│   policy_library_phase1/            │
│     M8_K6_kmeans_greedy/           │
│       trajectories.jsonl            │
│                                     │
│ 每行一個 trajectory:                │
│ {                                   │
│   "path": "angle_0/clip_000.npy",  │
│   "angle_index": 0,                 │
│   "steps": [                        │
│     {                               │
│       "expert": 0,                  │
│       "atom": 4,                    │
│       "dict_index": 4,              │
│       "resid_sq": 0.0174,          │
│       "p_true": 0.058,             │
│       "rtg_resid": 0.0,            │
│       "rtg_acc": 0.892             │
│     },                              │
│     ...  (K steps)                  │
│   ]                                 │
│ }                                   │
│                                     │
│ ⚠️ 注意: 不儲存 r_seq 完整向量      │
│ 原因: 可從物理方程重建，節省空間    │
└──────────┬──────────────────────────┘
           │
           ▼
     trajectories.jsonl 
     (輕量級，約 1-5 MB vs 完整數據 100+ MB)
```

### 階段 2: Trajectory 載入與 Token Embedding

```
┌──────────────────────────────────────────────────────────────────┐
│ Token Embedding Pipeline (DTMinCore.forward)                     │
└──────────────────────────────────────────────────────────────────┘

trajectories.jsonl
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Load Trajectories                │
│                                     │
│ For each trajectory:                │
│   • 從 JSONL 讀取 metadata          │
│   • 重建 r_seq (從物理方程)         │
│   • 提取 RTG_seq, action_seq        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 2. DataLoader Collation             │
│                                     │
│ batch = {                           │
│   'r_seq': (B, K, F=346),          │ ← 物理空間
│   'RTG_seq': (B, K, 2),            │ ← [rtg_resid, rtg_acc]
│   'STEP_seq': (B, K, 2),           │ ← [t/K, (K-t)/K]
│   'expert_gt': (B, K),             │
│   'atom_gt': (B, K)                │
│ }                                   │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 3. Token Embedding (關鍵步驟!)      │
│                                     │
│ ⚠️ 這裡才是真正儲存的表示!           │
│                                     │
│ For each timestep t:                │
│                                     │
│ Step 3a: 投影到 embedding space     │
│   r_tok = Linear_R(r_t)            │
│           + type_R                  │
│     → (B, K, d_model=128)          │ ← Residual token
│                                     │
│   rtg_tok = Linear_RTG(RTG_t)      │
│     → (B, K, d_model=128)          │ ← RTG token
│                                     │
│   step_tok = Linear_Step(STEP_t)   │
│     → (B, K, d_model=128)          │ ← Step token
│                                     │
│ Step 3b: Token 組合                 │
│   ⚠️ 使用拼接 (concat) 而非加總      │
│                                     │
│   token_concat = concat([           │
│     r_tok,                          │
│     rtg_tok,                        │
│     step_tok                        │
│   ], dim=-1)                        │
│     → (B, K, 3*d_model=384)        │
│                                     │
│ Step 3c: 投影到 Transformer 空間    │
│   token_t = Linear_projection(      │
│     token_concat                    │
│   )                                 │
│     → (B, K, d_model=128)          │ ← 最終 token
│                                     │
│   token_t = LayerNorm(token_t)     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 4. Transformer Encoding             │
│                                     │
│ H = Transformer(token_seq)          │
│     → (B, K, d_model=128)          │
│                                     │
│ ⚠️ 無 action tokens!                │
│ (Markov property: r_t 已包含所有資訊)│
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 5. Hierarchical Action Prediction   │
│                                     │
│ 使用凍結字典 embeddings 作為 Query: │
│                                     │
│ Query_expert = Frozen_H_embeddings  │
│   → (E=37, d_model)                │
│                                     │
│ Query_atom = Frozen_W_embeddings    │
│   → (P=296, d_model)               │
│                                     │
│ Expert scores:                      │
│   expert_logits = H @ Query_expert^T│
│     → (B, K, E=37)                 │
│                                     │
│ Atom scores:                        │
│   atom_logits = H @ Query_atom^T    │
│     → (B, K, P=296)                │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ 6. Hierarchical Loss                │
│                                     │
│ loss_expert = CrossEntropy(          │
│   expert_logits, expert_gt          │
│ )                                   │
│                                     │
│ loss_atom = CrossEntropy(            │
│   atom_logits, atom_gt              │
│ )                                   │
│                                     │
│ loss = loss_expert + loss_atom      │
└──────────┬──────────────────────────┘
           │
           ▼
     Gradient Backpropagation
     (只訓練 Linear 層和 Transformer，
      字典 embeddings 凍結)
```

**關鍵設計決策**:

1. **Token 拼接 (Concat) 而非加總 (Add)**:

   ```python
   # ✅ 正確: 保留所有資訊
   token = concat([r_tok, rtg_tok, step_tok])  # (3*d_model,)
   token = Linear(token)  # → (d_model,)

   # ❌ 錯誤: 資訊損失
   token = r_tok + rtg_tok + step_tok  # (d_model,)
   ```
2. **無 Action Tokens**:

   - 傳統 DT: `[s_0, a_0, s_1, a_1, ...]`
   - DTMin: `[r_0, r_1, r_2, ...]`
   - 原因: Markov property，r_t 已編碼所有歷史資訊
3. **凍結字典 Embeddings**:

   - H, W embeddings 從物理測量預計算
   - 訓練時不更新 (節省 ~50% 參數)
   - 提供強物理歸納偏差

---

## OMP Trajectory 生成與儲存

### 核心代碼文件

**Trajectory 生成**:

```
scripts/omp_trajectory.py
├── class OMPTrajectoryGenerator  # OMP trajectory 生成器
│   ├── __init__(D, E, M, K)      # 初始化字典參數
│   ├── generate_trajectory()      # 生成單個 trajectory
│   └── generate_dataset()         # 批量生成
│
└── def collate_trajectories()     # DataLoader collation function
```

**模型核心**:

```
scripts/dtmin_core.py
├── class DTMinCore(nn.Module)     # DTMin 主模型
│   ├── __init__()                 # 初始化架構
│   │   ├── Linear_R               # Residual projection (F → d_model)
│   │   ├── Linear_RTG             # RTG projection (2 → d_model)
│   │   ├── Linear_Step            # Step projection (2 → d_model)
│   │   ├── token_projection       # Token concat → d_model
│   │   └── transformer            # Transformer encoder
│   │
│   ├── init_dictionary_embeddings(D)  # 初始化並凍結字典 embeddings
│   │   ├── Linear_D (FROZEN)      # Dictionary projection
│   │   └── W_k (FROZEN)           # Key projection (~50% 參數凍結)
│   │
│   └── forward(R_seq, RTG_seq, STEP_seq)
│       ├── Token embedding        # r_tok, rtg_tok, step_tok
│       ├── Token concatenation    # concat (NOT add!)
│       └── Hierarchical prediction # Expert × Atom scores
│
└── def compute_hierarchical_loss()  # 階層式損失函數
```

**訓練腳本**:

```
scripts/train_dtmin_demo.py        # 演示訓練（即時生成 trajectories）
scripts/train_dtmin_real.py        # 真實數據訓練
scripts/train_dtmin.py             # 完整訓練腳本（支持多種配置）

Common components:
├── class TrajectoryDataset(Dataset)  # PyTorch Dataset wrapper
├── def train_epoch()                 # 訓練一個 epoch
├── def evaluate()                    # 評估模型
└── DataLoader(..., collate_fn=collate_trajectories)
```

### Trajectory 儲存策略

**設計理念**: 輕量級儲存 + 物理重建

```python
# ❌ 不儲存: r_seq (K, F) = 完整 residual vectors
# 原因: 可從物理方程重建，每個 trajectory 節省 ~8 KB

# ✅ 儲存: 
# - actions (K,) = action indices
# - rtg_resid, rtg_acc (K, 2) = return-to-go values  
# - resid_sq (K,) = ||r_t||² norms (用於驗證)
# - metadata: path, angle_index
```

### 數據路徑配置

```bash
# Trajectory 儲存位置 (統一標準)
TRAJECTORY_DIR="/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-diverse-policies/results"

# 可用配置 (policy_library_phase1)
mdp-diverse-policies/results/policy_library_phase1/
├── M8_K6_kmeans_greedy/
│   ├── trajectories.jsonl       # OMP trajectory 記錄
│   ├── W_reduced.pth             # 縮減後的字典 (F=346, M=8)
│   ├── config.json               # 配置參數
│   └── metrics.json              # 性能指標
│
├── M8_K6_kcenter_greedy/        # KCenter 原子選擇
├── M8_K6_random_greedy/         # Random 原子選擇
└── ... (更多配置)

# 原始數據集 (用於重建 r_seq)
DATASET_ROOT="/path/to/white_noise_box_data_no_edge_sync_vad_normalized"
white_noise_box_data_no_edge_sync_vad_normalized/
├── angle_0/
│   ├── clip_000.npy  # (samples,) raw waveform
│   ├── clip_001.npy
│   └── ...
├── angle_1/
└── ... (37 angles total)

# 共用字典
H_PATH="h_matrix_normalized_original_to_box.pth"  # (F=346, E=37)
```

### 完整執行流程

```bash
# 1. 環境設置
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jnrle/LDVReorientation/worktrees/dtmin-full-training:$PYTHONPATH

# 2. 運行訓練
python scripts/train_dtmin_demo.py \
    --h_path h_matrix_normalized_original_to_box.pth \
    --w_path nmf_localizer/usm_dict_37ang_111spk_50atoms.pth \
    --dataset_root /path/to/white_noise_box_data \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --d_model 128 \
    --n_heads 4 \
    --n_layers 1 \
    --device mps \
    --output_dir dtmin_outputs_omp
```

### 內部流程

```python
# train_dtmin_demo.py 內部執行順序

# Step 1: 加載字典
H = torch.load(h_path)  # (F=346, E=37)
W = torch.load(w_path)  # (F=346, M=50)

# Atom reduction (M=50 → M=8)
# 使用 kmeans/kcenter/random 選擇 8 個原子
W_reduced = reduce_atoms(W, M_target=8, method='kmeans')

# 構建完整字典
D = build_dictionary(H, W_reduced)  # (F=346, P=37×8=296)
E_map = build_expert_map(E=37, M=8)  # (296,) 每個原子的 expert ID

# Step 2: 加載數據
dataset = load_dataset(dataset_root)  # 加載 .npy 文件
Y_samples, labels = prepare_samples(dataset, num_samples=100)
# Y_samples: (N, F) 已經過 STFT
# labels: (N,) angle indices

# Step 3: 生成 trajectories
from omp_trajectory import OMPTrajectoryGenerator

generator = OMPTrajectoryGenerator(
    D=D,
    E_map=E_map,
    M=8,
    K=6,
    device='mps'
)

trajectories = generator.generate_dataset(
    Y_samples, 
    labels,
    verbose=True
)
# → List[Dict] 長度 N，每個 dict 包含完整 trajectory 數據

# Step 4: 創建 DataLoader
from omp_trajectory import collate_trajectories

train_loader = DataLoader(
    train_trajectories,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_trajectories
)

# Step 5: 訓練
model = DTMinCore(
    F=346, E=37, M=8, K=6,
    d_model=128, n_heads=4, n_layers=1,
    D=D  # Frozen dictionary embeddings
)

for epoch in range(50):
    for batch in train_loader:
        # batch['r_seq']: (B, K, F)
        # batch['RTG_seq']: (B, K, 2)
        # batch['STEP_seq']: (B, K, 2)
    
        expert_scores, atom_scores = model(
            batch['r_seq'],
            batch['RTG_seq'],
            batch['STEP_seq']
        )
    
        loss = compute_hierarchical_loss(
            expert_scores, atom_scores,
            batch['expert_gt'], batch['atom_gt']
        )
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## MDP Trajectory 訓練流程

### 數據路徑配置

```bash
# MDP trajectories (預先生成)
MDP_RESULTS_DIR="../mdp-diverse-policies/results/policy_library_phase1"
CONFIG_NAME="M8_K6_kmeans_greedy"

JSONL_PATH="${MDP_RESULTS_DIR}/${CONFIG_NAME}/trajectories.jsonl"
W_PATH="${MDP_RESULTS_DIR}/${CONFIG_NAME}/W_reduced.pth"

# 原始數據集 (與 OMP 相同)
DATASET_ROOT="/path/to/white_noise_box_data_no_edge_sync_vad_normalized"

# H matrix (共用)
H_PATH="h_matrix_normalized_original_to_box.pth"
```

### 可用的 MDP 配置

```bash
# Policy Library Phase 1 中的配置
mdp-diverse-policies/results/policy_library_phase1/
├── M8_K6_kmeans_greedy/
│   ├── trajectories.jsonl       # 輕量級 trajectory 記錄
│   ├── W_reduced.pth             # 縮減後的字典
│   └── config.json               # 配置參數
│
├── M8_K6_kcenter_greedy/
├── M8_K6_random_greedy/
├── M8_K6_kmeans_random/
└── ... (更多配置)
```

### 完整執行流程

```bash
# 方法 1: 使用便捷腳本
./run_train_mdp.sh M8_K6_kmeans_greedy

# 方法 2: 手動執行
python scripts/train_dtmin_from_mdp.py \
    --jsonl ../mdp-diverse-policies/results/policy_library_phase1/M8_K6_kmeans_greedy/trajectories.jsonl \
    --dataset_root /path/to/white_noise_box_data \
    --h_path h_matrix_normalized_original_to_box.pth \
    --w_path ../mdp-diverse-policies/results/policy_library_phase1/M8_K6_kmeans_greedy/W_reduced.pth \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --rtg_mode continuous \
    --device mps \
    --output_dir dtmin_outputs_mdp/M8_K6_kmeans_greedy
```

### 內部流程

```python
# train_dtmin_from_mdp.py 內部執行順序

# Step 1: 加載字典
H = torch.load(h_path)       # (F=346, E=37)
W = torch.load(w_path)       # (F=346, M=8) 已縮減
D = build_dictionary(H, W)   # (F=346, P=296)

# Step 2: 創建 MDP Adapter
from mdp_trajectory_adapter import MDPTrajectoryAdapter

adapter = MDPTrajectoryAdapter(
    jsonl_path=jsonl_path,
    D=D,
    dataset_root=dataset_root,
    fs=16000,
    n_fft=2048,
    freq_min=300.0,
    freq_max=3000.0,
    rtg_mode='continuous',  # 使用 MDP 的 RTG
    device='mps',
    verbose=True
)

# Step 3: 轉換 trajectories
trajectories = adapter.create_dataset()
# → List[Dict] 格式與 OMP 完全相同!

# Adapter 內部執行:
# for each line in trajectories.jsonl:
#   1. 解析 JSON
#   2. 從 path 加載原始 .npy
#   3. 計算 STFT → Y (F, N)
#   4. 時間平均 → y (F,)
#   5. 根據 actions 重建 r_seq:
#      r_0 = y / ||y||
#      for t in 1..K:
#        S = actions[0:t]
#        x_S = lstsq(D[:, S], y)
#        r_t = (y - D[:, S] @ x_S) / norm
#   6. 提取 RTG from JSON (continuous mode)
#   7. 打包為 DTMin 格式

# Step 4-5: 與 OMP 版本完全相同
# (創建 DataLoader, 訓練循環)
# ... (同 OMP 流程 Step 4-5)
```

---

## Trajectory 格式轉換詳解

### MDP JSONL 格式

```json
{
  "path": "angle_0/clip_000.npy",
  "angle_index": 0,
  "steps": [
    {
      "expert": 0,              // Expert ID (0-36)
      "atom": 4,                // Atom ID within expert (0-7)
      "dict_index": 4,          // Global dict index = expert × M + atom
      "resid_sq": 0.0174,       // ||r_t||² (只存 norm 的平方)
      "delta_resid_sq": 0.9826, // ||r_{t-1}||² - ||r_t||²
      "p_true": 0.058,          // P(angle_gt ∈ experts_selected)
      "rtg_resid": 0.0,         // Target_resid - current_resid
      "rtg_acc": 0.892          // Target_acc - current_acc
    },
    // ... K-1 more steps
  ]
}
```

**為什麼不存 r_seq?**

- r_seq 是 (K, F) = (6, 346) = 2076 floats
- 每個 trajectory 約 8.3 KB (只存 r_seq)
- 1000 個 trajectories = 8.3 MB
- **但可以從 Y 和 actions 重建** → 節省 99%+ 空間
- JSONL 只需約 1-5 MB (vs 完整數據 100+ MB)

### DTMin 統一格式

```python
trajectory = {
    'y': torch.Tensor,           # (F,) 原始信號
    'angle_gt': int,             # Ground truth angle index
    'K': int,                    # Selection budget
    'r_seq': torch.Tensor,       # (K, F) residual 序列
    'action_seq': torch.Tensor,  # (K,) global action indices
    'expert_seq': torch.Tensor,  # (K,) expert IDs
    'atom_seq': torch.Tensor,    # (K,) atom IDs
    'RTG_seq': torch.Tensor,     # (K, 2) [rtg_resid, rtg_acc]
    'STEP_seq': torch.Tensor,    # (K, 2) [t/K, (K-t)/K]
    'residual_norms': torch.Tensor,  # (K,) ||r_t||
    'final_accuracy': float      # Binary: angle_gt selected?
}
```

### 關鍵轉換步驟

#### 1. Residual 重建算法

```python
def _recompute_residuals(y, actions):
    """
    從 action history 重建完整 residual 序列
  
    Input:
        y: (F,) 原始信號
        actions: List[int] 長度 K，action indices
  
    Output:
        r_seq: (K, F) residual vectors
    """
    K = len(actions)
    F = y.shape[0]
    r_seq = torch.zeros(K, F)
  
    actions_so_far = []
  
    for t in range(K):
        if len(actions_so_far) == 0:
            # 初始: r_0 = normalized y
            r_t = y / (y.norm() + 1e-12)
        else:
            # 正交投影物理:
            # r_t = y - D[:, S_{0:t-1}] @ x
            # 其中 x = argmin ||y - D[:, S] @ x||²
        
            D_S = D[:, actions_so_far]  # (F, t)
        
            # 最小二乘求解
            x_S = torch.linalg.lstsq(D_S, y).solution
        
            # 重建信號
            y_hat = D_S @ x_S
        
            # 計算殘差
            r_t = y - y_hat
            r_t = r_t / (r_t.norm() + 1e-12)  # 歸一化
    
        r_seq[t] = r_t
        actions_so_far.append(actions[t])
  
    return r_seq
```

**物理意義**:

- 每一步選擇一個原子 d_j
- 更新重建: ŷ = D[:, S] @ x
- 殘差: r = y - ŷ (正交於已選原子)
- 保證: r ⊥ span(D[:, S])

#### 2. RTG 模式比較

**Continuous Mode (MDP 原生)**:

```python
# 直接從 JSON 提取
RTG_seq[t, 0] = step['rtg_resid']  # Continuous value
RTG_seq[t, 1] = step['rtg_acc']    # Based on p_true
```

**Binary Mode (DTMin 原生)**:

```python
# 重新計算 binary accuracy
experts_so_far = expert_seq[:t+1].unique()
current_acc = 1.0 if angle_gt in experts_so_far else 0.0

RTG_seq[t, 0] = target_resid - current_resid
RTG_seq[t, 1] = target_acc - current_acc  # Binary jump
```

---

## 完整操作指南

### 環境準備

```bash
# 1. 激活 conda 環境
source ~/.zshrc
conda activate trl-training

# 2. 設置 PYTHONPATH
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/dtmin-full-training
export PYTHONPATH=$(pwd):$PYTHONPATH

# 3. 驗證環境
python -c "import torch; print(torch.backends.mps.is_available())"
# 應該輸出: True (for Apple Silicon)
```

### 選項 1: 使用 MDP Trajectories (推薦)

```bash
# Step 1: 測試 adapter
./test_mdp_adapter.sh

# 預期輸出:
# ✅ Adapter created
# ✅ Residual norm consistency
# ✅ Action hierarchy consistency
# ✅ Residual monotonic decrease
# ✅ RTG non-negative
# ✅ STEP in [0,1]

# Step 2: 選擇配置並訓練
# 查看可用配置
ls ../mdp-diverse-policies/results/policy_library_phase1/

# 運行訓練
./run_train_mdp.sh M8_K6_kmeans_greedy

# Step 3: 監控訓練
tail -f dtmin_outputs_mdp/M8_K6_kmeans_greedy/train.log
```

### 選項 2: 使用 OMP Trajectories

```bash
# 直接運行
python scripts/train_dtmin_demo.py \
    --h_path h_matrix_normalized_original_to_box.pth \
    --w_path nmf_localizer/usm_dict_37ang_111spk_50atoms.pth \
    --dataset_root /path/to/white_noise_box_data \
    --epochs 50 \
    --batch_size 16 \
    --device mps \
    --output_dir dtmin_outputs_omp

# 監控訓練
tail -f dtmin_outputs_omp/train.log
```

### 訓練參數說明

```bash
# 必需參數
--jsonl              # MDP trajectories 路徑 (僅 MDP 版本)
--dataset_root       # 原始數據集根目錄
--h_path             # H matrix 路徑
--w_path             # W matrix 路徑

# 訓練超參數
--epochs 50          # 訓練輪數
--batch_size 16      # Batch size
--lr 1e-4            # Learning rate
--weight_decay 1e-5  # L2 regularization

# 模型架構
--d_model 128        # Embedding dimension
--n_heads 4          # Attention heads
--n_layers 1         # Transformer layers

# RTG 模式 (僅 MDP 版本)
--rtg_mode continuous  # 使用 MDP 的 continuous RTG
--rtg_mode binary      # 重新計算 binary RTG

# 硬件
--device mps         # Apple Silicon GPU
--device cpu         # CPU only

# 輸出
--output_dir path    # 輸出目錄
```

### 輸出文件結構

```
dtmin_outputs_mdp/M8_K6_kmeans_greedy/
├── checkpoints/
│   ├── epoch_10.ckpt
│   ├── epoch_20.ckpt
│   └── best_model.ckpt       # 最佳模型
│
├── train.log                  # 訓練日誌
├── config.json                # 訓練配置
├── metrics.json               # 訓練指標
│
└── tensorboard/              # TensorBoard logs
    └── events.out.tfevents.*
```

---

## 數據格式規範

### STFT 參數 (統一標準)

```python
# 所有腳本必須使用相同的 STFT 參數
STFT_CONFIG = {
    'fs': 16000,           # Sampling frequency
    'n_fft': 2048,         # FFT size
    'window': 'hann',      # Window type
    'freq_min': 300.0,     # Band limit (Hz)
    'freq_max': 3000.0,    # Band limit (Hz)
}

# 結果:
# F = 346 frequency bins (after band limiting)
# N = varies (time frames, depends on audio length)
```

### 維度約定

```python
# Dictionary
H: (F=346, E=37)      # Transfer functions, 37 angles
W: (F=346, M=8)       # Atoms per expert (after reduction)
D: (F=346, P=296)     # Full dictionary, P = E × M

# Signals
y: (F,)               # Time-averaged observation
Y: (F, N)             # STFT spectrogram

# Trajectories
r_seq: (K, F)         # Residual sequence, K=6 steps
RTG_seq: (K, 2)       # Return-to-go [resid, acc]
STEP_seq: (K, 2)      # Step encoding [t/K, (K-t)/K]
action_seq: (K,)      # Action indices
expert_seq: (K,)      # Expert IDs (0-36)
atom_seq: (K,)        # Atom IDs (0-7)

# Batched (from DataLoader)
r_seq: (B, K, F)      # B = batch_size
RTG_seq: (B, K, 2)
STEP_seq: (B, K, 2)
expert_gt: (B, K)
atom_gt: (B, K)

# Model outputs
expert_scores: (B, K, E=37)
atom_scores: (B, K, P=296)
```

### Action 層次關係

```python
# Hierarchical decomposition
expert_id = dict_index // M          # Expert ID (0-36)
atom_id = dict_index % M             # Atom ID (0-7)

# Reconstruction
dict_index = expert_id × M + atom_id

# Validation
assert 0 <= expert_id < E
assert 0 <= atom_id < M
assert 0 <= dict_index < P
assert dict_index == expert_id * M + atom_id
```

---

## 故障排除

### 常見錯誤 1: MPS Device Mismatch

**症狀**:

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and mps:0!
```

**原因**: Index tensors 在 MPS 上可能有問題

**解決方案**:

```python
# In forward pass, ensure indices are on CPU for gather operations
expert_gt = expert_gt.cpu()
atom_gt = atom_gt.cpu()
```

### 常見錯誤 2: JSONL File Not Found

**症狀**:

```
FileNotFoundError: JSONL file not found: .../trajectories.jsonl
```

**檢查清單**:

```bash
# 1. 確認 MDP 專案路徑
ls ../mdp-diverse-policies/results/policy_library_phase1/

# 2. 確認配置名稱正確
echo $CONFIG_NAME

# 3. 檢查 JSONL 是否存在
ls -lh ../mdp-diverse-policies/results/policy_library_phase1/$CONFIG_NAME/trajectories.jsonl
```

### 常見錯誤 3: Audio File Not Found

**症狀**:

```
FileNotFoundError: Y file not found: /path/to/angle_0/clip_000.npy
```

**原因**: `dataset_root` 路徑不正確

**解決方案**:

```bash
# 確認數據集路徑
export DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"

# 驗證結構
ls $DATASET_ROOT/angle_0/clip_000.npy
```

### 常見錯誤 4: Dimension Mismatch

**症狀**:

```
RuntimeError: Expected tensor to have size 346 at dimension 0, but got size 400
```

**原因**: STFT 參數不一致

**檢查**:

```python
# 確保所有腳本使用相同參數
fs = 16000
n_fft = 2048
freq_min = 300.0
freq_max = 3000.0

# 驗證 F
F = len([f for f in freqs if freq_min <= f <= freq_max])
assert F == 346
```

### 性能優化建議

```python
# 1. 使用更大的 batch size (如果內存允許)
--batch_size 32  # 默認 16

# 2. 啟用混合精度訓練 (未來功能)
--mixed_precision  # TODO

# 3. 數據預加載
--num_workers 4  # DataLoader workers

# 4. Gradient accumulation (模擬更大 batch)
--gradient_accumulation_steps 2
```

---

## 附錄: 關鍵代碼片段

### A. Collate Function (可重用)

```python
def collate_trajectories(trajectories):
    """
    Collate trajectories into batch.
    可用於 OMP 和 MDP trajectories (格式相同)
  
    Input: List[Dict] of trajectories
    Output: Dict of batched tensors
    """
    return {
        'r_seq': torch.stack([t['r_seq'] for t in trajectories]),
        'RTG_seq': torch.stack([t['RTG_seq'] for t in trajectories]),
        'STEP_seq': torch.stack([t['STEP_seq'] for t in trajectories]),
        'expert_gt': torch.stack([t['expert_seq'] for t in trajectories]),
        'atom_gt': torch.stack([t['atom_seq'] for t in trajectories]),
        'angle_gt': torch.tensor([t['angle_gt'] for t in trajectories])
    }
```

### B. Hierarchical Loss (可重用)

```python
def compute_hierarchical_loss(expert_scores, atom_scores, expert_gt, atom_gt):
    """
    計算階層式損失
  
    Input:
        expert_scores: (B, K, E)
        atom_scores: (B, K, P)
        expert_gt: (B, K)
        atom_gt: (B, K)
  
    Output:
        loss_expert: scalar
        loss_atom: scalar
    """
    B, K, E = expert_scores.shape
  
    # Flatten for cross entropy
    expert_logits = expert_scores.reshape(-1, E)
    expert_targets = expert_gt.reshape(-1)
  
    atom_logits = atom_scores.reshape(-1, atom_scores.size(-1))
    atom_targets = atom_gt.reshape(-1)
  
    loss_expert = F.cross_entropy(expert_logits, expert_targets)
    loss_atom = F.cross_entropy(atom_logits, atom_targets)
  
    return loss_expert, loss_atom
```

### C. Dictionary Construction

```python
def build_dictionary(H, W):
    """
    構建完整字典 D = H ⊗ W
  
    Input:
        H: (F, E) transfer functions
        W: (F, M) atoms per expert
  
    Output:
        D: (F, E×M) full dictionary
    """
    F, E = H.shape
    _, M = W.shape
  
    D = torch.zeros(F, E * M)
  
    for e in range(E):
        for m in range(M):
            # Hadamard product
            D[:, e * M + m] = H[:, e] * W[:, m]
  
    # Normalize columns
    D = D / (D.norm(dim=0, keepdim=True) + 1e-12)
  
    return D
```

---

## 結語

本文檔提供了 DTMin 訓練的完整流程，包括:

✅ **兩種 trajectory 生成方式**的詳細對比
✅ **完整的數據流程圖**，從原始音頻到訓練完成
✅ **MDP trajectory 轉換**的核心算法
✅ **統一的訓練管線**，80% 代碼可重用
✅ **詳細的操作指南**和故障排除

**關鍵洞察**:

1. MDP 和 OMP trajectories 產生**相同格式**的數據
2. **Collation 和訓練代碼完全可重用**
3. 差異僅在 trajectory 生成階段
4. MDP 方式提供更好的策略多樣性和訓練效率

**下一步**:

1. 運行 `./test_mdp_adapter.sh` 驗證轉換正確性
2. 選擇 MDP 配置並開始訓練
3. 監控訓練指標並調整超參數
4. 與 OMP baseline 比較性能

---

**文檔版本**: 1.0
**最後更新**: 2025-11-04
**維護者**: DTMin Development Team
