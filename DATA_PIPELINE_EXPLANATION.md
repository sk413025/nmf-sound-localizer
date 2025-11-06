# DTMin 完整資料處理流程說明

**日期**: 2025-11-06  
**目的**: 清楚說明從原始音訊到 Decision Transformer 訓練的完整資料處理流程

---

## 📋 目錄

1. [流程總覽](#流程總覽)
2. [階段 1: Trajectory 生成](#階段-1-trajectory-生成)
3. [階段 2: Trajectory 載入與轉換](#階段-2-trajectory-載入與轉換)
4. [階段 3: Decision Transformer 訓練](#階段-3-decision-transformer-訓練)
5. [涉及的檔案](#涉及的檔案)
6. [維度轉換詳解](#維度轉換詳解)

---

## 流程總覽

```
原始音訊 (.npy)
    ↓
[階段 1: offline_dt_dataset.py]
    ↓ 產生
trajectories.jsonl (儲存 actions + metadata，不儲存完整 residual vectors)
    ↓
[階段 2: dt_pointer_ldv.py 讀取]
    ↓ 重建
R_seq, RTG_seq, STEP_seq (完整 tensors)
    ↓
[階段 3: dt_pointer_ldv.py 訓練]
    ↓ Token embedding
Decision Transformer 預測 expert + atom
```

---

## 階段 1: Trajectory 生成

### 使用的檔案

**主要腳本**: `doa_rl/trajectories/offline_dt_dataset.py`

### 輸入

1. **原始音訊檔案**:
   ```
   /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized/
   ├── angle_0/
   │   ├── clip_000.npy  # Shape: (samples,) 原始波形 @ 16kHz
   │   ├── clip_001.npy
   │   └── clip_002.npy
   ├── angle_5/
   ├── ...
   └── angle_180/
   
   Total: 111 個 .npy 檔案 (37 angles × 3 clips)
   ```

2. **H 矩陣** (Transfer functions):
   ```
   Path: /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
   Shape: (F=346, E=37)
   含義: 37 個角度的聲學轉移函數
   ```

3. **W 矩陣** (Spectral atoms):
   ```
   Path: doa_normalized_config_c_corrected/models/usm.pth
   Shape: (F=346, M_full=50) 原始
        → (F=346, M=8) 縮減後 (kmeans/kcenter/random)
   含義: 頻譜原子字典
   ```

### 處理流程

```python
# 1. 載入字典
H = torch.load(h_path)  # (F=346, E=37)
W = torch.load(w_path)  # (F=346, M_full=50)

# 2. 原子縮減 (M_full=50 → M=8)
W_reduced = reduce_atoms_kmeans(W, n_clusters=8, random_state=42)
# Shape: (F=346, M=8)

# 3. 構建完整字典 D = H ⊙ W
D = torch.zeros(F, E * M)  # (346, 296)
for e in range(E):  # 37 experts
    for m in range(M):  # 8 atoms per expert
        D[:, e*M + m] = H[:, e] * W_reduced[:, m]  # Hadamard product
# 歸一化每個 column
D = D / (D.norm(dim=0, keepdim=True) + 1e-12)

# 4. 對每個音訊檔案執行 OMP
for audio_file in audio_files:  # 111 個檔案
    # 4a. 載入並計算 STFT
    waveform = np.load(audio_file)  # (samples,)
    Y = compute_stft(waveform, fs=16000, n_fft=2048)  # (F=346, N)
    
    # 4b. 時間平均
    y = Y.mean(dim=1)  # (F=346,)
    y = y / y.norm()    # 歸一化
    
    # 4c. OMP selection (K=6 steps)
    actions = []  # 儲存選擇的 atom indices
    r = y.clone()  # 初始 residual
    
    for t in range(K):  # K=6 steps
        # 計算 correlation
        g = D.T @ r  # (P=296,) = (296, 346) @ (346,)
        
        # Greedy selection
        j = torch.argmax(torch.abs(g)).item()  # 選擇最大 correlation
        actions.append(j)
        
        # 更新 residual (正交投影)
        D_selected = D[:, actions]  # (346, t+1)
        x = torch.linalg.lstsq(D_selected, y).solution
        y_hat = D_selected @ x
        r = y - y_hat
        r = r / (r.norm() + 1e-12)  # 歸一化
        
        # 計算 metrics
        resid_sq = (r @ r).item()  # ||r_t||²
        
        # 計算 RTG (return-to-go)
        # ... (細節見下方)
    
    # 5. 儲存 trajectory
    trajectory = {
        "path": "angle_0/clip_000.npy",
        "angle_index": 0,
        "steps": [
            {
                "expert": j // M,      # expert ID
                "atom": j % M,         # atom ID within expert
                "dict_index": j,       # global index
                "resid_sq": 0.0174,   # ||r_t||²
                "p_true": 0.058,      # accuracy metric
                "rtg_resid": 0.0,     # return-to-go (residual)
                "rtg_acc": 0.892      # return-to-go (accuracy)
            },
            # ... K-1 more steps
        ]
    }
```

### 輸出

**檔案**: `results/dt_traj_omp_480epochs_<timestamp>/trajectories.jsonl`

**格式**: 每行一個 JSON (111 行)

```jsonl
{"path": "angle_0/clip_000.npy", "angle_index": 0, "steps": [{"expert": 0, "atom": 4, ...}, ...]}
{"path": "angle_0/clip_001.npy", "angle_index": 0, "steps": [{"expert": 1, "atom": 2, ...}, ...]}
...
```

**重要**: ⚠️ **不儲存完整的 residual vectors** (r_seq)  
原因: 可以從 actions 重建，節省空間 (每個 trajectory 節省 ~8KB)

### 維度總結 (階段 1)

| 資料 | 維度 | 說明 |
|------|------|------|
| 原始波形 | `(samples,)` | 16kHz 採樣 |
| STFT | `(F=346, N)` | 頻譜，N=時間幀數 |
| 時間平均 y | `(F=346,)` | 單一頻譜向量 |
| H 矩陣 | `(F=346, E=37)` | 37 個角度 |
| W 矩陣 | `(F=346, M=8)` | 8 個原子 (縮減後) |
| 字典 D | `(F=346, P=296)` | P = E×M = 37×8 |
| Actions | `(K=6,)` | 每步選擇的 dict_index |

---

## 階段 2: Trajectory 載入與轉換

### 使用的檔案

**主要腳本**: `scripts/dt_pointer_ldv.py` (訓練腳本中的 data loading 部分)

### 輸入

1. **trajectories.jsonl** (從階段 1 產生)
2. **原始音訊檔案** (重新載入以重建 residuals)
3. **H, W 矩陣** (重建字典 D)

### 處理流程

```python
# 在 dt_pointer_ldv.py 的 main() 函數中

# 1. 載入 manifest 和 trajectories
manifest = load_manifest(traj_dir)  # manifest.json
trajs = load_trajectories(traj_dir)  # trajectories.jsonl → List[Dict]

# 2. 重建字典 (與階段 1 完全相同的參數)
H, W = load_H_W(h_path, w_path, selected_indices, device)
D, E, M = build_D(H, W, angles)
# D: (F=346, P=296)

# 3. 對每個 trajectory 重建完整資料
samples = []
for traj in trajs:  # 111 個 trajectories
    # 3a. 重新載入音訊並計算 y
    audio_path = dataset_root / traj['path']
    waveform = np.load(audio_path)
    Y = compute_stft(waveform, ...)  # (F, N)
    y = Y.mean(dim=1)  # (F,)
    y = y / y.norm()
    
    # 3b. 從 actions 重建 residual sequence
    actions = [step['dict_index'] for step in traj['steps']]
    r_seq = recompute_r_t(y, D, actions)  # 關鍵函數!
    # Shape: (K=6, F=346)
    
    # 3c. 提取 RTG 和 actions
    RTG_seq = torch.zeros(K, 2)
    expert_seq = torch.zeros(K, dtype=torch.long)
    atom_seq = torch.zeros(K, dtype=torch.long)
    
    for t, step in enumerate(traj['steps']):
        RTG_seq[t, 0] = step['rtg_resid']  # residual RTG
        RTG_seq[t, 1] = step['rtg_acc']    # accuracy RTG
        expert_seq[t] = step['expert']     # expert ID (0-36)
        atom_seq[t] = step['atom']         # atom ID (0-7)
    
    # 3d. 構建 STEP sequence
    STEP_seq = torch.zeros(K, 2)
    for t in range(K):
        STEP_seq[t, 0] = t / K           # progress
        STEP_seq[t, 1] = (K - t) / K     # budget remaining
    
    # 3e. 打包成 sample
    sample = {
        'R': r_seq,           # (K=6, F=346)
        'RTG': RTG_seq,       # (K=6, 2)
        'STEP': STEP_seq,     # (K=6, 2)
        'expert_gt': expert_seq,  # (K=6,)
        'atom_gt': atom_seq       # (K=6,)
    }
    samples.append(sample)

# 4. Train/test split
train_samples, test_samples = split_samples(samples, test_split=0.2, seed=42)
# Train: 74 samples (66.7%)
# Test:  37 samples (33.3%)
```

### 關鍵函數: `recompute_r_t()` (重建 residual sequence)

```python
def recompute_r_t(y: torch.Tensor, D: torch.Tensor, actions: List[int]) -> torch.Tensor:
    """
    從 action history 重建完整的 residual sequence
    
    這是為什麼我們可以不儲存 r_seq 的原因!
    
    Args:
        y: (F,) 原始信號 (時間平均後)
        D: (F, P) 字典
        actions: List[int] 長度 K，已選擇的 atom indices
    
    Returns:
        r_seq: (K, F) residual vectors
    """
    K = len(actions)
    F = y.shape[0]
    r_seq = torch.zeros(K, F, device=y.device)
    
    for t in range(K):
        if t == 0:
            # 第一步: r_0 = y (歸一化)
            r_t = y / (y.norm() + 1e-12)
        else:
            # 後續步驟: 從已選擇的 atoms 重建
            S = actions[:t]  # 已選擇的 atoms
            D_S = D[:, S]    # (F, t)
            
            # 最小二乘求解: x = argmin ||y - D_S @ x||²
            x_S = torch.linalg.lstsq(D_S, y).solution
            
            # 重建信號
            y_hat = D_S @ x_S
            
            # 計算 residual
            r_t = y - y_hat
            r_t = r_t / (r_t.norm() + 1e-12)
        
        r_seq[t] = r_t
    
    return r_seq
```

### 維度總結 (階段 2)

| 資料 | 載入時維度 | 說明 |
|------|-----------|------|
| trajectories.jsonl | 111 個 JSON objects | 輕量級，只有 metadata |
| 重建後 r_seq | `(K=6, F=346)` | **每個 sample** |
| RTG_seq | `(K=6, 2)` | [rtg_resid, rtg_acc] |
| STEP_seq | `(K=6, 2)` | [t/K, (K-t)/K] |
| expert_gt | `(K=6,)` | Ground truth expert IDs |
| atom_gt | `(K=6,)` | Ground truth atom IDs |

---

## 階段 3: Decision Transformer 訓練

### 使用的檔案

**主要腳本**: `scripts/dt_pointer_ldv.py`  
**模型定義**: `doa_rl/model/` (內建在 dt_pointer_ldv.py 中)

### DataLoader Batching

```python
# 使用 PyTorch DataLoader
def batch_iter(samples, batch_size, shuffle):
    """
    將 samples 打包成 batches
    """
    if shuffle:
        indices = torch.randperm(len(samples))
    else:
        indices = torch.arange(len(samples))
    
    for i in range(0, len(samples), batch_size):
        batch_idx = indices[i:i+batch_size]
        
        # Stack 成 batch tensors
        R_b = torch.stack([samples[j]['R'] for j in batch_idx])
        RTG_b = torch.stack([samples[j]['RTG'] for j in batch_idx])
        STEP_b = torch.stack([samples[j]['STEP'] for j in batch_idx])
        E_b = torch.stack([samples[j]['expert_gt'] for j in batch_idx])
        M_b = torch.stack([samples[j]['atom_gt'] for j in batch_idx])
        
        yield R_b, RTG_b, STEP_b, E_b, M_b

# 使用範例
train_loader = batch_iter(train_samples, batch_size=4, shuffle=True)
for R_b, RTG_b, STEP_b, E_b, M_b in train_loader:
    # R_b: (B=4, K=6, F=346)
    # RTG_b: (B=4, K=6, 2)
    # STEP_b: (B=4, K=6, 2)
    # E_b: (B=4, K=6)
    # M_b: (B=4, K=6)
    ...
```

### Token Embedding (在 forward pass 中)

```python
class DTMinPointer(nn.Module):
    def __init__(self, F, E, M, d_model=128, ...):
        super().__init__()
        
        # Projection layers
        self.P_R = nn.Linear(F, d_model, bias=False)       # Residual → embedding
        self.proj_rtg = nn.Linear(2, d_model, bias=False)  # RTG → embedding
        self.proj_step = nn.Linear(2, d_model, bias=False) # STEP → embedding
        
        # Type embeddings (可學習的 bias)
        self.type_R = nn.Parameter(torch.zeros(d_model))
        
        # LayerNorm
        self.ln = nn.LayerNorm(d_model)
        
        # Transformer
        self.encoder = nn.TransformerEncoder(...)
        
        # Dictionary embeddings (FROZEN)
        self.P_D = nn.Linear(F, d_model, bias=False)
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        # 預計算的 dictionary keys
        self.KD_em = ...  # (E=37, M=8, d_model)
    
    def forward(self, R_seq, RTG_seq, STEP_seq, causal_mask):
        """
        Args:
            R_seq: (B, K, F=346) residual sequence
            RTG_seq: (B, K, 2) return-to-go
            STEP_seq: (B, K, 2) step encoding
            causal_mask: (K, K) attention mask
        
        Returns:
            scores_e: (B, K, E=37) expert scores
            scores_em: (B, K, E=37, M=8) atom scores
        """
        B, K, F = R_seq.shape
        
        # Step 1: Project to embedding space
        r_tok = self.P_R(R_seq) + self.type_R      # (B, K, d_model)
        rtg_tok = self.proj_rtg(RTG_seq)           # (B, K, d_model)
        step_tok = self.proj_step(STEP_seq)        # (B, K, d_model)
        
        # Step 2: Combine tokens (加法，不是拼接)
        h = self.ln(r_tok + rtg_tok + step_tok)    # (B, K, d_model)
        
        # Step 3: Transformer encoding
        Ht = self.encoder(h, mask=causal_mask)     # (B, K, d_model)
        
        # Step 4: Compute queries
        Q = self.Wq(Ht)                            # (B, K, d_model)
        
        # Step 5: QK attention with frozen dictionary
        # KD_em: (E, M, d_model) 預計算的 keys
        qk = torch.einsum('bkd,emd->bkem', Q, self.KD_em)
        # qk: (B, K, E, M)
        
        # Step 6: Aggregate to expert scores (L2 over atoms)
        scores_e = torch.sqrt((qk.abs() ** 2).sum(dim=3) + 1e-12)
        # scores_e: (B, K, E)
        
        # Step 7: Atom scores
        scores_em = qk  # (B, K, E, M)
        
        return scores_e, scores_em
```

### Loss Computation

```python
# Forward pass
scores_e, scores_em, _ = model(R_b, RTG_b, STEP_b, causal_mask)
# scores_e: (B, K, E=37)
# scores_em: (B, K, E=37, M=8)

# Expert loss (階層 1)
loss_e = F.cross_entropy(
    scores_e.reshape(-1, E),  # (B*K, E)
    E_b.reshape(-1)           # (B*K,)
)

# Atom loss (階層 2)
# 先根據 ground truth expert 選擇對應的 atom scores
idx = E_b.reshape(-1)  # (B*K,)
se_flat = scores_em.reshape(-1, E, M)  # (B*K, E, M)
se_sel = se_flat.gather(1, idx.view(-1,1,1).expand(-1,1,M)).squeeze(1)
# se_sel: (B*K, M) - 只有正確 expert 的 atom scores

loss_a = F.cross_entropy(
    se_sel,           # (B*K, M)
    M_b.reshape(-1)   # (B*K,)
)

# Total loss
loss = loss_e + loss_a
```

### 訓練循環

```python
# 完整訓練循環
for epoch in range(epochs):
    model.train()
    
    for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(train_samples, batch_size=4, shuffle=True):
        # 移到 device
        R_b = R_b.to(device)
        RTG_b = RTG_b.to(device)
        STEP_b = STEP_b.to(device)
        E_b = E_b.to(device)
        M_b = M_b.to(device)
        
        # Forward
        scores_e, scores_em, _ = model(R_b, RTG_b, STEP_b, causal_mask)
        
        # Loss
        loss_e = F.cross_entropy(scores_e.reshape(-1, E), E_b.reshape(-1))
        
        se_flat = scores_em.reshape(-1, E, M)
        idx = E_b.reshape(-1)
        se_sel = se_flat.gather(1, idx.view(-1,1,1).expand(-1,1,M)).squeeze(1)
        loss_a = F.cross_entropy(se_sel, M_b.reshape(-1))
        
        loss = loss_e + loss_a
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(test_samples, batch_size=len(test_samples), shuffle=False):
            scores_e, scores_em, _ = model(R_b, RTG_b, STEP_b, causal_mask)
            
            # 計算 accuracy
            pred_e = scores_e.argmax(dim=-1)
            expert_acc = (pred_e == E_b).float().mean()
            
            # ... atom accuracy 類似
```

### 維度總結 (階段 3)

| 階段 | 資料 | 維度 | 說明 |
|------|------|------|------|
| **Input** | R_b | `(B=4, K=6, F=346)` | Batched residuals |
| | RTG_b | `(B=4, K=6, 2)` | Batched RTG |
| | STEP_b | `(B=4, K=6, 2)` | Batched steps |
| **Token Embedding** | r_tok | `(B=4, K=6, d=128)` | Residual tokens |
| | rtg_tok | `(B=4, K=6, d=128)` | RTG tokens |
| | step_tok | `(B=4, K=6, d=128)` | Step tokens |
| | h | `(B=4, K=6, d=128)` | Combined (加法) |
| **Transformer** | Ht | `(B=4, K=6, d=128)` | Encoded |
| **Prediction** | scores_e | `(B=4, K=6, E=37)` | Expert logits |
| | scores_em | `(B=4, K=6, E=37, M=8)` | Atom logits |
| **Ground Truth** | E_b | `(B=4, K=6)` | Expert labels |
| | M_b | `(B=4, K=6)` | Atom labels |

---

## 涉及的檔案

### 核心檔案 (必須)

| 檔案 | 用途 | 階段 |
|------|------|------|
| `doa_rl/trajectories/offline_dt_dataset.py` | 產生 trajectories | 階段 1 |
| `scripts/dt_pointer_ldv.py` | 訓練 Decision Transformer | 階段 2-3 |
| `doa_rl/model/physics_reconstruction.py` | Physics reconstruction head | 階段 3 (可選) |

### 資料檔案

| 檔案 | 內容 | 維度/格式 |
|------|------|----------|
| `h_matrix_box_ldv_correct.pth` | H 矩陣 | `(F=346, E=37)` |
| `doa_normalized_config_c_corrected/models/usm.pth` | W 矩陣 | `(F=346, M_full=50)` |
| `white_noise_box_data.../angle_X/clip_XXX.npy` | 原始音訊 | `(samples,)` 各個檔案 |
| `results/.../trajectories.jsonl` | OMP trajectories | JSONL, 111 行 |
| `results/.../manifest.json` | 配置資訊 | JSON |

### 輔助檔案

| 檔案 | 用途 |
|------|------|
| `run_physics_test.sh` | 執行 100 epoch 測試腳本 |
| `run_480epochs_omp.sh` | 執行完整 480 epoch 訓練 |
| `PHYSICS_RECONSTRUCTION_SUMMARY.md` | Physics 模組說明 |

---

## 維度轉換詳解

### 完整的維度變化鏈

```
原始音訊 (samples,)
    ↓ [STFT]
STFT (F=346, N=varies)
    ↓ [時間平均]
y (F=346,)
    ↓ [OMP selection] ← 階段 1
actions List[int] 長度 K=6
    ↓ [儲存到 trajectories.jsonl]
Lightweight JSON (只有 actions + metadata)
    ↓ [重建 residuals] ← 階段 2
r_seq (K=6, F=346)
    ↓ [Batching]
R_b (B=4, K=6, F=346)
    ↓ [Token projection] ← 階段 3
r_tok (B=4, K=6, d=128)
    ↓ [Combine with RTG + STEP]
h (B=4, K=6, d=128)
    ↓ [Transformer]
Ht (B=4, K=6, d=128)
    ↓ [QK attention]
scores_e (B=4, K=6, E=37)
scores_em (B=4, K=6, E=37, M=8)
```

### 關鍵維度對應

| 符號 | 值 | 含義 |
|------|-----|------|
| F | 346 | 頻率 bins (300-3000 Hz, STFT) |
| E | 37 | Experts (角度數量) |
| M | 8 | Atoms per expert (縮減後) |
| P | 296 | Dictionary size (E × M) |
| K | 6 | Selection budget (steps) |
| d_model | 128 | Token embedding dimension |
| B | 4 | Batch size |
| N | 111 | Total samples (37 angles × 3 clips) |

---

## 總結

### ✅ 您理解正確的部分

1. 使用 `offline_dt_dataset.py` 產生 trajectories
2. 引入 W, H 矩陣構建字典
3. 用傳統 OMP 產生 trajectories
4. 轉換成 (RTG, Residual, Step) tokens
5. 送給 Decision Transformer 預測

### 🔑 關鍵要點

1. **trajectories.jsonl 不儲存完整 r_seq**  
   → 節省空間，可從 actions 重建

2. **Token 轉換在訓練時才做**  
   → `offline_dt_dataset.py` 只產生 trajectories  
   → `dt_pointer_ldv.py` 載入並轉換成 tokens

3. **三個階段清楚分離**:
   - 階段 1: Trajectory 生成 (offline)
   - 階段 2: 重建完整資料 (training 開始時)
   - 階段 3: Token embedding + Transformer (每個 batch)

4. **維度追蹤很重要**:
   - 原始: `(F=346,)` → Batch: `(B, K=6, F=346)` → Token: `(B, K, d=128)`

希望這份文件清楚說明了整個流程！
