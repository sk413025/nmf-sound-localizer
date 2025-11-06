# Teacher Model Loading Issue - 診斷報告

## 問題描述

訓練時出現警告訊息：
```
[Teacher loader] Exception during load: invalid load key, 'v'.
```

## 根本原因

### 錯誤來源
錯誤發生在 `scripts/dt_pointer_ldv.py` 第 143 行：

```python
def _load_qk_teacher_from_scripts(...):
    ...
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = state.get('model_state_dict', state)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        print(f"[Teacher loader] Missing keys: {missing}")
        print(f"[Teacher loader] Unexpected keys: {unexpected}")
    except Exception as e:
        print(f"[Teacher loader] Exception during load: {e}")  # <- 錯誤在這裡
```

### 錯誤訊息解析

**`invalid load key, 'v'.`** 是 PyTorch/Pickle 嘗試載入二進位檔案時的錯誤，表示：
- 檔案不是有效的 PyTorch checkpoint（.pth）
- 實際上是文本文件，而非二進位序列化數據
- 第一個字元是 'v'（即 "version"），這是 LFS pointer 的標誌

### LFS Pointer 內容

Teacher checkpoint 文件 `results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth` 原本是：

```
version https://git-lfs.github.com/spec/v1
oid sha256:5f74c97e5dc649c6d4700b2eca7c81f6bcb45d314e1159b7068163f12434dbf0
size 3877237
```

- **文件大小**: 132 bytes（LFS pointer）
- **實際大小**: 3.7 MB（實際模型）
- **第一個字元**: 'v' ← 這就是錯誤訊息中的 'v'

當 `torch.load()` 嘗試用 pickle 解析這個文本文件時：
1. 讀取第一個字節 `'v'` (ASCII 118)
2. Pickle 期待這是一個 opcode
3. `'v'` 不是有效的 pickle opcode
4. 拋出錯誤：`invalid load key, 'v'.`

## Teacher Model 的作用

### 在訓練架構中的角色

Teacher model 是一個**可選的蒸餾組件**，用於：

```python
# 從 manifest 載入 teacher 配置
teacher_ckpt = args.teacher_ckpt
if not teacher_ckpt and isinstance(manifest.get('qk', None), dict):
    teacher_ckpt = manifest['qk'].get('ckpt', '')  # <- 從這裡讀取

teacher_model = None
if teacher_ckpt:
    teacher_model = _load_qk_teacher_from_scripts(...)  # <- 在這裡失敗
```

### Teacher 的功能（Knowledge Distillation）

在每個訓練 batch 中（第 473-489 行）：

```python
if teacher_model is not None and args.distill_weight > 0.0:
    # 1. 用 teacher 生成 expert scores（目標分佈）
    with torch.no_grad():
        te_list = []
        for t in range(K):
            for b in range(R_b.size(0)):
                qk_exp = qk_scores_with_config(teacher_model, D.to(device), R_b[b,t])
                te_t.append(qk_exp)
            te_logits = torch.stack(te_list, dim=1)  # (B,K,E)
    
    # 2. 計算 KL divergence（蒸餾損失）
    T = max(args.distill_T, 1.0)
    p_teacher = F.softmax(te_logits / T, dim=-1)
    log_p_student = F.log_softmax(scores_e / T, dim=-1)
    kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T*T)
    
    # 3. 加權蒸餾損失到總損失
    w = args.distill_weight if epoch < args.warmup_epochs else (0.5 * args.distill_weight)
    loss = loss + w * kl  # <- Teacher 影響訓練損失
```

### 蒸餾參數

從訓練命令可見：
```bash
--distill_weight 0.7    # 蒸餾損失權重
--distill_T 1.0         # 溫度參數
--warmup_epochs 3       # 前3個epoch用全權重，之後減半
```

**Warmup 階段（epoch 0-2）**:
```
loss = loss_ce + 0.7 * kl_divergence
```

**後續階段（epoch 3+）**:
```
loss = loss_ce + 0.35 * kl_divergence
```

## 對訓練的影響

### ⚠️ 目前狀態（Teacher 載入失敗）

```python
teacher_model = None  # 因為載入失敗，設為 None
```

**實際訓練損失**：
```python
if teacher_model is not None and args.distill_weight > 0.0:  # <- False，跳過整個區塊
    ...

# 只有基礎損失
loss = loss_e + loss_a  # expert CE + atom CE
```

**影響**：
1. ❌ **沒有 knowledge distillation**：學生模型無法從 teacher 學習
2. ❌ **訓練目標改變**：只優化 cross-entropy，缺少 teacher 引導
3. ❌ **收斂速度可能變慢**：teacher 提供的 soft labels 通常幫助更快收斂
4. ❌ **最終精度可能較低**：蒸餾通常能提升模型表現

### ✅ 修復後（Teacher 正常載入）

**實際訓練損失**：
```python
loss = loss_e + loss_a + 0.7 * kl_divergence  # (epoch 0-2)
loss = loss_e + loss_a + 0.35 * kl_divergence # (epoch 3+)
```

**預期改善**：
1. ✓ Teacher 引導學生模型學習更好的 expert selection 策略
2. ✓ Soft labels 提供更豐富的監督信號（不只是 one-hot 標籤）
3. ✓ 更快收斂到更好的區域最優解
4. ✓ 角度精度（angle accuracy）應該顯著提升

## 實驗結果對比

### 無 Teacher（目前）

從你的訓練日誌：
```
Chunk 1: loss=7.8347 → 5.2859, angle t=0: 0.243
Chunk 2: loss=6.1385 → 4.9477, angle t=0: 0.189  # 角度精度下降！
Chunk 3: loss=5.7050 → 4.7210, angle t=0: 0.189
Chunk 4: loss=5.4389 → 4.5593, angle t=0: 0.162  # 繼續下降
```

**觀察**：
- Loss 在下降（好）
- 但 angle accuracy 從 0.243 → 0.162 持續**下降**（壞）
- Teacher QK accuracy 只有 0.027（錯誤！應該是 ~0.95）

### 原始實驗（有 Teacher）

Commit 0cfe0c1 (40 epochs):
```
loss=6.9319 → 5.2181
angle t=0: 0.189
teacher qk t=0: 0.946  # <- 正確的 teacher 精度
```

Commit 8d5f10b (80 epochs):
```
loss=4.81 → 3.59
angle t=0: 0.405
teacher qk t=0: 0.946
```

**關鍵差異**：
- ✓ Loss 範圍相似
- ❌ **Teacher QK accuracy: 0.027 vs 0.946** ← 明確證據 teacher 未正常工作
- ❌ Angle accuracy 趨勢相反（你的下降，原始上升）

## 為什麼訓練仍能繼續？

Teacher 是**可選組件**：

```python
if teacher_model is not None and args.distill_weight > 0.0:
    # 蒸餾邏輯
    ...
else:
    # 跳過，只用基礎損失
    pass

# 訓練繼續
opt.zero_grad()
loss.backward()
opt.step()
```

**訓練不會崩潰**，但變成：
- 純監督學習（supervised learning）而非蒸餾（distillation）
- 從軌跡標籤學習，而非從 teacher 模型學習

## 解決方案

### 已修復

```bash
rsync -av /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/mdp-decision-transformer/results/exp_H_qk_encoder_on_atom_d128_20251026_233228/ \
          results/exp_H_qk_encoder_on_atom_d128_20251026_233228/
```

**結果**：
- ✓ Teacher checkpoint: 132B → 3.7MB
- ✓ 可以正常載入：`torch.load()` 成功
- ✓ 包含完整的 model_state_dict

### 驗證

```bash
python3 -c "
import torch
ckpt = torch.load('results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth', weights_only=False)
print('Model keys:', len(ckpt['model_state_dict']))
"
# 輸出: Model keys: 21
```

### 更新 setup 腳本

`setup_lfs_files.sh` 已更新，包含複製 teacher model。

## 重新訓練建議

### 清除舊訓練結果

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer
rm -rf results/dt_min_qk_kmeans_distill_reproduction/*
```

### 重新設置

```bash
./setup_lfs_files.sh  # 確保所有 LFS 文件都正確
```

### 重新訓練

```bash
./reproduce_8d5f10b.sh
```

### 預期結果

**應該看到**：
1. ✓ 無 `[Teacher loader] Exception` 錯誤
2. ✓ `[Teacher loader] Missing keys: []` 和 `Unexpected keys: []`
3. ✓ `teacher qk t=0: ~0.95`（接近 1.0）
4. ✓ Angle accuracy 持續上升（0.19 → 0.41）
5. ✓ Loss 收斂更快

## 技術細節

### Teacher Model 架構

`FullTransformerRoutedSoftOMP` from `scripts/omp-transformer-ldv.py`:
- Transformer encoder with QK routing
- 用於從殘差 r_t 預測 expert scores
- 配置：`d_model=128, nhead=2, nlayers=1`

### QK Scoring 機制

```python
def qk_scores_with_config(model, D, r):
    """
    用 teacher 的 QK attention 計算 expert scores
    """
    # 1. Teacher 接收殘差 r_t
    # 2. 生成 query vectors
    # 3. 與 dictionary atoms 的 keys 做 attention
    # 4. 聚合到 expert-level scores
    # 5. 返回 (E,) 維度的 logits
```

## 總結

| 項目 | 無 Teacher（錯誤） | 有 Teacher（正確） |
|------|-------------------|-------------------|
| **載入狀態** | ❌ Exception | ✓ 成功 |
| **訓練損失** | `loss_ce` | `loss_ce + 0.7*KL` |
| **Teacher QK acc** | 0.027 | 0.946 |
| **Angle acc 趨勢** | 下降 | 上升 |
| **最終 angle acc** | ~0.16 | ~0.41 |
| **收斂速度** | 較慢 | 較快 |
| **符合原始實驗** | ❌ | ✓ |

**結論**：`invalid load key, 'v'.` 是因為 teacher checkpoint 是 LFS pointer。這**嚴重影響**訓練效果，導致無蒸餾學習、角度精度下降。必須修復後重新訓練才能重現原始實驗結果。

---

**修復完成**，可以重新開始訓練了！
