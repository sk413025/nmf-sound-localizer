# Transformer 輸出分析與方向預測機制

## 一、架構類型：Sequence-to-Classification

### ❌ 不是 Sequence-to-Sequence
- **Seq2Seq 特徵**: 輸入序列 → 輸出序列（如翻譯）
- **有 Encoder + Decoder**
- **輸出可變長度序列**
- **每個位置輸出一個 token**

### ✅ 是 Sequence-to-Classification
- **輸入**: 127 個 patch tokens（序列）
- **輸出**: 1 個方向類別（24 選 1）
- **只有 Encoder**（無 Decoder）
- **使用 Mean Pooling 聚合信息**
- **Linear 層做最終分類**

## 二、Transformer 輸出維度詳解

### 完整維度流程

```
階段                     形狀                描述
─────────────────────────────────────────────────────
1. Input IDs           (1, 127)            Token 序列
2. Embeddings          (1, 127, 256)       Token + Position
3. Transformer Layer 1 (1, 127, 256)       局部關係
4. Transformer Layer 2 (1, 127, 256)       全局模式
5. Layer Norm          (1, 127, 256)       歸一化
6. Mean Pooling        (1, 256)            聚合成單向量
7. Linear Head         (1, 24)             24 個 logits
8. Softmax            (1, 24)             概率分布
9. Argmax             scalar              方向 index
```

### 關鍵轉換點

1. **Transformer 保持序列長度**
   - 輸入: `(1, 127, 256)`
   - 輸出: `(1, 127, 256)`
   - 每個 token 都有 contextualized representation

2. **Pooling 壓縮序列**
   - 輸入: `(1, 127, 256)` - 127 個向量
   - 輸出: `(1, 256)` - 1 個全局向量
   - 方法: Mean Pooling（平均所有 token）

3. **Classification Head 產生 logits**
   - 輸入: `(1, 256)` - 全局特徵
   - 輸出: `(1, 24)` - 24 個方向的分數
   - 方法: Linear(256, 24)

## 三、方向沒有 Tokens！

### 輸入側（有 tokens）
```python
# 129 種可能的 input tokens
tokens = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "<P_0_0_0>": 3,
    "<P_0_0_1>": 4,
    ...
    "<P_6_17_15>": 128
}
```

### 輸出側（沒有 tokens）
```python
# 24 個方向類別（不是 tokens）
directions = {
    0: "0°",
    1: "15°",
    2: "30°",
    ...
    23: "345°"
}

# 輸出是 class index，不是 token ID
prediction = 5  # 表示 75°，不是 token
```

### 為什麼不需要方向 tokens？
1. **分類任務 vs 生成任務**
   - 分類: 選擇預定義類別
   - 生成: 產生 token 序列

2. **固定輸出空間**
   - 只有 24 個可能的輸出
   - 不需要詞彙表或 decoder

## 四、Transformer 如何決定方向

### 4.1 Self-Attention 學習 Patch 關係

```python
# 每個 patch 通過 attention 關注其他 patches
Attention_matrix[i,j] = similarity(Q[i], K[j])

# 例如 P_3_8 會關注:
- P_3_7, P_3_9     # 時間相鄰
- P_2_8, P_4_8     # 頻率相鄰
- P_1_4, P_5_12    # 可能的諧波
```

### 4.2 兩層 Transformer 的分工

**Layer 1**: 學習局部關係
- 相鄰 patches 的時間連續性
- 頻譜上的諧波結構
- 局部能量模式

**Layer 2**: 學習全局模式
- 整體頻譜分布
- 時間演化軌跡
- 方向特異性模式

### 4.3 Mean Pooling 聚合信息

```python
# 將所有 token 的信息聚合
pooled = sum(h[i] * mask[i] for i in range(127)) / 127
# pooled: (256,) 包含所有 patches 的聚合信息
```

### 4.4 Linear Classification Head

```python
# 權重矩陣 W: (256, 24)
# 每一列是一個方向的 "檢測器"

W[:, 0]  # 檢測 0° 的權重向量
W[:, 1]  # 檢測 15° 的權重向量
...
W[:, 23] # 檢測 345° 的權重向量

# 計算每個方向的分數
logits = pooled @ W.T + bias
```

### 4.5 最終決策

```python
# 方法 1: Argmax（確定性）
prediction = torch.argmax(logits)  # 0-23

# 方法 2: Softmax（概率性）
probabilities = torch.softmax(logits, dim=-1)
# 可以看到每個方向的信心度

# 實例輸出
logits = [2.1, -0.5, 3.2, ..., 1.5]  # 24 個值
probs = [0.12, 0.02, 0.35, ..., 0.08]  # 總和 = 1.0
prediction = 2  # 30° (最高分數)
```

## 五、完整計算流程示例

### 輸入
```python
# 127 個 patch tokens
input_tokens = ["[CLS]", "<P_0_0_4>", "<P_0_1_5>", ..., "<P_6_17_8>"]
input_ids = [1, 10, 15, ..., 125]  # Shape: (1, 127)
```

### 處理過程
```python
# 1. Embedding
embeddings = tok_embed(input_ids) + pos_embed(positions)
# Shape: (1, 127, 256)

# 2. Transformer Encoder
h = transformer_encoder(embeddings)
# Shape: (1, 127, 256)

# 3. Pooling
pooled = mean(h, dim=1)
# Shape: (1, 256)

# 4. Classification
logits = linear(pooled)
# Shape: (1, 24)

# 5. Prediction
direction_idx = argmax(logits)  # 例如: 7
direction = direction_idx * 15  # 105°
```

### 實際輸出示例
```python
# Logits (未歸一化分數)
logits = tensor([[0.54, -0.21, 0.34, ..., -0.15]])

# Probabilities (歸一化概率)
probs = tensor([[0.041, 0.032, 0.055, ..., 0.033]])

# 預測結果
Top 1: 105° (5.85%)
Top 2: 30° (5.52%)
Top 3: 120° (5.08%)
```

## 六、訓練監督信號

### 標籤格式
```python
# 訓練數據
input: 127 patch tokens
target: 5  # 方向 index (75°)，不是 token
```

### Loss 計算
```python
# Cross-Entropy Loss
logits = model(input_ids)  # (1, 24)
loss = F.cross_entropy(logits, target)

# 梯度
∂L/∂logits[i] = probs[i] - 1(i==target)
```

### 學習目標
1. 最大化正確方向的 logit
2. 最小化錯誤方向的 logits
3. 學習 patch 模式 → 方向的映射

## 七、關鍵洞察

### 架構選擇的合理性

1. **為什麼不用 Seq2Seq？**
   - 輸出是單一類別，不是序列
   - 不需要生成 tokens
   - 分類任務更簡單高效

2. **為什麼用 Mean Pooling？**
   - 所有 patches 都對方向有貢獻
   - 順序不敏感（patch 位置已編碼）
   - 簡單有效的聚合方法

3. **為什麼只用 2 層 Transformer？**
   - 任務相對簡單（24 分類）
   - 避免過擬合
   - 計算效率高

### 模型容量分析

```
參數分布:
- Embeddings: 164K (主要部分)
- Transformer: ~1.5M
- Classification Head: 6K
總計: ~1.74M 參數

這對於 24 分類任務是合適的容量
```

## 總結

這個系統是典型的 **Sequence-to-Classification** 架構：
- **輸入**: 127 個 patch tokens（序列）
- **處理**: Transformer Encoder + Mean Pooling
- **輸出**: 24 個方向之一（分類）
- **沒有方向 tokens**，直接輸出類別 index
- **Transformer 輸出維度**: 始終保持 `(1, 127, 256)`，直到 pooling
- **最終決策**: Linear 層產生 24 個 logits，argmax 選擇方向