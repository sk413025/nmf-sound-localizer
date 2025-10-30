# Patches 與 Transformer 的關係詳解

## 一、Patch 的物理意義

### 1.1 什麼是 Patch？

每個 patch 是頻譜圖中的一個**局部時頻區塊**，包含特定頻率範圍在特定時間窗口內的能量分布信息。

**實際維度（從執行日誌驗證）：**
```
原始頻譜圖 Y: (116, 189)
- 116 個頻率 bins (300-3000 Hz)
- 189 個時間幀 (2秒音訊)

每個 Patch: 16×10 的矩陣
- 16 個頻率 bins (~385 Hz 範圍)
- 10 個時間幀 (~0.11 秒窗口)
- 總共 160 個數值

Patch 網格: 7×18 = 126 個 patches
```

### 1.2 Patch 的物理含義

每個 patch 代表的物理量：

```python
Patch P_ij 的物理意義：
- 頻率範圍: [300 + i*385, 300 + (i+1)*385] Hz
- 時間範圍: [j*0.11, (j+1)*0.11] 秒
- 內容: 該時頻區域的聲音能量分布

例如 P_3_8:
- 頻率: 1455-1840 Hz (中頻段)
- 時間: 0.88-0.99 秒
- 物理意義: 音訊在第 0.88-0.99 秒內，1455-1840 Hz 頻段的能量模式
```

### 1.3 Patch 的量化過程

```python
# 1. 提取原始 patch (16×10 矩陣)
patch_raw = Y[i*16:(i+1)*16, j*10:(j+1)*10]

# 2. 轉換為對數刻度（更符合人耳感知）
patch_log = log(max(patch_raw, 1e-12))

# 3. 計算平均能量
energy_mean = mean(patch_log)  # 單一數值

# 4. 量化為 16 個離散等級 (0-15)
level = clip((energy_mean + 15) / 30 * 15, 0, 15)

# 5. 生成 token
token = "<P_{i}_{j}_{level}>"
```

**物理解釋：**
- `level = 0`: 該區域幾乎無聲（背景噪音）
- `level = 7-8`: 中等能量
- `level = 15`: 該區域有很強的聲音信號

## 二、Patch Token 的結構

### 2.1 Token 編碼格式

```
<P_i_j_k>
 │ │ │ └── k: 能量等級 (0-15)
 │ │ └──── j: 時間位置 (0-17)
 │ └────── i: 頻率位置 (0-6)
 └──────── P: Patch 標識符
```

### 2.2 Token 的空間含義

```
頻率軸 (7個band)
↑
6 | P_6_0  P_6_1  ...  P_6_17  | 2615-3000 Hz (最高頻)
5 | P_5_0  P_5_1  ...  P_5_17  | 2230-2615 Hz
4 | P_4_0  P_4_1  ...  P_4_17  | 1845-2230 Hz
3 | P_3_0  P_3_1  ...  P_3_17  | 1460-1845 Hz (中頻)
2 | P_2_0  P_2_1  ...  P_2_17  | 1075-1460 Hz
1 | P_1_0  P_1_1  ...  P_1_17  | 690-1075 Hz
0 | P_0_0  P_0_1  ...  P_0_17  | 305-690 Hz (最低頻)
  +--------------------------------> 時間軸 (18個窗口)
    0.0s   0.11s  ...   1.98s
```

## 三、Transformer 如何處理 Patches

### 3.1 Token 序列化

126 個 patches 被展開成一維序列：

```python
序列 = [CLS, P_0_0, P_0_1, ..., P_0_17, P_1_0, ..., P_6_17]
長度 = 127 (包含 [CLS] token)
```

### 3.2 Embedding 階段（詳細解釋）

#### 3.2.1 Embedding 層的創建和初始化

**模型初始化時創建兩個 Embedding 層：**

```python
class TransformerPolicy(nn.Module):
    def __init__(self, vocab_size=129, n_dirs=24, d_model=256):
        super().__init__()
        # 創建 Token Embedding 層（可學習的查找表）
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        # 矩陣大小: 129 × 256 = 33,024 個參數
        # 初始化: N(0,1) 隨機正態分布

        # 創建 Position Embedding 層
        self.pos_embed = nn.Embedding(max_len=512, d_model)
        # 矩陣大小: 512 × 256 = 131,072 個參數
        # 初始化: N(0,1) 隨機正態分布
```

**Embedding 矩陣結構：**
- `tok_embed.weight[0]`: `[PAD]` token 的 256 維表示
- `tok_embed.weight[1]`: `[CLS]` token 的 256 維表示
- `tok_embed.weight[10]`: `<P_0_0_4>` 的 256 維表示
- `tok_embed.weight[125]`: `<P_6_17_8>` 的 256 維表示

#### 3.2.2 Forward Pass 中的計算過程

```python
def forward(self, input_ids, attention_mask):
    B, L = input_ids.shape  # B=1 (batch), L=127 (序列長度)

    # Step 1: Token Embedding 查找
    # input_ids = [1, 10, 11, 25, 48, ...] (token IDs)
    tok_embeddings = self.tok_embed(input_ids)
    # 過程: 對每個 ID，從 129×256 矩陣中查找對應行
    # tok_embeddings[0,0,:] = tok_embed.weight[1,:]  # [CLS]
    # tok_embeddings[0,1,:] = tok_embed.weight[10,:] # P_0_0_4
    # 輸出: (1, 127, 256)

    # Step 2: Position Embedding 查找
    pos_idx = torch.arange(L)  # [0, 1, 2, ..., 126]
    pos = pos_idx.unsqueeze(0).expand(B, L)  # (1, 127)
    pos_embeddings = self.pos_embed(pos)
    # 過程: 對每個位置，從 512×256 矩陣中查找對應行
    # pos_embeddings[0,0,:] = pos_embed.weight[0,:]  # 位置 0
    # pos_embeddings[0,63,:] = pos_embed.weight[63,:] # 位置 63
    # 輸出: (1, 127, 256)

    # Step 3: Element-wise 相加
    x = tok_embeddings + pos_embeddings
    # 每個位置的最終表示 = token 語義 + 位置信息
    # x[0,i,:] = tok_embeddings[0,i,:] + pos_embeddings[0,i,:]
    # 輸出: (1, 127, 256) → 進入 Transformer
```

#### 3.2.3 Embedding 的物理意義和學習目標

**Token Embedding 學習什麼：**
```python
# 初始（隨機）
embedding(<P_0_0_4>) ≈ random_vector_1
embedding(<P_3_8_12>) ≈ random_vector_2

# 訓練後（學習到的語義）
embedding(<P_0_0_4>) → 編碼 "低頻低能量" 的特徵向量
embedding(<P_3_8_12>) → 編碼 "中頻高能量" 的特徵向量

# 相似的 patches 會有相似的 embeddings
similarity(P_3_8_12, P_3_8_11) > similarity(P_3_8_12, P_0_0_1)
```

**Position Embedding 學習什麼：**
```python
# 編碼序列中的絕對位置
embedding(pos_0) → "序列開始，對應 patch P_0_0"
embedding(pos_63) → "序列中間，對應 patch P_3_9"
embedding(pos_126) → "序列末尾，對應 patch P_6_17"

# 使 Transformer 能夠理解時間順序
# 例如：早期 patches (0-40) vs 後期 patches (80-126)
```

#### 3.2.4 訓練過程中的更新

```python
# 梯度回傳路徑
Loss ← Direction_Logits ← Pooled ← Transformer ← Embeddings
                                                      ↑
                                              梯度更新這裡

# 更新規則（簡化）
if prediction_error:
    # 如果 P_3_8_12 應該指向 0° 但預測錯誤
    # 則調整 tok_embed.weight[token_id_of_P_3_8_12]
    # 使其更能表達 "指向 0°" 的特徵

    # 同時調整相關位置的 position embeddings
    # 使位置信息更好地輔助方向預測
```

#### 3.2.5 為什麼需要 Embedding？

1. **離散到連續的橋樑**: Transformer 需要連續向量，不能處理離散 IDs
2. **降維表示**: 129 個 tokens → 256 維連續空間（更豐富的表達）
3. **可學習性**: 通過訓練自動優化表示
4. **組合性**: Token + Position 信息同時編碼

### 3.3 Self-Attention 機制

Transformer 通過 attention 學習 patches 之間的關係：

#### A. 時間相關性
```
同頻率不同時間的 patches:
P_3_0 ← attention → P_3_1 ← attention → P_3_2
學習: 聲音在 1460-1845 Hz 頻段隨時間的演化
```

#### B. 頻譜相關性
```
同時間不同頻率的 patches:
P_0_5 ← attention → P_1_5 ← attention → P_2_5
學習: 在 0.55 秒時刻的頻譜結構（諧波關係）
```

#### C. 時頻軌跡
```
對角線 patches:
P_0_0 ← attention → P_1_2 ← attention → P_2_4
學習: 頻率隨時間變化的模式（如都卜勒效應）
```

### 3.4 方向特徵學習

不同方向產生不同的 patch 模式：

```python
# 方向 0° 的典型模式
Direction_0 = {
    "強能量": [P_2_*, P_3_*],  # 中頻段強
    "弱能量": [P_0_*, P_6_*],  # 低頻和高頻弱
    "時間模式": "能量集中在前半段"
}

# 方向 90° 的典型模式
Direction_90 = {
    "強能量": [P_0_*, P_1_*],  # 低頻段強
    "弱能量": [P_5_*, P_6_*],  # 高頻弱
    "時間模式": "能量均勻分布"
}
```

## 四、Attention 權重的物理解釋

### 4.1 Attention 矩陣含義

```python
Attention[i,j] = Patch_i 對 Patch_j 的注意力權重

高權重表示:
- 這兩個 patches 經常一起出現
- 它們共同指示某個方向
- 存在物理上的因果關係
```

### 4.2 學習到的模式示例

```python
# 1. 諧波結構
如果 P_1_5 (基頻) 有高能量
則 P_2_5 (二次諧波) 和 P_3_5 (三次諧波) 也應有能量
→ Attention[P_1_5, P_2_5] 會很高

# 2. 時間連續性
聲音不會突然消失
→ Attention[P_i_j, P_i_(j+1)] 通常較高

# 3. 方向特異性
某些 patch 組合只在特定方向出現
→ 這些 patches 之間的 attention 會很高
```

## 五、從 Patches 到方向預測

### 5.1 信息聚合流程

```
126 patches → 127 tokens (with CLS)
    ↓ (Embedding)
127 × 256 embeddings
    ↓ (2-layer Transformer)
127 × 256 contextualized representations
    ↓ (Mean Pooling)
1 × 256 global representation
    ↓ (Linear Classifier)
1 × 24 direction logits
    ↓ (Argmax)
Predicted direction (0-23)
```

### 5.2 決策過程

```python
# Transformer 學習的決策邏輯（簡化示例）
def predict_direction(patches):
    # 檢查低頻能量集中度
    if sum(P_0_* + P_1_*) > threshold_1:
        if P_0_5 > P_0_10:  # 能量在前半段
            return Direction_90
        else:
            return Direction_270

    # 檢查中頻模式
    if sum(P_3_* + P_4_*) > threshold_2:
        if temporal_pattern_matches("rising"):
            return Direction_0
        else:
            return Direction_180

    # 更複雜的組合模式...
```

## 六、關鍵洞察

### 6.1 為什麼使用 Patches？

1. **降維**: 從 116×189=21,924 個值降到 126 個 tokens
2. **離散化**: 連續值變成 16 個等級，減少噪音影響
3. **局部特徵**: 每個 patch 捕捉局部時頻模式
4. **可解釋性**: Token 格式明確表示位置和能量

### 6.2 Transformer 的優勢

1. **全局依賴**: Self-attention 可以捕捉任意距離的 patches 關係
2. **並行處理**: 所有 patches 同時處理，不像 RNN 需要序列處理
3. **位置編碼**: 保留 patches 的空間結構信息
4. **可學習關係**: 自動發現哪些 patch 組合對方向預測重要

### 6.3 物理直覺

- **不同方向** → **不同的聲學傳遞函數** → **不同的頻譜模式** → **不同的 patch 激活模式**
- Transformer 學習識別這些方向特異的 patch 模式組合

## 七、實際例子

### 聲源在 45° 方向時的 Patch 模式

```python
# 45° 方向的典型 patch 激活
Active_patches = [
    "P_1_5_10",   # 低頻，中間時刻，高能量
    "P_2_5_12",   # 中低頻，中間時刻，很高能量
    "P_2_6_11",   # 中低頻，稍後時刻，高能量
    "P_3_5_8",    # 中頻，中間時刻，中等能量
]

# Transformer 學到的規則
if (P_1_5 > 10 and P_2_5 > 12 and P_3_5 < 9):
    confidence_45_degree = high
```

這種模式識別是通過大量訓練數據自動學習的，而不是手工設計的規則。

## 總結

Patches 將連續的時頻信號離散化為可管理的 tokens，每個 token 編碼了**位置**（頻率-時間）和**能量等級**信息。Transformer 通過 self-attention 機制學習這些 patches 之間的複雜關係，最終聚合所有 patch 信息來預測聲源方向。這個過程本質上是學習不同方向的聲學特徵在時頻域的分布模式。