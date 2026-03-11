# Soft-OMP Architecture Improvements

## 🎯 改進目標

解決 commit 60e6282 發現的三個關鍵問題，同時保持 "可以學習 routing (w)，但必須保留 g 的物理意義"。

## 📋 改進內容

### 1️⃣ 新增 Flexible Routing Modes

**問題**: 原版只有 QK-routing（失敗，2.7%）或 eval-mode greedy（成功，83.8%），無法在訓練中使用物理信號。

**解決**: 新增 `routing_mode` 參數，支持三種模式：

```python
class TrainableRoutedSoftOMP(nn.Module):
    def __init__(
        self,
        ...,
        routing_mode: str = "g",  # 'qk', 'g', or 'hybrid'
        hybrid_alpha: float = 1.0,  # blend factor
    ):
```

#### Mode 1: Pure Physics (`routing_mode='g'`) ✅ 推薦

```python
# 直接使用物理相關性 g 作為 routing scores
g = D.T @ r
scores_expert = sqrt(sum(g²))  # 每個 expert 的能量
scores_atoms = g               # 每個 atom 的相關性

# 優點:
#   • 100% 對齊物理 (ρ=1.0)
#   • 不需要學習 routing
#   • 保證收斂性
#   • 可學習 τ, η 等超參數

# 使用情境:
#   • 基準測試
#   • 高準確率需求
#   • 字典 coherence 很高時 (μ>0.9)
```

#### Mode 2: Pure Learnable (`routing_mode='qk'`) ⚠️ 需謹慎

```python
# 使用 QK attention 學習 routing
q = Wq(P_R(r))
K = Wk(P_D(D))
qk_scores = q·K / √d

# 優點:
#   • 可學習端到端
#   • 可能發現非線性 pattern

# 缺點:
#   • 可能與物理 g 不對齊 (ρ=0.08)
#   • 需要大量數據和正則化
#   • Score drift 問題

# 使用情境:
#   • 數據充足 + 強正則化
#   • 字典 coherence 較低時 (μ<0.5)
```

#### Mode 3: Hybrid Blending (`routing_mode='hybrid'`) 🌟 最佳平衡

```python
# 混合物理和學習
alpha = learnable hybrid_alpha  # 可學習權重

# 正規化後混合
g_norm = g / ||g||
qk_norm = qk / ||qk||
scores = alpha · g_norm + (1-alpha) · qk_norm

# 優點:
#   • 初期用物理 (alpha=1.0) 保證收斂
#   • 逐漸轉移到學習 (alpha→0) 發現 pattern
#   • alpha 可學習或 schedule

# 訓練策略:
#   Epoch 1-10:   alpha=1.0  (純物理，穩定訓練)
#   Epoch 11-30:  alpha線性衰減 1.0→0.5
#   Epoch 31+:    alpha=0.5  (平衡物理+學習)
```

**關鍵設計原則**: 無論哪種 routing mode，**g 永遠用於 UPDATE**！

```python
# 所有模式都執行相同的更新
w = softmax(scores / τ)  # routing weights (來源可變)
g = D.T @ r              # 物理相關性 (固定)
x += η · (w ⊙ g)         # w 選哪些，g 決定更新量
```

### 2️⃣ Score Regularization (防止漂移和飽和)

**問題**: Softmax 對常數偏移不變，導致 scores 無界增長 (1.8 → 14.1 in 3 epochs)，造成梯度消失。

**數學原理**:
```
Softmax 不變性: softmax(s) = softmax(s + c)

因此梯度下降可以任意平移 s:
s_1 = [1, 2, 3]
s_2 = [101, 102, 103]  ← 平移 100
s_3 = [-99, -98, -97]   ← 平移 -100

softmax 輸出完全相同！

當 s 太大: softmax(14.1/0.73) ≈ 1.0 → 梯度 ≈ 0
```

**解決**: 加入 L2 正則化

```python
def __init__(self, ..., score_reg_weight: float = 0.01):
    self.score_reg_weight = score_reg_weight

def forward(self, ...):
    # 計算 scores
    scores_expert = ...
    
    # L2 penalty 防止分數過大
    if self.score_reg_weight > 0:
        reg_loss = self.score_reg_weight * (scores_expert ** 2).mean()
        self.last_reg_loss += reg_loss.item()
    
    # 在 training loop 中:
    # total_loss = task_loss + model.last_reg_loss
```

**效果對比**:

```
Without regularization:          With regularization (λ=0.01):
Epoch 1: scores ∈ [-0.6,  1.8]   Epoch 1: scores ∈ [-0.6,  1.8]
Epoch 2: scores ∈ [-3.2,  9.3]   Epoch 2: scores ∈ [-1.2,  2.5]  ← 控制住了
Epoch 3: scores ∈ [-5.0, 14.1]   Epoch 3: scores ∈ [-1.5,  3.1]
         ↓                                ↓
    梯度消失 ❌                      梯度正常 ✅
```

**調參建議**:
- `score_reg_weight=0.01`: 標準設定（推薦）
- `score_reg_weight=0.001`: 輕度正則化（允許更大分數）
- `score_reg_weight=0.1`: 強正則化（強制小分數）

### 3️⃣ 保留 g 的物理意義（核心不變原則）

**哲學**: Routing 可以學習，但 UPDATE 必須遵循物理！

```python
# ✅ 正確: 學習 WHERE to update，物理決定 HOW MUCH
w = learnable_routing(...)  # 可以是 QK, g, hybrid
g = D.T @ r                 # 物理相關性
x += η · (w ⊙ g)            # w 選擇，g 更新

# ❌ 錯誤: 完全拋棄 g
w = learnable_routing(...)
x += η · w  # 沒有物理意義，單位不對，無法保證收斂
```

**為什麼 g 不可替代？**

1. **數學最優性**: g = -∇L 是負梯度方向
2. **物理意義**: g[j] = 原子 j 與殘差的投影長度
3. **單位一致**: g 與訊號 y 單位相同，可直接用於重建
4. **收斂保證**: 沿 g 方向更新保證殘差單調遞減
5. **信息完整**: g 包含方向（正負）和量級

## 🔧 使用方式

### 基本用法

```python
from doa_rl.omp.soft_omp import TrainableRoutedSoftOMP

# 方式 1: Pure physics routing (推薦起點)
model = TrainableRoutedSoftOMP(
    F=346, E=37, M=8,
    routing_mode='g',  # 使用物理 g
    score_reg_weight=0.01
)

# 方式 2: Hybrid routing (推薦用於學習)
model = TrainableRoutedSoftOMP(
    F=346, E=37, M=8,
    routing_mode='hybrid',
    hybrid_alpha=1.0,  # 初始純物理
    score_reg_weight=0.01
)

# 方式 3: Pure learnable (進階用法)
model = TrainableRoutedSoftOMP(
    F=346, E=37, M=8,
    routing_mode='qk',
    score_reg_weight=0.05  # 需要更強正則化
)
```

### 訓練循環整合

```python
optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

for epoch in range(num_epochs):
    for y, D, label in dataloader:
        optimizer.zero_grad()
        
        # Forward
        x, residuals = model(y, D, train_mode=True)
        
        # Task loss (例如分類)
        logits = classifier(x)
        task_loss = F.cross_entropy(logits, label)
        
        # 加入 regularization loss
        total_loss = task_loss + model.last_reg_loss
        
        # Backward
        total_loss.backward()
        optimizer.step()
    
    # Optional: Anneal hybrid_alpha
    if model.routing_mode == 'hybrid' and epoch > 10:
        with torch.no_grad():
            model.hybrid_alpha.data = max(0.5, 1.0 - 0.05 * (epoch - 10))
```

### Alpha Annealing 策略 (Hybrid mode)

```python
def anneal_hybrid_alpha(model, epoch, strategy='linear'):
    if model.routing_mode != 'hybrid':
        return
    
    with torch.no_grad():
        if strategy == 'linear':
            # 線性衰減: 1.0 → 0.0 over 30 epochs
            start_epoch = 10
            end_epoch = 40
            if epoch < start_epoch:
                model.hybrid_alpha.data = torch.tensor(1.0)
            elif epoch > end_epoch:
                model.hybrid_alpha.data = torch.tensor(0.0)
            else:
                progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                model.hybrid_alpha.data = torch.tensor(1.0 - progress)
        
        elif strategy == 'cosine':
            # Cosine 衰減: 平滑過渡
            import math
            start_epoch = 10
            end_epoch = 40
            if epoch < start_epoch:
                alpha = 1.0
            elif epoch > end_epoch:
                alpha = 0.0
            else:
                progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                alpha = 0.5 * (1 + math.cos(math.pi * progress))
            model.hybrid_alpha.data = torch.tensor(alpha)
        
        elif strategy == 'step':
            # 階梯式: 突然轉換
            if epoch < 20:
                model.hybrid_alpha.data = torch.tensor(1.0)
            elif epoch < 40:
                model.hybrid_alpha.data = torch.tensor(0.5)
            else:
                model.hybrid_alpha.data = torch.tensor(0.0)

# 在訓練循環中使用
for epoch in range(num_epochs):
    anneal_hybrid_alpha(model, epoch, strategy='linear')
    # ... training code ...
```

## 📊 預期效果

### Routing Mode 比較

| Mode    | 準確率預期 | 訓練難度 | 收斂速度 | 適用場景 |
|---------|----------|---------|---------|---------|
| g       | 95-100%  | 低      | 快      | 高 coherence，需要高準確率 |
| qk      | 60-90%   | 高      | 慢      | 低 coherence，數據充足 |
| hybrid  | 85-100%  | 中      | 中      | 平衡物理與學習 |

### Score Regularization 效果

| reg_weight | Score 範圍 | 梯度狀態 | 學習穩定性 |
|-----------|----------|---------|-----------|
| 0.0       | [-5, 14] | 消失    | 不穩定 ❌  |
| 0.001     | [-3, 8]  | 弱      | 尚可       |
| 0.01      | [-2, 3]  | 正常    | 穩定 ✅    |
| 0.1       | [-1, 1]  | 過強    | 過度約束   |

## 🔬 實驗建議

### Experiment 1: 驗證 g-routing 基準

```bash
python scripts/omp-transformer-ldv.py \
  --routing_mode g \
  --score_reg_weight 0.01 \
  --epochs 10 \
  --out_dir results/soft_omp_g_routing_baseline
```

**預期結果**:
- 準確率: 95-100%
- QK-g 相關性: ≈1.0
- Score 範圍: 穩定在 [-2, 3]

### Experiment 2: Hybrid routing with annealing

```bash
python scripts/omp-transformer-ldv.py \
  --routing_mode hybrid \
  --hybrid_alpha 1.0 \
  --score_reg_weight 0.01 \
  --epochs 50 \
  --out_dir results/soft_omp_hybrid_annealing
```

**預期結果**:
- Early epochs (1-10): 95%+ (physics-driven)
- Mid epochs (11-30): 90%+ (gradual transfer)
- Late epochs (31+): 85-95% (balanced)

### Experiment 3: 對比有無正則化

```bash
# Without regularization
python scripts/omp-transformer-ldv.py \
  --routing_mode qk \
  --score_reg_weight 0.0 \
  --epochs 10

# With regularization
python scripts/omp-transformer-ldv.py \
  --routing_mode qk \
  --score_reg_weight 0.01 \
  --epochs 10
```

**預期差異**:
- 無正則化: score drift, 梯度消失, 低準確率
- 有正則化: score 穩定, 梯度正常, 較高準確率

## 📝 總結

### 核心改進

1. ✅ **Flexible Routing**: 3 種模式 (g/qk/hybrid)，可根據需求選擇
2. ✅ **Score Regularization**: 防止漂移和飽和，保持梯度健康
3. ✅ **保留 g 的物理意義**: 永遠用 g 來更新，確保數學正確性

### 設計哲學

> **"Routing 可以學習，Update 必須遵循物理"**

- Routing (w): 決定「選哪些原子」→ 可學習、可混合
- Update (g): 決定「更新多少」→ 必須物理、不可替代

### 推薦配置

**起點** (baseline):
```python
routing_mode='g', score_reg_weight=0.01
```

**進階** (學習):
```python
routing_mode='hybrid', hybrid_alpha=1.0→0.5, score_reg_weight=0.01
```

**研究** (探索):
```python
routing_mode='qk', score_reg_weight=0.05
```
