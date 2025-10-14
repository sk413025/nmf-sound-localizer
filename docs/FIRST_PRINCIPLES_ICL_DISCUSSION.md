# 第一性原理討論：為什麼需要 Multi-Modal ICL？

## 📚 目錄
1. [四個核心問題的第一性原理分析](#part1)
2. [你的 Domain Knowledge 至關重要的地方](#part2)
3. [我的設計假設 vs 你需要驗證的點](#part3)

---

<a name="part1"></a>
## Part 1: 四個核心問題的第一性原理分析

### ❌ 問題 1: 單一模態 — 只有 Patch tokens，缺乏物理意義

#### 🔬 第一性原理：什麼是「模態」？

在多模態學習中，**模態（Modality）** 指的是**不同的資訊表示方式**。

舉例：
- **視覺模態**：圖片 (pixels)
- **聽覺模態**：聲音 (waveform)
- **文字模態**：語言 (words)

在我們的聲學定位問題中，同一個聲音訊號 **Y(F, N)** 可以有多種表示方式：

```
同一個聲音 Y(F,N)
    ↓
├─【空間模態】頻譜小塊 → Patch tokens
├─【結構模態】頻譜組成 → NMF Atom tokens  
└─【物理模態】方向相關 → Direction Projection tokens
```

#### 📊 現狀問題：只看「樹」不見「林」

**現有系統**只用 Patch tokens：
```python
Y(F,N) → PatchTokenizer → [<P_0_0_5>, <P_0_1_8>, <P_1_0_12>, ...]
```

這就像把一張**臉部照片切成 16×16 的小方格**，然後說：
```
"這個方格是 RGB(120, 80, 90)"
"那個方格是 RGB(130, 85, 95)"
...
```

**問題在哪？**
1. ❌ **局部資訊**：每個 token 只看到一小塊時頻區域
2. ❌ **無全域結構**：不知道「這些小塊組合起來代表什麼」
3. ❌ **無物理意義**：不知道「這個頻譜對應哪個方向」

#### 🎯 類比：人類如何辨識方向？

假設你在一個房間聽到聲音，你的大腦會：

1. **全域感知**（對應 Direction tokens）
   - "聲音好像從右前方來"
   - "大概 80-90 度"
   
2. **結構分析**（對應 Atom tokens）
   - "這是人聲，不是音樂"
   - "有低頻共振，可能在牆角"
   
3. **細節驗證**（對應 Patch tokens）
   - "500Hz 那邊能量特別高"
   - "2000Hz 有個峰值"

**三者缺一不可！**

#### 🧪 數學上的證明

考慮兩個頻譜 Y₁ 和 Y₂：

```python
# 場景 1: 90° 的人聲
Y₁ = H[:, 90] ⊙ S_speech  
# → Patches 可能是: [<P_0_0_8>, <P_1_2_12>, ...]

# 場景 2: 95° 的噪音（頻譜相似但方向不同）
Y₂ = H[:, 95] ⊙ S_noise
# → Patches 可能也是: [<P_0_0_8>, <P_1_2_11>, ...]  ← 很接近！
```

**如果只看 Patches**，模型會混淆這兩個場景！

**但如果加入 Direction tokens**：
```python
Y₁ → [<R_090:14>, <P_0_0_8>, ...]  # 90° 相關性強
Y₂ → [<R_095:13>, <P_0_0_8>, ...]  # 95° 相關性強
```

現在模型有了**額外的物理線索**來區分！

---

### ❌ 問題 2: 無多模態融合 — 沒有整合 NMF atoms、Direction projection

#### 🔬 第一性原理：聲學的物理模型

回憶一下你的系統的物理公式：

```
Y(f, t) = H(f, d) ⊙ S(f, t) + noise
```

其中：
- **Y**: 觀測到的頻譜（麥克風錄到的）
- **H**: 轉移函數（方向 d 的聲學特性）
- **S**: 源訊號（聲音本身的頻譜）

#### 🧩 三種 Token 對應物理模型的不同部分

| Token 類型 | 物理對應 | 提供的資訊 |
|-----------|---------|-----------|
| **Patch tokens** | Y 的局部觀測 | 「我看到這個時頻點有能量」 |
| **Atom tokens** | S 的結構分解 | 「聲源由這些頻譜成分組成」 |
| **Direction tokens** | H 的方向特徵 | 「頻譜與這個方向的 H 最匹配」 |

#### 💡 為什麼需要多模態融合？

**第一性原理：反問題的多解性**

聲源定位是一個**病態反問題（Ill-posed Inverse Problem）**：

```
給定 Y，求 d 使得 Y ≈ H(·, d) ⊙ S
```

**問題**：
1. S 未知（你不知道聲源的原始頻譜）
2. H 和 S 糾纏在一起（無法單獨觀測）
3. 噪音和反射讓問題更複雜

**解法：提供更多約束**

```
單一模態：
  只看 Y → 資訊不足 → 多個 d 都可能

多模態：
  Y + S估計 + H相關性 → 約束增加 → 唯一解更明確
```

#### 🔬 實驗證明（你可以驗證）

假設有兩個方向 85° 和 90°：

**情況 A：只用 Patches**
```python
Y_85 和 Y_90 的 patch tokens 可能很相似
→ 模型困惑，準確度 ~60%
```

**情況 B：加入 Direction tokens**
```python
Y_85 → <R_085:15> (85° 相關性最強)
Y_90 → <R_090:14> (90° 相關性最強)
→ 模型有額外線索，準確度 → 75%
```

**情況 C：再加入 Atom tokens**
```python
Y_85 → <R_085:15> <AT_5:12> <AT_10:8> (特定頻譜結構)
Y_90 → <R_090:14> <AT_5:11> <AT_12:9> (略有不同)
→ 更多約束，準確度 → 85%
```

#### 🎯 這就是為什麼 Transformer 的 Attention 機制重要

Transformer 的 **Self-Attention** 可以學習：

```
Attention(Q, K, V) 中：
  Q = "當前 patch token 想問：我該屬於哪個方向？"
  K = "Direction token 回答：我代表 90°"
  V = "Atom token 補充：而且頻譜結構符合人聲"
  
→ 模型自動學會融合三種資訊！
```

---

### ❌ 問題 3: 無 ICL 能力 — 無法做 few-shot learning 或 context adaptation

#### 🔬 第一性原理：什麼是 In-Context Learning？

**ICL 的核心思想**：

> 「給模型看幾個例子，它就能推廣到新例子，而不需要重新訓練。」

這是 GPT-3 證明的重要能力。

#### 📖 類比：人類如何學習新任務

假設你從未見過「泰文」，我給你這些例子：

```
例子 1: "สวัสดี" → "你好"
例子 2: "ขอบคุณ" → "謝謝"
例子 3: "ลาก่อน" → "再見"

問題: "สบายดีไหม" → ?
```

你可能猜出：「這是在問安，大概是『你好嗎』」

**這就是 ICL！你沒有「訓練」，只是從 context 學習。**

#### 🎯 在聲學定位中為什麼需要 ICL？

**場景 1：環境適應**

假設模型在**室內**訓練，但要在**戶外**使用：

```
室內 H_indoor:  反射多、混響強
戶外 H_outdoor: 反射少、直達聲強
```

**傳統方法**：
```
需要收集大量戶外數據 → 重新訓練 → 耗時數天
```

**ICL 方法**：
```python
# 給模型幾個「戶外範例」
context = [
    (Y_outdoor_1, angle_1),  # 戶外 30°
    (Y_outdoor_2, angle_2),  # 戶外 60°
    (Y_outdoor_3, angle_3),  # 戶外 90°
]

# 模型看到這些例子後，推測新的戶外聲音
query = Y_outdoor_new
prediction = model.predict_with_context(query, context)
# → 模型自動適應戶外特性！
```

**場景 2：少樣本學習**

假設你只有**每個角度 3 個樣本**（數據稀缺）：

**傳統方法**：
```
3 samples/angle × 17 angles = 51 samples
→ 過擬合，泛化差
```

**ICL 方法**：
```python
# 訓練時教模型「如何從少量例子學習」
for angle in angles:
    samples = get_samples(angle)
    context = samples[:2]  # 用 2 個當 context
    query = samples[2]     # 用 1 個當 query
    
    # 模型學會：「看到 context 的模式，推測 query」
```

測試時，即使新角度只有 2 個樣本，模型也能推廣！

#### 🧪 數學上的理解：Meta-Learning

ICL 本質上是一種 **Meta-Learning（學習如何學習）**：

```python
傳統學習：
  θ* = argmin_θ Σ L(f_θ(x_i), y_i)
  # 學習固定的參數 θ

Meta-Learning (ICL):
  θ* = argmin_θ Σ_task [ E[L(f_θ(x_query | context), y_query)] ]
  # 學習如何利用 context 來預測 query
```

在 Transformer 中，**Attention 機制天然支援 ICL**：

```python
Attention(Query, Context_Keys, Context_Values)
# → Query 可以「查詢」Context 來獲得資訊！
```

---

### ❌ 問題 4: 缺乏可解釋性 — tokens 與聲學物理特徵脫節

#### 🔬 第一性原理：為什麼需要可解釋性？

**科學研究的基本要求**：

1. **驗證正確性**：模型做對了嗎？為什麼？
2. **診斷錯誤**：模型做錯了，哪裡出問題？
3. **建立信任**：模型可以部署到真實應用嗎？

#### 📊 現狀問題：黑盒模型

**當前系統**：

```python
Input: [<P_0_0_5>, <P_0_1_8>, <P_1_0_12>, ...]
        ↓
    Transformer (黑盒)
        ↓
Output: 90° (預測)
```

**你無法回答**：
- ❓ 為什麼模型認為是 90°？
- ❓ 模型關注了哪些頻率？
- ❓ 錯誤預測是因為噪音？還是模型本身問題？

#### 🎯 Multi-Modal Tokens 提供可解釋性

**改進後的系統**：

```python
Input: [<R_090:14>, <AT_5:12>, <P_3_5_8>, ...]
        ↓
    Transformer (可視化 Attention)
        ↓
Output: 90° (預測)

解釋：
✅ <R_090:14> 的 attention weight = 0.42  ← 模型最關注這個！
✅ <AT_5:12> 的 attention weight = 0.23   ← 低頻 atom 很重要
✅ <P_3_5_8> 的 attention weight = 0.08   ← 某個時頻點有貢獻
```

#### 🔬 具體例子：診斷錯誤

**場景**：模型把 85° 誤判為 90°

**無多模態 Token**：
```
Input: [<P_0_0_5>, <P_0_1_8>, ...]
Output: 90° (錯誤！應該是 85°)
原因: ？？？ 無法診斷
```

**有多模態 Token**：
```python
Input: [<R_090:13>, <R_085:12>, <AT_5:11>, ...]
                ↑ 注意！90° 和 85° 分數接近
Output: 90° (錯誤！)

診斷：
1. 檢查 Direction tokens:
   - <R_090:13> 相關性 = 0.78
   - <R_085:12> 相關性 = 0.75  ← 非常接近！
   
2. 結論：兩個方向的 H 矩陣太相似
   → 需要更多訓練數據 or 更精細的 H 估計

3. 改進方案：
   - 加入更多 Atom tokens（結構差異）
   - 收集更多 85° 附近的數據
```

#### 🎓 學術價值：可發表的研究

有了可解釋性，你可以寫論文：

```
Title: "Physics-Guided Multi-Modal Tokens for Interpretable 
        Sound Source Localization"

Contributions:
1. 證明 Direction tokens 提供物理先驗
2. 證明 Atom tokens 捕捉頻譜結構
3. Attention 可視化驗證模型推理符合物理直覺
4. 錯誤案例分析指出改進方向
```

這比「黑盒模型提升 5% 準確度」有價值得多！

---

<a name="part2"></a>
## Part 2: 你的 Domain Knowledge 至關重要的地方

### 🎯 重要聲明

我作為 AI assistant，可以設計**軟體架構**和**通用 ML 框架**，但以下問題**必須由你的聲學物理專業知識來回答**：

---

### 🔴 Critical Question 1: Direction Token 的計算方式

**我的假設**（需要你驗證）：

```python
class DirectionProjectionTokenizer:
    def __call__(self, Y, top_m=5):
        Ybar = Y.mean(axis=1)  # 時間平均
        
        for d in range(D):
            H_d = self.H[:, d]
            
            # 方法 A: 相關性
            score = np.dot(Ybar, H_d) / (norm(Ybar) * norm(H_d))
            
            # 方法 B: IS divergence
            score = -IS_divergence(Ybar, H_d)
            
            # 方法 C: ??? 你有更好的想法嗎？
```

**你需要回答**：

1. ✅ **方法 A（相關性）合理嗎？**
   - 問題：Ybar 和 H_d 的物理意義不同
   - Ybar = 混響後的頻譜
   - H_d = 轉移函數（impulse response 的頻域）
   - 直接算相關性有意義嗎？

2. ✅ **方法 B（IS divergence）更合理嗎？**
   - 但 IS 需要兩個頻譜，H_d 本身不是完整頻譜
   - 應該是 IS(Ybar, H_d ⊙ S_hat) 嗎？
   - 那 S_hat 從哪來？

3. ✅ **是否需要歸一化？**
   ```python
   H_d_normalized = H_d / H_d.sum()  # L1 歸一化？
   # 或
   H_d_normalized = (H_d - H_d.mean()) / H_d.std()  # 標準化？
   ```

4. ✅ **時間維度如何處理？**
   ```python
   # 選項 1: 全時間平均
   Ybar = Y.mean(axis=1)
   
   # 選項 2: 滑動窗口
   for window in sliding_windows(Y):
       Ybar_window = window.mean(axis=1)
       score = compute_score(Ybar_window, H_d)
   
   # 選項 3: 加權平均（能量高的時刻權重大）
   weights = Y.sum(axis=0)  # 每個時刻的總能量
   Ybar = np.average(Y, axis=1, weights=weights)
   ```

**這些選擇會顯著影響效果，需要你的物理直覺！**

---

### 🔴 Critical Question 2: NMF Atom Token 的選擇策略

**我的假設**：

```python
class NMFAtomTokenizer:
    def __call__(self, Y):
        Ybar = Y.mean(axis=1)
        z = estimate_z_is(Ybar, self.W)  # (K,) 激活向量
        
        # 選 top-k 個最強的 atoms
        top_indices = np.argsort(-z)[:self.top_k]
```

**你需要回答**：

1. ✅ **top_k 應該是多少？**
   - k=5: 只保留最主要的成分
   - k=10: 更多細節
   - k=20: 會不會太冗餘？
   
   這取決於：
   - 你的 W 矩陣有多少個 atoms（K=64? 128?）
   - 聲音的複雜度（人聲 vs 音樂 vs 噪音）

2. ✅ **z 的量化方式？**
   ```python
   # 方法 A: 線性量化
   level = int(z[idx] * scale_factor)
   
   # 方法 B: 對數量化（小值也能區分）
   level = int(np.log(z[idx] + eps) * scale_factor)
   
   # 方法 C: 相對量化（相對於最大值）
   level = int((z[idx] / z.max()) * (n_levels - 1))
   ```
   
   哪種更能反映物理意義？

3. ✅ **是否需要 Atom 的語義標註？**
   
   你的 W 矩陣的 atoms 有沒有明確的物理意義？比如：
   ```python
   Atom 0: 低頻基底（50-200 Hz）
   Atom 5: 人聲共振峰（800-1200 Hz）
   Atom 12: 高頻噪音（3000+ Hz）
   ```
   
   如果有，我們可以設計：
   ```python
   <AT_low_freq_0:12>   # 語義化的 token
   <AT_formant_5:8>
   <AT_high_freq_12:3>
   ```
   
   這樣更有可解釋性！

---

### 🔴 Critical Question 3: Multi-Modal Token 的排列順序

**這是最關鍵的設計決策！**

我提出了三種策略：

#### 策略 A: Physics-First（物理優先）

```python
[<CLS>] + Direction tokens + Atom tokens + Patch tokens
```

**理由**：
- Direction tokens 提供**全域先驗**（「大概在 90° 附近」）
- Atom tokens 提供**結構資訊**（「是人聲，不是音樂」）
- Patch tokens 提供**局部細節**（「某個頻率特別強」）

**類比**：人類推理順序
1. 先感知方向（視覺+聽覺）
2. 再分析聲音類型（頻譜結構）
3. 最後確認細節

#### 策略 B: Mixed（交錯）

```python
[<CLS>] + [Direction, Atom, Patch] 輪流出現
```

**理由**：
- 讓模型自己學習哪種資訊更重要
- Attention 可以跨模態融合

#### 策略 C: Hierarchical（層次）

```python
[<CLS>] 
+ Coarse-level Direction (每 10° 一個)
+ Fine-level Direction (每 5° 一個)
+ Coarse-level Atoms (top-5)
+ Fine-level Atoms (top-10)
+ Patches
```

**理由**：
- 由粗到細，符合多尺度感知
- 類似 Vision Transformer 的 patch hierarchy

**你需要決定**：

1. ✅ 哪種順序符合**聲學物理的推理邏輯**？
2. ✅ 是否需要**特殊分隔符**？
   ```python
   [<CLS>] <DIR_START> ... <DIR_END> <ATOM_START> ... <ATOM_END> <PATCH_START> ... <PATCH_END>
   ```
3. ✅ 不同模態的 token 數量比例？
   ```python
   # 選項 1: 均衡
   5 Direction + 8 Atoms + 126 Patches
   
   # 選項 2: 物理主導
   10 Direction + 12 Atoms + 50 Patches (減少 patch 密度)
   
   # 選項 3: 細節主導
   3 Direction + 5 Atoms + 126 Patches
   ```

---

### 🔴 Critical Question 4: ICL Context 的採樣策略

**問題**：在 ICL 模式下，如何選擇「好的」context examples？

#### 選項 A: Random Sampling（隨機）

```python
context = random.sample(all_samples, k=3)
```

**優點**：簡單
**缺點**：可能選到不相關的例子

#### 選項 B: Angle-Based Sampling（角度相關）

```python
query_angle = 90
# 選擇鄰近角度
context = sample_from_angles([85, 90, 95], k=1 each)
```

**優點**：Context 和 query 相關
**缺點**：太相似可能不夠多樣

#### 選項 C: Diversity Sampling（多樣性）

```python
# 確保 context 涵蓋不同角度
context = [
    sample_from_angle(30),   # 左側
    sample_from_angle(90),   # 中間
    sample_from_angle(150),  # 右側
]
```

**優點**：模型看到全域分佈
**缺點**：可能太分散

#### 選項 D: Hard Example Mining（困難樣本）

```python
# 選擇「模型容易混淆」的例子
context = [
    sample_from_angle(85),  # 和 90 很像
    sample_from_angle(90),  # 目標角度
    sample_from_angle(95),  # 和 90 很像
]
```

**你需要決定**：

1. ✅ 哪種策略符合**實際應用場景**？
   - 如果是環境適應：可能需要 diversity
   - 如果是少樣本學習：可能需要 angle-based

2. ✅ Context 數量多少合適？
   - 1-shot: 只給 1 個例子
   - 3-shot: 3 個例子（常見）
   - 10-shot: 10 個例子（可能太多）

---

### 🔴 Critical Question 5: 評估指標

**最後但最重要：如何評估多模態系統是否更好？**

#### 指標 A: Top-1 Accuracy（現有）

```python
accuracy = (predicted_angle == true_angle).mean()
```

**問題**：對於 ICL，這個指標不夠！

#### 指標 B: Cross-Domain Transfer

```python
# 訓練：室內數據
# 測試：戶外數據（zero-shot）
transfer_accuracy = test_on_outdoor(model_trained_on_indoor)

# ICL：給幾個戶外例子
icl_accuracy = test_on_outdoor_with_context(model, few_outdoor_examples)

# 期待：icl_accuracy >> transfer_accuracy
```

#### 指標 C: Few-Shot Learning Curve

```python
for n_shots in [1, 2, 3, 5, 10]:
    accuracy = evaluate_with_n_shot_context(model, n_shots)
    plot(n_shots, accuracy)

# 期待：隨 n_shots 增加，accuracy 快速提升
```

#### 指標 D: Token Importance Analysis

```python
# 可視化：哪種 token 最重要？
attention_weights = model.get_attention_weights()

direction_importance = attention_weights[direction_tokens].mean()
atom_importance = attention_weights[atom_tokens].mean()
patch_importance = attention_weights[patch_tokens].mean()

# 期待：direction_importance > atom_importance > patch_importance
# （驗證我們的「物理優先」假設）
```

**你需要決定**：

1. ✅ 主要評估指標是什麼？
2. ✅ 多模態的「成功」標準是什麼？
   - 準確度提升 5%？10%？
   - ICL 能力明顯優於 fine-tuning？
   - 可解釋性改善（定性 or 定量）？

---

<a name="part3"></a>
## Part 3: 我的設計假設 vs 你需要驗證的點

### 📋 Summary Table

| 設計模塊 | 我的貢獻（軟體架構） | 你的貢獻（Domain Knowledge） | 優先級 |
|---------|-------------------|---------------------------|--------|
| **NMFAtomTokenizer** | ✅ 程式碼框架、z 估計流程 | 🔴 top_k 選擇、量化方式、atoms 語義 | P0 |
| **DirectionProjectionTokenizer** | ✅ Token 生成邏輯 | 🔴 相關性計算公式、時間處理、歸一化 | P0 |
| **MultiModalPromptBuilder** | ✅ Token 組合框架 | 🔴 排列順序策略、模態比例 | P0 |
| **ICL Context Sampling** | ✅ 採樣介面 | 🔴 採樣策略、context 數量 | P1 |
| **Evaluation Metrics** | ✅ 指標計算程式 | 🔴 成功標準定義、實驗設計 | P1 |
| **HF Tokenizer Vocab** | ✅ Vocab 構建邏輯 | ⚪ 審查 token 命名合理性 | P2 |
| **Dataset Pipeline** | ✅ DoAICLDataset 類別 | ⚪ 驗證數據載入正確性 | P2 |

---

### 🎯 我的設計決策（可以直接實作）

這些部分我有足夠的通用 ML 知識可以決定：

1. ✅ **軟體架構**
   - Tokenizer 的抽象類別設計
   - PromptBuilder 的介面設計
   - Dataset 的繼承結構

2. ✅ **Transformer 整合**
   - HF tokenizer vocab 擴展
   - Attention 可視化工具
   - ICL prompt 格式（[CLS] ... [SEP] ...）

3. ✅ **訓練流程**
   - `--use-multi-modal` flag
   - `--icl-mode` flag
   - Dataloader 修改

4. ✅ **程式碼實作**
   - 所有 Python 類別和函式
   - 單元測試
   - 文檔

---

### 🔴 你的決策（必須在實作前確定）

這些問題我**無法**替你回答，因為需要**聲學物理的 domain knowledge**：

#### Phase 0: 理論驗證（現在就要決定）

1. **Direction token 計算公式**
   ```python
   # 你需要告訴我：
   score = ???  # 相關性？IS divergence？其他？
   ```

2. **Token 排列順序**
   ```python
   # 選擇一個：
   ordering = "physics_first"  # or "mixed" or "hierarchical"
   ```

3. **主要評估指標**
   ```python
   # 定義成功標準：
   success_criteria = "accuracy improvement > X%" 
   # or "ICL zero-shot works"
   ```

#### Phase 1: 參數調優（實作後實驗決定）

4. **Atom token 的 top_k**
5. **Direction token 的 top_m**
6. **ICL context 數量**
7. **模態比例** (Direction:Atom:Patch)

#### Phase 2: 進階設計（可選，看初步效果）

8. **Atom 語義標註**（如果 W 矩陣有物理意義）
9. **多尺度 Direction tokens**（coarse + fine）
10. **動態 token 選擇**（依據信噪比調整）

---

## 🎬 下一步行動建議

### 建議流程：

#### Week 1: 理論驗證 + 最小可行原型（MVP）

**Day 1-2（你的工作）**：
```
[ ] 決定 Direction token 計算公式
    - 寫出數學公式
    - 用 10 個樣本手動驗證合理性
    
[ ] 決定 Token 排列順序
    - 畫出 prompt 範例
    - 確認符合物理直覺

[ ] 定義評估指標
    - Top-1 accuracy
    - 至少一個 ICL 相關指標
```

**Day 3-5（我的工作 / 你執行）**：
```
[ ] 實作 NMFAtomTokenizer（50 行）
[ ] 實作 DirectionProjectionTokenizer（50 行）
[ ] 實作 MultiModalPromptBuilder（100 行）
[ ] 單元測試（驗證 token 生成正確）
```

**Day 6-7（一起）**：
```
[ ] Smoke test：10 個樣本
[ ] 檢查生成的 prompts 是否合理
[ ] 可視化 tokens（確認物理意義）
```

#### Week 2: 整合訓練 + 初步實驗

**Day 8-10**：
```
[ ] 整合到 train_reward_model_lora.py
[ ] 跑 smoke test（--use-multi-modal）
[ ] 對比 baseline（只用 patches）
```

**Day 11-14**：
```
[ ] 完整實驗（100-1000 samples）
[ ] 分析結果
[ ] 決定是否繼續 ICL 部分
```

---

## 💡 我的建議

基於我對你系統的理解，我建議：

### 🥇 Priority 0（必須做）

1. **Direction Token 用相關性**
   ```python
   # 理由：簡單、可解釋、計算快
   score = cosine_similarity(Ybar, H_d)
   ```

2. **Token 順序用 Physics-First**
   ```python
   # 理由：符合人類認知，便於解釋
   [<CLS>] + Direction (top-5) + Atom (top-8) + Patches
   ```

3. **先做 Multi-Modal，暫緩 ICL**
   ```python
   # 理由：ICL 需要 multi-modal 作為基礎
   # 先驗證 multi-modal 有效，再考慮 ICL
   ```

### 🥈 Priority 1（有時間再做）

4. **ICL 用 Angle-Based Sampling**
5. **加入 Cross-Domain 評估**

### 🥉 Priority 2（研究向）

6. **Atom 語義標註**（如果你的 W 有明確意義）
7. **Attention 可視化**（做 figure）

---

## ❓ 現在請你回答

請針對以下**三個最關鍵**的問題給我答案（其他可以之後再討論）：

### Q1: Direction Token 計算公式

```python
# 選項 A
score = np.dot(Ybar, H_d) / (norm(Ybar) * norm(H_d))

# 選項 B  
score = -IS_divergence(Ybar, H_d ⊙ S_hat)  # S_hat 從哪來？

# 選項 C（你提供）
score = ???
```

**你的選擇**：________

**理由（從物理意義解釋）**：________

---

### Q2: Token 排列順序

```python
# 選項 A: Physics-First
[<CLS>] <R_090:14> <R_095:12> ... <AT_5:12> <AT_10:8> ... <P_0_0_5> ...

# 選項 B: Mixed
[<CLS>] <R_090:14> <AT_5:12> <P_0_0_5> <R_095:12> <AT_10:8> ...

# 選項 C: Hierarchical  
[<CLS>] <R_090:14> <R_095:12> [MID] <AT_5:12> <AT_10:8> [DETAIL] <P_0_0_5> ...
```

**你的選擇**：________

**理由（為什麼這個順序反映聲學推理）**：________

---

### Q3: 評估指標與成功標準

```python
# 目標 A: 準確度提升
"Multi-modal accuracy - Baseline accuracy > 5%"

# 目標 B: ICL 能力
"ICL few-shot accuracy > Zero-shot accuracy + 10%"

# 目標 C: 可解釋性
"Direction token attention weight > 0.3 (定性驗證)"
```

**你的優先級**（1 > 2 > 3）：________

**原因**：________

---

請你仔細思考後回答這三個問題，然後我們就可以開始實作了！🚀
