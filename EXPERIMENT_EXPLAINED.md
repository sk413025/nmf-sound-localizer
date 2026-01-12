# Experiment: Frequency-Aware Decision Transformer (`exp-freq-aware-policy`)

## 1. 核心動機 (Motivation)

你當初創建這個 Experiment (`exp-freq-aware-policy`) 的核心目的是為了解決 **「相位模糊 (Phase Ambiguity)」** 問題。

### 背景
在聲源定位與優化中，我們發現單純看 Correlation (相關性) 的形狀是不夠的。
- 物理公式: $\Delta\phi = 2\pi f \Delta\tau$
- 這意味著：同樣的 **時間延遲 ($\Delta\tau$)**，在 **不同頻率 ($f$)** 下會產生完全不同的 **相位差 ($\Delta\phi$)**。
- 如果模型不知道現在是在處理哪個頻率 (Frequency Bin)，它就很難學會正確的 Lag 選擇策略，因為 50Hz 的最佳特徵跟 500Hz 的最佳特徵長得不一樣。

### 假設 (Hypothesis)
**"Frequency Awareness is Key"**
如果我們在 Decision Transformer (DT) 的輸入中加入 **頻率嵌入 (Frequency Embedding)**，模型就能根據當下的頻率調整它的策略。
- **輸入**: Correlation History + RTG (目標) + **Freq ID (頻率索引)**
- **預期效果**: 一個單一的大模型可以同時學會所有頻率的優化策略，而不是像之前那樣需要針對單一頻率訓練。

---

## 2. 全量數據生成 (Full Scale Data Generation)

這是腳本中的 `[Step 1]`。

### 這是什麼？
這是建立 **「標準答案庫 (Teacher Knowledge)」** 的過程。我們使用物理引擎 (OMP, Orthogonal Matching Pursuit) 對整個資料集進行暴力搜索，找出完美的 Lag 選擇。

- **數據量**: Speech-260 (全部 260 個真實語音片段，而非原本用來測試的 2 個片段)。
- **範圍**: 涵蓋所有頻率 (5Hz - 300Hz) 和所有發音特性。
- **輸出**: `lag_trajectories.pt`。這裡面存的是：「在這個 Correlation 狀態下，OMP 選擇了哪個 Lag (Action)，並獲得了多少能量下降 (Reduction)」。

### 為什麼要全量？
在你之前的 Smoke Test (2 epoch, 2 clips) 中，我們發現模型雖然學會了第一步 (Step 0)，但在後續步驟 (Step 1, 2) 會卡住 (Stall)。這是因為訓練資料太少，導致模型對於複雜的殘差動態 (Residual Dynamics) 擬合不足。全量數據是為了讓模型見過足夠多的「各種情況」，才能學會連續優化。

---

## 3. 完整訓練 (Full Training)

這是腳本中的 `[Step 2]`。

### 這是什麼？
這是訓練 **「學生模型 (Student Policy)」** 的過程。

- **架構**: `SeqDT_FreqAware` (時序決策 Transformer + 頻率感知)。
- **輸入**:
    1. **Correlations**: 過去的觀測值。
    2. **RTG**: Returns-to-Go (期望這一步能降低多少能量，比如 "我要降低 3dB")。
    3. **Freq Embed**: "我現在是 Bin 50" 或 "我是 Bin 200" (這是此 Exp 的關鍵新增)。
- **目標**: 預測出跟 OMP 一樣的 Lag Action。
- **設定**:
    - **Epochs**: 30 (Smoke Test 只跑了 2，導致欠擬合)。
    - **Freq Range**: 5-300 (低頻區，也是之前發現最難搞的區域)。

### 成功的定義 (Metrics)
1. **Accuracy (模仿率)**: 模型選的 Lag Index 跟 OMP 選的是否一樣？
2. **Reduction (能量消除率)**: 這是物理指標。我們剛修好的 `inspect_flow.py` 就是用來測這個。
    - 如果模型預測準確，Reduction 應該要跟 OMP 一樣 (e.g., 66.61%)。
    - 如果模型亂選，Reduction 就會很低甚至為負。

---

## 總結

**`exp-freq-aware-policy` 是一個從「單頻率實驗」走向「全頻率通用模型」的關鍵轉折點。**

- **過去**: 我們只敢測 Bin 50，或手動調整每個頻率。
- **現在 (正在跑的)**: 我們試圖訓練一個統一的大腦，給它頻率 ID，讓它自己去適應 5-300Hz 之間的所有物理變化。如果這個實驗成功，我們就證明了 **Frequency Conditioned Decision Transformer** 是可行的通用解法。
