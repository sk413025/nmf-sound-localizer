# InterSpeech 2026 論文詳細規劃與投稿分析

> **文檔創建日期**: 2026-01-13
> **最後修改日期**: 2026-01-15
> **目標會議**: InterSpeech 2026
> **截稿日期**: 2026-02-25
> **可用時間**: 6週（43天）
> **論文格式**: Regular Paper (4頁正文 + 2頁參考文獻)

---

## ⚠️ 方法論修正說明 (2026-01-15)

**本文件部分內容已過時。核心修正如下：**

| 舊敘事 | 正確敘事 |
|--------|----------|
| Stage 1 用白噪音學習 H | H 已從白噪音預先計算好 |
| Stage 2 遷移到語音任務 | Stage 2 是方向估計 fine-tuning |
| 準確率 32%→82% | Stage 1: 97.11% Energy Reduction |

**已更新的章節**：
- §1.3 核心故事 ✅
- §2.3 Abstract ✅
- §2.4 段落3-4 (Method概述) ✅
- §4.4.1 白噪音分析 (標記為過時) ✅

**未更新的章節** (需參考 `two_stage_notes.md` 獲取正確資訊)：
- §3 Method 詳細章節
- §4 Experiments 數字

---

---

## 目錄

1. [核心策略調整：從LDV應用到通用方法](#1-核心策略調整)
2. [論文詳細鋪陳規劃](#2-論文詳細鋪陳規劃)
3. [投稿成功率深度分析](#3-投稿成功率深度分析)
4. [關鍵實驗規劃](#4-關鍵實驗規劃)
5. [時間線與風險管理](#5-時間線與風險管理)
6. [決策建議](#6-決策建議)

---

## 1. 核心策略調整

### 1.1 問題識別

**原始錯誤定位（LDV-centric）**:
- 標題包含 "LDV-Based"
- 將LDV作為主要貢獻
- 論文讀起來像"為LDV設計的專用方法"
- 受眾：LDV用戶（極小眾）

**問題**：InterSpeech是以Microphone為主的會議，LDV硬體特定性會導致審稿人第一印象排斥。

---

### 1.2 核心策略轉變

| 維度 | ❌ 錯誤定位 | ✅ 正確定位 | 改善 |
|------|------------|------------|------|
| **主角** | LDV硬體 | Cross-sensor adaptation方法 | +++ |
| **問題** | LDV speech localization難 | 跨傳感器模態的聲學任務（通用問題） | +++ |
| **貢獻** | 為LDV設計的方法 | 通用方法，在challenging case驗證 | +++ |
| **受眾** | LDV用戶（小眾） | Speech community（廣泛） | +++ |
| **標題** | "LDV-Based..." | "Cross-Sensor Transfer Learning..." | +++ |
| **InterSpeech相關性** | 50% | **85%** | +35% |
| **方法通用性** | 60% | **90%** | +30% |
| **錄取機率** | 35-40% | **50-60%**（條件機率） | +15-20% |

---

### 1.3 新的敘事框架

#### **核心故事（3句話版本）**
```
1. In cross-sensor settings, the observation mixes signal content with sensor/environment transfer.
   Direct end-to-end DOA training can overfit to content shortcuts and fail to generalize.

2. We propose a general two-stage framework:
   Stage 0: measure H_cal(f,θ) from white-noise recordings (calibration).
   Stage 1: at a single fixed angle θ₀, learn Mic→LDV transformation features from speech (no angle labels).
   Stage 2: use Stage 1 features + the pre-computed H_cal(f,θ) library for direction classification.

3. In a Mic→LDV case study, Stage 1 reaches near-oracle reconstruction energy reduction on a
   representative validation window (97.11% vs 97.88% oracle; reproducible via the validation scripts).
   Stage 2 direction accuracy is pending.
```

> **重要修正**：H是預先從白噪音錄製計算好的頻率響應函數，
> 不是Stage 1學習的目標。Stage 1學習的是語音轉換特徵。

#### **關鍵賣點排序**
1. 🥇 **通用框架** - Cross-sensor transfer learning（適用任意sensor pair）
2. 🥈 **方法創新** - 兩階段解耦系統與內容
3. 🥉 **技術突破** - Frequency-Aware Policy解決相位混疊
4. 🏅 **實證驗證** - 在extreme case (Mic-to-LDV)上證明有效性

---

## 2. 論文詳細鋪陳規劃

### 2.1 論文結構概覽

**4頁分配**：
```
Page 1: Introduction (0.7頁) + Related Work (0.3頁)
Page 2: Method (1.3頁)
Page 3: Experiments (1.5頁)
Page 4: Discussion & Conclusion (0.3頁) + References開始
Page 5-6: References續
```

**字數預算**: ~3200-3500 words (4頁雙欄)

---

### 2.2 標題選項（推薦優先級）

#### **選項1（強烈推薦）**
```
"Frequency-Aware Transfer Learning for Cross-Sensor
Speech Source Localization"
```
- ✅ 無LDV，審稿人不會第一眼排斥
- ✅ "Cross-Sensor"是通用術語
- ✅ 適用於Mic-to-Mic, Mic-to-LDV, Mic-to-Accelerometer等

#### **選項2**
```
"Learning Acoustic Sensor Transformations via Two-Stage
Transfer Learning for Speech Localization"
```
- ✅ 強調"sensor transformation"通用概念
- ⚠️ 稍長

#### **選項3**
```
"Disentangling System and Content for Cross-Modality
Acoustic Source Localization"
```
- ✅ 強調核心技術貢獻
- ⚠️ 可能太抽象

---

### 2.3 Abstract（150 words）

```
Cross-sensor acoustic learning is challenging due to frequency-
dependent system transformations. We propose a two-stage learning
framework: Stage 1 learns Mic-to-LDV speech transformation
features at a single angle, while Stage 2 combines these features
with a pre-computed calibration library H_cal(f,θ) for direction
estimation. A key innovation is Frequency-Aware Policy, which
resolves phase-wrapped lag ambiguity by explicitly conditioning
the policy on the frequency bin. We validate on microphone-to-LDV
speech localization: Stage 1 reaches 97.11% reconstruction energy
reduction on a representative validation window, close to the
oracle value (97.88%) under the same protocol. Stage 2 direction
accuracy will evaluate whether the learned transformation features
transfer to downstream DOA classification. The proposed framework
separates angle-agnostic feature learning from angle-aware
direction estimation, enabling efficient adaptation to cross-sensor
scenarios.
```

> **Abstract 已更新**：移除「白噪音訓練」敘述，改為正確的兩階段描述。

**Keywords**:
```
cross-sensor learning, transfer learning, speech source localization,
frequency-aware neural networks, sensor adaptation
```

---

### 2.4 §1 Introduction (0.7頁，~550 words)

#### **段落1 (150 words): 通用問題設定**
```
在實際應用中，聲學傳感器的多樣性是常態：

• 不同傳感器模態：
  - Microphones (pressure)
  - Laser Doppler Vibrometry (vibration)
  - Accelerometers (acceleration)
  - Bone conduction sensors (solid vibration)

• 不同聲學環境：
  - 不同房間的混響特性
  - 不同材質的傳播介質

• 核心挑戰：
  在一個傳感器/環境訓練的模型，
  如何遷移到另一個傳感器/環境？

這是speech community面臨的普遍問題：
- Hearing aids: 骨導傳感器 ↔ 空氣麥克風
- Smart home: 嵌入式Mic ↔ 設備Mic
- Surveillance: 遠場LDV ↔ 近場Mic

關鍵：如何學習sensor-agnostic的聲學表示？
```

#### **段落2 (120 words): 引入LDV作為extreme case**
```
我們以Laser Doppler Vibrometry (LDV)為研究案例：

LDV優勢：
✓ 非接觸測量（隱私保護）
✓ 遠距離捕獲（10-100m）
✓ 無需部署傳感器（靈活）

LDV挑戰（代表cross-sensor的極端難度）：
✗ 測量表面振動 (vs Mic的空氣聲壓)
✗ 頻率響應差異大（共振/衰減）
✗ 相位關係複雜（傳播路徑不同）

我們的初步實驗揭示了嚴重性能差距：
┌────────────┬──────────┐
│ White Noise│   100%   │  ← 信號簡單時可以學
│ Speech     │   32%    │  ← 信號複雜時崩潰
└────────────┴──────────┘

問題本質：系統差異H與信號內容耦合
```

#### **段落3 (180 words): 通用方法論**
```
我們提出一個sensor-agnostic的框架：

核心思想：分離角度無關特徵與角度相關資訊
  H_cal(f,θ): pre-computed from white noise (37 angles)
  Features: 從語音學習的轉換特徵（角度無關）

兩階段學習策略：
  Stage 1: 固定單一角度θ₀，學習Mic→LDV語音轉換
    → 學習頻率相依的轉換特徵
    → 不使用角度資訊（angle-agnostic）

  Stage 2: 引入H_cal(f,θ)進行方向估計
    → Stage 1特徵 + 37角度H_cal → 方向分類
    → Fine-tune或凍結Stage 1 encoder

關鍵技術：Frequency-Aware Policy
  動機：延遲造成的相位項隨頻率縮放（φ(f) = -2π·f·τ mod 2π）
  實現：Frequency embedding解決跨頻率的策略衝突
  效果：使單一策略可在多頻段穩定選擇lag（具體增益以消融實驗呈現）

驗證：Stage 1達到97.11% Energy Reduction（接近Oracle 97.88%）
```

#### **段落4 (100 words): 貢獻總結**
```
本文貢獻（按重要性排序）：

1. 技術創新：Frequency-Aware Policy解決相位混疊
   以頻率條件化處理相位/延遲的wrap-around；增益以消融實驗量化

2. 兩階段框架：分離角度無關特徵學習與角度相關方向估計
   Stage 1學習轉換特徵，Stage 2驗證下游任務

3. 理論貢獻：接近OMP Oracle上界
   97.11% vs 97.88%（差距僅0.77%）

4. 實證驗證：在challenging case (Mic-to-LDV)上驗證
   證明學到的特徵對方向估計有用
```

---

### 2.5 §2 Related Work (0.3頁，~240 words)

#### **2.1 Cross-Domain Acoustic Learning (100 words)**
```
跨域聲學學習的既有工作：

Room Adaptation:
- [X] Domain adversarial training for reverb
- [Y] Few-shot adaptation to new rooms

Sensor Adaptation (少量工作):
- [Z] Bone conduction → Air conduction (hearing aids)
- [W] Contact microphone → Free-field microphone

Gap: 缺少systematic framework for arbitrary sensor pairs
→ 本文填補這個空白

Our positioning:
從specific application (LDV)出發，
提出general framework (two-stage transfer)
```

#### **2.2 Sound Source Localization (80 words)**
```
經典方法：SRP-PHAT [3], MUSIC [4], GCC-PHAT [5]
  → 基於time/phase delay估計
  → 依賴麥克風陣列幾何

神經方法：DOANet [6], SELDnet [7], Conformer-based [8]
  → End-to-end學習
  → 需要大量標註數據

Limitation: 都假設標準麥克風輸入，不適用於LDV振動信號
→ 本文擴展到cross-sensor scenarios
```

#### **2.3 Phase-Aware Representations (60 words)**
```
頻率依賴建模：
- Multi-resolution STFT [12]
- Learnable frequency filters [13]
- Phase unwrapping networks [14]

我們的創新：
Frequency Embedding for phase disambiguation
→ 顯式建模相位-延遲關係 φ(f) = -2π·f·τ (mod 2π)
→ 解決lag-phase混疊問題
```

---

### 2.6 §3 Method (1.3頁，~1000 words)

#### **3.1 Problem Formulation (0.3頁，~230 words)**

**General Setup（任意傳感器對）**:
```
Given:
- Source sensor S provides x(t) (e.g., microphone pressure)
- Target sensor T provides y_θ(t) (e.g., LDV surface velocity)
- Task label: source direction θ ∈ [0°, 180°]

Assumption (per direction θ): the cross-sensor mapping is approximately linear and time-invariant (LTI) within a clip:
  y_θ(t) = (h_θ * x)(t) + ε(t)

STFT notation:
  X(f,n), Y_θ(f,n) are complex STFTs (frequency bin f, frame index n).

Narrowband approximation (instantaneous TF):
  Y_θ(f,n) ≈ H_θ(f) · X(f,n)

More general STFT-frame convolution (to capture longer effective responses):
  Y_θ(f,n) ≈ Σ_{ℓ=0}^{L-1} H_θ(f,ℓ) · X(f,n-ℓ)

Key point:
- H_θ captures sensor+environment transfer characteristics and can vary with θ.
- θ is a geometric quantity shared across sensors, while speech content is not.

Why direct end-to-end DOA can fail in cross-sensor settings:
- The model may overfit to content shortcuts (speaker/formants) instead of learning stable transfer/geometry cues.
- This hurts cross-signal generalization (speech → music/noise) and cross-sensor transfer.
```

**Instantiation: Mic-to-LDV（本文實驗場景）**:
```
We use Mic→LDV as an extreme test case:
- Mic measures air pressure (Pa)
- LDV measures surface vibration velocity (m/s)
- The mapping includes acoustic propagation, structure coupling, and sensor responses.

Lag dictionary used in Stage 1 (discrete STFT-frame shifts):
For each frequency bin f, we approximate the LDV STFT window as a sparse combination of delayed Mic windows:
  Y(f, n:n+T_w-1) ≈ Σ_{ℓ∈K} a(f,ℓ) · X(f, n-ℓ : n-ℓ+T_w-1)

where:
- ℓ ∈ {0,…,15} are discrete STFT-frame lags
- τ_ℓ = ℓ · hop / fs, with hop=512 samples and fs=16 kHz (1 lag = 32 ms; max lag ≈ 480 ms)
- a(f,ℓ) ∈ ℂ are frequency-dependent coefficients
```

**圖示（小型diagram）**:
```
┌─────┐   Stage 1 (θ₀ only)        ┌───────────┐
│ Mic │───────────────┬──────────→│  Features │
└─────┘               │           └───────────┘
                      │                 │
                      │                 │  + pre-computed H_cal(f,θ)
                      │                 ↓
                      │           ┌───────────┐
                      └──────────→│ Direction │──→ θ
                                  │   Head    │
                                  └───────────┘
```

#### **3.2 Stage 1: System Transfer Learning (0.5頁，~400 words)**

**3.2.1 Teacher: Physics-Based OMP (150 words)**
```
Lag-OMP算法（迭代選擇最優延遲）：

初始化：Residual R = Y, Active_Set = ∅

For k = 1 to 4:
  1. 計算相關性：
     C_lag(ℓ) = |⟨Dict_ℓ, R⟩| / ||Dict_ℓ||
     其中Dict_ℓ = X(f, t-ℓ)

  2. 選擇最優lag：
     ℓ* = argmax C_lag(ℓ)

  3. 更新權重（Least Squares）：
     h = (A^H A)^{-1} A^H Y
     其中A = [Dict_ℓ for ℓ ∈ Active_Set]

  4. 更新殘差：
     R = Y - A·h

輸出軌跡：{State_k, Action_k, Reduction_k}
→ 用於Student distillation
```

**Table 1: Lag-OMP teacher statistics (reduction score)**
```
┌───────────────────┬─────────┐
│ Metric            │  Value  │
├───────────────────┼─────────┤
│ Mean reduction    │ 73.57%  │
│ Median            │ 75.39%  │
│ 90th Percentile   │  >93%   │
└───────────────────┴─────────┘
```

Note: this “reduction score” is the per-bin metric used in trajectory generation
(`(‖y‖ - ‖r‖)/‖y‖`). It is different from the “energy reduction” metric used in Table 2.

**3.2.1.5 從物理過程到 Algorithm Distillation：方法論的自然演化 (250 words)**

**A. 物理傳播模型與稀疏性**
```
聲學物理過程：
  聲源 → [空氣傳播] → 麥克風測量 p(f,t)
               ↓
          [牆面振動] → LDV測量 v(f,t)

數學建模：
  Time domain (per direction θ, within a clip):
    v_θ(t) = (h_θ * p)(t) + ε(t)

  STFT (narrowband approximation):
    V_θ(f,n) ≈ H_θ(f) · P(f,n)

  where H_θ(f) is the complex transfer function (magnitude + phase) determined by the sensor pair, geometry, and environment.

物理稀疏性：
  聲學路徑數量有限（直接路徑 + 少數反射）
  → H的時域表示：H(t) = Σ_{k=1}^K h_k δ(t - τ_k)
  → In many setups, a small K can capture the dominant early contributions (direct path + a few strong reflections),
    but the effective K depends on the environment and the analysis window.

這直接導出稀疏建模的物理必然性
```

**B. Why the pursuit is sequential (algorithmic dependency)**
```
OMP (and related pursuit methods) produce a sequence of decisions because the residual changes after each selected atom.

Step 1:
  Compute correlations g = D^H y and select the lag/atom with the largest |g|.

Step 2..K:
  Update the residual r_k = y - D_A h_A (using the current active set A),
  then recompute correlations D^H r_k and select the next atom.

Key dependency:
  The best action at step k depends on the previously selected atoms because the residual depends on them.
  With a highly coherent lag dictionary (e.g., μ≈0.998), this dependency can be strong,
  so one-shot predictors or naive greedy policies can be brittle.

Why we use a sequence model:
  We distill teacher trajectories that already have a sequential (state → action → updated residual) structure.
  A Decision-Transformer-style model is one convenient way to learn this mapping over multiple steps.
```

**C. Algorithm Distillation 的物理意義：目標導向的逆問題求解**
```
Behavioral Cloning（模仿動作）vs Algorithm Distillation（提取物理邏輯）：

BC問題：
  - 只模仿OMP的動作序列 (τ₁, τ₂, ..., τ_K)
  - 忽略物理目標：為什麼選這些路徑？
  - 累積誤差：第1步偏離 → 殘差污染 → 後續步驟崩潰

AD的物理視角：
  OMP不是隨機選路徑，而是朝著物理目標函數優化：

  目標1: 殘差能量最小化 → min ||r_K||²
  目標2: 重建精度最大化 → max coherence(y, ŷ)

Return-to-Go (RTG) 的物理對應：
  RTG[0] = ||r||_target - ||r_k||   (還需減少多少殘差能量)
  RTG[1] = acc_target - acc_k       (還需提升多少重建精度)

  → 不是RL技巧，而是物理目標函數的顯式編碼

多重監督信號的物理根源：
  1. g = D^T r  → 物理投影（correlation in signal space）
  2. RTG        → 物理優化目標（能量 + 精度）
  3. Teacher KL → 繼承物理先驗（OMP的physics-based策略）

AD本質：學習如何執行目標導向的物理逆問題求解
```

**D. DTMin 架構 = 物理結構的神經編碼**
```
三大創新的物理根源：

1. Hierarchical Pointer (Expert → Atom)
   物理依據：空間定位優先於頻譜細節
   - 聲源方向θ 決定主要傳播路徑（幾何）
   - 頻率只影響路徑的頻譜響應（material property）
   - D = H ⊙ W：空間(37方向) × 頻譜(8 atoms) 的物理解耦

2. Dictionary-Aware Keys
   物理意義：Query·Key = residual^T · atoms
   - 對應OMP的correlation：物理內積 ⟨r, D_k⟩
   - 不是attention機制的技巧，而是物理投影的神經實現
   - 可解釋性：注意力權重 = 物理相關性分數

3. Frequency Embedding
   物理關係：φ(f) = -2π·f·τ (mod 2π) （相位-延遲-頻率耦合）
   - 同樣延遲τ，不同頻率f → 不同相位φ
   - 模型必須"知道"當前頻率f才能解碼τ
   - 這是物理定律，不是網絡設計選擇

小結：DTMin不是為DT設計的變體，而是將物理結構映射到神經架構
```

**表格1.5: 方法比較**
```
┌────────────────────┬────────────────────────────────────────────┐
│ Method             │ Notes                                      │
├────────────────────┼────────────────────────────────────────────┤
│ Greedy pursuit     │ One-step selection; can be brittle with a   │
│                    │ highly coherent lag dictionary              │
│ Behavioral cloning │ Imitate actions; error can accumulate over  │
│                    │ multiple selection steps                    │
│ DTMin (ours)       │ Sequence model with RTG + frequency context │
│                    │ (distills multi-step teacher trajectories)  │
└────────────────────┴────────────────────────────────────────────┘
```

**3.2.2 Student: Frequency-Aware DTMin 架構實現 (200 words)**

**完整架構設計**:
```python
class FreqAware_DTmin(nn.Module):
    def __init__(self, d_model=128):
        # 核心創新：頻率embedding
        self.freq_embed = nn.Embedding(F_bins, d_model)

        # 物理輸入：OMP相關性
        self.state_embed = nn.Linear(M_lags, d_model)

        # 目標信號：RTG (Return-to-Go)
        self.rtg_embed = nn.Linear(1, d_model)

        # Sequential decision
        self.gru = nn.GRU(d_model, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, M_lags)

    def forward(self, corr, rtg, freq_idx):
        # 三路融合
        emb = (self.state_embed(corr) +
               self.rtg_embed(rtg) +
               self.freq_embed(freq_idx))  # ← KEY!

        h = self.gru(emb)
        return self.policy_head(h)
```

**為什麼需要Frequency Embedding？**
```
Physical motivation:
For a pure delay τ, the transfer function contributes a phase term:
  H(f) = exp(-j·2π·f·τ)  ⇒  φ(f) = -2π·f·τ (mod 2π)

In our discrete lag dictionary, τ_ℓ = ℓ·hop/fs (ℓ is an integer STFT-frame lag).
Therefore, the same lag produces different *wrapped* phases at different frequencies.
If a policy is trained to select discrete lags from correlation features, it must be conditioned on the frequency bin.

Concrete example (hop=512, fs=16 kHz → 1 lag = 32 ms):
  τ_1 = 0.032 s
  φ(100 Hz)  = -2π·100·0.032  = -20.1 rad ≡ -1.26 rad (mod 2π)
  φ(1000 Hz) = -2π·1000·0.032 = -201  rad ≡ ~0   rad (mod 2π)

This “phase wrapping” effect is one reason frequency information is necessary for stable lag selection across bins.
```

#### **3.3 Stage 2: Task-Specific Fine-tuning (0.3頁，~230 words)**

**遷移策略**:
```
Stage 0 (Calibration): measure H_cal(f,θ) from white noise recordings at all θ

Stage 1 (Pretraining @ single θ₀, no angle labels):
  - Train Stage 1 on paired Mic/LDV speech at a fixed θ₀
  - Objective: learn angle-agnostic transformation features / lag-selection policy

Stage 2 (Direction fine-tuning across angles):
  Step 1: Attach a direction head (37-way, 0°–180°, 5° step)
  Step 2: Train with Stage 1 features + the pre-computed H_cal(f,θ) library
  Step 3: Ablate frozen vs fine-tuned encoder

Training details (initial plan):
  - Learning rate: 1e-4 (lower than Stage 1)
  - Epochs: ~10
  - Augmentation: time shift, pitch shift (optional; must not leak angle labels)
```

**關鍵假設驗證**:
```
兩階段有效的前提：
H是系統特性（sensor-specific），與信號內容無關

驗證方法（在實驗部分）：
1. 跨信號泛化：Music, Environmental sound
2. Ablation: 直接Speech訓練vs兩階段
3. H(f)的物理合理性分析
```

**偽代碼**:
```python
# Stage 0: calibration (white noise)
H_cal = estimate_transfer_function_library(white_noise_recordings)  # H_cal(f,θ)

# Stage 1: speech pretraining at a single θ₀ (no angle labels)
stage1 = FreqAware_DTmin()
stage1.train(paired_speech_at_theta0, teacher=lag_omp_teacher)

# Stage 2: direction training across θ using H_cal
model = DirectionModel(stage1_encoder=stage1.encoder, H_cal=H_cal)
model.train(multiangle_speech, labels=theta)
```

---

### 2.7 §4 Experiments (1.5頁，~1150 words)

#### **4.1 Experimental Setup (0.3頁，~230 words)**

**4.1.1 Test Case: Mic-to-LDV Datasets (120 words)**
```
我們在Mic-to-LDV場景上驗證方法：

Why this test case?
1. Extreme sensor difference (pressure vs velocity)
2. High practical relevance (non-contact speech capture)
3. No existing benchmark (我們貢獻新數據集)

自建LDV Benchmark:

White Noise Set:
  - Angles: 37 (0°-180°, 5°step)
  - Duration: 2s per clip
  - Samples: 111 clips
  - SNR: Clean recording (>30dB)

Speech Set (Speech260):
  - Same 37 angles
  - Source: TIMIT sentences (male+female)
  - Duration: 1-3s per clip
  - Samples: 260 clips (train: 200, val: 60)
  - SNR: ~15-20dB (LDV hardware limit)

錄音設置：
  - Mic: Rode NT1 (omni)
  - LDV: Polytec OFV-5000
  - Box: 30×30×30cm MDF
  - Distance: Mic-to-Box 2m, Speaker-to-Box 1.5m

Note: 方法不限於LDV
雖然實驗在Mic-to-LDV上，
但framework可直接應用於其他sensor pairs
```

**4.1.2 Baselines & Ablations (110 words)**
```
對比方法：

Baseline:
  • H_cal template matching (non-learned):
    for each θ, predict Ŷ_θ(f,n) = H_cal(f,θ)·X(f,n),
    then choose θ that minimizes ||Y - Ŷ_θ||² (summed over f,n).

  • H-only nearest neighbor (non-learned):
    estimate H_test(f) from the test clip (cross/auto spectra),
    then choose θ that maximizes similarity between H_test and H_cal(·,θ).

  • Note: array-based DOA baselines (SRP-PHAT/MUSIC) require multi-channel measurements
    (mic array or multi-point vibration). They are not directly applicable to a single Mic + single LDV point.

Ablations:
  • End-to-end direction head: train DOA directly from inputs without Stage 1 pretraining
  • No H_cal: direction head uses Stage 1 features only
  • w/o frequency embedding: Stage 1 pretraining without conditioning on frequency bin
  • Frozen vs fine-tuned Stage 1 encoder in Stage 2

Training:
  - Optimizer: AdamW (lr=5e-4 Stage1, 1e-4 Stage2)
  - Batch size: 256 / 64
  - Hardware: Mac Studio (MPS GPU)
```

#### **4.2 Main Results (0.4頁，~300 words)**

**Table 2: Stage 1 Reconstruction (Mic→LDV, representative window)**
```
Metric: Energy reduction = 1 - (||r||² / ||y||²)
Window: clip_idx=0, start_t=32, Tw=16, MaxLag=16, K=3, STFT bins 5–300

┌─────────────────────────┬──────────────┬──────────────────────────────────────────────┐
│ Method                  │ Energy Red.   │ Evidence / Reproduction                      │
├─────────────────────────┼──────────────┼──────────────────────────────────────────────┤
│ Pursuit oracle          │ 97.88%        │ `python -m scripts.h_exploration.calc_global_omp --clip_idx 0` |
│ Freq-Aware DTMin (ours) │ 97.11%        │ `python -m scripts.h_exploration.validate_global_dt --model_path results/dt_freq_aware_full/dt_freq_aware_best.pth --clip_idx 0` |
└─────────────────────────┴──────────────┴──────────────────────────────────────────────┘
```

**Discussion (draft)**
```
Stage 1 is evaluated as a reconstruction sanity check at a fixed angle θ₀.
The near-oracle energy reduction on a representative window indicates that the learned policy can explain the LDV STFT using a small set of Mic-frame lags.

For the paper, this section should also report averages across many clips/windows (not just one representative window) and include the planned ablations (w/o frequency embedding, frozen vs fine-tuned encoder).
```

#### **4.3 Ablation Studies (0.4頁，~300 words)**

**4.3.1 Frequency-Aware的必要性 (150 words)**

**Planned ablations (no unverified numbers)**
```
Goal: isolate why conditioning on the frequency bin matters for lag selection.

Ablation variants:
  - w/o frequency embedding: remove freq conditioning in Stage 1
  - narrow-band vs wide-band training: train a policy on a restricted bin range and test outside it

Evaluation:
  - Energy reduction on the same reconstruction protocol as Table 2
  - Report both representative-window results and averages over many clips/windows
```

**Figure 1建議: Learned Lag Selection Strategy vs Frequency**
```
X-axis: frequency (Hz) or STFT bin index (b)
  - Use f(b) = b · fs / n_fft to convert bins to Hz (fs=16 kHz, n_fft=2048 in current codebase).
Y-axis: P(select lag ℓ | frequency bin)

Plot options:
  - Heatmap: frequency bins × lag index, colored by selection probability
  - Or line plots for a few representative bins

Caption (draft):
  Frequency-conditioned policies can learn different lag-selection behaviors across bins,
  consistent with phase-wrapping and frequency-dependent transfer characteristics.
```

**4.3.2 兩階段vs直接訓練 (150 words)**
```
Why can direct end-to-end DOA training be brittle in cross-sensor settings?

Hypothesis (to be verified experimentally):
  Speech carries many strong content cues (speaker, phonetic structure, formants).
  A model trained end-to-end may exploit these content shortcuts instead of learning stable, angle-dependent transfer cues.

Planned diagnostics:
  1) Confusion matrices across angles (systematic neighbor confusions vs dispersed errors)
  2) Inspect learned transfer characteristics (e.g., smoothness/physical plausibility of Ĥ(f,θ))
  3) Cross-content evaluation (speech → other signals) to test content reliance

Goal of the two-stage design:
  Separate “angle-agnostic transformation features” (Stage 1) from “angle inference using calibrated H_cal(f,θ)” (Stage 2).
```

#### **4.4 Analysis (0.4頁，~320 words)**

**4.4.1 ~~為什麼白噪音有效？~~ [已過時 - 見下方說明]**

> **⚠️ 方法論修正**：此章節基於舊的「白噪音訓練→語音遷移」敘事。
> 實際方法是：H已從白噪音預先計算，Stage 1直接在語音上訓練。
> 此章節保留供參考，但不應納入論文。

```
[DEPRECATED] 白噪音的關鍵特性：

1. 平坦頻譜：
   S_white(f) ≈ const for f ∈ [20Hz, 8kHz]
   → 所有頻率均勻激勵
   → 模型無法利用頻譜先驗

2. 隨機相位：
   φ_white(f,t) ~ Uniform(-π, π)
   → 無周期性/諧波結構
   → 模型無法記憶時域模式

3. 平穩性：
   E[X(t)] = const, E[X(t)X(t+τ)] = R(τ)
   → 統計特性不變
   → 模型只能學系統響應H

對比語音：
- 基頻周期 → 模型可以利用自相關
- 共振峰 → 模型可以記憶頻譜pattern
- 音素邊界 → 模型可以學時域結構
```

**4.4.2 Coherence與性能關係 (100 words)**
```
Data integrity check (planned):

Compute magnitude-squared coherence γ²(f) between Mic and LDV.

Interpretation:
✓ High coherence supports the existence of a stable *linear* relationship (consistent with an LTI transfer-function model).
✓ It suggests the two sensors observe strongly related physical phenomena in those bands (given the current setup).

Caveat:
⚠ High coherence does not imply DOA is “easy”. DOA difficulty depends on angle separability
  (how much H_cal(f,θ) changes with θ), noise, and modeling mismatch.

Future work:
  Report (i) coherence curves and (ii) an angle-separability metric for H_cal across θ.
```

**4.4.3 Limitations (100 words)**
```
當前實驗的局限性（非方法本身）：

1. 單一test case:
   僅在Mic-to-LDV上驗證
   需要更多sensor pairs來全面驗證通用性

2. 單聲源假設:
   未測試多聲源場景（cocktail party）
   可擴展，但需額外實驗

3. 靜態場景:
   聲源和傳感器靜止
   移動聲源需要tracking機制

4. 數據集規模:
   260 speech clips相對較小
   但足夠驗證proof-of-concept

強調：這些是實驗範圍的限制，非方法設計的限制
Framework本身是sensor-agnostic
```

---

### 2.8 §5 Discussion & Conclusion (0.3頁，~240 words)

#### **5.1 Why Does Two-Stage Work? (120 words)**
```
Intuition (general, not LDV-specific):

- Direct end-to-end DOA: a model maps observed signals to θ.
  In cross-sensor settings, it may exploit content shortcuts (speaker/formants) instead of learning stable, angle-dependent transfer cues.

- Two-stage framework:
  Stage 0 calibrates an angle-indexed transfer-function library H_cal(f,θ) from white noise (content-independent excitation).
  Stage 1 learns angle-agnostic transformation features at a fixed θ₀ using speech pairs but no angle labels.
  Stage 2 injects angle information by matching Stage 1 features to H_cal(f,θ) for direction classification.

Empirical evidence / plan:
  - Stage 1 reconstruction reaches near-oracle energy reduction on a representative window (Table 2).
  - Stage 2 direction accuracy and cross-content generalization will be reported once the Stage 2 pipeline is implemented.
```

#### **5.2 Generalizability Beyond LDV (80 words)**
```
方法的applicability beyond Mic-to-LDV:

Candidate scenarios (future work):

1. Bone Conduction → Air Microphone (Hearing Aids)
2. Contact Mic → Free-field Mic (Studio Recording)
3. Underwater Hydrophone → Air Microphone (Cross-medium)
4. Array Mic (Old room) → Array Mic (New room)

核心條件：
需要sensor-to-sensor correspondence (測量同一物理過程)

框架本身是sensor-agnostic，適用於任意adaptation場景
```

#### **5.3 Conclusion (40 words)**
```
We propose a systematic two-stage framework for cross-sensor acoustic transfer.
In the Mic→LDV case study, Stage 1 achieves near-oracle reconstruction energy reduction under a reproducible protocol.
Stage 2 direction estimation will evaluate downstream transfer once implemented.
```

---

### 2.9 關鍵可視化規劃

#### **Figure 1: Framework Overview (必須，0.2頁)**
```
┌───────────────────────────────────────────────────────┐
│  General Framework (Sensor-Agnostic)                  │
├───────────────────────────────────────────────────────┤
│                                                        │
│  Stage 0: Calibration (offline)                       │
│  White noise @ each θ  ──→  estimate H_cal(f,θ)       │
│                                                        │
│  Stage 1: Angle-agnostic feature learning (single θ₀) │
│  Paired speech (Mic, LDV) @ θ₀  ──→  encoder/policy z  │
│                                                        │
│  Stage 2: Angle-aware task head (multi-angle)          │
│  z + H_cal(f,θ)  ──→  direction head  ──→  θ           │
│                                                        │
├───────────────────────────────────────────────────────┤
│  Instantiation: Mic-to-LDV (This Work)                │
├───────────────────────────────────────────────────────┤
│  Sensor S: Microphone (pressure)                      │
│  Sensor T: LDV (vibration velocity)                   │
│  Task: Speech source localization (37 angles)         │
└───────────────────────────────────────────────────────┘
```

#### **Figure 2: Freq-Aware Ablation (強烈推薦，0.15頁)**
```
[Heatmap: Lag Selection Probability]

Y-axis: STFT frequency bins (e.g., 5–300)
X-axis: Lag index (0-15)
Color: Selection probability

Expected outcome:
  Structured variation of lag-selection behavior across frequency bins.
  (The exact pattern should be reported from the trained model; avoid hard-coding an unverified trend.)
```

---

### 2.10 寫作風格指南

#### **Tone & Language**
```
✓ DO:
- 主動語態："We propose..." (不是"It is proposed...")
- 具體數字："improves accuracy by 50%" (不是"significantly improves")
- 因果清晰："Because X, therefore Y"

✗ DON'T:
- 過度claim："revolutionary", "breakthrough"
- 模糊描述："quite good", "very effective"
- 冗長從句：keep sentences < 25 words
```

#### **術語一致性**
```
1. 統一術語：
   - "cross-sensor" (不是有時"cross-modality"有時"cross-sensor")
   - "Stage 1" / "Stage 2" (不是"Phase 1")
   - "Mic" or "Microphone" 統一

2. 數學符號：
   - Bold uppercase for matrices: X, Y, H
   - Bold lowercase for vectors: x, y, h
   - Italics for scalars: f, t, k

3. 縮寫首次全寫：
   - "LDV (Laser Doppler Vibrometry)"
   - "OMP (Orthogonal Matching Pursuit)"
```

#### **語言模式替換表（通篇適用）**

| ❌ 避免 (LDV-centric) | ✅ 使用 (Method-centric) |
|---------------------|----------------------|
| "LDV poses unique challenges" | "Cross-sensor adaptation poses challenges, exemplified by LDV" |
| "We propose a method for LDV" | "We propose a general method, validated on LDV" |
| "LDV applications include..." | "Cross-sensor methods enable applications such as LDV-based..." |
| "The LDV dataset..." | "The test case dataset (Mic-to-LDV)..." |
| "To solve LDV's problem" | "To address cross-sensor challenges" |
| "LDV-specific design" | "Sensor-agnostic design (instantiated for LDV)" |

---

## 3. 投稿成功率深度分析

### 3.1 為什麼不是更高的成功率？

#### **初步估計的問題**
之前提到的"50-60%"隱含了一個**關鍵假設**：
```
"假設pilot實驗顯示遷移效果 ≥70%"
```

這是一個**條件機率**：
```
P(Accept | pilot成功) = 50-60%
```

但**無條件成功率**需要考慮pilot失敗的可能性。

---

### 3.2 成功率的機率樹分解

#### **完整機率模型**

```
投稿決策
    │
	    ├─ P(技術實驗成功) = ?
	    │   │
	    │   ├─ H_cal calibration (white noise) stable      → 90% (已有pipeline)
	    │   ├─ 遷移到語音 ≥70%        → ⚠️ 70% (最大不確定性)
	    │   ├─ Classical baseline (H_cal matching)         → 85% (implementation risk)
	    │   └─ Ablation studies       → 90% (相對簡單)
    │
    │   綜合實驗成功率 = 0.9 × 0.7 × 0.85 × 0.9 = 0.48 (48%)
    │
    ├─ P(寫作質量|實驗成功) = ?
    │   │
    │   ├─ 4頁結構清晰            → 80% (時間緊)
    │   ├─ 圖表製作精良           → 85%
    │   └─ 語言潤色到位           → 75% (非母語)
    │
    │   綜合寫作質量 = 0.8 × 0.85 × 0.75 = 0.51 (51%)
    │
    └─ P(錄取|好論文) = ?
        │
        ├─ 技術創新性足夠         → 77% (重新定位後)
        ├─ 審稿人expertise匹配    → 70%
        ├─ 競爭對手質量不太強     → 75%
        └─ 避免運氣因素           → 80%

        綜合錄取率 = 0.77 × 0.7 × 0.75 × 0.8 = 0.32 (32%)
```

#### **無條件成功率計算**

```
P(Accept) = P(實驗成功) × P(寫作質量|實驗成功) × P(錄取|好論文)
          = 0.48 × 0.51 × 0.32
          = 0.078 ≈ 8%  😱
```

**這太悲觀了！問題在哪？**

---

### 3.3 更現實的評估模型

#### **問題1：機率獨立性假設過嚴**

實際上：
- 如果實驗很convincing，審稿人更可能接受（不是獨立的）
- 如果創新性強，即使有小瑕疵也可能接受

更合理的模型：
```
P(Accept) ≈ 技術創新性 × 實驗說服力 × 寫作清晰度 × 審稿運氣
```

#### **問題2：遷移實驗成功率的不確定性**

**為什麼遷移≥70%的機率只有70%？**

**悲觀因素**：
1. ⚠️ freq-aware-policy的97%是在**speech數據**上訓練的
2. ⚠️ 可能學到了speech-specific patterns（而非純系統H）
3. ⚠️ 白噪音→語音的gap可能比想像的大

**樂觀因素**：
1. ✅ High coherence (>0.96) 說明H確實存在且穩定
2. ✅ Lag結構物理合理（Lag 1>0>2>3衰減模式）
3. ✅ 已有完整pipeline，技術風險相對可控
4. ✅ 從commit history看，OMP方法在多個場景都有效

**重新評估**：P(遷移≥70%) = 70%（保持不變，但有理有據）

---

### 3.4 修正後的成功率評估

#### **場景A：激進路線（直接投稿，不做pilot）**

```
P(技術創新性足夠) = 0.77  (重新定位後大幅提升)

P(實驗convincing) = P(遷移≥70%) × P(baseline成功) × P(ablation成功)
                  = 0.70 × 0.85 × 0.90
                  = 0.54

P(寫作OK) = 0.75  (4頁可控)

P(審稿運氣|solid paper) = 0.60  (solid paper的錄取率)

P(Accept) = 0.77 × 0.54 × 0.75 × 0.60
          = 0.188 ≈ 19%
```

**結論：直接投稿成功率約 15-25%** ⚠️

---

#### **場景B：穩妥路線（Pilot + 條件投稿）**

```
Week 1-2: Pilot實驗（小規模）
├─ H_cal calibration (white noise) (subset)
└─ Stage 1/Stage 2 pilot on speech (subset)

Week 2末決策點：
┌─────────────────────────────────────────┐
│ IF pilot ≥ 75%:                        │
│   → 繼續投稿                            │
│   P(全量重現|pilot好) = 0.85           │
│   P(Accept|pilot好) = 0.77×0.85×0.75×0.60│
│                     = 0.29 (29%)       │
│                                         │
│   BUT: pilot成功已經validate了假設      │
│   → 審稿人會更信服                      │
│   → Boost錄取率 to 40-45%              │
│                                         │
├─────────────────────────────────────────┤
│ IF pilot 65-75%:                       │
│   → 邊界case，可以試                    │
│   P(Accept) ≈ 20-25%                   │
│                                         │
├─────────────────────────────────────────┤
│ IF pilot <65%:                         │
│   → 放棄InterSpeech，轉Nature Comm     │
│   (只損失2週，可接受)                   │
└─────────────────────────────────────────┘

P(pilot ≥75%) = 0.50  (樂觀估計)
P(pilot 65-75%) = 0.30
P(pilot <65%) = 0.20

加權平均成功率 = 0.50 × 0.42 + 0.30 × 0.22 + 0.20 × 0
               = 0.21 + 0.066
               = 0.276 ≈ 28%
```

**結論：Pilot策略成功率約 25-30%** ⚠️

---

#### **場景C：最樂觀情況（Pilot好 + 完美執行）**

```
假設：
1. ✅ Pilot顯示遷移≥75%（我們才投稿）
2. ✅ 全量實驗順利重現
3. ✅ 實現DOANet簡化版baseline（額外加分）
4. ✅ 寫作質量高
5. ✅ 重新定位為cross-sensor後相關性強

在這些條件都滿足時：

P(Accept|all above) = 0.77 × 0.90 × 0.85 × 0.70
                    = 0.41 ≈ 40-45%

加上主觀優化：
- 重新定位後相關性提升 (+10-15%)
- Freq-Aware ablation很convincing (+5%)

最終：40-45% + 15% = 55-60%
```

**這就是我之前說的"50-60%"！**

---

### 3.5 關鍵結論

| 場景 | 策略 | 成功率 | 備註 |
|------|------|--------|------|
| **無條件投稿** | 不管實驗結果如何都投 | **15-25%** | 不推薦，太冒險 |
| **Pilot+條件** | 先驗證，好才投 | **25-35%** | 更現實的估計 |
| **Pilot好+完美執行** | pilot≥75%且一切順利 | **50-60%** | 這是條件機率 |

#### **為什麼即使完美執行也只有50-60%？**

**制約因素排名**：

| 風險 | 影響 | 可控性 | 緩解方案 |
|------|------|--------|---------|
| **1. 遷移實驗不確定性** | 🔴 -30% | ⚠️ 中 | **必須做pilot驗證** |
| **2. InterSpeech基礎錄取率低** | 🟡 -15% | ❌ 不可控 | Regular: ~32%, Long: <30% |
| **3. 缺少神經baseline** | 🟡 -10% | ⚠️ 中 | 努力實現DOANet簡化版 |
| **4. 單一test case** | 🟡 -10% | ❌ 低 | 時間不夠做其他sensor pair |
| **5. 寫作時間緊** | 🟢 -5% | ✅ 高 | 提前規劃，並行工作 |

---

### 3.6 如何提升成功率？

#### **策略1：投入更多實驗（目標60-70%）**

**增加以下內容：**
```
✅ P0實驗全部完美完成
✅ 加做簡化版DOANet baseline
✅ 加做第二個sensor pair驗證（e.g., Contact Mic → Air Mic）
  （即使小規模，也能證明generalizability）
✅ 充足時間打磨論文（8-10週而非6週）
```

**時間需求**：8-10週（比現在多2-4週）
**成功率提升**：到60-70%

---

#### **策略2：接受現實，降低期望（務實）**

**理性思考：25-35%已經不錯**

理由：
```
✓ InterSpeech錄取率本來就只有32%
✓ 我們的工作是solid，但不是top-tier breakthrough
✓ 25-35%已經**高於平均錄取率**了

而且：
✓ 如果被拒，feedback很valuable
✓ 可以根據意見改進後投ICASSP 2027
✓ Nature Comm仍是保底（70%成功率）
```

---

#### **策略3：兩階段決策（推薦）**

```
Phase 1: Pilot (Week 1-2)
├─ 投入：2週時間，小規模驗證
└─ 產出：遷移準確率數據

Phase 2: 決策點 (Week 2末)
├─ IF pilot ≥75%:
│   → 全力衝刺InterSpeech (成功率40-50%)
│   → 6週完成
│
├─ IF pilot 65-75%:
│   → 評估是否值得（成功率20-25%）
│   → 或投入額外2週做第二個sensor pair
│
└─ IF pilot <65%:
    → 立即轉向Nature Comm
    → 只損失2週，可接受
```

**優點**：
- ✅ 風險可控（最多損失2週）
- ✅ 基於數據決策（不是盲目樂觀）
- ✅ 有Nature Comm保底

---

## 4. 關鍵實驗規劃

### 4.1 必做實驗（P0）

| 實驗 | 目的 | 工作量 | 預期結果 | 風險 |
|------|------|--------|---------|------|
| **1. H_cal calibration (white noise)** | 驗證Stage 0 | 2-3天 | H_cal(f,θ) across angles is stable & physically plausible | 🟢 低（有pipeline） |
| **2. Stage 2 direction training** | 驗證Stage 2 | 1週 | Report DOA accuracy (Top-1, ±10°) | 🔴 高（最大不確定） |
| **3. End-to-end DOA baseline** | Ablation | 3天 | Train/eval without Stage 1 pretraining and/or without H_cal | 🟢 低 |
| **4. Classical baseline (H_cal matching)** | 經典方法對比 | 2-3天 | Non-learned template matching using H_cal | 🟢 低 |
| **5. Frequency-conditioning ablation** | 證明創新點 | 2-3天 | w/ vs w/o frequency embedding (report deltas) | 🟡 中 |

**總時間：約3.5週**

---

### 4.2 強烈建議實驗（P1）

| 實驗 | 目的 | 工作量 | 價值 |
|------|------|--------|------|
| **6. 跨信號泛化（Music）** | 驗證H通用性 | 3天 | ⭐⭐⭐⭐⭐ 關鍵證據 |
| **7. DOANet簡化版** | 神經baseline | 5天 | ⭐⭐⭐⭐ 審稿人會問 |
| **8. H(f)物理分析** | 可解釋性 | 2天 | ⭐⭐⭐ 增強說服力 |

**額外時間：約1.5週**

---

### 4.3 可選實驗（P2）

| 實驗 | 目的 | 工作量 |
|------|------|--------|
| **9. 第二個sensor pair** | 證明generalizability | 1週 |
| **10. 不同SNR魯棒性** | 引用白噪音結果 | 3天 |
| **11. Masking策略ablation** | 完整性 | 5天 |

---

### 4.4 實驗優先級決策樹

```
IF 時間充裕 (8-10週):
  → P0 + P1 + P2部分
  → 成功率：60-70%

IF 時間緊迫 (6週):
  → P0 + P1中的實驗6
  → 成功率：40-50% (pilot好的情況下)

IF 極端緊迫 (4週):
  → P0 only
  → 成功率：25-35%
```

---

## 5. 時間線與風險管理

### 5.1 推薦時間線（Pilot + 條件投稿）

#### **6週執行計畫（假設今天1月13日開始）**

| 時間段 | 任務 | 交付物 | 里程碑 |
|--------|------|--------|--------|
| **Week 1** (1/13-1/19) | Pilot：H_cal calibration (white noise) | H_cal(f,θ) library + sanity plots | - |
| **Week 2** (1/20-1/26) | Pilot：Stage 1 validation (+ Stage 2 prototype if ready) | Stage 1 energy reduction + preliminary DOA metrics | 🔴 **決策點** |
| **Week 3** (1/27-2/2) | P0-2,3: 全量遷移+對比 | 完整表格2數據 | - |
| **Week 4** (2/3-2/9) | P0-4,5 + P1-6: Baseline+Ablation | 表格3+跨信號測試 | - |
| **Week 5** (2/10-2/16) | 論文撰寫：Method+Experiments | §3-4初稿 | - |
| **Week 6** (2/17-2/23) | 論文撰寫：Intro+Figures+潤色 | 完整終稿 | - |
| **2/24-2/25** | 最終檢查+提交 | 提交！ | 🎯 **截稿** |

---

#### **Week 2末決策點詳情**

```
2026年1月26日（週日）晚上評估：

測試指標：
├─ Pilot遷移準確率（20 speech clips）
├─ 跨信號測試（5 music clips）
└─ H(f)響應的物理合理性

決策規則：
┌────────────────────────────────────────┐
│ IF pilot ≥ 75%:                       │
│   ✅ GO: 全力衝刺InterSpeech           │
│   📅 Week 3-6: 執行剩餘實驗+寫作        │
│   🎯 預期成功率: 40-50%                │
│                                        │
├────────────────────────────────────────┤
│ IF pilot 65-75%:                      │
│   ⚠️ CONSIDER: 評估以下因素            │
│   • H(f)是否物理合理？                 │
│   • 跨信號是否>60%？                   │
│   • 是否有額外2週投入第二sensor pair？  │
│   如果都是Yes → GO (成功率25-30%)     │
│   否則 → STOP                          │
│                                        │
├────────────────────────────────────────┤
│ IF pilot <65%:                        │
│   ❌ STOP: 立即轉向Nature Comm         │
│   只損失2週，完全可接受                 │
│   Nature Comm成功率: 70%               │
└────────────────────────────────────────┘
```

---

### 5.2 風險識別與緩解

#### **風險矩陣**

| 風險 | 機率 | 影響 | 緩解策略 | 應急方案 |
|------|------|------|---------|---------|
| **遷移<70%** | 40% | 🔴 致命 | Pilot驗證 | 轉Nature Comm |
| **Classical baseline實現困難** | 20% | 🟡 重要 | 優先做H_cal matching（單通道可用） | 以H-only baseline替代 |
| **實驗結果不穩定** | 20% | 🟡 重要 | 多次訓練取平均 | 報告variance |
| **寫作時間不夠** | 50% | 🟠 嚴重 | 並行實驗+寫作 | 砍掉P2實驗 |
| **審稿人不買帳cross-sensor定位** | 15% | 🟡 重要 | Introduction強調通用性 | Rebuttal階段強調 |

---

#### **關鍵路徑分析**

```
Critical Path:
  Pilot驗證 → 遷移實驗 → Classical baseline (H_cal matching) → 論文撰寫

如果任何環節延誤：
├─ Pilot延誤1週 → 總時間變7週 → 來不及
├─ 遷移實驗延誤 → 砍掉P1實驗 → 成功率降至30%
└─ Baseline困難 → 用簡化版 → 審稿人可能質疑
```

---

### 5.3 並行工作策略

**為了節省時間，可以並行：**

```
Week 3-4 同時進行：
├─ 實驗線程：
│   ├─ 主力：全量遷移實驗（你或主要研究人員）
│   └─ 輔助：Classical baseline (H_cal matching)（如果有合作者）
│
└─ 寫作線程：
    ├─ Method部分可以提前寫（不依賴實驗結果）
    └─ Related Work可以提前寫（文獻調研）

Week 5 同時進行：
├─ 實驗線程：P1-6跨信號測試（快速）
└─ 寫作線程：Experiments部分（邊實驗邊寫）
```

---

## 6. 決策建議

### 6.1 三種策略對比

| 策略 | 時間 | 成功率 | 風險 | 推薦度 |
|------|------|--------|------|--------|
| **A. 直接投稿** | 6週 | 15-25% | 🔴 高 | ⭐⭐☆☆☆ |
| **B. Pilot+條件** | 2+6週 | 25-35% | 🟡 中 | ⭐⭐⭐⭐⭐ |
| **C. 專注Nature** | 4週 | 70% | 🟢 低 | ⭐⭐⭐⭐☆ |

---

### 6.2 決策流程圖

```
                    開始決策
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
    優先級1                          優先級2
  快速發表？                      進入Speech社群？
        │                               │
        Yes                            Yes
        ↓                               ↓
   Nature Comm                    做Pilot實驗
   (70%成功率)                    (2週投入)
   4週完成                             ↓
        ↓                        ┌──────┴──────┐
     完成！                      │             │
                            pilot≥75%      pilot<65%
                                 │             │
                                 ↓             ↓
                          衝刺InterSpeech   轉Nature Comm
                          (40-50%成功)      (只損失2週)
                          6週完成                ↓
                                 ↓            完成！
                              提交等結果
```

---

### 6.3 最終建議（基於不同優先級）

#### **如果目標是"盡快發高質量論文"**

➡️ **推薦：專注Nature Communication**

理由：
```
✅ 成功率高：70%
✅ 時間短：4週
✅ 影響力高：跨學科期刊，Nature系列
✅ 已有進展：SNR robustness study完成
✅ 風險低：白噪音100%結果很強

InterSpeech可以之後投：
- 2027年用Nature Comm的結果作為基礎
- 補充speech-specific內容
```

---

#### **如果想"進入Speech社群 + 願意承擔風險"**

➡️ **推薦：Pilot + 條件投稿策略**

執行步驟：
```
Week 1-2: Pilot實驗（小規模驗證）
  ├─ 投入：2週，可控損失
  └─ 產出：遷移準確率數據

Week 2決策點：
  ├─ pilot ≥75%:
  │   → 繼續（成功率40-50%）
  │   → 值得冒險
  │
  └─ pilot <65%:
      → 放棄，轉Nature Comm
      → 只損失2週

優點：
✓ 基於數據決策（不是盲目）
✓ 有Nature Comm保底
✓ 即使失敗也獲得valuable feedback
```

---

#### **如果有充足資源（團隊支持/更多時間）**

➡️ **考慮：增強版InterSpeech投稿**

需要滿足：
```
✓ 時間：8-10週（不是6週）
✓ 人力：可以並行多個實驗
✓ 資源：能做第二個sensor pair驗證

成功率可達：60-70%

額外收穫：
- 更完整的cross-sensor框架驗證
- 審稿人無法挑剔實驗完整性
- 即使被拒，改投後成功率>80%
```

---

### 6.4 作者個人建議

**基於以上所有分析，我的誠實建議：**

#### **推薦策略：Pilot + 條件投稿**（策略B）

理由：
1. ✅ **平衡風險與機會**
   - Nature Comm保底（70%）
   - InterSpeech有機會（40-50%，如果pilot好）

2. ✅ **基於數據決策，不是賭博**
   - 2週pilot可以validate核心假設
   - 避免6週全投入後發現不可行

3. ✅ **即使失敗也有收穫**
   - Pilot數據對Nature Comm也有價值
   - 可以在Discussion中討論遷移的挑戰

4. ✅ **符合科學方法**
   - 先驗證假設，再全力投入
   - 體現了嚴謹的研究態度

---

#### **不推薦策略：直接投稿**（策略A）

理由：
1. ❌ **成功率太低（15-25%）**
   - 不值得6週全力投入
   - 機會成本太高

2. ❌ **最大風險未驗證**
   - 遷移實驗的不確定性是最大風險
   - 直接賭這個風險不明智

3. ❌ **如果失敗，損失巨大**
   - 6週時間浪費
   - 可能延誤Nature Comm

---

### 6.5 執行檢查清單

如果決定採用**Pilot+條件策略**，請確認：

#### **Week 1-2 (Pilot階段)**
```
□ 環境準備：Conda環境、MPS GPU測試
□ 數據準備：白噪音校準數據（用於估計H_cal(f,θ)）
□ 代碼準備：確保freq-aware-policy代碼可運行
□ Stage 0：估計H_cal(f,θ)，檢查跨角度的一致性與物理合理性
□ Stage 1：單角度θ₀語音訓練（無角度標籤），並用validation scripts驗證reconstruction energy reduction
□ Stage 2：方向head的最小可行版本（如果時間允許）
□ 跨內容測試：非語音信號（如果有）作為generalization check
□ H_cal(f,θ)分析：繪製幅度/相位與角度依賴性
□ 決策會議：Week 2末評估結果
```

#### **Week 3-6 (如果GO，全力衝刺)**
```
實驗線：
□ P0-2: 全量遷移實驗（260 clips）
□ P0-3: 直接語音訓練對比
□ P0-4: Classical baseline (H_cal matching)實現
□ P0-5: Freq-Aware ablation
□ P1-6: 跨信號泛化（Music完整測試）
□ 數據整理：所有表格數字確認

寫作線：
□ Week 3-4: Method部分初稿
□ Week 4: Experiments部分初稿
□ Week 5: Introduction + Related Work
□ Week 5: Figure 1-2製作
□ Week 6: 整合+潤色
□ Week 6: 內部review
□ 2/24: 最終檢查
□ 2/25: 提交！
```

---

## 附錄A：審稿人可能的問題與預防

### A.1 關於LDV的問題

**Q1: "為什麼研究LDV這麼小眾的東西？"**

**預防性回答（在Introduction中）**:
```
"我們選擇LDV不是因為它是目標應用，而是因為它代表了
cross-sensor adaptation的extreme case。傳感器模態差異極大
（pressure vs velocity），如果方法在此場景有效，說明它在
其他less challenging scenarios也會有效。這類似於Computer
Vision用ImageNet作為benchmark - 不是為了分類狗，而是為了
驗證方法的通用性。"
```

---

### A.2 關於實驗的問題

**Q2: "為什麼只有一個classical baseline？神經baseline在哪？"**

**預防性回答（在4.1.2中）**:
```
"Array-based DOA methods such as SRP-PHAT/MUSIC are strong classical baselines when multi-channel measurements are available.
In our single Mic + single LDV-point setting, these array-based baselines are not directly applicable.
Therefore, we include a physically grounded classical baseline based on H_cal matching, and we discuss how multi-channel baselines could be added if multi-point measurements are collected."
```

如果有時間實現DOANet簡化版，這個問題就不存在了。

---

**Q3: "The transfer looks surprisingly strong — is there data leakage?"**

**預防性回答（在4.4.2或補充材料中）**:
```
"我們進行了嚴格的數據完整性檢查：
1. Train/val split基於clips，沒有重疊
2. Calibration (white noise) and speech recordings are separated by design (to be verified and documented)
3. We will report coherence and other integrity checks as supporting evidence, but we do not use coherence as proof of no leakage"
```

---

**Q4: "Music測試集多大？統計顯著性如何？"**

**預防性回答（在4.1.1中明確說明）**:
```
"Cross-content evaluation will use held-out non-speech clips (e.g., music and environmental sounds).
We will report the exact dataset size, angle coverage, and confidence intervals in §4.1.1 once the evaluation set is finalized."
```

---

### A.3 關於方法的問題

**Q5: "Frequency Embedding不就是簡單的lookup table嗎？有什麼創新？"**

**預防性回答（在3.2.2中）**:
```
"Frequency Embedding的創新在於：
1. 物理動機：延遲造成的相位項隨頻率縮放（φ(f) = -2π·f·τ mod 2π）
2. 解決實際問題：沒有frequency conditioning時，跨頻段lag選擇容易不穩定（以消融實驗量化）
3. 可解釋性：可視化顯示不同頻率段的lag選擇行為存在結構性差異（圖2）

而非盲目加入一個embedding layer。"
```

---

**Q6: "兩階段訓練會增加計算成本嗎？"**

**預防性回答（在Conclusion或補充材料中）**:
```
"Compute cost (planned wording):
- Stage 0 calibration (white noise → H_cal) is a lightweight offline step.
- Stage 1 + Stage 2 adds training cost versus direct end-to-end DOA, but Stage 1 is reusable once trained.
- Inference cost is similar to a standard encoder+classifier; H_cal lookup is negligible."
```

---

## 附錄B：與Nature Communication的對比

### B.1 兩個venue的優劣對比

| 維度 | InterSpeech 2026 | Nature Communication |
|------|------------------|---------------------|
| **適配度** | 65%（重新定位後） | 95%（天然契合） |
| **當前進度** | 30%（需要新實驗） | 80%（SNR study完成） |
| **所需時間** | 6-8週 | 4週 |
| **成功率** | 25-50%（取決於pilot） | 70% |
| **影響力** | 中（Speech社群） | 高（跨學科，Nature系列） |
| **受眾** | Speech研究者 | 物理+工程+ML |
| **審稿週期** | 3個月 | 6-9個月 |
| **引用潛力** | 中 | 高 |

---

### B.2 如果兩者都投（時間線）

**理想情況（如果有足夠資源）**:

```
Timeline:
├─ 2026年1月: Pilot實驗
├─ 2026年2月:
│   ├─ IF pilot好 → InterSpeech投稿
│   └─ 同時繼續Nature Comm準備
├─ 2026年3月: Nature Comm投稿
├─ 2026年5月: InterSpeech結果
│   ├─ Accept → 等Nature Comm結果
│   └─ Reject → 專注Nature Comm
└─ 2026年9月: Nature Comm結果

最壞情況：
- InterSpeech拒稿 + Nature Comm拒稿
- 但獲得兩輪審稿意見，改進後投ICASSP 2027幾乎必中

最好情況：
- InterSpeech接收 + Nature Comm接收
- 兩篇高質量論文
```

**風險**：
- Nature Comm可能會在review時質疑與InterSpeech的overlap
- 需要確保兩篇論文focus不同（InterSpeech: 方法，Nature: 物理）

---

## 附錄C：參考文獻建議

### C.1 必引文獻（20-25篇）

**DOA Methods (6-7篇)**:
```
[1] SRP-PHAT foundational paper
[2] MUSIC algorithm
[3] GCC-PHAT
[4] DOANet (Cao et al., ICASSP 2019)
[5] SELDnet (Adavanne et al., WASPAA 2019)
[6] Transformer-based DOA
```

**Transfer Learning (4-5篇)**:
```
[7] Domain adaptation survey
[8] Self-supervised audio (wav2vec, HuBERT)
[9] Cross-domain acoustic (room adaptation)
[10] Few-shot learning in acoustics
```

**Phase/Frequency (3-4篇)**:
```
[11] Phase unwrapping networks
[12] Multi-resolution STFT
[13] Learnable filterbanks
```

**LDV & Sensor Adaptation (3篇)**:
```
[14] LDV for speech capture
[15] Bone conduction → air conduction
[16] Vibration-based sensing
```

**RL & Decision Making (2-3篇)**:
```
[17] Decision Transformer (Chen et al., NeurIPS 2021)
[18] Behavioral cloning
```

**Physics-Informed ML (2篇)**:
```
[19] OMP algorithm (Mallat & Zhang, 1993)
[20] Physics-guided neural networks
```

---

## 文檔版本歷史

| 版本 | 日期 | 主要變更 | 作者 |
|------|------|---------|------|
| v1.0 | 2026-01-13 | 初始版本：完整論文規劃+投稿分析 | Claude |
| v1.1 | 2026-01-13 | 轉換為繁體中文 | Claude |

---

## 聯繫與反饋

如有問題或需要進一步討論，請參考：
- 之前的gap analysis: `docs/interspeech_submission_gap_analysis.md`
- 實驗分支: `experiment/freq-aware-policy-full`
- 主分支: `feature/master-figure-nature-comm`

---

**最後總結**：

這份文檔提供了：
1. ✅ 完整的4頁論文鋪陳（從標題到結論）
2. ✅ 誠實的成功率分析（25-60%，取決於策略）
3. ✅ 詳細的實驗規劃（P0/P1/P2優先級）
4. ✅ 可執行的時間線（6週Pilot+條件策略）
5. ✅ 風險管理與決策流程

**關鍵結論**：
- 推薦：**Pilot + 條件投稿策略**
- 預期成功率：**25-35%**（無條件），**40-50%**（pilot好的條件下）
- 保底方案：**Nature Comm**（70%成功率）

希望這份文檔能幫助你做出明智的決策！🎯
