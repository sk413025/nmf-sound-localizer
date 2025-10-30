結論先講清楚：**是的**——你現在的 **Routed attn soft-OMP (no solver)** 架構在**部署推論**時，**使用者完全不需要知道真實的方向濾波 (H)**。
這個「不知道 (H) 也能用」不是偶然，而是**刻意設計**出來，並且**正好就是 ICL 的語法**：

* **ICL 關鍵 1：no weight updates**（不改權重）
* **ICL 關鍵 2：context-driven**（行為只由「前綴/上下文」決定）
* **ICL 關鍵 3：forward ≈ algorithm**（單次前向就是「選→投影/更新→殘差」的多步）

在我們的網路裡，「不知道 (H)」這件事被**轉化**為「把 (H) 的資訊放在**字典 (D)** 的**專家/碼本**裡」，推論時**只餵 (y)**（和可選的 prefix 前綴），網路就能靠**上下文**自動路由到合適的方向族群，完成 soft-OMP 式的稀疏恢復與 DoA 估計。

---

## 1) (H) 現在「住在哪裡」？（使用者「不知道 (H)」如何被處理）

### 三種我們支援的「(H) 來源」——部署期都**不用**使用者提供

1. **Canonical Grid**（統一方向網格）
   離線準備一組（或幾組）標準方向響應 ({H_\theta^*}*{\theta\in\Theta^*})（例如每 2°）。**部署期固定** (D^*=[H*\theta^*\odot W_k])。

2. **VQ 原型碼本（Expert = 原型 (c)）**
   用混合來源的真實/合成 (H)（1°/2°/3°/5°）做 **K-means/VQ** 得到 ({\hat H_c}*{c=1}^E)。**部署期用**
   (;D*{\text{VQ}}=[\hat H_c\odot W_k])。
   （**好處**：不需要真實 (H)，碼本覆蓋方向流形）

3. **RVQ 殘差碼本（Expert = ((k_1,k_2))）**
   兩階（建議 S=2）：**center** (C^{(1)}) + **residual** (C^{(2)})。合成 (\hat H=C^{(1)}*{k_1}+C^{(2)}*{k_2})，**部署期固定**
   (;D_{\text{RVQ}}=[\hat H\odot W_k])。
   （**好處**：off-grid 誤差小；連續方向近似更好）

> 三種情境在部署期的共通點：**都先凍結字典 (D)**；使用者只要給 (y)，其它由路由與稀疏更新完成。

---

## 2) 與 ICL 的關係（為什麼這就是 ICL）

* **no weight updates**：推論時**不反向、不微調**；行為只由 ((D,\ r_0=y,\ S_0)) 決定。
* **context-driven**：不同裝置/場景的 (H) 差異，被「是否路由到哪個**原型/中心**」來吸收；這完全由**殘差 (r)** 驅動。
* **forward ≈ algorithm**：每一步都是 **coarse（群級相關性篩選）→ fine（群內 Top-L + soft-OMP 權重）→ GD 更新 → 殘差**，**層/loop＝步**。
* **Prefix ICL**：若你提供 prefix（例如少量窗用 OMP 先產生 (S_0)），**Ordered > Shuffled** 的增益曲線會出現——這是 ICL 的經典證據。

---

## 3) 這版網路裡的 **token** 與 **prompt**（特別是在「不知道 (H)」的情況）

### Tokens（連續向量，不是 NLP 字詞）

* **Residual token** (r\in\mathbb{R}^F)：當前殘差；像 **query** 一樣去量每個候選的相關性（**“attention as similarity”**）。
* **Expert tokens**：

  * **VQ**：群 (c)（對應 (\hat H_c)）；
  * **RVQ**：群 ((k_1,k_2))（對應 (C^{(1)}*{k_1}+C^{(2)}*{k_2})）。
    實作上是一組**索引到原子清單**。
* **Atom tokens** (d_{(\text{group},k)}=\hat H_{\text{group}}\odot W_k\in\mathbb{R}^F)：群內的基底組合原子。
* **Support token / mask**（(\mathcal{S}_{\text{union}})）：累積已選原子集合，**without-replacement**，控制總稀疏度。

### Prompt（推論前提供的上下文）

* **核心 prompt**：(\mathbf{Prompt}=\big(D,\ r_0=y,\ S_0\big))

  * **(D)**：由 **VQ/RVQ** 或 **Canonical grid** 得來，**部署期固定**；
  * **(r_0=y)**：量測 STFT；
  * **(S_0)**（可選）：prefix 支撐（例如 3–5 個原子），或用 teacher 先走 (L) 步。
* **語義**：這個 prompt 告訴網路「**有哪些候選原子（含方向結構）**、**當前要解釋的殘差長相**、**已經知道的部分支撐**」。**權重不改**，行為卻會因 prompt 不同而改變——**這正是 ICL**。

---

## 4) 部署推論流程（使用者不知道 (H)，實際上怎麼跑）

1. **固定字典** (D)（擇一）

   * **VQ**：用離線學好的 ({\hat H_c}) 組 (D_{\text{VQ}})；
   * **RVQ**：用 ({C^{(1)},C^{(2)}}) 合成 (\hat H) 組 (D_{\text{RVQ}})；
   * **Canonical**：用離線的 ({H^*_\theta}) 組 (D^*)。

2. **（可選）ICL Calibration**（幾十毫秒）
   蒐集幾窗 (y)，用 OMP/Soft-OMP 產生一個小 (S_0)（或直接用 Top-L），當作 prefix。

   > 這是「**prefix ICL**」；**no weight updates**，只給更好的上下文。

3. **Routed soft-OMP 迭代**（每步）

   * **Coarse（群級）**：
     VQ：(\alpha_c=\max_{j\in c}|(D^\top r)*j|)，選 **Top-(K_e)** 群；
     RVQ：先在 center 取 **Top-(K*{e1})**。
   * **Fine（群內）**：在每個選到的群內取 **Top-(L)** 原子，權重 (w=\mathrm{softmax}(|s|/\tau))。
   * **Update**：(x[idx]\leftarrow x[idx]+\eta,(w\odot (D^\top r)[idx]))；(r\leftarrow y-Dx)。
   * **Support**：(\mathcal{S}*{\text{union}}\gets\mathcal{S}*{\text{union}}\cup \mathcal{S}_t)（**不放回**），(|\mathrm{supp}(x)|\le S\cdot K_e\cdot L)。

4. **DoA 讀出**
   (A_{\text{group}}=\sum_k |x_{\text{group},k}|)；離散取 (\arg\max)，或以群中心做 **barycenter** 得到連續角度。

> **整個過程不需要使用者知道 (H)**。(H) 的角色已在 **字典 (D)** 與 **路由規則**裡實現，**ICL** 則保障了「只靠上下文」（(y) 與小 prefix）就能對新場景做即時調整。

---

## 5) （可選）加入「可訓練」的版本，仍然不需要 (H)

* **最保守（推薦）**：**凍結字典 (D)**，**訓練連續參數**（**“learned similarity”**）：
  (\tau,\ \eta,\ P_R,\ P_D,\ W_q,\ W_k)；路由用 **Gumbel-Softmax** 或 **entmax**（**train = soft**，**deploy = hard**）。
  → **ICL/CS 本質不變**（**fixed (D)** + **no weight update at inference**）。

* **碼本也學（VQ-VAE/ST/EMA；RVQ 兩階 ST）**：
  訓練期用 ST/EMA 更新碼字；**部署**仍是「**先定碼字→固定 (D)**→routed soft-OMP」，**使用者依舊不知道 (H)**。

---

## 6) 小抉擇與風險提示

* **VQ vs RVQ**：
  VQ 較「CS 正統」、解釋性好；RVQ off-grid 誤差小，但**務必**保持「**先定碼字→固定 (D)**」，避免字典在迭代中隨殘差漂移（那會變成**自適應字典**，偏離 CS 固定線性假設）。
* **相干度**：碼本過密會拉高 **“mutual coherence”**，支撐恢復會變難；建議在碼本學習時加**相干正則**、單位化。
* **角度不齊的資料**：訓練期**balanced sampling** + **angular smoothing（von-Mises）**，讓不同來源（1°/2°/3°/5°）都能為碼本/路由提供資訊。

---

## 7) 極簡「圖解表」— 在**不知道 (H)** 的部署期，我們的 **token / prompt / ICL** 是：

* **Tokens**：

  * **Residual token**：(r\in\mathbb{R}^F)（查詢者）
  * **Expert tokens**（VQ：原型 (c)；RVQ：((k_1,k_2))）
  * **Atom tokens**：(d_{(\text{group},k)}=\hat H_{\text{group}}\odot W_k\in\mathbb{R}^F)
  * **Support mask**：(\mathcal{S}_{\text{union}})

* **Prompt**：(\mathbf{Prompt}=(D,\ r_0=y,\ S_0))

  * **(D)** 由碼本或 canonical grid 來；部署固定
  * **(r_0=y)**；**(S_0)** 可為 prefix（ICL）
  * **no weight updates**；行為由 prompt 決定（完全 **ICL**）

* **Forward**：coarse（群級相關性）→ fine（群內稀疏）→ GD 更新 → 殘差；**層/loop＝步**。

---

先給結論、再把對應關係講清楚：

* **結論 1（是不是 Transformer？）**
  你現在的 **Routed attn soft-OMP (no solver)** 是一個「**特化過的 Transformer block**」——更精確地說，它等價於**單一 query 的 cross-attention**（Residual token 作 query，字典原子作 keys/values），再配上**殘差式迭代**（loop = step）。它不是完整的 GPT/ViT 編碼器，但**屬於 Transformer 家族的「最小可動核心」**。

* **結論 2（還對得上 ICL 論文嗎？）**
  對得上，而且是以兩個經典訊號詞對齊：
  **“attention as similarity”**（注意力打分≈相關性 (D^\top r)）與 **“forward pass implements GD/OLS”**（我們用 GD-like 更新模擬投影/重估）。
  不同的是：我們採的是**algorithmic-prior ICL（白盒迭代）**，不是「完全由權重湧現」的 **emergent-ICL**。若要逼近後者，只要把離散路由/外部 LS 替換為**可微/可學**的注意力與更新（下面給做法）。

---

## A) 與標準 Transformer 的「一一對應」

| 我們的元件                                                   | 對應到 Transformer                           | 物理/數學意義（shape）                         |
| ------------------------------------------------------- | ----------------------------------------- | -------------------------------------- |
| **Residual token** (r\in\mathbb{R}^F)                   | **Query**（單 token）                        | 當前殘差，代表「要被解釋的訊號」                       |
| **Dictionary atoms** (d_{(\cdot,\cdot)}\in\mathbb{R}^F) | **Keys/Values**（多 token）                  | (H\odot W) 原子；可來自 VQ/RVQ 原型（retrieval） |
| **打分** (s\propto K q) 或 (D^\top r)                      | **Scaled dot-prod attention**             | **attention ≈ similarity**             |
| **Top-k / entmax**                                      | **稀疏注意力 / MoE gating**                    | 讓少數原子活躍（近似 OMP 選擇）                     |
| **GD-like 更新** (x\leftarrow x+\eta(w\odot D^\top r))    | **Attention 輸出 + FFN 對 state 的更新**        | **forward ≈ GD/OLS** 的一步               |
| **loop = step**                                         | **疊層 / Looped Transformers**              | 層/回圈 = 迭代步（**iterative inference**）    |
| **VQ/RVQ 原型/殘差**                                        | **Retrieval / Codebook（RETRO、VQ-VAE 家族）** | 外部字典；參數不必內化在權重                         |

> 直白說：你目前做的是「**單 query cross-attn + 稀疏路由 + 殘差迭代**」這個 Transformer 的最小內核。

---

## B) 與「Transformer 能 ICL」的研究脈絡是否對齊？

* **對齊處**

  * **attention ≈ similarity**：我們的打分直接就是 (D^\top r) 或它的線性嵌入版本（(q=W_q P_R(r), K=W_k P_D(D))）。
  * **forward ≈ GD/OLS**：無 solver 版用 **GD-like** 更新；有 solver 版則在功能上對等 OMP/LS（但那是白盒插入）。
  * **iterative inference**：loop = step，殘差逐步下降，與「殘差流逐層演化」的觀察一致。
  * **retrieval/外部記憶**：VQ/RVQ codebook 對應到 **retrieval-augmented Transformers**（RETRO 類）——把知識放在外部庫，而非全部內化於權重。

* **偏離處（如果你要「更像」論文展示）**

  * 你目前的 **hard Top-k 路由**與 **VQ/RVQ 近鄰選碼**是**離散**的；這更像 **algorithmic-prior ICL**，而非權重**湧現**。
  * 想貼近 **emergent-ICL**：把路由與更新改成**可微/可學**（例如 Gumbel-Softmax/entmax 路由、可學步長/投影、多頭注意力＋FFN），並用重建＋正交＋單調等損失來訓練。

---

## C) 如果你要「更像標準/先進 Transformer」——三個漸進升級

1. **Minimal-Transformer 化（不改任務幾何）**

   * 把目前的打分換成**多頭 cross-attention**（**multi-head QK^T/sqrt(d)**）；
   * 在每步加入 **Pre-Norm + 殘差** 與 **FFN**（例如 (\text{LayerNorm}(r) \rightarrow \text{Attn}\rightarrow) 殘差，接 (\text{FFN}) 再殘差）；
   * 路由改 **entmax / soft top-k**（train soft / infer hard）；
   * **loop = step** 保留，狀態向量就是殘差/係數的某種投影。
     → 這樣幾乎就是「**Perceiver/Set Transformer 的單 latent cross-attn**」特例。

2. **Emergent-ICL 化（去掉 solver／保留可微）**

   * 用 **GD-unroll** 或 **Neumann/CG 展開** 取代任何顯式 `solve()`；
   * 訓練 (\tau,\eta,P_R,P_D,W_q,W_k) 與（可選）小 FFN head，讓「**forward ≈ GD**」成為**學得行為**；
   * 損失：重建 + 殘差單調 + 正交 +（可選）支撐 CE。
     → 這最貼「**forward pass implements GD/OLS**」的論述。

3. **Retrieval-Transformer 化（對齊先進檢索式架構）**

   * 把 VQ/RVQ codebook 視為**外部記憶**；coarse-to-fine 路由 ≈ **MoE/RETRO** 的檢索與專家選擇；
   * 將粗路由實作為**檢索層**（例如 ANN/MIPS 取候選），細選由 cross-attn 完成；
   * 這條線保留「使用者推論時不知 (H)」的優點，同時與**現代檢索式 Transformer**一致。

---

## D) Token 與 Prompt（在「使用者不知道 (H)」的部署情境）

* **Tokens（連續向量）**

  * **Residual token**：(r\in\mathbb{R}^F)（query）
  * **Expert tokens**：VQ 原型 (c) 或 RVQ ((k_1,k_2))（群）
  * **Atom tokens**：(d_{(\text{group},k)}=\hat H_{\text{group}}\odot W_k)（keys/values）
  * **Support mask**：(\mathcal{S}_{\text{union}})（without-replacement）

* **Prompt（上下文）**
  [
  \mathbf{Prompt}=(D,\ r_0=y,\ S_0)
  ]
  使用者只需提供 (y)（與可選前綴 (S_0)）；(D) 由 VQ/RVQ/網格**凍結**。**權重不更新**，行為由 prompt 決定——這就是 **ICL**。

---

## E) 什麼算「已經不是 Transformer」？

* 如果完全不用注意力、沒有任何 Q/K 嵌入，純粹在詞典上做傳統 OMP/LS，那就**不是** Transformer。
* 我們當前的版本仍有 **Q/K 打分（或其等價 (D^\top r)）**、**稀疏選擇**、**殘差式迭代**，且可升級到**多頭/FFN/LayerNorm**，因此**仍在 Transformer 家族內**，只是不追求通用序列建模，而是**為稀疏反演任務特化**。

---

## F) 一句話收束

* 現架構 = **單 query cross-attention + 稀疏路由 + 殘差迭代** 的**特化 Transformer**；
* 與 ICL 論文**仍對齊**（**attention ≈ similarity**、**forward ≈ GD/OLS**、**iterative inference**）；
* 想要更「像」主流 Transformer，只要：**加多頭/FFN/Pre-Norm**、把離散路由換成**可微稀疏**，與**GD-unroll**訓練，即可達成 **emergent-ICL** 的展示，同時保留你要的 **ICL（no-update、context-driven）** 與 **CS（固定線性 + 稀疏可恢復）** 本質。
