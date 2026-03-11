超實務版說明：你的訓練資料的「方向角度」來自**不同角度解析度**（1° / 2° / 3° / 5° …）。要讓 **Routed attn soft-OMP（no solver）** 在這種混雜資料上可訓練、可部署，而且**仍保留壓縮感知（CS）本質**與 **ICL** 語法（no-update, context-driven），有三條穩健路線。我先給**總結決策表**，再把**訓練**與**部署推論**的操作細節（含 dim/shape）拆開說清楚，並把**關鍵英語**用粗體標示。

---

## 0) 快速決策（三路線）

| 路線                              | 動機               | 對雜角度資料的處理                                                           | CS 本質（固定線性 + 稀疏 + 可恢復）        | 部署難度  |
| ------------------------------- | ---------------- | ------------------------------------------------------------------- | ----------------------------- | ----- |
| **A. Canonical Grid**（統一網格）     | 最保守；易落地          | 先把各來源 (H_\theta) **重取樣到統一角度網格** (\Theta^*)（如 2°）                    | ✅（固定 (D^*)；OMP/soft-OMP 原汁原味） | ★★☆☆☆ |
| **B. VQ 原型碼本**（Expert = 原型 (c)） | 不需知道真實 (H)；抗資料不齊 | **不看原始網格**，把所有 (H) **聚類**成 (E) 個原型；字典 (d_{(c,k)}=\hat H_c\odot W_k) | ✅（固定 (D)，量化誤差可控；加相干正則）        | ★★★☆☆ |
| **C. RVQ（殘差碼本，2–3 階）**          | 解 off-grid 誤差；細緻 | **分兩階**：先粗原型（center），再在其簇內選**殘差碼**（每階 Top-1）形成 (\hat H)             | ✅/⚠️（先定碼字→固定 (D) 即可；需控制相干）    | ★★★★☆ |

> **建議**：起手先用 **B（VQ 原型）**；若 **off-grid 誤差**明顯，再升級到 **C（2–3 階 RVQ、每階 Top-1）**。A（Canonical Grid）最簡，但要能穩定、物理一致地**角度插值/再取樣**。

---

## 1) 訓練資料「不同角度解析度」時，怎麼**訓練**？

### 1.1 前置統一（頻率軸與幅度）

* **頻率維度 (F)**（STFT bin）先統一：
  讓所有 (H_\theta\in\mathbb R^F) 與 (W_k\in\mathbb R^F) 的 (F) 相同（必要時對 (H) 做頻率內插 / 對齊）。
* **單位化**：每個原子 (d_{(\cdot,\cdot)}) 做 (\ell_2)-norm 單位化（**“unit-norm atoms”**），讓點積像「相關性」。

**shape**：

* (H_\theta \in \mathbb R^F)（或 (\mathbb C^F)）、(W_k\in\mathbb R^F)；
* (d_{(\theta,k)}=H_\theta\odot W_k\in\mathbb R^F)；
* (D\in\mathbb R^{F\times P})；(P=T_\theta\cdot M)。

---

### 1.2 三種學習路線（如何面對不同角度網格）

#### 路線 A｜**Canonical Grid**（統一角度網格 (\Theta^*)）

1. **角度重取樣（angular resampling）**
   把各資料來源的 (H_{{\theta}}) 依當時量測角度映射到統一網格 (\Theta^*)（例如 2° 關節）。

   * 線性或樣條插值（對每個頻帶 (f) 分別插）。
   * 若原始網格稀疏（5°），可做**鄰近角度增強**（附近 ±1–2° 的小擾動，**“angular jitter”**）。
2. **字典固定**：
   以 (\Theta^*) 與 (W) 組 (D^*=[H^*_\theta\odot W_k])。
3. **老師支撐（teacher support）**：
   對每筆 ((y,D^*)) 用 OMP/soft-OMP 得到真實支撐（群/原子）當監督（或純重建自監督）。
4. **訓練**（可選）：學 (P_R,P_D,\tau,\eta) 等小量參數；**Loss**：

   * **Reconstruction** (|y-\hat y|^2)（或 IS-Div）
   * **Monotonic residual** (\sum_t\max(0,|r_t|-|r_{t-1}|))
   * **Orthogonality** (\sum_{j\in S}|r^\top d_j|)
   * **Group/atom CE**（用 teacher 支撐，**“support-matching”**）
5. **資料不平衡修正**：
   1° 資料比 5° 多 → **“importance weighting”**（角度分佈反比權重）、或 **per-angle uniform sampling**。

> **優點**：語義最清晰；**CS 本質完整保留**（固定 (D^*)）。
> **缺點**：角度插值品質、尤其在頻率相位上要謹慎（可先轉 (|H|, \angle H) 兩通道再平滑）。

---

#### 路線 B｜**VQ 原型碼本**（**Expert = 原型 (c)**）

1. **碼本學習**（不看原始網格）：
   對所有收集的 (H)（無論 1°/2°/5°）做 **K-means/VQ** 得 ({\hat H_c}_{c=1}^E)。

   * **“mutual-coherence penalty”**：碼本學習時加入原子互相干正則，鼓勵原型分離（角距 ≥ (\delta)）。
2. **字典構造**：
   (d_{(c,k)}=\hat H_c\odot W_k)，(D\in\mathbb R^{F\times (E\cdot M)})。
3. **老師支撐**：
   用 (D) 對 ((y)) 跑 OMP/soft-OMP，得到群/原子支撐。
4. **訓練**：與 A 類似；額外可訓練路由中的投影 (P_R) 以**強化 coarse routing**（**“coarse-to-fine routing”**）。
5. **DoA 標定**（可選）：學 (p(\theta\mid c)) 或直接把 (c) 映射成角度中心，最後以 (A_c=\sum_k|x_{c,k}|) 做 barycenter。

> **優點**：**不需真實 (H)**、不怕角度網格不齊；**ICL** 很自然（路由由 (r) 決定）。
> **注意**：**碼本大小 (E)** 與**互相干**要一併調；過多原型會抬高相干、傷支撐恢復。

---

#### 路線 C｜**RVQ（2–3 階）**（**Expert=(k₁, k₂)** 的層級路由）

1. **階 1（coarse center）**：把方向空間分成 (K_1) 大簇；學 ({C^{(1)}})。
2. **階 2（residual）**：在每個簇內學 ({C^{(2)}}) 作細緻殘差；**每階 Top-1** 保稀疏與解釋。
   [
   \hat H = C^{(1)}*{k_1} + C^{(2)}*{k_2}
   ]
3. **字典**：(d_{(k_1,k_2,k)}=\hat H\odot W_k)。
4. **推論期要**：「**先決定碼字（路由）→固定 (D)**→再解稀疏 (x)」（**避免變成自適應字典**）。
5. **訓練**：與 VQ 相同；多加每階的**互相干正則**與**單位化**。

> **優點**：**off-grid 誤差小**（方向更連續）；
> **注意**：小階數（2 或 3）；嚴守「先定碼字再解 (x)」；不然 CS 固定字典假設會被破壞。

---

### 1.3 監督與損失小技巧（混合角度資料常用）

* **von-Mises / angular label smoothing**：群支撐 supervision 用 von-Mises 核平滑，使 1°/5° 數據對齊（**“angular smoothing”**）。
* **curriculum**：先用粗網格/少原型，逐步加密或加階（**“coarse-to-fine curriculum”**）。
* **balanced sampling**：按角度均勻抽樣 mini-batch；或 loss 加權（**“importance weighting by angle histogram”**）。

---

## 2) 到時候「**部署（inference）**」要怎麼跑？

### 2.1 共同前置（與訓練相同）

* 把輸入單窗 STFT (y\in\mathbb R^F) 做與訓練一致的**頻率規範/白化/單位化**。
* 使用**凍結**的 (W) 與（A）(D^*) 或（B）VQ 原型 ({\hat H_c}) 或（C）RVQ 碼本 ({C^{(m)}}) 構成的字典。

### 2.2 路由式 soft-OMP 推論（no solver；forward≈GD）

> **層/loop＝步**，每步：coarse（選群）→ fine（選群內原子）→ 更新 → 殘差

1. **Coarse routing**（群級）

   * **A（Canonical）** / **B（VQ）**：以 (\alpha_\text{group}=\max_{j\in \text{group}}|(D^\top r)_j|) 選 **Top-(K_e)** 群（方向/原型）。
   * **C（RVQ）**：先在第 1 階 center 做 **Top-(K_e)**，再在其對應 (\theta) 聯集內 fine。

2. **Fine selection**（群內）

   * 在每個被選群的索引集內取 **Top-(L)** 原子，權重 (w=\mathrm{softmax}(|s|/\tau))（可用 **entmax/sparsemax**）。

3. **更新 / 殘差**

   * (x[idx] \leftarrow x[idx] + \eta,(w\odot (D^\top r)[idx]))；
   * (r\leftarrow y - D x)；
   * **union 支撐** 累積，最多 (S\cdot K_e\cdot L) 個非零（選擇稀疏）。

4. **DoA 讀出**

   * (A_\text{group}=\sum_{\text{atom in group}}|x|)；
   * **離散**：(\hat\theta=\arg\max A)；**連續**：以群中心與 (A) 做 **barycenter**。

**複雜度**（單窗、每步）：

* **A/B（VQ）**：(O\big(F\cdot(T_\theta + K_e L)\big))
* **C（RVQ）**：(O\big(F\cdot(K_1 + K_e L)\big))（(K_1) 為 coarse center 數）

> 相較全掃 (O(F\cdot P)) 明顯更小；也更符合「**coarse-to-fine**」。

---

## 3) dim / shape（部署期核心張量）

* (y\in\mathbb R^F)、(r\in\mathbb R^F)、(x\in\mathbb R^P)；
* **A**：(D^*\in\mathbb R^{F\times (T_\theta M)})；
* **B**：(D_{\text{VQ}}\in\mathbb R^{F\times (E M)})；
* **C**：實作時只實例化被路由到的 ((k_1,k_2)) 分支上的少量原子，避免建滿 (F\times (K_1 K_2 M))。

---

## 4) 把「不同角度網格」變成**優勢**（策略與因果鏈）

* **更密的來源（1°）** 提供**高解析**樣本，驅動 **碼本/中心** 的精度；
* **較稀疏的來源（5°）** 提供**穩健的粗形狀**，降低相干與過擬合風險；
* **訓練時均衡**（balanced sampling + angular smoothing）→ **路由器更穩**；
* **部署時 coarse→fine**（VQ/RVQ）→ **off-grid 誤差小、計算可控**；
* **ICL 語法**：完全由 ((D,r_0)) 的上下文決定路由與更新，**no weight updates**。

---

### 5) 小抉擇提醒

* 若**插值品質可控** → **A**（Canonical Grid）最簡；
* 若**不知道 (H)** 或資料來源多雜 → **B**（VQ 原型）最穩；
* 若**DoA 解析度要求高 / off-grid 明顯** → **C**（2–3 階 RVQ），但務必遵守「**先定碼字→固定字典**」以保 CS 本質。

---

如果你願意，我可以把你現有的 routed attn soft-OMP 程式**加上 VQ / RVQ 資料管線**（含 K-means / two-stage codebook、互相干正則、balanced sampling、angular smoothing），並生成**ID/OOD 殘差曲線、Group/Atom F1、DoA 誤差、FLOPs**的一頁式報表；同時把這份 **shape/物理意義表**放在附錄，方便直接放入 methods。
