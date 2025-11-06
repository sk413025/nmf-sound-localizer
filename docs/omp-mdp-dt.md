下面我用**第一性原理**把「**Soft‑OMP for LDV DoA**」和「**Decision Transformer（DT）**」之間的關係拆成 5 個層次：**(A) 背景基礎** → **(B) 把 OMP/Soft‑OMP 鑄造成 MDP** → **(C) 何以能套用 Decision Transformer** → **(D) Token 設計能否共用（設計範式）** → **(E) 何時值得、何時不值得**。每一步都標出**關鍵英語詞**（**粗體**）並給出直白釋義；關鍵數據與設計細節以你的投影片為準並標註。

---

## (A) 背景基礎（最小知識集）

1. **OMP/Soft‑OMP 的本質**

* **OMP（Orthogonal Matching Pursuit）**：逐步從字典 (D) 中選原子，更新殘差 (r)，貪婪逼近稀疏解（**greedy sparse approximation**）。經典結果說明在低互相干與稀疏條件下可恢復正確支撐集。([tropp.caltech.edu](https://tropp.caltech.edu/papers/TG07-Signal-Recovery-preprint.pdf?utm_source=chatgpt.com "Signal Recovery from Random Measurements via Orthogonal ..."))
* 你的 **Soft‑OMP**：把 OMP 的「相關性打分」變成**可微分的注意力打分**，並加入**兩級分層路由（expert→atom）**與**多任務損失**（重建+單調性+分類），字典以 (D=H\\odot W) 物理結構化設計降低**互相干（**mutual coherence**：原子間相似度）**，最終在 37 角、K=6 上達到 100% 測試準確率（以降低 (\\mu) 爲關鍵）。

2. **MDP 與離線 RL**

* **MDP（Markov Decision Process）**：由 **state（狀態）**、**action（動作）**、**transition（轉移）**、**reward（獎勵）** 組成；**Markov property** = 下一步只依賴當前狀態與動作。
* **Offline RL（離線強化學習）**：只用**既有軌跡（trajectories）**學策略，不再與環境互動；核心風險是**OOD（out‑of‑distribution）動作**導致價值高估，如 **Conservative Q‑Learning (CQL)** 以「保守」估值抑制樂觀偏差。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"))

3. **Decision Transformer（DT）**

* **核心想法**：把 RL **視為序列建模**。用**因果遮罩（causal masking）**的 Transformer，條件化在 **return‑to‑go（RTG，尚未獲得的未來回報）**、過去 **states** 與 **actions** 上，**自回歸**輸出下一個 **action**。本質上是「**條件序列建模**」而不是學 Q 函數或做 policy gradient，特別適合**離線資料**。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via ..."))
* **同一脈絡的工作**：**Trajectory Transformer** 把整條 **(s,a,r)** 轉成序列，配合 **beam search** 做規劃；**Diffuser** 用**擴散模型**直接在「軌跡空間」規劃。([arXiv](https://arxiv.org/abs/2106.02039?utm_source=chatgpt.com "Offline Reinforcement Learning as One Big Sequence ..."))

---

## (B) 把 OMP／Soft‑OMP 鑄造成一個 MDP（可行的最小表述）

> 目標：釐清「如果把（Soft‑）OMP 看成 **MDP**，會得到什麼？」——這決定了能否直接使用 **DT**。

* **State (s\_t)**（**詞義**：當前決策所需的全部資訊）
  建議包含：
  1. **殘差 (r\_t)**（連續向量；你的實作用連續投影做 **Query**），
  2. **已選支撐集 (S\_t)**（集合；可用 one‑hot mask 或集合嵌入），
  3. **字典 (D)**（固定但可用**Key 嵌入**表示每個 atom；你已用線性投影把 (D) 變成 **Keys**），
  4. **步數 (t)** 與**剩餘預算 (K-t)**（**budget**），
     5)（視任務）**任務提示**：例如目標類別或 RTG。
     這樣定義時，**Markov property** 成立：(s\_{t+1}) 只由 (s\_t) 和 (a\_t) 決定（因為下一個殘差 (r\_{t+1}) 是把 (a\_t) 加入支撐後做正交回歸得到）。
* **Action (a\_t)**：從**尚未選過**的 atom（或 expert→atom 層級）中選一個（或一組）加入 (S\_t)。
  你已經用**兩級分層路由**把 296 候選壓到「先 37 expert→再 8 atom」的指標選擇，這恰好是一個**層級動作空間（hierarchical action space）**。
* **Transition**：執行 (a\_t) 後，更新係數、重算殘差 (r\_{t+1}=Y-Dc\_{t+1})；這是**確定性轉移（deterministic transition）**。
* **Reward (r\_t)**（多種設計）：
  * **重建式**：(\\Delta|r\_t|\_2^2) 的下降量（**reward shaping**：鼓勵每一步最大化殘差下降）。
  * **任務式**：分類 logit margin 的提升、或最終是否把 DoA 分對（終局回報）。
  * **結構式**：懲罰高互相干選擇，或鼓勵**monotonicity**（早選更關鍵）。你已有**單調性損失**，可映射成逐步獎勵。

> **因果鏈**：把支撐選擇問題表述成 MDP → 可以把「長期（到 (K) 步）總體表現」轉為回報設計 → 任何**離線 RL**或**序列模型**（如 DT）都能作用在這個 MDP 上。

---

## (C) 能不能直接用 Decision Transformer？——「三種使用場景」比較

> **關鍵詞：****conditional sequence modeling**（*以條件生成動作的序列模型*），**return‑to‑go (RTG)**（*未來剩餘回報*），**offline trajectories**（*先驗收集的軌跡*）。

### 場景 C‑1：**以 Soft‑OMP 當行為示範（imitation‑style DT）**

* **動機**：不改變你的訓練目標；收集你現有 **Soft‑OMP** 在各種 (Y)、各種字典/噪聲/邊界條件下的**決策軌跡** ((r\_1,a\_1,r\_2,a\_2,\\ldots,r\_K))。
* **做法**：把每條軌跡的**return（例如最終 (-|r\_K|^2) + 分類正確獎勵）**寫成 **RTG**，用 DT 做**條件行為複現**。
* **效果**（推論）：DT ≈ 在**更大的分佈**上 distill 你的策略，並可**條件化**不同 RTG（比如「更快收斂 vs 更準確」的 trade‑off）。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via ..."))

### 場景 C‑2：**以 DT 作為新策略學習器（policy learner）**

* **動機**：希望在**不同的任務回報**下自動學到「挑 atom 的策略」，甚至超越現有 Soft‑OMP 的局部貪婪。
* **做法**：同樣離線蒐集軌跡，但回報非僅重建，可加入**最終分類、罰互相干、耗時/能耗**等多目標（**multi‑objective RTG**）。
* **風險**：典型的**offline RL OOD 風險**——DT雖不用 Q，但**資料覆蓋不到的狀態‑動作**區域仍可能導致性能不穩。需要資料增豐或以**保守策略**、**篩選示範**緩解（CQL 類洞見仍適用於資料治理）。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"))

### 場景 C‑3：**把 DT 當規劃器（planner）**

* **動機**：近似 **Trajectory Transformer**/**Diffuser** 的用法：用序列模型或擴散模型在「**支撐選擇的序列空間**」規劃，配合 beam search/denoising 尋找高回報序列。
* **適配度**：你的決策地平線很短（(K\\approx 6)），序列規劃成本低，可嘗試**beam search** 在「可行 atom 序列」上找更好的組合。([arXiv](https://arxiv.org/abs/2106.02039?utm_source=chatgpt.com "Offline Reinforcement Learning as One Big Sequence ..."))

> **小結（不下結論，僅列條件）**：若你已有大量**離線軌跡**、想用**條件化回報**調節策略或做**跨場景遷移**，DT 是自然選項；若資料稀少、且你的 **Soft‑OMP** 已明確地利用了**物理結構與分層歸一化**，DT 帶來的淨效益取決於能否從更廣的軌跡分佈與回報設計中**擷取額外結構**。

---

## (D) Token 設計能否共用？——三個「可共用」的設計範式

> 你的系統已經有**連續投影 tokens**（**Query**=由 (r) 投影；**Keys**=由 (D) 投影）與**兩級路由**（expert→atom）。我們把 DT 需要的 **(RTG, state, action)** token 語法和你現有設計對齊。

### 範式 D‑1：**Residual‑as‑State + Candidate‑as‑Memory（指針式決策）**

* **核心英語**：**pointer head**（*用注意力把輸出指到「候選集合」的某一個索引*）。
* **做法**：
  1. 用你現有的**殘差投影**作 **state token**；
  2. 把所有（或候選子集）的 **atom keys** 當「記憶槽 tokens」；
  3. 用 **pointer head** 在這些 key 上做注意力分佈，**輸出動作 = 選哪個 atom**（等價於你當前的軟選擇，但由 DT 的解碼器觸發）。
  4. 把 **(RTG, state, past action)** 以 DT 的順序串接：([ \\textbf{RTG}\_t, \\textbf{state}*t, \\textbf{action}*{t-1} ] \\to \\textbf{action}\_t)。
* **因果鏈**：**共用連續投影** → state 與候選的幾何仍被保留 → DT 只負責**序列條件化與全局規劃**。
* **對應文獻**：這與 **Pointer Networks** 的「指向輸入位置」相似，只是你的輸入是**物理字典的嵌入**。([NeurIPS Papers](https://papers.nips.cc/paper/5866-pointer-networks?utm_source=chatgpt.com "Pointer Networks"))

### 範式 D‑2：**Hierarchical Tokens（先 expert 後 atom）**

* **核心英語**：**hierarchical action space**（*先在粗層級決定、再在細層級精選*）。
* **做法**：
  1. 以 37 個 **expert tokens**（由 (H) 生成）先做一次 DT 決策，得到 **Top‑k experts**；
  2. 僅對被選 experts 展開對應 **atom tokens**（由 (W) 與 expert 組合）做第二次決策。
* **因果鏈**：這直接**重用你的二階路由歸一化**（先去除專家內 bias、聚合 L2、再專家間正規化），把**計算/比較**聚焦在具物理意義的子空間。

### 範式 D‑3：**MDP‑aware 序列欄位（DT 的欄位設計）**

* **必要欄位**（皆可**連續嵌入**）：
  * **RTG token**（**return‑to‑go**；*控制想要的未來總回報*）
  * **State token**：([\\hat r\_t, \\text{mask}(S\_t), t, K-t])（*含殘差、支撐遮罩與步序/預算*）
  * **Action token**：上一步所選 atom 的**索引嵌入**；
  * **Dictionary tokens**：可整體池化成「上下文」，或以**記憶槽**形式供 pointer 使用；
  * **Constraint token**（選配）：例如**目標精度/延遲/能耗**等條件化需求。
* **因果鏈**：把**MDP 所需資訊**完整結構化為 tokens → DT 能在**條件化序列**中學到可控行爲。

> **重點**：上述三個範式都**共用**你原本「**連續投影 token**（geometry‑preserving）」與「**分層路由**」這兩個歸納偏置； DT 僅把**序列條件化與 RTG 控制**接上去。

---

## (E) 何時值得、何時不值得 —— 用「共同因子」來判斷

> **共同因子（共通本質）**
>
> 1. **Autoregressive selection**（*逐步增長支撐的自回歸選擇*）
> 2. **Budget‑constrained planning**（*固定 (K) 的規劃*）
> 3. **Structure‑aware tokens**（*以物理字典結構化的嵌入*）
> 4. **Compatibility scoring via attention**（*用注意力做相容性打分*）
> 5. **Offline trajectories**（*可批量產生的離線決策序列*）

用這 5 個共同因子來看你的問題：

* **如果**你希望在**不同任務權衡**（重建 vs 分類、準確 vs 延遲/能耗）之間**可控**，並能匯集大量**跨場景的離線軌跡**（模擬/實驗都可），**那麼**把 OMP/Soft‑OMP 鑄成 MDP，再用 **DT 的 RTG** 來「撥盤」是自然的延伸（因為**autoregressive selection + budget** 與 **conditional sequence modeling** 本來就對齊）。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via ..."))
* **如果**你的資料量仍像目前（每角 3 例，總 111）且主力證據在於**物理結構 + 分層歸一化**帶來的**可解釋性與穩定性**，**那麼**引入 DT 的增益很可能受限（因為**短地平線 K=6**、**確定性轉移**、**已強的歸納偏置**讓「序列建模優勢」難以完全施展；離線 RL 的 OOD 風險也需要額外治理）。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"))

---

## 實作藍圖（一步步，從小到大）

> **英語詞提醒**：**ablation**（*消融實驗*）、**calibration**（*機率校準*）、**OOD**（*分佈外*）。

1. **生成離線軌跡資料庫**（minimal）

* 對多種條件（不同 (H, W)、噪聲、材料、邊界、角度密度）跑 **Soft‑OMP** 與若干 baseline（flat OMP、MUSIC/ESPRIT/SBL）得到\*\*(RTG, state, action)\*\* 序列。

2. **DT‑Pointer 原型（範式 D‑1）**

* **State token** = 你的殘差投影；**Dictionary tokens** = 你的 atom keys；**Action** 由 pointer head 在 296（或二階 61）候選上輸出。
* **RTG**：用 (-|r\_K|^2 + \\lambda\\cdot \\text{Acc}) 或「(+) 正確、(-) 錯誤」。
* **Ablation**：無 RTG（純模仿）、帶 RTG；有/無二階路由；有/無你現有**三步正規化**。([NeurIPS Papers](https://papers.nips.cc/paper/5866-pointer-networks?utm_source=chatgpt.com "Pointer Networks"))

3. **規劃版（範式 C‑3）**

* 訓練 **Trajectory Transformer** 或 **DT**；**推論時**在動作序列上做 **beam search**（深度 (K) 很小，計算可控），比較是否能在相同 (K) 下取得更低殘差/更高準確或**更好校準**。([arXiv](https://arxiv.org/abs/2106.02039?utm_source=chatgpt.com "Offline Reinforcement Learning as One Big Sequence ..."))

4. **評估指標**

* **角度 MAE / Top‑1 準確**、**殘差曲線**、**ECE**（機率校準）、**延遲/能耗**、**對 (\\mu)** 的敏感度、**OOD**（新材料/新角度間隔）。
* **失敗案例可視化**：DT 與 Soft‑OMP 的選擇軌跡差異（每步選了哪個 expert/atom）。

---

## 針對你的三個提問，逐一用「因果鏈」回覆

### Q1.「這篇研究跟 Decision Transformer 有什麼關聯？」

* **共同因子**：兩者都在做 **autoregressive selection**；都可用**attention**來計算**compatibility score**；你的方法已有**連續投影 tokens**與**分層路由**這些強歸納偏置。
* **因果鏈**：把 OMP 鑄成 MDP → 用軌跡序列學策略/規劃 → **DT 作為條件序列模型**恰好可利用 **RTG** 調控策略行為。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via ..."))

### Q2.「把（Soft‑）OMP 看成 MDP，就能把它當 RL 算法，然後用 DT 嗎？」

* **可行路徑**：是的，**在 MDP 表述清楚**（state/action/transition/reward）且**有離線軌跡**時，DT 可以直接學**條件政策**或做**規劃**。
* **前置條件**：回報設計需與目標對齊（只重建 vs 加分類/互相干/能耗）；資料需涵蓋足夠多樣的 (r,S,D) 組合，以減少 **OOD**。
* **效果變數**：地平線短（(K=6)）與轉移確定 → DT 的長期 credit assignment 優勢較小；但 **RTG** 帶來的**可控性**與**跨場景蒐集**帶來的**泛化**仍可能是價值來源。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"))

### Q3.「Token design 策略可以共用嗎？」

* **可以以「範式 D‑1/D‑2/D‑3」共用**：沿用你**連續投影**生成 **state/keys**，再按 DT 語法補上 **RTG / step / budget / mask** 等 **MDP‑aware tokens**；動作以**pointer**選 atom，或先選 **expert** 再**atom**。
* **因果鏈**：共用連續投影（保持**geometry**）→ 共享兩級路由（保留物理結構）→ DT 只加上 **序列條件化與 RTG 控制**。([NeurIPS Papers](https://papers.nips.cc/paper/5866-pointer-networks?utm_source=chatgpt.com "Pointer Networks"))

---

## 參考與對照來源

* **Decision Transformer**（把 RL 當序列建模，條件在 RTG / states / actions 上）：NeurIPS 2021 與對應預印本。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via ..."))
* **Trajectory Transformer**（序列模型 + beam search 規劃）。([arXiv](https://arxiv.org/abs/2106.02039?utm_source=chatgpt.com "Offline Reinforcement Learning as One Big Sequence ..."))
* **Diffuser**（擴散式軌跡規劃）。([arXiv](https://arxiv.org/abs/2205.09991?utm_source=chatgpt.com "Planning with Diffusion for Flexible Behavior Synthesis"))
* **CQL**（離線 RL 的保守估值以抑制 OOD 風險）。([NeurIPS Proceedings](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"))
* **Pointer Networks**（指向候選集合的注意力決策，類比 atom 選擇）。([NeurIPS Papers](https://papers.nips.cc/paper/5866-pointer-networks?utm_source=chatgpt.com "Pointer Networks"))
* 你的投影片（Soft‑OMP 架構、分層路由、token 投影、實驗與消融）。

---

### 一句話把全局勾起來（**不作結論，只描繪路徑**）

> **"If we cast sparse support selection as a **budget‑constrained MDP**, then **Decision Transformer** becomes a **conditional sequence model** over support‑selection trajectories. Because your method already provides **structure‑aware tokens** and **hierarchical routing**, the two can share tokenization and differ mainly in whether we learn from **offline trajectories** with **RTG control** or via **end‑to‑end supervised routing**."**
>

---

# 分析：設計經典 OMP 版本產生專家軌跡

## ✅ 可行性結論：**完全可行！**

實際上，**現有程式碼已經支援這個功能**。讓我詳細說明：

## 1. 現狀分析

### 當前軌跡生成器支援兩種教師模式

從 [offline_dt_dataset.py](doa_rl/trajectories/offline_dt_dataset.py) 的程式碼：

```python
# Line 529-537
for t in range(args.K):
    # Hierarchical pick
    if args.teacher == 'g':  # ← 經典 OMP (|g| 能量)
        e, m, j, energy_e_max, a_score = hierarchical_pick_g(D, r, E=E, M=M)
    else:  # args.teacher == 'qk'
        assert qk_model is not None
        with torch.no_grad():
            qk_expert, qk_atoms = _qk_scores_with_config(qk_model, D, r)
            e = int(torch.argmax(qk_expert).item())
            m = int(torch.argmax(qk_atoms[e].abs()).item())
            j = e * M + m
    S.append(j)
```

## 2. 經典 OMP 算法詳解

### 已經實現的 `hierarchical_pick_g` 函數

```python
def hierarchical_pick_g(D: torch.Tensor, r: torch.Tensor, E: int, M: int):
    """
    經典 OMP 的階層式選擇策略

    輸入:
        D: (F, P) 字典，P = E × M
        r: (F,) 當前殘差
        E: 專家數量 (37)
        M: 每個專家的原子數 (8)

    輸出:
        e: 選擇的專家索引 (0-36)
        m: 選擇的原子索引 (0-7)
        j: 字典索引 j = e*M + m (0-295)
        energy_e_max: 該專家的能量總和
        a_score: 該原子的能量分數
    """
    # Step 1: 計算所有原子與殘差的相關性
    g = D.T @ r  # (P,) - 物理相關性 (內積)

    # Step 2: 重塑為階層式結構
    g_em = g.view(E, M)  # (E, M)

    # Step 3: 專家級聚合 (L1 範數)
    energy_e = g_em.abs().sum(dim=1)  # (E,)
    e = argmax(energy_e)

    # Step 4: 原子級選擇
    a_scores = g_em[e, :].abs()  # (M,)
    m = argmax(a_scores)

    j = e * M + m
    return e, m, j, energy_e[e], a_scores[m]
```

## 3. 與 Transformer Routed OMP 的接口對比

| 維度 | Transformer Routed OMP | 經典 OMP (hierarchical_pick_g) |
|------|----------------------|-------------------------------|
| **輸入** | `(D, r, E, M)` | `(D, r, E, M)` ✅ **相同** |
| **輸出** | `(e, m, j)` | `(e, m, j, ...)` ✅ **相同** |
| **選擇策略** | QK attention scores | \|g\| correlation scores |
| **專家聚合** | L2: `√(Σ|qk|²)` | L1: `Σ|g|` |
| **原子選擇** | `argmax(qk_atoms[e])` | `argmax(g[e])` |
| **後續步驟** | 正交投影、殘差更新 | 正交投影、殘差更新 ✅ **相同** |

**關鍵發現**：兩者的輸入輸出接口**完全一致**！

## 4. 產生經典 OMP 軌跡的方法

### 方法 1：使用現有程式碼 (最簡單)

```bash
# 當前 commit 使用 QK 教師
PYTHONPATH=$(pwd) python -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher qk \
  --qk_ckpt results/exp_H_qk_encoder_on_atom_d128_20251026_233228/model_best.pth \
  --out_dir results/dt_traj_kmeans_v3 \
  ...

# 切換為經典 OMP 教師 (只需改一個參數！)
PYTHONPATH=$(pwd) python -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher g \
  --out_dir results/dt_traj_classic_omp \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --atom_reduce_mode kmeans \
  --K 6
```

**生成的軌跡格式完全相同**：
- `trajectories.jsonl`: 每步記錄 `(expert, atom, dict_index, resid_sq, rtg_resid, rtg_acc, ...)`
- `manifest.json`: 包含 `teacher: "g"` 標記
- `numeric_diagnostics.jsonl`: 統計資訊

### 方法 2：如果需要更細緻的控制

可以修改 `hierarchical_pick_g` 來模仿 Transformer 的行為模式：

```python
def hierarchical_pick_g_l2_variant(D: torch.Tensor, r: torch.Tensor, E: int, M: int):
    """
    經典 OMP 變體：使用 L2 聚合 (模仿 Transformer expert_agg='l2')
    """
    g = D.T @ r  # (P,)
    g_em = g.view(E, M)  # (E, M)

    # 使用 L2 範數聚合 (與 Transformer 一致)
    energy_e = torch.sqrt((g_em.abs() ** 2).sum(dim=1) + 1e-12)  # (E,)
    e = int(torch.argmax(energy_e).item())

    a_scores = g_em[e, :].abs()  # (M,)
    m = int(torch.argmax(a_scores).item())

    j = e * M + m
    return e, m, j, float(energy_e[e].item()), float(a_scores[m].item())
```

## 5. 實驗設計建議

### 對比實驗：經典 OMP vs Transformer OMP

```bash
# 實驗 1: 經典 OMP 軌跡 + DT-min 訓練
python doa_rl/trajectories/offline_dt_dataset.py \
  --teacher g --out_dir results/dt_traj_classic_omp

python scripts/dt_pointer_ldv.py \
  --traj_dir results/dt_traj_classic_omp \
  --out_dir results/dt_min_classic_omp \
  --epochs 120 --batch_size 8

# 實驗 2: Transformer OMP 軌跡 + DT-min 訓練 (baseline)
python scripts/dt_pointer_ldv.py \
  --traj_dir results/dt_traj_kmeans_v3 \
  --out_dir results/dt_min_kmeans_v3 \
  --epochs 120 --batch_size 8
```

### 預期對比指標

| 指標 | 經典 OMP 軌跡 | Transformer OMP 軌跡 |
|------|--------------|---------------------|
| 教師 t=0 準確度 | ~83.8% | ~94.6% |
| 軌跡殘差下降 | 較慢 | 較快 |
| DT-min 學習後 t=0 準確度 | ? | ~54.1% |
| DT-min 收斂速度 | ? | 120 epochs |
| 知識蒸餾效果 | 較差 (教師弱) | 較好 (教師強) |

## 6. 關鍵差異分析

### 專家選擇策略對比

```python
# 經典 OMP (L1 聚合)
g_em = (D.T @ r).view(E, M)        # (E, M)
energy_e = g_em.abs().sum(dim=1)   # (E,) - L1 範數
e_star = argmax(energy_e)
# 特點: 專家能量 = 該專家所有原子的絕對值之和

# Transformer OMP (L2 聚合)
qk_em = QK_attention(r, D).view(E, M)
energy_e = sqrt((qk_em.abs()**2).sum(dim=1))  # (E,) - L2 範數
e_star = argmax(energy_e)
# 特點: 專家能量 = 該專家所有原子的能量平方和開根號
```

**物理意義**:
- **L1**: 線性累加，對異常值敏感
- **L2**: 歐式距離，對大值更敏感，有平滑效果

### 軌跡質量預測

| 軌跡特徵 | 經典 OMP | Transformer OMP |
|---------|---------|-----------------|
| 初始 t=0 選擇準確度 | 83.8% | 94.6% |
| 殘差下降單調性 | 保證 (數學性質) | 近似保證 (學習策略) |
| 分類信心度 p_true | 較低 | 較高 |
| RTG 達標率 | 較低 | 較高 |
| 策略多樣性 | 低 (固定策略) | 高 (情境依賴) |

## 7. 程式碼修改指南 (如果需要)

### 選項 A: 零修改 (直接使用)

```bash
# 只需修改命令列參數
--teacher g  # 從 'qk' 改為 'g'
```

### 選項 B: 微調聚合策略

在 `offline_dt_dataset.py` 中添加：

```python
# 在 Line 201 後添加
def hierarchical_pick_g_l2(D: torch.Tensor, r: torch.Tensor, E: int, M: int):
    """經典 OMP with L2 aggregation (match Transformer config)"""
    g = (D.T @ r)
    g_em = g.view(E, M)
    # Use L2 aggregation like Transformer
    energy_e = torch.sqrt((g_em.abs() ** 2).sum(dim=1) + 1e-12)
    e = int(torch.argmax(energy_e).item())
    a_scores = g_em[e, :].abs()
    m = int(torch.argmax(a_scores).item())
    j = e * M + m
    return e, m, j, float(energy_e[e].item()), float(a_scores[m].item())

# 在 Line 529 修改
if args.teacher == 'g':
    e, m, j, _, _ = hierarchical_pick_g_l2(D, r, E=E, M=M)  # 使用 L2 版本
```

### 選項 C: 添加後處理 (模仿 Transformer 配置)

```python
def hierarchical_pick_g_with_config(D, r, E, M,
                                    score_norm_mode='std',
                                    score_center_expert=True,
                                    score_center_atoms=True):
    """經典 OMP + Transformer 風格的分數處理"""
    g = (D.T @ r)
    g_em = g.view(E, M)
    energy_e = torch.sqrt((g_em.abs() ** 2).sum(dim=1) + 1e-12)

    # 模仿 Transformer 的分數正規化
    if score_norm_mode == 'std':
        energy_e = (energy_e - energy_e.mean()) / (energy_e.std() + 1e-8)
    if score_center_expert:
        energy_e = energy_e - energy_e.mean()

    e = int(torch.argmax(energy_e).item())
    a_scores = g_em[e, :].abs()

    if score_center_atoms:
        a_scores = a_scores - a_scores.mean()

    m = int(torch.argmax(a_scores).item())
    j = e * M + m
    return e, m, j, float(energy_e[e].item()), float(a_scores[m].item())
```

## 8. 實驗假設與預測

### 假設 1: 教師質量影響 DT-min 性能上限

```
經典 OMP 軌跡 (83.8% 教師準確度)
    ↓ DT-min 學習
預測: DT-min t=0 準確度 < 54.1%
原因: 軌跡質量較差，難以學習良好策略
```

### 假設 2: DT 可能學到比教師更好的策略

```
可能性: DT-min 通過蒸餾和泛化，t=K-1 性能 > 教師
機制: 條件化 RTG 允許探索不同於教師的路徑
```

### 假設 3: 收斂速度差異

```
經典 OMP 軌跡 → 預測需要更多 epochs (>120)
Transformer 軌跡 → 120 epochs 足夠
```

## 9. 建議實驗步驟

```bash
# Step 1: 生成經典 OMP 軌跡
PYTHONPATH=$(pwd) python -u doa_rl/trajectories/offline_dt_dataset.py \
  --teacher g \
  --h_path /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --dataset_root /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --atom_reduce_mode kmeans \
  --out_dir results/dt_traj_classic_omp_g \
  --K 6 \
  2>&1 | tee results/dt_traj_classic_omp_g/run.log

# Step 2: 檢查生成的軌跡質量
cat results/dt_traj_classic_omp_g/manifest.json | python3 -m json.tool | grep -A 5 "teacher_eval"
# 預期: accuracy_t0_currentD ≈ 0.838

# Step 3: 訓練 DT-min
for i in {1..24}; do \
  PYTHONPATH=$(pwd) python -u scripts/dt_pointer_ldv.py \
    --traj_dir results/dt_traj_classic_omp_g \
    --out_dir results/dt_min_classic_omp_g \
    --epochs 5 --batch_size 8 --d_model 128 --nhead 2 --nlayers 1 \
    --distill_weight 0.7 --distill_T 1.0 --warmup_epochs 3 --device cpu \
    2>&1 | tee -a results/dt_min_classic_omp_g/run.log || break; \
done

# Step 4: 對比結果
echo "=== 經典 OMP 軌跡 ==="
tail -20 results/dt_min_classic_omp_g/run.log | grep "Angle acc"

echo "=== Transformer OMP 軌跡 ==="
tail -20 results/dt_min_kmeans_v3/run.log | grep "Angle acc"
```

## 10. 總結

| 問題 | 答案 |
|------|------|
| **是否可行？** | ✅ **完全可行**，接口已經統一 |
| **需要修改程式碼？** | ❌ **不需要**，只需改參數 `--teacher g` |
| **軌跡格式相同？** | ✅ **完全相同** |
| **能訓練 DT-min？** | ✅ **可以**，無需任何修改 |
| **性能預期？** | ⚠️ **可能較差** (教師準確度 83.8% vs 94.6%) |
| **科學價值？** | ✅ **很高**，可對比教師質量對 DT 學習的影響 |

**這是一個非常有價值的消融實驗 (Ablation Study)**，可以回答：
1. DT-min 的性能上限是否受限於教師質量？
2. 簡單的物理啟發策略能否通過 DT 學習得到提升？
3. Transformer 的額外複雜度是否必要？

推薦立即執行實驗！
