太好了—我把 **Routed attn soft-OMP (no solver)** 與 **LDV 聲源估計**整個流程中會用到的數學符號，全部補上 **dim / shape**（含物理意義與是否可複數），也把 **VQ / RVQ 原型化**與 **ICL prompt** 的張量尺寸列清楚。你可以直接當作 methods 的「符號與維度表」。

---

# 0) 索引集合與尺寸（Index sets & sizes）

* (F)：頻帶數（number of STFT frequency bins），例如 256/512。
* (T)：時間窗數（number of STFT frames）。
* (|\Theta|=T_\theta)：方向（DoA）網格數（experts 數目，亦可為原型數 (E)）。
* (|\mathcal K|=M)：語音頻譜基底數（NMF/learned bases）。
* (P = T_\theta \cdot M)：**字典原子總數**。
* (K_e)：每步選取的 **experts**（方向群）數（coarse）。
* (L)：每個被選 expert 內選取的 **atoms** 數（fine）。
* (S)（或 `steps`）：迭代步數（層/loop = 步）。

---

# 1) 量測與頻譜（Signals）

* (Y \in \mathbb C^{F\times T})（或 (\mathbb R^{F\times T})）：**LDV STFT**；單步時窗為 (y \in \mathbb C^{F})（或 (\mathbb R^{F})）。

  * 若用**實值兩通道**堆疊：(\tilde y \in \mathbb R^{2F})（([\Re(y);\Im(y)]) 或 ([|y|;\angle y])）。
* (r \in \mathbb R^{F}) / (\mathbb C^{F})：**殘差**（Residual）；初值 (r_0=y)。
* 單位：頻帶為 Hz 的離散索引；量測值常以「幅度或能量」表達（相對單位）。

---

# 2) 方向濾波與語音基底（Atoms & Dictionary）

* (H_\theta \in \mathbb R^{F}) / (\mathbb C^{F})：**方向濾波（steering frequency response）**；每一 (\theta\in\Theta)。
* (W_k \in \mathbb R^{F}_{\ge 0})（NMF）或 (\mathbb R^{F})（learned basis）：**語音頻譜基底**；每一 (k\in\mathcal K)。
* (d_{(\theta,k)} = H_\theta \odot W_k \in \mathbb R^{F}) / (\mathbb C^{F})：**字典原子**（逐頻相乘；可單位化）。
* (D = [,d_{(\theta,k)},] \in \mathbb R^{F\times P})：**字典矩陣**（欄為原子，(P=T_\theta M)）。
* (x \in \mathbb R^{P})（或 (\mathbb R_{\ge 0}^{P})）：**稀疏係數**（單窗；多窗則 (X\in\mathbb R^{P\times T})）。
* **線性前向模型**：(y \approx D,x + \varepsilon)。

---

# 3) Experts（方向群）與分片（Shards）

* `group_indices[θ]`：第 (\theta) 個 **expert**（方向群）包含的原子索引集合
  (\mathcal J_\theta={j\mid j \leftrightarrow (\theta,k),\ k=1\ldots M})。
* **coarse score**（群級相關性）：
  (\alpha_\theta = \max_{j\in\mathcal J_\theta} |(D^\top r)*j|)，
  (\alpha \in \mathbb R^{T*\theta})。

---

# 4) Attention 版本（可選）之嵌入與打分（Tokens & Projections）

> （若採純 (D^\top r) 也可；此處給與 ICL 文獻一致的「attention ≈ similarity」表法）

* (P_D \in \mathbb R^{d\times F})、(P_R \in \mathbb R^{d\times F})：**線性投影**（字典/殘差→嵌入）。
* (D_{\text{emb}} = P_D,D \in \mathbb R^{d\times P})（或 (D_{\text{emb}}^\top \in \mathbb R^{P\times d})）。
* (q = W_q,P_R,r \in \mathbb R^{d})、(K = W_k,D_{\text{emb}}^\top \in \mathbb R^{P\times d})。
* **scores**：(s = K,q / \sqrt d \in \mathbb R^{P})。
* **coarse** 可用 block pooling：(\alpha_\theta = \operatorname{pool},(s_{\mathcal J_\theta}))。

---

# 5) Fine 選擇與 soft-OMP 權重（Fine selection）

* **選擇集合**：

  * chosen **experts**：(\mathcal E_t \subset \Theta)，(|\mathcal E_t| = K_e)。
  * expert (e) 內 chosen **atoms**：(\mathcal S_{t,e} \subset \mathcal J_e)，(|\mathcal S_{t,e}| = L)。
  * 當步 union：(\mathcal S_t = \bigcup_{e\in\mathcal E_t}\mathcal S_{t,e})，(|\mathcal S_t| \le K_e L)。
* **權重**（soft-OMP）：
  (w_t = \mathrm{softmax}\Big(\frac{|s_{\mathcal S_t}|}{\tau}\Big)\in \mathbb R^{K_eL})，(\sum w_t=1)。
  （可替換 **sparsemax/α-entmax** 以產生精確 0。）

---

# 6) 更新與殘差（Update & Residual）

* **梯度式更新**（no solver；forward≈GD）：
  令 (g = D^\top r_{t-1}\in\mathbb R^{P})，僅在 (\mathcal S_t) 更新：
  [
  x_{\mathcal S_t} \leftarrow x_{\mathcal S_t} + \eta;\big( w_t \odot g_{\mathcal S_t}\big),\quad
  x_{\text{others}}\ \text{不變}.
  ]
  其中 (\eta\in\mathbb R_{+}) 為步長。
* **殘差更新**：(r_t \leftarrow y - D,x \in \mathbb R^{F})。
* **支撐集合**（union，without-replacement）：
  (\mathcal S_{\text{union}}\leftarrow\mathcal S_{\text{union}}\cup\mathcal S_t)，
  (|\mathcal S_{\text{union}}| \le S\cdot K_e L)。
* **監測**：(|r_t|*2) 單調↓；**orthogonality score** (\sum*{j\in\mathcal S_{\text{union}}}|r_t^\top d_j|)。

> 若要硬貼 **OMP**：設 (L{=}1)（或 Top-K），(\tau\to0)，並在 (\mathcal S_{\text{union}}) 達到目標稀疏度 (k) 時早停。

---

# 7) 稀疏與結構化稀疏（Sparsity）

* **選擇稀疏（(\ell_0) 上界）**：(|\mathrm{supp}(x)| \le S\cdot K_e L)。
* **群稀疏**：只少數 **experts** 被啟用（接近 **Block-OMP**）；群內再少數原子被啟用（**group→atom**）。
* **值稀疏（可選）**：GD 後加 **ISTA** 近端
  (x\leftarrow \operatorname{sign}(x)\cdot\max(|x|-\lambda\eta,0))；或 **非負/單純形**投影。

---

# 8) DoA 估計（Direction of Arrival）

* 群活躍度（方向能量）：
  (A_\theta = \sum_{k=1}^M |x_{(\theta,k)}| \in \mathbb R^{T_\theta})。
* **離散網格** DoA：(\hat\theta = \arg\max_\theta A_\theta)。
* **連續估計**（barycenter）：
  (\hat\theta = \mathrm{barycenter}\big({\theta,,A_\theta}\big)) 或用 von-Mises 平滑。

---

# 9) ICL 的 prompt 與前綴（Prompt & Prefix）

* **Prompt**：((D,\ r_0=y,\ S_0))。

  * (D \in\mathbb R^{F\times P})：由 ((H,W)) 或其學得替代（見下一節）拼成；**固定於推論期**。
  * (r_0=y \in\mathbb R^{F})：量測的該窗 STFT 向量。
  * (S_0 \subset {1,\ldots,P})：（可選）**prefix 支撐**；或用 teacher 先走 (L_{\text{prefix}}) 步產生。
* **ICL 判準**：推論 **無權重更新**；行為完全由 **context**（((D,r_0),S_0)）決定。

---

# 10) 原型碼本（VQ）版（H 不在網路；以資料學原型）

* **Codebook**：({\hat H_c}_{c=1}^{E})，(\hat H_c\in\mathbb R^{F})；(E) = 原型數。
* **Expert**：(c)（取代 (\theta)）；`group_indices[c] = { (c,k) })。
* **Atom**：(d_{(c,k)} = \hat H_c \odot W_k \in \mathbb R^{F})。
* 字典：(D \in \mathbb R^{F\times (E\cdot M)})。
* **DoA**：以 (p(\theta\mid c)) 的查表或小頭，把 (\sum_k|x_{(c,k)}|) 投回角度。

---

# 11) 多階殘差量化（RVQ）版（階數 S：2–3 階為宜）

* **第 (m) 階碼本**：(C^{(m)}\in\mathbb R^{F\times K_m})；**選擇向量** (e^{(m)}\in{0,1}^{K_m},\ |e^{(m)}|_0=1)。
* 方向響應重建：(\hat H = \sum_{m=1}^{S} C^{(m)} e^{(m)} \in \mathbb R^{F})。
* 原子：(d_{(k_1,\ldots,k_S,,k)} = \hat H \odot W_k \in \mathbb R^{F})。
* **層級路由**：每階 **Top-1**（保持稀疏與解釋性），**先定碼字** ((k_1,\dots,k_S)) → **固定 (D)** → 再解 (x)。
* 字典：(D \in \mathbb R^{F\times (K_1\cdots K_S \cdot M)})（實作時只實例化被路由的少數分支）。

---

# 12) 量測／監測向量的 shape 摘要（速查）

| 變數                  |                                   shape | 說明                     |
| ------------------- | --------------------------------------: | ---------------------- |
| (y)                 |       (\mathbb R^{F}) 或 (\mathbb C^{F}) | 單窗 LDV STFT（量測）        |
| (\tilde y)          |                        (\mathbb R^{2F}) | ([\Re;\Im]) 實值堆疊（可選）   |
| (H_\theta)          |       (\mathbb R^{F}) / (\mathbb C^{F}) | 方向濾波（每 (\theta)）       |
| (W_k)               |                 (\mathbb R^{F})（(\ge0)） | 語音頻譜基底（每 (k)）          |
| (d_{(\theta,k)})    |                         (\mathbb R^{F}) | 原子（逐頻乘）                |
| (D)                 |                 (\mathbb R^{F\times P}) | 字典矩陣，(P=T_\theta M)    |
| (x)                 |                         (\mathbb R^{P}) | 稀疏係數                   |
| (r)                 |                         (\mathbb R^{F}) | 殘差                     |
| (s=D^\top r)        |                         (\mathbb R^{P}) | 相關性／打分                 |
| (\alpha)            |                  (\mathbb R^{T_\theta}) | 群級打分（coarse）           |
| (w)                 |                     (\mathbb R^{K_e L}) | soft-OMP 權重（fine）      |
| (A_\theta)          |                  (\mathbb R^{T_\theta}) | 方向能量（DoA 讀出）           |
| (D_{\text{emb}})    |                 (\mathbb R^{d\times P}) | 嵌入後字典（attention 版）     |
| (q,K)               | (\mathbb R^{d},\ \mathbb R^{P\times d}) | attention query / keys |
| codebooks (C^{(m)}) |               (\mathbb R^{F\times K_m}) | RVQ 第 (m) 階碼本          |
| (\hat H)            |                         (\mathbb R^{F}) | RVQ 重建方向響應             |

---

## 備註與小技巧

* **單位化**：對每個原子 (d_{(\cdot,\cdot)}) **(\ell_2) 單位化**，讓 (s=D^\top r) 更像「相關性」而非能量偏置。
* **coarse pooling**：群級打分可用 (\ell_\infty)（貼 OMP 的「最大相關」）或 (\ell_2)（穩定）。
* **稀疏權重**：把 softmax 換 **sparsemax / α-entmax（α≈1.3–1.5）** 可得到精確 0，行為更像 soft-OMP。
* **值稀疏**：加入 ISTA/FISTA 近端（(\ell_1)）可使 (x) 更稀疏（與 LASSO 靠近）。
* **ICL 前綴**：(S_0\subset{1\ldots P})，(|S_0|\le K_{\text{total}})；或 teacher 先走 (L_{\text{prefix}}) 步產生前綴，再接 forward 迭代。

---

