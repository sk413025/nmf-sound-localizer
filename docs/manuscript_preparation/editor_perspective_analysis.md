# Nature Communications 主編視角分析

**文件目的**：整合投稿策略、主編心理分析、以及寫作指南，作為 Nature Communications 投稿的完整參考手冊。

**最後更新**：2026-01-14

---

## 目錄

1. [文件總覽](#1-文件總覽)
2. [主編 Sarah 的世界](#2-主編-sarah-的世界)
3. [Sarah 的偏好清單](#3-sarah-的偏好清單)
4. [策略文件精華摘要](#4-策略文件精華摘要)
5. [研究定位策略](#5-研究定位策略)
6. [Sarah 的決策樹](#6-sarah-的決策樹)
7. [投稿檢查清單](#7-投稿檢查清單)
8. [文獻引用清單](#8-文獻引用清單)

---

## 1. 文件總覽

### 1.1 研究文件地圖

```
本專案文件架構
├── docs/
│   ├── research_context/
│   │   └── comprehensive_research_narrative.md  ← 完整技術敘事（1261 行）
│   ├── related_work_development_map.md          ← 文獻脈絡與引用定位
│   ├── research_context_related_work_summary.md ← NC 對齊摘要
│   └── manuscript_preparation/
│       ├── editor_perspective_analysis.md       ← 本文件（背景參考）
│       ├── nc_crossdisciplinary_reframing_guide.md  ← 操作指南（撰寫中快速查閱）★
│       └── auditory_physiology_literature.md    ← 聽覺生理學文獻
│
└── writing_notes/
    ├── 策略.md    ← Sarah 心理剖析 + 寫作原則
    ├── 策略2.md   ← 各章節詳細寫作指南
    └── 策略3.md   ← 四篇基礎文獻深度解析
```

### 1.2 策略文件關聯性

| 文件 | 角色 | 核心內容 |
|------|------|----------|
| **comprehensive_research_narrative.md** | 完整故事書 | 物理原理、實驗細節、技術架構、消融實驗 |
| **related_work_development_map.md** | 文獻地圖 | 時間線、概念脈絡、Related Work 草稿 |
| **research_context_related_work_summary.md** | 橋接摘要 | 內容濃縮 + NC 策略對齊 + 期刊清單 |
| **策略.md** | Sarah 心理剖析 | 四層思考、三大恐懼、寫作原則 |
| **策略2.md** | 章節指南 | Abstract/Introduction/Results/Discussion/Methods |
| **策略3.md** | 文獻基礎 | [R1,R4] 時間反轉 + [R5,R6] 逆散射理論 |

### 1.3 文件使用場景

| 撰寫階段 | 主要參考文件 |
|----------|-------------|
| 規劃敘事架構 | 本文件 + 策略.md |
| 撰寫 Introduction | 策略.md 下半部（6 段結構） |
| 撰寫 Results | 策略2.md（假說驅動結構） |
| 撰寫 Related Work | related_work_development_map.md |
| **撰寫中快速查閱** | **nc_crossdisciplinary_reframing_guide.md** ★ |
| 技術細節確認 | comprehensive_research_narrative.md |
| 投稿前檢查 | 本文件第 7 章（檢查清單） |

---

## 2. 主編 Sarah 的世界

### 2.1 日常壓力：47 篇稿件的早晨

> **虛構人物 Dr. Sarah Chen**：Nature Communications 物理科學領域資深編輯

```
Sarah 的一天
─────────────────────────────────────────────────────
早上 9:00   收件匣：47 篇新投稿等待初審
            │
            │  每篇只有 5-10 分鐘做決定
            │
            ├─ ~20% 會被接受（8-10 篇）
            │
            ├─ ~80% 需要「禮貌性拒絕」
            │
下午 5:00   還有：期刊會議、審稿人催促、作者申訴...
─────────────────────────────────────────────────────
```

**關鍵洞察**：你的論文必須在**前 2 頁**（Abstract + Introduction 第一段）就讓她做出「繼續或拒絕」的判斷。

### 2.2 四層思考邏輯

Sarah 讀每篇論文時，腦中同時運轉著四個層次的思考：

#### 第一層：風險評估
> 「這會不會害我出事？」

| 擔憂 | Sarah 的掃描 |
|------|-------------|
| 發表後被撤稿 | 數據是否太完美？宣稱是否太大膽？ |
| 統計分析有問題 | 有沒有誤差條？顯著性檢驗？ |
| 可重複性危機 | 有沒有承諾公開程式碼/數據？ |

#### 第二層：工作量評估
> 「這會不會讓我的工作變得很痛苦？」

| 擔憂 | Sarah 的預測 |
|------|-------------|
| 找不到審稿人 | 論文定位是否清楚？能立刻想到誰審嗎？ |
| 審稿人意見分歧 | 有沒有「可被攻擊的面」？ |
| 作者難以溝通 | 論文語氣是謙遜還是自我膨脹？ |

#### 第三層：成就評估
> 「這會讓我看起來聰明嗎？」

| 期待 | Sarah 的想像 |
|------|-------------|
| 高引用潛力 | 這篇會被誰引用？在什麼脈絡下？ |
| 跨領域影響 | 其他領域的人會感興趣嗎？ |
| 媒體報導潛力 | 這有「故事性」嗎？ |

#### 第四層：科學判斷
> 「這真的是科學嗎？」

| 區分 | Sarah 的直覺 |
|------|-------------|
| 科學問題 | 「X 為什麼可能？極限在哪裡？」 |
| 工程問題 | 「如何把 X 做得更好？」 |
| 深刻洞察 | 揭示某種關於世界的真理 |
| 花哨技巧 | 只是展示技術能力 |

### 2.3 三大恐懼分析

#### 恐懼一：發表後被撤稿（⚠️⚠️⚠️⚠️⚠️ 致命）

**Sarah 的內心**：
> 「Nature 系列期刊的聲譽建立在幾十年的信任之上。每一次撤稿都是對這個信任的打擊。如果我的判斷出了問題，這會成為科學新聞，會影響期刊的影響因子，會讓上司質疑我的判斷力。」

**你的論文如何化解**：
- ✅ 消融實驗清楚顯示每個組件的貢獻
- ✅ 誠實承認失敗案例（純物理 1.7%、純 DL 2.7%）
- ✅ 統計顯著性報告（p < 0.001，n=5）
- ✅ 承諾公開程式碼和資料

#### 恐懼二：拒絕經典論文（⚠️⚠️ 中等）

**Sarah 的心理**：
> 「拒絕好論文的後果是『錯失機會』，接受壞論文的後果是『主動傷害』。人類心理上更害怕後者。所以在不確定時，我傾向拒絕。」

**意義**：你不能讓 Sarah 處於「不確定」狀態。5 分鐘內必須讓她清楚知道：
- 你解決了什麼問題
- 為什麼這個問題重要
- 你怎麼解決的
- 結果有多好

#### 恐懼三：審稿人意見嚴重分歧（⚠️⚠️⚠️ 高）

**Sarah 的惡夢**：
> 審稿人 A：「這是突破性的工作」
> 審稿人 B：「這完全是錯誤的」

**你的論文如何化解**：
- 主動承認弱點（審稿人就沒東西可攻擊）
- 清楚定義範圍和假設
- 誠實的限制討論

### 2.4 時間壓力與審稿人困境

#### 找審稿人的困難

| 理想審稿人條件 | 現實困難 |
|----------------|----------|
| 懂這個領域 | 跨領域論文難找專家 |
| 有時間 | 忙碌的教授常拒絕 |
| 無利益衝突 | 小領域人人認識 |
| 願意免費工作 | 越來越多人拒審 |

#### 你的論文如何幫助 Sarah

讓她在讀完 Introduction 後能立刻想到：
- 「光學振動量測 → 找 Prof. A」
- 「物理資訊機器學習 → 找 Prof. B」
- 「稀疏編碼/逆問題 → 找 Prof. C」

**關鍵**：明確定位為「物理資訊機器學習 + 遠距聲學感測」

---

## 3. Sarah 的偏好清單

### 3.1 她喜歡看到的（✅）

| 偏好 | 原因 | 如何滿足 |
|------|------|----------|
| **清晰的一句話核心宣稱** | 她需要能在會議上說「這篇論文證明了 X」 | 「結構物理是天然的稀疏編碼器」 |
| **誠實的限制討論** | 減少她被審稿人打臉的風險 | 消融實驗清楚顯示每個組件的貢獻 |
| **統計顯著性** | 可信度的基本門檻 | p < 0.001，n=5 獨立實驗 |
| **跨領域連結** | 高引用潛力 = 她的成績單 | 連結神經科學、壓縮感測、材料科學 |
| **公開程式碼/數據承諾** | 減少可重複性危機的風險 | Data/Code Availability 聲明 |
| **範式轉移的語言** | NC 的定位：改變理解的研究 | 「散射不是雜訊，是編碼」 |
| **假說驅動的實驗** | 區分科學與工程 | 「If hypothesis H is true, we should observe P」 |
| **因果推理** | Nature 等級論文的標誌 | 控制實驗證明「為什麼有效」 |

### 3.2 她不想看到的（❌）

| 禁忌 | 原因 | 如何避免 |
|------|------|----------|
| **只報告數字沒有解釋** | 這是工程報告，不是科學論文 | 解釋「為什麼」93.5% 是有意義的 |
| **誇大其詞** | 審稿人會攻擊，她會丟臉 | 用「suggest」「may」而非「prove」「will」 |
| **隱藏弱點** | 沒有完美的研究 → 隱藏 = 不誠實 | 主動討論計算成本、特定條件的限制 |
| **模糊的定位** | 找不到審稿人 → 直接拒絕 | 明確說「物理資訊機器學習 + 遠距聲學感測」 |
| **純技術描述** | 「我們用了 Transformer」→「所以呢？」 | 解釋 Transformer 為什麼在物理上有意義 |
| **過長的背景介紹** | Sarah 不需要 LDV 的歷史 | 第一頁內就要建立定位 |
| **防禦性語氣** | 預示作者難溝通 | 保持謙遜、開放的語氣 |

### 3.3 決定接受/拒絕的關鍵因素

**在 5 分鐘內，Sarah 需要能回答三個問題**：

| 問題 | 正確答案範例 | 錯誤答案範例 |
|------|-------------|-------------|
| 這篇論文屬於什麼領域？ | 「物理資訊機器學習 + 遠距聲學感測」 | 「可能是光學？或聲學？或 ML？」 |
| 核心宣稱是什麼？（一句話） | 「結構物理是天然的稀疏編碼器」 | 「我們達到了 93.5% 準確率」 |
| 如果這是真的，會改變什麼？ | 「工程師一直試圖消除的『雜訊』是未被利用的資訊」 | 「LDV 使用者會覺得有用」 |

---

## 4. 策略文件精華摘要

### 4.1 策略.md：心理剖析與寫作原則

#### 核心洞察：科學層級 vs 工程層級

```
工程層級的論文：
「我們設計了一把更好的鑰匙來開這把鎖。」
   ↓
結果：投 IEEE Transactions

科學層級的論文：
「我們發現這把鎖的結構本身編碼了製造它的工匠的資訊，
 而且這個編碼遵循一個普遍的數學規律。」
   ↓
結果：投 Nature Communications
```

#### 你們的論文應該說

> 「當聲波與結構交互作用時，結構不是被動地『傳遞』聲音——它主動地『編碼』聲音。這個編碼過程把聲源的空間資訊（方向、位置）轉化為振動模式的稀疏表徵。我們發現，稀疏解碼是解開這個物理編碼的通用鑰匙。LDV 只是我們用來讀取這個編碼的窗口。」

#### 關鍵轉變

| 之前（錯誤） | 之後（正確） |
|-------------|-------------|
| 主角：LDV 和 Transformer | 主角：結構物理作為編碼器 |
| 問題：如何改進 LDV 訊號處理 | 問題：結構物理如何編碼環境資訊 |
| 貢獻：一個更好的演算法 | 貢獻：一個關於物理-資訊關係的洞察 |
| 連結：技術遷移 | 連結：概念統一（物理系統 ≈ 神經系統） |

### 4.2 策略2.md：各章節寫作指南

#### Abstract（150-200 字）

**結構**：問題 → 缺口 → 方法 → 結果 → 意義

**Sarah 的期待**：30 秒電梯遊說，不是論文縮短版

**範例開頭**：
> ❌ 「We propose a Transformer-based architecture for LDV signal processing...」
> ✅ 「When waves interact with complex structures, the resulting vibrations are conventionally treated as noise...」

#### Introduction（約 1000-1500 字，6 段結構）

| 段落 | 功能 | 關鍵句型 |
|------|------|----------|
| 1 | 建立普遍現象 + 傳統觀點 | 「But what if...?」 |
| 2 | 引入編碼視角 | 「The question becomes...」 |
| 3 | 提出稀疏性假說 | 「We hypothesize that...」 |
| 4 | 現有研究的缺口 | 「Neither approach asks...」 |
| 5 | 你們的方法和結果 | 「If hypothesis H is correct...」 |
| 6 | 更廣泛的意義 | 「The implications extend beyond...」 |

#### Results（約 2000-3000 字）

**結構**：每個子章節 = 「宣稱 → 證據 → 解釋」

**子章節建議**：
1. 框架驗證：OMP 展開的可行性
2. 物理約束的效果
3. 實際語音復原性能
4. 泛化測試
5. 極限分析（物理上的資訊邊界）

#### Discussion（約 1000-1500 字）

**功能**：幫助 Sarah 向別人解釋為什麼發表這篇

**必須包含**：
- 核心發現的提煉（可被直接引用的句子）
- 與現有工作的關係
- 對更廣泛領域的啟示
- 限制討論
- 未來方向

### 4.3 策略3.md：文獻基礎與學術家譜

#### 四篇基礎文獻的關係

```
┌─────────────────────────────────────────────────────────────┐
│                 「散射後的訊號能還原嗎？」                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [R1] & [R4] 時間反轉聲學        [R5] & [R6] 逆散射理論       │
│  ─────────────────────          ─────────────────────       │
│  物理學家的樂觀回答：            數學家的謹慎警告：            │
│  「可以！散射是可逆的映射」       「可以，但極度敏感」          │
│                                                             │
│  實驗證明：把訊號倒播回去        理論分析：這是「病態問題」     │
│  → 聲波會自動聚焦回源頭          → 微小誤差會被劇烈放大        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     你的研究填補的空白：
              「如何在實際條件下穩定地解碼散射訊號？」
```

#### 給 Sarah 的一句話解釋

> 「三十年前，Fink 證明了散射是可逆的映射 [R1, R4]，數學家警告了反演的病態性需要先驗知識 [R5, R6]——我們的工作首次展示如何把這些物理先驗『編譯』成可學習的神經網路結構，讓單點 LDV 測量可以穩定地解碼聲學散射來實現聲源定位。」

---

## 5. 研究定位策略

### 5.1 工程敘事 vs 科學敘事

| 面向 | 工程敘事（❌） | 科學敘事（✅） |
|------|---------------|---------------|
| **開場** | 「LDV 是一種非接觸式振動量測技術...」 | 「聲音作為波動現象，在與物質交互作用後留下的痕跡中，保留了多少可被解碼的資訊？」 |
| **問題** | 「如何改進 LDV 訊號處理？」 | 「自然界的物理過程如何與資訊處理產生關聯？」 |
| **宣稱** | 「我們達到了 93.5% 準確率」 | 「結構物理本身是天然的稀疏編碼器」 |
| **意義** | 「LDV 使用者會覺得有用」 | 「這個原理可能揭示物理系統和神經系統的共同規律」 |
| **Sarah 的分類** | 「工程報告」→ 禮貌性拒絕 | 「科學探索」→ 認真審稿 |

### 5.2 核心宣稱的一句話表達

**對不同聽眾的版本**：

| 聽眾 | 一句話版本 |
|------|-----------|
| Sarah（主編） | 「結構物理是天然的稀疏編碼器，散射不是雜訊而是資訊」 |
| 物理學家 | 「波-結構交互作用產生的模態疊加編碼了聲源的方向資訊」 |
| ML 研究者 | 「我們把迭代物理演算法『展開』成可學習的神經網路層」 |
| 神經科學家 | 「物理系統可能和大腦一樣使用稀疏編碼來表徵環境資訊」 |
| 工程師 | 「單點 LDV 測量可以達到傳統麥克風陣列的聲源定位性能」 |

### 5.3 跨領域影響力論述

**為什麼不同領域的人會引用這篇論文？**

| 領域 | 連結點 | 可能的引用脈絡 |
|------|--------|---------------|
| **神經科學** | 稀疏編碼 | 「物理系統和感覺皮質使用類似的編碼原理 [本文]」 |
| **壓縮感測** | 稀疏表徵 | 「自然界存在天然的稀疏結構 [本文]」 |
| **結構健康監測** | 振動分析 | 「結構響應包含可解碼的環境資訊 [本文]」 |
| **物理資訊學習** | 演算法展開 | 「物理先驗可以『編譯』成網路結構 [本文]」 |
| **聲學感測** | 單點定位 | 「不需要陣列也能做方向估計 [本文]」 |

---

## 6. Sarah 的決策樹

### 完整決策流程圖

```
                    Sarah 收到你的投稿
                           │
                           ▼
              ┌─────────────────────────┐
              │ 這是科學問題還是工程問題？│
              └───────────┬─────────────┘
                          │
           ┌──────────────┴──────────────┐
           │                             │
           ▼                             ▼
     「工程問題」                   「科學問題」
     「如何做更好」                 「為什麼可能」
           │                             │
           ▼                             ▼
     禮貌性拒絕                    繼續評估
     「投 IEEE」                         │
                                        ▼
                         ┌─────────────────────────┐
                         │ 我能在 5 分鐘內理解核心嗎？│
                         └───────────┬─────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │                             │
                      ▼                             ▼
                   「不能」                       「能」
                   模糊 → 風險                   清晰 → 安全
                      │                             │
                      ▼                             ▼
                   拒絕                        繼續評估
                                                   │
                                                   ▼
                                   ┌─────────────────────────┐
                                   │ 如果這是真的，會改變什麼？│
                                   └───────────┬─────────────┘
                                               │
                                ┌──────────────┴──────────────┐
                                │                             │
                                ▼                             ▼
                          「只影響小圈子」             「跨領域影響」
                                │                             │
                                ▼                             ▼
                         投專業期刊                    繼續評估
                                                           │
                                                           ▼
                                           ┌─────────────────────────┐
                                           │ 這會讓我丟臉嗎？         │
                                           │ （可信度、可重複性）      │
                                           └───────────┬─────────────┘
                                                       │
                                        ┌──────────────┴──────────────┐
                                        │                             │
                                        ▼                             ▼
                                  有風險信號                    看起來可信
                                  （數據太完美、                 （誠實限制、
                                   隱藏弱點）                    統計顯著）
                                        │                             │
                                        ▼                             ▼
                                     拒絕                       ✅ 送審
```

### Sarah 在內部會議的辯護

如果 Sarah 被問：「為什麼接受這篇？」

她需要能說：

> 「這篇論文回答了一個三十年的老問題。Fink 在 1996 年證明了散射是可逆的，但那需要換能器陣列。數學家警告反演是病態的，需要先驗知識。這篇論文**首次**展示如何把物理先驗『編譯』成神經網路結構，讓單點雷射測量就能解碼聲源方向。」
>
> 「更重要的是，它揭示了一個**普遍原理**：結構物理本身就是稀疏編碼器。這個洞察可能影響從材料科學到神經科學的多個領域。消融實驗清楚顯示純物理（1.7%）和純學習（2.7%）都失敗，只有結合才成功（93.5%）——這是**因果證據**，不是相關性。」

---

## 7. 投稿檢查清單

### 7.1 Abstract 檢查點

- [ ] 第一句建立「傳統觀點」而非直接說技術
- [ ] 核心宣稱用一句話清楚表達
- [ ] LDV 被框架為「readout interface」而非主角
- [ ] 包含具體的性能數字
- [ ] 最後一句指向跨領域影響
- [ ] 總字數在 150-200 字之間
- [ ] 沒有使用只有專家才懂的術語

### 7.2 Introduction 檢查點

- [ ] 第一頁內就清楚定位（「物理資訊機器學習 + 遠距聲學感測」）
- [ ] 有「傳統觀點 → 但是... → 新視角」的轉折
- [ ] 「We hypothesize」而非「We demonstrate」（在方法出現前）
- [ ] 明確指出「沒有人從這個角度看問題」
- [ ] 有「If hypothesis H is correct, we should observe P」的科學推理
- [ ] 控制實驗的預告（結構變簡單 → 性能下降）
- [ ] 最後一段連結神經科學/壓縮感測
- [ ] 總字數約 1000-1500 字

### 7.3 Results 檢查點

- [ ] 每個子章節有清楚的「宣稱 → 證據 → 解釋」結構
- [ ] 不是只描述圖表，而是引導讀者看到你想讓他們看到的
- [ ] 基線比較是公平的（說明如何調參）
- [ ] 報告多次實驗的平均值和標準差
- [ ] 「分布外」測試清楚定義什麼是「不同的條件」
- [ ] 有因果推理（「這證明了...因為...」）
- [ ] 性能下降的案例也有展示（誠實）

### 7.4 Discussion 檢查點

- [ ] 第一段是可被直接引用的核心發現提煉
- [ ] 與 Visual Microphone、事件相機方法形成對話
- [ ] 討論「演算法展開 + 物理約束」範式的通用性
- [ ] 誠實討論限制（計算成本、特定條件）
- [ ] 限制討論包含「為什麼這不影響核心結論」
- [ ] 未來方向是具體的、有洞察力的
- [ ] 語氣是「有信心但謙遜」

### 7.5 整體風險評估

**Sarah 會問的問題**：

- [ ] 「這篇論文的核心宣稱是什麼？」→ 能用一句話回答
- [ ] 「如果我接受，審稿人會打我臉嗎？」→ 誠實討論了限制
- [ ] 「如果我接受，一年後會有人說無法重複嗎？」→ 公開程式碼/數據
- [ ] 「如果我接受，會被說 over-hyped 嗎？」→ 用「suggest」「may」
- [ ] 「我能找到審稿人嗎？」→ 定位清楚，能想到至少 3 個名字

---

## 8. 文獻引用清單

### 8.1 時間反轉聲學 [R1, R4, R34]

| 標籤 | 文獻 | DOI |
|------|------|-----|
| [R1] | Time reversal in acoustics. *Contemporary Physics* (1996) | [10.1080/00107519608230338](https://doi.org/10.1080/00107519608230338) |
| [R4] | An overview of time-reversal acoustics. *JASA* (2008) | [10.1121/1.2933288](https://doi.org/10.1121/1.2933288) |
| [R34] | Time-Reversal Acoustics in Biomedical Engineering. *Annual Review of Biomedical Engineering* (2003) | [10.1146/annurev.bioeng.5.040202.121630](https://doi.org/10.1146/annurev.bioeng.5.040202.121630) |

**核心貢獻**：證明散射是可逆映射——把訊號時間反轉後播放，聲波會自動聚焦回源頭。

### 8.2 逆散射理論 [R5, R6]

| 標籤 | 文獻 | DOI |
|------|------|-----|
| [R5] | On an optimisation method for inverse acoustic scattering. *Inverse Problems* (1989) | [10.1088/0266-5611/5/2/009](https://doi.org/10.1088/0266-5611/5/2/009) |
| [R6] | Inverse acoustic scattering by small-obstacle expansion. *Inverse Problems* (2008) | [10.1088/0266-5611/24/3/035022](https://doi.org/10.1088/0266-5611/24/3/035022) |

**核心貢獻**：警告逆散射是「病態問題」——微小測量誤差會被劇烈放大，需要先驗知識來穩定。

### 8.3 LDV 與聲場測量 [R7-R10]

| 標籤 | 文獻 | DOI |
|------|------|-----|
| [R7] | Laser Doppler vibrometry and near-field acoustic holography. *MSSP* (2006) | [10.1016/j.ymssp.2005.11.011](https://doi.org/10.1016/j.ymssp.2005.11.011) |
| [R8] | Visualising scattering underwater acoustic fields using LDV. *JSV* (2007) | [10.1016/j.jsv.2007.04.026](https://doi.org/10.1016/j.jsv.2007.04.026) |
| [R9] | Transducer characterization by LDV. *JASA* (2009) | [10.1121/1.4783677](https://doi.org/10.1121/1.4783677) |
| [R10] | Laser Doppler multi-beam differential vibrometry. *JASA* (2020) | [10.1121/1.5147034](https://doi.org/10.1121/1.5147034) |

**核心貢獻**：LDV 提供非接觸式高保真振動測量，可用於聲場視覺化和換能器表徵。

### 8.4 物理資訊機器學習 [R13-R18]

| 標籤 | 文獻 | DOI |
|------|------|-----|
| [R13] | CNNs for inverse problems in imaging: A review. *IEEE SPM* (2017) | [10.1109/MSP.2017.2739299](https://doi.org/10.1109/MSP.2017.2739299) |
| [R14] | Physics-informed neural networks. *JCP* (2019) | [10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045) |
| [R15] | Deep Unfolding for Snapshot Compressive Imaging. *IJCV* (2023) | [10.1007/s11263-023-01844-4](https://doi.org/10.1007/s11263-023-01844-4) |
| [R16] | Far-Field Subwavelength Acoustic Imaging by Deep Learning. *PRX* (2020) | [10.1103/PhysRevX.10.031029](https://doi.org/10.1103/PhysRevX.10.031029) |
| [R17] | Physics-constrained deep learning for acoustic inverse scattering. *MSSP* (2022) | [10.1016/j.ymssp.2021.108190](https://doi.org/10.1016/j.ymssp.2021.108190) |
| [R18] | Neural network warm-start for inverse acoustic obstacle scattering. *JCP* (2023) | [10.1016/j.jcp.2023.112341](https://doi.org/10.1016/j.jcp.2023.112341) |

**核心貢獻**：展示學習先驗可以穩定逆問題，演算法展開可以把迭代物理編碼為可訓練層。

### 8.5 非接觸聲學解碼 [R11, R12, R20, R21]

| 標籤 | 文獻 | DOI |
|------|------|-----|
| [R11] | The visual microphone. *ACM TOG* (2014) | [10.1145/2601097.2601119](https://doi.org/10.1145/2601097.2601119) |
| [R12] | Event-Based Visual Microphone. *ICASSP* (2023) | [10.1109/ICASSP49357.2023.10094677](https://doi.org/10.1109/ICASSP49357.2023.10094677) |
| [R20] | Transmission matrix inversion in scattering media. *Optics Express* (2017) | [10.1364/OE.25.027234](https://doi.org/10.1364/OE.25.027234) |
| [R21] | Online learning of transmission matrix in dynamic media. *Optica* (2023) | [10.1364/OPTICA.479962](https://doi.org/10.1364/OPTICA.479962) |

**核心貢獻**：展示光學感測可以恢復聲學資訊，傳輸矩陣可以建模散射媒介。

### 8.6 Classical DOA and Array Signal Processing [R22-R26]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R22] | High-resolution frequency-wavenumber spectrum analysis. *Proceedings of the IEEE* (1969) | [10.1109/PROC.1969.7278](https://doi.org/10.1109/PROC.1969.7278) |
| [R23] | Multiple emitter location and signal parameter estimation. *IEEE Transactions on Antennas and Propagation* (1986) | [10.1109/TAP.1986.1143830](https://doi.org/10.1109/TAP.1986.1143830) |
| [R24] | ESPRIT - Estimation of signal parameters via rotational invariance techniques. *IEEE Transactions on Acoustics, Speech, and Signal Processing* (1989) | [10.1109/29.32276](https://doi.org/10.1109/29.32276) |
| [R25] | Beamforming: a versatile approach to spatial filtering. *IEEE ASSP Magazine* (1988) | [10.1109/53.665](https://doi.org/10.1109/53.665) |
| [R26] | Two decades of array signal processing research: the parametric approach. *IEEE Signal Processing Magazine* (1996) | [10.1109/79.526899](https://doi.org/10.1109/79.526899) |

Core contribution: establishes classic array-based DOA baselines and the signal processing context that single-point LDV-based decoding departs from.

### 8.7 NMF and IS-Divergence Foundations [R27-R29]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R27] | Learning the parts of objects by non-negative matrix factorization. *Nature* (1999) | [10.1038/44565](https://doi.org/10.1038/44565) |
| [R28] | Nonnegative Matrix Factorization with the Itakura-Saito Divergence: With Application to Music Analysis. *Neural Computation* (2009) | [10.1162/neco.2008.04-08-771](https://doi.org/10.1162/neco.2008.04-08-771) |
| [R29] | Algorithms for Nonnegative Matrix Factorization with the beta-divergence. *Neural Computation* (2011) | [10.1162/neco_a_00168](https://doi.org/10.1162/neco_a_00168) |

Core contribution: formalizes NMF and IS-divergence optimization that underpins the speech dictionary and the IS-geometry inner loop.

### 8.8 Algorithm Unrolling and Sparse Inference [R30-R31]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R30] | A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems. *SIAM Journal on Imaging Sciences* (2009) | [10.1137/080716542](https://doi.org/10.1137/080716542) |
| [R31] | Algorithm Unrolling: Interpretable, Efficient Deep Learning for Signal and Image Processing. *IEEE Signal Processing Magazine* (2021) | [10.1109/MSP.2020.3016905](https://doi.org/10.1109/MSP.2020.3016905) |

Core contribution: positions unrolling as a principled bridge between iterative physics-based solvers and trainable networks.

### 8.9 Transmission Matrix and Scattering Operators (Optics Analog) [R32-R33]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R32] | Measuring the Transmission Matrix in Optics: An Approach to the Study and Control of Light Propagation in Disordered Media. *Physical Review Letters* (2010) | [10.1103/PhysRevLett.104.100601](https://doi.org/10.1103/PhysRevLett.104.100601) |
| [R33] | Focusing coherent light through opaque strongly scattering media. *Optics Letters* (2007) | [10.1364/OL.32.002309](https://doi.org/10.1364/OL.32.002309) |

Core contribution: treats scattering as a linear operator (transmission matrix), reinforcing the mapping view used in the manuscript.

### 8.10 Single-Sensor DOA and NMF-Based Localization [R35-R36]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R35] | Direction of arrival estimation of an acoustic wave using a single structural vibration sensor. *Journal of Sound and Vibration* (2023) | [10.1016/j.jsv.2023.117671](https://doi.org/10.1016/j.jsv.2023.117671) |
| [R36] | Direction of Arrival With One Microphone, a Few LEGOs, and Non-Negative Matrix Factorization. *IEEE/ACM Transactions on Audio, Speech, and Language Processing* (2018) | [10.1109/TASLP.2018.2867081](https://doi.org/10.1109/TASLP.2018.2867081) |

Core contribution: demonstrates single-sensor DOA feasibility and NMF-based localization, directly contextualizing the LDV single-point setting.

### 8.11 Acoustic Structures and Mobile Systems for Localization [R37-R39]

| Tag | Reference | DOI |
|------|-----------|-----|
| [R37] | Spatial information coding with artificially engineered structures for acoustic and elastic wave sensing. *Frontiers in Physics* (2022) | [10.3389/fphy.2022.1024964](https://doi.org/10.3389/fphy.2022.1024964) |
| [R38] | EarCase: Sound Source Localization Leveraging Mini Acoustic Structure Equipped Phone Cases for Hearing-challenged People. *MobiHoc* (2023) | [10.1145/3565287.3610270](https://doi.org/10.1145/3565287.3610270) |
| [R39] | Owlet: Enabling Spatial Information in Ubiquitous Acoustic Devices. *MobiSys* (2021) | [10.1145/3458864.3467880](https://doi.org/10.1145/3458864.3467880) |

Core contribution: highlights engineered structures and mobile-system deployments for spatial audio sensing, supporting the broader impact narrative.

### 8.12 Auditory Physiology and Spatial Hearing [A1-A12, L1-L5]

#### Spatial Hearing Foundations [A1, A3-A6]

| Tag | Reference | DOI |
|------|-----------|-----|
| [A1] | Spatial Hearing: The Psychophysics of Human Sound Localization. Blauert, J. *MIT Press* (1997) | [10.1121/1.392109](https://doi.org/10.1121/1.392109) |
| [A3] | The role of the pinna in human localization. Batteau, D.W. *Proc. R. Soc. B* (1967) | [10.1098/rspb.1967.0058](https://doi.org/10.1098/rspb.1967.0058) |
| [A4] | Transformation of sound-pressure level from the free field to the eardrum. Shaw, E.A.G. *JASA* (1974) | [10.1121/1.1903522](https://doi.org/10.1121/1.1903522) |
| [A5] | Spectral cues used in the localization of sound sources on the median plane. Hebrank, J. & Wright, D. *JASA* (1974) | [10.1121/1.1903520](https://doi.org/10.1121/1.1903520) |
| [A6] | The influence of pinnae-based spectral cues on sound localization. Musicant, A.D. & Butler, R.A. *JASA* (1984) | [10.1121/1.390773](https://doi.org/10.1121/1.390773) |

Core contribution: establishes pinna as direction-dependent spectral filter, directly inspiring the plate-as-scattering-structure analogy.

#### Monaural Localization and Neural Plasticity [A8-A12]

| Tag | Reference | DOI |
|------|-----------|-----|
| [A8] | Contribution of Head Shadow and Pinna Cues to Chronic Monaural Sound Localization. Van Wanrooij, M.M. & Van Opstal, A.J. *J. Neurosci.* (2004) | [10.1523/JNEUROSCI.4163-03.2004](https://doi.org/10.1523/JNEUROSCI.4163-03.2004) |
| [A9] | Monaural sound localization: Acute versus chronic unilateral impairment. Slattery, W.H. & Middlebrooks, J.C. *Hearing Research* (1994) | [10.1016/0378-5955(94)90053-1](https://doi.org/10.1016/0378-5955(94)90053-1) |
| [A10] | Monaural sound localization revisited. Wightman, F.L. & Kistler, D.J. *JASA* (1997) | [10.1121/1.418029](https://doi.org/10.1121/1.418029) |
| [A11] | Adapting to supernormal auditory localization cues. Shinn-Cunningham, B.G. et al. *JASA* (1998) | [10.1121/1.423088](https://doi.org/10.1121/1.423088) |
| [A12] | Relearning sound localization with new ears. Hofman, P.M. et al. *Nature Neurosci.* (1998) | [10.1038/2226](https://doi.org/10.1038/2226) |

Core contribution: demonstrates spectral-to-spatial mappings require learning; supports necessity of data-driven decoder.

#### Advisor Publications - Prof. Ying-Hui Lai (賴穎暉) [L1-L5]

| Tag | Reference | DOI |
|------|-----------|-----|
| [L1] | A Deep Denoising Autoencoder Approach to Improving the Intelligibility of Vocoded Speech in Cochlear Implant Simulation. *IEEE TBME* (2017) | [10.1109/TBME.2016.2613960](https://doi.org/10.1109/TBME.2016.2613960) |
| [L2] | Deep Learning-Based Noise Reduction Approach to Improve Speech Intelligibility for Cochlear Implant Recipients. *Ear and Hearing* (2018) | [10.1097/AUD.0000000000000537](https://doi.org/10.1097/AUD.0000000000000537) |
| [L3] | An Audio-Visual Speech Enhancement Model Using Multimodal Deep Learning. *IEEE TETCI* (2018) | [10.1109/TETCI.2017.2784878](https://doi.org/10.1109/TETCI.2017.2784878) |
| [L4] | Speech enhancement for hearing-impaired listeners using deep neural networks with auditory-mask motivated loss function. *JASA* (2019) | [10.1121/1.5094063](https://doi.org/10.1121/1.5094063) |
| [L5] | Time-frequency attention for monaural speech enhancement. *IEEE ICASSP* (2020) | [10.1109/ICASSP40776.2020.9054182](https://doi.org/10.1109/ICASSP40776.2020.9054182) |

Core contribution: establishes advisor expertise in deep learning for auditory applications; supports the BME/hearing context of this research.

#### Biological Analogy Summary

| Human Auditory System | This Research |
|----------------------|---------------|
| Pinna (outer ear) | Vibrating plate |
| HRTF (spectral filtering) | Modal transfer function H_d |
| Spectral cues | Direction-dependent spectral features |
| Neural learning/adaptation | Deep learning decoder |
| Monaural localization ability | Single-point LDV localization |

**Key differentiation**: Unlike evolved (pinna) or engineered (metamaterial) scattering structures, plate modes arise naturally from structural dynamics—representing a third category of spatial encoding.

---

## 附錄：相關文件連結

- [完整技術敘事](../research_context/comprehensive_research_narrative.md)
- [文獻發展地圖](../related_work_development_map.md)
- [NC 對齊摘要](../research_context_related_work_summary.md)
- [聽覺生理學文獻](./auditory_physiology_literature.md)
