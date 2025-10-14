# ICL 系統架構詳解 - 回答你的三個核心問題

## 🎯 問題 1: Transformer 在哪裡？

### 答案：Transformer 在 `doa_rl/hf/model.py`，被三個訓練腳本**共用**

讓我畫出完整的資料流向：

```
原始音訊數據 (--data-root)
    ↓
┌─────────────────────────────────────────────────┐
│  DoADataset / DoAICLDataset                    │
│  (doa_rl/data.py)                              │
│                                                 │
│  讀取 .npy → STFT → Y(F,N) → Tokenizers        │
│                                   ↓             │
│                            生成 prompt 字串      │
└─────────────────────────────────────────────────┘
    ↓ prompt 字串
┌─────────────────────────────────────────────────┐
│  HF Tokenizer                                   │
│  (doa_rl/hf/tokenizer.py)                      │
│                                                 │
│  prompt 字串 → token IDs (數字序列)             │
└─────────────────────────────────────────────────┘
    ↓ token IDs
┌─────────────────────────────────────────────────┐
│  🤖 Transformer Model                          │
│  (doa_rl/hf/model.py)                          │
│                                                 │
│  GPT2LMHeadModel (d=256, layers=2, heads=8)    │
│  + Value Head (for RL)                         │
└─────────────────────────────────────────────────┘
    ↓ logits / value
┌─────────────────────────────────────────────────┐
│  訓練邏輯                                        │
│  ├─ train_reward_model_lora.py: 訓練 RM        │
│  ├─ train_sft_policy_with_rm.py: 訓練 Policy   │
│  └─ train_trl_ppo_with_rm.py: PPO 強化學習     │
└─────────────────────────────────────────────────┘
```

### 關鍵理解：Transformer 是共用的基礎設施

**三個訓練腳本都用同一個 Transformer，但訓練目標不同**：

1. **train_reward_model_lora.py**
   ```python
   # 位置：scripts/train_reward_model_lora.py L45-50
   from doa_rl.hf import build_patch_tokenizer, build_value_head_model
   
   tokenizer = build_patch_tokenizer(direction_angles)  # 建立詞彙表
   rm_model, _ = build_value_head_model(tokenizer)      # 建立 Transformer
   
   # 訓練：讓 v_head 學會「評分」
   # Input: [BOS] <D_090> <P_0_0_5> ...
   # Output: score (scalar) ← v_head 輸出
   ```

2. **train_sft_policy_with_rm.py**
   ```python
   # 位置：scripts/train_sft_policy_with_rm.py L180-185
   policy, _ = build_value_head_model(tokenizer)  # 同樣的 Transformer
   
   # 訓練：讓 LM head 學會「預測方向 token」
   # Input: [BOS] <P_0_0_5> <P_0_1_8> ...
   # Output: <D_090> ← LM head 預測下一個 token
   ```

3. **train_trl_ppo_with_rm.py**
   ```python
   # 位置：scripts/train_trl_ppo_with_rm.py L95-100
   policy, reference = build_value_head_model(tokenizer)
   
   # 訓練：用 RM 的獎勵優化 policy
   # Input: [BOS] <P_0_0_5> ...
   # Policy 生成: <D_090>
   # RM 評分: score = 0.85
   # PPO 更新 policy 參數
   ```

---

## 🎯 問題 2: `--data-root` 存的是什麼？

### 答案：**永遠是原始音訊數據**，ICL 處理在 **runtime 動態生成**

這是關鍵設計決策！讓我詳細解釋：

### 🗂️ 資料目錄結構（不變）

```bash
/Users/sbplab/jiawei/datasets/.../white_noise_box_data_no_edge_sync_vad_normalized/
├── angle_080/
│   ├── clip_000.npy   # 原始音訊波形 (145920,)
│   ├── clip_001.npy
│   └── ...
├── angle_085/
│   ├── clip_000.npy
│   └── ...
├── angle_090/
│   └── ...
└── ...
```

**重點**：
- ✅ `--data-root` 指向的是**原始 .npy 音訊檔案**
- ✅ **不需要**預先生成 ICL prompts 存成檔案
- ✅ ICL 轉換在 `DoAICLDataset.__getitem__()` **即時進行**

### 📊 資料流向對比

#### 現有系統（Simple）

```python
# scripts/train_reward_model_lora.py
ds = DoADataset(
    args.data_root,  # ← 原始音訊目錄
    direction_angles,
    ...
)

# 每次讀取一個樣本時：
batch = ds[0]
# batch = {
#     "Y": Tensor(116, 189),           # STFT 頻譜
#     "angle_deg": 90.0,
#     "path": ".../angle_090/clip_000.npy"
# }

# 然後 tokenize：
prompt = " ".join(PatchTokenizer()(batch["Y"].numpy()))
# → "<P_0_0_5> <P_0_1_8> <P_1_0_12> ..."
```

#### ICL 系統（Multi-Modal）

```python
# scripts/train_reward_model_lora.py（修改後）
from doa_rl.features.tokenizers_extended import NMFAtomTokenizer, DirectionProjectionTokenizer
from doa_rl.features.prompt_builder import MultiModalPromptBuilder

# 建立 tokenizers
patch_tok = PatchTokenizer()
atom_tok = NMFAtomTokenizer(W.numpy())           # 載入 W 矩陣
dir_tok = DirectionProjectionTokenizer(H.numpy()) # 載入 H 矩陣

prompt_builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok)

ds = DoAICLDataset(
    args.data_root,  # ← 還是原始音訊目錄！
    direction_angles,
    prompt_builder=prompt_builder,  # ← 傳入 prompt builder
    icl_mode=args.icl_mode,
    ...
)

# 每次讀取一個樣本時：
batch = ds[0]
# batch = {
#     "Y": Tensor(116, 189),
#     "angle_deg": 90.0,
#     "path": "...",
#     "prompt": "<R_090:14> <AT_5:12> <P_0_0_5> ..."  # ← 新增！動態生成
# }
```

### 🔑 關鍵理解：為什麼不預先生成 ICL prompts？

**原因 1：靈活性**
```python
# 不同實驗可以用不同的 tokenizer 設定
experiment_1 = DoAICLDataset(..., top_k_atoms=5)   # 5 個 atoms
experiment_2 = DoAICLDataset(..., top_k_atoms=10)  # 10 個 atoms
# 不需要重新處理數據！
```

**原因 2：ICL context 是隨機的**
```python
# 每次訓練，ICL context 都不同（增加多樣性）
batch_1 = ds[0]  # context: [angle_030, angle_060, angle_120]
batch_2 = ds[0]  # context: [angle_045, angle_075, angle_105] ← 不同！
```

**原因 3：節省空間**
```python
# 原始數據：10,000 samples × 145KB = 1.45 GB
# 如果存 prompts：10,000 × (512 tokens × 4 bytes) ≈ 20 MB
# → 但每個 epoch 都要重新生成（因為 ICL context 隨機）
# → 不如 runtime 計算
```

---

## 🎯 問題 3: 實作步驟與目標的關聯

讓我重新解釋每一步**為什麼要做**、**做什麼**、**產出什麼**：

### 📅 Week 1: 核心功能實作

#### Day 1-2: 實作 Tokenizers（你需要的新能力）

**目標**：讓系統能生成多模態 tokens

**要做的事**：
```bash
# 建立新檔案
touch doa_rl/features/tokenizers_extended.py
```

**檔案內容**（~150 行）：
```python
# doa_rl/features/tokenizers_extended.py

class NMFAtomTokenizer:
    """把頻譜分解成 NMF atoms 並生成 tokens"""
    
    def __init__(self, W: np.ndarray, top_k: int = 8):
        self.W = W  # (F, K) 從 usm.pth 載入
        self.top_k = top_k
    
    def __call__(self, Y: np.ndarray) -> List[str]:
        # Y(F,N) → Ybar(F) → z(K) → top-k atoms
        Ybar = Y.mean(axis=1)
        z = estimate_z_is(Ybar, self.W)
        top_indices = np.argsort(-z)[:self.top_k]
        
        tokens = []
        for idx in top_indices:
            level = quantize(z[idx])
            tokens.append(f"<AT_{idx}:{level}>")
        return tokens
        # → ["<AT_5:12>", "<AT_23:8>", ...]

class DirectionProjectionTokenizer:
    """計算頻譜與方向轉移函數的相關性"""
    
    def __init__(self, H: np.ndarray, angles: List[int]):
        self.H = H  # (F, D) 從 h_matrix.pth 載入
        self.angles = angles
    
    def __call__(self, Y: np.ndarray, top_m: int = 5) -> List[str]:
        Ybar = Y.mean(axis=1)
        scores = []
        for d in range(self.H.shape[1]):
            score = cosine_similarity(Ybar, self.H[:, d])
            scores.append((d, score))
        
        top_dirs = sorted(scores, key=lambda x: -x[1])[:top_m]
        tokens = []
        for dir_idx, score in top_dirs:
            level = quantize(score)
            angle = self.angles[dir_idx]
            tokens.append(f"<R_{angle:03d}:{level}>")
        return tokens
        # → ["<R_090:14>", "<R_085:12>", ...]
```

**產出**：
- ✅ `doa_rl/features/tokenizers_extended.py`
- ✅ 能從 Y 生成 `<AT_...>` 和 `<R_...>` tokens

**為什麼重要**：
- 這是**物理知識注入**的關鍵！
- Direction tokens = 用 H 矩陣提供物理先驗
- Atom tokens = 用 W 矩陣捕捉頻譜結構

---

#### Day 3-4: 實作 Prompt Builder（組合 tokens）

**目標**：把三種 tokens 組合成完整 prompt

**要做的事**：
```bash
touch doa_rl/features/prompt_builder.py
```

**檔案內容**（~100 行）：
```python
# doa_rl/features/prompt_builder.py

class MultiModalPromptBuilder:
    def __init__(self, patch_tok, atom_tok, dir_tok, config):
        self.patch_tok = patch_tok
        self.atom_tok = atom_tok
        self.dir_tok = dir_tok
        self.config = config
    
    def build_prompt(self, Y: np.ndarray) -> str:
        tokens = []
        
        # 步驟 1: Direction tokens（物理先驗）
        if self.config.use_directions:
            dir_tokens = self.dir_tok(Y, top_m=5)
            tokens.extend(dir_tokens)
            # → ["<R_090:14>", "<R_085:12>", ...]
        
        # 步驟 2: Atom tokens（結構資訊）
        if self.config.use_atoms:
            atom_tokens = self.atom_tok(Y)
            tokens.extend(atom_tokens)
            # → ["<AT_5:12>", "<AT_23:8>", ...]
        
        # 步驟 3: Patch tokens（細節）
        if self.config.use_patches:
            patch_tokens = self.patch_tok(Y)
            tokens.extend(patch_tokens)
            # → ["<P_0_0_5>", "<P_0_1_8>", ...]
        
        return " ".join(tokens)
        # 最終 prompt:
        # "<R_090:14> <R_085:12> <AT_5:12> <AT_23:8> <P_0_0_5> <P_0_1_8> ..."
```

**產出**：
- ✅ `doa_rl/features/prompt_builder.py`
- ✅ 能組合三種 tokens 成完整 prompt

**為什麼重要**：
- 定義了**資訊呈現順序**（物理 → 結構 → 細節）
- 未來可以調整順序/比例做實驗

---

#### Day 5-6: 擴展 Dataset（整合到資料流）

**目標**：讓訓練腳本能使用新的 prompt

**要做的事**：修改 `doa_rl/data.py`

**修改內容**（新增 ~80 行）：
```python
# doa_rl/data.py（在現有 DoADataset 之後新增）

class DoAICLDataset(DoADataset):
    """支援多模態 prompts 的 Dataset"""
    
    def __init__(self, root, angles, prompt_builder, **kwargs):
        super().__init__(root, angles, **kwargs)
        self.prompt_builder = prompt_builder  # ← 關鍵！
    
    def __getitem__(self, idx):
        # 1. 讀取原始數據（和之前一樣）
        path, angle_deg, _ = self.index[idx]
        wav = np.load(path)  # ← 從 --data-root 讀取
        Y = self._compute_spectrogram(wav)
        
        # 2. 生成 multi-modal prompt（新增！）
        prompt = self.prompt_builder.build_prompt(Y.numpy())
        # → "<R_090:14> <AT_5:12> <P_0_0_5> ..."
        
        return {
            "Y": Y,
            "angle_deg": float(angle_deg),
            "angle_index": self.angles.index(float(angle_deg)),
            "path": str(path),
            "prompt": prompt,  # ← 新增這個欄位！
        }
```

**產出**：
- ✅ `DoAICLDataset` 類別
- ✅ 每個 batch 包含 `"prompt"` 欄位

**為什麼重要**：
- 這是**橋接層**！
- 原始數據（.npy）→ prompt 字串 → 訓練腳本
- 訓練腳本幾乎不用改（只需用 `batch["prompt"]`）

---

#### Day 7: 更新 HF Tokenizer Vocab（讓 Transformer 認識新 tokens）

**目標**：擴展詞彙表，包含 `<AT_...>` 和 `<R_...>` tokens

**要做的事**：修改 `doa_rl/hf/tokenizer.py`

**修改內容**：
```python
# doa_rl/hf/tokenizer.py

def _build_vocab_extended(direction_tokens, n_atoms=64):
    vocab = []
    
    # Special tokens
    vocab.extend(["<PAD>", "<BOS>", "<EOS>", "<UNK>"])
    
    # Patch tokens（原有）
    for i in range(7):
        for j in range(18):
            for level in range(16):
                vocab.append(f"<P_{i}_{j}_{level}>")
    
    # NMF Atom tokens（新增！）
    for atom_id in range(n_atoms):
        for level in range(16):
            vocab.append(f"<AT_{atom_id}:{level}>")
    
    # Direction Projection tokens（新增！）
    for angle in range(0, 181, 5):
        for level in range(16):
            vocab.append(f"<R_{angle:03d}:{level}>")
    
    # Direction tokens（原有）
    vocab.extend(direction_tokens)
    
    return vocab
```

**產出**：
- ✅ 擴展的詞彙表（~3000 個 tokens）
- ✅ Transformer 能處理新 tokens

**為什麼重要**：
- Transformer 必須「認識」所有 tokens
- 每個 token 有自己的 embedding

---

### 📅 Week 2: 整合訓練

#### Day 8-9: 修改訓練腳本（讓它們使用新系統）

**目標**：加入 `--use-multi-modal` 參數

**要做的事**：修改 `scripts/train_reward_model_lora.py`

**修改內容**（新增 ~30 行）：
```python
# scripts/train_reward_model_lora.py

def main():
    ap = argparse.ArgumentParser(...)
    
    # === 新增參數 ===
    ap.add_argument("--use-multi-modal", action="store_true",
                    help="Use multi-modal tokens (Atoms + Direction)")
    args = ap.parse_args()
    
    # === 載入物理矩陣 ===
    W = torch.load(args.w_path)["W"]  # NMF 字典
    H = load_H(args.tf_path)           # 轉移函數
    
    # === 建立 tokenizers ===
    patch_tok = PatchTokenizer()
    
    if args.use_multi_modal:
        from doa_rl.features.tokenizers_extended import NMFAtomTokenizer, DirectionProjectionTokenizer
        atom_tok = NMFAtomTokenizer(W.numpy())
        dir_tok = DirectionProjectionTokenizer(H.numpy(), direction_angles)
    else:
        atom_tok = None
        dir_tok = None
    
    prompt_builder = MultiModalPromptBuilder(patch_tok, atom_tok, dir_tok)
    
    # === 使用新 Dataset ===
    from doa_rl.data import DoAICLDataset
    ds = DoAICLDataset(
        args.data_root,  # ← 還是原始數據目錄！
        direction_angles,
        prompt_builder=prompt_builder,
    )
    
    # === 訓練循環（幾乎不變）===
    for batch in dataloader:
        prompt = batch["prompt"]  # ← 用這個代替手動 tokenize
        input_ids = tokenizer.encode(prompt, ...)
        # ... 後續訓練邏輯不變
```

**產出**：
- ✅ 三個訓練腳本都支援 `--use-multi-modal`
- ✅ 向後兼容（不加參數 = 原有行為）

---

#### Day 10-14: 實驗驗證

**目標**：證明多模態系統更好

**實驗設計**：

```bash
# 實驗 A: Baseline（只用 Patches）
python scripts/train_reward_model_lora.py \
  --data-root /path/to/data \
  --max-samples 100 \
  --rm-epochs 10 \
  --out results/baseline

# 實驗 B: Multi-Modal（Patches + Atoms + Directions）
python scripts/train_reward_model_lora.py \
  --data-root /path/to/data \  # ← 同樣的數據！
  --use-multi-modal \           # ← 只加這個參數
  --max-samples 100 \
  --rm-epochs 10 \
  --out results/multimodal
```

**評估**：
```python
# 對比準確度
baseline_acc = evaluate(baseline_model)     # 假設 65%
multimodal_acc = evaluate(multimodal_model) # 期待 70-75%

# 可視化 attention
attention_weights = multimodal_model.get_attention()
# 分析：Direction tokens 的 attention 是否最高？
```

---

## 🗺️ 完整架構總結圖

```
📂 專案結構
├── 數據（不變）
│   └── /path/to/data/
│       ├── angle_080/
│       │   └── clip_*.npy  ← 原始音訊
│       └── ...
│
├── 物理資產（已有）
│   ├── h_matrix.pth  ← 轉移函數 H
│   └── usm.pth       ← NMF 字典 W
│
├── 新增代碼（Week 1）
│   ├── doa_rl/features/tokenizers_extended.py
│   │   ├── NMFAtomTokenizer        ← 用 W 生成 <AT_...>
│   │   └── DirectionProjectionTokenizer ← 用 H 生成 <R_...>
│   │
│   ├── doa_rl/features/prompt_builder.py
│   │   └── MultiModalPromptBuilder ← 組合 tokens
│   │
│   └── doa_rl/data.py（擴展）
│       └── DoAICLDataset           ← 生成 multi-modal prompts
│
├── 修改代碼（Week 2）
│   ├── doa_rl/hf/tokenizer.py（擴展 vocab）
│   └── scripts/train_*.py（加 --use-multi-modal）
│
└── 不變的核心
    ├── doa_rl/hf/model.py  ← Transformer（不變！）
    └── 訓練邏輯（幾乎不變）
```

---

## 💡 關鍵理解總結

### Q1: Transformer 在哪裡？
**答**：`doa_rl/hf/model.py` 的 `GPT2LMHeadModel`，三個訓練腳本共用，只是訓練目標不同。

### Q2: `--data-root` 存什麼？
**答**：永遠是**原始 .npy 音訊檔**，ICL 轉換在 `Dataset.__getitem__()` 動態進行。

### Q3: 每一步做什麼？
- **Day 1-2**: 寫能生成新 tokens 的類別（物理知識注入）
- **Day 3-4**: 寫組合 tokens 的類別（定義資訊順序）
- **Day 5-6**: 寫新 Dataset（橋接原始數據和 prompts）
- **Day 7**: 更新詞彙表（讓 Transformer 認識新 tokens）
- **Day 8-9**: 修改訓練腳本（加參數，用新 Dataset）
- **Day 10-14**: 跑實驗驗證效果

---

## ❓ 現在還有疑問嗎？

如果還不清楚，告訴我你卡在哪個環節，我可以：
1. 📝 畫更詳細的圖
2. 🔍 展示具體的程式碼片段
3. 🎯 用更簡單的例子說明

準備好開始實作了嗎？🚀
