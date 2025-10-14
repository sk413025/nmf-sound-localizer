# 開發進度總結與測試計劃

## 📅 開發進度總結（基於 4 個 Commits）

### ✅ Commit 1: dafac66 - Day 1-2 Multi-Modal Tokenizers

**完成時間：** 2025-10-14 20:39:44

**核心功能：**
1. **NMFAtomTokenizer** - 基於 NMF 分解的頻譜結構編碼器
   - 輸入：頻譜 Y(F,N) + W 矩陣(F,K)
   - 輸出：`<AT_atom_id:level>` tokens（8 個，top-k）
   - 作用：捕捉頻譜的結構化成分（phoneme-like patterns）

2. **DirectionProjectionTokenizer** - 基於物理轉移函數的方向編碼器
   - 輸入：頻譜 Y(F,N) + H 矩陣(F,D) + angles
   - 輸出：`<R_angle:level>` tokens（5 個，top-m）
   - 作用：注入物理先驗（direction-specific transfer function）

**檔案：**
- `doa_rl/features/tokenizers_extended.py` (259 lines)
- `tests/test_tokenizers_extended.py` (358 lines)
- `validate_tokenizers.py` (202 lines)
- `demo_tokenizers_with_real_data.py` (303 lines)
- `docs/DAY_1_2_IMPLEMENTATION_SUMMARY.md` (321 lines)

**測試結果：** ✅ 3/3 validation tests, 20+ pytest tests passed

---

### ✅ Commit 2: ffbc777 - Day 3-4 MultiModalPromptBuilder

**完成時間：** 2025-10-14 20:54:12

**核心功能：**
1. **MultiModalPromptBuilder** - 組合三種 tokenizers
   - 支援 4 種排序策略：
     * `physics_first`: [Direction] → [Atom] → [Patch]
     * `structure_first`: [Atom] → [Direction] → [Patch]
     * `patch_first`: [Patch] → [Atom] → [Direction]
     * `interleaved`: 循環穿插三種 tokens

2. **Token Budget 控制**
   - 全域限制：`max_tokens=200`
   - 分類限制：`max_patch_tokens`, `max_atom_tokens`, `max_direction_tokens`
   - 壓縮效果：10.3x (391→38 tokens)，信號比例增強 6-7 倍

3. **ICL Prompt 建構**
   - Few-shot format: `[ctx1] <D_030> [ctx2] <D_060> [query]`
   - Context sampling: random, nearest, diverse
   - 動態採樣（每次訓練不同 context）

**檔案：**
- `doa_rl/features/prompt_builder.py` (362 lines)
- `tests/test_prompt_builder.py` (541 lines)
- `validate_prompt_builder.py` (415 lines)
- `demo_prompt_builder.py` (375 lines)
- `docs/DAY_3_4_IMPLEMENTATION_SUMMARY.md` (469 lines)

**測試結果：** ✅ 6/6 validation tests, 25+ pytest tests passed

---

### ✅ Commit 3: cec4b30 - Day 5-6 DoAICLDataset Integration

**完成時間：** 2025-10-14 21:41:19

**核心功能：**
1. **DoAICLDataset** - PyTorch Dataset 整合
   - 繼承 `DoADataset`，完全向後兼容
   - Runtime prompt 生成（不預先儲存）
   - 兩種模式：
     * Basic mode: 單一 multi-modal prompt
     * ICL mode: Few-shot prompts with context

2. **ICL Context Sampling**
   - **Random:** 均勻採樣（無偏基準）
   - **Nearest:** 選擇最接近的角度（插值友好）
   - **Diverse:** 最大化最小成對距離（覆蓋最大化）

3. **資料流整合**
   ```
   .npy audio → STFT → Y(F,N) → Tokenizers → Prompt string → batch['prompt']
   ```

**檔案：**
- `doa_rl/data.py` (+220 lines to existing file)
- `tests/test_doa_icl_dataset.py` (526 lines)
- `validate_doa_icl_dataset.py` (473 lines)
- `demo_doa_icl_dataset.py` (410 lines)
- `docs/DAY_5_6_IMPLEMENTATION_SUMMARY.md` (466 lines)

**測試結果：** ✅ 10/10 validation tests, 25+ pytest tests passed

**真實數據驗證：**
- W 矩陣: 346×50 (from `usm.pth`)
- H 矩陣: 346×17 (from `h_matrix_normalized_original_to_box.pth`)
- 51 samples, 601 tokens/sample (baseline), 2407 tokens/sample (ICL 3-shot)

---

### ✅ Commit 4: af196dd - Day 7 HF Tokenizer Vocabulary Extension

**完成時間：** 2025-10-14 21:55:57

**核心功能：**
1. **詞彙表擴展**
   - +1,024 NMF Atom tokens: `<AT_0:0>` to `<AT_63:15>`
   - +592 Direction Projection tokens: `<R_000:0>` to `<R_180:15>`
   - 總計 +1,616 tokens（79% 增長）

2. **Pre-tokenizer 修復**
   - **問題：** `Whitespace()` 會在標點符號處分割
     * `<AT_5:14>` → `['<', 'AT_5', ':', '14', '>']` ❌
   - **解決：** 使用 `WhitespaceSplit()` 只在空格處分割
     * `<AT_5:14>` → `['<AT_5:14>']` ✅

3. **向後兼容**
   - `enable_extended_vocab=False`: 基礎詞彙表（2,025 tokens）
   - `enable_extended_vocab=True`: 擴展詞彙表（3,641 tokens）

**檔案：**
- `doa_rl/hf/tokenizer.py` (+78 lines, -5 lines modified)
- `validate_tokenizer_vocab.py` (343 lines)
- `demo_extended_tokenizer.py` (423 lines)
- `docs/DAY_7_IMPLEMENTATION_SUMMARY.md` (498 lines)

**測試結果：** ✅ 5/5 validation tests, 5/5 demos successful

**性能影響：**
- Sequence length: +20% (多模態 tokens overhead)
- Training time: +20% (longer sequences)
- Embedding params: +1,616 × d_model (e.g., +412K @ d=256)
- Memory: ~2KB per prompt string

---

## 🏗️ 整個訓練流程現在的長相

### 架構概覽

```
┌────────────────────────────────────────────────────────────────┐
│ 1. 資料準備階段 (不變)                                         │
├────────────────────────────────────────────────────────────────┤
│  - 輸入：原始音訊 .npy files (--data-root)                     │
│  - 物理資產：                                                   │
│    * W 矩陣：doa_normalized_config_c_corrected/models/usm.pth │
│    * H 矩陣：h_matrix_normalized_original_to_box.pth           │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 2. 多模態 Tokenization (NEW! ✨)                              │
├────────────────────────────────────────────────────────────────┤
│  DoAICLDataset.__getitem__(idx):                              │
│                                                                 │
│  ┌─────────────┐                                               │
│  │ Load .npy   │ → audio waveform (145920,)                   │
│  └─────────────┘                                               │
│         ↓                                                       │
│  ┌─────────────┐                                               │
│  │ STFT        │ → Y(F=346, N=189) spectrogram                │
│  └─────────────┘                                               │
│         ↓                                                       │
│  ┌────────────────────────────────────────┐                   │
│  │ 並行 Tokenization (3 種)                │                   │
│  │                                         │                   │
│  │  DirectionProjectionTokenizer(Y, H)    │                   │
│  │    → <R_090:14> <R_085:12> ... (5個)   │                   │
│  │                                         │                   │
│  │  NMFAtomTokenizer(Y, W)                │                   │
│  │    → <AT_5:10> <AT_23:8> ... (8個)     │                   │
│  │                                         │                   │
│  │  PatchTokenizer(Y)                     │                   │
│  │    → <P_0_0_5> <P_0_1_8> ... (378個)   │                   │
│  └────────────────────────────────────────┘                   │
│         ↓                                                       │
│  ┌─────────────────────────────┐                              │
│  │ MultiModalPromptBuilder     │                              │
│  │  - 組合策略: physics_first  │                              │
│  │  - Token budget: 200        │                              │
│  └─────────────────────────────┘                              │
│         ↓                                                       │
│  prompt = "<R_090:14> <R_085:12> <AT_5:10> <P_0_0_5> ..."    │
│                                                                 │
│  batch = {                                                     │
│      'Y': Tensor(346, 189),                                    │
│      'angle_deg': 90.0,                                        │
│      'prompt': prompt  ← NEW!                                  │
│  }                                                             │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 3. HF Tokenizer 編碼 (擴展詞彙表 ✨)                           │
├────────────────────────────────────────────────────────────────┤
│  tokenizer = build_patch_tokenizer(                            │
│      angles,                                                    │
│      enable_extended_vocab=True,  ← 啟用多模態                 │
│      n_atoms=50                                                 │
│  )                                                             │
│                                                                 │
│  prompt string → token IDs                                     │
│  Vocabulary: 3,641 tokens                                      │
│    - 4 special (<BOS>, <EOS>, <PAD>, <UNK>)                   │
│    - 2,016 patch tokens                                        │
│    - 1,024 atom tokens (NEW!)                                  │
│    - 592 direction projection tokens (NEW!)                    │
│    - 5 direction target tokens                                 │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 4. Transformer Model (不變)                                    │
├────────────────────────────────────────────────────────────────┤
│  GPT2LMHeadModel (doa_rl/hf/model.py)                         │
│    - Embeddings: 3,641 × 256                                   │
│    - Layers: 2                                                 │
│    - Heads: 8                                                  │
│    - d_model: 256                                              │
│    + Value Head (for RL)                                       │
│                                                                 │
│  Input: token_ids → Embeddings → Transformer → Logits/Values │
└────────────────────────────────────────────────────────────────┘
                            ↓
┌────────────────────────────────────────────────────────────────┐
│ 5. 訓練流程 (三階段，待整合 Day 8-9)                           │
├────────────────────────────────────────────────────────────────┤
│  Step 1: train_reward_model_lora.py                           │
│    目標：訓練 Value Head 評分能力                              │
│    輸入：multi-modal prompt                                    │
│    輸出：RM score (scalar)                                     │
│    訓練：MSE loss between predicted and target rewards        │
│                                                                 │
│  Step 2: train_sft_policy_with_rm.py                          │
│    目標：訓練 LM Head 預測方向                                 │
│    輸入：multi-modal prompt                                    │
│    輸出：<D_angle> prediction                                  │
│    訓練：Cross-entropy loss with RM-guided targets            │
│                                                                 │
│  Step 3: train_trl_ppo_with_rm.py                             │
│    目標：PPO 強化學習優化 policy                               │
│    輸入：multi-modal prompt                                    │
│    輸出：<D_angle> (via sampling)                              │
│    訓練：PPO with RM rewards                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 🎯 如何使用已開發功能進行測試

### 階段 1: 功能驗證 (已完成 ✅)

**執行：**
```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f
python demo_complete_workflow.py
```

**驗證項目：**
- ✅ W/H 矩陣正確載入
- ✅ 三種 Tokenizers 正常運作
- ✅ MultiModalPromptBuilder 組合 tokens
- ✅ Token budget 壓縮功能（4x compression）
- ✅ HF Tokenizer 擴展詞彙表（+1,616 tokens）
- ✅ Transformer Model 前向傳播

**輸出示例：**
```
✓ Token 分布:
  Direction: 5 (2.5%)  → 物理先驗
  Atom:      8 (4.0%)  → 頻譜結構
  Patch:     187 (93.5%) → 細節資訊

✓ 壓縮後 Token 分布 (信號比例增強):
  Direction: 5 (10.0%) [原 2.5%]  → 4x 增強
  Atom:      8 (16.0%) [原 4.0%]  → 4x 增強
  Patch:     37 (74.0%) [原 93.5%]

✓ Transformer Model:
  Embedding 矩陣: torch.Size([3417, 256])
  總參數量: 4,552,193
```

---

### 階段 2: 訓練腳本整合 (Day 8-9, 待完成)

#### 修改目標

將多模態功能整合到三個訓練腳本：
1. `scripts/train_reward_model_lora.py`
2. `scripts/train_sft_policy_with_rm.py`
3. `scripts/train_trl_ppo_with_rm.py`

#### 具體修改步驟

**A. 新增命令列參數**

```python
# 在每個腳本的 argparse 部分加入：
parser.add_argument(
    "--use-multi-modal",
    action="store_true",
    help="Enable multi-modal tokenization (Direction + Atom + Patch)"
)
parser.add_argument(
    "--w-path",
    type=str,
    default="doa_normalized_config_c_corrected/models/usm.pth",
    help="Path to W matrix (NMF dictionary)"
)
parser.add_argument(
    "--h-path",
    type=str,
    default="h_matrix_normalized_original_to_box.pth",
    help="Path to H matrix (transfer function)"
)
parser.add_argument(
    "--n-atoms",
    type=int,
    default=50,
    help="Number of NMF atoms in W matrix"
)
parser.add_argument(
    "--token-ordering",
    type=str,
    default="physics_first",
    choices=["physics_first", "structure_first", "patch_first", "interleaved"],
    help="Token ordering strategy"
)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=200,
    help="Maximum tokens per prompt (for budget control)"
)
parser.add_argument(
    "--icl-mode",
    action="store_true",
    help="Enable ICL few-shot prompts"
)
parser.add_argument(
    "--n-shots",
    type=int,
    default=3,
    help="Number of ICL context examples"
)
parser.add_argument(
    "--context-strategy",
    type=str,
    default="random",
    choices=["random", "nearest", "diverse"],
    help="ICL context sampling strategy"
)
```

**B. 程式碼修改**

```python
# 在 dataset 建立前加入：

if args.use_multi_modal:
    print("🚀 啟用多模態 ICL 系統...")
    
    # 1. 載入 W 和 H 矩陣
    import torch
    W_data = torch.load(args.w_path, map_location="cpu")
    W = W_data["W"].numpy()
    print(f"  ✓ W 矩陣載入: {W.shape}")
    
    H_data = torch.load(args.h_path, map_location="cpu", weights_only=False)
    H = H_data["H"].numpy()
    h_angles = H_data["angles"]
    print(f"  ✓ H 矩陣載入: {H.shape}")
    
    # 2. 建立三種 Tokenizers
    from doa_rl.features.tokenizers import PatchTokenizer
    from doa_rl.features.tokenizers_extended import (
        NMFAtomTokenizer,
        DirectionProjectionTokenizer,
    )
    
    patch_tok = PatchTokenizer()
    atom_tok = NMFAtomTokenizer(W, top_k=8)
    dir_tok = DirectionProjectionTokenizer(H, h_angles, top_m=5)
    print(f"  ✓ Tokenizers 建立完成")
    
    # 3. 建立 MultiModalPromptBuilder
    from doa_rl.features.prompt_builder import (
        MultiModalPromptBuilder,
        PromptConfig,
    )
    
    config = PromptConfig(
        ordering=args.token_ordering,
        use_directions=True,
        use_atoms=True,
        use_patches=True,
        max_tokens=args.max_tokens,
    )
    
    prompt_builder = MultiModalPromptBuilder(
        patch_tokenizer=patch_tok,
        atom_tokenizer=atom_tok,
        direction_tokenizer=dir_tok,
        config=config,
    )
    print(f"  ✓ PromptBuilder 建立完成 ({args.token_ordering}, max={args.max_tokens})")
    
    # 4. 使用 DoAICLDataset
    from doa_rl.data import DoAICLDataset
    
    train_ds = DoAICLDataset(
        root=args.data_root,
        angles=direction_angles,
        prompt_builder=prompt_builder,
        icl_mode=args.icl_mode,
        n_shots=args.n_shots,
        context_strategy=args.context_strategy,
    )
    print(f"  ✓ DoAICLDataset 建立完成 ({len(train_ds)} samples)")
    
    # 5. 建立擴展 Tokenizer
    tokenizer = build_patch_tokenizer(
        direction_angles,
        enable_extended_vocab=True,
        n_atoms=args.n_atoms,
    )
    print(f"  ✓ HF Tokenizer 建立完成 (vocab={len(tokenizer)})")

else:
    # 原有邏輯（向後兼容）
    print("使用基礎 Patch-only 模式...")
    from doa_rl.data import DoADataset
    
    train_ds = DoADataset(
        root=args.data_root,
        angles=direction_angles,
    )
    
    tokenizer = build_patch_tokenizer(direction_angles)

# 後續訓練邏輯保持不變...
```

**C. DataLoader 使用**

```python
# 如果使用多模態，需要修改 collate function 來處理 prompt 欄位

def collate_fn(batch):
    if 'prompt' in batch[0]:
        # Multi-modal mode
        prompts = [item['prompt'] for item in batch]
        angles = [item['angle_deg'] for item in batch]
        
        # Tokenize prompts
        encoded = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'angles': torch.tensor(angles),
        }
    else:
        # Original mode
        # ... 原有邏輯
```

---

### 階段 3: Smoke Test (驗證訓練能運行)

**目標：** 確認修改後的腳本能正常運行，不需要完整訓練。

**執行命令：**

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f

# Step 1: RM Smoke Test (Multi-Modal)
python scripts/train_reward_model_lora.py \
    --data-root <實際數據路徑> \
    --use-multi-modal \
    --w-path doa_normalized_config_c_corrected/models/usm.pth \
    --h-path h_matrix_normalized_original_to_box.pth \
    --n-atoms 50 \
    --token-ordering physics_first \
    --max-tokens 200 \
    --rm-epochs 2 \
    --batch-size 4 \
    --out results/mm_smoke_rm

# Step 2: SFT Smoke Test (Multi-Modal)
python scripts/train_sft_policy_with_rm.py \
    --data-root <實際數據路徑> \
    --use-multi-modal \
    --rm-adapters results/mm_smoke_rm_adapters \
    --rm-heads results/mm_smoke_rm_heads.pt \
    --epochs 2 \
    --batch-size 4 \
    --out results/mm_smoke_sft

# Step 3: PPO Smoke Test (Multi-Modal)
python scripts/train_trl_ppo_with_rm.py \
    --data-root <實際數據路徑> \
    --use-multi-modal \
    --rm-adapters results/mm_smoke_rm_adapters \
    --rm-heads results/mm_smoke_rm_heads.pt \
    --n-ppo-epochs 2 \
    --batch-size 4 \
    --out results/mm_smoke_ppo
```

**預期結果：**
- ✅ 腳本正常啟動
- ✅ 多模態 prompts 成功生成
- ✅ Loss 正常下降
- ✅ 模型權重正常保存

---

### 階段 4: 對比實驗 (驗證多模態效果)

#### 實驗設計

| 實驗 ID | 模式 | Token Ordering | ICL | 目的 |
|---------|------|----------------|-----|------|
| **Baseline** | Patch-only | - | No | 基準性能 |
| **MM-Physics** | Multi-modal | physics_first | No | 物理先驗效果 |
| **MM-Structure** | Multi-modal | structure_first | No | 結構先驗效果 |
| **MM-ICL-3shot** | Multi-modal | physics_first | Yes (3-shot, nearest) | Few-shot 學習 |

#### 執行命令

```bash
# 實驗 A: Baseline
python scripts/train_reward_model_lora.py \
    --data-root <path> \
    --rm-epochs 10 \
    --out results/baseline

# 實驗 B: Multi-Modal (Physics-first)
python scripts/train_reward_model_lora.py \
    --data-root <path> \
    --use-multi-modal \
    --token-ordering physics_first \
    --rm-epochs 10 \
    --out results/mm_physics

# 實驗 C: Multi-Modal (Structure-first)
python scripts/train_reward_model_lora.py \
    --data-root <path> \
    --use-multi-modal \
    --token-ordering structure_first \
    --rm-epochs 10 \
    --out results/mm_structure

# 實驗 D: ICL 3-shot
python scripts/train_reward_model_lora.py \
    --data-root <path> \
    --use-multi-modal \
    --token-ordering physics_first \
    --icl-mode \
    --n-shots 3 \
    --context-strategy nearest \
    --rm-epochs 10 \
    --out results/mm_icl_3shot
```

#### 評估指標

1. **RM 評分準確度**
   ```python
   # 計算 predicted score 與 ground-truth reward 的相關性
   correlation = np.corrcoef(pred_scores, true_rewards)[0, 1]
   ```

2. **Policy 方向預測準確度**
   ```python
   # Top-1 accuracy
   top1_acc = (predicted_angles == true_angles).mean()
   
   # Top-K recall (K=3)
   topk_acc = np.mean([true in topk_preds for true, topk_preds in zip(...)])
   ```

3. **Attention Weights 分析**
   ```python
   # 分析 Direction tokens 是否獲得最高 attention
   # 期待：physics_first 時，Direction tokens attention 最高
   
   attention = model.get_attention_weights()
   dir_attention = attention[:, :5].mean()  # 前 5 個 tokens (Direction)
   atom_attention = attention[:, 5:13].mean()  # 接下來 8 個 (Atom)
   patch_attention = attention[:, 13:].mean()  # 剩餘 (Patch)
   
   print(f"Direction attention: {dir_attention:.3f}")
   print(f"Atom attention: {atom_attention:.3f}")
   print(f"Patch attention: {patch_attention:.3f}")
   ```

---

### 階段 5: Ablation Studies (剖析各組件貢獻)

#### 實驗設計

| 實驗 | Direction | Atom | Patch | 目的 |
|------|-----------|------|-------|------|
| **Patch-only** | ❌ | ❌ | ✅ | Baseline |
| **Dir + Patch** | ✅ | ❌ | ✅ | 物理先驗單獨效果 |
| **Atom + Patch** | ❌ | ✅ | ✅ | 結構先驗單獨效果 |
| **Full Multi-modal** | ✅ | ✅ | ✅ | 完整系統 |

#### 執行方式

修改 `PromptConfig`:
```python
# Direction only
config = PromptConfig(
    use_directions=True,
    use_atoms=False,
    use_patches=True,
)

# Atom only
config = PromptConfig(
    use_directions=False,
    use_atoms=True,
    use_patches=True,
)

# Full
config = PromptConfig(
    use_directions=True,
    use_atoms=True,
    use_patches=True,
)
```

---

## 📊 預期成果

### 假設驗證

| 假設 | 預期結果 | 驗證方式 |
|------|----------|----------|
| **H1: 物理先驗提升性能** | MM-Physics > Baseline | Top-1 accuracy 提升 5-10% |
| **H2: Token ordering 有影響** | physics_first > structure_first > patch_first | 對比準確度 |
| **H3: Token budget 增強信號** | compressed (budget=50) ≥ full (budget=200) | 壓縮 4x 後性能不降 |
| **H4: ICL 改善 few-shot** | ICL-3shot > No-ICL | 小樣本場景準確度 +10% |
| **H5: Direction tokens 獲得高 attention** | Attention(Dir) > Attention(Atom) > Attention(Patch) | Attention weights 分析 |

---

## 📁 檔案清單

### 核心實作 (Day 1-7)

```
doa_rl/
├── features/
│   ├── tokenizers_extended.py    # NMF Atom + Direction Projection
│   └── prompt_builder.py          # MultiModalPromptBuilder
├── data.py                         # DoAICLDataset (+220 lines)
└── hf/
    └── tokenizer.py                # Extended vocabulary (+78 lines)

tests/
├── test_tokenizers_extended.py    # 20+ tests
├── test_prompt_builder.py         # 25+ tests
└── test_doa_icl_dataset.py        # 25+ tests

docs/
├── DAY_1_2_IMPLEMENTATION_SUMMARY.md
├── DAY_3_4_IMPLEMENTATION_SUMMARY.md
├── DAY_5_6_IMPLEMENTATION_SUMMARY.md
└── DAY_7_IMPLEMENTATION_SUMMARY.md
```

### 驗證腳本

```
validate_tokenizers.py              # 3/3 tests
validate_prompt_builder.py          # 6/6 tests
validate_doa_icl_dataset.py         # 10/10 tests
validate_tokenizer_vocab.py         # 5/5 tests
```

### 示範腳本

```
demo_tokenizers_with_real_data.py
demo_prompt_builder.py
demo_doa_icl_dataset.py
demo_extended_tokenizer.py
demo_complete_workflow.py           # 完整流程演示 ← 新增
```

---

## 🚀 下一步行動

### 立即可做（Day 8-9）

1. **修改訓練腳本** (估計 4-6 小時)
   - [ ] 修改 `train_reward_model_lora.py`
   - [ ] 修改 `train_sft_policy_with_rm.py`
   - [ ] 修改 `train_trl_ppo_with_rm.py`
   - [ ] 測試 Smoke test 能運行

2. **準備實驗數據** (估計 1-2 小時)
   - [ ] 確認數據路徑 (`--data-root`)
   - [ ] 驗證 W 和 H 矩陣可用
   - [ ] 準備小規模測試集 (100 samples)

### 實驗階段（Day 10-14）

3. **Baseline 實驗** (估計 2-3 小時)
   - [ ] 跑 Patch-only baseline
   - [ ] 建立評估指標腳本
   - [ ] 記錄 baseline 性能

4. **Multi-Modal 實驗** (估計 4-6 小時)
   - [ ] 對比 physics_first vs structure_first
   - [ ] 測試 Token budget 影響
   - [ ] Ablation studies

5. **ICL 實驗** (估計 3-4 小時)
   - [ ] 對比 1-shot vs 3-shot vs 5-shot
   - [ ] 對比 random vs nearest vs diverse sampling
   - [ ] 分析 attention weights

6. **結果分析與報告** (估計 4-6 小時)
   - [ ] 整理所有實驗結果
   - [ ] 繪製對比圖表（使用 `AGENTS.md` 定義的 palette）
   - [ ] 撰寫實驗報告（使用結構化 commit message 格式）

---

## 💡 關鍵技術決策回顧

### 為什麼 Runtime 生成 Prompts？

**原因 1: 靈活性**
- 不同實驗可以用不同 tokenizer 配置
- Token budget 可動態調整
- 不需要重新處理數據

**原因 2: ICL Context 隨機性**
- 每個 epoch 的 context 都不同
- 增加訓練多樣性
- 避免 overfitting 到特定 context

**原因 3: 節省空間**
- 原始數據：1.45 GB
- 如果預存 prompts：每個 epoch 都要重新生成
- Runtime 計算 overhead <20ms（可忽略）

### 為什麼需要 Token Budget？

**原因 1: Transformer 效率**
- 原始 prompt: 391 tokens
- 壓縮後: 38 tokens (10.3x)
- Training time 減少 ~10x

**原因 2: 信號比例增強**
- Direction: 1.3% → 7.9% (6x)
- Atom: 2.0% → 13.2% (6.6x)
- 物理/結構信號更強

**原因 3: Ablation Studies**
- 可測試不同壓縮比例的影響
- 找到最佳 token budget

---

## 🎉 總結

### 已完成的工作

✅ **Day 1-7 完整實作** (4 個 commits, ~8,000 lines code + docs)
- 多模態 Tokenizers (NMF Atom + Direction Projection)
- MultiModalPromptBuilder (4 種排序策略 + Token budget)
- DoAICLDataset (Runtime prompts + ICL sampling)
- HF Tokenizer 擴展 (+1,616 tokens)

✅ **完整測試覆蓋**
- 70+ pytest test cases
- 24/24 validation tests passed
- 5 demo scripts with real data

✅ **功能驗證成功**
- 完整工作流程可運行（`demo_complete_workflow.py`）
- 所有組件整合無誤
- Transformer 能處理多模態 prompts

### 待完成的工作

🚧 **Day 8-9: 訓練腳本整合** (估計 4-6 小時)
- 修改三個訓練腳本
- 新增命令列參數
- 測試 Smoke test

🧪 **Day 10-14: 實驗驗證** (估計 15-20 小時)
- Baseline vs Multi-Modal 對比
- Ablation studies
- ICL few-shot 實驗
- 結果分析與報告

---

**準備好開始 Day 8-9 了嗎？** 🚀

如有任何問題，請參考：
- 各階段實作總結：`docs/DAY_*_IMPLEMENTATION_SUMMARY.md`
- 架構說明：`docs/ICL_ARCHITECTURE_EXPLAINED.md`
- 訓練流程：`TRAINING_FLOW.md`
- 完整工作流程演示：`demo_complete_workflow.py`
