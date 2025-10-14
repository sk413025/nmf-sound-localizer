# In-Context Learning Bridge Design for DOA_RL

## 📋 目標

將原始音訊數據轉換為支援 **In-Context Learning (ICL)** 的 token 序列，橋接：
- **輸入端**：原始音訊數據（angle_*/clip_*.npy）
- **輸出端**：三個訓練腳本（train_reward_model_lora.py, train_sft_policy_with_rm.py, train_trl_ppo_with_rm.py）

---

## 🔍 現狀分析

### 現有系統（Simple Token System）

```
原始音訊 → STFT → Y(F,N) → PatchTokenizer → [<P_i_j_level>, ...] → Transformer
```

**問題**：
1. ❌ **單一模態**：只有 Patch tokens，缺乏物理意義
2. ❌ **無多模態融合**：沒有整合 NMF atoms、Direction projection
3. ❌ **無 ICL 能力**：無法做 few-shot learning 或 context adaptation
4. ❌ **缺乏可解釋性**：tokens 與聲學物理特徵脫節

### 目標系統（Multi-Modal ICL Token System）

```
原始音訊 → STFT → Y(F,N) 
                    ↓
         ┌──────────┼──────────┐
         ↓          ↓          ↓
    Patch      NMF Atoms   Direction
    Tokens     Tokens      Projection
         ↓          ↓          ↓
         └──────────┼──────────┘
                    ↓
         Multi-Modal Token Sequence
                    ↓
         [CLS] + Context + Query → Transformer → Prediction
```

---

## 🏗️ 架構設計

### 階段 1：擴展 Token 系統

#### 1.1 NMF Atom Tokenizer

**目的**：捕捉頻譜的**構成元素**（音素、共振峰等）

```python
# doa_rl/features/tokenizers.py (擴展)

class NMFAtomTokenizer:
    """將頻譜分解為 NMF atoms 並生成 token。
    
    Token 格式: <AT_atom_id:level>
    - atom_id: 原子索引 (0-K)
    - level: 激活強度的量化值 (0-15)
    """
    
    def __init__(self, W: np.ndarray, n_levels: int = 16, top_k: int = 8):
        """
        Args:
            W: NMF 字典 (F, K)，從 usm.pth 載入
            n_levels: 量化級數
            top_k: 選取 top-k 個最強的原子
        """
        self.W = W  # (F, K)
        self.K = W.shape[1]
        self.n_levels = n_levels
        self.top_k = top_k
    
    def __call__(self, Y: np.ndarray) -> List[str]:
        """
        Args:
            Y: 頻譜 (F, N)
        
        Returns:
            List of atom tokens
        """
        from doa_rl.features.nmf_utils import estimate_z_is
        
        # 估計激活向量 z (K,)
        Ybar = Y.mean(axis=1)  # (F,)
        z = estimate_z_is(Ybar, self.W, n_iter=50)
        
        # 選取 top-k 原子
        top_indices = np.argsort(-z)[:self.top_k]
        
        tokens = []
        for idx in top_indices:
            # 量化激活強度
            level = int(np.clip(z[idx] * self.K * 2, 0, self.n_levels - 1))
            tokens.append(f"<AT_{idx}:{level}>")
        
        return tokens
```

#### 1.2 Direction Projection Tokenizer

**目的**：編碼頻譜與各方向轉移函數的**相關性**

```python
class DirectionProjectionTokenizer:
    """計算頻譜與方向轉移函數的投影強度。
    
    Token 格式: <R_direction_id:level>
    - direction_id: 方向角度 (e.g., 030, 090)
    - level: 投影強度 (correlation 或 IS divergence)
    """
    
    def __init__(self, H: np.ndarray, angles: List[int], 
                 n_levels: int = 16, metric: str = "correlation"):
        """
        Args:
            H: 轉移函數矩陣 (F, D)
            angles: 方向角度列表
            metric: 'correlation' 或 'is_divergence'
        """
        self.H = H  # (F, D)
        self.angles = angles
        self.n_levels = n_levels
        self.metric = metric
    
    def __call__(self, Y: np.ndarray, top_m: int = 5) -> List[str]:
        """
        Args:
            Y: 頻譜 (F, N)
            top_m: 選取 top-m 個最強的方向
        
        Returns:
            List of direction projection tokens
        """
        Ybar = Y.mean(axis=1)  # (F,)
        
        scores = []
        for d in range(self.H.shape[1]):
            H_d = self.H[:, d]
            if self.metric == "correlation":
                # 歸一化相關性
                score = np.dot(Ybar, H_d) / (np.linalg.norm(Ybar) * np.linalg.norm(H_d) + 1e-12)
            elif self.metric == "is_divergence":
                # 負 IS divergence（越小越好，所以取負）
                ratio = Ybar / (H_d + 1e-12)
                score = -np.sum(ratio - np.log(ratio + 1e-12) - 1.0)
            scores.append((d, score))
        
        # 選取 top-m
        scores.sort(key=lambda x: -x[1])
        top_dirs = scores[:top_m]
        
        tokens = []
        for dir_idx, score in top_dirs:
            # 量化分數
            level = int(np.clip((score + 1) / 2 * (self.n_levels - 1), 0, self.n_levels - 1))
            angle = self.angles[dir_idx]
            tokens.append(f"<R_{angle:03d}:{level}>")
        
        return tokens
```

---

### 階段 2：Multi-Modal Prompt 構建器

#### 2.1 Prompt Builder

```python
# doa_rl/features/prompt_builder.py (新檔案)

from typing import List, Dict, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class PromptConfig:
    """Prompt 構建配置"""
    use_patches: bool = True
    use_atoms: bool = True
    use_directions: bool = True
    
    # Token 順序策略
    ordering: str = "physics_first"  # "physics_first" | "mixed" | "hierarchical"
    
    # ICL 支援
    include_context: bool = False
    context_shots: int = 3  # Few-shot 範例數量


class MultiModalPromptBuilder:
    """將頻譜轉換為多模態 prompt token 序列。"""
    
    def __init__(
        self,
        patch_tokenizer,
        atom_tokenizer: Optional[NMFAtomTokenizer] = None,
        direction_tokenizer: Optional[DirectionProjectionTokenizer] = None,
        config: Optional[PromptConfig] = None,
    ):
        self.patch_tok = patch_tokenizer
        self.atom_tok = atom_tokenizer
        self.dir_tok = direction_tokenizer
        self.config = config or PromptConfig()
    
    def build_prompt(self, Y: np.ndarray, angle: Optional[int] = None) -> str:
        """構建單個樣本的 prompt。
        
        Args:
            Y: 頻譜 (F, N)
            angle: 已知角度（用於 context examples）
        
        Returns:
            Token 序列字符串
        """
        tokens = []
        
        # 1. 物理引導的 tokens（Direction + Atoms）
        if self.config.use_directions and self.dir_tok:
            dir_tokens = self.dir_tok(Y, top_m=5)
            tokens.extend(dir_tokens)
        
        if self.config.use_atoms and self.atom_tok:
            atom_tokens = self.atom_tok(Y)
            tokens.extend(atom_tokens)
        
        # 2. 細節 tokens（Patches）
        if self.config.use_patches:
            patch_tokens = self.patch_tok(Y)
            tokens.extend(patch_tokens)
        
        # 3. 如果有已知角度，加入方向標籤
        if angle is not None:
            tokens.append(f"<D_{angle:03d}>")
        
        return " ".join(tokens)
    
    def build_icl_prompt(
        self,
        query_Y: np.ndarray,
        context_examples: List[Dict[str, any]],
    ) -> str:
        """構建 In-Context Learning prompt。
        
        Format:
        [CLS] <context_1> [SEP] <context_2> [SEP] ... [SEP] <query>
        
        Args:
            query_Y: 查詢頻譜
            context_examples: List of {"Y": np.ndarray, "angle": int}
        
        Returns:
            ICL prompt 字符串
        """
        tokens = ["<CLS>"]
        
        # 加入 context examples（few-shot）
        for example in context_examples[:self.config.context_shots]:
            example_prompt = self.build_prompt(example["Y"], example["angle"])
            tokens.append(example_prompt)
            tokens.append("<SEP>")
        
        # 加入 query（沒有角度標籤）
        query_prompt = self.build_prompt(query_Y, angle=None)
        tokens.append(query_prompt)
        
        return " ".join(tokens)
```

---

### 階段 3：資料管道整合

#### 3.1 擴展 DoADataset

```python
# doa_rl/data.py (修改)

class DoAICLDataset(DoADataset):
    """支援 In-Context Learning 的 DoA Dataset。"""
    
    def __init__(
        self,
        root: str,
        angles: List[float],
        prompt_builder: MultiModalPromptBuilder,
        icl_mode: bool = False,
        context_pool_size: int = 50,
        **kwargs,
    ):
        super().__init__(root, angles, **kwargs)
        self.prompt_builder = prompt_builder
        self.icl_mode = icl_mode
        
        # 為 ICL 預先建立 context pool
        if icl_mode:
            self._build_context_pool(context_pool_size)
    
    def _build_context_pool(self, size: int):
        """建立 context examples 池。"""
        import random
        self.context_pool = []
        
        # 每個角度取 size // len(angles) 個樣本
        samples_per_angle = max(1, size // len(self.angles))
        
        for angle in self.angles:
            angle_samples = [
                item for item in self.index if item[1] == angle
            ]
            selected = random.sample(
                angle_samples,
                min(samples_per_angle, len(angle_samples))
            )
            
            for path, angle_deg, _ in selected:
                wav = np.load(path)
                Y = self._compute_spectrogram(wav)
                self.context_pool.append({
                    "Y": Y.numpy(),
                    "angle": int(angle_deg)
                })
    
    def _compute_spectrogram(self, wav: np.ndarray):
        """計算頻譜（抽取為方法以便重用）。"""
        from nmf_localizer.utils.audio_utils import AudioProcessor
        
        freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
            wav, fs=self.fs, nperseg=self.n_fft, window=self.window
        )
        mask = (freqs >= self.freq_min) & (freqs <= self.freq_max)
        mag_band = magnitude[mask, :].astype(np.float32)
        return torch.from_numpy(mag_band)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """返回帶有 prompt 的樣本。"""
        path, angle_deg, _ = self.index[idx]
        wav = np.load(path)
        Y = self._compute_spectrogram(wav)
        
        # 構建 prompt
        if self.icl_mode:
            # 隨機選取 context examples（排除當前角度）
            import random
            context_candidates = [
                ex for ex in self.context_pool
                if ex["angle"] != int(angle_deg)
            ]
            context_examples = random.sample(
                context_candidates,
                min(3, len(context_candidates))
            )
            prompt = self.prompt_builder.build_icl_prompt(
                Y.numpy(), context_examples
            )
        else:
            prompt = self.prompt_builder.build_prompt(Y.numpy())
        
        return {
            "Y": Y,
            "angle_deg": float(angle_deg),
            "angle_index": self.angles.index(float(angle_deg)),
            "path": str(path),
            "prompt": prompt,  # 新增
        }
```

---

### 階段 4：訓練腳本適配

#### 4.1 修改 train_reward_model_lora.py

```python
# scripts/train_reward_model_lora.py (主要修改)

# 在 main() 開頭加入：

def main():
    # ... (現有參數解析)
    
    # === 新增：Multi-Modal Tokenizer 設定 ===
    ap.add_argument("--use-multi-modal", action="store_true",
                    help="Enable multi-modal tokenization (Atoms + Direction)")
    ap.add_argument("--icl-mode", action="store_true",
                    help="Enable In-Context Learning mode")
    args = ap.parse_args()
    
    # ... (現有代碼)
    
    # === 載入 NMF 字典和轉移函數 ===
    W = torch.load(args.w_path)["W"]  # (F, K)
    H = load_H(args.tf_path)  # (F, D)
    
    # === 建立 Multi-Modal Tokenizers ===
    from doa_rl.features.tokenizers import (
        PatchTokenizer,
        NMFAtomTokenizer,
        DirectionProjectionTokenizer,
    )
    from doa_rl.features.prompt_builder import (
        MultiModalPromptBuilder,
        PromptConfig,
    )
    
    patch_tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
    
    if args.use_multi_modal:
        atom_tok = NMFAtomTokenizer(W.numpy(), top_k=8)
        dir_tok = DirectionProjectionTokenizer(
            H.numpy(),
            direction_angles,
            metric="correlation"
        )
    else:
        atom_tok = None
        dir_tok = None
    
    config = PromptConfig(
        use_patches=True,
        use_atoms=args.use_multi_modal,
        use_directions=args.use_multi_modal,
        include_context=args.icl_mode,
    )
    
    prompt_builder = MultiModalPromptBuilder(
        patch_tok, atom_tok, dir_tok, config
    )
    
    # === 使用新的 Dataset ===
    from doa_rl.data import DoAICLDataset
    
    ds = DoAICLDataset(
        args.data_root,
        direction_angles,
        prompt_builder=prompt_builder,
        icl_mode=args.icl_mode,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    
    # 後續訓練使用 batch["prompt"] 而非手動 tokenize
```

---

## 📊 執行流程對比

### Before（現有系統）

```
音訊數據 (clip_*.npy)
    ↓
STFT → Y(F,N)
    ↓
PatchTokenizer → [<P_0_0_5>, <P_0_1_8>, ...]  (單一模態)
    ↓
Tokenizer.encode() → IDs
    ↓
訓練腳本
```

### After（Multi-Modal ICL 系統）

```
音訊數據 (clip_*.npy)
    ↓
STFT → Y(F,N)
    ↓
    ├─→ PatchTokenizer → [<P_i_j_level>, ...]
    ├─→ NMFAtomTokenizer → [<AT_5:12>, <AT_23:8>, ...]
    └─→ DirectionProjectionTokenizer → [<R_090:14>, <R_095:11>, ...]
    ↓
Multi-Modal Prompt Builder
    ↓
[CLS] <R_090:14> <AT_5:12> <P_0_0_5> ... → 物理引導 + 細節
    ↓
(Optional) ICL Context: [CLS] example1 [SEP] example2 [SEP] query
    ↓
Tokenizer.encode() → IDs
    ↓
訓練腳本（無需修改核心邏輯）
```

---

## 🛠️ 實作步驟

### Step 1: 實作新的 Tokenizers（1-2 天）

```bash
# 建立新檔案
touch doa_rl/features/tokenizers_extended.py
touch doa_rl/features/prompt_builder.py
```

**Priority**:
1. ✅ NMFAtomTokenizer
2. ✅ DirectionProjectionTokenizer  
3. ✅ MultiModalPromptBuilder

### Step 2: 擴展 Dataset（1 天）

```bash
# 修改 doa_rl/data.py
# 新增 DoAICLDataset 類別
```

### Step 3: 更新 HF Tokenizer Vocab（半天）

```python
# doa_rl/hf/tokenizer.py

def _build_vocab_extended(direction_tokens: Sequence[str], 
                          n_atoms: int = 64) -> List[str]:
    """擴展詞彙表以包含 Atom 和 Direction Projection tokens。"""
    vocab: List[str] = []
    
    # Special tokens
    vocab.extend([_PAD_TOKEN, _BOS_TOKEN, _EOS_TOKEN, _UNK_TOKEN, "<CLS>", "<SEP>"])
    
    # Patch tokens
    vocab.extend(_generate_patch_tokens())
    
    # NMF Atom tokens: <AT_id:level>
    for atom_id in range(n_atoms):
        for level in range(16):
            vocab.append(f"<AT_{atom_id}:{level}>")
    
    # Direction Projection tokens: <R_angle:level>
    for angle in range(0, 181, 5):  # 0-180 度，每 5 度一個
        for level in range(16):
            vocab.append(f"<R_{angle:03d}:{level}>")
    
    # Direction tokens (原有)
    vocab.extend(direction_tokens)
    
    return vocab
```

### Step 4: 適配訓練腳本（1 天）

為三個訓練腳本加入 `--use-multi-modal` 和 `--icl-mode` 參數。

### Step 5: 測試與驗證（1-2 天）

```bash
# Smoke test with multi-modal tokens
python scripts/train_reward_model_lora.py \
  --data-root <path> \
  --use-multi-modal \
  --max-samples 10 \
  ...

# Smoke test with ICL
python scripts/train_reward_model_lora.py \
  --data-root <path> \
  --use-multi-modal \
  --icl-mode \
  --max-samples 10 \
  ...
```

---

## 🎯 預期效果

### 1. **更強的物理引導**
- Direction tokens 提供**粗略定位**（物理先驗）
- Atom tokens 捕捉**頻譜結構**
- Patch tokens 保留**細節資訊**

### 2. **In-Context Learning 能力**
```
Few-shot Example:
[CLS]
  例子1: <R_090:15> <AT_5:12> ... <D_090>  [SEP]
  例子2: <R_095:14> <AT_5:11> ... <D_095>  [SEP]
  查詢:   <R_092:14> <AT_5:12> ... → 預測 092°
```

### 3. **跨域遷移**
- 在室內訓練，推廣至戶外（不同 H 矩陣）
- Direction tokens 提供環境適應能力

### 4. **可解釋性**
```
模型預測 90°，可視化：
  - Direction token <R_090:15> 權重最高
  - Atom tokens <AT_5:12>, <AT_23:9> 對應低頻能量
  → 符合物理直覺
```

---

## 📝 總結

### 關鍵創新點

1. **三層 Token 架構**：Physics (Direction) → Structure (Atoms) → Details (Patches)
2. **統一 Prompt 介面**：訓練腳本無需大改，只需替換 Dataset
3. **漸進式擴展**：可選擇性啟用 multi-modal 和 ICL
4. **物理知識注入**：H、W 矩陣直接參與 tokenization

### 開發優先級

**Phase 1（必須）**:
- ✅ NMFAtomTokenizer
- ✅ DirectionProjectionTokenizer
- ✅ MultiModalPromptBuilder
- ✅ DoAICLDataset

**Phase 2（增強）**:
- ⭕ ICL context sampling 策略優化
- ⭕ Dynamic token ordering
- ⭕ Cross-domain adaptation 機制

**Phase 3（研究）**:
- ⭕ Attention visualization
- ⭕ Token importance analysis
- ⭕ Multi-source extension (K>1)

---

## 🚀 下一步行動

1. **立即開始**：實作 `tokenizers_extended.py`
2. **2-3 天內完成**：Bridge 層全部程式碼
3. **1 週內驗證**：Smoke test + 小規模實驗
4. **2 週內評估**：與 baseline 對比效果

讓我知道你想從哪個部分開始實作！🎉
