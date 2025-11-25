# Physics-Informed Decision Transformer 實作計畫

## 目標
將目前的 DTMin（行為複製）升級為符合 Decision Transformer 原論文精神的 **Physics-Informed DT**，同時保留物理創新（殘差作為充分統計量、凍結字典）。

---

## 現況分析

### 目前架構問題
```
┌─────────────────────────────────────────────────────────────────────┐
│                        目前 DTMin 架構                              │
├─────────────────────────────────────────────────────────────────────┤
│  輸入: h_seq = P_R(r_t) + pos_embed                                 │
│  注意力: Bidirectional (違反因果假設)                                │
│  監督: CE(expert_pred, expert_omp) + CE(atom_pred, atom_omp)        │
│  缺少: RTG conditioning (無法控制生成品質)                           │
│  AWR: 基於 teacher 正確率，非環境 reward                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 目標架構
```
┌─────────────────────────────────────────────────────────────────────┐
│                   Physics-Informed DT 架構                          │
├─────────────────────────────────────────────────────────────────────┤
│  輸入: h_seq = P_R(r_t) + RTG_proj(rtg_t) + step_embed              │
│  注意力: Causal Mask (符合序列決策)                                  │
│  監督: CE(expert_pred, expert_omp) + CE(atom_pred, atom_omp)        │
│  RTG: rtg_t = [mean_residual, cumulative_accuracy]                  │
│  推論: 可通過設定 RTG 控制生成軌跡品質                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 實作階段

### Phase 1: 定義環境 Reward 函數
**檔案**: `doa_rl/domain_randomization/generator.py`

**目標**: 定義明確的 reward function，使 RTG 有意義

```python
# 提議的 reward function (每步)
def compute_step_reward(step_info: dict, angle_idx: int) -> float:
    """
    r_t = α * is_correct_expert + β * (prev_resid - curr_resid) / init_resid

    Components:
    - is_correct_expert: 1 if expert == angle_idx, else 0 (稀疏獎勵)
    - residual_reduction: 正規化的殘差下降量 (稠密獎勵)
    """
    alpha = 1.0  # 正確角度權重
    beta = 0.1   # 殘差下降權重

    correct_bonus = alpha * float(step_info["is_correct_expert"])
    resid_bonus = beta * step_info["delta_resid_sq"] / step_info.get("init_resid", 1.0)

    return correct_bonus + resid_bonus
```

**RTG 計算**:
```python
# Return-to-Go = 從 t 到 T 的累積 reward
rtg_t = sum(r_i for i in range(t, T))
```

**修改內容**:
1. 在 `_run_omp()` 中記錄 `init_resid`
2. 新增 `_compute_rewards()` 方法計算每步 reward
3. 修改 `_convert_to_embeddings()` 計算真正的 RTG

---

### Phase 2: 修改 Dataset 載入 RTG
**檔案**: `doa_rl/domain_randomization/dataset.py`

**目標**: 讓 Dataset 載入並返回 `rtg_seq`

```python
# 修改 _load_all()
def _load_all(self) -> None:
    for shard_dir in self.shard_dirs:
        data = np.load(npz_path)
        # 新增載入 rtg_seq
        rtg_seq = data["rtg_seq"] if "rtg_seq" in data else None
        step_seq = data["step_seq"] if "step_seq" in data else None

        for i in range(h_seq.shape[0]):
            sample = {
                "h_seq": torch.from_numpy(h_seq[i]).float(),
                "expert_gt": torch.from_numpy(expert_gt[i]).long(),
                "atom_gt": torch.from_numpy(atom_gt[i]).long(),
                "angle_gt": int(angle_gt[i]),
                "rtg_seq": torch.from_numpy(rtg_seq[i]).float() if rtg_seq is not None else None,
                "step_seq": torch.from_numpy(step_seq[i]).float() if step_seq is not None else None,
                ...
            }
```

**修改 collate function**:
```python
def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    result = {
        "h_seq": torch.stack([b["h_seq"] for b in batch]),
        "expert_gt": torch.stack([b["expert_gt"] for b in batch]),
        "atom_gt": torch.stack([b["atom_gt"] for b in batch]),
        "angle_gt": torch.tensor([b["angle_gt"] for b in batch], dtype=torch.long),
    }
    # 條件載入 RTG
    if batch[0].get("rtg_seq") is not None:
        result["rtg_seq"] = torch.stack([b["rtg_seq"] for b in batch])
    if batch[0].get("step_seq") is not None:
        result["step_seq"] = torch.stack([b["step_seq"] for b in batch])
    return result
```

---

### Phase 3: 修改模型架構
**檔案**: `scripts/train_angle_range_dtmin.py` 或新建 `doa_rl/models/physics_dt.py`

#### 3.1 新增 RTG Projection Layer
```python
class PhysicsInformedDT(nn.Module):
    """Physics-Informed Decision Transformer.

    Key differences from TinyDTMin:
    - RTG conditioning: P_RTG(rtg_t) added to token embedding
    - Causal attention mask: Prevents future information leakage
    - Step embedding: Normalized timestep information
    """

    def __init__(self, d_model: int, nhead: int, nlayers: int,
                 E: int, M: int, max_K: int, dropout: float = 0.1,
                 rtg_dim: int = 2):
        super().__init__()

        # Transformer encoder with causal mask support
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # Positional encoding
        self.pos = nn.Embedding(max_K, d_model)

        # RTG projection (NEW)
        self.rtg_proj = nn.Sequential(
            nn.Linear(rtg_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Layer norm and heads
        self.norm = nn.LayerNorm(d_model)
        self.expert_head = nn.Linear(d_model, E)
        self.atom_head = nn.Linear(d_model, M)

        # Store max_K for causal mask generation
        self.max_K = max_K

    def _generate_causal_mask(self, K: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask.

        Returns: (K, K) mask where mask[i,j] = True means position i
                 cannot attend to position j (j > i is masked)
        """
        mask = torch.triu(torch.ones(K, K, device=device), diagonal=1).bool()
        return mask

    def forward(self, h_seq: torch.Tensor,
                rtg_seq: torch.Tensor = None,
                use_causal_mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        B, K, _ = h_seq.shape
        device = h_seq.device

        # Positional embedding
        pos_idx = torch.arange(K, device=device).unsqueeze(0).expand(B, K)
        h = h_seq + self.pos(pos_idx)

        # RTG conditioning (if provided)
        if rtg_seq is not None:
            rtg_embed = self.rtg_proj(rtg_seq)
            h = h + rtg_embed

        # Causal mask for training
        mask = self._generate_causal_mask(K, device) if use_causal_mask else None

        # Transformer encoding
        h = self.encoder(h, mask=mask)
        h = self.norm(h)

        # Prediction heads
        expert_logits = self.expert_head(h)
        atom_logits = self.atom_head(h)

        return expert_logits, atom_logits
```

#### 3.2 Causal Mask 視覺化
```
Causal Attention Mask (K=5):
     t=0  t=1  t=2  t=3  t=4
t=0   ✓    ✗    ✗    ✗    ✗
t=1   ✓    ✓    ✗    ✗    ✗
t=2   ✓    ✓    ✓    ✗    ✗
t=3   ✓    ✓    ✓    ✓    ✗
t=4   ✓    ✓    ✓    ✓    ✓

✓ = can attend, ✗ = masked (cannot attend)

目前 TinyDTMin (Bidirectional):
     t=0  t=1  t=2  t=3  t=4
t=0   ✓    ✓    ✓    ✓    ✓   ← 問題：t=0 可以看到 t=4
t=1   ✓    ✓    ✓    ✓    ✓
...
```

---

### Phase 4: 修改訓練循環
**檔案**: `scripts/train_angle_range_dtmin.py`

#### 4.1 更新 train_epoch()
```python
def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                device: torch.device, mode: str, label_mode: str, use_atom_loss: bool,
                use_rtg: bool = True) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        h_seq = batch["h_seq"].to(device)
        expert_gt = batch["expert_gt"].to(device)
        atom_gt = batch["atom_gt"].to(device)

        # RTG conditioning (NEW)
        rtg_seq = batch.get("rtg_seq")
        if rtg_seq is not None and use_rtg:
            rtg_seq = rtg_seq.to(device)
        else:
            rtg_seq = None

        optimizer.zero_grad()

        # Forward with RTG and causal mask
        expert_logits, atom_logits = model(h_seq, rtg_seq=rtg_seq, use_causal_mask=True)

        # Loss computation (same as before)
        target_expert = compute_targets(batch, label_mode)
        mask = _mask_from_batch(batch, device)

        expert_ce = F.cross_entropy(
            expert_logits.transpose(1, 2), target_expert,
            reduction="none", ignore_index=-100
        )
        loss = (expert_ce * mask).sum() / mask.sum().clamp_min(1)

        if use_atom_loss:
            atom_mask = mask & (atom_gt != -100)
            atom_ce = F.cross_entropy(
                atom_logits.transpose(1, 2), atom_gt,
                reduction="none", ignore_index=-100
            )
            loss = loss + (atom_ce * atom_mask).sum() / atom_mask.sum().clamp_min(1)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1

    return {"loss": total_loss / max(total_batches, 1)}
```

#### 4.2 新增命令列參數
```python
ap.add_argument("--use-rtg", action="store_true", default=True,
                help="Enable RTG conditioning (Physics-Informed DT mode)")
ap.add_argument("--use-causal-mask", action="store_true", default=True,
                help="Use causal attention mask for sequential decision making")
ap.add_argument("--rtg-dim", type=int, default=2,
                help="RTG embedding dimension (default: 2 for [resid, accuracy])")
```

---

### Phase 5: 實作自回歸推論
**檔案**: 新建 `doa_rl/inference/autoregressive_inference.py`

**目標**: 實作可控制的自回歸生成

```python
class AutoregressiveInference:
    """Autoregressive inference with RTG control.

    使用方式:
    - 設定高 RTG: 生成高品質軌跡（高 expert accuracy）
    - 設定低 RTG: 生成較差軌跡（用於研究 failure modes）
    """

    def __init__(self, model: PhysicsInformedDT, dictionary: torch.Tensor,
                 device: torch.device):
        self.model = model
        self.dictionary = dictionary  # D = H ⊗ W
        self.device = device

    def generate_trajectory(self, y: torch.Tensor,
                           target_rtg: float = 1.0,
                           max_steps: int = 5,
                           temperature: float = 1.0) -> List[Dict]:
        """
        Generate OMP trajectory autoregressively.

        Args:
            y: Input spectrum [F]
            target_rtg: Desired return-to-go (higher = better quality)
            max_steps: Maximum OMP steps
            temperature: Sampling temperature (1.0 = deterministic)

        Returns:
            List of step dictionaries with expert, atom, residual
        """
        self.model.eval()
        steps = []

        # Initialize residual
        r_t = y.clone()
        h_seq_list = []
        rtg_seq_list = []

        with torch.no_grad():
            for t in range(max_steps):
                # Compute residual embedding
                h_t = self._embed_residual(r_t)
                h_seq_list.append(h_t)

                # Compute RTG for this step (decaying based on expected future reward)
                remaining_steps = max_steps - t
                rtg_t = target_rtg * (remaining_steps / max_steps)
                rtg_seq_list.append(torch.tensor([rtg_t, 0.0], device=self.device))

                # Stack history
                h_seq = torch.stack(h_seq_list, dim=0).unsqueeze(0)  # [1, t+1, d]
                rtg_seq = torch.stack(rtg_seq_list, dim=0).unsqueeze(0)  # [1, t+1, 2]

                # Forward pass (with causal mask, only need last position output)
                expert_logits, atom_logits = self.model(h_seq, rtg_seq=rtg_seq)

                # Sample or argmax from last position
                expert_probs = F.softmax(expert_logits[0, -1] / temperature, dim=-1)
                atom_probs = F.softmax(atom_logits[0, -1] / temperature, dim=-1)

                if temperature < 0.01:  # Deterministic
                    expert_idx = expert_probs.argmax().item()
                    atom_idx = atom_probs.argmax().item()
                else:  # Stochastic sampling
                    expert_idx = torch.multinomial(expert_probs, 1).item()
                    atom_idx = torch.multinomial(atom_probs, 1).item()

                # Get dictionary atom and update residual
                global_atom_idx = expert_idx * self.M + atom_idx
                d_j = self.dictionary[global_atom_idx]  # [F]
                alpha = (r_t @ d_j) / (d_j @ d_j + 1e-8)
                r_t = r_t - alpha * d_j

                steps.append({
                    "step": t,
                    "expert": expert_idx,
                    "atom": atom_idx,
                    "residual_norm": r_t.norm().item(),
                    "expert_probs": expert_probs.cpu().numpy(),
                    "atom_probs": atom_probs.cpu().numpy(),
                })

        return steps
```

---

## 實作順序與依賴關係

```
┌─────────────────────────────────────────────────────────────────────┐
│                        實作依賴圖                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Phase 1: Reward Function                                           │
│      │                                                              │
│      ▼                                                              │
│  Phase 2: Dataset 修改 ──────────────┐                              │
│      │                               │                              │
│      ▼                               ▼                              │
│  Phase 3: 模型架構 ◄─────────────────┘                              │
│      │                                                              │
│      ▼                                                              │
│  Phase 4: 訓練循環                                                   │
│      │                                                              │
│      ▼                                                              │
│  Phase 5: 自回歸推論                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 待修改檔案清單

| 檔案 | 修改內容 | 優先級 |
|------|----------|--------|
| `generator.py` | 新增 reward function, 修正 RTG 計算 | P0 |
| `dataset.py` | 載入 rtg_seq, step_seq | P0 |
| `train_angle_range_dtmin.py` | 新增 PhysicsInformedDT class, 修改 train loop | P0 |
| `train_angle_range_dtmin.py` | 新增 CLI 參數 (--use-rtg, --use-causal-mask) | P1 |
| 新建 `autoregressive_inference.py` | 自回歸推論實作 | P2 |

---

## 驗證計畫

### 單元測試
```python
def test_causal_mask():
    """Verify causal mask prevents future attention."""
    model = PhysicsInformedDT(...)
    h_seq = torch.randn(1, 5, 128)

    # Modify h_seq[0, 4, :] should NOT affect output at t=0
    with torch.no_grad():
        out1, _ = model(h_seq, use_causal_mask=True)
        h_seq[0, 4, :] = torch.randn(128)
        out2, _ = model(h_seq, use_causal_mask=True)

    assert torch.allclose(out1[0, 0], out2[0, 0]), "Causal mask broken!"

def test_rtg_conditioning():
    """Verify RTG affects output distribution."""
    model = PhysicsInformedDT(...)
    h_seq = torch.randn(1, 5, 128)
    rtg_high = torch.tensor([[[1.0, 0.0]]] * 5)
    rtg_low = torch.tensor([[[0.1, 0.0]]] * 5)

    out_high, _ = model(h_seq, rtg_seq=rtg_high)
    out_low, _ = model(h_seq, rtg_seq=rtg_low)

    assert not torch.allclose(out_high, out_low), "RTG has no effect!"
```

### 整合測試
1. **向後相容**: 當 `--use-rtg=False` 時，行為應與原 TinyDTMin 相同
2. **RTG 控制性**: 高 RTG 應產生更高的 voted accuracy
3. **Causal mask 正確性**: 訓練損失應與 bidirectional 相近（資料已包含因果結構）

### 效能基準
| 指標 | 目前 TinyDTMin | 預期 Physics-DT |
|------|---------------|-----------------|
| Voted Accuracy | 0.325 | ≥0.35 |
| Expert Accuracy | ~0.5 | ≥0.5 |
| 訓練時間 (5 epochs) | ~2 min | ~2.5 min |
| 推論可控性 | ✗ | ✓ |

---

## 風險與緩解

| 風險 | 可能性 | 影響 | 緩解措施 |
|------|--------|------|----------|
| Causal mask 降低準確度 | 中 | 中 | 保留 bidirectional 作為 fallback |
| RTG 維度設計不當 | 低 | 中 | 消融實驗：1D vs 2D RTG |
| 自回歸累積誤差 | 高 | 高 | 使用 teacher forcing 比例漸減 |
| 訓練不穩定 | 低 | 中 | 學習率預熱 + gradient clipping |

---

## 成功標準

1. **功能完整**: 支援 RTG conditioning + causal mask
2. **向後相容**: 可關閉新功能回到原 TinyDTMin 行為
3. **可控生成**: 推論時可通過 RTG 控制軌跡品質
4. **效能不降**: Voted accuracy ≥ 現有水準 (0.325)
5. **文檔完整**: 更新 CLAUDE.md 和相關文檔

---

## 預估工作量

- Phase 1: 1-2 小時 (Reward function)
- Phase 2: 0.5 小時 (Dataset 修改)
- Phase 3: 2-3 小時 (模型架構)
- Phase 4: 1 小時 (訓練循環)
- Phase 5: 2-3 小時 (自回歸推論)
- 測試與驗證: 2-3 小時

**總計**: 約 8-12 小時

---

## 附錄：關鍵公式

### Decision Transformer 原始 Loss
```
L_DT = E_τ∼D [ Σ_t -log π(a_t | s_{<t}, a_{<t}, R̂_t) ]

其中:
- τ: trajectory from dataset D
- s_{<t}: states before time t
- a_{<t}: actions before time t
- R̂_t: return-to-go at time t
```

### Physics-Informed DT Loss (我們的版本)
```
L_PI-DT = E_τ∼D [ Σ_t -log π(e_t | r_{<t}, RTG_t) - log π(m_t | r_{<t}, RTG_t) ]

其中:
- r_{<t}: 殘差序列 (替代 state-action 序列)
- RTG_t: [mean_residual_t, cumulative_accuracy_t]
- e_t: expert prediction
- m_t: atom prediction
```

### RTG 計算
```
RTG_t = Σ_{i=t}^{T} r_i

r_t = α * 𝟙[expert_t == angle] + β * (||r_{t-1}|| - ||r_t||) / ||r_0||
```
