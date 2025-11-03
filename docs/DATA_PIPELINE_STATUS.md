# Data Pipeline Status Report

**Date**: 2025-11-03  
**Commits**: 
- Design: 0fca4de (DT Atom-Sequence Design)
- Results: 8a59dc3 (g-teacher trajectories)

---

## 🎯 Quick Answer to Your Questions

### Q1: 目前对于资料的处理做到哪裡？

**Answer**: 我们已完成 **Phase 1** (Trajectory Generation)，但尚未实现 **Phase 2** (Atom-Sequence Input)。

**当前状态**:
- ✅ **Trajectory 数据**: 已生成并验证 (`results/dt_traj_g_full/`)
- ✅ **DT 训练器**: 可运行，但使用**旧的加法模式** (不是 atom-sequence)
- ❌ **Atom-Sequence Input**: 仅设计文档，未实现代码

### Q2: 我們是否已經完成可以給DT閱讀的資料了？

**Answer**: **部分完成**。有两种理解：

1. **从旧 DT 架构角度** (DTMinPointer):
   - ✅ **可以训练**: `trajectories.jsonl` 已足够
   - ✅ **格式正确**: Residual, RTG, labels 都可用
   - ✅ **动态构建**: 训练时从 trajectory 重建 `r_t`
   - ❌ **非最佳**: 不使用 atom-sequence（缺少完整状态）

2. **从新 DT 架构角度** (DTMinPointerV2, commit 0fca4de 设计):
   - ✅ **Trajectory 可重用**: 同样的 `trajectories.jsonl`
   - ❌ **需要新模型**: DTMinPointerV2 未实现
   - ❌ **需要新 forward()**: 构建 atom-sequence 的代码未写

### Q3: 或是在哪一部執行的時候會自動做這件事？

**Answer**: **训练时动态构建**，在 `dt_pointer_ldv.py` 的 main loop 中。

**当前流程** (已实现):
```python
# 位置: scripts/dt_pointer_ldv.py, lines ~350-380
for obj in trajs:  # 遍历 trajectories.jsonl
    y = load_y_for_path(path)  # 加载原始信号
    actions_prev = []
    
    for t, s in enumerate(steps):  # 每个时间步
        # 1. 重建 residual r_t (从 y 和之前的 actions)
        r_t = recompute_r_t(y, D, actions_prev)
        
        # 2. 构建输入特征 (当前加法模式)
        R_list.append(r_t)                    # (F,)
        RTG_list.append([rtg_resid, rtg_acc]) # (2,)
        STEP_list.append([t/K, (K-t)/K])      # (2,)
        
        # 3. 提取标签
        lab_e.append(s['expert'])  # 监督信号
        lab_m.append(s['atom'])
        
        # 4. 更新已选动作
        actions_prev.append(s['dict_index'])
```

**需要的新流程** (未实现):
```python
# DTMinPointerV2 的 forward() 应该构建:
for t, s in enumerate(steps):
    r_t = recompute_r_t(y, D, actions_prev)
    
    # 构建 atom-sequence (每步 2+P 个 tokens)
    RTG_tok = proj_rtg([rtg_resid, rtg_acc])  # (d,)
    R_tok = P_R(r_t) + type_R                  # (d,)
    Atom_toks = [P_D(D[:, j]) + type_D for j in range(P)]  # P × (d,)
    
    # 组合成序列
    seq_t = torch.stack([RTG_tok, R_tok] + Atom_toks)  # (2+P, d)
    seq_list.append(seq_t)

# 最终序列: (K, 2+P, d) 或展平为 (K×(2+P), d)
```

---

## 📊 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Trajectory Generation (✅ DONE - Commit 8a59dc3)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Raw LDV data (.npy files)                               │
│    ↓                                                             │
│  offline_dt_dataset.py                                          │
│    • Teacher policy: g (classic OMP)                            │
│    • For each sample:                                           │
│        - Load Y (STFT magnitude)                                │
│        - Build D = H ⊙ W                                        │
│        - Run K=6 greedy steps                                   │
│        - Record: (expert, atom, resid_sq, rtg, ...)            │
│    ↓                                                             │
│  Output: trajectories.jsonl                                     │
│    • Format: One JSON per sample                                │
│    • Size: 111 samples × K=6 steps = 666 step records          │
│    • Storage: ~171 KB (compact)                                 │
│                                                                  │
│  ✅ Status: Complete                                            │
│  ✅ Location: results/dt_traj_g_full/trajectories.jsonl        │
│  ✅ Verification: All tests passed                              │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2: DT Training Data Construction (⚠️  PARTIAL)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: trajectories.jsonl + manifest.json                      │
│    ↓                                                             │
│  dt_pointer_ldv.py (main loop)                                  │
│    • Load trajectories                                          │
│    • Rebuild D from manifest (H, W, selected_indices)           │
│    • For each trajectory:                                       │
│        - Load y from dataset (using path)                       │
│        - For each step t:                                       │
│            ├─ Recompute r_t = y - D_S @ lstsq(D_S, y)          │
│            ├─ Extract RTG from trajectory record                │
│            ├─ Compute step encoding [t/K, (K-t)/K]             │
│            └─ Extract labels (expert, atom)                     │
│    ↓                                                             │
│  Current Output (DTMinPointer):                                 │
│    • R_seq: (B, K, F)     - Residuals                           │
│    • RTG_seq: (B, K, 2)   - RTG targets                         │
│    • STEP_seq: (B, K, 2)  - Step/budget encoding                │
│    • Labels: (B, K) expert, (B, K) atom                         │
│    ↓                                                             │
│  forward() constructs:                                          │
│    h = LN(P_R(R) + type_R + proj_rtg(RTG) + proj_step(STEP))   │
│    Sequence shape: (B, K, d)  ← Only K tokens per sample       │
│                                                                  │
│  ⚠️  Issues:                                                     │
│    • Atoms NOT in sequence (stored in KD_em buffer)             │
│    • Token addition instead of sequential attention             │
│    • Cannot learn atom-specific attention patterns              │
│                                                                  │
│  ✅ Status: Implemented & working                               │
│  ❌ Limitation: Not the atom-sequence design from 0fca4de       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Step 2 (Target): DT Atom-Sequence Input (❌ NOT DONE)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: Same trajectories.jsonl + manifest.json                 │
│    ↓                                                             │
│  DTMinPointerV2.forward() (planned, not implemented)            │
│    • For each time step t:                                      │
│        ├─ RTG_tok = proj_rtg([rtg_resid, rtg_acc])   (d,)      │
│        ├─ R_tok = P_R(r_t) + type_R                  (d,)      │
│        └─ Atom_toks = [P_D(d_j) + type_D for j in range(P)]    │
│            └─ P tokens, each (d,)                               │
│    ↓                                                             │
│  Sequence construction:                                         │
│    seq_t = [RTG_tok, R_tok, Atom_0_tok, ..., Atom_{P-1}_tok]   │
│    Shape per step: (2+P, d) = (2+296, d) = (298, d)            │
│    ↓                                                             │
│  Full sequence (K steps):                                       │
│    Option A: (B, K, 2+P, d)      ← 4D tensor                   │
│    Option B: (B, K×(2+P), d)     ← Flatten to 3D               │
│                                                                  │
│  Attention mechanism:                                           │
│    • Causal mask within episode: step t can attend to step ≤t  │
│    • Within-step mask: Full attention to all 298 tokens        │
│    • Learns: d_j^T · r_t implicitly via attention weights      │
│                                                                  │
│  Action selection:                                              │
│    • Query from R_tok position                                  │
│    • Attention scores over Atom positions → argmax             │
│                                                                  │
│  ❌ Status: NOT IMPLEMENTED                                     │
│  ❌ Blockers:                                                    │
│     1. DTMinPointerV2 class not written                         │
│     2. Sequence construction logic not coded                    │
│     3. Mask generation for 4D structure needed                  │
│     4. Pointer head needs redesign                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Trajectory vs. DT Input Sequence: Conceptual Separation

### 术语定义 (避免混淆)

| 术语 | 英文 | 含义 | 文件格式 | 维度 | 生成阶段 |
|------|------|------|----------|------|----------|
| **Trajectory** | Trajectory | 教师策略的决策序列记录 | JSON (紧凑) | Per-step dict | Offline (预生成) |
| **DT Input Sequence** | DT Input Sequence | Transformer 输入的 token 序列 | Tensor | (B, seq_len, d) | Training (动态) |
| **Episode** | Episode | 同义于 Trajectory (RL 术语) | - | - | - |
| **Token Sequence** | Token Sequence | 同义于 DT Input Sequence | - | - | - |

### 沟通建议

**使用场景区分**:

1. **讨论数据生成时** → 用 "Trajectory"
   - ✅ "We generated 111 trajectories using g-teacher"
   - ✅ "Trajectory includes step-by-step actions and RTGs"
   - ✅ "Stored in trajectories.jsonl"

2. **讨论 DT 模型输入时** → 用 "Input Sequence" 或 "Token Sequence"
   - ✅ "DT input sequence has 298 tokens per step"
   - ✅ "We construct the token sequence during training"
   - ✅ "Sequence shape is (B, K×(2+P), d)"

3. **讨论两者关系时** → 明确说明转换
   - ✅ "Trajectory is **converted to** DT input sequence"
   - ✅ "Training loop reads trajectory and **builds** token sequence"
   - ✅ "Atom-sequence design requires **new sequence construction**"

### 示例对话

**❌ 易混淆**:
> "我们的 trajectory 有 298 个 tokens"  
> → 混淆了存储格式和 Transformer 输入

**✅ 清晰**:
> "我们的 trajectory 记录了 K=6 步的动作。训练时，每步会被转换为 298 个 tokens 的 DT input sequence（2+296: RTG + Residual + Atoms）"

---

## 🚧 Implementation Roadmap

### Phase 1: ✅ DONE (Commit 8a59dc3)
- [x] Generate trajectories using g-teacher
- [x] Validate trajectory format
- [x] Store in trajectories.jsonl with manifest
- [x] Verify reproducibility

### Phase 2: ⚠️  PARTIAL (Current dt_pointer_ldv.py)
- [x] Load trajectories
- [x] Rebuild D from manifest
- [x] Reconstruct r_t dynamically
- [x] Train DT with additive tokens (baseline)
- [ ] **NOT DONE**: Atom-sequence input
- [ ] **NOT DONE**: Sequential attention over atoms

### Phase 3: ❌ TODO (DTMinPointerV2)
- [ ] Implement DTMinPointerV2 class
- [ ] Build atom-sequence construction
- [ ] Design causal + within-step mask
- [ ] Pointer head over atom positions
- [ ] Training loop integration
- [ ] Smoke test (3 samples, 1 epoch)
- [ ] Full training (111 samples, 480 epochs)

---

## 📝 Current vs. Target Implementation

### Current (DTMinPointer - Working but Limited)

**File**: `scripts/dt_pointer_ldv.py`

**Data construction** (lines ~350-380):
```python
# Per step, extract from trajectory
R_list.append(r_t)                      # (F,) residual
RTG_list.append([rtg_resid, rtg_acc])   # (2,) targets
STEP_list.append([t/K, (K-t)/K])        # (2,) encoding
lab_e.append(expert)                    # Label
lab_m.append(atom)

# Batch construction
R_seq = torch.stack(R_list)       # (K, F)
RTG_seq = torch.stack(RTG_list)   # (K, 2)
STEP_seq = torch.stack(STEP_list) # (K, 2)
```

**Forward pass** (lines ~230-250):
```python
def forward(R_seq, RTG_seq, STEP_seq, causal_mask):
    # Token projections
    r_tok = self.P_R(R_seq) + self.type_R      # (B, K, d)
    rtg_tok = self.proj_rtg(RTG_seq)           # (B, K, d)
    step_tok = self.proj_step(STEP_seq)        # (B, K, d)
    
    # Additive combination (not sequential!)
    h = self.ln(r_tok + rtg_tok + step_tok)    # (B, K, d)
    
    # Transformer encoder
    Ht = self.encoder(h, mask=causal_mask)     # (B, K, d)
    
    # Query for pointer head
    Q = self.Wq(Ht)                            # (B, K, d)
    
    # Dot product with precomputed atom keys
    qk = einsum('bkd,emd->bkem', Q, self.KD_em)  # (B,K,E,M)
    
    # Expert scores (L2 aggregation)
    scores_e = sqrt(sum(qk.abs()^2, dim=M))    # (B, K, E)
    
    return scores_e, qk
```

**Limitations**:
1. ❌ Atoms are NOT in the sequence (stored in `KD_em` buffer)
2. ❌ Cannot learn attention over atoms (fixed keys)
3. ❌ Token addition loses sequential structure
4. ❌ No explicit state representation per atom

---

### Target (DTMinPointerV2 - Planned, Not Implemented)

**File**: `scripts/dt_pointer_ldv_v2.py` (does not exist)

**Data construction** (pseudo-code):
```python
# Per step, build atom-sequence
for t, s in enumerate(steps):
    r_t = recompute_r_t(y, D, actions_prev)
    
    # Construct token sequence for this step
    RTG_tok = proj_rtg([rtg_resid, rtg_acc])   # (d,)
    R_tok = P_R(r_t) + type_R                   # (d,)
    
    # All P atoms as individual tokens
    Atom_toks = []
    for j in range(P):
        atom_tok = P_D(D[:, j]) + type_D        # (d,)
        Atom_toks.append(atom_tok)
    
    # Stack into step sequence
    seq_t = torch.stack([RTG_tok, R_tok] + Atom_toks)  # (2+P, d)
    seq_list.append(seq_t)
    
    actions_prev.append(s['dict_index'])

# Full episode sequence
seq_episode = torch.stack(seq_list)  # (K, 2+P, d)
```

**Forward pass** (pseudo-code):
```python
def forward(seq_episode, causal_mask_4d):
    # seq_episode: (B, K, 2+P, d)
    B, K, seq_len, d = seq_episode.shape
    
    # Option A: Flatten to 3D for standard Transformer
    seq_flat = seq_episode.view(B, K*(2+P), d)  # (B, K×seq_len, d)
    
    # Transformer encoder with causal mask
    H = self.encoder(seq_flat, mask=causal_mask_4d)  # (B, K×seq_len, d)
    
    # Reshape back to per-step
    H_steps = H.view(B, K, 2+P, d)  # (B, K, 2+P, d)
    
    # Extract residual token representations (position 1 in each step)
    H_R = H_steps[:, :, 1, :]  # (B, K, d)
    
    # Extract atom token representations (positions 2 to 2+P-1)
    H_atoms = H_steps[:, :, 2:, :]  # (B, K, P, d)
    
    # Attention-based action selection
    # Query from residual position, keys from atom positions
    Q = self.Wq(H_R)                        # (B, K, d)
    K_atoms = self.Wk(H_atoms)              # (B, K, P, d)
    
    # Dot product: Q @ K^T
    scores = einsum('bkd,bkpd->bkp', Q, K_atoms) / sqrt(d)  # (B, K, P)
    
    # Reshape to (E, M) for hierarchical selection
    scores_em = scores.view(B, K, E, M)     # (B, K, E, M)
    
    # Expert-level aggregation
    scores_e = sqrt(sum(scores_em.abs()^2, dim=M))  # (B, K, E)
    
    return scores_e, scores_em
```

**Key differences**:
1. ✅ Atoms ARE in the sequence (explicit tokens)
2. ✅ Attention learns to compute d_j^T · r_t
3. ✅ Sequential structure preserved
4. ✅ Full state representation
5. ⚠️  Longer sequences (298 vs 1 token/step)

---

## 🎯 Immediate Action Items

### To Use Current Implementation (DTMinPointer)

**You can train NOW**:
```bash
python scripts/dt_pointer_ldv.py \
    --traj_dir results/dt_traj_g_full \
    --out_dir results/dt_min_g_baseline \
    --epochs 480 \
    --batch_size 16 \
    --lr 3e-3 \
    --d_model 128 \
    --nhead 2 \
    --nlayers 1 \
    --device cpu \
    --test_split 0.2 \
    --split_seed 42
```

**What you get**:
- ✅ Working baseline DT
- ✅ Can compare to teacher (g) performance
- ✅ Validates trajectory format
- ❌ Not using atom-sequence design

### To Implement Atom-Sequence (DTMinPointerV2)

**Steps needed**:

1. **Create new model file**:
   ```bash
   cp scripts/dt_pointer_ldv.py scripts/dt_pointer_ldv_v2.py
   ```

2. **Modify DTMinPointer → DTMinPointerV2**:
   - Change sequence construction (see pseudo-code above)
   - Add 4D mask generation
   - Update pointer head logic

3. **Smoke test**:
   ```bash
   python scripts/dt_pointer_ldv_v2.py \
       --traj_dir results/dt_traj_g_full \
       --subset_angles "0,5,10" \
       --epochs 1 \
       --batch_size 2
   ```

4. **Full training**:
   - After smoke test passes
   - Same hyperparameters as baseline

**Estimated effort**: 2-4 hours for experienced developer

---

## 📚 Summary Table

| Aspect | Trajectory | DT Input Sequence (Current) | DT Input Sequence (Target) |
|--------|-----------|----------------------------|----------------------------|
| **生成阶段** | Offline (预生成) | Training (动态) | Training (动态) |
| **存储格式** | JSON (trajectories.jsonl) | Tensor (in-memory) | Tensor (in-memory) |
| **数据量** | 111 samples × K=6 steps | Same data, different shape | Same data, different shape |
| **每步大小** | ~250 bytes (JSON) | 1 token (d=128) | 298 tokens (2+P) |
| **总 tokens** | N/A | K tokens/episode | K×(2+P) tokens/episode |
| **包含 atoms?** | No (仅 indices) | No (buffer 中) | Yes (显式 tokens) |
| **实现状态** | ✅ Done (8a59dc3) | ✅ Done (dt_pointer_ldv.py) | ❌ Not done |
| **可用性** | ✅ 可复现 | ✅ 可训练 | ❌ 需实现 |

---

## 🔍 Terminology Cheat Sheet

**避免混淆的用词指南**:

| 想说的概念 | ✅ 推荐用词 | ❌ 易混淆用词 |
|-----------|------------|--------------|
| 教师策略的决策记录 | "trajectory", "episode" | "sequence" (太泛) |
| Transformer 的输入 | "DT input sequence", "token sequence" | "trajectory" (指存储) |
| 每步的动作记录 | "step record", "action at step t" | "token" (是表示) |
| 298 个 tokens | "tokens per step in DT input" | "trajectory has 298 tokens" |
| JSON 文件 | "trajectories.jsonl" | "DT input file" |
| 转换过程 | "construct sequence from trajectory" | "load sequence" |

---

## 💡 Recommendations

### For Current Work:
1. ✅ **可以开始训练**: 使用当前 `dt_pointer_ldv.py`
2. ✅ **建立 baseline**: 在 g-teacher trajectories 上训练
3. ✅ **验证流程**: 确认 trajectory → training 流程正确

### For Future Work (Atom-Sequence):
1. ⚠️  **评估必要性**: 当前模型是否已满足需求？
2. ⚠️  **权衡计算**: 298 tokens/step vs 1 token/step (速度差异)
3. ⚠️  **渐进实现**: 先 smoke test，再 full training

### Communication:
1. ✅ 明确区分 "trajectory" (数据) 和 "sequence" (模型输入)
2. ✅ 讨论时说明是哪个阶段 (offline generation vs online training)
3. ✅ 使用术语表避免歧义

---

**需要我帮你实现 DTMinPointerV2 吗？还是先用当前的 baseline 跑一次训练看效果？**
