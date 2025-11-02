# DT Atom Sequence Design - Development Plan

## 🎯 Objective

Extend the Decision Transformer (DT) input representation to include **all atom vectors as explicit State Tokens**, enabling the Transformer to learn the physical operation `argmax(D.T @ r)` from data.

## 📐 Current State (Commit: 29b3fad)

### Trajectory Format
- **Source**: `doa_rl/trajectories/offline_dt_dataset.py`
- **Teachers**: 
  - `g` (classical OMP): L1 aggregation, `hierarchical_pick_g`
  - `qk` (Transformer OMP): L2 aggregation, learned QK scores
- **Output**: `trajectories.jsonl` with per-step records
  ```json
  {
    "step": t,
    "expert": e,
    "atom": m,
    "dict_index": j,
    "resid_sq": ||r_t||²,
    "delta_resid_sq": Δ||r||²,
    "p_true": p(y_true|ŷ),
    "rtg_resid": RTG_resid,
    "rtg_acc": RTG_acc
  }
  ```

### DT Model Architecture  
- **Source**: `scripts/dt_pointer_ldv.py` (`DTMinPointer`)
- **Current Token Design** (additive, not sequential):
  ```python
  r_tok = self.P_R(R_seq) + self.type_R      # Residual token
  rtg_tok = self.proj_rtg(RTG_seq)           # RTG token
  step_tok = self.proj_step(STEP_seq)        # Step/budget token
  h = self.ln(r_tok + rtg_tok + step_tok)    # Sum and feed to Transformer
  ```
- **Atom Representation**: Precomputed `KD_em` (E, M, d) buffer, used for dot product
- **Action Selection**: Pointer mechanism via `Q @ KD`

### Limitations
1. **No Explicit Atom State**: Atoms are not part of the input sequence; DT cannot learn to attend to atoms based on residual
2. **Token Addition**: Tokens are summed instead of forming a sequence
3. **Fixed Routing**: Action is computed via precomputed keys, not learned from full state context

## 🚀 Target Design: Atom-Sequence State Representation

### Principle
**State Token Sequence at time t**:
```
[RTG_token, Residual_token, Atom_0_token, Atom_1_token, ..., Atom_{P-1}_token]
```

Where:
- **RTG_token**: (rtg_resid, rtg_acc) → d dimensions
- **Residual_token**: r_t (F,) → d dimensions  
- **Atom_j_token**: d_j (F,) → d dimensions, for j ∈ {0, ..., P-1}

**Sequence Length per Time Step**: `2 + P`
- For E=37, M=8: P = 37×8 = 296 → **298 tokens/step**
- For K=6 steps: Total sequence = 6 × 298 = **1788 tokens**

### Physical Meaning
The Transformer will learn:
1. **From RTG**: Understand the goal (minimize residual vs maximize accuracy)
2. **From Residual r_t**: Current state of reconstruction error
3. **From Atoms {d_j}**: Available dictionary columns (physical structure)
4. **Attention Mechanism**: Learns to compute inner products `g_j = d_j^T · r_t` implicitly
5. **Output**: Next atom to select (equivalent to `argmax(g)`)

### Why This Design

#### ✅ Aligns with Physical Intuition
- OMP selection is fundamentally: `j* = argmax_j |d_j^T · r|`
- Transformer with full atom context can learn this operation
- No need for explicit Action Tokens (action is deterministic from state)

#### ✅ Consistent with Project Goals
- **Multi-OMP Variants**: Each teacher (`g`, `qk`, future hybrids) generates trajectories
- **Trajectory → Sequence**: Each trajectory becomes a complete state sequence
- **DT Training**: DT learns conditional policy from diverse trajectories
- **Controllability**: RTG tokens enable goal-directed behavior

#### ✅ Matches Sequence Length to Atom Count (M)
- Current: M=8 (K-means reduced from 50)
- Sequence length: `2 + E×M = 2 + 37×8 = 298`
- Parameterized by M: Reducing M → shorter sequences (e.g., M=4 → 150 tokens/step)

## 🏗️ Implementation Plan

### Phase 1: Model Architecture Extension (DTMinPointerV2)

**New Model**: `DTMinPointerV2` with atom sequence input

#### Changes:
1. **Input Format**:
   ```python
   forward(
       R_seq: Tensor,      # (B, K, F) - Residuals
       RTG_seq: Tensor,    # (B, K, 2) - RTG targets  
       D: Tensor,          # (F, P) - Dictionary (for atom tokens)
       causal_mask: Tensor # (K*SeqLen, K*SeqLen) - Cross-step causality
   )
   ```

2. **Token Construction** (per time step t):
   ```python
   # RTG token
   rtg_token = self.proj_rtg(RTG_seq[:, t, :])  # (B, d)
   
   # Residual token
   res_token = self.proj_residual(R_seq[:, t, :])  # (B, d)
   
   # Atom tokens (shared across all time steps)
   atom_tokens = self.proj_atom(D.T)  # (P, F) → (P, d)
   
   # Sequence: [RTG, Residual, Atom_0, ..., Atom_{P-1}]
   step_seq = torch.cat([
       rtg_token.unsqueeze(1),           # (B, 1, d)
       res_token.unsqueeze(1),           # (B, 1, d)
       atom_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, P, d)
   ], dim=1)  # (B, 2+P, d)
   ```

3. **Position Embeddings**:
   ```python
   self.pos_emb = nn.Parameter(torch.randn(1, 2 + P, d_model))
   # Position 0: RTG
   # Position 1: Residual  
   # Position 2~(P+1): Atoms
   ```

4. **Causal Masking** (two-level):
   - **Within-step**: RTG and Residual can attend to all atoms
   - **Cross-step**: Step t can only attend to steps ≤ t
   
5. **Output Heads**:
   ```python
   # Extract decision token (e.g., RTG position)
   decision_tokens = encoded[:, ::SeqLen, :]  # (B, K, d)
   
   # Predict expert and atom
   scores_expert = self.expert_head(decision_tokens)  # (B, K, E)
   scores_atom = self.atom_head(decision_tokens)      # (B, K, M)
   ```

### Phase 2: Trajectory Compatibility

**No changes needed** to `offline_dt_dataset.py`:
- Trajectories already contain `(r_t, RTG_t, expert, atom)` per step
- Dictionary D is reconstructed from manifest during DT training
- Atom tokens dynamically generated from D at training time

### Phase 3: Training Script Update

**File**: `scripts/dt_pointer_ldv_v2.py` (new script)

**Changes**:
1. Pass dictionary `D` to model in addition to trajectories
2. Construct full sequence per batch
3. Use new causal mask construction
4. Rest of training loop unchanged (loss, optimizer, logging)

### Phase 4: Ablation Studies

Compare architectures:
1. **Baseline** (`DTMinPointer`): Current additive tokens + precomputed keys
2. **Full Atoms** (`DTMinPointerV2`): Atom sequence (298 tokens/step)
3. **Expert-level** (optional): Aggregated expert tokens (39 tokens/step)

Metrics:
- Training convergence (epochs to 95% test accuracy)
- Final test accuracy (angle classification)
- Inference time per sample
- Attention pattern interpretability

## 📊 Expected Outcomes

### Success Criteria
1. **Learns Physical Operation**: Attention weights show correlation between residual and selected atoms
2. **Matches or Exceeds Baseline**: Test accuracy ≥ current 98.2% (480-epoch baseline)
3. **RTG Controllability**: Different RTG values lead to predictable behavior changes
4. **Interpretability**: Attention maps reveal which atoms are considered at each step

### Risks & Mitigations
1. **Long Sequence**: 1788 tokens → Higher memory/compute
   - **Mitigation**: Start with smaller M (e.g., M=4) or expert-level aggregation
2. **Training Stability**: Larger model may overfit on 111 samples
   - **Mitigation**: Strong regularization, data augmentation (noise, scaling)
3. **Causal Mask Complexity**: Cross-step masking may be tricky
   - **Mitigation**: Use block-diagonal mask construction, test on toy data first

## 🔄 Integration with Multi-OMP Workflow

### Workflow:
```
OMP Variant 1 (g-teacher, L1) ─┐
OMP Variant 2 (qk-teacher, L2) ├──→ Trajectories (JSONL) ──→ DT Training ──→ Comparison
OMP Variant 3 (hybrid, ...)   ─┘
```

### Per Variant:
1. Generate trajectories: `offline_dt_dataset.py --teacher <type>`
2. Train DT: `dt_pointer_ldv_v2.py --traj_dir <variant_dir>`
3. Evaluate: Accuracy, RTG sensitivity, attention analysis
4. Compare: Which teacher's trajectories lead to best DT performance?

## 📚 References

- **omp-mdp-dt.md**: Full first-principles analysis of OMP-MDP-DT connection
- **rl_formulation_analysis.md**: Physical formulation rationale
- **AGENTS.md**: Commit requirements (atomic code+results, reproducibility)
- Current implementation: `scripts/dt_pointer_ldv.py`, `doa_rl/trajectories/offline_dt_dataset.py`

## ✅ Next Immediate Action

**Code-only commit**: Document this design plan (this file)  
**Status**: Planning phase - no executable code yet  
**Type**: Design document (allowed per AGENTS.md code-only branch)

**Future Results Commits** (each atomic):
1. Implement `DTMinPointerV2` + smoke test
2. Train on g-teacher trajectories + metrics
3. Train on qk-teacher trajectories + comparison
4. Ablation: Full atoms vs expert-level vs baseline
