# Implementation Plan: DTMinPointerV2 (Atom-Sequence Input)

**Date**: 2025-11-03  
**Based on**: Design document `docs/dt_atom_sequence_design.md` (commit 0fca4de)  
**Goal**: Extend DT to include all P atom vectors as explicit state tokens

---

## 🎯 Objective

Enable the Decision Transformer to learn the physical operation `argmax(D^T · r)` by providing complete state information including all dictionary atoms as input tokens.

---

## 📋 Implementation Steps

### Step 1: Create DTMinPointerV2 Model Class ✅ (Next)

**File**: `scripts/dt_pointer_ldv_v2.py`

**Changes from DTMinPointer**:

1. **Input format**:
   ```python
   forward(
       R_seq: Tensor,       # (B, K, F) - Still needed for reconstruction
       RTG_seq: Tensor,     # (B, K, 2) - RTG targets
       D: Tensor,           # (F, P) - Dictionary atoms (NEW)
       STEP_seq: Tensor,    # (B, K, 2) - Step encoding
       causal_mask: Tensor  # (K×seq_len, K×seq_len) - 2D causal mask
   )
   ```

2. **Sequence construction per step**:
   ```python
   # Per time step t, build sequence of 2+P tokens:
   tokens_t = [
       proj_rtg(RTG_t),              # (d,) - RTG token
       P_R(R_t) + type_R,             # (d,) - Residual token
       P_D(D[:, 0]) + type_D,         # (d,) - Atom 0 token
       P_D(D[:, 1]) + type_D,         # (d,) - Atom 1 token
       ...
       P_D(D[:, P-1]) + type_D        # (d,) - Atom P-1 token
   ]
   # Shape: (2+P, d)
   ```

3. **Full episode sequence**:
   ```python
   # Stack K steps
   seq_episode = torch.stack([tokens_0, ..., tokens_{K-1}])  # (K, 2+P, d)
   
   # Option A: Keep 3D structure
   # Input to Transformer: (B, K, 2+P, d)
   # Requires custom attention mask
   
   # Option B: Flatten to 2D (RECOMMENDED)
   seq_flat = seq_episode.view(B, K*(2+P), d)  # (B, K×(2+P), d)
   # Standard Transformer input
   ```

4. **Causal mask design**:
   ```python
   # Mask shape: (K×(2+P), K×(2+P))
   # - Within step t: all 2+P tokens can attend to each other
   # - Across steps: step t can attend to all tokens in step ≤t
   
   # Block structure:
   # [t=0: 2+P×2+P full attention]
   # [t=1: 2+P×2+P full, can see t=0]
   # [t=2: 2+P×2+P full, can see t=0,1]
   # ...
   ```

5. **Pointer head redesign**:
   ```python
   # After Transformer encoding
   H = self.encoder(seq_flat, mask=causal_mask)  # (B, K×(2+P), d)
   H_steps = H.view(B, K, 2+P, d)
   
   # Extract positions
   H_RTG = H_steps[:, :, 0, :]      # (B, K, d) - RTG position
   H_R = H_steps[:, :, 1, :]        # (B, K, d) - Residual position
   H_atoms = H_steps[:, :, 2:, :]   # (B, K, P, d) - Atom positions
   
   # Query from residual position
   Q = self.Wq(H_R)                 # (B, K, d)
   
   # Keys from atom positions (learned inner product)
   K_atoms = self.Wk(H_atoms)       # (B, K, P, d)
   
   # Attention scores (analogous to g_j = d_j^T · r)
   scores = einsum('bkd,bkpd->bkp', Q, K_atoms) / sqrt(d)  # (B, K, P)
   
   # Reshape to (E, M) for hierarchical selection
   scores_em = scores.view(B, K, E, M)
   scores_e = sqrt(sum(scores_em.abs()^2, dim=M))  # L2 aggregation
   ```

### Step 2: Sequence Construction in Training Loop

**File**: Same `scripts/dt_pointer_ldv_v2.py` (main function)

**Changes**:
```python
# Current: Build R_seq, RTG_seq, STEP_seq as (K, ...) per sample
# New: Keep same, but pass D explicitly to forward()

# No change to data loading (trajectories still used the same way)
for obj in trajs:
    y = load_y_for_path(path)
    actions_prev = []
    
    for t, s in enumerate(steps):
        r_t = recompute_r_t(y, D, actions_prev)
        
        # Same as before
        R_list.append(r_t)
        RTG_list.append([s['rtg_resid'], s['rtg_acc']])
        STEP_list.append([t/K, (K-t)/K])
        
        # Labels unchanged
        lab_e.append(s['expert'])
        lab_m.append(s['atom'])
        
        actions_prev.append(s['dict_index'])
    
    # Build batch tensors (same)
    R_seq = torch.stack(R_list)
    RTG_seq = torch.stack(RTG_list)
    STEP_seq = torch.stack(STEP_list)
    
    samples.append((R_seq, RTG_seq, STEP_seq, lab_e, lab_m))

# In training loop: pass D to forward()
for R, RTG, STEP, le, lm in dataloader:
    scores_e, scores_em = model(R, RTG, D, STEP, causal_mask)
    # Loss computation unchanged
```

### Step 3: Causal Mask Generation

**Function**: `generate_causal_mask_atom_seq(K: int, seq_len: int, device)`

```python
def generate_causal_mask_atom_seq(K: int, seq_len: int, device: torch.device):
    """
    Generate causal mask for atom-sequence input.
    
    Args:
        K: Number of time steps
        seq_len: Tokens per step (2 + P)
        device: torch device
    
    Returns:
        mask: (K×seq_len, K×seq_len) with -inf for masked positions
    
    Mask structure:
        - Diagonal blocks (within-step): Full attention (0)
        - Lower triangle blocks (past steps): Full attention (0)
        - Upper triangle blocks (future steps): Masked (-inf)
    """
    total_len = K * seq_len
    mask = torch.zeros(total_len, total_len, device=device)
    
    for t in range(K):
        # Current step block: [t×seq_len : (t+1)×seq_len]
        # Can attend to all previous steps and within current step
        start_t = t * seq_len
        end_t = (t + 1) * seq_len
        
        # Mask future steps (t' > t)
        for t_future in range(t + 1, K):
            start_future = t_future * seq_len
            end_future = (t_future + 1) * seq_len
            mask[start_t:end_t, start_future:end_future] = float('-inf')
    
    return mask
```

### Step 4: Update Unroll Policy (Inference)

**Function**: `unroll_policy_v2(model, D, y, K, ...)`

**Changes**:
```python
@torch.no_grad()
def unroll_policy_v2(model, D, y, K, rtg_targets, device):
    """
    Greedy unroll with atom-sequence input.
    
    At each step t:
    1. Recompute r_t from y and actions[:t]
    2. Build sequences up to step t (not full K)
    3. Forward pass with current history
    4. Select action from step t predictions
    5. Update actions list
    """
    actions = []
    E, M = model.E, model.M
    
    for t in range(K):
        r_t = recompute_r_t(y, D, actions)
        
        # Build sequences for steps 0..t (incremental)
        R_history = []
        RTG_history = []
        STEP_history = []
        
        for tau in range(t + 1):
            r_tau = recompute_r_t(y, D, actions[:tau])
            
            # Compute RTG at step tau (using current estimate)
            y_hat = D[:, actions[:tau]] @ lstsq(D[:, actions[:tau]], y) if actions[:tau] else 0
            p_est = estimate_p_true(D, y_hat, E, M)
            rtg_r = max(0, float((r_tau @ r_tau).item()) - rtg_targets['resid'])
            rtg_a = max(0, rtg_targets['acc'] - p_est)
            
            R_history.append(r_tau)
            RTG_history.append([rtg_r, rtg_a])
            STEP_history.append([tau/K, (K-tau)/K])
        
        # Stack into batch dimension 1
        R_seq = torch.stack(R_history).unsqueeze(0)      # (1, t+1, F)
        RTG_seq = torch.stack(RTG_history).unsqueeze(0)  # (1, t+1, 2)
        STEP_seq = torch.stack(STEP_history).unsqueeze(0) # (1, t+1, 2)
        
        # Generate causal mask for current history length
        cmask = generate_causal_mask_atom_seq(t+1, 2+model.P, device)
        
        # Forward pass
        scores_e, scores_em = model(R_seq, RTG_seq, D, STEP_seq, cmask)
        
        # Extract predictions for step t (last step in history)
        e = int(scores_e[0, t].argmax().item())
        m = int(scores_em[0, t, e].argmax().item())
        j = e * M + m
        
        # Avoid duplicates
        if j in actions:
            scores_em_filtered = scores_em[0, t, e].clone()
            scores_em_filtered[m] = -1e9
            m = int(scores_em_filtered.argmax().item())
            j = e * M + m
        
        actions.append(j)
    
    # Final predictions
    y_hat = D[:, actions] @ lstsq(D[:, actions], y)
    g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)
    e_pred = int(g_hat.argmax().item())
    
    return actions, e_pred, y_hat
```

### Step 5: Testing Strategy

**Smoke Test** (3 samples, 1 epoch):
```bash
python scripts/dt_pointer_ldv_v2.py \
    --traj_dir results/dt_traj_g_full \
    --out_dir results/dt_v2_smoke \
    --subset_angles "0,5,10" \
    --epochs 1 \
    --batch_size 2 \
    --lr 1e-3 \
    --device cpu
```

**Verification**:
- [ ] Model initializes without errors
- [ ] Forward pass completes
- [ ] Sequence shapes are correct: (B, K×(2+P), d)
- [ ] Causal mask has correct structure
- [ ] Loss is computed successfully
- [ ] Gradients flow properly (check with torch.autograd.grad)
- [ ] Unroll policy runs without errors

**Functional Test** (full dataset, 10 epochs):
```bash
python scripts/dt_pointer_ldv_v2.py \
    --traj_dir results/dt_traj_g_full \
    --out_dir results/dt_v2_functional \
    --epochs 10 \
    --batch_size 8 \
    --lr 3e-3 \
    --device cpu \
    --test_split 0.2
```

**Acceptance Criteria**:
- Train/val loss converge (decrease over epochs)
- Expert accuracy > 50% on test set (random baseline: 2.7%)
- Atom accuracy > 40% on test set (random: 12.5%)
- No NaN or Inf in losses
- Model checkpoints saved successfully

---

## 🚨 Key Design Decisions

### 1. Sequence Flattening (Option B)

**Choice**: Flatten (K, 2+P, d) → (K×(2+P), d)

**Rationale**:
- ✅ Standard Transformer API (no custom attention)
- ✅ Easier mask generation (2D instead of 4D)
- ✅ Compatible with existing TransformerEncoder
- ✅ Proven in Decision Transformer paper

**Alternative** (Option A): Keep 3D structure
- ❌ Requires custom multi-head attention
- ❌ More complex mask logic
- ❌ No clear benefit over flattening

### 2. Positional Encoding

**Choice**: Use STEP_seq encoding + implicit position via causal mask

**Rationale**:
- ✅ Step information already encoded in STEP_seq
- ✅ Causal mask provides temporal ordering
- ✅ Within-step position is semantically meaningful (RTG, R, atoms)
- ✅ Avoids adding extra position embeddings

**Alternative**: Add explicit positional embeddings
- ❌ May interfere with RTG/R/Atom semantics
- ❌ Step position already captured in STEP_seq

### 3. Pointer Head Position

**Choice**: Query from H_R (residual position), Keys from H_atoms (atom positions)

**Rationale**:
- ✅ Physically meaningful: r queries which atom to select
- ✅ Matches OMP operation: argmax(d_j^T · r)
- ✅ Attention weights interpretable as g_j scores

**Alternative**: Query from all positions
- ❌ Less interpretable
- ❌ Doesn't align with physical intuition

### 4. Expert Aggregation

**Choice**: L2 aggregation (same as teacher qk-model)

**Rationale**:
- ✅ Consistency with teacher training
- ✅ Preserves magnitude information
- ✅ Proven to work in commit fe4ef60

**Alternative**: L1 or max aggregation
- May work but less tested

---

## 📊 Expected Outcomes

### Sequence Length Analysis

**Current (DTMinPointer)**:
- Tokens per step: 1 (summed)
- Total sequence: K = 6 tokens
- Computation: O(K²·d²) for self-attention

**DTMinPointerV2**:
- Tokens per step: 2 + P = 2 + 296 = 298
- Total sequence: K×(2+P) = 6×298 = 1788 tokens
- Computation: O((K·seq_len)²·d²) = O(1788²·d²)

**Computational overhead**:
- Sequence length: 298× longer
- Attention complexity: ~90,000× more compute per sample
- Batch size impact: May need to reduce batch size

**Mitigation**:
- Use smaller M (e.g., M=4 → P=148 → 150 tokens/step)
- Gradient accumulation for effective larger batches
- Consider linear attention variants (future work)

### Performance Expectations

**Baseline (DTMinPointer on g-trajectories)**:
- Expected expert acc: ~80% (following teacher's stickiness)
- Expected atom acc: ~70%
- Test p_true: ~0.04 (inherits teacher's low discrimination)

**DTMinPointerV2 on g-trajectories**:
- Expected expert acc: ~85% (better state representation)
- Expected atom acc: ~75%
- Test p_true: ~0.04-0.06 (slight improvement from attention learning)
- Key benefit: Interpretable attention weights showing learned g_j

**DTMinPointerV2 on qk-trajectories** (future):
- Expected expert acc: >90%
- Expected atom acc: >85%
- Test p_true: >0.90 (inherits teacher's high discrimination)

---

## 🔧 Implementation Checklist

### Phase 1: Model Implementation
- [ ] Create `scripts/dt_pointer_ldv_v2.py`
- [ ] Implement `DTMinPointerV2` class
- [ ] Implement sequence construction in forward()
- [ ] Implement causal mask generation
- [ ] Implement pointer head with atom positions
- [ ] Add docstrings and type hints

### Phase 2: Training Loop
- [ ] Modify main() to pass D to forward()
- [ ] Update batch collation (if needed)
- [ ] Verify loss computation
- [ ] Add logging for sequence shapes

### Phase 3: Inference
- [ ] Implement `unroll_policy_v2()`
- [ ] Update evaluation loop
- [ ] Add metrics computation

### Phase 4: Testing
- [ ] Smoke test (3 samples, 1 epoch)
- [ ] Verify sequence shapes
- [ ] Check gradient flow
- [ ] Functional test (10 epochs)
- [ ] Compare to baseline DTMinPointer

### Phase 5: Documentation
- [ ] Update code comments
- [ ] Document hyperparameter choices
- [ ] Record experimental results
- [ ] Create reproduction script

---

## 📝 File Structure

```
scripts/
├── dt_pointer_ldv.py          # Baseline (DTMinPointer)
└── dt_pointer_ldv_v2.py        # New (DTMinPointerV2) ← To create

results/
├── dt_traj_g_full/             # Trajectories (reused)
├── dt_v2_smoke/                # Smoke test output
├── dt_v2_functional/           # Functional test output
└── dt_v2_g_full/               # Full training output

docs/
├── dt_atom_sequence_design.md  # Design document (0fca4de)
└── DATA_PIPELINE_STATUS.md     # Implementation status (d7560dc)
```

---

## 🎯 Success Criteria

**Smoke Test**:
- ✅ No runtime errors
- ✅ Sequence shapes match expected: (B, 1788, 128)
- ✅ Loss is finite and decreases
- ✅ Completes in <5 minutes on CPU

**Functional Test**:
- ✅ Train loss < 2.0 after 10 epochs
- ✅ Val loss within 20% of train loss (no severe overfitting)
- ✅ Expert accuracy > 50%
- ✅ Attention weights show reasonable patterns

**Full Training** (480 epochs):
- ✅ Expert accuracy ≥ 85%
- ✅ Atom accuracy ≥ 75%
- ✅ Comparable to or better than DTMinPointer baseline
- ✅ Attention weights interpretable as g_j scores

---

**Next**: Begin implementation of `scripts/dt_pointer_ldv_v2.py`
