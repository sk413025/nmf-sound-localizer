# 🚨 Architecture Violation: Current Implementation Is Not True DTMin

**Status**: CRITICAL - Must refactor before production use  
**Date**: 2025-11-24  
**Branch**: exp/domain-randomizationV3  
**Affected commits**: 5a0d57da, c1bc82b3, and related angle-supervised experiments

---

## Executive Summary

The current implementation in `scripts/train_angle_range_dtmin.py` with `label_mode=angle` is **NOT a Decision Transformer Minimum (DTMin)**. It is a **Supervised Residual-to-Angle Classifier** that bypasses trajectory imitation entirely.

While this approach achieves high accuracy (voted=0.927-1.000) when the OMP teacher fails (voted=0.018), it **violates the core DTMin philosophy** of learning hierarchical decision-making from offline trajectories.

---

## What DTMin Should Be (Original Design)

### Core Principles (References: commits a9241ef, bd7ecbc)

**DTMin = Decision Transformer for Minimum Viable OMP Policy**

1. **Input**: Sequential state tokens `{r_t, RTG_resid_t, RTG_acc_t, step_t}` for t ∈ [0, K)
2. **Supervision**: Imitate OMP teacher's hierarchical actions `(expert_t, atom_t)`
3. **Loss**: `CE(expert_pred, expert_omp) + CE(atom_pred, atom_omp)`
4. **Goal**: Learn to generate trajectories that match teacher's decision process
5. **Evaluation**: 
   - Joint accuracy: `P(expert_t = expert_omp AND atom_t = atom_omp)` (trajectory matching)
   - Voted accuracy: `majority_vote(expert[0:K]) = angle_gt` (localization quality)

### Key Insight from Historical Failures

**Commit a9241ef showed**:
- Joint acc = 68.6% (good trajectory imitation)
- Voted acc = 17.3% (poor angle classification)
- **Conclusion**: "Trajectory imitation ≠ angle classification"

**Commit bd7ecbc showed**:
- Pure DT: voted acc = 0.027 (random)
- QK distillation: voted acc = 0.054 (slightly better, still poor)
- **Root cause**: Residual norms `r_t` + RTG do not uniquely encode angle in high-coherence dictionary (μ_max=0.9977)

---

## What Current Implementation Does

### Architecture in commits c1bc82b, 5a0d57d

**Actually: Supervised Angle Classifier from Residuals**

```python
# Input: residual sequence embeddings
h_seq = P_R(r[0:K]) + positional_encoding

# Supervision: DIRECTLY classify angle (not imitate trajectory)
target_expert = angle_gt.view(B, 1).expand(-1, K)  # Force every step to predict angle
loss = CE(expert_logits, target_expert)  # Angle classification loss
     + 0 * CE(atom_logits, atom_gt)      # Atom loss disabled

# Evaluation: only angle accuracy matters
voted_acc = (majority_vote(expert_pred[0:K]) == angle_gt).mean()
```

### Critical Differences

| Aspect | True DTMin | Current Implementation |
|--------|-----------|------------------------|
| **Input tokens** | r_t + RTG (decision state) | r_t (feature vector) |
| **Supervision** | (expert, atom) trajectory | angle labels |
| **Learning objective** | Imitate OMP decisions | Classify angles |
| **Output interpretation** | Action sequence | Class prediction |
| **Philosophy** | Imitation Learning | Supervised Classification |
| **Dependencies** | Requires teacher trajectory | Requires ground-truth angles |

---

## Why Current Approach Works (But Isn't DTMin)

### Physical Justification

The residuals `r[0:K]` contain angle information because:

1. **Transfer function uniqueness**: Each angle θ has unique `H(f, θ)`
2. **Residual encoding**: `r_t = Y - Ŷ_t` preserves frequency-domain patterns correlated with θ
3. **Grid alignment**: `Y.F == H.F == W.F = 346` ensures frequency consistency
4. **Transformer extraction**: Multi-head attention extracts angle-discriminative features from residual sequence

### Success Metrics (commit 5a0d57d)

- Teacher (OMP): voted = 0.018 (failed due to k-means compression + normalization)
- Student (angle-supervised): voted = 0.927-1.000 (peak)
- **Improvement**: +90 percentage points over teacher

### Why This Is NOT DTMin

1. **No trajectory imitation**: Model never learns to predict which (expert, atom) OMP would choose
2. **No RTG conditioning**: No return-to-go tokens guiding decision quality
3. **No hierarchical decomposition**: Expert head directly predicts angles, not intermediate OMP actions
4. **No controllability**: Cannot adjust trajectory quality via RTG at inference time

---

## Consequences of Architecture Violation

### What We Gained

✅ **High angle accuracy**: When teacher fails, direct supervision works  
✅ **Proof of concept**: Residuals contain angle information  
✅ **Pragmatic workaround**: Achieves localization goal despite bad teacher  

### What We Lost

❌ **Decision-making capability**: Cannot learn or improve OMP strategy  
❌ **Trajectory reasoning**: No understanding of multi-step selection process  
❌ **Controllability**: Cannot condition on desired trajectory quality  
❌ **Transfer function fidelity**: Loses (expert, atom) → H·W mapping  
❌ **Interpretability**: Cannot analyze which atoms were selected and why  

### Technical Debt

1. **Misleading naming**: File named `train_angle_range_dtmin.py` but doesn't implement DTMin
2. **Mixed abstractions**: Code structure implies trajectory learning but does classification
3. **Evaluation confusion**: Reports "expert accuracy" but expert ≠ angle in true DTMin
4. **Future scaling**: Cannot extend to RL/PPO without redesigning state representation

---

## Mandatory Refactoring Requirements

### Phase 1: Honest Naming (Immediate)

1. **Rename module**:
   ```bash
   scripts/train_angle_range_dtmin.py → scripts/train_residual_angle_classifier.py
   ```

2. **Update class names**:
   ```python
   # Before
   class TinyDTMin(nn.Module):
   
   # After
   class ResidualAngleClassifier(nn.Module):
       """Supervised classifier: residual sequence → angle.
       
       NOT a Decision Transformer - no trajectory imitation.
       """
   ```

3. **Clarify configuration**:
   ```python
   ap.add_argument("--method", default="supervised_classification",
                   choices=["supervised_classification", "dtmin_imitation"])
   ```

### Phase 2: True DTMin Implementation (Required Before Production)

**Design Requirements (per commit bd7ecbc "Next steps")**:

1. **Enhanced state tokens**:
   ```python
   # Add angle-discriminative evidence to token sequence
   state_t = {
       "r_t": residual[t],              # Current
       "g_scores": g(D, r_t),           # Per-angle greedy scores (NEW)
       "qk_logits": teacher_qk(r_t),    # QK teacher priors (NEW)
       "rtg_resid": target_resid - resid_t,
       "rtg_acc": target_acc - p_true_t,
       "step": t / K
   }
   ```

2. **Trajectory supervision**:
   ```python
   # Imitate teacher's hierarchical actions
   target_expert = batch["expert_gt"][t]  # OMP's expert choice at step t
   target_atom = batch["atom_gt"][t]      # OMP's atom choice at step t
   loss = CE(expert_logits, target_expert) + CE(atom_logits, target_atom)
   ```

3. **RTG-based controllability**:
   ```python
   # Inference with desired quality
   trajectory = model.generate(
       r_0, 
       rtg_target_resid=0.02,  # Control reconstruction quality
       rtg_target_acc=0.95,     # Control classification confidence
       K=6
   )
   ```

4. **Evaluation separation**:
   ```python
   metrics = {
       "joint_acc": ...,        # Trajectory matching (imitation quality)
       "voted_acc": ...,        # Angle classification (localization quality)
       "controllability": ...,  # RTG correlation with outcomes
   }
   ```

### Phase 3: Hybrid Architecture (Optional Enhancement)

**Combine both approaches**:

```python
class HybridDTMinClassifier(nn.Module):
    """Two-head architecture:
    1. DTMin head: Learns OMP trajectory imitation
    2. Angle head: Direct angle classification
    
    Training: Multi-task loss with configurable weights
    """
    def forward(self, state_seq, rtg_seq):
        h = self.encoder(state_seq, rtg_seq)
        
        # DTMin outputs (trajectory imitation)
        expert_logits = self.expert_head(h)  # → expert_omp
        atom_logits = self.atom_head(h)      # → atom_omp
        
        # Direct angle classification
        angle_logits = self.angle_head(h.mean(dim=1))  # → angle_gt
        
        return expert_logits, atom_logits, angle_logits
```

---

## Migration Path

### Step 1: Baseline Preservation (Week 1)

- [ ] Archive current implementation as `scripts/legacy/train_residual_angle_classifier.py`
- [ ] Document performance: voted=0.927-1.000 on speech260 (commit 5a0d57d)
- [ ] Tag baseline: `baseline/supervised-angle-classifier-v1`

### Step 2: True DTMin Implementation (Week 2-3)

- [ ] Implement enhanced state tokens (g_scores, qk_logits)
- [ ] Restore trajectory supervision (expert_omp, atom_omp)
- [ ] Add RTG conditioning and controllability metrics
- [ ] Smoke test: Joint acc > 60%, Voted acc > 40% on white noise (c1bc82b baseline)

### Step 3: Validation (Week 4)

- [ ] Reproduce historical DTMin experiments (commits a9241ef, bd7ecbc)
- [ ] Compare against supervised baseline
- [ ] Verify controllability via RTG sweeps
- [ ] Document trade-offs and use cases

### Step 4: Hybrid Exploration (Optional)

- [ ] Implement dual-head architecture
- [ ] Multi-task training experiments
- [ ] Analyze when each head contributes

---

## Decision Criteria: When to Use Each Approach

### Use Supervised Classifier (Current) When:

- Teacher is unreliable (voted < 0.1)
- Angle labels are available
- Only localization accuracy matters
- No need for trajectory interpretability
- Deployment as black-box classifier is acceptable

### Use True DTMin When:

- Teacher is reasonable (voted > 0.5)
- Need to understand OMP strategy
- Require controllable trajectory quality (RTG)
- Plan to extend to RL/PPO
- Want interpretable (expert, atom) selections

### Use Hybrid When:

- Both trajectory imitation AND angle classification matter
- Multi-task learning beneficial
- Ensemble predictions desired
- Comparing decision strategies

---

## Historical Context

### Successful Experiments with Current (Wrong) Approach

| Commit | Data | Teacher | Student (Supervised) | Notes |
|--------|------|---------|---------------------|-------|
| c1bc82b | white noise | 0.864 | 0.880 | Teacher good, student slightly better |
| 5a0d57d | speech260 | 0.018 | 0.927-1.000 | Teacher failed, supervised succeeded |

### Failed Experiments with True DTMin

| Commit | Method | Joint Acc | Voted Acc | Root Cause |
|--------|--------|-----------|-----------|------------|
| a9241ef | Pure DT | 68.6% | 17.3% | Trajectory ≠ angle |
| bd7ecbc | Pure DT | - | 0.027 | r_t insufficient |
| bd7ecbc | QK distill | - | 0.054 | Still insufficient |

**Lesson**: True DTMin requires angle-discriminative tokens in state representation, not just residuals.

---

## Action Items (Mandatory)

### Before Next Production Deploy:

1. ✅ Create this architecture violation document
2. ⚠️ Rename misleading files and classes
3. ⚠️ Add explicit `--method` flag to distinguish approaches
4. ⚠️ Update all result commits to clarify "supervised classification, not DTMin"
5. ❌ Implement true DTMin with enhanced state tokens
6. ❌ Validate against historical baselines
7. ❌ Document decision criteria for method selection

### Before Paper Submission:

- Must clearly distinguish supervised baseline vs DTMin in methods section
- Cannot claim "DTMin" results without proper trajectory imitation
- Must cite failure modes (a9241ef, bd7ecbc) and architectural differences

---

## References

- **Commit a9241ef**: "DTMin voted accuracy reveals trajectory imitation ≠ angle classification"
- **Commit bd7ecbc**: "DTMin trajectories — pure DT vs QK distillation (K=6, M=8, LDV box)"
- **Commit c1bc82b**: "Clean rerun K3 domain-rand BC/AWR vs teacher" (white noise, supervised)
- **Commit 5a0d57d**: "speech260 angle-range shards — OMP teacher baseline vs DTMin (BC/AWR, angle labels)" (supervised)

---

**CRITICAL**: This document must be reviewed and acknowledged before any claims of "DTMin implementation" in papers, presentations, or production systems.

---

**Signed**: AI Agent (Copilot)  
**Date**: 2025-11-24  
**Worktree**: domain-randomizationV3  
**Branch**: exp/domain-randomizationV3
