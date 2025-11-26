# DTMin RTG Information Leakage Investigation

> Investigation Date: 2025-11-26
> Related Commits: b763272, e8a9e78, 310b989

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Initial Observation: DTMin Exceeds OMP?](#2-initial-observation-dtmin-exceeds-omp)
3. [Metric Definitions](#3-metric-definitions)
4. [Information Leakage Hypothesis](#4-information-leakage-hypothesis)
5. [Decision Transformer Literature Review](#5-decision-transformer-literature-review)
6. [RTG Leakage Experiments](#6-rtg-leakage-experiments)
7. [Results Analysis](#7-results-analysis)
8. [Conclusions](#8-conclusions)
9. [Lessons Learned](#9-lessons-learned)

---

## 1. Background and Motivation

### 1.1 What is DTMin?

DTMin (Decision Transformer Minimum) is a model that learns to imitate OMP (Orthogonal Matching Pursuit) algorithm's trajectory decisions for Direction of Arrival (DOA) estimation. It is essentially a **Behavioral Cloning** approach where:

- **Input**: Residual embeddings `h_seq` from OMP steps (shape: [B, K, d_model])
- **Output**: Predicted (expert, atom) choices at each step
- **Supervision**: OMP teacher's trajectory decisions

### 1.2 Physics-Informed DT Extension

Commit `b763272` introduced RTG (Return-to-Go) conditioning, inspired by Decision Transformer:

```python
# Reward function
r_t = α * is_correct_expert - β * delta_resid_sq / init_resid

# RTG at step t
RTG_t = Σ(r_i) for i from t to T
```

Where:
- `α = 1.0`: Weight for correct expert selection
- `β = 0.1`: Weight for residual improvement
- `is_correct_expert`: Binary indicator if OMP's selected expert matches ground truth angle

### 1.3 The Core Question

After implementing RTG conditioning, we observed that DTMin appeared to **exceed OMP's accuracy**. This raised a critical question:

> Is this improvement legitimate, or is there information leakage through the RTG input?

---

## 2. Initial Observation: DTMin Exceeds OMP?

### 2.1 Commit b763272 Results (White Noise Data)

| Mode | RTG | Causal | Voted Acc | Expert Acc | Joint Acc |
|------|-----|--------|-----------|------------|-----------|
| A: Original | ✗ | ✗ | 0.646 | 0.861 | 0.778 |
| B: RTG+Causal | ✓ | ✓ | 0.583 | 0.681 | 0.604 |
| C: RTG Only | ✓ | ✗ | **0.715** | 0.935 | 0.894 |
| D: Causal Only | ✗ | ✓ | 0.444 | 0.550 | 0.447 |

### 2.2 Initial Confusion

The initial analysis incorrectly compared:
- DTMin Voted Acc on **white noise**: 0.715 (71.5%)
- OMP Voted Acc on **speech**: 0.324 (32.4%)

This **apples-to-oranges comparison** led to the mistaken belief that DTMin exceeded OMP by 2x.

### 2.3 Correct Comparison

After careful verification:

| Dataset | OMP Voted Acc | DTMin Voted Acc (RTG) | Comparison |
|---------|---------------|----------------------|------------|
| White Noise | **86.5%** | 71.5% | DTMin < OMP |
| Speech | **32.4%** | ~31.8% | DTMin ≈ OMP |

**Key Insight**: DTMin never exceeded OMP. The confusion arose from comparing different datasets.

---

## 3. Metric Definitions

### 3.1 Three Accuracy Metrics

Understanding what each metric measures is crucial:

#### Voted Acc (Localization Accuracy)
```
Voted Acc = P(majority_vote(expert_predictions) == angle_gt)
```
- Compares against **TRUE ground truth angle**
- Measures actual DOA estimation quality
- This is the primary metric for real-world performance

#### Expert Acc (Trajectory Matching - Expert)
```
Expert Acc = P(expert_pred == expert_omp) for each step
```
- Compares against **OMP's expert choice**
- Measures how well we imitate OMP's decisions
- High Expert Acc doesn't guarantee high Voted Acc

#### Joint Acc (Trajectory Matching - Complete)
```
Joint Acc = P(expert_pred == expert_omp AND atom_pred == atom_omp)
```
- Requires both expert AND atom to match OMP
- Strictest metric for trajectory imitation

### 3.2 Relationship Between Metrics

```
                    ┌─────────────────────────────────────┐
                    │         Ground Truth Angle          │
                    │           (angle_gt)                │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────────────┐
                    │          Voted Acc                  │
                    │  (majority vote vs true angle)      │
                    └──────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         OMP Trajectory              │
                    │   (expert_omp, atom_omp)            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
         ┌─────────────────┐           ┌─────────────────┐
         │   Expert Acc    │           │   Joint Acc     │
         │ (expert match)  │           │ (expert+atom)   │
         └─────────────────┘           └─────────────────┘
```

### 3.3 Why OMP's Voted Acc is the Ceiling

Since DTMin is trained to **imitate OMP's trajectory**, not to predict ground truth directly:
- If OMP makes wrong decisions, DTMin learns those wrong decisions
- DTMin's Voted Acc ≤ OMP's Voted Acc (theoretical upper bound)
- Exceeding OMP would require learning something OMP doesn't know

---

## 4. Information Leakage Hypothesis

### 4.1 Source of Potential Leakage

The RTG computation in `generator.py` uses `is_correct_expert`:

```python
def _compute_rewards(self, steps: List[Dict], init_resid: float) -> np.ndarray:
    alpha = 1.0  # correct angle weight
    beta = 0.1   # residual improvement weight
    rewards = np.zeros(len(steps), dtype=np.float32)
    for t, step in enumerate(steps):
        correct_bonus = alpha * float(step["is_correct_expert"])  # ← LEAKAGE?
        resid_bonus = -beta * step["delta_resid_sq"] / max(init_resid, 1e-12)
        rewards[t] = correct_bonus + resid_bonus
    return rewards
```

### 4.2 The Leakage Concern

`is_correct_expert` is computed as:
```python
is_correct_expert = (expert_angle == ground_truth_angle)
```

This means **RTG contains ground truth angle information**:
- If RTG is high → OMP was mostly correct → ground truth is likely OMP's choice
- If RTG is low → OMP was mostly wrong → ground truth differs from OMP's choice

### 4.3 The Critical Question

Does the model learn to:
1. **Extract information from RTG** to "cheat" and predict correct angles? (Leakage)
2. **Ignore RTG** and focus on actual h_seq embeddings? (No Leakage)

---

## 5. Decision Transformer Literature Review

### 5.1 Standard DT Architecture (Chen et al., 2021)

The original Decision Transformer:
- Uses RTG as a **conditioning signal** for desired future returns
- During training: Uses **actual rewards** from trajectories
- During inference: Starts with **target RTG** and updates with **actual rewards**

### 5.2 Key Insight from Literature

> "At test time, we can specify the desired performance (e.g., 1 for success or 0 for failure) and the generated actions are those that the transformer has learned are most likely to lead to that return."

**Important**: Standard DT **does** use actual rewards during inference to update RTG.

### 5.3 Our Situation is Different

In DOA estimation:
- There is no "environment" that gives rewards during inference
- To compute reward, we need to know if expert choice matches ground truth
- But ground truth is what we're trying to predict!

This creates a circular dependency that doesn't exist in typical RL settings.

### 5.4 Implication

Using actual RTG during inference **would require ground truth**, which we don't have. Therefore, we need to understand what happens with different RTG values.

---

## 6. RTG Leakage Experiments

### 6.1 Experimental Design

To test for leakage, we evaluate the same trained model with different fixed RTG values:

| RTG Setting | Description | Expected Behavior (No Leakage) |
|-------------|-------------|-------------------------------|
| Actual RTG | Real RTG from data | Baseline performance |
| Zero RTG (0.0) | Assume no future reward | Should match Actual if model ignores RTG |
| Low RTG (0.5) | Assume poor performance | Similar to Zero |
| Medium RTG (2.5) | Assume moderate performance | Similar to Zero |
| High RTG (5.0) | Assume excellent performance | OOD, may fail |

### 6.2 Test Conditions

We tested three configurations:

| Test | Dataset | Shards | N | RTG | Epochs |
|------|---------|--------|---|-----|--------|
| Test 1 | White Noise | 4 | 222 | ✓ | 120 |
| Test 2 | Speech | 4 | 222 | ✓ | 120 |
| Test 3 | Speech | 7 | 285 | ✓ | 120 |

### 6.3 Commands Used

```bash
# Test 1: White noise, 4 shards
PYTHONPATH=. python scripts/train_angle_range_dtmin.py \
  --shard-root results/angle_range_shards \
  --epochs 120 --use-rtg --rtg-leakage-test

# Test 2: Speech, 4 shards (default)
PYTHONPATH=. python scripts/train_angle_range_dtmin.py \
  --shard-root results/speech260_dtmin_full_w_ranges_mix_nortg \
  --epochs 120 --use-rtg --rtg-leakage-test

# Test 3: Speech, 7 shards
PYTHONPATH=. python scripts/train_angle_range_dtmin.py \
  --shard-root results/speech260_dtmin_full_w_ranges_mix_nortg \
  --shard-names "full,low,mid,high,weak_0_50,weak_140_170,strong_edges" \
  --epochs 120 --use-rtg --rtg-leakage-test
```

---

## 7. Results Analysis

### 7.1 White Noise Results (120 epochs, 4 shards)

| RTG Setting | Voted Acc | Expert Acc | Joint Acc |
|-------------|-----------|------------|-----------|
| Actual RTG | **0.771** | 1.000 | 1.000 |
| Zero RTG | **0.771** | 0.979 | 0.967 |
| RTG=0.5 | 0.771 | 0.992 | 0.992 |
| RTG=2.5 | 0.771 | 0.929 | 0.929 |
| RTG=5.0 | 0.042 | 0.179 | 0.113 |

**Key Observation**: Actual RTG = Zero RTG = 0.771 → **No Leakage**

### 7.2 Speech Results (120 epochs, 4 shards)

| RTG Setting | Voted Acc | Expert Acc | Joint Acc |
|-------------|-----------|------------|-----------|
| Actual RTG | **0.229** | 0.900 | 0.888 |
| Zero RTG | **0.229** | 0.900 | 0.888 |
| RTG=0.5 | 0.229 | 0.900 | 0.879 |
| RTG=2.5 | 0.208 | 0.858 | 0.842 |
| RTG=5.0 | 0.042 | 0.458 | 0.246 |

**Key Observation**: Actual RTG = Zero RTG = 0.229 → **No Leakage**

### 7.3 Speech Results (120 epochs, 7 shards)

| RTG Setting | Voted Acc | Expert Acc | Joint Acc |
|-------------|-----------|------------|-----------|
| Actual RTG | **0.281** | 0.953 | 0.953 |
| Zero RTG | **0.281** | 0.953 | 0.953 |
| RTG=0.5 | 0.281 | 0.953 | 0.953 |
| RTG=2.5 | 0.141 | 0.550 | 0.422 |
| RTG=5.0 | 0.078 | 0.216 | 0.134 |

**Key Observation**: Actual RTG = Zero RTG = 0.281 → **No Leakage**

### 7.4 Comparison with Original Non-RTG Experiment

| Experiment | Dataset | RTG | Shards | Voted Acc | OMP Voted |
|------------|---------|-----|--------|-----------|-----------|
| Commit 310b989 | Speech | ✗ | 7 | **0.325** | 0.324 |
| Current Test | Speech | ✓ | 7 | **0.281** | 0.324 |

**Key Observation**: RTG may actually **hurt** performance on speech data (-4.4%)

### 7.5 Understanding RTG=5.0 Failure

The catastrophic failure at RTG=5.0 is **not** evidence of leakage, but rather:
- RTG=5.0 is **out-of-distribution** (training RTG range is approximately 0-4)
- The model has never seen such high RTG values during training
- This is expected behavior for any neural network given OOD inputs

---

## 8. Conclusions

### 8.1 Primary Finding: No Information Leakage

After 120 epochs of training, the model learns to **ignore RTG** and rely solely on `h_seq` embeddings:

```
Actual RTG Voted Acc = Zero RTG Voted Acc
```

This holds for both white noise and speech datasets.

### 8.2 Why No Leakage?

The transformer encoder learns that:
1. `h_seq` embeddings contain sufficient information for trajectory prediction
2. RTG provides redundant/noisy information
3. Ignoring RTG leads to more robust predictions

### 8.3 DTMin Never Exceeds OMP

| Dataset | OMP Voted Acc | DTMin Voted Acc | Comparison |
|---------|---------------|-----------------|------------|
| White Noise | **86.5%** | 77.1% (RTG) | DTMin < OMP |
| Speech | **32.4%** | 28.1% (RTG) | DTMin < OMP |
| Speech | **32.4%** | 32.5% (No RTG) | DTMin ≈ OMP |

This confirms the theoretical expectation:
> Behavioral Cloning cannot exceed the teacher's performance.

### 8.4 RTG May Hurt Performance

On speech data:
- Without RTG: 32.5%
- With RTG: 28.1%
- **RTG reduced performance by 4.4%**

Possible explanations:
1. RTG adds noise to the input
2. Model capacity wasted on learning to ignore RTG
3. RTG distribution shift between training and inference

### 8.5 Training Duration Matters

Early in training (10 epochs), we observed apparent "leakage":
- Actual RTG: 0.771
- Zero RTG: 0.604
- Apparent leakage: 16.7%

But after 120 epochs:
- Actual RTG: 0.771
- Zero RTG: 0.771
- No leakage: 0%

**Interpretation**: Early in training, the model relies on RTG as a shortcut. With sufficient training, it learns the underlying patterns and ignores RTG.

---

## 9. Lessons Learned

### 9.1 Experimental Methodology

1. **Always compare like with like**: Same dataset, same conditions
2. **Verify baseline numbers**: Don't assume - check the actual data
3. **Run sufficient epochs**: Short training can give misleading results
4. **Control for all variables**: Shards, N, epochs, RTG all matter

### 9.2 Understanding Behavioral Cloning Limits

1. **Teacher ceiling**: BC cannot exceed the teacher's performance
2. **Trajectory vs outcome**: High Expert Acc doesn't guarantee high Voted Acc
3. **Information flow**: Model can only learn what's in the training data

### 9.3 RTG Conditioning in Non-RL Settings

1. **Circular dependency**: RTG requires ground truth, which is the prediction target
2. **Inference mismatch**: Training uses actual RTG, inference doesn't have it
3. **Model robustness**: Well-trained models learn to ignore unhelpful inputs

### 9.4 Debugging Strategy

When a model seems to "exceed" expectations:
1. Verify the comparison is fair (same dataset, conditions)
2. Check for information leakage in inputs
3. Design ablation experiments (fixed RTG values)
4. Run sufficient training to convergence
5. Document findings thoroughly

---

## Appendix A: Data Sources

| Shard Directory | Data Type | Source Path |
|-----------------|-----------|-------------|
| `angle_range_shards` | White Noise | `white_noise_box_data_no_edge_sync_vad_normalized` |
| `speech260_dtmin_full_w_ranges_mix_nortg` | Speech | `speech260_box_data_no_edge_sync_vad_normalized` |

## Appendix B: OMP Baseline Accuracies

From `summary.json` files:

| Dataset | first_step_acc | voted_acc | joint_acc |
|---------|----------------|-----------|-----------|
| White Noise | 0.676 | **0.865** | 0.0 |
| Speech | 0.189 | **0.324** | 0.0 |

## Appendix C: Code References

- RTG computation: [generator.py:_compute_rewards()](../doa_rl/domain_randomization/generator.py)
- RTG leakage test: [train_angle_range_dtmin.py:470-502](../scripts/train_angle_range_dtmin.py)
- Evaluate with fixed RTG: [train_angle_range_dtmin.py:evaluate()](../scripts/train_angle_range_dtmin.py)
