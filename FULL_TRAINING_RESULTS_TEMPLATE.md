# Full Training Results: OMP → Decision Transformer (480 Epochs)

**Date:** [YYYY-MM-DD]  
**Run ID:** [TIMESTAMP]  
**Total Runtime:** [X hours Y minutes]

---

## Executive Summary

| Metric | OMP Baseline | DT (480 epochs) | Improvement |
|--------|--------------|-----------------|-------------|
| Expert accuracy | X.X% | X.X% | +X.X% |
| Atom accuracy | X.X% | X.X% | +X.X% |
| Training loss | N/A | X.XXX | - |
| Test loss | N/A | X.XXX | - |

**Key Finding:** [One-sentence summary of main result]

---

## 1. Experimental Setup

### Environment
- **Conda env:** trl-training
- **Device:** CPU
- **Python:** 3.x
- **PYTHONPATH:** /Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer

### OMP Trajectory Generation
```bash
Teacher: omp
K (selection budget): 6
M (atom set size): 8
Atom reduce mode: kcenter
Sampling rate: 16000 Hz
STFT n_fft: 2048
Frequency band: [300, 3000] Hz
Random seed: 42
```

**Data:**
- H matrix: /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
- W matrix: doa_normalized_config_c_corrected/models/usm.pth
- Dataset: /Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized

### Decision Transformer Training
```bash
Epochs: 480
Batch size: 4
Learning rate: 3e-3
Model architecture:
  - d_model: 128
  - n_heads: 2
  - n_layers: 1
Train/test split: 80/20 (seed=42)
```

**Physics Reconstruction (if enabled):**
- Use physics: [true/false]
- Physics weight: X.X
- Warmup epochs: XXX
- Task weights:
  - Residual prediction: X.X
  - Direction classification: X.X
  - Spectral coherence: X.X

---

## 2. OMP Trajectory Quality

### Generation Metrics
```
Total trajectories: XXX
Total steps: XXXXX
Average trajectory length: X.XX steps

First-step accuracy (expert): XX.X%
First-step accuracy (atom): XX.X%
Mean reward per step: X.XXX
```

### Trajectory Statistics
| Statistic | Value |
|-----------|-------|
| Expert selections (K=0) | XX.X% |
| Atom selections (K=1-7) | XX.X% |
| Average reward | X.XXX |
| Reward std dev | X.XXX |

**Analysis:** [Explain OMP baseline performance]

---

## 3. Training Dynamics

### Loss Curves (Key Epochs)

| Epoch | Train Loss | Test Loss | Expert Acc | Atom Acc | Notes |
|-------|------------|-----------|------------|----------|-------|
| 10 | X.XXX | X.XXX | XX.X% | XX.X% | Early training |
| 50 | X.XXX | X.XXX | XX.X% | XX.X% | |
| 100 | X.XXX | X.XXX | XX.X% | XX.X% | Physics warmup complete |
| 200 | X.XXX | X.XXX | XX.X% | XX.X% | Mid-training |
| 300 | X.XXX | X.XXX | XX.X% | XX.X% | |
| 400 | X.XXX | X.XXX | XX.X% | XX.X% | |
| 480 | X.XXX | X.XXX | XX.X% | XX.X% | **Final** |

### Training Observations
- **Convergence behavior:** [Fast/slow/unstable]
- **Best test loss achieved:** X.XXX at epoch XXX
- **Overfitting signs:** [Yes/No - explain]
- **Learning rate effectiveness:** [Analysis]

### Physics Reconstruction Metrics (if enabled)

| Epoch | Residual MSE (dB) | Direction Acc | Coherence Loss | Physics Total |
|-------|-------------------|---------------|----------------|---------------|
| 10 | -XX.X | XX.X% | X.XXX | X.XXX |
| 100 | -XX.X | XX.X% | X.XXX | X.XXX |
| 200 | -XX.X | XX.X% | X.XXX | X.XXX |
| 480 | -XX.X | XX.X% | X.XXX | X.XXX |

**Physics Analysis:** [Do physics metrics correlate with action accuracy?]

---

## 4. Final Performance Analysis

### Quantitative Results

**Test Set Performance (Epoch 480):**
```
Expert accuracy: XX.X% (baseline: OMP first-step XX.X%)
Atom accuracy: XX.X% (baseline: OMP first-step XX.X%)
Mean test loss: X.XXX
Test set size: XXX trajectories
```

**Improvement over OMP:**
- Expert selection: +X.X percentage points
- Atom selection: +X.X percentage points
- Overall: [Better/Worse/Similar]

### Action Distribution
| Action Type | OMP Distribution | DT Distribution | Difference |
|-------------|------------------|-----------------|------------|
| Expert (K=0) | XX.X% | XX.X% | ±X.X% |
| Atom 1 (K=1) | XX.X% | XX.X% | ±X.X% |
| Atom 2 (K=2) | XX.X% | XX.X% | ±X.X% |
| ... | ... | ... | ... |

---

## 5. Physical Interpretation

### Physics Reconstruction Quality (if enabled)

**Residual Prediction:**
- Final MSE: -XX.X dB
- Interpretation: [Can model predict mixing residuals?]

**Direction Classification:**
- Final accuracy: XX.X%
- Interpretation: [Does model learn spatial structure?]

**Spectral Coherence:**
- Final loss: X.XXX
- Interpretation: [Are embeddings frequency-aligned?]

### Embedding Analysis
[Analyze learned representations:]
- Are expert vs atom embeddings separable?
- Do physics constraints improve embedding quality?
- Can we visualize meaningful structure (e.g., angle ordering)?

---

## 6. Cross-Experiment Comparison

### Historical Context
| Experiment | Epochs | Expert Acc | Atom Acc | Notes |
|------------|--------|------------|----------|-------|
| [Previous run] | 100 | XX.X% | XX.X% | Physics warmup only |
| **This run** | **480** | **XX.X%** | **XX.X%** | **Full training** |
| [Baseline] | N/A | XX.X% | XX.X% | OMP teacher |

### Pattern Recognition (BECAUSE/DUE TO)
- **Success factors:** DT achieves XX.X% BECAUSE [reason based on training dynamics]
- **Limitations:** Expert accuracy plateaus at XX.X% DUE TO [fundamental constraint]
- **Physics impact:** Physics reconstruction [improves/degrades] performance BECAUSE [mechanism]

---

## 7. Reproduction Instructions

### Complete Workflow
```bash
# 1. Activate environment
source ~/.zshrc
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jnrle/LDVReorientation/worktrees/mdp-decision-transformer:$PYTHONPATH

# 2. Run full training pipeline
chmod +x run_full_training.sh
./run_full_training.sh

# 3. Expected outputs
# - results/dt_traj_omp_full_YYYYMMDD_HHMMSS/trajectories.jsonl
# - results/dt_traj_omp_full_YYYYMMDD_HHMMSS/generation.log
# - results/dt_full_training_YYYYMMDD_HHMMSS/training.log
# - results/dt_full_training_YYYYMMDD_HHMMSS/best_model.ckpt
# - results/dt_full_training_YYYYMMDD_HHMMSS/final_metrics.npz
```

### Verification
```bash
# Check trajectory quality
cat results/dt_traj_omp_full_*/generation.log | grep "First-step accuracy"

# Check final training metrics
tail -50 results/dt_full_training_*/training.log

# Verify checkpoint exists
ls -lh results/dt_full_training_*/best_model.ckpt
```

### Data Lineage
- **Trajectory fingerprint:** [MD5 of trajectories.jsonl]
- **H matrix version:** [Git commit or file hash]
- **W matrix version:** [Git commit or file hash]
- **Dataset version:** [Description or fingerprint]

---

## 8. Next Experiments

### Based on Results
1. **If DT >> OMP:** [What to try next to improve further]
2. **If DT ≈ OMP:** [Why might this be? How to break through?]
3. **If DT < OMP:** [Root cause analysis, what went wrong?]

### Physics Reconstruction Follow-ups (if applicable)
- [ ] Ablation: Train without physics, compare
- [ ] Ablation: Remove each physics task individually
- [ ] Analysis: Visualize learned embeddings (PCA/t-SNE)
- [ ] Transfer: Can physics-trained embeddings generalize?

### Architectural Explorations
- [ ] Increase model capacity (d_model=256, n_layers=2)
- [ ] Longer context (include previous K-1 steps)
- [ ] Attention analysis (which tokens matter most?)

---

## 9. Extracted Principles

### Design Principles
[Convert findings into actionable rules for future work]

**THEREFORE:**
- [Principle 1 based on results]
- [Principle 2 based on physics metrics]
- [Principle 3 based on training dynamics]

### Hypothesis Formation
**GIVEN** [these results], **PREDICT:**
- [Hypothesis 1 for next experiment]
- [Hypothesis 2 for architecture choice]

### Resource Allocation
**BECAUSE** [observation], **INVEST IN:**
- [Where to focus effort]
- [What not to waste time on]

---

## 10. Files and Artifacts

### Generated Files
```
results/dt_traj_omp_full_YYYYMMDD_HHMMSS/
├── trajectories.jsonl          # OMP rollout data
├── manifest.json                # Trajectory metadata
├── generation.log               # OMP generation log
└── numeric_diagnostics.jsonl   # Per-sample OMP metrics

results/dt_full_training_YYYYMMDD_HHMMSS/
├── training.log                 # Full training log
├── best_model.ckpt              # Best checkpoint (by test loss)
├── final_model.ckpt             # Epoch 480 checkpoint
├── final_metrics.npz            # NumPy metrics archive
└── config.json                  # Training configuration
```

### Log Locations
- **OMP generation:** `results/dt_traj_omp_full_*/generation.log`
- **DT training:** `results/dt_full_training_*/training.log`

---

## Commit Message Draft

```
Results: Full 480-epoch training - OMP→DT performance evaluation

Experiment context:
- Background: Physics reconstruction implemented, tested at 100 epochs
- Motivation: Need full 480-epoch training for fair OMP baseline comparison
- Purpose: Evaluate whether Decision Transformer can match/exceed OMP teacher
- Expected: DT should achieve 90%+ accuracy matching OMP baseline (64.9%/94.6%)

Training results:
- OMP first-step accuracy: XX.X% expert, XX.X% atom
- DT final accuracy (480 epochs): XX.X% expert, XX.X% atom
- Training time: X.X hours on CPU
- Best test loss: X.XXX at epoch XXX
- Hardware: MacBook Pro M1, conda env: trl-training

Key findings:
- [Main discovery 1]
- [Main discovery 2]
- [Main discovery 3]

Comparison to expectation:
✓ [What matched predictions]
✗ [What differed from predictions]
! [Unexpected discoveries]

Physical/mathematical analysis:
- First principles: [Explain from RL/imitation learning theory]
- Mathematical relationships: [Loss landscape, convergence properties]
- Signal processing: [STFT representation quality, atom separability]

Cross-experiment analysis (commits: [XXX, YYY, ZZZ]):
- Pattern recognition: [Patterns emerge BECAUSE physics/theory]
- Success factors: [What works BECAUSE math/constraints]
- Failure modes: [What fails DUE TO limitations]

Extracted principles:
- Design: THEREFORE [actionable rule]
- Hypothesis: GIVEN [results], PREDICT [next outcome]
- Resources: BECAUSE [finding], INVEST IN [direction]

Reproduction:
./run_full_training.sh
Expected: results/dt_full_training_*/training.log shows final accuracy XX.X%

Data lineage:
- Trajectory fingerprint: [MD5]
- Total trajectories: XXX
- Train/test split: 80/20 (seed=42)

Next experiments:
- [Follow-up 1 based on results]
- [Follow-up 2 based on physics metrics]
```
