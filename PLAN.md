# Plan: Frequency-Aware Policy for Lag-Exploration

## 1. Background & Motivation
- **Problem**: The global decision transformer (DTmin) achieves ~50% accuracy on the Lag-Selection task, significantly underperforming OMP (74%).
- **Discovery**: A "Single-Frequency" experiment (Bins 50-60) achieved 74% accuracy, matching OMP. This suggests the "Phase-Lag" relationship is frequency-dependent and the global model suffers from task interference (averaging conflicting signals).
- **Hypothesis**: Conditioning the policy on the specific Frequency Bin (or Band) will allow a single model to master the entire spectrum by learning frequency-specific phase relationships.

## 2. Methodology
We will transition from a "Global Blind Policy" to a "Frequency-Aware Policy".

### 2.1 Architecture Changes
- **Input**:
  - Current: `Correlation Sequence` (Batch, Seq, 1), `RTG` (Batch, Seq, 1).
  - New: `Frequency Index` (Batch, 1) or `Frequency Value` (Batch, 1).
- **Model (`SeqDT_RTG`)**:
  - Add `self.freq_embed = nn.Embedding(num_bins, hidden_dim)` (or Linear projection).
  - Strategy: Add Frequency Embedding to the Input Token Embeddings (similar to Positional Encoding).
  - $Input = LayerNorm(CorrProj(c) + RTGProj(R) + FreqEmb(f) + PosEmb(t))$

### 2.2 Dataset Changes (`dataset_lag.py`)
- Modify `RandomFreqDataset` (or equivalent) to return specific `freq_bin_idx` alongside the correlation vector.
- Ensure coverage of the full frequency range (e.g., bins 5 to 300) during training.

## 3. Execution Phase
1.  **Code Refactor**: Implement `FreqAwareDT` and update dataloader.
2.  **Smoke Test**: Run 2 epochs on small subset to verify pipeline.
3.  **Full Training**: Train on full spectrum (Bins 5-300).
4.  **Evaluation**:
    - Compare `FreqAwareDT` vs `GlobalDT` vs `OMP`.
    - Analyze `Accuracy` per Frequency Bin.

## 4. Success Metrics
- **Primary**: Overall Accuracy > 65% (OMP is ~74%).
- **Secondary**: "Step 0 Reduction" matching OMP-Single-Freq (>60% reduction).

## 5. Next Steps
- If successful, integrate this Policy back into the full RL/Inference pipeline.
