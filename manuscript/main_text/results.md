# Results

[Target: ~2,500 words]

## System Architecture and Core Performance

Our system achieved 93.5% accuracy on the 37-class acoustic localization task.

**Evidence**: Linked to commit 872aa65, see `evidence_tracking/核心結果證據.md`

## Ablation Study: Mechanism Attribution

We conducted comprehensive ablation experiments to understand the contribution
of each component (see Figure 4).

**Key finding**: Neither physics alone (1.7%) nor learning alone (2.7%) works—
synergy is essential (93.5%).

**Evidence**: Linked to commit b9dcafa, see `evidence_tracking/消融實驗證據.md`

### Physics-Learning Synergy is Essential

[詳細說明消融實驗結果]

### Sparsity is Fundamental

[說明稀疏性的重要性: 90.7% 崩潰]

### Transformer Enables Feature Learning

[說明 Transformer 的貢獻: 30.4% 差距]

## SNR Robustness

**Evidence**: Linked to commits bd88710, cfdc4d9, e37f512

## Routing Analysis

[說明路由機制如何工作]
