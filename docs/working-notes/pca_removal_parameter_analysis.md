# PCA Removal and Parameter Reduction Analysis

## Summary

**Modifications made:**
1. ✅ Removed all PCA code from `scripts/omp-transformer-ldv.py`
2. ✅ Added `--d_model` parameter to control Transformer embedding dimension
3. ✅ Updated default `--nhead` from 3 to 2 (divides d_model=64)
4. ✅ Changed to use full frequency resolution F=346 (no PCA bottleneck)

## Changes Overview

### Code Deletions
- Removed `from sklearn.decomposition import PCA` import
- Removed `reduce_frequency_pca()` function (lines 104-168)
- Removed `--pca_dim` argument from argparse

### Code Modifications
- `load_ldv_samples()`: Removed `pca` parameter, now returns F=346 spectra
- `main()`: Removed PCA preprocessing step, uses `H_final = H_raw` and `W_final = W_reduced`
- Model initialization: Changed from `d=F` to `d=args.d_model`

### New Parameters
```python
parser.add_argument('--d_model', type=int, default=64,
                    help='Transformer embedding dimension (default: 64, must be < F=346 to reduce parameters)')
```

## Parameter Comparison

### Previous (with PCA=45)
```
Input: F=45, E=37, M=8
Model: d_model=45, nhead=3, nlayers=2

Token projections: 2 × (45 × 45) = 4,050
Type embeddings: 2 × 45 = 90
TransformerEncoder (nlayers=2):
  Self-attention per layer: 4 × (45 × 45) = 8,100
  FFN per layer: 2 × (45 × 180) = 16,200
  Total per layer: ~24,300
  Two layers: 48,600
Routing (Wq, Wk): 2 × (45 × 45) = 4,050

Total parameters: ~57,000
```

### Current (F=346, d=64, nlayers=1)
```
Input: F=346, E=37, M=8
Model: d_model=64, nhead=2, nlayers=1

Token projections: 2 × (346 × 64) = 44,288
Type embeddings: 2 × 64 = 128
TransformerEncoder (nlayers=1):
  Self-attention per layer: 4 × (64 × 64) = 16,384
  FFN per layer: 2 × (64 × 256) = 32,768
  Total: ~49,152
Routing (Wq, Wk): 2 × (64 × 64) = 8,192

Estimated total parameters: ~102,000
```

**Parameter increase: 57k → 102k (~79% increase)**

This is acceptable because:
- We now use **full frequency information** (F=346 vs 45) - 7.7× more input features
- Token projections are only 44k (vs naive 2×346×346 = 239k if d=F)
- By using d=64 < F=346, we achieve **81% parameter reduction** in projections
- Reduced from nlayers=2 to nlayers=1 saves one Transformer layer (~24k params)

### Alternative Configuration (Lower Parameters)

If we want to match the previous 57k parameter budget:

```bash
python scripts/omp-transformer-ldv.py --d_model 32 --nhead 2 --nlayers 1
```

```
Token projections: 2 × (346 × 32) = 22,144
TransformerEncoder: ~12,288 (self-attn) + ~16,384 (FFN) = ~28,672
Routing: 2 × (32 × 32) = 2,048

Estimated total: ~53,000 (similar to previous budget)
```

## Rationale for PCA Removal

### Problem with PCA
1. **sklearn constraint**: `n_components ≤ min(n_samples, n_features)`
   - Our case: `min(37+8, 346) = 45` → Forced to F=45
   - Setting `--pca_dim 346` still resulted in F=45
2. **Information loss**: PCA maximizes variance, but discriminative features may exist in low-variance components
3. **Greedy baseline**: Achieves 83.8% with F=346, suggesting full frequency info is needed

### Evidence for Full Frequency Resolution
- **Greedy Soft-OMP**: 83.8% accuracy (93/111) using F=346
- **Transformer with PCA (F=45)**: 5.4% accuracy (6/111) after 30 epochs
  - Classification loss stuck at 3.61 ± 0.001
  - Only 2/37 angles learned (essentially random guessing)
- **Hypothesis**: PCA discards angle-discriminative features that exist in the "tail" of variance distribution

## Expected Outcome

With full frequency resolution (F=346) and reduced Transformer parameters (d=64):

1. **Better feature representation**: All 346 frequency bins preserved
2. **Controlled parameters**: d=64 prevents parameter explosion (vs d=346)
3. **Test hypothesis**: Does Transformer need full frequency info like greedy baseline?

### Success Criterion
- Classification loss decreases from 3.6 to <3.0 over 30 epochs
- Accuracy >10% (better than 5.4% random guessing baseline)
- Learning curve shows consistent improvement (not flat)

### If This Fails
Next hypotheses to test:
1. Insufficient training data (111 samples for 37 classes)
2. High dictionary coherence (μ=0.9995) causes gradient issues
3. Need higher learning rate (try 3e-3 or 1e-2)
4. Need more regularization (dropout, weight decay)

## Usage

### Default (d=64, nhead=2, nlayers=1)
```bash
conda activate trl-training
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace:$PYTHONPATH

python scripts/omp-transformer-ldv.py \
  --epochs 30 \
  --batch_size 16 \
  --device mps
```

### Lower parameters (d=32 to match previous budget)
```bash
python scripts/omp-transformer-ldv.py \
  --epochs 30 \
  --batch_size 16 \
  --d_model 32 \
  --nhead 2 \
  --nlayers 1 \
  --device mps
```

### Higher capacity (d=128 for more expressiveness)
```bash
python scripts/omp-transformer-ldv.py \
  --epochs 30 \
  --batch_size 16 \
  --d_model 128 \
  --nhead 4 \
  --nlayers 1 \
  --device mps
```

## Files Modified

- `scripts/omp-transformer-ldv.py`:
  - Lines 1-12: Updated docstring
  - Line 25: Removed PCA import
  - Lines 104-168: Deleted `reduce_frequency_pca()` function
  - Lines 147-218: Updated `load_ldv_samples()` signature and logic
  - Lines 606-614: Removed `--pca_dim`, added `--d_model`
  - Lines 669-684: Removed PCA preprocessing step in main()
  - Lines 690-732: Updated model initialization with d_model parameter
  - Lines 924-930: Removed PCA from saved artifacts

## Next Steps

1. **Smoke test**: Run 2 epochs to verify no crashes
   ```bash
   python scripts/omp-transformer-ldv.py --epochs 2 --device mps
   ```

2. **Full training**: 30 epochs with default d=64
   ```bash
   python scripts/omp-transformer-ldv.py --epochs 30 --device mps
   ```

3. **Parameter sweep**: Test d ∈ {32, 64, 128} to find optimal capacity

4. **Compare learning curves**: F=346 vs previous F=45 experiments
