# Critical Bug Fix: Stage 3 H Matrix Estimation Error

**Date**: 2025-10-25  
**Severity**: CRITICAL - Affects all Stage 3-4 experiments in commit 3f3d8eb  
**Status**: FIXED

---

## Executive Summary

**Commit 3f3d8eb** (LDV-data Stage 3-4) produced **invalid H matrices** (all values = 1.0) due to using **the same dataset as both X and Y** in transfer function estimation. This caused DOA localization to fail completely (10% accuracy, 75° mean error).

**Root Cause**: H = Box/Box = 1.0 (self-reference instead of Original→Box)  
**Correct Method**: H = Box/Original (cross-domain transfer function)  
**Impact**: 90% accuracy achieved after fix (vs 10% with invalid H)

---

## Detailed Problem Analysis

### 1. The Error in Commit 3f3d8eb

#### What Happened
Stage 3 H matrix estimation used **only Box data** for both X (reference) and Y (observation):
```bash
# WRONG (as executed in 3f3d8eb):
python scripts/estimate_transfer_functions.py \
  ~/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  --output ~/LDV-data-processed/h_matrix_normalized_box_ldv.pth \
  --freq-min 300 --freq-max 3000 --files-per-angle 3
```

#### Why This Failed
The script `estimate_transfer_functions.py` accepts only ONE positional argument (`noise_data_root`). When only one path is provided, `DataProcessor.estimate_transfer_functions()` uses it for BOTH X and Y:

```python
# From nmf_localizer/core/data_processor.py
def estimate_transfer_functions(self, original_root, box_root=None, ...):
    if box_root is None:
        box_root = original_root  # ← BUG: Uses same path for both!
        logger.info(f"Using unified data root for both X and Y: {original_root}")
```

This results in:
```python
H(f,t) = Y_stft(f,t) / X_stft(f,t)
       = Box_stft / Box_stft  
       = 1.0  # ← All frequencies, all angles!
```

#### Evidence
```python
# Checking the generated H matrix from 3f3d8eb:
import torch
data = torch.load('~/LDV-data-processed/h_matrix_normalized_box_ldv.pth')
H = data['H']
print(H.shape)          # torch.Size([346, 37])
print(H.min(), H.max()) # 1.0, 1.0
print(torch.all(H == 1.0))  # True ← ALL VALUES ARE 1!
```

### 2. Physical Consequences

#### Why H=1 Breaks Localization
The NMF DOA algorithm relies on:
```
A = [diag(H₁)W, diag(H₂)W, ..., diag(H_D)W]
```

Where H_d encodes the directional transfer function for angle d.

With H=1:
```
diag(H_d) = I (identity matrix for all d)
A_d = I @ W = W  (same for ALL directions!)
```

**All angle blocks are identical → NMF cannot distinguish directions!**

#### Observed Failure Mode
- **Accuracy**: 10% (3/30 correct)
- **Mean error**: 75.2° (worse than random guessing ~45°)
- **NMF loss oscillation**: 10K ↔ 2.2M (non-convergent)
- **Predicted angles**: Essentially random

---

## The Correct Method

### 1. Proper Data Flow

```
X = Original playback (white_noise_original_data_no_edge_sync_vad)
    ↓ Speaker output → Air propagation
    
Y = Box LDV recording (white_noise_box_data_no_edge_sync_vad)  
    ↓ LEGO box reflection + LDV vibrometry

H(f, θ) = Y(f,t) / X(f,t)  
    = Transfer function encoding:
      • Angular-dependent reflection (LEGO box geometry)
      • Frequency-dependent filtering
      • LDV system response
```

### 2. Correct Implementation

```python
# Direct API call (bypassing broken script):
from nmf_localizer.core.stft_unified_processor import STFTUnifiedProcessor
from nmf_localizer.config.defaults import NMFConfig

config = NMFConfig(n_files_per_angle=3, freq_min=300.0, freq_max=3000.0)
processor = STFTUnifiedProcessor(config)

H, angles, _, metadata = processor.estimate_transfer_functions_stft(
    original_root=Path("~/LDV-data-processed/white_noise_original_data_no_edge_sync_vad"),
    box_root=Path("~/LDV-data-processed/white_noise_box_data_no_edge_sync_vad"),
    method='stft_unified',
    time_pooling='geometric'
)
```

### 3. Validation Results

**Correct H matrix statistics:**
```
Shape: [346 × 37]
Min: 0.000677
Max: 0.148538  
Mean: 0.015817
Unique values: 12,802 (not just 1!)
```

**DOA Localization with correct H:**
```
Accuracy: 90.0% (9/10 correct)
Mean error: 0.0°
Per-source accuracy: 100.0%
Success rate: 100.0%
Processing time: 29.3s per sample
```

---

## Files Affected by Original Error

### Invalid Files (DO NOT USE)
These were generated with H=1 bug in commit 3f3d8eb:
```
❌ ~/LDV-data-processed/h_matrix_normalized_box_ldv.pth
❌ ~/LDV-data-processed/h_matrix_normalized_irregularbox_ldv.pth
```

### Correct Replacement Files
```
✓ ~/LDV-data-processed/h_matrix_box_ldv_correct.pth
  (Generated: 2025-10-25 17:58, using Original→Box)
```

### Unaffected Files (Still Valid)
USM training used single-domain data, not affected by H bug:
```
✓ ~/LDV-data-processed/usm_box_ldv.pth
✓ ~/LDV-data-processed/usm_irregularbox_ldv.pth  
✓ ~/LDV-data-processed/usm_original_ldv.pth
```

---

## Reproduction Instructions

### Reproduce the Bug (for verification)
```bash
# This will generate H=1 (WRONG):
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace
conda activate trl-training

python scripts/estimate_transfer_functions.py \
  ~/LDV-data-processed/white_noise_box_data_no_edge_sync_vad \
  --output /tmp/h_matrix_bug_demo.pth \
  --freq-min 300 --freq-max 3000 --files-per-angle 3

# Verify it's all 1s:
python -c "
import torch
h = torch.load('/tmp/h_matrix_bug_demo.pth', weights_only=False)['H']
print(f'Min: {h.min():.6f}, Max: {h.max():.6f}')
print(f'All ones: {torch.all(h == 1.0)}')
"
```

### Generate Correct H Matrix
```bash
# Method 1: Direct API call (recommended)
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace:$PYTHONPATH

python << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from nmf_localizer.config.defaults import NMFConfig
from nmf_localizer.core.stft_unified_processor import STFTUnifiedProcessor
import torch

original_root = Path.home() / "LDV-data-processed/white_noise_original_data_no_edge_sync_vad"
box_root = Path.home() / "LDV-data-processed/white_noise_box_data_no_edge_sync_vad"
output_path = Path.home() / "LDV-data-processed/h_matrix_box_ldv_correct.pth"

config = NMFConfig(n_files_per_angle=3, freq_min=300.0, freq_max=3000.0)
processor = STFTUnifiedProcessor(config)

H, angles, angle_folders, metadata = processor.estimate_transfer_functions_stft(
    original_root, box_root, method='stft_unified', time_pooling='geometric'
)

torch.save({
    'H': H,
    'angles': angles,
    'angle_folders': [str(f) for f in angle_folders],
    'metadata': metadata
}, output_path)

print(f"✓ H matrix saved: {output_path}")
print(f"  Shape: {H.shape}")
print(f"  Range: [{H.min():.6f}, {H.max():.6f}]")
print(f"  Mean: {H.mean():.6f}")
EOF
```

### Test DOA Localization
```bash
# Smoke test (10 samples):
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace:$PYTHONPATH

conda run -n trl-training python scripts/run_localization.py \
  --tf-path ~/LDV-data-processed/h_matrix_box_ldv_correct.pth \
  --speech-data-root ~/LDV-data-processed/white_noise_box_data_no_edge_sync_vad \
  --output results/ldv_doa_correct_h_test \
  --usm-path ~/LDV-data-processed/usm_box_ldv.pth \
  --use-all-angles-for-usm \
  --freq-min 300 --freq-max 3000 \
  --beta 0.0 \
  --max-iter 100 \
  --n-test-examples 10

# Expected: Accuracy ≥80%, Mean error ≤10°
```

---

## Why Didn't We Catch This Earlier?

### 1. Misleading Commit Message
Commit 3f3d8eb recorded **incorrect commands** in the message:
```bash
# Commit message claimed:
python scripts/estimate_transfer_functions.py \
  --data_root ~/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized \
  ...
```

But the script never had a `--data_root` parameter! It was always a positional argument.

### 2. Coherence = 1.0 Was Misleading
The commit reported "coherence=1.000" which seemed good, but:
- **Coherence measures temporal synchronization** (X vs Y alignment)
- **NOT directional discriminability** (whether H varies across angles)
- With X=Y (same files), coherence=1.0 is guaranteed but meaningless!

### 3. No H Value Validation
The commit didn't include H statistics (min/max/mean), only shape and coherence.

**Lesson**: Always validate transfer function **value ranges**, not just shapes!

---

## Preventive Measures

### 1. Script Interface Fix (Recommended)
Modify `scripts/estimate_transfer_functions.py` to explicitly require two paths:

```python
parser.add_argument(
    'original_root',
    type=str,
    help='Path to Original playback data (X, reference signal)'
)
parser.add_argument(
    'box_root', 
    type=str,
    help='Path to Box/LDV recording data (Y, observation signal)'
)
```

### 2. Validation Checks
Add to `STFTUnifiedProcessor.estimate_transfer_functions_stft()`:

```python
# After H estimation:
if torch.all(H == 1.0):
    raise ValueError(
        "H matrix is all 1s! Check if X and Y are from the same dataset. "
        "Expected: H = Y/X where X=Original, Y=Box."
    )

if H.std() < 0.001:
    logger.warning(
        f"H matrix has very low variance (std={H.std():.6f}). "
        "This may indicate X≈Y (same dataset used for both)."
    )
```

### 3. Mandatory Commit Fields
For Stage 3 H matrix commits, REQUIRE:
```
- H statistics: min, max, mean, std
- Coherence: mean, range
- Data paths: explicit X (original_root) and Y (box_root)
- Validation: sample H values for 3 angles
```

---

## Timeline

- **2025-10-25 16:58** - Commit 3f3d8eb created with invalid H=1 matrices
- **2025-10-25 17:23** - First DOA test failed (10% accuracy) 
- **2025-10-25 17:30** - Investigated failure, suspected USM/pipeline issues
- **2025-10-25 17:43** - Discovered H=1 bug via direct matrix inspection
- **2025-10-25 17:58** - Generated correct H matrix, achieved 90% accuracy
- **2025-10-25 18:03** - Validated fix with smoke test

**Total debug time**: ~2 hours  
**Root cause**: Single-path script API + undocumented assumption

---

## References

- **Broken commit**: 3f3d8eb (Results: LDV-data Stage 3-4)
- **Fix commit**: [This commit]
- **Test results**: `results/ldv_doa_correct_h_smoke/`
- **Correct H matrix**: `~/LDV-data-processed/h_matrix_box_ldv_correct.pth`
