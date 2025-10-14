# Day 10-14 Quick Reference

## 🚀 Quick Start

### Run Smoke Test
```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f
bash run_day10_14_smoke_test.sh
```

### Check Results
```bash
ls -lh results/day10_14_smoke/
```

---

## 📊 Experiment Commands

### 1. Baseline (Patch-only)
```bash
python scripts/train_reward_model_lora.py \
    --data-root doa_normalized_config_c_corrected \
    --tf-path h_matrix_normalized_original_to_box.pth \
    --w-path doa_normalized_config_c_corrected/models/usm.pth \
    --s-root doa_normalized_config_c_corrected \
    --K 2 --rm-epochs 2 --batch-size 2 --max-samples 5 \
    --out results/baseline_rm
```

### 2. Multi-Modal (Physics-First)
```bash
python scripts/train_reward_model_lora.py \
    --data-root doa_normalized_config_c_corrected \
    --tf-path h_matrix_normalized_original_to_box.pth \
    --w-path doa_normalized_config_c_corrected/models/usm.pth \
    --s-root doa_normalized_config_c_corrected \
    --use-multi-modal \
    --token-ordering physics_first \
    --max-tokens 150 --top-k-atoms 5 --top-m-directions 3 \
    --K 2 --rm-epochs 2 --batch-size 2 --max-samples 5 \
    --out results/multimodal_rm
```

### 3. Full Comparison Suite
```bash
bash run_comparison_experiments.sh
```

---

## 🔍 Verification

### Check Module Installation
```bash
python -c "import doa_rl; print('✓ OK')"
```

### Verify Tokenizers
```bash
python -c "
from doa_rl.features.tokenizers_extended import NMFAtomTokenizer, DirectionProjectionTokenizer
print('✓ Extended tokenizers available')
"
```

### Test Data Creation
```bash
python -c "
from pathlib import Path
import numpy as np

data_root = Path('doa_normalized_config_c_corrected')
for angle in [80, 85, 90, 95, 100]:
    angle_dir = data_root / f'angle_{angle:03d}'
    angle_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        audio = np.random.randn(145920).astype(np.float32) * 0.1
        np.save(angle_dir / f'clip_{i:03d}.npy', audio)
print('✓ Test data created')
"
```

---

## 📈 Expected Outputs

### Smoke Test Results
```
results/day10_14_smoke/
├── baseline_rm_adapters/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── README.md
├── baseline_rm_heads.pt          (2.0 MB)
├── multimodal_rm_adapters/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   └── README.md
└── multimodal_rm_heads.pt        (3.3 MB)
```

### Training Logs
- BT Pair Loss: ~0.66 (epoch 0, untrained)
- Top-1 Accuracy: 0% (baseline for random init)
- Trainable Parameters: ~541K / 4.2M (12.8%)

---

## ⚙️ Configuration Presets

### Smoke Test (Fast)
```bash
--rm-epochs 2 --batch-size 2 --max-samples 5 --device cpu
```

### Development (Medium)
```bash
--rm-epochs 10 --batch-size 8 --max-samples 100 --device auto
```

### Production (Full)
```bash
--rm-epochs 50 --batch-size 16 --max-samples 0 --device cuda
```

---

## 🐛 Troubleshooting

### Issue: Module not found
```bash
# Solution 1: Set PYTHONPATH
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Solution 2: Reinstall
pip install -e .
```

### Issue: Missing data
```bash
# Smoke test creates synthetic data automatically
# Or manually create:
python -c "from pathlib import Path; import numpy as np; ..."
```

### Issue: Out of memory
```bash
# Reduce parameters:
--batch-size 1 --max-samples 3
```

---

## 📚 Documentation Links

- [Full Summary](DAY_10_14_SMOKE_TEST_SUMMARY.md)
- [中文總結](DAY_10_14_煙霧測試總結_中文.md)
- [Script Guide](SCRIPTS_EXECUTION_GUIDE.md)
- [Architecture](docs/ICL_ARCHITECTURE_EXPLAINED.md)

---

## ✅ Success Criteria

- [x] Both experiments complete without errors
- [x] Extended vocabulary (3,641 tokens) loads
- [x] Multi-modal prompts generated correctly
- [x] Checkpoints saved in correct format
- [x] Backward compatibility preserved

---

**Status**: ✅ All tests passing  
**Next**: Run full comparison experiments with `run_comparison_experiments.sh`
