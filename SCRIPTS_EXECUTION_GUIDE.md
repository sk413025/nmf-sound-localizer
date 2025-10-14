# 執行指南：三個訓練腳本

本指南說明如何執行以下三個訓練腳本的 smoke test：
1. `train_reward_model_lora.py` - 訓練 Reward Model (RM) with LoRA
2. `train_sft_policy_with_rm.py` - 使用 RM 訓練 SFT Policy
3. `train_trl_ppo_with_rm.py` - 使用 RM 訓練 TRL PPO

## 環境設置

### 1. 激活 Conda 環境
```bash
conda activate trl-training
```

### 2. 確認當前目錄
```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f
```

## 腳本執行順序

### Script 1: train_reward_model_lora.py

**用途**: 訓練 Reward Model，這是後續兩個腳本的基礎

**必要參數**:
- `--data-root`: 訓練數據目錄（包含 angle_* 子目錄）
- `--tf-path`: 轉移函數文件路徑（.pth 文件）
- `--w-path`: W 矩陣路徑（.pth 文件）
- `--s-root`: S 數據集根目錄

**Smoke Test 執行範例**:
```bash
python scripts/train_reward_model_lora.py \
  --data-root doa_normalized_config_c_corrected \
  --tf-path h_matrix_normalized_original_to_box.pth \
  --w-path doa_normalized_config_c_corrected/models/usm.pth \
  --s-root doa_normalized_config_c_corrected \
  --K 2 \
  --teacher fit \
  --rm-epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --lora-r 4 \
  --lora-alpha 8 \
  --device auto \
  --out results/rm_smoke_test
```

**Smoke Test 關鍵參數說明**:
- `--rm-epochs 2`: 只訓練 2 個 epoch（快速測試）
- `--max-samples 10`: 只使用 10 個樣本
- `--batch-size 4`: 小 batch size
- `--K 2`: 使用 2 個方向（減少計算）
- `--lora-r 4`: 較小的 LoRA rank

**輸出**:
- `results/rm_smoke_test_adapters/`: LoRA 適配器
- `results/rm_smoke_test_heads.pt`: Embeddings + V-head

---

### Script 2: train_sft_policy_with_rm.py

**用途**: 使用訓練好的 RM 作為 teacher 來訓練 SFT Policy

**必要參數**:
- `--data-root`: 訓練數據目錄
- `--rm-adapters`: RM LoRA 適配器路徑（來自 Script 1）
- `--rm-heads`: RM heads 檔案路徑（來自 Script 1）

**Smoke Test 執行範例**:
```bash
python scripts/train_sft_policy_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --K 2 \
  --epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --lr 1e-4 \
  --device auto \
  --out results/sft_smoke_test
```

**Smoke Test 關鍵參數說明**:
- `--epochs 2`: 只訓練 2 個 epoch
- `--max-samples 10`: 只使用 10 個樣本
- `--batch-size 4`: 小 batch size
- `--K 2`: 與 RM 訓練保持一致

**輸出**:
- `results/sft_smoke_test_policy_adapters/`: Policy 適配器
- `results/sft_smoke_test_policy_heads.pt`: Policy embeddings

---

### Script 3: train_trl_ppo_with_rm.py

**用途**: 使用 TRL PPO 和 RM 進行強化學習訓練

**必要參數**:
- `--data-root`: 訓練數據目錄
- `--rm-adapters`: RM LoRA 適配器路徑（來自 Script 1）
- `--rm-heads`: RM heads 檔案路徑（來自 Script 1）

**可選參數**（用於 warm start）:
- `--sft-policy-adapters`: SFT policy 適配器路徑（來自 Script 2）
- `--sft-policy-heads`: SFT policy embeddings（來自 Script 2）

**Smoke Test 執行範例（不使用 SFT warm start）**:
```bash
python scripts/train_trl_ppo_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --K 2 \
  --epochs 1 \
  --ppo-epochs 1 \
  --batch-size 2 \
  --max-samples 10 \
  --lr 1e-4 \
  --device auto
```

**Smoke Test 執行範例（使用 SFT warm start）**:
```bash
python scripts/train_trl_ppo_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --sft-policy-adapters results/sft_smoke_test_policy_adapters \
  --sft-policy-heads results/sft_smoke_test_policy_heads.pt \
  --K 2 \
  --epochs 1 \
  --ppo-epochs 1 \
  --batch-size 2 \
  --max-samples 10 \
  --lr 1e-4 \
  --device auto
```

**Smoke Test 關鍵參數說明**:
- `--epochs 1`: 只訓練 1 個 epoch
- `--ppo-epochs 1`: PPO 內部迭代 1 次
- `--max-samples 10`: 只使用 10 個樣本
- `--batch-size 2`: 小 batch size（PPO 需要較大記憶體）

---

## 完整 Smoke Test 執行流程

執行以下命令進行完整的 smoke test：

```bash
#!/bin/bash
# 設置環境
conda activate trl-training
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f

# 創建輸出目錄
mkdir -p results

echo "=========================================="
echo "Step 1: 訓練 Reward Model (RM)"
echo "=========================================="
python scripts/train_reward_model_lora.py \
  --data-root doa_normalized_config_c_corrected \
  --tf-path h_matrix_normalized_original_to_box.pth \
  --w-path doa_normalized_config_c_corrected/models/usm.pth \
  --s-root doa_normalized_config_c_corrected \
  --K 2 \
  --teacher fit \
  --rm-epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --lora-r 4 \
  --lora-alpha 8 \
  --device auto \
  --out results/rm_smoke_test

echo ""
echo "=========================================="
echo "Step 2: 訓練 SFT Policy"
echo "=========================================="
python scripts/train_sft_policy_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --K 2 \
  --epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --lr 1e-4 \
  --device auto \
  --out results/sft_smoke_test

echo ""
echo "=========================================="
echo "Step 3: 訓練 TRL PPO (使用 SFT warm start)"
echo "=========================================="
python scripts/train_trl_ppo_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --sft-policy-adapters results/sft_smoke_test_policy_adapters \
  --sft-policy-heads results/sft_smoke_test_policy_heads.pt \
  --K 2 \
  --epochs 1 \
  --ppo-epochs 1 \
  --batch-size 2 \
  --max-samples 10 \
  --lr 1e-4 \
  --device auto

echo ""
echo "=========================================="
echo "Smoke Test 完成！"
echo "=========================================="
```

---

## 參數詳細說明

### 共同參數

| 參數 | 說明 | Smoke Test 建議值 | 正式訓練建議值 |
|------|------|-------------------|----------------|
| `--data-root` | 數據根目錄 | `doa_normalized_config_c_corrected` | 實際數據路徑 |
| `--K` | 方向數量 | `2` | `3-5` |
| `--device` | 計算設備 | `auto` | `auto/mps/cuda` |
| `--max-samples` | 最大樣本數 | `10` | `0` (全部) |
| `--batch-size` | Batch 大小 | `2-4` | `8-32` |
| `--seed` | 隨機種子 | `0` | `0` |

### Script 1 特有參數

| 參數 | 說明 | Smoke Test 建議值 | 正式訓練建議值 |
|------|------|-------------------|----------------|
| `--tf-path` | 轉移函數路徑 | `h_matrix_normalized_original_to_box.pth` | 實際 TF 路徑 |
| `--w-path` | W 矩陣路徑 | `doa_normalized_config_c_corrected/models/usm.pth` | 實際 W 路徑 |
| `--s-root` | S 數據集根目錄 | 與 data-root 相同 | 實際 S 路徑 |
| `--rm-epochs` | RM 訓練 epochs | `2` | `20-100` |
| `--teacher` | Teacher 類型 | `fit` | `fit` |
| `--lora-r` | LoRA rank | `4` | `8-16` |
| `--lora-alpha` | LoRA alpha | `8` | `16-32` |

### Script 2 特有參數

| 參數 | 說明 | Smoke Test 建議值 | 正式訓練建議值 |
|------|------|-------------------|----------------|
| `--rm-adapters` | RM 適配器路徑 | Script 1 輸出 | Script 1 輸出 |
| `--rm-heads` | RM heads 路徑 | Script 1 輸出 | Script 1 輸出 |
| `--epochs` | SFT 訓練 epochs | `2` | `3-10` |
| `--use-lora` | 是否使用 LoRA | 不使用 | 可選 |

### Script 3 特有參數

| 參數 | 說明 | Smoke Test 建議值 | 正式訓練建議值 |
|------|------|-------------------|----------------|
| `--epochs` | 訓練 epochs | `1` | `5-20` |
| `--ppo-epochs` | PPO 內部 epochs | `1` | `4-8` |
| `--sft-policy-adapters` | SFT 適配器（可選） | Script 2 輸出 | Script 2 輸出 |
| `--sft-policy-heads` | SFT heads（可選） | Script 2 輸出 | Script 2 輸出 |

---

## 常見問題

### Q1: 找不到數據目錄
**A**: 確認 `doa_normalized_config_c_corrected` 目錄存在且包含 `angle_*` 子目錄。

### Q2: CUDA/MPS 錯誤
**A**: 使用 `--device cpu` 或 `--device auto` 讓系統自動選擇。

### Q3: 記憶體不足
**A**: 降低 `--batch-size` 和 `--max-samples` 參數。

### Q4: 訓練時間太長
**A**: 對於 smoke test，已經設置了最小參數。如果還是太慢，可以進一步降低 `--max-samples` 到 5。

### Q5: RM 模型載入失敗
**A**: 確認 Script 1 成功完成並產生了 `_adapters` 目錄和 `_heads.pt` 文件。

---

## 進階配置

### 數據格式要求

數據目錄結構：
```
doa_normalized_config_c_corrected/
├── angle_000/
│   ├── sample_001.npy
│   ├── sample_002.npy
│   └── ...
├── angle_005/
│   └── ...
├── angle_010/
│   └── ...
└── models/
    └── usm.pth
```

### 輸出文件說明

**RM 訓練輸出**:
- `<out>_adapters/`: LoRA 適配器（目錄）
- `<out>_heads.pt`: 包含 embeddings 和 v_head 的檢查點

**SFT 訓練輸出**:
- `<out>_policy_adapters/`: Policy LoRA 適配器（目錄）
- `<out>_policy_heads.pt`: Policy embeddings 檢查點

**PPO 訓練輸出**:
- 根據 TRL 配置，可能輸出到 `trl-output/` 目錄

---

## 檢查點管理

### 保存已訓練的模型
```bash
# 創建備份
mkdir -p checkpoints
cp -r results/rm_smoke_test_adapters checkpoints/
cp results/rm_smoke_test_heads.pt checkpoints/
```

### 使用已有檢查點
如果已經有訓練好的 RM 模型，可以直接從 Script 2 開始：
```bash
python scripts/train_sft_policy_with_rm.py \
  --rm-adapters checkpoints/rm_smoke_test_adapters \
  --rm-heads checkpoints/rm_smoke_test_heads.pt \
  # ... 其他參數
```

---

## 監控訓練進度

訓練過程會輸出以下資訊：
- Loss 值
- Epoch 進度
- 評估指標（Top-1 accuracy, Recall@K）
- Embedding 質量分析

注意觀察：
1. Loss 是否在下降
2. Accuracy 是否在提升
3. 是否有錯誤訊息或警告

---

## 完整參數列表

如需查看腳本的所有可用參數，執行：
```bash
python scripts/train_reward_model_lora.py --help
python scripts/train_sft_policy_with_rm.py --help
python scripts/train_trl_ppo_with_rm.py --help
```
