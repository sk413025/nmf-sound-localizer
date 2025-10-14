# 快速執行參考

## 🚀 一鍵執行 Smoke Test

```bash
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f
conda activate trl-training
bash run_smoke_test.sh
```

## 📋 三個腳本的執行命令

### 1️⃣ 訓練 Reward Model

```bash
python scripts/train_reward_model_lora.py \
  --data-root doa_normalized_config_c_corrected \
  --tf-path h_matrix_normalized_original_to_box.pth \
  --w-path doa_normalized_config_c_corrected/models/usm.pth \
  --s-root doa_normalized_config_c_corrected \
  --K 2 \
  --rm-epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --device auto \
  --out results/rm_smoke_test
```

**輸出**:
- `results/rm_smoke_test_adapters/` (LoRA 適配器)
- `results/rm_smoke_test_heads.pt` (Embeddings + V-head)

---

### 2️⃣ 訓練 SFT Policy

**依賴**: 必須先完成 Step 1

```bash
python scripts/train_sft_policy_with_rm.py \
  --data-root doa_normalized_config_c_corrected \
  --rm-adapters results/rm_smoke_test_adapters \
  --rm-heads results/rm_smoke_test_heads.pt \
  --K 2 \
  --epochs 2 \
  --batch-size 4 \
  --max-samples 10 \
  --device auto \
  --out results/sft_smoke_test
```

**輸出**:
- `results/sft_smoke_test_policy_adapters/` (Policy 適配器)
- `results/sft_smoke_test_policy_heads.pt` (Policy Embeddings)

---

### 3️⃣ 訓練 TRL PPO

**依賴**: 必須先完成 Step 1；Step 2 為可選（warm start）

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
  --device auto
```

## 📊 關鍵參數對照表

| 用途 | Smoke Test | 正式訓練 |
|------|------------|----------|
| `--max-samples` | 10 | 0 (全部) |
| `--rm-epochs` | 2 | 20-100 |
| `--epochs` (SFT) | 2 | 3-10 |
| `--epochs` (PPO) | 1 | 5-20 |
| `--batch-size` | 2-4 | 8-32 |
| `--K` | 2 | 3-5 |

## 🔧 常用參數說明

### 所有腳本共用
- `--data-root`: 數據根目錄
- `--K`: 方向數量（預測幾個角度）
- `--device`: `auto`/`cpu`/`mps`/`cuda`
- `--max-samples`: 限制樣本數（0=全部）
- `--seed`: 隨機種子（預設 0）

### Script 1 專用
- `--tf-path`: 轉移函數檔案
- `--w-path`: W 矩陣檔案
- `--s-root`: S 數據集根目錄
- `--rm-epochs`: RM 訓練 epochs
- `--teacher`: `fit` 或 `euc`
- `--lora-r`: LoRA rank (4/8/16)

### Script 2 專用
- `--rm-adapters`: RM 適配器路徑
- `--rm-heads`: RM heads 檔案
- `--epochs`: SFT 訓練 epochs
- `--use-lora`: 是否在 policy 使用 LoRA

### Script 3 專用
- `--rm-adapters`: RM 適配器路徑
- `--rm-heads`: RM heads 檔案
- `--sft-policy-adapters`: (可選) SFT 適配器
- `--sft-policy-heads`: (可選) SFT heads
- `--ppo-epochs`: PPO 內部迭代次數

## ⚠️ 注意事項

1. **執行順序**: 必須按照 1 → 2 → 3 的順序執行
2. **環境**: 確保使用 `trl-training` conda 環境
3. **數據**: 確認 `doa_normalized_config_c_corrected/` 存在
4. **檔案**: 確認 `h_matrix_normalized_original_to_box.pth` 和 `usm.pth` 存在
5. **記憶體**: PPO 訓練較耗記憶體，如遇問題降低 batch-size

## 🐛 疑難排解

### 找不到模組
```bash
conda activate trl-training
pip install peft trl transformers torch
```

### 記憶體不足
```bash
# 降低參數
--batch-size 2
--max-samples 5
```

### 裝置錯誤
```bash
# 使用 CPU
--device cpu
```

## 📖 詳細文檔

查看 `SCRIPTS_EXECUTION_GUIDE.md` 獲取完整說明和進階配置。
