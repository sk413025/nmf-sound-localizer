# 訓練流程圖

## 執行流程

```
┌─────────────────────────────────────────────────────────────┐
│                    環境準備                                  │
│  conda activate trl-training                                │
│  cd worktrees/sync-from-0bed93f                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: 訓練 Reward Model (RM)                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  腳本: train_reward_model_lora.py                          │
│                                                              │
│  輸入:                                                       │
│    • 數據目錄 (doa_normalized_config_c_corrected)           │
│    • 轉移函數 (h_matrix_normalized_original_to_box.pth)    │
│    • W 矩陣 (usm.pth)                                       │
│    • S 數據集根目錄                                         │
│                                                              │
│  訓練參數 (Smoke Test):                                     │
│    • --rm-epochs 2                                          │
│    • --max-samples 10                                       │
│    • --K 2                                                  │
│    • --batch-size 4                                         │
│                                                              │
│  輸出:                                                       │
│    ✓ results/rm_smoke_test_adapters/  (LoRA 適配器)       │
│    ✓ results/rm_smoke_test_heads.pt   (Embeddings+V-head) │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: 訓練 SFT Policy                                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  腳本: train_sft_policy_with_rm.py                         │
│                                                              │
│  輸入:                                                       │
│    • 數據目錄                                               │
│    • RM 適配器 (從 Step 1)                                  │
│    • RM heads (從 Step 1)                                   │
│                                                              │
│  訓練參數 (Smoke Test):                                     │
│    • --epochs 2                                             │
│    • --max-samples 10                                       │
│    • --K 2                                                  │
│    • --batch-size 4                                         │
│                                                              │
│  輸出:                                                       │
│    ✓ results/sft_smoke_test_policy_adapters/  (Policy LoRA)│
│    ✓ results/sft_smoke_test_policy_heads.pt   (Embeddings) │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: 訓練 TRL PPO (強化學習)                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│  腳本: train_trl_ppo_with_rm.py                            │
│                                                              │
│  輸入:                                                       │
│    • 數據目錄                                               │
│    • RM 適配器 (從 Step 1) [必要]                          │
│    • RM heads (從 Step 1) [必要]                           │
│    • SFT Policy 適配器 (從 Step 2) [可選 warm start]       │
│    • SFT Policy heads (從 Step 2) [可選 warm start]        │
│                                                              │
│  訓練參數 (Smoke Test):                                     │
│    • --epochs 1                                             │
│    • --ppo-epochs 1                                         │
│    • --max-samples 10                                       │
│    • --K 2                                                  │
│    • --batch-size 2                                         │
│                                                              │
│  輸出:                                                       │
│    ✓ PPO 訓練完成的 Policy 模型                            │
│    ✓ (輸出位置依 TRL 配置，可能在 trl-output/)             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   訓練完成！   │
                    └───────────────┘
```

## 數據流向

```
┌──────────────┐
│  原始音訊數據  │
│  angle_*/    │
└──────┬───────┘
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌────────────┐        ┌──────────┐
│ 轉移函數 H  │        │ W 矩陣   │
│  .pth      │        │  .pth    │
└─────┬──────┘        └────┬─────┘
      │                    │
      └──────────┬─────────┘
                 │
                 ▼
         ┌──────────────┐
         │  Reward Model │
         │  (LoRA + V-head)│
         └───────┬────────┘
                 │
                 ├──────────────┐
                 │              │
                 ▼              ▼
         ┌──────────┐    ┌──────────┐
         │ SFT Policy│    │ PPO Policy│
         │ (Teacher) │    │ (RL)      │
         └───────────┘    └───────────┘
```

## 依賴關係

```
train_reward_model_lora.py (獨立運行)
              │
              ├─── 輸出: RM 模型
              │
              ▼
┌─────────────────────────────────┐
│                                 │
▼                                 ▼
train_sft_policy_with_rm.py    train_trl_ppo_with_rm.py
(需要 RM)                       (需要 RM)
│                                 │
├─── 輸出: SFT Policy ────────────┤
                                  │
                                  ▼
                            train_trl_ppo_with_rm.py
                            (可選: SFT warm start)
```

## 檔案關係圖

```
worktrees/sync-from-0bed93f/
│
├── 腳本 (scripts/)
│   ├── train_reward_model_lora.py     [1]
│   ├── train_sft_policy_with_rm.py    [2]
│   └── train_trl_ppo_with_rm.py       [3]
│
├── 輸入數據
│   ├── doa_normalized_config_c_corrected/
│   │   ├── angle_000/, angle_005/, ...  (音訊數據)
│   │   └── models/usm.pth               (W 矩陣)
│   └── h_matrix_normalized_original_to_box.pth (轉移函數)
│
└── 輸出結果 (results/)
    ├── rm_smoke_test_adapters/          [1] 輸出
    ├── rm_smoke_test_heads.pt           [1] 輸出
    ├── sft_smoke_test_policy_adapters/  [2] 輸出
    └── sft_smoke_test_policy_heads.pt   [2] 輸出
```

## 參數傳遞鏈

```
Step 1: train_reward_model_lora.py
  輸入參數:
    --data-root          → 數據目錄
    --tf-path            → 轉移函數檔案
    --w-path             → W 矩陣檔案
    --s-root             → S 數據集根目錄
  
  輸出:
    ${out}_adapters/     ────┐
    ${out}_heads.pt      ────┼──→ 傳遞給 Step 2 & 3
                             │
                             │
Step 2: train_sft_policy_with_rm.py  │
  輸入參數:                           │
    --data-root          → 數據目錄   │
    --rm-adapters        ←────────────┘
    --rm-heads           ←────────────┐
                                      │
  輸出:                               │
    ${out}_policy_adapters/  ─────┐  │
    ${out}_policy_heads.pt   ─────┼──┼──→ 傳遞給 Step 3 (可選)
                                  │  │
                                  │  │
Step 3: train_trl_ppo_with_rm.py  │  │
  輸入參數:                        │  │
    --data-root          → 數據目錄│  │
    --rm-adapters        ←───────────┘
    --rm-heads           ←────────────┐
    --sft-policy-adapters ←──────┘    │ (可選)
    --sft-policy-heads    ←───────────┘ (可選)
```

## 記憶體與時間估計 (Smoke Test)

| 步驟 | 預估時間 | 預估記憶體 |
|------|----------|------------|
| Step 1 (RM) | 2-5 分鐘 | ~2-4 GB |
| Step 2 (SFT) | 1-3 分鐘 | ~2-4 GB |
| Step 3 (PPO) | 2-5 分鐘 | ~3-6 GB |
| **總計** | **5-13 分鐘** | **~6 GB peak** |

*實際時間依硬體而定 (CPU/GPU/MPS)*

## 成功指標

### Step 1 完成
- ✓ Loss 下降
- ✓ 生成 `_adapters/` 目錄
- ✓ 生成 `_heads.pt` 檔案
- ✓ 輸出 embedding 質量分析

### Step 2 完成
- ✓ Loss 下降
- ✓ 生成 `_policy_adapters/` 目錄
- ✓ 生成 `_policy_heads.pt` 檔案

### Step 3 完成
- ✓ 訓練運行無錯誤
- ✓ 獎勵信號正常計算
- ✓ PPO 更新成功執行
