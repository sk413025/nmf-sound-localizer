#!/bin/bash
# Smoke Test 執行腳本
# 用途：快速測試三個訓練腳本是否能正常運行
# 使用方式：bash run_smoke_test.sh

set -e  # 遇到錯誤立即停止

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=========================================="
echo "開始 Smoke Test"
echo "==========================================${NC}"

# 檢查環境
echo -e "${YELLOW}檢查環境...${NC}"
if ! command -v conda &> /dev/null; then
    echo -e "${RED}錯誤: conda 未安裝${NC}"
    exit 1
fi

# 激活環境
echo -e "${YELLOW}激活 trl-training 環境...${NC}"
eval "$(conda shell.bash hook)"
conda activate trl-training

# 確認目錄
cd /Users/sbplab/jnrle/LDVReorientation/worktrees/sync-from-0bed93f

# 創建輸出目錄
mkdir -p results

# 檢查必要文件
echo -e "${YELLOW}檢查必要文件...${NC}"
if [ ! -f "h_matrix_normalized_original_to_box.pth" ]; then
    echo -e "${RED}錯誤: 找不到 h_matrix_normalized_original_to_box.pth${NC}"
    exit 1
fi

if [ ! -f "doa_normalized_config_c_corrected/models/usm.pth" ]; then
    echo -e "${RED}錯誤: 找不到 usm.pth${NC}"
    exit 1
fi

if [ ! -d "doa_normalized_config_c_corrected" ]; then
    echo -e "${RED}錯誤: 找不到數據目錄 doa_normalized_config_c_corrected${NC}"
    exit 1
fi

echo -e "${GREEN}環境檢查通過！${NC}"
echo ""

# ===========================
# Step 1: 訓練 Reward Model
# ===========================
echo -e "${GREEN}=========================================="
echo "Step 1/3: 訓練 Reward Model (RM)"
echo "==========================================${NC}"
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

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 1 完成${NC}"
else
    echo -e "${RED}✗ Step 1 失敗${NC}"
    exit 1
fi
echo ""

# ===========================
# Step 2: 訓練 SFT Policy
# ===========================
echo -e "${GREEN}=========================================="
echo "Step 2/3: 訓練 SFT Policy"
echo "==========================================${NC}"
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

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 2 完成${NC}"
else
    echo -e "${RED}✗ Step 2 失敗${NC}"
    exit 1
fi
echo ""

# ===========================
# Step 3: 訓練 TRL PPO
# ===========================
echo -e "${GREEN}=========================================="
echo "Step 3/3: 訓練 TRL PPO (with SFT warm start)"
echo "==========================================${NC}"
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

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Step 3 完成${NC}"
else
    echo -e "${RED}✗ Step 3 失敗${NC}"
    exit 1
fi
echo ""

# ===========================
# 完成總結
# ===========================
echo -e "${GREEN}=========================================="
echo "✓ Smoke Test 全部完成！"
echo "==========================================${NC}"
echo ""
echo "輸出文件位置："
echo "  1. RM 模型:"
echo "     - results/rm_smoke_test_adapters/"
echo "     - results/rm_smoke_test_heads.pt"
echo ""
echo "  2. SFT Policy 模型:"
echo "     - results/sft_smoke_test_policy_adapters/"
echo "     - results/sft_smoke_test_policy_heads.pt"
echo ""
echo "  3. PPO 輸出:"
echo "     - (根據 TRL 配置，可能在 trl-output/)"
echo ""
echo -e "${YELLOW}下一步：${NC}"
echo "  - 檢查輸出文件是否生成"
echo "  - 查看訓練日誌確認無錯誤"
echo "  - 如需正式訓練，請修改參數（參考 SCRIPTS_EXECUTION_GUIDE.md）"
