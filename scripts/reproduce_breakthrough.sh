#!/bin/bash
# 94.1% DOA突破性實驗一鍵重現腳本
# Usage: bash scripts/reproduce_breakthrough.sh

set -e  # Exit on any error

echo "=========================================="
echo "94.1% DOA 突破性實驗重現腳本"
echo "=========================================="

# 環境檢查
if [[ "$CONDA_DEFAULT_ENV" != "wavtokenizer" ]]; then
    echo "❌ 錯誤：請先激活conda環境"
    echo "執行: source ~/.zshrc && conda activate wavtokenizer"
    exit 1
fi

# 設置環境變量
export PYTHONPATH="/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace:$PYTHONPATH"

# 數據路徑配置
BOX_DATA_PATH="/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original/white_noise_box_data_no_edge_sync_vad"
TF_FILE="h_matrix_80_150_freq_300_3000.pth"
OUTPUT_DIR="doa_breakthrough_reproduction_$(date +%Y%m%d_%H%M%S)"

echo "環境配置完成 ✅"
echo "數據路徑: $BOX_DATA_PATH"
echo "輸出目錄: $OUTPUT_DIR"
echo ""

# 檢查數據路徑
if [[ ! -d "$BOX_DATA_PATH" ]]; then
    echo "❌ 錯誤：數據路徑不存在 $BOX_DATA_PATH"
    exit 1
fi

# 檢查傳遞函數文件
if [[ ! -f "$TF_FILE" ]]; then
    echo "⚠️  傳遞函數文件不存在，開始重新估計..."
    echo "步驟 1/2: 傳遞函數估計"
    python scripts/estimate_transfer_functions.py \
        "$BOX_DATA_PATH" \
        --output "$TF_FILE" \
        --freq-min 300.0 --freq-max 3000.0 \
        --files-per-angle 50 --time-pooling geometric
    
    if [[ $? -eq 0 ]]; then
        echo "✅ 傳遞函數估計完成"
    else
        echo "❌ 傳遞函數估計失敗"
        exit 1
    fi
else
    echo "✅ 找到現有傳遞函數文件: $TF_FILE"
fi

echo ""
echo "步驟 2/2: DOA定位評估"
echo "關鍵配置："
echo "  - 傳遞函數: $TF_FILE"
echo "  - 測試數據: $BOX_DATA_PATH (麥克風陣列數據)"
echo "  - 頻率範圍: 300-3000 Hz"
echo "  - 期望結果: 94.1% 準確率"

python scripts/run_localization.py \
    --tf-path "$TF_FILE" \
    --speech-data-root "$BOX_DATA_PATH" \
    --output "$OUTPUT_DIR" \
    --freq-min 300.0 --freq-max 3000.0 \
    --tolerance-degrees 10 --n-sources 1 \
    --device cpu

# 結果驗證
REPORT_FILE="$OUTPUT_DIR/evaluation/evaluation_report.txt"
if [[ -f "$REPORT_FILE" ]]; then
    echo ""
    echo "=========================================="
    echo "🎉 實驗完成！結果驗證："
    echo "=========================================="
    
    # 提取關鍵指標
    ACCURACY=$(grep "Accuracy:" "$REPORT_FILE" | head -1 | awk '{print $2}')
    MEAN_ERROR=$(grep "Mean Error:" "$REPORT_FILE" | awk '{print $3}')
    
    echo "總體準確率: $ACCURACY"
    echo "平均誤差: $MEAN_ERROR"
    
    # 檢查是否達到突破性結果
    ACCURACY_NUM=$(echo $ACCURACY | sed 's/%//')
    if (( $(echo "$ACCURACY_NUM >= 94.0" | bc -l) )); then
        echo "✅ 成功重現突破性結果！"
        echo ""
        echo "詳細結果："
        echo "----------------------------------------"
        cat "$REPORT_FILE"
        echo "----------------------------------------"
        echo ""
        echo "實驗結果保存在: $OUTPUT_DIR"
        echo "關鍵文件:"
        echo "  - 結果報告: $REPORT_FILE"
        echo "  - 詳細數據: $OUTPUT_DIR/evaluation/evaluation_results.pth"
        echo "  - USM模型: $OUTPUT_DIR/models/usm.pth"
        echo "  - 定位器: $OUTPUT_DIR/models/localizer.pth"
    else
        echo "❌ 實驗結果未達到預期 (期望: ≥94.0%, 實際: $ACCURACY)"
        echo "請檢查配置或參考故障排除指南"
        exit 1
    fi
else
    echo "❌ 未找到結果報告文件"
    exit 1
fi

echo ""
echo "🎯 實驗重現成功！"
echo "詳細重現指南: docs/breakthrough_experiment_reproduction_guide.md"