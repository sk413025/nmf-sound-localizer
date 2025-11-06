#!/bin/bash
# G-Teacher 完整驗證流程
# 日期: 2025-11-06
# 目的: 生成 g-teacher 軌跡並立即驗證準確性

set -e  # Exit on error

echo "================================================================================"
echo "G-Teacher 完整驗證流程"
echo "================================================================================"
echo "開始時間: $(date)"
echo ""

# Configuration
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="doa_normalized_config_c_corrected/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
OUT_DIR="results/dt_traj_g_verification_$(date +%Y%m%d_%H%M%S)"
TEACHER="g"
K=6
N_ATOMS=8
ATOM_REDUCE_MODE="kcenter"
FS=16000
N_FFT=2048
FREQ_MIN=300.0
FREQ_MAX=3000.0
RTG_TARGET_RESID=0.02
RTG_TARGET_ACC=0.95
SEED=42
DEVICE="cpu"

echo "配置參數:"
echo "  Teacher policy: $TEACHER (classic OMP with |g| energies)"
echo "  H matrix: $H_PATH"
echo "  W matrix: $W_PATH"
echo "  Dataset: $DATASET_ROOT"
echo "  Output: $OUT_DIR"
echo "  K (trajectory steps): $K"
echo "  M (atoms per expert): $N_ATOMS (reduced via $ATOM_REDUCE_MODE)"
echo "  STFT: fs=$FS, n_fft=$N_FFT, band=[$FREQ_MIN, $FREQ_MAX] Hz"
echo "  RTG targets: resid=$RTG_TARGET_RESID, acc=$RTG_TARGET_ACC"
echo "  Random seed: $SEED"
echo "  Device: $DEVICE"
echo ""

# Verify prerequisites
echo "檢查前置條件..."
if [ ! -f "$H_PATH" ]; then
    echo "✗ ERROR: H matrix not found at $H_PATH"
    exit 1
fi

if [ ! -f "$W_PATH" ]; then
    echo "✗ ERROR: W matrix not found at $W_PATH"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "✗ ERROR: Dataset root not found at $DATASET_ROOT"
    exit 1
fi

echo "✓ 所有前置條件滿足"
echo ""

# Create output directory
mkdir -p "$OUT_DIR"

# Export Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================================================================
# 階段 1: 生成 G-Teacher 軌跡
# ============================================================================
echo "============================================================================"
echo "階段 1: 生成 G-Teacher 軌跡"
echo "============================================================================"
echo ""

python -u doa_rl/trajectories/offline_dt_dataset.py \
    --teacher "$TEACHER" \
    --K "$K" \
    --n_atoms "$N_ATOMS" \
    --atom_reduce_mode "$ATOM_REDUCE_MODE" \
    --h_path "$H_PATH" \
    --w_path "$W_PATH" \
    --dataset_root "$DATASET_ROOT" \
    --out_dir "$OUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --fs "$FS" \
    --n_fft "$N_FFT" \
    --freq_min "$FREQ_MIN" \
    --freq_max "$FREQ_MAX" \
    --rtg_target_resid "$RTG_TARGET_RESID" \
    --rtg_target_acc "$RTG_TARGET_ACC" \
    2>&1 | tee "$OUT_DIR/generation.log"

echo ""
echo "✓ 軌跡生成完成"
echo ""

# ============================================================================
# 階段 2: 驗證軌跡準確性
# ============================================================================
echo "============================================================================"
echo "階段 2: 驗證軌跡準確性"
echo "============================================================================"
echo ""

python -u verify_g_teacher_trajectories.py \
    --traj_jsonl "$OUT_DIR/trajectories.jsonl" \
    --h_path "$H_PATH" \
    --w_path "$W_PATH" \
    --n_atoms "$N_ATOMS" \
    --atom_reduce_mode "$ATOM_REDUCE_MODE" \
    --fs "$FS" \
    --n_fft "$N_FFT" \
    --freq_min "$FREQ_MIN" \
    --freq_max "$FREQ_MAX" \
    --seed "$SEED" \
    2>&1 | tee "$OUT_DIR/verification.log"

VERIFY_EXIT=$?

echo ""
echo "============================================================================"
echo "完整驗證流程結束"
echo "============================================================================"
echo "結束時間: $(date)"
echo "輸出目錄: $OUT_DIR"
echo ""

if [ $VERIFY_EXIT -eq 0 ]; then
    echo "✅ 驗證成功! G-Teacher 軌跡準確無誤。"
    echo ""
    echo "生成的文件:"
    ls -lh "$OUT_DIR"
    echo ""
    echo "下一步: 使用此軌跡訓練 Decision Transformer"
    echo "  指令: ./run_dt_training.sh --traj_dir $OUT_DIR --epochs 480"
else
    echo "⚠️  驗證失敗! 請檢查 verification.log 了解詳情。"
    echo ""
    echo "常見問題:"
    echo "  1. STFT 參數不一致"
    echo "  2. Atom reduction seed 不同"
    echo "  3. G-teacher 實現邏輯差異"
    echo ""
    echo "請查看詳細日誌:"
    echo "  生成日誌: $OUT_DIR/generation.log"
    echo "  驗證日誌: $OUT_DIR/verification.log"
fi

echo "================================================================================"

exit $VERIFY_EXIT
