#!/usr/bin/env bash
# Phase 6: Batch Evaluation - White Noise SNR Sweep (Transformer Model)
# Uses omp-transformer-ldv.py with FullTransformerRoutedSoftOMP for accurate evaluation
# Expected: SNR=Inf achieves 100% accuracy (baseline commit 1f6b68c)

set -e  # Exit on error

# Setup environment
source ~/.zshrc
conda activate wavtokenizer

# Directories
WORKSPACE="$HOME/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace"
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"
DATA_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-16k"
RESULTS_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/results/white_noise_transformer"
H_MATRIX="${HOME}/LDV-data-processed/h_matrix_box_ldv_correct.pth"
USM_MODEL="${WORKSPACE}/doa_normalized_config_c_corrected/models/usm.pth"

# Training/Evaluation parameters (match baseline commit 1f6b68c)
EPOCHS=10
BATCH_SIZE=16
LR=3e-3
D_MODEL=64
NHEAD=2
NLAYERS=1
N_ATOMS=8
STEPS=6
TOP_E=2
TOP_L=2
DEVICE="cpu"

# SNR levels to evaluate
SNR_LEVELS=("Inf" "30dB" "20dB" "15dB" "10dB" "5dB" "0dB")

echo "========================================================================"
echo "Phase 6: White Noise SNR Sweep - Transformer Model Evaluation"
echo "========================================================================"
echo "Total evaluation runs: ${#SNR_LEVELS[@]} (7 SNR levels)"
echo "Expected: SNR=∞ achieves 100% accuracy (baseline verification)"
echo "Goal: Identify SNR threshold where performance degrades below 90%"
echo ""
echo "Model configuration:"
echo "  - Architecture: FullTransformerRoutedSoftOMP"
echo "  - d_model=${D_MODEL}, nhead=${NHEAD}, nlayers=${NLAYERS}"
echo "  - Atom reduction: K-means 50→${N_ATOMS}"
echo "  - Routing: g-mode (physics-based)"
echo "  - Training: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}, lr=${LR}"
echo ""
echo "Data configuration:"
echo "  - H matrix: ${H_MATRIX}"
echo "  - USM (W):  ${USM_MODEL}"
echo "  - Device: ${DEVICE}"
echo "========================================================================"
echo ""

# Create results directory
mkdir -p "${RESULTS_BASE}"

# Function to evaluate one SNR level
evaluate_snr_level() {
    local snr_label=$1

    # Special case: Use BASELINE data for SNR=Inf verification
    if [ "$snr_label" = "Inf" ]; then
        # Use 48kHz baseline dataset (matches H matrix training)
        local dataset_root="${HOME}/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snrInf_sync_vad_normalized"
        echo "  Using BASELINE data for SNR=Inf verification (48kHz)"
    else
        # Use synthetic SNR data for other levels (48kHz version, NOT 16kHz resampled)
        local dataset_root="${HOME}/LDV-data-experiments/snr-synthetic-2025-12/processed-48k/white_noise_box_snr${snr_label}_sync_vad_normalized"
        echo "  Using SYNTHETIC SNR data (48kHz, same as baseline)"
    fi

    # Output directory
    local out_dir="${RESULTS_BASE}/snr_${snr_label}"

    echo "[EVAL] Processing SNR=${snr_label}"
    echo "  Dataset: ${dataset_root}"
    echo "  Output:  ${out_dir}"

    # Check dataset exists
    if [ ! -d "${dataset_root}" ]; then
        echo "  ✗ ERROR: Dataset not found: ${dataset_root}"
        return 1
    fi

    # Count files
    local file_count=$(find "${dataset_root}" -name "*.npy" | wc -l | tr -d ' ')
    echo "  Files: ${file_count} (expected: 111)"

    if [ "${file_count}" -ne 111 ]; then
        echo "  ⚠️  WARNING: Expected 111 files, found ${file_count}"
    fi

    # Run evaluation with Transformer model
    python "${WORKSPACE}/scripts/omp-transformer-ldv.py" \
        --dataset_root "${dataset_root}" \
        --h_path "${H_MATRIX}" \
        --w_path "${USM_MODEL}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LR} \
        --d_model ${D_MODEL} \
        --nhead ${NHEAD} \
        --nlayers ${NLAYERS} \
        --routing_mode g \
        --device ${DEVICE} \
        --n_atoms ${N_ATOMS} \
        --steps ${STEPS} \
        --top_e ${TOP_E} \
        --top_l ${TOP_L} \
        --out_dir "${out_dir}"

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "  ✓ Evaluation complete: ${out_dir}"
        echo ""
    else
        echo "  ✗ Evaluation failed with exit code ${exit_code}"
        echo ""
        return ${exit_code}
    fi
}

# Main evaluation loop
echo "========================================================================"
echo "EVALUATION PHASE: White Noise SNR Sweep (Transformer Model)"
echo "========================================================================"
echo ""

# Evaluate baseline (SNR=∞) first
echo "--- BASELINE EVALUATION (SNR=∞) ---"
evaluate_snr_level "Inf"

echo "========================================================================"
echo "Baseline (SNR=∞) evaluation complete!"
echo ""
echo "CRITICAL CHECKPOINT:"
echo "  - Check results in: ${RESULTS_BASE}/snr_Inf/"
echo "  - Expected: Overall accuracy = 100%"
echo "  - Baseline (commit 1f6b68c): 100% accuracy on white noise"
echo ""
echo "If baseline matches expectations → Proceed with SNR sweep"
echo "If baseline DOES NOT match → Debug before continuing"
echo "========================================================================"
echo ""

# Ask user before continuing (optional - can be commented out for automation)
# read -p "Press Enter to continue with SNR sweep, or Ctrl+C to stop..."

# Evaluate SNR degradation sweep
echo ""
echo "--- SNR DEGRADATION SWEEP ---"
for snr in "${SNR_LEVELS[@]:1}"; do  # Skip first element (Inf, already evaluated)
    evaluate_snr_level "$snr"
done

echo "========================================================================"
echo "Phase 6 (White Noise Transformer) Evaluation Complete!"
echo "========================================================================"
echo ""
echo "Evaluation results saved to: ${RESULTS_BASE}/snr_*/eval_results.json"
echo ""
echo "Summary:"
for snr in "${SNR_LEVELS[@]}"; do
    result_file="${RESULTS_BASE}/snr_${snr}/eval_results.json"
    if [ -f "${result_file}" ]; then
        # Extract accuracy using Python
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('${result_file}'))['overall_accuracy']*100:.2f}\")" 2>/dev/null || echo "N/A")
        echo "  ✓ SNR=${snr}: ${accuracy}% accuracy"
    else
        echo "  ✗ SNR=${snr}: MISSING (evaluation may have failed)"
    fi
done
echo ""
echo "Next steps:"
echo "1. Analyze accuracy degradation curve"
echo "2. Identify 90% accuracy threshold"
echo "3. Generate SNR vs Accuracy plot"
echo "4. Compare with baseline (without Transformer): 83.78%"
echo ""
