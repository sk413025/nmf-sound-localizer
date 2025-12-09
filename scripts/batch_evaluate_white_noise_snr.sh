#!/usr/bin/env bash
# Phase 6: Batch Evaluation - White Noise SNR Sweep
# Evaluates full validation accuracy across all samples for each SNR level

set -e  # Exit on error

# Setup environment
source ~/.zshrc
conda activate wavtokenizer

# Directories
WORKSPACE="$HOME/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace"
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH}"
DATA_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-16k"
RESULTS_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/results/white_noise"
H_MATRIX="${HOME}/LDV-data-processed/h_matrix_box_ldv_correct.pth"
USM_MODEL="${WORKSPACE}/doa_normalized_config_c_corrected/models/usm.pth"

# Evaluation parameters (match training)
FREQ_MIN=300.0
FREQ_MAX=3000.0
STEPS=6
TOP_E=2
L=2
ITERS=2000  # Match training iterations for proper convergence
BATCH_SIZE=4
DEVICE="cpu"

# SNR levels to evaluate
SNR_LEVELS=("Inf" "30dB" "20dB" "15dB" "10dB" "5dB" "0dB")

echo "========================================================================"
echo "Phase 6: White Noise SNR Sweep - Full Validation Accuracy"
echo "========================================================================"
echo "Total evaluation runs: ${#SNR_LEVELS[@]} (7 SNR levels)"
echo "Expected: SNR=∞ achieves ~100% accuracy (baseline verification)"
echo "Goal: Identify SNR threshold where performance degrades below 90%"
echo ""
echo "Evaluation configuration:"
echo "  - H matrix: ${H_MATRIX}"
echo "  - USM (W):  ${USM_MODEL}"
echo "  - Freq band: [${FREQ_MIN}, ${FREQ_MAX}]Hz → F=346 bins @ 16kHz"
echo "  - Steps: ${STEPS}, Top-E: ${TOP_E}, L: ${L}"
echo "  - Iterations: ${ITERS}, Batch size: ${BATCH_SIZE}"
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
        # Use original baseline dataset (same as commit 1f6b68c)
        local dataset_root="${HOME}/LDV-data-processed/white_noise_box_data_no_edge_sync_vad_normalized"
        echo "  Using BASELINE data for SNR=Inf verification"
    else
        # Use synthetic SNR data for other levels
        local dataset_root="${DATA_BASE}/white_noise_box_snr${snr_label}_16k_sync_vad_normalized"
    fi

    # Output directory (same as training)
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

    # Run evaluation
    python "${WORKSPACE}/scripts/omp/evaluate_snr_accuracy.py" \
        --dataset_root "${dataset_root}" \
        --H_path "${H_MATRIX}" \
        --W_path "${USM_MODEL}" \
        --freq_min ${FREQ_MIN} \
        --freq_max ${FREQ_MAX} \
        --steps ${STEPS} \
        --top_e ${TOP_E} \
        --L ${L} \
        --iters ${ITERS} \
        --batch_size ${BATCH_SIZE} \
        --device ${DEVICE} \
        --routing_mode g \
        --out_dir "${out_dir}"

    local exit_code=$?

    if [ ${exit_code} -eq 0 ]; then
        echo "  ✓ Evaluation complete: ${out_dir}/accuracy_results.json"
        echo ""
    else
        echo "  ✗ Evaluation failed with exit code ${exit_code}"
        echo ""
        return ${exit_code}
    fi
}

# Main evaluation loop
echo "========================================================================"
echo "EVALUATION PHASE: White Noise SNR Sweep"
echo "========================================================================"
echo ""

# Evaluate baseline (SNR=∞) first
echo "--- BASELINE EVALUATION (SNR=∞) ---"
evaluate_snr_level "Inf"

echo "========================================================================"
echo "Baseline (SNR=∞) evaluation complete!"
echo ""
echo "CRITICAL CHECKPOINT:"
echo "  - Check results in: ${RESULTS_BASE}/snr_Inf/accuracy_results.json"
echo "  - Expected: Overall accuracy ~100%"
echo "  - Baseline (commit 1f6b68c): 100% accuracy on white noise"
echo ""
echo "If baseline matches expectations → Proceed with SNR sweep"
echo "If baseline DOES NOT match → Debug before continuing"
echo "========================================================================"
echo ""

# Evaluate SNR degradation sweep
echo ""
echo "--- SNR DEGRADATION SWEEP ---"
for snr in "${SNR_LEVELS[@]:1}"; do  # Skip first element (Inf, already evaluated)
    evaluate_snr_level "$snr"
done

echo "========================================================================"
echo "Phase 6 (White Noise) Evaluation Complete!"
echo "========================================================================"
echo ""
echo "Evaluation results saved to: ${RESULTS_BASE}/snr_*/accuracy_results.json"
echo ""
echo "Summary:"
for snr in "${SNR_LEVELS[@]}"; do
    result_file="${RESULTS_BASE}/snr_${snr}/accuracy_results.json"
    if [ -f "${result_file}" ]; then
        # Extract accuracy using Python
        accuracy=$(python3 -c "import json; print(f\"{json.load(open('${result_file}'))['overall_accuracy']*100:.2f}\")")
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
echo "4. Proceed to Phase 6 (Speech) if results look good"
echo ""
