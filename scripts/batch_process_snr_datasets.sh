#!/bin/bash
# Batch Process SNR Datasets - Phase 4: VAD + Normalization
# Processes all 68,117 files (white noise + speech) through Stage 1 (VAD) and Stage 2 (normalization)

set -e  # Exit on error

# Base directories
RAW_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/raw"
PROCESSED_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-48k"
SCRIPT_DIR="$HOME/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace/scripts"

# SNR levels to process
SNR_LEVELS=("Inf" "30dB" "20dB" "15dB" "10dB" "5dB" "0dB")

# VAD parameters (48 kHz)
VAD_THRESHOLD="1e-5"
VAD_METHOD="soft"
SAMPLE_RATE="48000"
N_FFT="2048"
HOP_LENGTH="512"
FREQ_MIN="300"
FREQ_MAX="3000"

echo "========================================================================"
echo "Phase 4: SNR Dataset Processing - VAD + Normalization"
echo "========================================================================"
echo "Total files to process: 68,117"
echo "  - White noise: 777 files (111 clips × 7 SNR)"
echo "  - Speech: 67,340 files (9,620 clips × 7 SNR)"
echo ""
echo "Processing stages:"
echo "  1. VAD @ 48kHz (synchronized X-Y)"
echo "  2. Normalization to [0, 1]"
echo "========================================================================"
echo ""

# Function to process VAD for one dataset
process_vad() {
    local dataset_type=$1  # "white_noise" or "speech260"
    local snr_level=$2     # "Inf", "30dB", etc.

    # Construct directory names
    if [ "$snr_level" = "Inf" ]; then
        local y_suffix="box_snrInf_data_no_edge"
        local y_output_suffix="box_snrInf_sync_vad"
    else
        local y_suffix="box_snr${snr_level}_data_no_edge"
        local y_output_suffix="box_snr${snr_level}_sync_vad"
    fi

    local x_input="${RAW_BASE}/${dataset_type}_original_data_no_edge"
    local y_input="${RAW_BASE}/${dataset_type}_${y_suffix}"
    local x_output="${PROCESSED_BASE}/${dataset_type}_original_sync_vad"
    local y_output="${PROCESSED_BASE}/${dataset_type}_${y_output_suffix}"

    echo "[VAD] Processing ${dataset_type} SNR=${snr_level}"
    echo "  X: ${x_input}"
    echo "  Y: ${y_input}"

    python "${SCRIPT_DIR}/apply_spectrogram_vad.py" \
        --x_input_dir "${x_input}" \
        --y_input_dir "${y_input}" \
        --x_output_dir "${x_output}" \
        --y_output_dir "${y_output}" \
        --vad_threshold ${VAD_THRESHOLD} \
        --vad_method ${VAD_METHOD} \
        --sample_rate ${SAMPLE_RATE} \
        --n_fft ${N_FFT} \
        --hop_length ${HOP_LENGTH} \
        --freq_min ${FREQ_MIN} \
        --freq_max ${FREQ_MAX}

    echo "  ✓ VAD complete"
    echo ""
}

# Function to process normalization for one dataset
process_normalization() {
    local dataset_type=$1  # "white_noise" or "speech260"
    local snr_level=$2     # "Inf", "30dB", etc.

    # Construct directory names
    if [ "$snr_level" = "Inf" ]; then
        local y_suffix="box_snrInf_sync_vad"
        local output_suffix="box_snrInf_sync_vad_normalized"
    else
        local y_suffix="box_snr${snr_level}_sync_vad"
        local output_suffix="box_snr${snr_level}_sync_vad_normalized"
    fi

    local input_dir="${PROCESSED_BASE}/${dataset_type}_${y_suffix}"
    local output_dir="${PROCESSED_BASE}/${dataset_type}_${output_suffix}"

    echo "[NORM] Processing ${dataset_type} SNR=${snr_level}"
    echo "  Input: ${input_dir}"

    python "${SCRIPT_DIR}/conversion/normalize_to_unit_range.py" \
        --in_dir "${input_dir}" \
        --out_dir "${output_dir}"

    echo "  ✓ Normalization complete"
    echo ""
}

# Main processing loop
echo "========================================================================"
echo "STAGE 1: VAD Processing (48 kHz)"
echo "========================================================================"
echo ""

# Process white noise datasets
echo "--- WHITE NOISE DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    process_vad "white_noise" "$snr"
done

# Process speech datasets
echo "--- SPEECH DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    process_vad "speech260" "$snr"
done

echo "========================================================================"
echo "STAGE 2: Normalization"
echo "========================================================================"
echo ""

# Process white noise datasets
echo "--- WHITE NOISE DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    process_normalization "white_noise" "$snr"
done

# Process speech datasets
echo "--- SPEECH DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    process_normalization "speech260" "$snr"
done

echo "========================================================================"
echo "Phase 4 Complete!"
echo "========================================================================"
echo ""
echo "Processed datasets (48 kHz):"
echo "  - White noise: 7 SNR levels × 2 stages = 14 operations"
echo "  - Speech: 7 SNR levels × 2 stages = 14 operations"
echo "  - Total operations: 28"
echo ""
echo "Next step: Phase 5 - Resampling (48 kHz → 16 kHz)"
echo ""
