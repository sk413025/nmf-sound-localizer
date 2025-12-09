#!/bin/bash
# Batch Process SNR Datasets - White Noise Only
# Processes white noise files through Stage 1 (VAD) and Stage 2 (normalization)

set -e  # Exit on error

# Base directories
RAW_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/raw"
PROCESSED_BASE="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-48k"
SCRIPT_DIR="$HOME/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace/scripts"
ORIGINAL_DATA_ROOT="$HOME/LDV-data-processed"

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
echo "Phase 4: SNR Dataset Processing - VAD + Normalization (White Noise Only)"
echo "========================================================================"

# Function to process VAD for one dataset
process_vad() {
    local dataset_type=$1  # "white_noise"
    local snr_level=$2     # "Inf", "30dB", etc.

    # Construct directory names
    if [ "$snr_level" = "Inf" ]; then
        local y_suffix="box_snrInf_data_no_edge"
        local y_output_suffix="box_snrInf_sync_vad"
    else
        local y_suffix="box_snr${snr_level}_data_no_edge"
        local y_output_suffix="box_snr${snr_level}_sync_vad"
    fi

    # X input is always the clean original source
    local x_input="${ORIGINAL_DATA_ROOT}/${dataset_type}_original_data_no_edge"
    local y_input="${RAW_BASE}/${dataset_type}_${y_suffix}"
    
    # We don't really need X output for SNR experiments, but the script might produce it
    local x_output="${PROCESSED_BASE}/${dataset_type}_original_sync_vad" 
    local y_output="${PROCESSED_BASE}/${dataset_type}_${y_output_suffix}"

    echo "[VAD] Processing ${dataset_type} SNR=${snr_level}"
    echo "  X Input: ${x_input}"
    echo "  Y Input: ${y_input}"
    echo "  Output:  ${y_output}"

    if [ ! -d "${y_input}" ]; then
        echo "  ✗ ERROR: Input directory not found: ${y_input}"
        return 1
    fi

    # Run VAD script
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

# Function to process Normalization
process_normalization() {
    local dataset_type=$1  # "white_noise"
    local snr_level=$2     # "Inf", "30dB", etc.

    # Construct directory names
    if [ "$snr_level" = "Inf" ]; then
        local input_suffix="box_snrInf_sync_vad"
        local output_suffix="box_snrInf_sync_vad_normalized"
    else
        local input_suffix="box_snr${snr_level}_sync_vad"
        local output_suffix="box_snr${snr_level}_sync_vad_normalized"
    fi

    local input_dir="${PROCESSED_BASE}/${dataset_type}_${input_suffix}"
    local output_dir="${PROCESSED_BASE}/${dataset_type}_${output_suffix}"

    echo "[NORM] Processing ${dataset_type} SNR=${snr_level}"
    echo "  Input:  ${input_dir}"
    echo "  Output: ${output_dir}"

    if [ ! -d "${input_dir}" ]; then
        echo "  ✗ ERROR: Input directory not found: ${input_dir}"
        return 1
    fi

    # Run Normalization script
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

echo "========================================================================"
echo "STAGE 2: Normalization"
echo "========================================================================"
echo ""

# Process white noise datasets
echo "--- WHITE NOISE DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    process_normalization "white_noise" "$snr"
done

echo "========================================================================"
echo "Processing Complete!"
echo "========================================================================"
