#!/bin/bash
# Batch Resample SNR Datasets - Phase 5: 48kHz → 16kHz
# Resamples all 68,117 normalized files from 48kHz to 16kHz for training

set -e  # Exit on error

# Base directories
PROCESSED_48K="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-48k"
PROCESSED_16K="$HOME/LDV-data-experiments/snr-synthetic-2025-12/processed-16k"
SCRIPT_DIR="$HOME/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace/scripts"

# SNR levels to process
SNR_LEVELS=("Inf" "30dB" "20dB" "15dB" "10dB" "5dB" "0dB")

echo "========================================================================"
echo "Phase 5: SNR Dataset Resampling - 48kHz → 16kHz"
echo "========================================================================"
echo "Total files to resample: 68,117"
echo "  - White noise: 777 files (111 clips × 7 SNR)"
echo "  - Speech: 67,340 files (9,620 clips × 7 SNR)"
echo ""
echo "Resampling: 48,000 Hz → 16,000 Hz (down by 3×)"
echo "========================================================================"
echo ""

# Function to resample one dataset
resample_dataset() {
    local dataset_type=$1  # "white_noise" or "speech260"
    local snr_level=$2     # "Inf", "30dB", etc.

    # Construct directory names
    if [ "$snr_level" = "Inf" ]; then
        local input_suffix="box_snrInf_sync_vad_normalized"
        local output_suffix="box_snrInf_16k_sync_vad_normalized"
    else
        local input_suffix="box_snr${snr_level}_sync_vad_normalized"
        local output_suffix="box_snr${snr_level}_16k_sync_vad_normalized"
    fi

    local input_dir="${PROCESSED_48K}/${dataset_type}_${input_suffix}"
    local output_dir="${PROCESSED_16K}/${dataset_type}_${output_suffix}"

    echo "[RESAMPLE] Processing ${dataset_type} SNR=${snr_level}"
    echo "  Input:  ${input_dir}"
    echo "  Output: ${output_dir}"

    python "${SCRIPT_DIR}/conversion/resample_to_16k.py" \
        --in_dir "${input_dir}" \
        --out_dir "${output_dir}" \
        --src_sr 48000 \
        --dst_sr 16000

    echo "  ✓ Resampling complete"
    echo ""
}

# Main processing loop
echo "========================================================================"
echo "STAGE 4.5: Resampling (48 kHz → 16 kHz)"
echo "========================================================================"
echo ""

# Process white noise datasets
echo "--- WHITE NOISE DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    resample_dataset "white_noise" "$snr"
done

# Process speech datasets
echo "--- SPEECH DATASETS ---"
for snr in "${SNR_LEVELS[@]}"; do
    resample_dataset "speech260" "$snr"
done

echo "========================================================================"
echo "Phase 5 Complete!"
echo "========================================================================"
echo ""
echo "Resampled datasets (16 kHz):"
echo "  - White noise: 7 SNR levels × 111 clips = 777 files"
echo "  - Speech: 7 SNR levels × 9,620 clips = 67,340 files"
echo "  - Total files: 68,117"
echo ""
echo "Next step: Phase 6 - Training"
echo ""
