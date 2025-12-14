#!/bin/bash

# Session name
SESSION="snr-5fold"

# SNRs to sweep (Only Inf available in default path)
SNRS=("Inf")

# Base paths
DATA_BASE="/Users/sbplab/jiawei/datasets/test_nmf_output_no_edge_with_original"
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="../atom-budget-sweep/doa_normalized_config_c_corrected/models/usm.pth"

# Create new session
tmux new-session -d -s $SESSION -n "fold0"

# Loop through folds 0-2 (Dataset has only 3 clips, so only 3 folds are possible)
for fold in {0..2}; do
    if [ $fold -ne 0 ]; then
        tmux new-window -t $SESSION:$fold -n "fold$fold"
    fi
    
    # Construct the command for this fold
    # Initialize conda correctly without relying on ~/.zshrc
    CMD="eval \"\$(/opt/anaconda3/bin/conda shell.zsh hook)\" && conda activate wavtokenizer && export PYTHONPATH=$(pwd):\$PYTHONPATH && "
    
    for snr in "${SNRS[@]}"; do
        if [ "$snr" == "Inf" ]; then
            DATASET="${DATA_BASE}/white_noise_box_data_no_edge_sync_vad_normalized"
        else
            DATASET="${DATA_BASE}/white_noise_box_snr${snr}_sync_vad_normalized"
        fi
        
        OUT_DIR="results/qk_snr${snr}_30epochs_fold${fold}_$(date +%Y%m%d)"
        
        CMD+="echo 'Running Fold $fold SNR $snr...' && "
        CMD+="python -u scripts/omp-transformer-ldv.py "
        CMD+="--routing_mode qk --device cpu "
        CMD+="--dataset_root $DATASET "
        CMD+="--h_path $H_PATH "
        CMD+="--w_path $W_PATH "
        CMD+="--epochs 30 --batch_size 16 --d_model 128 --nhead 2 --nlayers 1 "
        CMD+="--no_type_bias --score_center_atoms --score_center_expert --score_norm std "
        CMD+="--fold $fold "
        CMD+="--out_dir $OUT_DIR && "
    done
    
    CMD+="echo 'Fold $fold completed.'"
    
    # Send command to tmux window
    tmux send-keys -t $SESSION:$fold "$CMD" C-m
done

echo "Tmux session '$SESSION' created. Attach with: tmux attach -t $SESSION"
