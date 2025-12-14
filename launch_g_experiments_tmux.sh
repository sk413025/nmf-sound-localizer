#!/bin/bash
set -e

# Define paths
WORKTREE_ROOT="/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-gteacher-20251214"
CONDA_BIN="/Users/jiawei/miniconda3/bin/conda"
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="${WORKTREE_ROOT}/doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized"

cd "$WORKTREE_ROOT"

# Function to start a session and run a command loop
start_session() {
    local SESSION_NAME=$1
    local CMD_LOOP=$2
    
    # Check if session exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists. Skipping creation."
    else
        echo "Creating session $SESSION_NAME..."
        # Start detached session with a shell
        tmux new-session -d -s "$SESSION_NAME" -c "$WORKTREE_ROOT"
        
        # Send setup commands
        tmux send-keys -t "$SESSION_NAME" "export PYTHONPATH=$WORKTREE_ROOT" C-m
        tmux send-keys -t "$SESSION_NAME" "export CONDA_BIN=$CONDA_BIN" C-m
        
        # Send the main loop command
        # We use a slightly complex send-keys to ensure the loop runs correctly
        tmux send-keys -t "$SESSION_NAME" "$CMD_LOOP" C-m
    fi
}

# Command for g-teacher F (Identity Encoder)
CMD_G_TEACHER="
for seed in 42 1 2 3 4; do
    RUN_DIR=results/ablate_g_teacher_F_seed\${seed}_\$(date +%Y%m%d_%H%M%S)
    echo \"[g-teacher F] seed \${seed} -> \${RUN_DIR}\"
    mkdir -p \"\$RUN_DIR\"
    PYTHONUNBUFFERED=1 PYTHONPATH=$WORKTREE_ROOT \"\$CONDA_BIN\" run -n trl-training \\
      python -u scripts/omp-transformer-ldv.py \\
      --h_path $H_PATH \\
      --w_path $W_PATH \\
      --dataset_root $DATASET_ROOT \\
      --n_atoms 8 --atom_reduce_mode kmeans --atom_min_cos 0.98 \\
      --d_model 346 --nhead 2 --nlayers 1 --steps 2 --top_e 2 --top_l 2 \\
      --epochs 20 --batch_size 16 --lr 0.001 --seed \${seed} \\
      --alpha 1.0 --beta 0.2 --gamma 0.5 \\
      --routing_mode g --routing_activation_e gumbel --routing_activation_a gumbel \\
      --score_norm none --score_reg_weight 0.0 --expert_agg l2 \\
      --device mps --encoder_identity \\
      --out_dir \"\$RUN_DIR\" 2>&1 | tee \"\$RUN_DIR/run.log\"
done
"

# Command for g-fixed F (Transformer ON)
CMD_G_FIXED="
for seed in 42 1 2 3 4; do
    RUN_DIR=results/ablate_g_fixed_F_seed\${seed}_\$(date +%Y%m%d_%H%M%S)
    echo \"[g-fixed F] seed \${seed} -> \${RUN_DIR}\"
    mkdir -p \"\$RUN_DIR\"
    PYTHONUNBUFFERED=1 PYTHONPATH=$WORKTREE_ROOT \"\$CONDA_BIN\" run -n trl-training \\
      python -u scripts/omp-transformer-ldv.py \\
      --h_path $H_PATH \\
      --w_path $W_PATH \\
      --dataset_root $DATASET_ROOT \\
      --n_atoms 8 --atom_reduce_mode kmeans --atom_min_cos 0.98 \\
      --d_model 346 --nhead 2 --nlayers 1 --steps 2 --top_e 2 --top_l 2 \\
      --epochs 20 --batch_size 16 --lr 0.001 --seed \${seed} \\
      --alpha 1.0 --beta 0.2 --gamma 0.5 \\
      --routing_mode g --routing_activation_e gumbel --routing_activation_a gumbel \\
      --score_norm none --score_reg_weight 0.0 --expert_agg l2 \\
      --device mps \\
      --out_dir \"\$RUN_DIR\" 2>&1 | tee \"\$RUN_DIR/run.log\"
done
"

# Start the sessions
start_session "gteacherF_seq" "$CMD_G_TEACHER"
start_session "gfixedF_seq" "$CMD_G_FIXED"

echo "Done. Use 'tmux attach -t gteacherF_seq' or 'tmux attach -t gfixedF_seq' to view progress."
