#!/bin/bash
set -euo pipefail

# Define paths
CONDA_BIN="/Users/jiawei/miniconda3/bin/conda"
WORKTREE_ROOT="/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-gteacher-20251214"
H_PATH="/Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth"
W_PATH="${WORKTREE_ROOT}/doa_speech260_config_c_16k_smoke_mps_20251114_184322/models/usm.pth"
DATASET_ROOT="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized"

cd "$WORKTREE_ROOT"

echo "Starting sequential runs..."

# Sequence 1: g-teacher F (Transformer OFF / Identity Encoder)
# Corresponds to "gteacherF_seq" in user prompt
for seed in 42 1 2 3 4; do
    RUN_DIR="results/ablate_g_teacher_F_seed${seed}_$(date +%Y%m%d_%H%M%S)"
    echo "----------------------------------------------------------------"
    echo "[g-teacher F] Starting seed ${seed} -> ${RUN_DIR}"
    echo "----------------------------------------------------------------"
    mkdir -p "$RUN_DIR"
    
    PYTHONUNBUFFERED=1 PYTHONPATH="$WORKTREE_ROOT" "$CONDA_BIN" run -n trl-training \
      python -u scripts/omp-transformer-ldv.py \
      --h_path "$H_PATH" \
      --w_path "$W_PATH" \
      --dataset_root "$DATASET_ROOT" \
      --n_atoms 8 --atom_reduce_mode kmeans --atom_min_cos 0.98 \
      --d_model 346 --nhead 2 --nlayers 1 --steps 2 --top_e 2 --top_l 2 \
      --epochs 20 --batch_size 16 --lr 0.001 --seed ${seed} \
      --alpha 1.0 --beta 0.2 --gamma 0.5 \
      --routing_mode g --routing_activation_e gumbel --routing_activation_a gumbel \
      --score_norm none --score_reg_weight 0.0 --expert_agg l2 \
      --device mps --encoder_identity \
      --out_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
      
    echo "[g-teacher F] Finished seed ${seed}"
done

# Sequence 2: g-fixed F (Transformer ON)
# Corresponds to "gfixedF_seq" in user prompt
for seed in 42 1 2 3 4; do
    RUN_DIR="results/ablate_g_fixed_F_seed${seed}_$(date +%Y%m%d_%H%M%S)"
    echo "----------------------------------------------------------------"
    echo "[g-fixed F] Starting seed ${seed} -> ${RUN_DIR}"
    echo "----------------------------------------------------------------"
    mkdir -p "$RUN_DIR"
    
    PYTHONUNBUFFERED=1 PYTHONPATH="$WORKTREE_ROOT" "$CONDA_BIN" run -n trl-training \
      python -u scripts/omp-transformer-ldv.py \
      --h_path "$H_PATH" \
      --w_path "$W_PATH" \
      --dataset_root "$DATASET_ROOT" \
      --n_atoms 8 --atom_reduce_mode kmeans --atom_min_cos 0.98 \
      --d_model 346 --nhead 2 --nlayers 1 --steps 2 --top_e 2 --top_l 2 \
      --epochs 20 --batch_size 16 --lr 0.001 --seed ${seed} \
      --alpha 1.0 --beta 0.2 --gamma 0.5 \
      --routing_mode g --routing_activation_e gumbel --routing_activation_a gumbel \
      --score_norm none --score_reg_weight 0.0 --expert_agg l2 \
      --device mps \
      --out_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
      
    echo "[g-fixed F] Finished seed ${seed}"
done

echo "All sequential runs completed."
