#!/bin/bash
cd /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/mdp-decision-transformer
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/mdp-decision-transformer

for i in {1..48}; do
  echo "=== Running epoch chunk $i/48 (240 total epochs) ==="
  python -u scripts/dt_pointer_ldv.py \
    --traj_dir results/dt_traj_classic_omp_g_M50 \
    --out_dir results/dt_min_classic_omp_g_M50 \
    --epochs 5 --batch_size 8 --d_model 128 --nhead 2 --nlayers 1 \
    --distill_weight 0.7 --distill_T 1.0 --warmup_epochs 3 --device cpu \
    2>&1 | tee -a results/dt_min_classic_omp_g_M50/run.log

  if [ $? -ne 0 ]; then
    echo "Training failed at chunk $i"
    break
  fi
done
