# Important Artifacts and Paths

This document records the absolute paths and associated git commits for key experimental results and visualizations.

## 1. Baseline vs Transformer Heatmap
**Description**: Heatmaps comparing the confusion matrices of the Baseline model (Transformer-Routed Soft-OMP) and the No-Transformer ablation.
**Absolute Path**: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablation_analysis_20251219`
**Git Commit**: `88a89405c942c873d6b6ae53256fe0162eb03bb9` (in `exp-omp-ablation-20251209` worktree)

## 2. Modal Data Visualization
**Description**: Visualization of modal data (likely related to feature extraction or OMP transformer modal analysis).
**Absolute Path**: `/Users/jnrle/Documents/LDVReorientation/worktrees/feat-omp-transformer-modal-viz/results/modal_viz_repro_e3d8462_3955a18`
**Git Commit**: `15b2981d660d97c8925bcfe97f1e57fbbcd6da54` (in `feat-omp-transformer-modal-viz` worktree)

## 3. Ablation Experiments & Documentation
**Description**: Documentation and run notes for the ablation study, including paths to specific experiment runs.

**Key Documentation Files**:
- `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/PLAN_BASELINE_COMPARISON.md`
- `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/README_run_notes.md`

**Git Commit**: `88a89405c942c873d6b6ae53256fe0162eb03bb9` (in `exp-omp-ablation-20251209` worktree)

**Experiment Run Paths (from README_run_notes.md)**:

*   **Baseline (Speech260 val-split)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_20251202_192153`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_seed1_20251203_105731`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_seed2_20251203_105739`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_seed3_20251203_202610`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_seed4_20251203_202619`
    *   Seed 5: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/omp_transformer_speech260_trainval_split_full_seed5_20251203_202628`

*   **No Transformer (`--encoder_identity`)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed42_20251210_203029`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed1_20251211_113851`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed2_20251211_185200`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed3_20251212_161859`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_identity_speech260_seed4_20251213_043329`

*   **Fixed Heuristic Routing (`--routing_mode g`)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_routing_speech260_seed42_20251210_203034`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_routing_speech260_seed1_20251211_122649`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_routing_speech260_seed2_20251211_210227`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_routing_speech260_seed3_20251212_164010`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_routing_speech260_seed4_20251213_081907`

*   **No Type Bias (`--no_type_bias`)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_no_type_bias_speech260_seed42_20251210_203042`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_no_type_bias_speech260_seed1_20251211_124027`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_no_type_bias_speech260_seed2_20251212_025820`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_no_type_bias_speech260_seed3_20251212_173046`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_no_type_bias_speech260_seed4_20251213_142618`

*   **Dense Routing (`--disable_omp_sparsity`)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_disable_omp_sparsity_speech260_seed42_20251210_203049`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_disable_omp_sparsity_speech260_seed1_20251211_022038`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_disable_omp_sparsity_speech260_seed2_20251211_032717`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_disable_omp_sparsity_speech260_seed3_20251211_125643`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_disable_omp_sparsity_speech260_seed4_20251211_171758`

*   **G-Teacher F (Identity + G-Routing)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_teacher_F_seed42_20251214_104517`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_teacher_F_seed1_20251214_233904`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_teacher_F_seed2_20251215_104733`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_teacher_F_seed3_20251215_164354`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_teacher_F_seed4_20251216_041222`

*   **G-Fixed F (Transformer + G-Routing)**:
    *   Seed 42: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_fixed_F_seed42_20251214_104517`
    *   Seed 1: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_fixed_F_seed1_20251214_235111`
    *   Seed 2: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_fixed_F_seed2_20251215_121914`
    *   Seed 3: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_fixed_F_seed3_20251215_170749`
    *   Seed 4: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-omp-ablation-20251209/results/ablate_g_fixed_F_seed4_20251216_070534`

## 4. SNR Robustness (5-Fold CV & Noise Levels)
**Description**: Evaluation of model performance across varying Signal-to-Noise Ratios (0dB, 5dB, 10dB, 15dB, 20dB, 30dB, Inf) using 5-fold cross-validation.
**Absolute Path**: `/Users/jnrle/Documents/LDVReorientation/worktrees/exp-snr-5fold-repro/results`
**Git Commit**: `f687aa085318309296205304dfcc39eff346093f` (in `exp-snr-5fold-repro` worktree)

**Experiment Run Paths**:
*   **0dB**: results/qk_snr0dB_30epochs_fold*_20251214
*   **5dB**: results/qk_snr5dB_30epochs_fold*_20251214
*   **10dB**: results/qk_snr10dB_30epochs_fold*_20251214
*   **15dB**: results/qk_snr15dB_30epochs_fold*_20251214
*   **20dB**: results/qk_snr20dB_30epochs_fold*_20251214
*   **30dB**: results/qk_snr30dB_30epochs_fold*_20251214
*   **Inf dB (Clean)**: results/qk_snrInf_30epochs_fold*_20251214
