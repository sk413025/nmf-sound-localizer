# Worktree: H Exploration (Speech)

This worktree explores the estimation of the H matrix (Transfer Function / Beamforming weights) using Complex Speech data.

## Goal
Use OMP to find optimal weights mapping Microphone Data -> LDV Data, and distill this into a Decision Transformer (DTmin).

## Experiment
- **Dataset**: `speech260` (16kHz). Original Microphone + Box LDV.
- **Method**: 
    1. Generate OMP trajectories (Mic x H = LDV).
    2. Train DTmin to predict H (channel selection) from Mic observations.

## Results
- Run `exp_h_full`: 336,700 trajectories.
- DTmin Loss: ~0.0 (Perfect convergence). 
- **Critical Finding**: The loaded "Original" Speech data appears to be **Mono** (1 Channel). 
    - OMP only had 1 choice (Index 0).
    - DTmin effectively learned a constant function.
    - To perform meaningful Channel Selection / Beamforming, ensure Multi-channel data is used data.

## Scripts
- `scripts/h_exploration/dataset.py`: Paired Dataset.
- `scripts/h_exploration/generate_omp_trajectories.py`: OMP Teacher.
- `scripts/h_exploration/train_dtmin_h.py`: Student.
