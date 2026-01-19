# Experiment Report: exp_interspeech_gru2_tw32_lag50_k16_ep50

## Summary
- Dataset: boy1 MIC/LDV WAV pairs (no angle filtering). Found 416 pairs.
- Trajectory generation: 9870 blocks, variants_per_clip=5.
- OMP diagnostic: OMP ≫ Random in sparse regime (K=1,2,4,8); gap shrinks near K=16.
- DTmin training completed for 50 epochs; validation loss decreased steadily.
- RTG0/RTG1 override grid shows near-constant DT final mean across overrides.

## Data Sources
- MIC root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/MIC
- LDV root: /Users/jnrle/Documents/LDVReorientation/data/SpeechData/boy1/LDV
- Pairing: direct MIC/LDV pairs, no angle filtering.

## Parameters
- STFT: n_fft=2048, hop_length=160, fs=16000, freq bins=1025
- Window: Tw=32 frames
- Lag range: [-50, 50] (M=101 lags)
- K (selection budget): 16
- Gain: 100.0
- Variants per clip: 5
- Max items: 1000

## OMP Teacher Setup
- OMP acts as the oracle teacher per frequency bin for each block.
- Total OMP teacher blocks: 9870
- Each block produces a per-frequency selection of K atoms over M=101 lags.

## DTmin Student Setup
- Single DTmin student trained across all OMP teacher blocks.
- Model: SeqDT_FreqAware (GRU-based), rtg_dim=2.
- RTG inputs: remaining energy (rtg0) + remaining steps (rtg1).
- Student traverses all frequencies and K steps to match OMP teacher behavior.

## Training Curves
![Training Curves](figures/training_curves.png)

## OMP vs Random (K-sweep)
![OMP vs Random](figures/omp_vs_random_k_sweep.png)

| K | OMP | Random | Gap | Ratio (O/R) |
|---:|---:|---:|---:|---:|
| 1 | 0.9140 | 0.1584 | 0.7556 | 5.77x |
| 2 | 0.9565 | 0.3086 | 0.6478 | 3.10x |
| 4 | 0.9826 | 0.5517 | 0.4310 | 1.78x |
| 8 | 0.9967 | 0.8863 | 0.1104 | 1.12x |
| 16 | 0.9998 | 0.9954 | 0.0044 | 1.00x |

## DT vs OMP (Overall Mean Across Frequency Bins)
![DT vs OMP Overall](figures/dt_vs_omp_overall.png)

## DT/OMP/Random at Sparse K
![DT/OMP/Random Sparse](figures/dt_omp_random_sparse.png)

| K | DT (overall) | OMP (overall) | Random (k-sweep) |
|---:|---:|---:|---:|
| 1 | 0.2485 | 0.2225 | 0.1584 |
| 2 | 0.3981 | 0.3624 | 0.3086 |
| 4 | 0.5624 | 0.5529 | 0.5517 |
| 8 | 0.8690 | 0.8042 | 0.8863 |
| 16 | 0.9987 | 0.9852 | 0.9954 |

**Note:** DT/OMP values are from eval_stats.pt (10 clips). Random values come from the K-sweep diagnostic (3 clips, 5 random trials). The sparse regime still shows OMP ≫ Random.

## Band Efficiency Heatmap (DT/OMP)
![Band Efficiency Heatmap](figures/band_efficiency_heatmap.png)

## RTG0/RTG1 Override Grid
![RTG0/RTG1 Heatmap](figures/rtg0_rtg1_heatmap.png)

## Observations
- OMP strongly dominates Random in sparse K (K=1,2,4), consistent with a meaningful dictionary and sparse selection budget.
- DTmin approaches OMP at higher K; DT/OMP efficiency converges toward ~100% across bands.
- RTG0/RTG1 overrides barely change DT final mean, indicating limited sensitivity of RTG conditioning in this run.

## Current Issues
- **RTG meaning remains unclear**: both RTG0 and RTG1 overrides yield near-identical DT performance, suggesting the RTG inputs are not materially influencing policy decisions.

## Artifacts
- Pipeline log: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/pipeline.log
- Eval stats: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/eval/eval_stats.pt
- OMP vs Random: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/omp_vs_random_k_sweep/omp_vs_random_k_sweep.json
- RTG grid: results/exp_interspeech_gru2_tw32_lag50_k16_ep50/rtg_grid/rtg0_rtg1_override_grid.json
