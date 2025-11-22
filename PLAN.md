# Domain Randomization Agent V2 (angle-range variation)

Branch: exp/domain-randomization-AgentV2  
Codex session: 019aa1de-78c5-73e0-aa62-381459e57d2e

## Objective
Keep W/H fixed and M=8, K=6, normalized W/D, but create multiple datasets with different angle ranges. Evaluate AWR per shard and (optionally) balanced multi-shard to see if angle-range diversity helps without changing the dictionary.

## Fixed settings
- H: /Users/sbplab/LDV-data-processed/h_matrix_box_ldv_correct.pth
- W: doa_normalized_config_c_corrected/models/usm.pth
- K=6, M=8, normalize_W/D on, perturbation_sigma=0.0
- STFT: fs=16000, n_fft=2048, band 300–3000 Hz
- Reduction: kmeans (single method to keep W consistent)
- Seed: keep a single seed (e.g., 200) for W reduction across all shards to keep the dictionary identical.

## Angle ranges to generate (each shard: 37 angles × clips_per_angle=3 capped to the range)
- Full: 0–180 (every 5°) — baseline.
- Low: 0–60 (angles 0–12).
- Mid: 65–115 (angles 13–23).
- High: 120–180 (angles 24–36).
- Optional: Two overlapping ranges to test robustness (e.g., 30–90, 90–150).

## Plan
1) Generate 4–5 shards with the above angle ranges, same W/H, same reduced W (seed=200), normalize_W/D on, K=6, M=8.
2) Per-shard AWR with best single-shard recipe: β=0.5, w_max=4, normalize_weights=True, compute_voted_every=1.
3) Balanced multi-shard AWR (if needed): use only the angle-range shards, balanced sampling, try β in {0.3, 0.5, 0.7}, w_max {4,5}, both with/without normalize_weights; monitor per-angle metrics and eff_bs.
4) Select the best-performing shard(s) and report whether angle-range specialization helps vs full-range.

## Success criteria
- Per-shard AWR improves within its angle range (voted > 0.6) and does not collapse on its slice.
- If multi-shard is used, balanced training achieves voted > 0.6 without eff_bs explosion or flat weights.
