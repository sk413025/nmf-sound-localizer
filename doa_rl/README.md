DoA-RL (Mirror Descent View)
================================

Implements the code structure described in docs/code.md:

- Features: tokenizers (Patch/LEAF/Scattering/NMF-atom) and NMF utils.
- Env: STFT utils, dataset builder, DoA math (advantage ★), H loaders.
- Algos: PPO/GRPO runners over a Transformer policy (no model architecture change).
- Scripts: prepare_hrtf, prepare_dict, train_single, train_multi, infer_demo.
- Eval: metrics and plotting helpers.

The implementation reuses measured H and USM W assets, and computes the advantage
(★) exactly in IS geometry: A_d = Σ_f (H_d ⊙ s_hat)_f (Y_f/Ŷ_f^2 − 1/Ŷ_f).

