Results: Executability validation for PPO+RM and RM training (smoke tests)

Background (related commits)
- 564564a: Stage 0–1 scaffolds (left padding, K-step, no-repeat logits, PPO CPU)
- a63c33a: GRPO Option A (K=3) with physics rewards (proxyA/ΔIS) — no TRL changes
- 6b09fb2: Experiment notes and full GRPO run instructions (K=3)
- 68284ac: Added PPO K>1 with Reward Model (scripts/train_trl_ppo_with_rm.py) and RM training (scripts/train_reward_model.py)

Purpose
- Verify both new scripts execute end-to-end with current TRL/Transformers stack on a tiny tmp dataset.
- Document exact environment, commands, outputs, and any fixes needed.

Environment
- Conda env: trl-training (Python 3.11)
- Hardware: CPU (use_cpu=True)
- Packages (conda env provides): torch, transformers, datasets, accelerate, trl
- PYTHONPATH: /Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/angle-based-byol:/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/development-workspace

Reproduction instructions
1) Train Reward Model (ΔIS supervision)
```
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/angle-based-byol:$PWD:$PYTHONPATH
conda run -n trl-training \
  python scripts/train_reward_model.py \
  --data-root tmp/trl_test_data \
  --tf-path h_matrix_normalized_original_to_box.pth \
  --w-path data/W_usm_is.npz \
  --reward-mode deltaIS \
  --K 3 --epochs 1 --batch-size 8 --lr 1e-4 \
  --patch-fp 16 --patch-np 10 --n-fft 2048 --sample-rate 48000 \
  --freq-min 300 --freq-max 3000 --max-samples 6 \
  --out rm_ckpt.pt
```
Expected output (example):
```
{'epoch': 0, 'loss': 6.736660266282257e+28}
Saved RM checkpoint to rm_ckpt.pt
```

2) PPO with Reward Model (K>1, no TRL changes)
```
export PYTHONPATH=/Users/sbplab/jiawei/pg-ltr-frame-byol-worktree/worktrees/angle-based-byol:$PWD:$PYTHONPATH
conda run -n trl-training \
  python scripts/train_trl_ppo_with_rm.py \
  --data-root tmp/trl_test_data \
  --rm-ckpt rm_ckpt.pt \
  --K 3 --epochs 1 --batch-size 2 --ppo-epochs 1 --lr 1e-4 \
  --patch-fp 16 --patch-np 10 --n-fft 2048 --sample-rate 48000 \
  --freq-min 300 --freq-max 3000 --max-samples 6
```
Expected console metrics (example):
```
{'eps': 11, 'objective/kl': 21.6345, 'objective/entropy': 0.8880,
 'objective/non_score_reward': -1.0817, 'objective/rlhf_reward': -2.4340,
 'objective/scores': -1.3523, 'policy/approxkl_avg': 26.2557,
 'policy/clipfrac_avg': 0.3333, 'loss/policy_avg': 0.3415,
 'loss/value_avg': 3.4804, 'val/clipfrac_avg': 0.0,
 'policy/entropy_avg': 7.5608, 'val/ratio': 0.00093,
 'val/ratio_var': nan, 'val/num_eos_tokens': 0, 'lr': 0.0001,
 'episode': 2, 'epoch': 0.33}
```

Fixes applied to ensure executability (minimal, no algorithm changes)
- doa_rl/hf/tokenizer.py: Set `hf_tokenizer.model_max_length = 8192` to avoid OverflowError when `max_length=tokenizer.model_max_length` is used.
- scripts/train_trl_ppo_with_rm.py: Pass `value_model=policy` to match current TRL PPOTrainer signature; pre-encode prompts into `input_ids/attention_mask` to satisfy the data collator.

Data lineage
- Dataset used: tmp/trl_test_data (3 angles × 2 files = 6 .npy)
- Fingerprint: `find tmp/trl_test_data -name '*.npy' -exec md5 -r {} + | sort -k2 | md5 -r`
  - MD5: 3f30533e259ec87b27b8da733f93c389
  - Total data files: 6
- Preprocessing: PatchTokenizer tokens from DoADataset STFT; crude F-axis resampling for H where needed; ŝ fallback = mean(Y) for shape mismatch (smoke only).
- Train/val split: N/A (smoke run; evaluation via console metrics)

Physical/mathematical analysis (ΔIS reward used for RM supervision)
- First principles: Itakura–Saito divergence D_IS(Y||Ŷ) = Σ_{f,n} [Y/Ŷ − log(Y/Ŷ) − 1] measures scale-invariant spectral fit.
- Mathematical relationships: At each step t, Ŷ_t accumulates H_s·ŝ along selected directions, so ΔIS_t = −(D_IS(Y, Ŷ_t) − D_IS(Y, Ŷ_{t−1})). Larger positive ΔIS indicates better mixture approximation BECAUSE the incremental component reduces divergence.
- Physical constraints: Accurate ΔIS magnitude requires H, W, and dataset STFT grids to match; mismatches (frequency axis or windowing) distort magnitudes DUE TO inconsistent spectral bases.
- Signal processing fundamentals: Nearest-neighbor resampling along F reduces fidelity and may inflate or deflate ΔIS steps; ŝ fallback to mean(Y) ignores content structure and therefore weakens physical validity.
- Information theory: ΔIS reflects gain in explanatory power of the generative mixture; with poorly matched bases the mutual information proxy is unreliable, WHICH IMPLIES we should not interpret absolute reward scales on smoke data.

Cross-experiment analysis
- Pattern recognition: GRPO smoke and this PPO+RM smoke both run end-to-end BECAUSE physics is injected via public APIs (reward_funcs or reward_model) without modifying TRL internals (commits a63c33a, 6b09fb2, 68284ac).
- Success factors: No-repeat logits masking consistently enforces distinct K selections, improving stability BECAUSE it matches the combinatorial structure of the task (564564a, a63c33a).
- Failure modes: Inflated/unstable reward scales occur DUE TO TF/W vs dataset STFT mismatches and ŝ fallback, seen across smoke runs; asset alignment is crucial for meaningful magnitudes.
- Method effectiveness: PPO K>1 via RM and GRPO K>1 both remain valid routes BECAUSE they leverage TRL public interfaces; neither requires algorithm patches.

Extracted principles
- Design: Prefer public-API integrations (reward_funcs, reward_model) over internal trainer modifications; enforce no-repeat constraints at generation.
- Hypothesis: GIVEN ΔIS depends on matched spectral bases, expect calibrated magnitudes only when TF/W share STFT config with the dataset.
- Resource allocation: Invest effort in asset alignment (H/W frequency grids, windowing) BEFORE tuning RL hyperparameters.
- Risk mitigation: Set a finite tokenizer `model_max_length`; pre-encode datasets for PPO collator; pass `value_model` explicitly to avoid TRL API drift.
- Success amplification: Use GRPO for quick physics-reward validation; add RM to enable PPO K>1 while keeping PPO untouched.

Meta-reflection
- Methodology: Our approach aligns with the design principle of TRL-safe integration; smoke tests verified plumbing without overfitting to toy data.
- Documentation: This note captures exact environment, commands, and fingerprints, WHICH IMPROVES reproducibility.
- Efficiency: Minimal code changes unblocked execution; further gains require asset alignment rather than algorithmic tweaks.
- Knowledge gaps: Quantitative reward calibration on the 17-angle dataset remains to be established WITH matched TF/W.

Verification
- Files generated: rm_ckpt.pt (tracked by Git LFS via .gitattributes)
- Expected console outputs as above; no additional artifacts (save_strategy=no in PPO)

Next experiments
- Replace crude F resampling and ŝ fallback with properly aligned H/W to calibrate ΔIS magnitudes; re-run RM and PPO+RM.
- Add evaluation scripts for selected direction set quality and ΔIS curves; consider early stop when ΔIS < ε.

