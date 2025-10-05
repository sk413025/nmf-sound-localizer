#!/usr/bin/env python3
"""TRL PPO with a Reward Model (RM) for K>1 episodes, without modifying TRL.

Assumes a trained RM checkpoint containing v_head weights. RM predicts a scalar
reward for (prompt + K-token completion). PPOTrainer uses RM as reward_model.
"""

from __future__ import annotations

import argparse
from typing import List

from datasets import Dataset
from transformers import set_seed
from trl import PPOConfig, PPOTrainer
from trl.models import create_reference_model

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids
from doa_rl.hf.logits_mask import NoRepeatDirectionLogitsProcessor
import torch


def _discover_angles(root: str) -> List[int]:
    from pathlib import Path
    base = Path(root)
    angles = sorted({int(p.name.split("_")[1]) for p in base.glob("angle_*") if p.is_dir() and p.name.split("_")[1].isdigit()})
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _prepare_prompts(args, direction_angles: List[int]) -> List[str]:
    ds = DoADataset(args.data_root, direction_angles, fs=args.sample_rate, n_fft=args.n_fft, freq_min=args.freq_min, freq_max=args.freq_max)
    dl = create_dataloader(ds, batch_size=1, shuffle=False)
    tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
    prompts: List[str] = []
    for batch in dl:
        Y_np = batch["Y"].squeeze(0).numpy()
        prompts.append(" ".join(tok(Y_np)))
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts


def main():
    ap = argparse.ArgumentParser(description="TRL PPO with RM for K>1 option A")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--rm-ckpt", type=str, required=True, help="Path to RM checkpoint (v_head state dict)")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)

    # Angles and prompts
    direction_angles = _discover_angles(args.data_root)
    prompts = _prepare_prompts(args, direction_angles)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Policy and reference
    policy, _ = build_value_head_model(tokenizer)
    reference = create_reference_model(policy)

    # Reward model (RM): same architecture, load v_head
    rm_model, _ = build_value_head_model(tokenizer)
    for p in rm_model.pretrained_model.parameters():
        p.requires_grad = False
    state = torch.load(args.rm_ckpt, map_location="cpu")
    rm_model.v_head.load_state_dict(state["v_head"])  # restore value head
    rm_model.score = rm_model.v_head  # expose score for TRL

    # Dataset for PPO: pre-encode prompts to input_ids/attention_mask to satisfy collator
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    ds = Dataset.from_dict(
        {
            "input_ids": [ids.tolist() for ids in enc["input_ids"]],
            "attention_mask": [m.tolist() for m in enc["attention_mask"]],
        }
    )

    # PPO config
    bs = max(1, min(args.batch_size, len(ds)))
    cfg = PPOConfig(
        output_dir="trl-output-ppo-rm",
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        learning_rate=args.lr,
        use_cpu=True,
        fp16=False,
        bf16=False,
    )
    cfg.batch_size = bs
    cfg.mini_batch_size = bs
    cfg.num_ppo_epochs = max(1, args.ppo_epochs)
    cfg.response_length = args.K
    cfg.temperature = 1.0
    cfg.num_total_batches = max(1, args.epochs)
    cfg.total_episodes = cfg.num_total_batches * bs
    cfg.whiten_rewards = False
    cfg.save_strategy = "no"
    cfg.logging_strategy = "steps"
    cfg.logging_steps = 1
    cfg.eval_strategy = "no"

    trainer = PPOTrainer(
        cfg,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=rm_model,
        train_dataset=ds,
        value_model=policy,  # share value head with policy for PPO baseline
        eval_dataset=ds,
    )

    # Enforce no-repeat selections
    processor = NoRepeatDirectionLogitsProcessor(direction_token_ids(tokenizer))
    def _wrap_generate(module):
        orig = module.generate
        def gen_with_mask(*args, **kwargs):
            procs = list(kwargs.get("logits_processor", []))
            procs.append(processor)
            kwargs["logits_processor"] = procs
            return orig(*args, **kwargs)
        module.generate = gen_with_mask
    _wrap_generate(trainer.policy_model.pretrained_model)
    _wrap_generate(trainer.model.policy)

    trainer.train()


if __name__ == "__main__":
    main()
