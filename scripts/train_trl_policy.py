#!/usr/bin/env python3
"""Minimal TRL PPO training over patch tokens using a Hugging Face causal LM."""

from __future__ import annotations

import argparse
from typing import Dict, List

from datasets import Dataset
from transformers import set_seed
from trl import PPOConfig, PPOTrainer
from trl.models import create_reference_model

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model


def _discover_angles(root: str) -> List[int]:
    from pathlib import Path

    base = Path(root)
    angles = sorted(
        {
            int(dir_path.name.split("_")[1])
            for dir_path in base.glob("angle_*")
            if dir_path.is_dir() and dir_path.name.split("_")[1].isdigit()
        }
    )
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _prepare_samples(args) -> tuple[List[Dict], List[int]]:
    direction_angles = _discover_angles(args.data_root)
    dataset = DoADataset(
        args.data_root,
        direction_angles,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dataloader = create_dataloader(dataset, batch_size=1, shuffle=False)
    patch_tokenizer = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)

    records: List[Dict] = []
    for batch in dataloader:
        Y = batch["Y"].squeeze(0).numpy()
        tokens = patch_tokenizer(Y)
        prompt = " ".join(tokens)
        target = int(batch["angle_index"].item())
        records.append({
            "prompt": prompt,
            "angle_idx": target,
            "path": batch["path"][0],
        })
        if args.max_samples and len(records) >= args.max_samples:
            break
    return records, direction_angles


def _tokenize_prompts(samples: List[Dict], tokenizer, max_len: int) -> Dataset:
    encoded = tokenizer(
        [s["prompt"] for s in samples],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    ds = Dataset.from_dict(
        {
            "input_ids": [row.tolist() for row in encoded.input_ids],
            "attention_mask": [row.tolist() for row in encoded.attention_mask],
        }
    )
    return ds


def _build_trainer(args, tokenizer, train_dataset: Dataset) -> PPOTrainer:
    policy, _ = build_value_head_model(
        tokenizer,
        d_model=args.d_model,
        n_layer=args.layers,
        n_head=args.nhead,
        n_positions=args.max_seq_len,
    )
    reference = create_reference_model(policy)

    # Configure PPO — keep values small so smoke tests run quickly on CPU/MPS.
    batch_size = max(1, min(args.batch_size, len(train_dataset)))
    total_batches = max(1, args.epochs)

    config = PPOConfig(
        output_dir="trl-output",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=args.lr,
    )
    config.batch_size = batch_size
    config.mini_batch_size = batch_size
    config.num_ppo_epochs = max(1, args.ppo_epochs)
    config.response_length = args.response_length
    config.temperature = args.temperature
    config.num_total_batches = total_batches
    config.total_episodes = total_batches * batch_size
    config.whiten_rewards = False
    config.save_strategy = "no"
    config.logging_strategy = "steps"
    config.logging_steps = 1
    config.eval_strategy = "no"
    config.push_to_hub = False
    config.report_to = []
    # Align auxiliary knobs used inside the trainer loop
    config.local_rollout_forward_batch_size = batch_size

    trainer = PPOTrainer(
        config,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=policy,
        value_model=policy,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
    )
    return trainer


def main():
    ap = argparse.ArgumentParser(description="TRL PPO smoke test for patch-token policy")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=8)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--response-length", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    args = ap.parse_args()

    set_seed(args.seed)

    samples, direction_angles = _prepare_samples(args)
    if not samples:
        raise RuntimeError("No samples available for training; check data-root path")

    tokenizer = build_patch_tokenizer(direction_angles)
    train_dataset = _tokenize_prompts(samples, tokenizer, args.max_seq_len)
    trainer = _build_trainer(args, tokenizer, train_dataset)

    trainer.train()
    trainer.generate_completions(sampling=True)


if __name__ == "__main__":
    main()
