#!/usr/bin/env python3
"""TRL PPO with a Reward Model (RM) for K-step episodes via generate → step.

We keep TRL's public API. We do not modify TRL internals. Physics stays in the
RM training path; PPO uses the frozen RM only for scoring per-step rewards.
"""

from __future__ import annotations

import argparse
from typing import List, Sequence

from datasets import Dataset
from transformers import set_seed
from trl import PPOConfig, PPOTrainer
from trl.models import create_reference_model

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids
from doa_rl.hf.logits_mask import NoRepeatDirectionLogitsProcessor
import torch
import torch.nn.functional as F

try:
    from peft import PeftModel
    HAS_PEFT = True
except Exception:
    HAS_PEFT = False


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
    ap = argparse.ArgumentParser(description="TRL PPO with RM (K-step, generate→step)")
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--rm-adapters", type=str, required=True, help="Path to RM LoRA adapters dir (save_pretrained)")
    ap.add_argument("--rm-heads", type=str, required=True, help="Path to RM heads checkpoint (embeddings + v_head)")
    # Optional SFT warm start for policy
    ap.add_argument("--sft-policy-adapters", type=str, default=None, help="Optional: path to SFT policy adapters")
    ap.add_argument("--sft-policy-heads", type=str, default=None, help="Optional: path to SFT policy embeddings heads .pt")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--ppo-epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps","cuda"])
    ap.add_argument("--sample-rate", type=int, default=16000)
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

    # Device selection
    dev = args.device
    if dev == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    device = torch.device(dev)
    print(f"Device: {device}")

    # Policy and (optional) SFT warm start before reference snapshot
    policy, _ = build_value_head_model(tokenizer)
    policy.to(device)
    if args.sft_policy_adapters:
        if not HAS_PEFT:
            raise RuntimeError("peft is required to load SFT policy adapters. Install with: pip install peft")
        policy.pretrained_model = PeftModel.from_pretrained(policy.pretrained_model, args.sft_policy_adapters)
        if args.sft_policy_heads:
            heads = torch.load(args.sft_policy_heads, map_location="cpu")
            emb_layer = policy.pretrained_model.get_input_embeddings()
            sd = heads.get("embeddings")
            if hasattr(emb_layer, "base_layer"):
                emb_layer.base_layer.load_state_dict(sd)
            else:
                emb_layer.load_state_dict(sd)
        print("Loaded SFT policy warm start before reference snapshot")
    reference = create_reference_model(policy)
    reference.to(device)

    # Reward model (RM): same architecture, load LoRA adapters + embeddings + v_head, then freeze
    rm_model, _ = build_value_head_model(tokenizer)
    if not HAS_PEFT:
        raise RuntimeError("peft is required to load RM LoRA adapters. Install with: pip install peft")
    rm_model.to(device)
    rm_model.eval()
    # Attach adapters to the backbone
    rm_model.pretrained_model = PeftModel.from_pretrained(rm_model.pretrained_model, args.rm_adapters)
    # Load embeddings and v_head
    heads = torch.load(args.rm_heads, map_location="cpu")
    emb = rm_model.pretrained_model.get_input_embeddings()
    state_dict = heads.get("embeddings", None)
    if state_dict is None:
        raise RuntimeError("RM heads checkpoint missing 'embeddings' state dict")
    # PEFT may wrap embedding; support both
    if hasattr(emb, "base_layer"):
        emb.base_layer.load_state_dict(state_dict)
    else:
        emb.load_state_dict(state_dict)
    rm_model.v_head.load_state_dict(heads["v_head"])  # restore value head
    rm_model.score = rm_model.v_head  # expose score
    # Freeze
    for p in rm_model.parameters():
        p.requires_grad = False
    rm_model.eval()

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
        use_cpu=(device.type == "cpu"),
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
        reward_model=rm_model,  # required by TRL API; we will supply external rewards to step()
        train_dataset=ds,
        value_model=policy,  # share value head with policy for PPO baseline
        eval_dataset=ds,
    )

    # Generation constraints: only direction tokens, no repeats
    allowed = direction_token_ids(tokenizer)
    processor = NoRepeatDirectionLogitsProcessor(allowed)

    # Helper: build RM scoring input for a prefix of directions
    def rm_score_for_prefix(dir_prefix_ids: Sequence[int], patch_ids: Sequence[int]) -> float:
        bos = tokenizer.bos_token_id
        # Build sequence: [BOS] + dir_prefix + patch_ids
        ids = [bos] + list(dir_prefix_ids) + list(patch_ids)
        inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        attn = torch.ones_like(inp, device=device)
        with torch.no_grad():
            out = rm_model.pretrained_model(input_ids=inp, attention_mask=attn, output_hidden_states=True, return_dict=True)
            last_hidden = out.hidden_states[-1]
            vals = rm_model.v_head(last_hidden).squeeze(-1)[0]  # (S,)
        # Patch tokens are appended after the prefix; select that slice directly
        start = 1 + len(dir_prefix_ids)
        end = start + len(patch_ids)
        if end > vals.numel():
            raise RuntimeError("RM scoring: constructed sequence shorter than expected for patch slice")
        patch_vals = vals[start:end]
        return float(patch_vals.mean().item())

    # Extract patch token ids from the prompt (strip specials; keep only <P_...>)
    def get_patch_ids(ids_row: torch.Tensor) -> List[int]:
        toks = tokenizer.convert_ids_to_tokens(ids_row.tolist())
        # Prompt rows contain only patch tokens (no directions); keep all non-special tokens
        patch = [tid for tid, tok in zip(ids_row.tolist(), toks) if tok not in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token)]
        return patch

    # Wrap generate on trainer models to inject logits processor
    def _wrap_generate(module):
        if not hasattr(module, 'generate'):
            return
        orig = module.generate
        def gen_with_mask(*g_args, **g_kwargs):
            procs = list(g_kwargs.get("logits_processor", []))
            procs.append(processor)
            g_kwargs["logits_processor"] = procs
            g_kwargs.setdefault("max_new_tokens", args.K)
            g_kwargs.setdefault("min_new_tokens", args.K)
            g_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
            return orig(*g_args, **g_kwargs)
        module.generate = gen_with_mask

    _wrap_generate(trainer.model.policy)
    if hasattr(trainer.model, 'pretrained_model'):
        _wrap_generate(trainer.model.pretrained_model)

    # Prefer manual generate→step if this TRL version exposes it; otherwise run trainer.train() for a smoke step
    if hasattr(trainer, 'step'):
        total = len(ds)
        batch_size = bs
        batches = 0
        for start in range(0, total, batch_size):
            if batches >= cfg.num_total_batches:
                break
            end = min(start + batch_size, total)
            input_batch = torch.tensor(enc["input_ids"][start:end], dtype=torch.long, device=device)
            attn_batch = torch.tensor(enc["attention_mask"][start:end], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = policy.pretrained_model.generate(
                    input_ids=input_batch,
                    attention_mask=attn_batch,
                    max_new_tokens=args.K,
                    min_new_tokens=args.K,
                    do_sample=True,
                )
            responses_batch = gen[:, -args.K:]
            rewards_list: List[torch.Tensor] = []
            query_tensors = [row.detach().clone() for row in input_batch]
            response_tensors = [row.detach().clone() for row in responses_batch]
            for i in range(end - start):
                patch_ids = get_patch_ids(input_batch[i])
                dir_ids = response_tensors[i].tolist()
                if any(t not in allowed for t in dir_ids):
                    bad = [t for t in dir_ids if t not in allowed]
                    raise RuntimeError(f"Generated non-direction tokens found: ids={bad}")
                if len(set(dir_ids)) != len(dir_ids):
                    raise RuntimeError(f"Duplicate directions in a trajectory are disallowed: {dir_ids}")
                scores = []
                for j in range(1, args.K + 1):
                    Sj = rm_score_for_prefix(dir_ids[:j], patch_ids)
                    scores.append(Sj)
                deltas = [scores[0] - 0.0] + [scores[j] - scores[j-1] for j in range(1, len(scores))]
                rewards_list.append(torch.tensor(deltas, dtype=torch.float32, device=device))
            trainer.step(query_tensors, response_tensors, rewards_list)
            batches += 1
        print("Training run completed (manual step path).")
    else:
        print("Info: PPOTrainer.step() not available in this TRL version; running trainer.train() for smoke test.")
        trainer.train()
        print("Training run completed (trainer.train path).")


if __name__ == "__main__":
    main()
