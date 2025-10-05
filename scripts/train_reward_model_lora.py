#!/usr/bin/env python3
"""Train Reward Model with LoRA - Efficient compromise between frozen and full fine-tuning.

This addresses the cold-start problem (docs/reward_model_cold_start_analysis.md)
while being much more efficient than full fine-tuning.

Key features:
1. LoRA adapters for transformer layers (low-rank updates)
2. Trainable embeddings for new tokens (essential!)
3. Trainable v_head for reward prediction
4. ~99% fewer trainable parameters than full fine-tuning

Based on:
- Hu et al. (2021): LoRA paper
- Aghajanyan et al. (2020): Low intrinsic dimensionality of fine-tuning
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import set_seed

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("Warning: peft library not found. Install with: pip install peft")

from doa_rl.data import DoADataset, create_dataloader
from doa_rl.features import PatchTokenizer
from doa_rl.hf import build_patch_tokenizer, build_value_head_model
from doa_rl.assets import load_H, load_W
from nmf_localizer.core.usm_trainer import USMTrainer


def _discover_angles(root: str) -> List[int]:
    base = Path(root)
    angles = sorted(
        {
            int(p.name.split("_")[1])
            for p in base.glob("angle_*")
            if p.is_dir() and p.name.split("_")[1].isdigit()
        }
    )
    if not angles:
        raise RuntimeError(f"No angle_* directories found under {root}")
    return angles


def _prepare_samples(args, direction_angles: List[int]) -> Tuple[List[str], Dict[str, Dict[str, torch.Tensor]]]:
    ds = DoADataset(
        args.data_root,
        direction_angles,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    dl = create_dataloader(ds, batch_size=1, shuffle=False)
    tok = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
    prompts: List[str] = []
    cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for batch in dl:
        Y_t: torch.Tensor = batch["Y"].squeeze(0)
        Y_np = Y_t.numpy()
        prompt = " ".join(tok(Y_np))
        prompts.append(prompt)
        cache[prompt] = {"Y": Y_t.clone()}
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts, cache



def _parse_completion(text: str, K: int, direction_angles: List[int]) -> List[int]:
    dirs: List[int] = []
    for m in re.finditer(r"<D_(\d{3})>", text):
        angle = int(m.group(1))
        if angle in direction_angles:
            dirs.append(direction_angles.index(angle))
        if len(dirs) >= K:
            break
    return dirs


def _compute_deltaIS_reward(
    Y: torch.Tensor,
    s_hat: torch.Tensor,
    H: torch.Tensor,
    selected: List[int],
    trace: List[Dict[str, float]],
) -> float:
    eps = 1e-12
    Y = Y.clone().float()
    F, N = Y.shape
    # Enforce exact frequency grid match; no resampling allowed
    if H.shape[0] != F:
        raise ValueError(
            f"H and Y must have the same number of frequency bins (F). "
            f"Got H.F={int(H.shape[0])} vs Y.F={int(F)}. "
            f"Align STFT config (fs/n_fft/band) to match assets."
        )
    Hs = (H.float() * s_hat.view(-1, 1).float())
    
    def is_div(Ytrue: torch.Tensor, Ymix: torch.Tensor) -> torch.Tensor:
        Yhat = torch.clamp(Ymix.view(-1, 1).expand(F, N), min=eps)
        ratio = torch.clamp(Ytrue, min=eps) / Yhat
        return torch.sum(ratio - torch.log(ratio) - 1.0)
    
    Y_mix = torch.full((F,), eps, dtype=torch.float32)
    prev = is_div(Y, Y_mix)
    total = 0.0
    for d in selected:
        Y_mix = torch.clamp(Y_mix + Hs[:, d], min=eps)
        cur = is_div(Y, Y_mix)
        # Absolute improvement (exposes magnitude issues if present)
        step_val = (prev - cur)
        total += float(step_val.item())
        trace.append({
            "prev": float(prev.item()),
            "cur": float(cur.item()),
            "delta_abs": float((prev - cur).item()),
            "delta_rel": float(((prev - cur) / torch.clamp(prev, min=eps)).item()),
        })
        prev = cur
    return total


def main():
    ap = argparse.ArgumentParser(
        description="Train RM with LoRA (efficient compromise)"
    )
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    
    # LoRA hyperparameters
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank (4, 8, 16)")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (typically 2×r)")
    ap.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    ap.add_argument("--lora-target-modules", type=str, default="c_attn,c_proj",
                    help="Comma-separated list of modules to add LoRA")
    
    # Training hyperparameters
    ap.add_argument("--rm-epochs", type=int, default=20, help="RM training epochs")
    ap.add_argument("--lr-lora", type=float, default=1e-4, help="Learning rate for LoRA adapters")
    ap.add_argument("--lr-embed", type=float, default=1e-4, help="Learning rate for embeddings")
    ap.add_argument("--lr-vhead", type=float, default=1e-3, help="Learning rate for v_head")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"],
                    help="Compute device for training (auto→mps>cuda>cpu)")
    
    # Data parameters
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-fft", type=int, default=2048)
    ap.add_argument("--freq-min", type=float, default=300.0)
    ap.add_argument("--freq-max", type=float, default=3000.0)
    ap.add_argument("--patch-fp", type=int, default=16)
    ap.add_argument("--patch-np", type=int, default=10)
    ap.add_argument("--max-samples", type=int, default=0)
    
    ap.add_argument("--out", type=str, default="rm_ckpt_lora")
    args = ap.parse_args()

    if not HAS_PEFT:
        print("\n" + "=" * 60)
        print("ERROR: peft library is required for LoRA training")
        print("Install with: pip install peft")
        print("=" * 60)
        return

    set_seed(args.seed)

    # Select device
    dev = args.device
    if dev == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    device = torch.device(dev)
    print(f"\nDevice: {device}")

    # Prepare data
    direction_angles = _discover_angles(args.data_root)
    prompts, cache = _prepare_samples(args, direction_angles)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Load assets
    H_full, anglesT = load_H(args.tf_path)
    H_full = H_full.float().cpu()
    anglesT = anglesT.cpu().numpy().astype(int).tolist()
    col_idx: List[int] = []
    for a in direction_angles:
        # Require exact angle match; no nearest fallback
        if int(a) not in anglesT:
            raise RuntimeError(
                f"Angle {int(a)}° not found in TF angles {anglesT}. "
                f"Ensure dataset angles exactly match TF asset."
            )
        j = anglesT.index(int(a))
        col_idx.append(j)
    if len(set(col_idx)) != len(col_idx):
        raise RuntimeError(
            f"Duplicate TF column mappings detected: {col_idx}. "
            f"This indicates repeated/ambiguous angles; fix assets or dataset."
        )
    H = H_full[:, col_idx].contiguous()

    # Precompute ŝ
    if args.w_path.endswith('.npz'):
        W_np = np.load(args.w_path)["W"]
    else:
        W_t = load_W(args.w_path)
        W_np = W_t.cpu().numpy()
    for p in prompts:
        Y_t = cache[p]["Y"]
        s_hat_np = USMTrainer.compute_content_s_hat(Y=Y_t.numpy(), W=W_np, mode="S1", n_iter=50, l1=0.0)
        cache[p]["s_hat"] = torch.from_numpy(s_hat_np.astype(np.float32))

    # Build base RM model
    rm_model, _ = build_value_head_model(tokenizer)
    
    print("\n" + "=" * 60)
    print("Applying LoRA to Reward Model")
    print("=" * 60)
    
    # Configure LoRA
    target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    
    print(f"LoRA config:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    print(f"  - Target modules: {target_modules}")
    
    # Apply LoRA to the backbone
    rm_model.pretrained_model = get_peft_model(
        rm_model.pretrained_model,
        lora_config
    )
    
    # CRITICAL: Unfreeze embeddings for new tokens
    # This is essential for learning patch/direction token meanings!
    embedding_layer = rm_model.pretrained_model.get_input_embeddings()
    if hasattr(embedding_layer, 'base_layer'):
        # PEFT wraps the embedding layer
        embedding_layer.base_layer.weight.requires_grad = True
    else:
        embedding_layer.weight.requires_grad = True
    
    # V-head is already trainable
    rm_model.score = rm_model.v_head
    rm_model.to(device)
    rm_model.train()
    
    # Count parameters
    def count_parameters(model, pattern: str = None):
        if pattern:
            return sum(
                p.numel() for n, p in model.named_parameters()
                if p.requires_grad and pattern.lower() in n.lower()
            )
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    lora_params = count_parameters(rm_model, "lora")
    # GPT2 uses 'wte' for word token embeddings, not 'embed'
    embed_params = count_parameters(rm_model, "wte") + count_parameters(rm_model, "wpe")
    vhead_params = sum(p.numel() for p in rm_model.v_head.parameters())
    total_trainable = sum(p.numel() for p in rm_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in rm_model.parameters())
    
    print(f"\nParameter counts:")
    print(f"  - LoRA adapters:  {lora_params:,}")
    print(f"  - Embeddings:     {embed_params:,}")
    print(f"  - V-head:         {vhead_params:,}")
    print(f"  - Total trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.2f}%)")
    print(f"  - Reduction vs full FT: {100*(1 - total_trainable/total_params):.2f}%")
    print("=" * 60 + "\n")

    # Construct RM training dataset
    dir_tokens = list(tokenizer.direction_tokens)
    texts: List[str] = []
    rewards: List[float] = []
    is_prev_list: List[float] = []
    is_final_list: List[float] = []
    is_step_delta_abs: List[float] = []
    is_step_delta_rel: List[float] = []
    for p in prompts:
        idxs = np.random.choice(len(dir_tokens), size=args.K, replace=False)
        comp = " ".join(dir_tokens[i] for i in idxs)
        text = p + " " + comp
        texts.append(text)
        dirs = [tokenizer.direction_tokens.index(dir_tokens[i]) for i in idxs]
        Y = cache[p]["Y"]
        s_hat = cache[p]["s_hat"]
        trace_steps: List[Dict[str, float]] = []
        R = _compute_deltaIS_reward(
            Y, s_hat, H, dirs,
            trace=trace_steps,
        )
        rewards.append(float(R))
        is_prev_list.append(trace_steps[0]["prev"])  # before first selection
        is_final_list.append(trace_steps[-1]["cur"])  # after last selection
        is_step_delta_abs.extend([t["delta_abs"] for t in trace_steps])
        is_step_delta_rel.extend([t["delta_rel"] for t in trace_steps])

    # Tokenize
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    tgt = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    print(f"Training data:")
    print(f"  - Samples: {len(texts)}")
    print(f"  - Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"  - Reward mean±std: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
    if is_prev_list:
        prev = np.array(is_prev_list, dtype=float)
        final = np.array(is_final_list, dtype=float)
        step_abs = np.array(is_step_delta_abs, dtype=float)
        step_rel = np.array(is_step_delta_rel, dtype=float)
        def stats(x):
            return {
                "min": float(np.min(x)),
                "median": float(np.median(x)),
                "mean": float(np.mean(x)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "max": float(np.max(x)),
            }
        print("IS divergence (absolute) stats:")
        print("  - prev (before any selection):", stats(prev))
        print("  - final (after K selections):", stats(final))
        print("  - per-step ΔIS abs:", stats(step_abs))
        print("  - per-step ΔIS rel:", stats(step_rel), "(clipped by eps when prev≈0)")

    # Optimizer with differential learning rates
    # Group 1: LoRA adapters (medium LR)
    # Group 2: Embeddings (medium LR, critical for new tokens!)  
    # Group 3: V-head (highest LR)
    optimizer = torch.optim.Adam([
        {
            "params": [p for n, p in rm_model.named_parameters() 
                      if p.requires_grad and "lora" in n.lower()],
            "lr": args.lr_lora,
        },
        {
            # GPT2 uses 'wte' (word token embeddings) and 'wpe' (position embeddings)
            "params": [p for n, p in rm_model.named_parameters() 
                      if p.requires_grad and ("wte" in n.lower() or "wpe" in n.lower())],
            "lr": args.lr_embed,
        },
        {
            "params": rm_model.v_head.parameters(),
            "lr": args.lr_vhead,
        },
    ])
    
    print(f"Optimizer:")
    print(f"  - LoRA LR:       {args.lr_lora}")
    print(f"  - Embedding LR:  {args.lr_embed}")
    print(f"  - V-head LR:     {args.lr_vhead}\n")
    
    loss_fn = nn.MSELoss()
    
    # Training loop
    print("=" * 60)
    print("Training")
    print("=" * 60)
    
    best_loss = float('inf')
    best_corr = -1.0
    
    for epoch in range(args.rm_epochs):
        rm_model.zero_grad(set_to_none=True)
        
        # Forward pass
        out = rm_model.pretrained_model(
            input_ids=input_ids,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden = out.hidden_states[-1]
        value = rm_model.v_head(last_hidden).squeeze(-1)
        
        # Take last token's value as scalar reward
        lengths = attn.sum(dim=1) - 1
        pred = value[torch.arange(value.size(0)), lengths]
        
        # MSE loss
        loss = loss_fn(pred, tgt)
        
        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred - tgt))
            correlation = torch.corrcoef(torch.stack([pred, tgt]))[0, 1]
            if torch.isnan(correlation):
                correlation = torch.tensor(0.0)
        
        print({
            "epoch": epoch,
            "loss": float(loss.item()),
            "mae": float(mae.item()),
            "correlation": float(correlation.item()),
        })
        
        if loss.item() < best_loss:
            best_loss = loss.item()
        if correlation.item() > best_corr:
            best_corr = correlation.item()

    print(f"\nTraining completed!")
    print(f"  - Best loss: {best_loss:.4f}")
    print(f"  - Best correlation: {best_corr:.4f}")
    
    # Save LoRA adapters and heads
    lora_dir = f"{args.out}_adapters"
    heads_path = f"{args.out}_heads.pt"
    
    rm_model.pretrained_model.save_pretrained(lora_dir)
    torch.save({
        "embeddings": embedding_layer.state_dict(),
        "v_head": rm_model.v_head.state_dict(),
        "config": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "best_loss": best_loss,
            "best_correlation": best_corr,
        }
    }, heads_path)
    
    print(f"\nSaved:")
    print(f"  - LoRA adapters: {lora_dir}/")
    print(f"  - Embeddings + V-head: {heads_path}")
    
    # Analyze embedding quality
    print("\n" + "=" * 60)
    print("Embedding Quality Analysis")
    print("=" * 60)
    
    embeddings = rm_model.pretrained_model.get_input_embeddings()
    dir_ids = tokenizer.convert_tokens_to_ids(list(tokenizer.direction_tokens))
    dir_embs = embeddings(torch.tensor(dir_ids, device=device))
    
    # Compute pairwise cosine similarity
    similarity = torch.nn.functional.cosine_similarity(
        dir_embs.unsqueeze(1),
        dir_embs.unsqueeze(0),
        dim=2,
    )
    
    off_diag_mean = (similarity.sum() - similarity.trace()).item() / (len(dir_ids) * (len(dir_ids) - 1))
    
    print(f"Direction token embeddings:")
    print(f"  - Similarity range: [{similarity.min():.3f}, {similarity.max():.3f}]")
    print(f"  - Mean off-diagonal: {off_diag_mean:.3f}")
    print(f"\nInterpretation:")
    print(f"  - Random embeddings: ~0.0")
    print(f"  - Learned structure: >0.3 for nearby angles")
    print("=" * 60)


if __name__ == "__main__":
    main()
