#!/usr/bin/env python3
"""Train Reward Model with LoRA - Efficient compromise between frozen and full fine-tuning.
# Code snapshot aligned with results commit c7b3d41 (per‑patch IS targets)

Reward path (current): deltaIS_localizer with per‑bin target
- Reconstruction: Y_hat via localizer‑style A·X using selected [diag(H_d)W] blocks with IS updates.
- Target: reward = − IS(Y||Y_hat) / (F·N) (single scalar per sample; pred shape [B], tgt shape [B]).
- Diagnostics: also print Absolute IS (sum over F·N) for fail‑fast visibility.
- IS updates: default 100 with adaptive early stopping (tol, min_iters, patience).

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
import json
import os
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
from doa_rl.hf import build_patch_tokenizer, build_value_head_model, direction_token_ids
from doa_rl.assets import load_H, load_W
# Reuse scoring helpers to ensure identical semantics with SFT/eval
from scripts.train_sft_policy_with_rm import (
    rm_score_for_prefix,
    get_patch_ids,
    compute_rm_greedy_teacher,
)
# USMTrainer not needed after switching to localizer-style reward



def _is_factorize_with_selected_blocks(
    Y: torch.Tensor,  # (F,N)
    W: torch.Tensor,  # (F,K)
    H: torch.Tensor,  # (F,D)
    selected: List[int],
    n_iter: int = 20,
    tol: float = 1e-4,
    min_iters: int = 5,
    patience: int = 3,
    eps: float = 1e-12,
    clip: float = 1e3,
) -> Tuple[torch.Tensor, float]:
    """Localizer-style reconstruction: build A from selected directions and solve X via IS updates.

    Returns Y_hat (F,N) and IS(Y||Y_hat).
    """
    # Fail-fast validations (no fallbacks allowed)
    if not selected:
        raise RuntimeError(
            "No directions selected (selected is empty). Set K>=1 and ensure selection produces valid indices."
        )
    Yc = torch.clamp(Y.detach().cpu().float(), min=eps)
    Wc = torch.clamp(W.detach().cpu().float(), min=eps)
    Hc = torch.clamp(H.detach().cpu().float(), min=eps)
    F, N = Yc.shape
    Fw, K = Wc.shape
    Fh, D = Hc.shape
    if Fw != F or Fh != F:
        raise RuntimeError(
            f"Shape mismatch: Y.F={F}, W.F={Fw}, H.F={Fh}. Align STFT config (fs/n_fft/band) and assets."
        )
    if any((d < 0 or d >= D) for d in selected):
        raise RuntimeError(
            f"Selected direction indices out of range: {selected}; valid range is [0,{D-1}]"
        )
    if len(set(selected)) != len(selected):
        raise RuntimeError(
            f"Duplicate directions in selection: {selected}. Duplicates are disallowed (fail-fast)."
        )
    # Build A_sel = [diag(H_d) W] for selected directions
    blocks: List[torch.Tensor] = []
    for d in selected:
        Hd = Hc[:, d:d+1]  # (F,1)
        A_d = Wc * Hd      # broadcast diag(H_d) @ W
        blocks.append(A_d)
    A_sel = torch.cat(blocks, dim=1)
    P = A_sel.shape[1]
    X = torch.full((P, N), 1.0 / max(P, 1), dtype=Yc.dtype)
    Yhat = torch.clamp(A_sel @ X, min=eps)
    # Initial IS
    r0 = Yc / Yhat
    is_prev = torch.sum(r0 - torch.log(torch.clamp(r0, min=eps)) - 1.0).item()
    stable_steps = 0
    iters_run = 0
    for it in range(max(int(n_iter), 0)):
        Yhat = torch.clamp(Yhat, min=eps)
        invYhat = 1.0 / Yhat
        y_over_yhat2 = Yc / (Yhat * Yhat)
        num = A_sel.t() @ y_over_yhat2
        den = A_sel.t() @ invYhat
        ratio = num / torch.clamp(den, min=eps)
        ratio = torch.clamp(ratio, min=eps, max=clip)
        X = X * torch.sqrt(ratio)
        Yhat = A_sel @ X
        # Convergence check on IS relative change
        r = Yc / torch.clamp(Yhat, min=eps)
        is_cur = torch.sum(r - torch.log(torch.clamp(r, min=eps)) - 1.0).item()
        rel_change = abs(is_prev - is_cur) / max(abs(is_prev), eps)
        iters_run = it + 1
        if iters_run >= min_iters:
            if rel_change < tol:
                stable_steps += 1
                if stable_steps >= patience:
                    break
            else:
                stable_steps = 0
        is_prev = is_cur
    Yhat = torch.clamp(Yhat, min=eps)
    r = Yc / Yhat
    is_val = torch.sum(r - torch.log(torch.clamp(r, min=eps)) - 1.0).item()
    return Yhat, is_val

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
        # Track source path for diagnostics
        src_path = batch.get("path", None)
        if isinstance(src_path, (list, tuple)):
            src_path = src_path[0] if src_path else None
        cache[prompt] = {"Y": Y_t.clone(), "path": str(src_path) if src_path is not None else ""}
        if args.max_samples and len(prompts) >= args.max_samples:
            break
    return prompts, cache



def _parse_completion(*args, **kwargs):
    # No longer used (we sample directions directly); kept as stub to preserve namespace stability if imported
    return []


def main():
    ap = argparse.ArgumentParser(
        description="Train RM with LoRA (efficient compromise)"
    )
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--tf-path", type=str, required=True)
    ap.add_argument("--w-path", type=str, required=True)
    ap.add_argument("--K", type=int, default=3)
    # Reward path fixed to localizer-style A·X reconstruction (deltaIS_localizer)
    ap.add_argument("--is-iters", type=int, default=100, help="Max iterations for IS updates in localizer reward")
    ap.add_argument("--is-tol", type=float, default=1e-4, help="Relative IS convergence tolerance")
    ap.add_argument("--is-min-iters", type=int, default=5, help="Minimum iterations before checking convergence")
    ap.add_argument("--is-patience", type=int, default=3, help="Consecutive convergence steps required to stop early")
    
    # LoRA hyperparameters
    ap.add_argument("--lora-r", type=int, default=8, help="LoRA rank (4, 8, 16)")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (typically 2×r)")
    ap.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    ap.add_argument("--lora-target-modules", type=str, default="c_attn,c_proj",
                    help="Comma-separated list of modules to add LoRA")
    
    # Training hyperparameters
    ap.add_argument("--rm-epochs", type=int, default=20, help="RM training epochs")
    ap.add_argument("--eval-every", type=int, default=1, help="Evaluate Top-1/Top-K every N epochs (directions-first)")
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
    ap.add_argument("--debug-info", action="store_true", help="Print tensor shapes and sample pred/target values")
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
        W_t = torch.from_numpy(W_np).float()
    else:
        W_t = load_W(args.w_path)
        W_np = W_t.cpu().numpy()

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
    rewards: List[float] = []  # per-sample scalar (mean over patches) for logging only
    is_final_list: List[float] = []
    sample_logs: List[Dict[str, float]] = []
    patch_targets_all: List[torch.Tensor] = []  # per-sample per-patch targets (training)
    results_dir = os.path.join("results", f"{args.out}")
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = os.path.join(results_dir, "numeric_diagnostics.jsonl")
    for p in prompts:
        idxs = np.random.choice(len(dir_tokens), size=args.K, replace=False)
        comp = " ".join(dir_tokens[i] for i in idxs)
        # Directions-first so patch tokens can attend to selected directions (fixes causal visibility)
        text = comp + " " + p
        texts.append(text)
        dirs = [tokenizer.direction_tokens.index(dir_tokens[i]) for i in idxs]
        Y = cache[p]["Y"]
        # Localizer-style: A·X with IS updates on selected blocks; use per-patch reward
        Yhat_sel, is_val = _is_factorize_with_selected_blocks(
            Y=Y, W=W_t, H=H, selected=dirs,
            n_iter=args.is_iters, tol=args.is_tol,
            min_iters=args.is_min_iters, patience=args.is_patience
        )
        F, N = Y.shape
        # Per-patch IS: average g(r) over bins in each patch; negate for reward convention
        eps = 1e-12
        Y_cl = torch.clamp(Y, min=eps)
        ratio_map = Y_cl / torch.clamp(Yhat_sel, min=eps)
        g_map = ratio_map - torch.log(torch.clamp(ratio_map, min=eps)) - 1.0
        Fp, Np = int(args.patch_fp), int(args.patch_np)
        Lf, Lt = F // Fp, N // Np
        patch_vals: List[float] = []
        for i in range(Lf):
            for j in range(Lt):
                patch = g_map[i*Fp:(i+1)*Fp, j*Np:(j+1)*Np]
                patch_vals.append(float(patch.mean().item()))
        patch_targets = -torch.tensor(patch_vals, dtype=torch.float32)  # reward per patch
        patch_targets_all.append(patch_targets)
        # Per-sample scalar (mean over patches) for logging
        rewards.append(float(patch_targets.mean().item()))
        is_final_list.append(float(is_val))

        # Numeric diagnostics per sample: per‑patch IS only
        patch_is = np.array(patch_vals, dtype=float)
        def stat(v):
            return float(v)
        log_row = {
            "path": cache[p].get("path", ""),
            "F": int(F),
            "N": int(N),
            "patch_is_mean": stat(patch_is.mean()),
            "patch_is_p95": stat(np.percentile(patch_is, 95)),
            "patch_is_p99": stat(np.percentile(patch_is, 99)),
            "patch_is_max": stat(patch_is.max()),
        }
        sample_logs.append(log_row)
        # Persist as JSONL incrementally to avoid data loss on interruption
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

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
    if args.debug_info:
        tgt_dbg = torch.tensor(rewards, dtype=torch.float32)
        print("Debug — batch tensors:")
        print(f"  input_ids: {tuple(input_ids.shape)}  attention: {tuple(attn.shape)}  per-sample scalar tgt (logging): {tuple(tgt_dbg.shape)}")
        print(f"  tgt (scalar, log) stats: min={float(tgt_dbg.min().item()):.4f} max={float(tgt_dbg.max().item()):.4f} mean={float(tgt_dbg.mean().item()):.4f}")
    
    print(f"Training data:")
    print(f"  - Samples: {len(texts)}")
    print(f"  - Reward range: [{min(rewards):.2f}, {max(rewards):.2f}]")
    print(f"  - Reward mean±std: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}\n")
    if sample_logs:
        # Per‑patch IS stats across samples (diagnostics)
        def stats(x):
            return {
                "min": float(np.min(x)),
                "median": float(np.median(x)),
                "mean": float(np.mean(x)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "max": float(np.max(x)),
            }
        patch_means = np.array([row.get("patch_is_mean", float('nan')) for row in sample_logs], dtype=float)
        patch_means = patch_means[~np.isnan(patch_means)]
        if patch_means.size > 0:
            print("Per‑patch IS (diagnostics) — mean over patches per sample:", stats(patch_means))
        # Print concise per-sample diagnostics (first 5 for brevity)
        print("\nNumeric diagnostics (first 5 samples):")
        for row in sample_logs[:5]:
            print({
                "path": os.path.basename(row.get("path", "")) or "(n/a)",
                "F": row["F"], "N": row["N"],
                "patch_is_mean": round(row["patch_is_mean"], 4),
                "patch_is_p99": round(row["patch_is_p99"], 4),
            })
        print(f"Saved numeric diagnostics JSONL: {jsonl_path}")

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
        value = rm_model.v_head(last_hidden).squeeze(-1)  # [B, S]
        # Build patch token mask per sample and gather predictions/targets
        pred_list: List[torch.Tensor] = []
        tgt_list: List[torch.Tensor] = []
        bos_tok = tokenizer.bos_token or "<BOS>"
        eos_tok = tokenizer.eos_token or "<EOS>"
        pad_tok = tokenizer.pad_token or "<PAD>"
        for b in range(input_ids.size(0)):
            toks = tokenizer.convert_ids_to_tokens(input_ids[b].tolist())
            # Directions-first: take first P non-special tokens AFTER the last <D_> token
            P = patch_targets_all[b].numel()
            last_dir = max([i for i, t in enumerate(toks) if t.startswith("<D_")], default=-1)
            usable = [i for i, t in enumerate(toks[last_dir+1:]) if t not in (bos_tok, eos_tok, pad_tok)]
            if len(usable) < P:
                raise RuntimeError(
                    f"Patch token alignment failed: found {len(usable)} usable tokens after direction prefix, "
                    f"but need {P}. Ensure prompts contain all patch tokens and lengths align with PatchTokenizer (Fp={args.patch_fp}, Np={args.patch_np})."
                )
            patch_pos = [last_dir + 1 + i for i in usable[:P]]
            pred_b = value[b, torch.tensor(patch_pos, device=value.device)]
            tgt_b = patch_targets_all[b].to(value.device)
            if pred_b.numel() != tgt_b.numel():
                raise RuntimeError(
                    f"Patch length mismatch: pred={pred_b.numel()} vs tgt={tgt_b.numel()}. "
                    f"Ensure tokenizer patch grid matches PatchTokenizer (Fp={args.patch_fp}, Np={args.patch_np})."
                )
            pred_list.append(pred_b)
            tgt_list.append(tgt_b)
        pred_all = torch.cat(pred_list, dim=0)
        tgt_all = torch.cat(tgt_list, dim=0)
        loss = loss_fn(pred_all, tgt_all)
        if args.debug_info and epoch == 0:
            k = min(5, pred_all.numel())
            print("  sample pred_patch[:k] vs tgt_patch[:k] (per-patch IS target):")
            print("  pred:", [float(x) for x in pred_all[:k].detach().cpu()])
            print("  tgt :", [float(x) for x in tgt_all[:k].detach().cpu()])
        
        # Backward with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            mae = torch.mean(torch.abs(pred_all - tgt_all))
            correlation = torch.corrcoef(torch.stack([pred_all, tgt_all]))[0, 1]
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

        # Periodic directions-first evaluation (Top-1 and recall@K)
        do_eval = (args.eval_every > 0) and (((epoch + 1) % args.eval_every == 0) or (epoch == args.rm_epochs - 1))
        if do_eval:
            rm_model.eval()
            try:
                # Build patch-only prompts and ground-truth angles in dataset order
                direction_angles = [int(a) for a in _discover_angles(args.data_root)]
                ds_eval = DoADataset(
                    args.data_root,
                    direction_angles,
                    fs=args.sample_rate,
                    n_fft=args.n_fft,
                    freq_min=args.freq_min,
                    freq_max=args.freq_max,
                )
                dl_eval = create_dataloader(ds_eval, batch_size=1, shuffle=False)
                tok_patch = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
                eval_prompts: List[str] = []
                gt_angles: List[int] = []
                for batch in dl_eval:
                    Y_np = batch["Y"].squeeze(0).numpy()
                    eval_prompts.append(" ".join(tok_patch(Y_np)))
                    gt_angles.append(int(batch["angle_deg"]))
                    if args.max_samples and len(eval_prompts) >= args.max_samples:
                        break
                enc_eval = tokenizer(
                    eval_prompts,
                    padding=True,
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                )
                input_ids_prompt = enc_eval["input_ids"].to(device)
                allowed = set(direction_token_ids(tokenizer))
                if args.K < 1 or args.K > len(allowed):
                    raise RuntimeError(f"Invalid K={args.K}; valid range is [1, {len(allowed)}]")

                # Teacher sequences for recall@K
                teacher_dir_ids = compute_rm_greedy_teacher(
                    rm_model=rm_model,
                    tokenizer=tokenizer,
                    input_ids_prompt=input_ids_prompt,
                    K=args.K,
                    device=device,
                )

                # Top-1 scoring at t=0
                top1_hits = 0
                teacher_hits = 0
                for i in range(input_ids_prompt.size(0)):
                    row = input_ids_prompt[i]
                    patch_ids = get_patch_ids(tokenizer, row)
                    if not patch_ids:
                        raise RuntimeError("Eval prompt row has no patch tokens; cannot score")
                    cand_scores = {}
                    for cand in allowed:
                        cand_scores[cand] = float(rm_score_for_prefix(rm_model, tokenizer, [cand], patch_ids, device))
                    gt_angle = int(gt_angles[i])
                    gt_token = f"<D_{gt_angle:03d}>"
                    gt_id = tokenizer.convert_tokens_to_ids(gt_token)
                    if gt_id not in allowed:
                        raise RuntimeError(f"Ground-truth token '{gt_token}' not in allowed direction tokens")
                    best_id = max(cand_scores, key=cand_scores.get)
                    if best_id == gt_id:
                        top1_hits += 1
                    if gt_id in teacher_dir_ids[i]:
                        teacher_hits += 1
                total = input_ids_prompt.size(0)
                top1_acc = top1_hits / total
                recall_k = teacher_hits / total
                print({
                    "epoch": epoch,
                    "eval_top1_acc": float(top1_acc),
                    "eval_recall_at_K": float(recall_k),
                    "eval_samples": int(total),
                    "K": int(args.K),
                })
            finally:
                rm_model.train()

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
