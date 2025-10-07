#!/usr/bin/env python3
"""Train Reward Model with LoRA — Pairwise Bradley–Terry (LTR: fit/EUC).

Switches RM training to Learning‑to‑Rank with pairwise Bradley–Terry loss using
forward‑model teachers from docs/LTR.md:

  Y_hat(d) = diag(H_d) * S,   s_d = -D(Y | Y_hat(d))   (fit=IS default; euc optional)

We generate labeled pairs (d_pos ≻ d_neg) per sample and optimize BCEWithLogits
on β*(pred_pos − pred_neg), where pred is the mean v_head over patch token
positions for sequences “[BOS] <D_d> + patches(Y)”.

Guardrails:
- Exact angle match (dataset ↔ TF asset); no nearest fallback
- Strict grid alignment: Y.F == S.F == H.F; band [freq_min, freq_max]
- Fail fast on duplicates or shape mismatches

Diagnostics: Write per‑sample score/pair stats to results/<out>/numeric_diagnostics.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence

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
from doa_rl.assets import load_H
from nmf_localizer.utils.audio_utils import AudioProcessor
# Reuse greedy eval helpers for directions‑first evaluation
from scripts.train_sft_policy_with_rm import (
    rm_score_for_prefix,
    get_patch_ids,
    compute_rm_greedy_teacher,
)



def _load_S_for_Y_path(
    s_root: str,
    y_path: str,
    fs: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    """Load counterpart waveform from S‑root and compute band‑limited magnitude STFT (F,N)."""
    p = Path(y_path)
    if not p.name.endswith('.npy'):
        raise RuntimeError(f"Expected .npy file for Y path; got: {y_path}")
    angle_dir = p.parent.name
    if not angle_dir.startswith('angle_'):
        raise RuntimeError(f"Y path does not reside in angle_* directory: {y_path}")
    s_path = Path(s_root) / angle_dir / p.name
    if not s_path.exists():
        raise RuntimeError(f"S file not found: {s_path} (derived from {y_path})")
    wav = np.load(s_path)
    assert wav.ndim == 1, f"Expected mono waveform in S: {s_path}"
    freqs, times, stft, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=n_fft, window='hann'
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


def _is_divergence(Y: torch.Tensor, Yhat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    Yc = torch.clamp(Y, min=eps)
    Yhc = torch.clamp(Yhat, min=eps)
    ratio = Yc / Yhc
    return torch.sum(ratio - torch.log(torch.clamp(ratio, min=eps)) - 1.0)


def _teacher_scores(
    Y: torch.Tensor,
    S: torch.Tensor,
    H: torch.Tensor,
    dir_indices: Sequence[int],
    teacher: str = "fit",
    eps: float = 1e-12,
) -> List[float]:
    """Compute s_d for each direction index in dir_indices.

    - fit (IS): s_d = -IS(Y || diag(H_d) * S)
    - euc:      s_d = -||Y - diag(H_d) * S||^2
    """
    F, N = Y.shape
    if S.shape != Y.shape:
        raise RuntimeError(f"Y and S shapes must match. Got Y={tuple(Y.shape)} S={tuple(S.shape)}")
    if H.shape[0] != F:
        raise RuntimeError(f"H.F must equal Y.F. Got H.F={int(H.shape[0])} vs Y.F={int(F)}")
    Yc = torch.clamp(Y.float(), min=eps)
    Sc = torch.clamp(S.float(), min=eps)
    scores: List[float] = []
    for d in dir_indices:
        Hd = torch.clamp(H[:, d], min=eps)
        Yhat = Hd.view(-1, 1) * Sc
        if teacher == "fit":
            val = -_is_divergence(Yc, Yhat, eps=eps).item()
        elif teacher == "euc":
            diff = (Yc - Yhat)
            val = -float(torch.sum(diff * diff).item())
        else:
            raise ValueError(f"Unknown teacher: {teacher}")
        scores.append(val)
    return scores


def _pad_to_batch(rows: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(r) for r in rows)
    out = []
    for r in rows:
        out.append([pad_id] * (max_len - len(r)) + r)
    return torch.tensor(out, dtype=torch.long)


def _rm_pred_score_train(
    rm_model,
    tokenizer,
    dir_id: int,
    patch_ids: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Compute RM score with gradient: mean v_head over patch positions.

    Sequence: [BOS] <D> + patch_ids (no EOS). Returns a scalar tensor.
    """
    bos = tokenizer.bos_token_id
    ids = [bos, int(dir_id)] + [int(x) for x in patch_ids]
    inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(inp, device=device)
    out = rm_model.pretrained_model(input_ids=inp, attention_mask=attn, output_hidden_states=True, return_dict=True)
    vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)[0]
    patch_vals = vals[-len(patch_ids):] if len(patch_ids) > 0 else vals.new_tensor([0.0])
    return patch_vals.mean()

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
    ap.add_argument("--s-root", type=str, required=True, help="Root for S dataset (mirrors Y structure)")
    ap.add_argument("--K", type=int, default=3)
    # Teacher scoring (LTR)
    ap.add_argument("--teacher", type=str, choices=["fit", "euc"], default="fit", help="Teacher score (docs/LTR.md §1/§3)")
    ap.add_argument("--bt-beta", type=float, default=1.0, help="Bradley–Terry temperature β for logits scaling")
    ap.add_argument("--directions-per-sample", type=int, default=0, help="0=all directions; else random subset size")
    ap.add_argument("--pairs-per-sample", type=int, default=64, help="Max pairwise examples per sample")
    
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

    # Note: ŝ/W unused in LTR pairwise training; kept CLI for compatibility

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

    # Build pairwise BT dataset from forward‑model teachers
    dir_token_ids_all: List[int] = list(direction_token_ids(tokenizer))
    d_count = len(dir_token_ids_all)
    tokenid_to_dindex = {tid: i for i, tid in enumerate(dir_token_ids_all)}

    results_dir = os.path.join("results", f"{args.out}")
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = os.path.join(results_dir, "numeric_diagnostics.jsonl")
    sample_logs: List[Dict[str, float]] = []

    # Pre-tokenize patch prompts into token ids (no specials)
    enc_patch = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors=None,
    )
    specials = {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
    if isinstance(enc_patch["input_ids"], list):
        patch_ids_list = [[tid for tid in row if tid not in specials] for row in enc_patch["input_ids"]]
    else:
        patch_ids_list = [[tid for tid in row.tolist() if tid not in specials] for row in enc_patch["input_ids"]]

    pair_rows: List[Dict[str, int]] = []
    for si, p in enumerate(prompts):
        Y = cache[p]["Y"].float()
        F, N = Y.shape
        y_src = cache[p].get("path", "")
        if not y_src:
            raise RuntimeError("Dataset did not provide source path for Y; cannot derive S counterpart")
        S = _load_S_for_Y_path(
            s_root=args.s_root,
            y_path=y_src,
            fs=args.sample_rate,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        ).float()
        if S.shape != Y.shape:
            raise RuntimeError(
                f"Y and S shapes must match per sample. Y={tuple(Y.shape)} S={tuple(S.shape)} path={y_src}"
            )
        # Candidate direction indices
        if args.directions_per_sample and args.directions_per_sample > 0:
            sel_idx = np.random.choice(d_count, size=min(args.directions_per_sample, d_count), replace=False)
            d_indices = [int(i) for i in sel_idx]
        else:
            d_indices = list(range(d_count))
        # Teacher scores
        scores = _teacher_scores(Y=Y, S=S, H=H, dir_indices=d_indices, teacher=args.teacher)
        s_arr = np.asarray(scores, dtype=float)
        def stats(x):
            return {
                "min": float(np.min(x)),
                "median": float(np.median(x)),
                "mean": float(np.mean(x)),
                "p95": float(np.percentile(x, 95)),
                "p99": float(np.percentile(x, 99)),
                "max": float(np.max(x)),
            }
        # Build pairs emphasizing margins
        order = list(range(len(d_indices)))
        order.sort(key=lambda i: scores[i], reverse=True)
        top_k = order[: max(1, len(order)//4)]
        bot_k = order[-max(1, len(order)//4):]
        pairs: List[Tuple[int, int]] = []
        for a in top_k:
            for b in bot_k:
                if a == b or scores[a] == scores[b]:
                    continue
                pairs.append((a, b))
        # Fill with random distinct pairs if needed
        rng = np.random.default_rng(seed=si + args.seed)
        while len(pairs) < args.pairs_per_sample and len(pairs) < len(order)*(len(order)-1):
            i, j = rng.integers(0, len(order), size=2)
            if i == j or scores[i] == scores[j]:
                continue
            if scores[i] > scores[j]:
                pairs.append((i, j))
            else:
                pairs.append((j, i))
        if len(pairs) > args.pairs_per_sample:
            pairs = pairs[: args.pairs_per_sample]
        # Emit rows
        for (ai, bi) in pairs:
            d_pos = d_indices[ai]
            d_neg = d_indices[bi]
            pair_rows.append({
                "sample_index": si,
                "dir_pos": int(dir_token_ids_all[d_pos]),
                "dir_neg": int(dir_token_ids_all[d_neg]),
                "patch_len": int(len(patch_ids_list[si])),
            })
        # Per-sample log
        deltas = np.array([scores[a] - scores[b] for (a, b) in pairs], dtype=float) if pairs else np.array([0.0])
        log_row = {
            "path": os.path.basename(y_src) or "",
            "F": int(F),
            "N": int(N),
            "teacher": args.teacher,
            "beta": float(args.bt_beta),
            "dirs_considered": int(len(d_indices)),
            "pairs": int(len(pairs)),
            "score_stats": stats(s_arr),
            "delta_stats": stats(deltas),
        }
        sample_logs.append(log_row)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

    print("Training data (pairwise BT):")
    print(f"  - Samples: {len(prompts)}")
    avg_dirs = float(np.mean([r["dirs_considered"] for r in sample_logs])) if sample_logs else 0.0
    print(f"  - Directions/sample (avg): {avg_dirs:.1f}")
    print(f"  - Pairs total: {len(pair_rows)} (~{len(pair_rows)/max(1,len(prompts)):.1f}/sample)\n")
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
    
    bce = nn.BCEWithLogitsLoss()
    
    # Training loop (pairwise BT)
    print("=" * 60)
    print("Training")
    print("=" * 60)

    def _iter_batches(rows: List[Dict[str, int]], bs: int):
        step = max(1, bs)
        for i in range(0, len(rows), step):
            yield rows[i:i+step]

    for epoch in range(args.rm_epochs):
        rm_model.train()
        total_loss = 0.0
        total_pairs = 0
        for batch in _iter_batches(pair_rows, args.batch_size):
            # Build sequences for pos/neg
            seqs: List[List[int]] = []
            patch_lens: List[int] = []
            for row in batch:
                si = row["sample_index"]
                patch_ids = patch_ids_list[si]
                seqs.append([tokenizer.bos_token_id, row["dir_pos"], *patch_ids])
                seqs.append([tokenizer.bos_token_id, row["dir_neg"], *patch_ids])
                patch_lens.extend([len(patch_ids), len(patch_ids)])
            inp = _pad_to_batch(seqs, tokenizer.pad_token_id).to(device)
            attn = (inp != tokenizer.pad_token_id).to(device)
            out = rm_model.pretrained_model(
                input_ids=inp,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )
            vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)  # (2B, S)
            means: List[torch.Tensor] = []
            for r in range(vals.size(0)):
                plen = patch_lens[r]
                seqlen = int(attn[r].sum().item())
                if plen == 0:
                    means.append(vals[r, seqlen-1:seqlen].mean())
                else:
                    means.append(vals[r, seqlen-plen:seqlen].mean())
            means_t = torch.stack(means, dim=0)
            pos_pred = means_t[0::2]
            neg_pred = means_t[1::2]
            logits = float(args.bt_beta) * (pos_pred - neg_pred)
            target = torch.ones_like(logits)
            loss = bce(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_pairs += len(batch)
        avg_loss = total_loss / max(1, total_pairs)
        print({"epoch": epoch, "bt_pair_loss": float(avg_loss), "pairs": int(total_pairs)})

        # Evaluation: directions-first Top-1 and recall@K
        do_eval = (args.eval_every > 0) and (((epoch + 1) % args.eval_every == 0) or (epoch == args.rm_epochs - 1))
        if do_eval:
            rm_model.eval()
            try:
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
                for batch_eval in dl_eval:
                    Y_np = batch_eval["Y"].squeeze(0).numpy()
                    eval_prompts.append(" ".join(tok_patch(Y_np)))
                    gt_angles.append(int(batch_eval["angle_deg"]))
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
                teacher_dir_ids = compute_rm_greedy_teacher(
                    rm_model=rm_model,
                    tokenizer=tokenizer,
                    input_ids_prompt=input_ids_prompt,
                    K=args.K,
                    device=device,
                )
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
                top1_acc = top1_hits / max(1, total)
                recall_k = teacher_hits / max(1, total)
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
            "teacher": args.teacher,
            "bt_beta": float(args.bt_beta),
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
