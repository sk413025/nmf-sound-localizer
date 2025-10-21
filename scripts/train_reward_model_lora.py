#!/usr/bin/env python3
"""Train Reward Model with LoRA — Pairwise LTR (Bradley–Terry) using USM ŝ.

Flow:
- Physics teacher uses USM content spectrum ŝ(F,) with dictionary W(F,K)
  to score directions by ΔIS(d|∅) (IS‑OMP step0); higher is better.
- Supervision: pairwise Bradley–Terry on β·(pred_pos − pred_neg).

Guardrails (No‑fallback policy):
- Exact angle match (dataset ↔ TF asset); no nearest mapping
- Strict grid alignment: Y.F == H.F == W.F; band [freq_min, freq_max]
- ŝ must be non‑negative, finite, shape (F,); else raise

Diagnostics: minimal per‑sample JSONL with STFT grid and invariants.
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
from doa_rl.assets import load_H, load_W
from nmf_localizer.utils.audio_utils import AudioProcessor
from doa_rl.features.nmf_utils import estimate_s_hat_torch
# Reuse greedy eval helpers for directions‑first evaluation
from scripts.train_sft_policy_with_rm import (
    get_patch_ids,
)
from doa_rl.omp.is_omp import compute_deltais_step0



def _derive_content_path(content_root: str, y_path: str) -> Path:
    """Map a test sample path to its Original (content) counterpart.

    Enforces angle_/clip_ structure and .npy filename parity.
    """
    p = Path(y_path)
    if not p.name.endswith('.npy'):
        raise RuntimeError(f"Expected .npy file for Y; got: {y_path}")
    angle_dir = p.parent.name
    if not angle_dir.startswith('angle_'):
        raise RuntimeError(f"Y path does not reside in angle_* directory: {y_path}")
    s_path = Path(content_root) / angle_dir / p.name
    if not s_path.exists():
        raise RuntimeError(f"Content file not found: {s_path} (derived from {y_path})")
    return s_path


def _load_band_spectrogram_from_npy(
    npy_path: Path,
    fs: int,
    n_fft: int,
    freq_min: float,
    freq_max: float,
) -> torch.Tensor:
    """Load waveform .npy and compute band‑limited magnitude STFT (F,N)."""
    wav = np.load(str(npy_path))
    if wav.ndim != 1:
        raise RuntimeError(f"Expected mono waveform in {npy_path}")
    freqs, _, _, magnitude = AudioProcessor.compute_stft_spectrogram(
        wav, fs=fs, nperseg=n_fft, window='hann'
    )
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    mag_band = magnitude[mask, :].astype(np.float32)
    return torch.from_numpy(mag_band)


## Removed alternative teacher score helpers (fit/euc). OMP ΔIS only.


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
    """Compute RM score with gradient: MAX over patch positions (sharper margins).

    Sequence: [BOS] <D> + patch_ids (no EOS). Returns a scalar tensor.
    """
    bos = tokenizer.bos_token_id
    ids = [bos, int(dir_id)] + [int(x) for x in patch_ids]
    inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    attn = torch.ones_like(inp, device=device)
    out = rm_model.pretrained_model(input_ids=inp, attention_mask=attn, output_hidden_states=True, return_dict=True)
    vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)[0]
    patch_vals = vals[-len(patch_ids):] if len(patch_ids) > 0 else vals.new_tensor([0.0])
    return patch_vals.max()

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
    ap.add_argument("--content-root", type=str, required=True, help="Root of Original/content dataset (mirrors angle_/clip_ structure)")
    ap.add_argument("--K", type=int, default=3)
    # Teacher scoring (LTR): fixed to OMP ΔIS step0 (IS‑OMP)
    ap.add_argument("--bt-beta", type=float, default=1.0, help="Bradley–Terry temperature β for base pairs")
    # Hard-negative controls
    ap.add_argument("--hn-per-sample", type=int, default=1, help="Hard-negative pairs per sample per epoch (GT vs model top-N non-GT)")
    ap.add_argument("--hn-beta", type=float, default=1.0, help="β for hard-negative pairs (overrides --bt-beta for HN rows)")
    ap.add_argument("--hn-include-teacher", action="store_true", help="Also add teacher (top vs second/third) pairs to fill HN quota")
    ap.add_argument("--hn-only-wrong", action="store_true", help="Allocate hard negatives only to samples mispredicted by current RM (pre-eval)")
    # Always use all directions per sample (no subsampling)
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
    # No verbose progress printing; only report eval loss↔metric alignment
    
    ap.add_argument("--out", type=str, default="rm_ckpt_lora")
    ap.add_argument("--debug-info", action="store_true", help="Print tensor shapes and sample pred/target values")
    # Pairwise-only supervision (simplified)
    ap.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon for IS divergence and clamping")
    # Listwise disabled in this branch
    args = ap.parse_args()

    # Track epoch loss and eval metrics to report alignment of loss vs Top-1/Recall@K
    loss_history: List[Dict[str, object]] = []
    eval_history: List[Dict[str, object]] = []

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

    # Preflight removed to simplify codepath

    # Load W (USM dictionary) for ŝ estimation
    # Load W (USM dictionary) for ŝ estimation
    W_t = load_W(args.w_path).float().cpu()

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
    
    print("LoRA config:")
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
    
    # Omit verbose parameter table in simplified script

    # Build pairwise BT dataset from forward‑model teachers
    dir_token_ids_all: List[int] = list(direction_token_ids(tokenizer))
    d_count = len(dir_token_ids_all)
    # No reverse index needed in simplified trainer

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
    # Track training pairs per sample (by absolute source path) for confusor coverage checks
    pair_sets_by_path: Dict[str, set] = {}
    # Per-sample metadata for hard-negative mining
    sample_meta: Dict[int, Dict[str, object]] = {}
    for si, p in enumerate(prompts):
        Y = cache[p]["Y"].float()
        F, N = Y.shape
        y_src = cache[p].get("path", "")
        if not y_src:
            raise RuntimeError("Dataset did not provide source path for Y; cannot derive content counterpart")
        # Load corresponding content spectrogram and estimate ŝ via W
        content_path = _derive_content_path(args.content_root, y_src)
        Y_content = _load_band_spectrogram_from_npy(
            content_path,
            fs=args.sample_rate,
            n_fft=args.n_fft,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        ).float()
        if Y_content.shape != Y.shape:
            raise RuntimeError(
                f"Y_content shape {tuple(Y_content.shape)} must equal Y {tuple(Y.shape)}; align STFT grid (fs/n_fft/band)."
            )
        # Grid checks: H.F == Y.F and W.F == Y.F
        if H.shape[0] != F:
            raise RuntimeError(
                f"H.F ({int(H.shape[0])}) != Y.F ({int(F)}); check tf_path or STFT grid (fs/n_fft/band)."
            )
        if (W_t.dim() != 2) or (int(W_t.shape[0]) != int(F)):
            raise RuntimeError(
                f"W.F ({int(W_t.shape[0])}) != Y.F ({int(F)}); ensure USM W matches STFT grid."
            )
        # Estimate ŝ(F,) using torch twin (returns cpu tensors)
        s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=50, l1=0.0)
        if s_hat_t.shape[0] != F:
            raise RuntimeError(f"ŝ shape mismatch: got {tuple(s_hat_t.shape)} vs F={int(F)}")
        if not torch.isfinite(s_hat_t).all():
            raise RuntimeError("ŝ contains non-finite values; aborting")
        if (s_hat_t < 0).any():
            raise RuntimeError("ŝ contains negatives; aborting")
        # Candidate direction indices: use all directions
        d_indices = list(range(d_count))
        # Teacher scores: ΔIS(d|∅) via IS‑OMP step0
        ord_all, deltas_all = compute_deltais_step0(
            Y=Y.numpy().astype(np.float64),
            H=H.numpy().astype(np.float64),
            W=W_t.numpy().astype(np.float64),
            s_hat=s_hat_t.numpy().astype(np.float64),
            prefilter_M=None,
            mu_iter=10,
            baseline_k=2,
            eps=float(args.eps),
        )
        d_indices = ord_all
        delta_map = {ord_all[i]: deltas_all[i] for i in range(len(ord_all))}
        scores = [float(delta_map[idx]) for idx in d_indices]
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
        # Build supervision rows
        order = list(range(len(d_indices)))
        order.sort(key=lambda i: scores[i], reverse=True)
        top_k = order[: max(1, len(order)//4)]
        bot_k = order[-max(1, len(order)//4):]
        # Build pairwise supervision rows (top vs bottom)
        pairs: List[Tuple[int, int]] = []
        for a in top_k:
            for b in bot_k:
                if a == b or scores[a] == scores[b]:
                    continue
                pairs.append((a, b))
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
        # Derive GT from source path for coverage stats
        try:
            angle_dir = os.path.basename(os.path.dirname(y_src))
            gt_angle = int(angle_dir.split('_')[1])
        except Exception:
            gt_angle = -1
        gt_tok = tokenizer.convert_tokens_to_ids(f"<D_{gt_angle:03d}>") if gt_angle >= 0 else -1
        # Build a per-sample set of unordered token-id pairs for later coverage checks
        pair_set = set()
        gt_pairs_count = 0

        for (ai, bi) in pairs:
            d_pos = d_indices[ai]
            d_neg = d_indices[bi]
            t_pos = int(dir_token_ids_all[d_pos])
            t_neg = int(dir_token_ids_all[d_neg])
            pair_rows.append({
                "sample_index": si,
                "dir_pos": t_pos,
                "dir_neg": t_neg,
                "patch_len": int(len(patch_ids_list[si])),
                "beta": float(args.bt_beta),
            })
            # Update coverage structures
            pair_set.add((min(t_pos, t_neg), max(t_pos, t_neg)))
            if gt_tok >= 0 and (t_pos == gt_tok or t_neg == gt_tok):
                gt_pairs_count += 1

        # Record per-sample pair set by absolute path for eval-time coverage checks
        try:
            path_abs = os.path.abspath(str(y_src)) if y_src else ""
        except Exception:
            path_abs = str(y_src) if y_src else ""
        if path_abs:
            pair_sets_by_path[path_abs] = pair_set
        # Record per-sample meta
        # Teacher top-K token ids (first three) for optional HN teacher pairs
        teacher_sorted_dir_idxs = [d_indices[i] for i in order]
        teacher_top_token_ids = [int(dir_token_ids_all[j]) for j in teacher_sorted_dir_idxs[:3]]

        sample_meta[int(si)] = {
            "path_abs": path_abs,
            "gt_tok": int(gt_tok) if gt_tok >= 0 else None,
            "patch_len": int(len(patch_ids_list[si])),
            "teacher_top_ids": teacher_top_token_ids,
        }
        # Per-sample log
        deltas = np.array([scores[a] - scores[b] for (a, b) in pairs], dtype=float) if ('pairs' in locals() and pairs) else np.array([0.0])
        # Signal stats and baseline_k=2
        Y_np = Y.numpy()
        s_np = s_hat_t.numpy()
        Y_min, Y_mean, Y_max = float(np.min(Y_np)), float(np.mean(Y_np)), float(np.max(Y_np))
        s_min, s_mean, s_max = float(np.min(s_np)), float(np.mean(s_np)), float(np.max(s_np))
        try:
            Hs = (H * s_hat_t.view(-1, 1)).float().cpu().numpy()  # (F,D)
            k = 2
            part = np.partition(Hs, kth=k-1, axis=1)[:, :k]
            Y_base = np.maximum(np.sum(part, axis=1), float(args.eps))
            Y_base_exp = np.repeat(Y_base[:, None], N, axis=1)
            ratio = np.clip(Y_np / np.maximum(Y_base_exp, float(args.eps)), 1e-12, 1e12)
            ratio_p50 = float(np.percentile(ratio, 50))
            ratio_p95 = float(np.percentile(ratio, 95))
            ratio_p99 = float(np.percentile(ratio, 99))
            mix_base_min = float(np.min(Y_base))
            mix_base_mean = float(np.mean(Y_base))
            mix_base_max = float(np.max(Y_base))
        except Exception:
            k = 2
            ratio_p50 = ratio_p95 = ratio_p99 = 0.0
            mix_base_min = mix_base_mean = mix_base_max = 0.0
        log_row = {
            "path": os.path.basename(y_src) or "",
            "F": int(F),
            "N": int(N),
            "beta": float(args.bt_beta),
            "fs": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "freq_min": float(args.freq_min),
            "freq_max": float(args.freq_max),
            "eps": float(args.eps),
            "dirs_considered": int(len(d_indices)),
            "pairs": int(len(pairs)),
            "score_stats": stats(s_arr),
            "delta_stats": stats(deltas),
            "Y_min": Y_min, "Y_mean": Y_mean, "Y_max": Y_max,
            "s_hat_min": s_min, "s_hat_mean": s_mean, "s_hat_max": s_max,
            "baseline_k": int(k),
            "mix_base_min": mix_base_min,
            "mix_base_mean": mix_base_mean,
            "mix_base_max": mix_base_max,
            "ratio_base_p50": ratio_p50,
            "ratio_base_p95": ratio_p95,
            "ratio_base_p99": ratio_p99,
            "gt_angle": int(gt_angle) if gt_angle >= 0 else None,
            "gt_tok": int(gt_tok) if gt_tok >= 0 else None,
            "pairs_total": int(len(pairs)),
            "gt_pairs_count": int(gt_pairs_count),
            "gt_pair_ratio": float(gt_pairs_count / max(1, len(pairs))),
        }
        # No teacher softmax diagnostics in simplified trainer
        sample_logs.append(log_row)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

    print("Training data:")
    print(f"  - Samples: {len(prompts)}")
    avg_dirs = float(np.mean([r["dirs_considered"] for r in sample_logs])) if sample_logs else 0.0
    print(f"  - Directions/sample (avg): {avg_dirs:.1f}")
    print(f"  - Pairs total: {len(pair_rows)} (~{len(pair_rows)/max(1,len(prompts)):.1f}/sample)")
    print(f"Saved numeric diagnostics JSONL: {jsonl_path}")
    # No teacher softmax/Q summaries

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
    
    print("Optimizer:")
    print(f"  - LoRA LR:       {args.lr_lora}")
    print(f"  - Embedding LR:  {args.lr_embed}")
    print(f"  - V-head LR:     {args.lr_vhead}\n")
    
    bce = nn.BCEWithLogitsLoss()
    
    # Training loop
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
        denom = 0
        epoch_loss_name = "loss"
        epoch_loss_value = 0.0
        # Build hard-negative pairs for this epoch: (GT vs current model top non-GT)
        hard_rows: List[Dict[str, int]] = []
        allowed_ids = list(direction_token_ids(tokenizer))
        # Optional pre-eval to focus HN on current wrong samples
        consider_indices: List[int] = []
        cand_scores_cache: Dict[int, Dict[int, float]] = {}
        top_pred_cache: Dict[int, int] = {}
        wrong_set: set = set()
        for si in range(len(patch_ids_list)):
            meta = sample_meta.get(int(si), {})
            gt_tok = meta.get("gt_tok", None)
            if gt_tok is None:
                continue
            patch_ids = patch_ids_list[si]
            if not patch_ids:
                continue
            # Score all candidate directions with current RM
            cand_scores: Dict[int, float] = {}
            for cand in allowed_ids:
                cand_scores[int(cand)] = float(_rm_pred_score_train(rm_model, tokenizer, int(cand), patch_ids, device))
            cand_scores_cache[int(si)] = cand_scores
            best_id = max(cand_scores, key=cand_scores.get)
            top_pred_cache[int(si)] = int(best_id)
            if args.hn_only_wrong:
                if int(best_id) != int(gt_tok):
                    wrong_set.add(int(si))
            else:
                consider_indices.append(int(si))

        if args.hn_only_wrong:
            consider_indices = sorted(list(wrong_set))

        # Build HN rows for considered samples
        for si in consider_indices:
            meta = sample_meta.get(int(si), {})
            gt_tok = meta.get("gt_tok", None)
            if gt_tok is None:
                continue
            patch_ids = patch_ids_list[si]
            if not patch_ids:
                continue
            cand_scores = cand_scores_cache.get(int(si)) or {}
            ranked = [k for k,_ in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)]
            model_non_gt = [cid for cid in ranked if int(cid) != int(gt_tok)]
            hn_quota = max(0, int(args.hn_per_sample))
            # 1) GT vs model top-N non-GT
            for cid in model_non_gt[:hn_quota]:
                hard_rows.append({
                    "sample_index": int(si),
                    "dir_pos": int(gt_tok),
                    "dir_neg": int(cid),
                    "patch_len": int(len(patch_ids)),
                    "beta": float(args.hn_beta),
                })
            # 2) Optionally teacher top vs teacher second/third
            if args.hn_include_teacher and hn_quota > 0:
                teacher_ids = meta.get("teacher_top_ids", []) or []
                if len(teacher_ids) >= 2:
                    t1 = int(teacher_ids[0])
                    for tid in teacher_ids[1:3]:
                        hard_rows.append({
                            "sample_index": int(si),
                            "dir_pos": int(t1),
                            "dir_neg": int(int(tid)),
                            "patch_len": int(len(patch_ids)),
                            "beta": float(args.hn_beta),
                        })

        # Build epoch coverage sets including HN pairs
        pair_sets_by_path_epoch: Dict[str, set] = {k: set(v) for k, v in pair_sets_by_path.items()}
        for row in hard_rows:
            si = int(row.get("sample_index", -1))
            if si < 0:
                continue
            p_meta = sample_meta.get(int(si), {})
            p_abs = p_meta.get("path_abs", None)
            if not p_abs:
                continue
            up = (min(int(row["dir_pos"]), int(row["dir_neg"])), max(int(row["dir_pos"]), int(row["dir_neg"])))
            if p_abs not in pair_sets_by_path_epoch:
                pair_sets_by_path_epoch[p_abs] = set()
            pair_sets_by_path_epoch[p_abs].add(up)

        # Compose epoch rows: base pairs + hard negatives
        rows_epoch = list(pair_rows)
        rows_epoch.extend(hard_rows)

        # Log HN allocation stats for this epoch
        try:
            targeted = len(consider_indices)
            print({
                "epoch": epoch,
                "hard_negative_stats": {
                    "hn_rows": int(len(hard_rows)),
                    "targeted_samples": int(targeted),
                    "avg_hn_per_targeted": float(len(hard_rows) / max(1, targeted)),
                    "pairs_epoch_total": int(len(rows_epoch)),
                }
            })
        except Exception:
            pass

        # Pairwise BT training
        for batch in _iter_batches(rows_epoch, args.batch_size):
            # Build sequences for pos/neg
            seqs: List[List[int]] = []
            patch_lens: List[int] = []
            betas: List[float] = []
            for row in batch:
                si = row["sample_index"]
                patch_ids = patch_ids_list[si]
                seqs.append([tokenizer.bos_token_id, row["dir_pos"], *patch_ids])
                seqs.append([tokenizer.bos_token_id, row["dir_neg"], *patch_ids])
                patch_lens.extend([len(patch_ids), len(patch_ids)])
                betas.append(float(row.get("beta", float(args.bt_beta))))
            inp = _pad_to_batch(seqs, tokenizer.pad_token_id).to(device)
            attn = (inp != tokenizer.pad_token_id).to(device)
            out = rm_model.pretrained_model(
                input_ids=inp,
                attention_mask=attn,
                output_hidden_states=True,
                return_dict=True,
            )
            vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)  # (2B, S)
            pooled_vals: List[torch.Tensor] = []
            for r in range(vals.size(0)):
                plen = patch_lens[r]
                seqlen = int(attn[r].sum().item())
                if plen == 0:
                    pooled_vals.append(vals[r, seqlen-1:seqlen].max())
                else:
                    pooled_vals.append(vals[r, seqlen-plen:seqlen].max())
            pooled_t = torch.stack(pooled_vals, dim=0)
            pos_pred = pooled_t[0::2]
            neg_pred = pooled_t[1::2]
            beta_vec = torch.tensor(betas, dtype=pos_pred.dtype, device=pos_pred.device)
            logits = beta_vec * (pos_pred - neg_pred)
            target = torch.ones_like(logits)
            loss = bce(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            denom += len(batch)
        avg_loss = total_loss / max(1, denom)
        epoch_loss_name = "bt_pair_loss"
        epoch_loss_value = float(avg_loss)
        print({"epoch": epoch, epoch_loss_name: epoch_loss_value, "pairs": int(denom)})

        # Record epoch loss for downstream correlation with eval metrics
        try:
            loss_history.append({"epoch": int(epoch), "name": epoch_loss_name, "value": float(epoch_loss_value)})
        except Exception:
            pass

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
                tok_patch = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
                eval_prompts: List[str] = []
                gt_angles: List[int] = []
                eval_records: List[Dict[str, object]] = []
                # Deterministic: reuse existing manifest if present; else create once from dataloader
                results_dir = os.path.join("results", f"{args.out}")
                os.makedirs(results_dir, exist_ok=True)
                manifest_path = os.path.join(results_dir, "eval_subset_manifest.json")
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as mf:
                            manifest = json.load(mf)
                        files = manifest.get("files", [])
                        for rec in files:
                            pth_abs = rec.get("path_abs") or rec.get("path") or ""
                            angle = rec.get("angle_deg")
                            if not pth_abs or angle is None:
                                continue
                            Y_np = _load_band_spectrogram_from_npy(Path(pth_abs), fs=args.sample_rate, n_fft=args.n_fft, freq_min=args.freq_min, freq_max=args.freq_max).numpy()
                            eval_prompts.append(" ".join(tok_patch(Y_np)))
                            gt_angles.append(int(angle))
                            eval_records.append({"path_abs": os.path.abspath(pth_abs), "angle_deg": int(angle)})
                    except Exception:
                        eval_prompts = []
                        eval_records = []
                        gt_angles = []
                if not eval_prompts:
                    dl_eval = create_dataloader(ds_eval, batch_size=1, shuffle=False)
                    for batch_eval in dl_eval:
                        Y_np = batch_eval["Y"].squeeze(0).numpy()
                        eval_prompts.append(" ".join(tok_patch(Y_np)))
                        gt_a = int(batch_eval["angle_deg"]) if "angle_deg" in batch_eval else None
                        gt_angles.append(int(gt_a) if gt_a is not None else 0)
                        pth = batch_eval.get("path", "")
                        if isinstance(pth, (list, tuple)):
                            pth = pth[0] if pth else ""
                        try:
                            pth_abs = os.path.abspath(str(pth)) if pth else ""
                        except Exception:
                            pth_abs = str(pth) if pth else ""
                        rec = {"path_abs": pth_abs, "angle_deg": int(gt_a) if gt_a is not None else None}
                        eval_records.append(rec)
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
                # Compute Top-1 and Recall@K directly from RM scores (no greedy teacher)
                top1_hits = 0
                teacher_hits = 0
                # Confusor coverage counters (for wrong samples)
                wrong_cnt = 0
                covered_cnt = 0
                for i in range(input_ids_prompt.size(0)):
                    row = input_ids_prompt[i]
                    patch_ids = get_patch_ids(tokenizer, row)
                    if not patch_ids:
                        raise RuntimeError("Eval prompt row has no patch tokens; cannot score")
                    cand_scores = {}
                    for cand in allowed:
                        # Use local max-pooling scorer for sharper margins
                        cand_scores[cand] = float(_rm_pred_score_train(rm_model, tokenizer, int(cand), patch_ids, device))
                    gt_angle = int(gt_angles[i])
                    gt_token = f"<D_{gt_angle:03d}>"
                    gt_id = tokenizer.convert_tokens_to_ids(gt_token)
                    if gt_id not in allowed:
                        raise RuntimeError(f"Ground-truth token '{gt_token}' not in allowed direction tokens")
                    # Top-1 by RM scores
                    best_id = max(cand_scores, key=cand_scores.get)
                    if best_id == gt_id:
                        top1_hits += 1
                    # Recall@K by RM scores
                    topk_ids = [k for k,_ in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:int(args.K)]]
                    if gt_id in topk_ids:
                        teacher_hits += 1
                    # Confusor coverage: if wrong, check if (gt, best) pair existed in training pairs for this sample
                    # Map back to training sample via absolute path (manifest stores path_abs)
                    p_abs = eval_records[i].get("path_abs") if i < len(eval_records) else None
                    if p_abs and best_id != gt_id:
                        wrong_cnt += 1
                        # Use epoch-inclusive coverage (base + current HN)
                        ps = pair_sets_by_path_epoch.get(p_abs, set())
                        upair = (min(int(gt_id), int(best_id)), max(int(gt_id), int(best_id)))
                        if upair in ps:
                            covered_cnt += 1
                    # No noisy eval progress spam
                total = input_ids_prompt.size(0)
                top1_acc = top1_hits / max(1, total)
                recall_k = teacher_hits / max(1, total)
                # Record eval
                eval_rec = {"epoch": int(epoch), "top1": float(top1_acc), "recallK": float(recall_k), "K": int(args.K), "samples": int(total)}
                prev_eval = eval_history[-1] if len(eval_history) > 0 else None
                eval_history.append(eval_rec)
                # Compute deltas and alignment
                cur_loss = loss_history[-1]["value"] if len(loss_history) > 0 else None
                prev_loss = loss_history[-2]["value"] if len(loss_history) > 1 else None
                d_top1 = (eval_rec["top1"] - prev_eval["top1"]) if prev_eval else 0.0
                d_recall = (eval_rec["recallK"] - prev_eval["recallK"]) if prev_eval else 0.0
                d_loss = (float(cur_loss) - float(prev_loss)) if (cur_loss is not None and prev_loss is not None) else 0.0
                aligned = (d_loss < 0.0) and ((d_top1 > 0.0) or (d_recall > 0.0)) if prev_eval and prev_loss is not None else None
                # Emit raw eval and correlation report
                print({
                    "epoch": epoch,
                    "eval_top1_acc": float(top1_acc),
                    "eval_recall_at_K": float(recall_k),
                    "eval_samples": int(total),
                    "K": int(args.K),
                })
                # Print confusor coverage summary for this epoch
                try:
                    ratio = float(covered_cnt) / float(max(1, wrong_cnt))
                except Exception:
                    ratio = 0.0
                print({
                    "epoch": epoch,
                    "confusor_coverage": {
                        "wrong_samples": int(wrong_cnt),
                        "covered": int(covered_cnt),
                        "covered_ratio": float(ratio),
                    }
                })
                print({
                    "epoch": epoch,
                    "progress_report": {
                        "loss_name": epoch_loss_name,
                        "loss": float(cur_loss) if cur_loss is not None else None,
                        "loss_delta": float(d_loss),
                        "top1": float(eval_rec["top1"]),
                        "top1_delta": float(d_top1),
                        "recall_at_K": float(eval_rec["recallK"]),
                        "recall_delta": float(d_recall),
                        "aligned_loss_vs_metrics": (bool(aligned) if aligned is not None else None),
                    }
                })
                # Persist deterministic eval manifest
                try:
                    with open(manifest_path, "w") as mf:
                        json.dump({"files": eval_records}, mf, indent=2)
                    print({"epoch": epoch, "eval_manifest": manifest_path, "eval_records": int(len(eval_records))})
                except Exception as _e:
                    print(f"Warning: failed to write eval manifest: {_e}")

                # OMP alignment diagnostics removed (simplification)
            finally:
                rm_model.train()

    print("\nTraining completed!")
    
    # Save LoRA adapters and heads under results/<out>/
    results_dir = os.path.join("results", f"{args.out}")
    os.makedirs(results_dir, exist_ok=True)
    lora_dir = os.path.join(results_dir, "adapters")
    heads_path = os.path.join(results_dir, "heads.pt")

    rm_model.pretrained_model.save_pretrained(lora_dir)
    torch.save({
        "embeddings": embedding_layer.state_dict(),
        "v_head": rm_model.v_head.state_dict(),
        "config": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "teacher": "omp",
            "bt_beta": float(args.bt_beta),
        }
    }, heads_path)
    
    print("\nSaved:")
    print(f"  - LoRA adapters: {lora_dir}/")
    print(f"  - Embeddings + V-head: {heads_path}")


if __name__ == "__main__":
    main()
