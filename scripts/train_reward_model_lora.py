#!/usr/bin/env python3
"""Train Reward Model with LoRA — LTR (pairwise/pointwise) using USM ŝ.

Conceptual alignment with docs/rm.md while preserving project data flow:
- Physics teacher uses USM content spectrum ŝ(F,) computed from real data and a
  pre-trained dictionary W(F,K): Y_hat(d) = diag(H_d) ⊙ ŝ, score s_d = -D_IS(Y || Y_hat(d))
  (or -||Y - Y_hat(d)||^2 for euc). No S(F,N) waveform teacher or fallback.

Supervision modes:
- pairwise (default): Bradley–Terry on β·(pred_pos − pred_neg)
- pointwise: MSE on teacher scores per direction

Guardrails (No‑fallback policy):
- Exact angle match (dataset ↔ TF asset); no nearest mapping
- Strict grid alignment: Y.F == H.F == W.F; band [freq_min, freq_max]
- ŝ must be non‑negative, finite, shape (F,); else raise

Diagnostics: per‑sample JSONL under results/<out>/numeric_diagnostics.jsonl
including STFT grid, eps, signal stats (Y/ŝ), and teacher score stats.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Sequence
import subprocess, sys
import time

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
    rm_score_for_prefix,
    get_patch_ids,
    compute_rm_greedy_teacher,
)
from doa_rl.omp.is_omp import compute_deltais_step0, is_omp_select



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


def _is_divergence(Y: torch.Tensor, Yhat: torch.Tensor, eps: float) -> torch.Tensor:
    """Itakura–Saito divergence with shared eps; supports vector Yhat.

    Y: (F,N), Yhat: (F,N) or (F,). If (F,), it will be broadcast across time.
    """
    Yc = torch.clamp(Y, min=eps)
    if Yhat.dim() == 1:
        Yhc = torch.clamp(Yhat.view(-1, 1).expand_as(Yc), min=eps)
    else:
        Yhc = torch.clamp(Yhat, min=eps)
    ratio = Yc / Yhc
    return torch.sum(ratio - torch.log(torch.clamp(ratio, min=eps)) - 1.0)


def _teacher_scores(
    Y: torch.Tensor,
    s_hat: torch.Tensor,
    H: torch.Tensor,
    dir_indices: Sequence[int],
    teacher: str = "fit",
    eps: float = 1e-8,
) -> List[float]:
    """Compute s_d for each direction index using USM ŝ:

    - fit (IS): s_d = -IS(Y || diag(H_d) ⊙ ŝ)
    - euc:      s_d = -||Y - diag(H_d) ⊙ ŝ||^2
    """
    F, N = Y.shape
    if H.shape[0] != F:
        raise RuntimeError(f"H.F must equal Y.F. Got H.F={int(H.shape[0])} vs Y.F={int(F)}")
    if s_hat.dim() != 1 or int(s_hat.shape[0]) != int(F):
        raise RuntimeError(f"ŝ must be (F,), got {tuple(s_hat.shape)} vs F={int(F)}")
    Yc = torch.clamp(Y.float(), min=eps)
    sh = torch.clamp(s_hat.float(), min=eps)
    scores: List[float] = []
    for d in dir_indices:
        Hd = torch.clamp(H[:, d], min=eps)
        Yhat_vec = Hd * sh  # (F,)
        if teacher == "fit":
            val = -_is_divergence(Yc, Yhat_vec, eps=eps).item()
        elif teacher == "euc":
            diff = (Yc - Yhat_vec.view(-1, 1).expand_as(Yc))
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
    ap.add_argument("--content-root", type=str, required=True, help="Root of Original/content dataset (mirrors angle_/clip_ structure)")
    ap.add_argument("--K", type=int, default=3)
    # Teacher scoring (LTR)
    ap.add_argument("--teacher", type=str, choices=["omp", "fit", "euc"], default="omp", help="Teacher: omp=ΔIS step0 (IS‑OMP), fit=−IS, euc=−L2")
    ap.add_argument("--eval-omp-align", action="store_true", help="Enable OMP alignment metrics during eval (Intersection@K, Spearman vs ΔIS)")
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
    ap.add_argument("--progress-every", type=int, default=16, help="Print progress every N rows/samples (0=off)")
    
    ap.add_argument("--out", type=str, default="rm_ckpt_lora")
    ap.add_argument("--debug-info", action="store_true", help="Print tensor shapes and sample pred/target values")
    ap.add_argument("--supervision", type=str, choices=["pairwise", "pointwise"], default="pairwise",
                    help="LTR supervision: pairwise BT or pointwise regression (listwise disabled in this branch)")
    ap.add_argument("--eps", type=float, default=1e-8, help="Numerical epsilon for IS divergence and clamping")
    # Listwise disabled in this branch
    # Optional preflight evaluator (detection-only) to gate training
    ap.add_argument("--preflight", action="store_true", help="Run physics-teacher alignment evaluator and gate training")
    ap.add_argument("--preflight-max-samples", type=int, default=64)
    ap.add_argument("--preflight-directions-per-sample", type=int, default=0)
    ap.add_argument("--preflight-strict-teacher-top1", type=float, default=0.7)
    ap.add_argument("--preflight-strict-recallK", type=float, default=0.9)
    ap.add_argument("--preflight-strict-h-top1", type=float, default=0.7)
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

    # Optional: run detection-only preflight to ensure teacher→DoA alignment and H validity
    if args.preflight:
        pf_out = f"{args.out}_preflight"
        cmd = [
            sys.executable, "scripts/eval/eval_physics_teacher_alignment.py",
            "--data-root", args.data_root,
            "--content-root", args.content_root,
            "--tf-path", args.tf_path,
            "--w-path", args.w_path,
            "--teacher", args.teacher,
            "--K", str(args.K),
            "--max-samples", str(args.preflight_max_samples),
            "--directions-per-sample", str(args.preflight_directions_per_sample),
            "--eps", str(args.eps),
            "--strict-teacher-top1", str(args.preflight_strict_teacher_top1),
            "--strict-recallK", str(args.preflight_strict_recallK),
            "--strict-h-top1", str(args.preflight_strict_h_top1),
            "--out", pf_out,
        ]
        print("\n[Preflight] Running:", " ".join(cmd))
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # Echo evaluator output for visibility
        print(res.stdout)
        if res.returncode != 0:
            raise RuntimeError(
                f"Preflight failed (exit {res.returncode}). See results/{pf_out}/summary.json for details."
            )
        # Print concise summary if available
        try:
            with open(Path("results")/pf_out/"summary.json", "r") as sf:
                summary = json.load(sf)
            print("[Preflight] Teacher alignment:",
                  f"top1={summary.get('teacher_top1_acc'):.3f}",
                  f"recall@K={summary.get('teacher_recall_at_K'):.3f}",
                  f"median_rank={summary.get('teacher_median_rank')}")
        except Exception:
            pass

    # Load W (USM dictionary) for ŝ estimation
    if args.w_path.endswith('.npz'):
        W_np = np.load(args.w_path)["W"].astype(np.float32)
        W_t = torch.from_numpy(W_np)
    else:
        W_t = load_W(args.w_path).float().cpu()
        W_np = W_t.cpu().numpy().astype(np.float32)

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
    point_rows: List[Dict[str, int]] = []
    list_rows: List[Dict[str, object]] = []
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
        # Candidate direction indices
        if args.directions_per_sample and args.directions_per_sample > 0:
            sel_idx = np.random.choice(d_count, size=min(args.directions_per_sample, d_count), replace=False)
            d_indices = [int(i) for i in sel_idx]
        else:
            d_indices = list(range(d_count))
        # Teacher scores
        if args.teacher == "omp":
            # Use ΔIS(d|S=∅) via IS‑OMP step0; returns sorted indices and deltas
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
            # Respect directions_per_sample if set
            if args.directions_per_sample and args.directions_per_sample > 0:
                ord_use = ord_all[: min(int(args.directions_per_sample), len(ord_all))]
            else:
                ord_use = ord_all
            d_indices = ord_use
            # Map deltas for selected indices
            delta_map = {ord_all[i]: deltas_all[i] for i in range(len(ord_all))}
            scores = [float(delta_map[idx]) for idx in d_indices]
        else:
            # −IS / −L2 teacher
            scores = _teacher_scores(
                Y=Y, s_hat=s_hat_t, H=H, dir_indices=d_indices, teacher=args.teacher, eps=float(args.eps)
            )
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
        teacher_diag = {}
        if args.supervision == "pairwise":
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
        elif args.supervision == "pointwise":
            for idx_local, d_local in enumerate(d_indices):
                point_rows.append({
                    "sample_index": si,
                    "dir_id": int(dir_token_ids_all[d_local]),
                    "score": float(scores[idx_local]),
                    "patch_len": int(len(patch_ids_list[si])),
                })
        else:  # listwise (DoA-aligned)
            # Parse GT angle from path for logging and angle-based fallback
            try:
                angle_dir = os.path.basename(os.path.dirname(y_src))
                gt_angle = int(angle_dir.split('_')[1])
            except Exception:
                raise RuntimeError(f"Cannot parse ground-truth angle from path: {y_src}")
            # Listwise distribution: default now uses ΔIS teacher (or fallback to angle-based if teacher!=omp)
            if args.teacher == "omp":
                # Softmax over ΔIS/τ on selected candidates
                tau = max(1e-6, float(args.listwise_tau_deg))
                # token ids and angles for selected direction indices (needed for logging/rows)
                cand_token_ids = [int(dir_token_ids_all[j]) for j in d_indices]
                cand_tokens = tokenizer.convert_ids_to_tokens(cand_token_ids)
                cand_angles: List[int] = []
                for tok in cand_tokens:
                    try:
                        deg = int(tok.strip('<>').split('_')[1])
                    except Exception:
                        raise RuntimeError(f"Invalid direction token format: {tok}")
                    cand_angles.append(deg)
                deltas_sel = np.asarray([scores[d_indices.index(j)] for j in d_indices], dtype=float)
                # Optional normalization to avoid softmax saturation
                if args.listwise_normalize == "per_bin":
                    # ΔIS averaged per time-frequency bin
                    denom = max(1.0, float(F) * float(N))
                    deltas_use = deltas_sel / denom
                else:
                    deltas_use = deltas_sel
                logits = deltas_use / tau
                z = logits - np.max(logits)
                e = np.exp(z)
                P = (e / np.sum(e)).astype(float).tolist()
            else:
                # Angle‑based teacher (previous behavior)
                tol = float(args.listwise_tol_deg)
                tau = float(args.listwise_tau_deg)
                try:
                    angle_dir = os.path.basename(os.path.dirname(y_src))
                    gt_angle = int(angle_dir.split('_')[1])
                except Exception:
                    raise RuntimeError(f"Cannot parse ground-truth angle from path: {y_src}")
                cand_token_ids = [int(dir_token_ids_all[j]) for j in d_indices]
                cand_tokens = tokenizer.convert_ids_to_tokens(cand_token_ids)
                cand_angles: List[int] = []
                for tok in cand_tokens:
                    try:
                        deg = int(tok.strip('<>').split('_')[1])
                    except Exception:
                        raise RuntimeError(f"Invalid direction token format: {tok}")
                    cand_angles.append(deg)
                def circ_dist(a, b):
                    d = abs(a - b) % 360
                    return min(d, 360 - d)
                weights = []
                for a in cand_angles:
                    d = circ_dist(a, gt_angle)
                    w = 1.0 if d <= tol else np.exp(-d / max(tau, 1e-6))
                    weights.append(w)
                w_arr = np.asarray(weights, dtype=float)
                if not np.isfinite(w_arr).all() or np.all(w_arr <= 0):
                    raise RuntimeError("Listwise teacher produced invalid weights")
                P = (w_arr / np.sum(w_arr)).astype(float).tolist()
            # Compute teacher softmax diagnostics for numeric logging (listwise only)
            teacher_diag = {}
            try:
                if args.supervision == "listwise":
                    if args.teacher == "omp":
                        # z, P were computed above
                        z_arr = np.asarray(z, dtype=float) if 'z' in locals() else np.asarray([], dtype=float)
                        P_arr = np.asarray(P, dtype=float) if 'P' in locals() else np.asarray([], dtype=float)
                    else:
                        # For angle-based weights, derive pseudo-logits from weights to inspect concentration
                        # Avoid log(0): clamp
                        P_arr = np.asarray(P, dtype=float)
                        z_arr = np.log(np.clip(P_arr, 1e-12, 1.0))
                        z_arr = z_arr - np.max(z_arr)
                    p_max = float(P_arr.max()) if P_arr.size > 0 else 0.0
                    entropy = float(-(P_arr * np.log(np.clip(P_arr, 1e-12, 1.0))).sum()) if P_arr.size > 0 else 0.0
                    # margin in logit space (max - second max); requires at least 2 candidates
                    if z_arr.size >= 2:
                        idx_sorted = np.argsort(z_arr)[::-1]
                        margin = float(z_arr[idx_sorted[0]] - z_arr[idx_sorted[1]])
                        z_min = float(z_arr.min())
                        z_med = float(np.median(z_arr))
                        z_max = float(z_arr.max())
                    else:
                        margin = 0.0
                        z_min = z_med = z_max = 0.0
                    # GT position in candidates
                    try:
                        gt_idx = cand_angles.index(int(gt_angle)) if 'cand_angles' in locals() else -1
                    except Exception:
                        gt_idx = -1
                    gt_rank = None
                    if gt_idx >= 0 and z_arr.size > 0:
                        # rank 1 is best
                        order_desc = np.argsort(z_arr)[::-1].tolist()
                        gt_rank = int(order_desc.index(gt_idx) + 1) if gt_idx in order_desc else None
                    p_gt = float(P_arr[gt_idx]) if gt_idx >= 0 and P_arr.size > gt_idx else 0.0
                    deltais_min = float(np.min(deltas_sel)) if 'deltas_sel' in locals() and len(deltas_sel) > 0 else 0.0
                    deltais_med = float(np.median(deltas_sel)) if 'deltas_sel' in locals() and len(deltas_sel) > 0 else 0.0
                    deltais_max = float(np.max(deltas_sel)) if 'deltas_sel' in locals() and len(deltas_sel) > 0 else 0.0
                    deltais_std = float(np.std(deltas_sel)) if 'deltas_sel' in locals() and len(deltas_sel) > 0 else 0.0
                    deltais_range_over_tau = float((deltais_max - deltais_min) / max(tau, 1e-6)) if 'deltas_sel' in locals() and len(deltas_sel) > 0 else 0.0
                    # Per-bin averaged ΔIS (for diagnostics)
                    if 'deltas_use' in locals() and args.listwise_normalize == "per_bin":
                        deltais_pb = deltas_use
                        deltais_pb_min = float(np.min(deltais_pb)) if deltais_pb.size > 0 else 0.0
                        deltais_pb_med = float(np.median(deltais_pb)) if deltais_pb.size > 0 else 0.0
                        deltais_pb_max = float(np.max(deltais_pb)) if deltais_pb.size > 0 else 0.0
                        deltais_pb_std = float(np.std(deltais_pb)) if deltais_pb.size > 0 else 0.0
                    else:
                        deltais_pb_min = deltais_pb_med = deltais_pb_max = deltais_pb_std = 0.0
                    teacher_diag = {
                        "tau_deg": float(tau),
                        "teacher_p_max": p_max,
                        "teacher_entropy": entropy,
                        "teacher_margin_logit": margin,
                        "logits_z_min": z_min,
                        "logits_z_median": z_med,
                        "logits_z_max": z_max,
                        "deltais_min": deltais_min,
                        "deltais_median": deltais_med,
                        "deltais_max": deltais_max,
                        "deltais_std": deltais_std,
                        "deltais_range_over_tau": deltais_range_over_tau,
                        "deltais_per_bin_min": deltais_pb_min,
                        "deltais_per_bin_median": deltais_pb_med,
                        "deltais_per_bin_max": deltais_pb_max,
                        "deltais_per_bin_std": deltais_pb_std,
                        "listwise_normalize": str(args.listwise_normalize),
                        "gt_in_candidates": bool(gt_idx >= 0),
                        "gt_rank": int(gt_rank) if gt_rank is not None else None,
                        "teacher_p_gt": p_gt,
                    }
            except Exception:
                teacher_diag = {}

            list_rows.append({
                "sample_index": si,
                "cand_ids": cand_token_ids,
                "cand_angles": cand_angles,
                "teacher_P": P,
                "patch_len": int(len(patch_ids_list[si])),
                "gt_angle": int(gt_angle),
            })
        # Per-sample log
        deltas = np.array([scores[a] - scores[b] for (a, b) in pairs], dtype=float) if (args.supervision == "pairwise" and 'pairs' in locals() and pairs) else np.array([0.0])
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
            "teacher": args.teacher,
            "beta": float(args.bt_beta),
            "fs": int(args.sample_rate),
            "n_fft": int(args.n_fft),
            "freq_min": float(args.freq_min),
            "freq_max": float(args.freq_max),
            "eps": float(args.eps),
            "dirs_considered": int(len(d_indices)),
            "pairs": int(len(pairs)) if args.supervision == "pairwise" else 0,
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
        }
        # Attach teacher softmax diagnostics if available
        if 'teacher_diag' in locals() and teacher_diag:
            log_row.update(teacher_diag)
        sample_logs.append(log_row)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

    print("Training data:")
    print(f"  - Samples: {len(prompts)}")
    avg_dirs = float(np.mean([r["dirs_considered"] for r in sample_logs])) if sample_logs else 0.0
    print(f"  - Directions/sample (avg): {avg_dirs:.1f}")
    if args.supervision == "pairwise":
        print(f"  - Pairs total: {len(pair_rows)} (~{len(pair_rows)/max(1,len(prompts)):.1f}/sample)")
    else:
        print(f"  - Points total: {len(point_rows)} (~{len(point_rows)/max(1,len(prompts)):.1f}/sample)")
    print(f"Saved numeric diagnostics JSONL: {jsonl_path}")
    # Summarize teacher softmax diagnostics if present
    try:
        keys = [
            "teacher_p_max",
            "teacher_entropy",
            "teacher_margin_logit",
            "deltais_range_over_tau",
        ]
        def _q(vals, q):
            return float(np.percentile(vals, q)) if vals else 0.0
        summary = {"samples": int(len(sample_logs))}
        for k in keys:
            vals = [float(r.get(k, 0.0)) for r in sample_logs if k in r]
            if vals:
                summary[f"{k}_p50"] = _q(vals, 50)
                summary[f"{k}_p90"] = _q(vals, 90)
                summary[f"{k}_p95"] = _q(vals, 95)
        # Saturation fraction: p_max>0.9
        pmax = [float(r.get("teacher_p_max", 0.0)) for r in sample_logs if "teacher_p_max" in r]
        if pmax:
            summary["sat_frac_pmax_gt_0.9"] = float(np.mean([1.0 if v > 0.9 else 0.0 for v in pmax]))
        print({"teacher_softmax_summary": summary})
    except Exception:
        pass

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
        if args.supervision == "pairwise":
            pw_total = len(pair_rows)
            pw_done = 0
            pw_t0 = time.time()
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
                denom += len(batch)
                # Progress
                pw_done += len(batch)
                if args.progress_every and (pw_done % max(1, args.progress_every) == 0 or pw_done == pw_total):
                    elapsed = time.time() - pw_t0
                    pct = 100.0 * pw_done / max(1, pw_total)
                    eta = elapsed * (pw_total / max(1, pw_done) - 1.0)
                    print({
                        "epoch": epoch,
                        "phase": "train",
                        "supervision": "pairwise",
                        "progress_rows": int(pw_done),
                        "total_rows": int(pw_total),
                        "pct": float(pct),
                        "elapsed_s": float(elapsed),
                        "eta_s": float(max(0.0, eta)),
                    }, flush=True)
            avg_loss = total_loss / max(1, denom)
            print({"epoch": epoch, "bt_pair_loss": float(avg_loss), "pairs": int(denom)})
        elif args.supervision == "pointwise":
            pt_total = len(point_rows)
            pt_done = 0
            pt_t0 = time.time()
            for batch in _iter_batches(point_rows, args.batch_size):
                # Build sequences and targets
                seqs: List[List[int]] = []
                targets: List[float] = []
                patch_lens: List[int] = []
                for row in batch:
                    si = row["sample_index"]
                    patch_ids = patch_ids_list[si]
                    seqs.append([tokenizer.bos_token_id, row["dir_id"], *patch_ids])
                    patch_lens.append(len(patch_ids))
                    targets.append(float(row["score"]))
                inp = _pad_to_batch(seqs, tokenizer.pad_token_id).to(device)
                attn = (inp != tokenizer.pad_token_id).to(device)
                out = rm_model.pretrained_model(
                    input_ids=inp,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)
                means: List[torch.Tensor] = []
                for r in range(vals.size(0)):
                    plen = patch_lens[r]
                    seqlen = int(attn[r].sum().item())
                    if plen == 0:
                        means.append(vals[r, seqlen-1:seqlen].mean())
                    else:
                        means.append(vals[r, seqlen-plen:seqlen].mean())
                pred = torch.stack(means, dim=0)
                tgt = torch.tensor(targets, dtype=pred.dtype, device=pred.device)
                loss = nn.MSELoss()(pred, tgt)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(batch)
                denom += len(batch)
                # Progress
                pt_done += len(batch)
                if args.progress_every and (pt_done % max(1, args.progress_every) == 0 or pt_done == pt_total):
                    elapsed = time.time() - pt_t0
                    pct = 100.0 * pt_done / max(1, pt_total)
                    eta = elapsed * (pt_total / max(1, pt_done) - 1.0)
                    print({
                        "epoch": epoch,
                        "phase": "train",
                        "supervision": "pointwise",
                        "progress_rows": int(pt_done),
                        "total_rows": int(pt_total),
                        "pct": float(pct),
                        "elapsed_s": float(elapsed),
                        "eta_s": float(max(0.0, eta)),
                    }, flush=True)
            avg_loss = total_loss / max(1, denom)
            print({"epoch": epoch, "point_mse": float(avg_loss), "points": int(denom)})
        else:  # listwise (DoA-aligned)
            # Cross-entropy between teacher P and RM softmax over candidate directions per sample
            ce = 0.0
            # Collect predicted Q statistics per epoch to diagnose saturation/flatness
            q_max_list: List[float] = []
            q_entropy_list: List[float] = []
            q_margin_list: List[float] = []
            lw_total = len(list_rows)
            lw_done = 0
            lw_t0 = time.time()
            for row in list_rows:
                si = int(row["sample_index"])
                patch_ids = patch_ids_list[si]
                cand_ids = row["cand_ids"]
                teacher_P = torch.tensor(row["teacher_P"], dtype=torch.float32, device=device)
                seqs = [[tokenizer.bos_token_id, int(cid), *patch_ids] for cid in cand_ids]
                inp = _pad_to_batch(seqs, tokenizer.pad_token_id).to(device)
                attn = (inp != tokenizer.pad_token_id).to(device)
                out = rm_model.pretrained_model(
                    input_ids=inp,
                    attention_mask=attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
                vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)
                preds: List[torch.Tensor] = []
                for r in range(vals.size(0)):
                    seqlen = int(attn[r].sum().item())
                    plen = len(patch_ids)
                    if plen == 0:
                        preds.append(vals[r, seqlen-1:seqlen].mean())
                    else:
                        preds.append(vals[r, seqlen-plen:seqlen].mean())
                logits = torch.stack(preds, dim=0)
                Q = torch.softmax(logits, dim=0)
                # Predicted distribution diagnostics
                with torch.no_grad():
                    q_np = Q.detach().cpu().numpy()
                    if q_np.size > 0:
                        q_max_list.append(float(np.max(q_np)))
                        q_entropy_list.append(float(-(q_np * np.log(np.clip(q_np, 1e-12, 1.0))).sum()))
                        # logit margin (max - second max)
                        lg = logits.detach().cpu().numpy()
                        if lg.size >= 2:
                            idx = np.argsort(lg)[::-1]
                            q_margin_list.append(float(lg[idx[0]] - lg[idx[1]]))
                        else:
                            q_margin_list.append(0.0)
                loss = torch.sum(-teacher_P * torch.log(torch.clamp(Q, min=1e-12)))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item())
                ce += float(loss.item())
                denom += 1
                # Progress
                lw_done += 1
                if args.progress_every and (lw_done % max(1, args.progress_every) == 0 or lw_done == lw_total):
                    elapsed = time.time() - lw_t0
                    pct = 100.0 * lw_done / max(1, lw_total)
                    eta = elapsed * (lw_total / max(1, lw_done) - 1.0)
                    print({
                        "epoch": epoch,
                        "phase": "train",
                        "supervision": "listwise",
                        "progress_rows": int(lw_done),
                        "total_rows": int(lw_total),
                        "pct": float(pct),
                        "elapsed_s": float(elapsed),
                        "eta_s": float(max(0.0, eta)),
                    }, flush=True)
            avg_loss = total_loss / max(1, denom)
            print({"epoch": epoch, "listwise_ce": float(avg_loss), "samples": int(denom)})
            # Summarize predicted Q distribution stats for the epoch
            try:
                def _q(vals, p):
                    return float(np.percentile(vals, p)) if vals else 0.0
                q_summary = {"samples": int(len(q_max_list))}
                if q_max_list:
                    q_summary.update({
                        "q_max_p50": _q(q_max_list, 50),
                        "q_max_p90": _q(q_max_list, 90),
                        "q_max_p95": _q(q_max_list, 95),
                    })
                if q_entropy_list:
                    q_summary.update({
                        "q_entropy_p50": _q(q_entropy_list, 50),
                        "q_entropy_p90": _q(q_entropy_list, 90),
                        "q_entropy_p95": _q(q_entropy_list, 95),
                    })
                if q_margin_list:
                    q_summary.update({
                        "q_margin_logit_p50": _q(q_margin_list, 50),
                        "q_margin_logit_p90": _q(q_margin_list, 90),
                        "q_margin_logit_p95": _q(q_margin_list, 95),
                    })
                print({"epoch": epoch, "q_softmax_summary": q_summary})
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
                # Use shuffle=True to avoid sampling a single-angle prefix when max_samples is small
                dl_eval = create_dataloader(ds_eval, batch_size=1, shuffle=True)
                tok_patch = PatchTokenizer(Fp=args.patch_fp, Np=args.patch_np)
                eval_prompts: List[str] = []
                gt_angles: List[int] = []
                # Collect eval sample meta for manifest
                eval_records: List[Dict[str, object]] = []
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
                    # Keep path for OMP teacher alignment
                    if "path" in batch_eval:
                        pass
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
                ev_total = int(input_ids_prompt.size(0))
                ev_done = 0
                ev_t0 = time.time()
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
                    # Eval progress
                    ev_done += 1
                    if args.progress_every and (ev_done % max(1, args.progress_every) == 0 or ev_done == ev_total):
                        elapsed = time.time() - ev_t0
                        pct = 100.0 * ev_done / max(1, ev_total)
                        eta = elapsed * (ev_total / max(1, ev_done) - 1.0)
                        print({
                            "epoch": epoch,
                            "phase": "eval",
                            "progress_samples": int(ev_done),
                            "total_samples": int(ev_total),
                            "pct": float(pct),
                            "elapsed_s": float(elapsed),
                            "eta_s": float(max(0.0, eta)),
                        }, flush=True)
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
                # Write eval subset manifest for reproducibility and downstream verification
                try:
                    results_dir = os.path.join("results", f"{args.out}")
                    os.makedirs(results_dir, exist_ok=True)
                    manifest_path = os.path.join(results_dir, "eval_subset_manifest.json")
                    with open(manifest_path, "w") as mf:
                        json.dump({"files": eval_records}, mf, indent=2)
                    print({"epoch": epoch, "eval_manifest": manifest_path, "eval_records": int(len(eval_records))})
                except Exception as _e:
                    print(f"Warning: failed to write eval manifest: {_e}")

                if args.eval_omp_align:
                    # Optional: OMP alignment metrics (may be slow)
                    try:
                        from scipy.stats import spearmanr  # Optional; fallback below if missing
                        HAVE_SCIPY = True
                    except Exception:
                        HAVE_SCIPY = False

                    inter_hits = 0
                    rhos = []
                    for batch_eval in dl_eval:
                        Y_t = batch_eval["Y"].squeeze(0)
                        y_src = batch_eval.get("path", None)
                        if isinstance(y_src, (list, tuple)):
                            y_src = y_src[0] if y_src else None
                        if not y_src:
                            continue
                        content_path = _derive_content_path(args.content_root, y_src)
                        Y_content = _load_band_spectrogram_from_npy(
                            content_path,
                            fs=args.sample_rate,
                            n_fft=args.n_fft,
                            freq_min=args.freq_min,
                            freq_max=args.freq_max,
                        ).float()
                        if Y_content.shape != Y_t.shape:
                            continue
                        s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=50, l1=0.0)
                        S_omp, _, _ = is_omp_select(
                            Y_t.numpy().astype(np.float64), H.numpy().astype(np.float64), W_t.numpy().astype(np.float64),
                            K=int(args.K), s_hat=s_hat_t.numpy().astype(np.float64), prefilter_M=16, mu_iter_warm=5, mu_iter_accept=20, eps=float(args.eps)
                        )
                        # Map RM top-K tokens to direction indices
                        dir_ids_all_list = list(direction_token_ids(tokenizer))
                        rm_topk_idx = []
                        for tid in teacher_dir_ids[i]:
                            if int(tid) in dir_ids_all_list:
                                rm_topk_idx.append(int(dir_ids_all_list.index(int(tid))))
                        inter_hits += len(set(S_omp).intersection(set(rm_topk_idx)))
                        # Spearman: ΔIS vs RM scores across all directions
                        ord_all, deltas_all = compute_deltais_step0(
                            Y=Y_t.numpy().astype(np.float64), H=H.numpy().astype(np.float64), W=W_t.numpy().astype(np.float64),
                            s_hat=s_hat_t.numpy().astype(np.float64), prefilter_M=None, mu_iter=10, baseline_k=2, eps=float(args.eps)
                        )
                        rm_scores_all = []
                        row = input_ids_prompt[i]
                        patch_ids = get_patch_ids(tokenizer, row)
                        for j in ord_all:
                            tid = int(dir_ids_all_list[j])
                            rm_scores_all.append(float(rm_score_for_prefix(rm_model, tokenizer, [tid], patch_ids, device)))
                        if len(rm_scores_all) >= 2:
                            if HAVE_SCIPY:
                                rho = float(spearmanr(rm_scores_all, deltas_all[:len(rm_scores_all)]).correlation)
                            else:
                                import numpy as _np
                                def _rank(a):
                                    order = _np.argsort(a)
                                    ranks = _np.empty_like(order, dtype=float)
                                    ranks[order] = _np.arange(1, len(a)+1)
                                    return ranks
                                r1 = _rank(_np.asarray(rm_scores_all))
                                r2 = _rank(_np.asarray(deltas_all[:len(rm_scores_all)]))
                                rho = float(_np.corrcoef(r1, r2)[0,1])
                            rhos.append(rho)
                    if total > 0:
                        print({"epoch": epoch, "eval_intersection_at_K_sum": int(inter_hits), "eval_spearman_mean": float(np.mean(rhos) if rhos else 0.0)})
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
