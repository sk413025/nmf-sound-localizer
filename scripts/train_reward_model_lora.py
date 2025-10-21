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
from scripts.train_sft_policy_with_rm import get_patch_ids
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

def _rm_pred_scores_batched(
    rm_model,
    tokenizer,
    dir_ids: Sequence[int],
    patch_ids: Sequence[int],
    device: torch.device,
) -> Dict[int, float]:
    """Score multiple direction ids in one forward using max pooling over patch tokens.

    Returns a dict {dir_id: score} computed without gradient for efficiency.
    """
    if not dir_ids:
        return {}
    seqs: List[List[int]] = []
    patch_len = len(patch_ids)
    bos = tokenizer.bos_token_id
    for did in dir_ids:
        seqs.append([bos, int(did), *[int(x) for x in patch_ids]])
    with torch.no_grad():
        inp = _pad_to_batch(seqs, tokenizer.pad_token_id).to(device)
        attn = (inp != tokenizer.pad_token_id).to(device)
        out = rm_model.pretrained_model(
            input_ids=inp,
            attention_mask=attn,
            output_hidden_states=True,
            return_dict=True,
        )
        vals = rm_model.v_head(out.hidden_states[-1]).squeeze(-1)  # (B, S)
        scores: List[float] = []
        for r in range(vals.size(0)):
            seqlen = int(attn[r].sum().item())
            if patch_len == 0:
                pooled = vals[r, seqlen-1:seqlen].max()
            else:
                pooled = vals[r, seqlen-patch_len:seqlen].max()
            scores.append(float(pooled.item()))
    return {int(dir_ids[i]): float(scores[i]) for i in range(len(dir_ids))}

## Unified batched scorer is used for eval (training uses pooled logits directly)


# -------------------------
# Small utilities (refactor)
# -------------------------

def select_device(dev_arg: str) -> torch.device:
    dev = dev_arg
    if dev == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = "mps"
        elif torch.cuda.is_available():
            dev = "cuda"
        else:
            dev = "cpu"
    device = torch.device(dev)
    print(f"\nDevice: {device}")
    return device


def apply_lora_and_unfreeze(rm_model, args):
    print("\n" + "=" * 60)
    print("Applying LoRA to Reward Model")
    print("=" * 60)
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
    rm_model.pretrained_model = get_peft_model(rm_model.pretrained_model, lora_config)
    embedding_layer = rm_model.pretrained_model.get_input_embeddings()
    if hasattr(embedding_layer, 'base_layer'):
        embedding_layer.base_layer.weight.requires_grad = True
    else:
        embedding_layer.weight.requires_grad = True
    rm_model.score = rm_model.v_head
    return rm_model, embedding_layer


def build_optimizer_for_rm(rm_model, args):
    return torch.optim.Adam([
        {
            "params": [p for n, p in rm_model.named_parameters() if p.requires_grad and "lora" in n.lower()],
            "lr": args.lr_lora,
        },
        {
            "params": [p for n, p in rm_model.named_parameters() if p.requires_grad and ("wte" in n.lower() or "wpe" in n.lower())],
            "lr": args.lr_embed,
        },
        {
            "params": rm_model.v_head.parameters(),
            "lr": args.lr_vhead,
        },
    ])


def pool_patch(vals_row: torch.Tensor, attn_row: torch.Tensor, patch_len: int, pooling: str, tau: float) -> torch.Tensor:
    seqlen = int(attn_row.sum().item())
    if patch_len <= 0:
        patch_slice = vals_row[seqlen-1:seqlen]
    else:
        patch_slice = vals_row[seqlen - patch_len: seqlen]
    if pooling == "lse":
        t = max(1e-6, float(tau))
        return torch.logsumexp(patch_slice / t, dim=-1) * t
    return patch_slice.max()


def iter_batches(rows: List[Dict[str, int]], bs: int):
    step = max(1, bs)
    for i in range(0, len(rows), step):
        yield rows[i:i+step]


def derive_gt_from_path(tokenizer, y_src: str) -> Tuple[int, int]:
    try:
        angle_dir = os.path.basename(os.path.dirname(y_src))
        gt_angle = int(angle_dir.split('_')[1])
    except Exception:
        gt_angle = -1
    gt_tok = tokenizer.convert_tokens_to_ids(f"<D_{gt_angle:03d}>") if gt_angle >= 0 else -1
    return int(gt_tok), int(gt_angle)


def qstats_array(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(list(x), dtype=float)
    if a.size == 0:
        a = np.array([0.0], dtype=float)
    return {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(np.max(a)),
    }


def estimate_shat_and_validate(y_src: str, Y: torch.Tensor, args, H: torch.Tensor, W_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load Original/content counterpart, compute ŝ, and enforce grid checks."""
    F, N = int(Y.shape[0]), int(Y.shape[1])
    content_path = _derive_content_path(args.content_root, y_src)
    Y_content = _load_band_spectrogram_from_npy(
        content_path,
        fs=args.sample_rate,
        n_fft=args.n_fft,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    ).float()
    if list(Y_content.shape) != list(Y.shape):
        raise RuntimeError(f"Y_content shape {tuple(Y_content.shape)} must equal Y {tuple(Y.shape)}; align STFT grid (fs/n_fft/band).")
    if int(H.shape[0]) != F:
        raise RuntimeError(f"H.F ({int(H.shape[0])}) != Y.F ({F}); check tf_path or STFT grid (fs/n_fft/band).")
    if (W_t.dim() != 2) or (int(W_t.shape[0]) != F):
        raise RuntimeError(f"W.F ({int(W_t.shape[0])}) != Y.F ({F}); ensure USM W matches STFT grid.")
    s_hat_t, _ = estimate_s_hat_torch(Y_content, W_t, mode="S1", H=None, n_iter=50, l1=0.0)
    if int(s_hat_t.shape[0]) != F:
        raise RuntimeError(f"ŝ shape mismatch: got {tuple(s_hat_t.shape)} vs F={F}")
    if not torch.isfinite(s_hat_t).all():
        raise RuntimeError("ŝ contains non-finite values; aborting")
    if (s_hat_t < 0).any():
        raise RuntimeError("ŝ contains negatives; aborting")
    return s_hat_t, Y_content


def teacher_scores_ord(Y: torch.Tensor, H: torch.Tensor, W_t: torch.Tensor, s_hat_t: torch.Tensor, eps: float) -> Tuple[List[int], List[float]]:
    ord_all, deltas_all = compute_deltais_step0(
        Y=Y.numpy().astype(np.float64),
        H=H.numpy().astype(np.float64),
        W=W_t.numpy().astype(np.float64),
        s_hat=s_hat_t.numpy().astype(np.float64),
        prefilter_M=None,
        mu_iter=10,
        baseline_k=2,
        eps=float(eps),
    )
    scores = [float(deltas_all[i]) for i in range(len(deltas_all))]
    return list(ord_all), scores


def build_all_pairs(order_idx: List[int], scores: List[float]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for ii in range(len(order_idx)):
        ai = order_idx[ii]
        for jj in range(ii + 1, len(order_idx)):
            bi = order_idx[jj]
            if scores[ai] == scores[bi]:
                continue
            pairs.append((ai, bi))
    return pairs


def batch_inputs_from_pairs(batch: List[Dict[str, int]], tokenizer, patch_ids_list: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[float]]:
    seqs: List[List[int]] = []
    patch_lens: List[int] = []
    betas: List[float] = []
    for row in batch:
        si = int(row.get("sample_index", 0))
        patch_ids = patch_ids_list[si]
        seqs.append([tokenizer.bos_token_id, row["dir_pos"], *patch_ids])
        seqs.append([tokenizer.bos_token_id, row["dir_neg"], *patch_ids])
        patch_lens.extend([len(patch_ids), len(patch_ids)])
        betas.append(float(row.get("beta", 1.0)))
    inp = _pad_to_batch(seqs, tokenizer.pad_token_id)
    attn = (inp != tokenizer.pad_token_id)
    return inp, attn, patch_lens, betas


def pooled_pair_logits(rm_model, inp: torch.Tensor, attn: torch.Tensor, patch_lens: List[int], pooling: str, tau: float) -> Tuple[torch.Tensor, torch.Tensor]:
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
        pooled_vals.append(pool_patch(vals[r], attn[r], plen, pooling, tau))
    pooled_t = torch.stack(pooled_vals, dim=0)
    pos_pred = pooled_t[0::2]
    neg_pred = pooled_t[1::2]
    return pos_pred, neg_pred


def write_sample_log_row(
    *,
    y_src: str,
    Y: torch.Tensor,
    s_hat_t: torch.Tensor,
    H: torch.Tensor,
    pairs: List[Tuple[int, int]],
    ord_all: List[int],
    scores: List[float],
    gt_tok: int,
    gt_angle: int,
    args,
    jsonl_path: str,
):
    Y_np = Y.numpy(); s_np = s_hat_t.numpy()
    F, N = int(Y_np.shape[0]), int(Y_np.shape[1])
    try:
        Hs = (H * s_hat_t.view(-1, 1)).float().cpu().numpy()
        k = 2
        part = np.partition(Hs, kth=k-1, axis=1)[:, :k]
        Y_base = np.maximum(np.sum(part, axis=1), float(args.eps))
        Y_base_exp = np.repeat(Y_base[:, None], N, axis=1)
        ratio = np.clip(Y_np / np.maximum(Y_base_exp, float(args.eps)), 1e-12, 1e12)
        ratio_p50 = float(np.percentile(ratio, 50)); ratio_p95 = float(np.percentile(ratio, 95)); ratio_p99 = float(np.percentile(ratio, 99))
        mix_base_min = float(np.min(Y_base)); mix_base_mean = float(np.mean(Y_base)); mix_base_max = float(np.max(Y_base))
    except Exception:
        k = 2
        ratio_p50 = ratio_p95 = ratio_p99 = 0.0
        mix_base_min = mix_base_mean = mix_base_max = 0.0
    log_row = {
        "path": os.path.basename(y_src) or "",
        "F": int(F), "N": int(N), "beta": float(args.bt_beta),
        "fs": int(args.sample_rate), "n_fft": int(args.n_fft),
        "freq_min": float(args.freq_min), "freq_max": float(args.freq_max), "eps": float(args.eps),
        "dirs_considered": int(len(ord_all)), "pairs": int(len(pairs)),
        "score_stats": qstats_array(scores),
        "delta_stats": qstats_array([scores[a]-scores[b] for (a,b) in pairs]) if pairs else qstats_array([0.0]),
        "Y_min": float(np.min(Y_np)), "Y_mean": float(np.mean(Y_np)), "Y_max": float(np.max(Y_np)),
        "s_hat_min": float(np.min(s_np)), "s_hat_mean": float(np.mean(s_np)), "s_hat_max": float(np.max(s_np)),
        "baseline_k": int(k),
        "mix_base_min": mix_base_min, "mix_base_mean": mix_base_mean, "mix_base_max": mix_base_max,
        "ratio_base_p50": ratio_p50, "ratio_base_p95": ratio_p95, "ratio_base_p99": ratio_p99,
        "gt_angle": int(gt_angle) if gt_angle >= 0 else None,
        "gt_tok": int(gt_tok) if gt_tok >= 0 else None,
        "pairs_total": int(len(pairs)), "gt_pairs_count": int(sum(1 for (a,b) in pairs if gt_tok>=0)),
        # Note: gt_pairs_count above is conservative; detailed count computed earlier if needed
    }
    with open(jsonl_path, "a") as jf:
        jf.write(json.dumps(log_row) + "\n")


def load_and_align_assets(args, direction_angles: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load TF H and USM W, align H columns to dataset angles, and return (H_aligned, W)."""
    H_full, anglesT = load_H(args.tf_path)
    H_full = H_full.float().cpu()
    anglesT = list(map(int, (anglesT.cpu().numpy().tolist() if hasattr(anglesT, 'cpu') else anglesT)))
    col_idx: List[int] = []
    for a in direction_angles:
        if int(a) not in anglesT:
            raise RuntimeError(
                f"Angle {int(a)}° not found in TF angles {anglesT}. Ensure dataset angles exactly match TF asset."
            )
        j = anglesT.index(int(a))
        col_idx.append(j)
    if len(set(col_idx)) != len(col_idx):
        raise RuntimeError(f"Duplicate TF column mappings detected: {col_idx}. Fix assets or dataset.")
    H = H_full[:, col_idx].contiguous()
    W_t = load_W(args.w_path).float().cpu()
    return H, W_t

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



# Removed unused completion parser (simplification)


def pretokenize_patch_ids(tokenizer, prompts: List[str]) -> List[List[int]]:
    enc = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors=None,
    )
    specials = {tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
    if isinstance(enc["input_ids"], list):
        return [[tid for tid in row if tid not in specials] for row in enc["input_ids"]]
    return [[tid for tid in row.tolist() if tid not in specials] for row in enc["input_ids"]]


def build_pairs_and_diagnostics(
    args,
    prompts: List[str],
    cache: Dict[str, Dict[str, torch.Tensor]],
    tokenizer,
    H: torch.Tensor,
    W_t: torch.Tensor,
    dir_token_ids_all: List[int],
    results_dir: str,
    jsonl_path: str,
) -> List[Dict[str, int]]:
    pair_rows: List[Dict[str, int]] = []
    sample_logs: List[Dict[str, float]] = []
    for si, p in enumerate(prompts):
        Y = cache[p]["Y"].float()
        F, N = Y.shape
        y_src = cache[p].get("path", "")
        if not y_src:
            raise RuntimeError("Dataset did not provide source path for Y; cannot derive content counterpart")

        # ŝ estimation + grid checks
        s_hat_t, _ = estimate_shat_and_validate(y_src, Y, args, H, W_t)

        # Teacher scores and ranking
        ord_all, scores = teacher_scores_ord(Y, H, W_t, s_hat_t, args.eps)
        order_idx = list(range(len(ord_all)))
        order_idx.sort(key=lambda i: scores[i], reverse=True)
        pairs = build_all_pairs(order_idx, scores)

        # Map to token rows
        gt_tok, gt_angle = derive_gt_from_path(tokenizer, y_src)
        gt_pairs_count = 0
        for (ai, bi) in pairs:
            d_pos = ord_all[ai]
            d_neg = ord_all[bi]
            t_pos = int(dir_token_ids_all[d_pos])
            t_neg = int(dir_token_ids_all[d_neg])
            pair_rows.append({
                "sample_index": si,
                "dir_pos": t_pos,
                "dir_neg": t_neg,
                "patch_len": 0,
                "beta": float(args.bt_beta),
            })
            if gt_tok >= 0 and (t_pos == gt_tok or t_neg == gt_tok):
                gt_pairs_count += 1

        # Minimal per‑sample diagnostics row
        Y_np = Y.numpy(); s_np = s_hat_t.numpy()
        try:
            Hs = (H * s_hat_t.view(-1, 1)).float().cpu().numpy()
            k = 2
            part = np.partition(Hs, kth=k-1, axis=1)[:, :k]
            Y_base = np.maximum(np.sum(part, axis=1), float(args.eps))
            Y_base_exp = np.repeat(Y_base[:, None], N, axis=1)
            ratio = np.clip(Y_np / np.maximum(Y_base_exp, float(args.eps)), 1e-12, 1e12)
            ratio_p50 = float(np.percentile(ratio, 50)); ratio_p95 = float(np.percentile(ratio, 95)); ratio_p99 = float(np.percentile(ratio, 99))
            mix_base_min = float(np.min(Y_base)); mix_base_mean = float(np.mean(Y_base)); mix_base_max = float(np.max(Y_base))
        except Exception:
            k = 2
            ratio_p50 = ratio_p95 = ratio_p99 = 0.0
            mix_base_min = mix_base_mean = mix_base_max = 0.0
        log_row = {
            "path": os.path.basename(y_src) or "",
            "F": int(F), "N": int(N), "beta": float(args.bt_beta),
            "fs": int(args.sample_rate), "n_fft": int(args.n_fft),
            "freq_min": float(args.freq_min), "freq_max": float(args.freq_max), "eps": float(args.eps),
            "dirs_considered": int(len(ord_all)), "pairs": int(len(pairs)),
            "score_stats": qstats_array(scores),
            "delta_stats": qstats_array([scores[a]-scores[b] for (a,b) in pairs]) if pairs else qstats_array([0.0]),
            "Y_min": float(np.min(Y_np)), "Y_mean": float(np.mean(Y_np)), "Y_max": float(np.max(Y_np)),
            "s_hat_min": float(np.min(s_np)), "s_hat_mean": float(np.mean(s_np)), "s_hat_max": float(np.max(s_np)),
            "baseline_k": int(k),
            "mix_base_min": mix_base_min, "mix_base_mean": mix_base_mean, "mix_base_max": mix_base_max,
            "ratio_base_p50": ratio_p50, "ratio_base_p95": ratio_p95, "ratio_base_p99": ratio_p99,
            "gt_angle": int(gt_angle) if gt_angle >= 0 else None,
            "gt_tok": int(gt_tok) if gt_tok >= 0 else None,
            "pairs_total": int(len(pairs)), "gt_pairs_count": int(gt_pairs_count),
            "gt_pair_ratio": float(gt_pairs_count / max(1, len(pairs))),
        }
        sample_logs.append(log_row)
        with open(jsonl_path, "a") as jf:
            jf.write(json.dumps(log_row) + "\n")

    # Print dataset summary
    print("Training data:")
    print(f"  - Samples: {len(prompts)}")
    avg_dirs = float(np.mean([r["dirs_considered"] for r in sample_logs])) if sample_logs else 0.0
    print(f"  - Directions/sample (avg): {avg_dirs:.1f}")
    print(f"  - Pairs total: {len(pair_rows)} (~{len(pair_rows)/max(1,len(prompts)):.1f}/sample)")
    print(f"Saved numeric diagnostics JSONL: {jsonl_path}")
    return pair_rows


def save_training_pairs(rows: List[Dict[str, int]], out_dir: str, base_count: int):
    os.makedirs(out_dir, exist_ok=True)
    pairs_out = os.path.join(out_dir, "training_pairs.json")
    with open(pairs_out, "w") as pf:
        json.dump({"rows": rows, "counts": {"base": int(base_count), "hn": 0}}, pf, indent=2)
    print({"training_pairs": pairs_out, "base_rows": int(base_count), "hn_rows": 0, "total": int(len(rows))})


def train_one_epoch(
    args,
    rm_model,
    tokenizer,
    optimizer,
    patch_ids_list: List[List[int]],
    rows_epoch: List[Dict[str, int]],
    device: torch.device,
) -> float:
    bce = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    denom = 0
    batches = list(iter_batches(rows_epoch, args.batch_size))
    for bi, batch in enumerate(batches):
        inp, attn, patch_lens, betas = batch_inputs_from_pairs(batch, tokenizer, patch_ids_list)
        inp = inp.to(device); attn = attn.to(device)
        pos_pred, neg_pred = pooled_pair_logits(rm_model, inp, attn, patch_lens, args.pooling, args.pool_temp)
        beta_vec = torch.tensor(betas, dtype=pos_pred.dtype, device=pos_pred.device)
        logits = beta_vec * (pos_pred - neg_pred)
        target = torch.ones_like(logits)
        loss = bce(logits, target)
        (loss / float(max(1, args.grad_accum))).backward()
        step_now = ((bi + 1) % max(1, args.grad_accum) == 0) or (bi == len(batches) - 1)
        if step_now:
            torch.nn.utils.clip_grad_norm_(rm_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item()) * len(batch)
        denom += len(batch)
    return total_loss / max(1, denom)


def evaluate_rm(
    args,
    rm_model,
    tokenizer,
    direction_angles: List[int],
    device: torch.device,
    results_dir: str,
    eval_history: List[Dict[str, object]],
    loss_history: List[Dict[str, object]],
):
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
        allowed = list(direction_token_ids(tokenizer))
        if args.K < 1 or args.K > len(allowed):
            raise RuntimeError(f"Invalid K={args.K}; valid range is [1, {len(allowed)}]")
        top1_hits = 0
        teacher_hits = 0
        for i in range(input_ids_prompt.size(0)):
            row = input_ids_prompt[i]
            patch_ids = get_patch_ids(tokenizer, row)
            if not patch_ids:
                raise RuntimeError("Eval prompt row has no patch tokens; cannot score")
            cand_scores = _rm_pred_scores_batched(rm_model, tokenizer, allowed, patch_ids, device)
            gt_angle = int(gt_angles[i])
            gt_token = f"<D_{gt_angle:03d}>"
            gt_id = tokenizer.convert_tokens_to_ids(gt_token)
            if gt_id not in allowed:
                raise RuntimeError(f"Ground-truth token '{gt_token}' not in allowed direction tokens")
            best_id = max(cand_scores, key=cand_scores.get)
            if best_id == gt_id:
                top1_hits += 1
            topk_ids = [k for k,_ in sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)[:int(args.K)]]
            if gt_id in topk_ids:
                teacher_hits += 1
        total = input_ids_prompt.size(0)
        top1_acc = top1_hits / max(1, total)
        recall_k = teacher_hits / max(1, total)
        eval_rec = {"epoch": int(len(eval_history)), "top1": float(top1_acc), "recallK": float(recall_k), "K": int(args.K), "samples": int(total)}
        prev_eval = eval_history[-1] if len(eval_history) > 0 else None
        eval_history.append(eval_rec)
        cur_loss = loss_history[-1]["value"] if len(loss_history) > 0 else None
        prev_loss = loss_history[-2]["value"] if len(loss_history) > 1 else None
        d_top1 = (eval_rec["top1"] - prev_eval["top1"]) if prev_eval else 0.0
        d_recall = (eval_rec["recallK"] - prev_eval["recallK"]) if prev_eval else 0.0
        d_loss = (float(cur_loss) - float(prev_loss)) if (cur_loss is not None and prev_loss is not None) else 0.0
        aligned = (d_loss < 0.0) and ((d_top1 > 0.0) or (d_recall > 0.0)) if prev_eval and prev_loss is not None else None
        print({"epoch": int(eval_rec["epoch"]), "eval_top1_acc": float(top1_acc), "eval_recall_at_K": float(recall_k), "eval_samples": int(total), "K": int(args.K)})
        print({"epoch": int(eval_rec["epoch"]), "progress_report": {"loss_name": loss_history[-1]["name"] if loss_history else "bt_pair_loss", "loss": float(cur_loss) if cur_loss is not None else None, "loss_delta": float(d_loss), "top1": float(eval_rec["top1"]), "top1_delta": float(d_top1), "recall_at_K": float(eval_rec["recallK"]), "recall_delta": float(d_recall), "aligned_loss_vs_metrics": (bool(aligned) if aligned is not None else None)}})
        try:
            with open(os.path.join(results_dir, "eval_subset_manifest.json"), "w") as mf:
                json.dump({"files": eval_records}, mf, indent=2)
            print({"epoch": int(eval_rec["epoch"]), "eval_manifest": os.path.join(results_dir, "eval_subset_manifest.json"), "eval_records": int(len(eval_records))})
        except Exception as _e:
            print(f"Warning: failed to write eval manifest: {_e}")
    finally:
        rm_model.train()


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
    # Pooling over patch span
    ap.add_argument("--pooling", type=str, default="max", choices=["max", "lse"],
                    help="Pooling over patch tokens: 'max' or 'lse' (log-sum-exp)")
    ap.add_argument("--pool-temp", type=float, default=1.0,
                    help="Temperature τ for LSE pooling (τ>0; ignored if --pooling=max)")
    # Removed hard-negative and retargeting controls (simplified trainer)
    ap.add_argument("--fixed-pairs", action="store_true", help="Construct training pairs once before training and keep them fixed across epochs")
    # Always use all directions per sample (teacher-all-pairs is the default behavior)
    ap.add_argument("--teacher-all-pairs", action="store_true", help="Use all teacher-consistent pairs per sample (pos>neg for all direction pairs)")
    
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
    # Gradient accumulation to emulate larger batches without increasing memory
    ap.add_argument("--grad-accum", type=int, default=1, help="Accumulate gradients over N micro-batches before optimizer.step()")
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
    device = select_device(args.device)

    # Prepare data and tokenizer
    direction_angles = _discover_angles(args.data_root)
    prompts, cache = _prepare_samples(args, direction_angles)
    tokenizer = build_patch_tokenizer(direction_angles)
    tokenizer.padding_side = "left"

    # Load TF/W assets and align
    H, W_t = load_and_align_assets(args, direction_angles)

    # Build base RM model and apply LoRA
    rm_model, _ = build_value_head_model(tokenizer)
    rm_model, embedding_layer = apply_lora_and_unfreeze(rm_model, args)
    rm_model.to(device)
    rm_model.train()
    
    # Omit parameter counting in simplified script

    # Build pairwise BT dataset from forward‑model teachers
    dir_token_ids_all: List[int] = list(direction_token_ids(tokenizer))
    results_dir = os.path.join("results", f"{args.out}")
    os.makedirs(results_dir, exist_ok=True)
    jsonl_path = os.path.join(results_dir, "numeric_diagnostics.jsonl")
    patch_ids_list = pretokenize_patch_ids(tokenizer, prompts)
    pair_rows = build_pairs_and_diagnostics(
        args,
        prompts,
        cache,
        tokenizer,
        H,
        W_t,
        dir_token_ids_all,
        results_dir,
        jsonl_path,
    )
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
    print("Pooling:")
    print(f"  - Type:          {args.pooling}")
    print(f"  - Temperature τ: {args.pool_temp}\n")
    
    bce = nn.BCEWithLogitsLoss()
    
    # Training loop
    print("=" * 60)
    print("Training")
    print("=" * 60)

    # helpers moved to module level: iter_batches, pool_patch

    # Precompute fixed pairs (base pairs only in simplified trainer)
    rows_fixed: List[Dict[str, int]] = []
    if args.fixed_pairs:
        base_rows = list(pair_rows)
        rows_fixed = list(base_rows)
        # Coverage tracking removed in simplified path
        # Persist training pairs for reproducibility
        try:
            pairs_out = os.path.join("results", f"{args.out}", "training_pairs.json")
            os.makedirs(os.path.dirname(pairs_out), exist_ok=True)
            with open(pairs_out, "w") as pf:
                json.dump({"rows": rows_fixed, "counts": {"base": len(base_rows), "hn": 0}}, pf, indent=2)
            print({"training_pairs": pairs_out, "base_rows": len(base_rows), "hn_rows": 0, "total": len(rows_fixed)})
        except Exception as _e:
            print(f"Warning: failed to write training_pairs.json: {_e}")

    for epoch in range(args.rm_epochs):
        rm_model.train()
        total_loss = 0.0
        denom = 0
        # Select rows once per epoch (simplified path)
        if args.fixed_pairs:
            rows_epoch = list(rows_fixed)
        else:
            rows_epoch = list(pair_rows)

        avg_loss = train_one_epoch(args, rm_model, tokenizer, optimizer, patch_ids_list, rows_epoch, device)
        epoch_loss_name = "bt_pair_loss"
        epoch_loss_value = float(avg_loss)
        print({"epoch": epoch, epoch_loss_name: epoch_loss_value, "pairs": int(denom)})
        try:
            loss_history.append({"epoch": int(epoch), "name": epoch_loss_name, "value": float(epoch_loss_value)})
        except Exception:
            pass

        # Evaluation: directions-first Top-1 and recall@K
        do_eval = (args.eval_every > 0) and (((epoch + 1) % args.eval_every == 0) or (epoch == args.rm_epochs - 1))
        if do_eval:
            evaluate_rm(args, rm_model, tokenizer, direction_angles, device, results_dir, eval_history, loss_history)

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
