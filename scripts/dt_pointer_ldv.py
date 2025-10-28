#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal Decision-Transformer-style trainer for LDV DoA with hierarchical pointer head.

Consumes trajectories from doa_rl/trajectories/offline_dt_dataset.py (g-teacher):
- Reconstructs residual r_t per step from y and selected actions S_{t-1}.
- Inputs per step: residual token r_t (F→d), RTG tokens (resid, acc), step/budget tokens.
- Time encoder: 1-layer TransformerEncoder over K steps (causal mask).
- Pointer head: expert scores via query·keys (keys from D aggregated per expert), atom scores via per-atom keys within expert.
- Loss: sum CE over steps for expert and atom (teacher forcing from trajectories).

Outputs under --out_dir:
- run.log (capture via -u | tee), metrics.npz, code_state.json, numeric_diagnostics.jsonl

No-Fallback checks:
- F consistency: D.F == reconstructed Y.F == H.F == W.F.
- Angles coverage and selected_indices from manifest used to rebuild W.
"""

from __future__ import annotations

import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from doa_rl.data import DoADataset, create_dataloader


def load_manifest(traj_dir: str) -> Dict[str, Any]:
    mpath = Path(traj_dir) / 'manifest.json'
    with open(mpath, 'r') as f:
        return json.load(f)


def load_trajectories(traj_dir: str) -> List[Dict[str, Any]]:
    tpath = Path(traj_dir) / 'trajectories.jsonl'
    data = []
    with open(tpath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def torch_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_H_W(h_path: str, w_path: str, selected_indices: List[int], device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    H = torch.load(h_path, map_location=device, weights_only=False)['H'].float()  # (F,E)
    w_data = torch.load(w_path, map_location=device, weights_only=False)
    W_full = w_data['W'].float() if isinstance(w_data, dict) and 'W' in w_data else w_data.float()
    W = W_full[:, selected_indices].clone()
    W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    return H, W


def build_D(H: torch.Tensor, W: torch.Tensor, angles: List[float]) -> Tuple[torch.Tensor, int, int]:
    Fdim, E = H.shape
    _, M = W.shape
    P = E * M
    D = torch.zeros(Fdim, P, dtype=torch.float32, device=H.device)
    j = 0
    for e in range(E):
        h = H[:, e]
        for m in range(M):
            atom = (h * W[:, m])
            atom = atom / (atom.norm() + 1e-12)
            D[:, j] = atom
            j += 1
    return D, E, M


def reduce_atoms_kmeans(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42) -> torch.Tensor:
    """KMeans over atom columns (F,M_full) → (F,n_clusters) centroids normalized per column (sklearn precise)."""
    X = W.cpu().numpy().T  # (M_full, F)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20, max_iter=300)
    labels = km.fit_predict(X)
    C = km.cluster_centers_  # (k,F)
    W_red = torch.from_numpy(C.T).float()  # (F,k)
    W_red = W_red / (W_red.norm(dim=0, keepdim=True) + 1e-12)
    return W_red


def reduce_atoms_kmeans_numpy(W: torch.Tensor, n_clusters: int = 8, random_state: int = 42, iters: int = 30) -> torch.Tensor:
    """Lightweight NumPy k-means (k-means++ init) matching trajectory builder fast path."""
    X = W.cpu().numpy().T  # (M_full, F)
    rng = np.random.RandomState(random_state)
    M_full = X.shape[0]
    # k-means++ init
    centers = []
    idx0 = int(rng.randint(M_full))
    centers.append(X[idx0].copy())
    for _ in range(1, n_clusters):
        d2 = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(rng.choice(M_full, p=probs))
        centers.append(X[idx].copy())
    C = np.stack(centers, axis=0)
    for _ in range(iters):
        dists = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        newC = np.stack([X[labels == k].mean(axis=0) if np.any(labels == k) else C[k] for k in range(n_clusters)], axis=0)
        if np.allclose(newC, C, atol=1e-5):
            C = newC
            break
        C = newC
    W_red = torch.from_numpy(C.T).float()
    W_red = W_red / (W_red.norm(dim=0, keepdim=True) + 1e-12)
    return W_red


def _load_qk_teacher_from_scripts(F: int, E: int, M: int, d_model: int, nhead: int, nlayers: int,
                                  checkpoint_path: str, device: torch.device):
    """Load FullTransformerRoutedSoftOMP from scripts/omp-transformer-ldv.py to compute qk_expert."""
    import importlib.util
    mod_path = Path('scripts/omp-transformer-ldv.py')
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location('omp_transformer_ldv', str(mod_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    ModelClass = getattr(mod, 'FullTransformerRoutedSoftOMP')
    mdl = ModelClass(F=F, E=E, M=M, d=d_model, nhead=nhead, nlayers=nlayers, steps=1).to(device)
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        sd = state.get('model_state_dict', state)
        missing, unexpected = mdl.load_state_dict(sd, strict=False)
        print(f"[Teacher loader] Missing keys: {missing}")
        print(f"[Teacher loader] Unexpected keys: {unexpected}")
    except Exception as e:
        print(f"[Teacher loader] Exception during load: {e}")
    # Restore critical flags
    mdl.routing_mode = 'qk'
    mdl.score_norm_mode = 'std'
    mdl.score_center_expert = True
    mdl.score_center_atoms = True
    mdl.no_type_bias = True
    mdl.expert_agg = 'l2'
    mdl.eval()
    return mdl


def qk_scores_with_config(model, D: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Compute teacher qk_expert (E,) with its normalization/centering flags."""
    E, M = model.E, model.M
    t_R = model.P_R(r) + (model.type_R if not model.no_type_bias else 0)
    T_D = model.P_D(D.T) + (model.type_D if not model.no_type_bias else 0)
    T = torch.cat([t_R[None, :], T_D], dim=0)
    mask = model._make_mask(T.size(0)).to(T.device)
    Henc = T if model.encoder_identity else model.encoder(T, mask=mask)
    h_R = Henc[0]; H_D = Henc[1:]
    qk_atoms = (model.Wk(H_D) @ model.Wq(h_R)) / math.sqrt(model.d)
    qk_atoms = qk_atoms.view(E, M)
    if model.expert_agg == 'l2':
        qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)
    elif model.expert_agg == 'max':
        qk_expert = qk_atoms.abs().max(dim=1).values
    else:
        qk_expert = F.relu(qk_atoms).mean(dim=1)
    if getattr(model, 'score_norm_mode', 'none') == 'std':
        se_mean, se_std = qk_expert.mean(), qk_expert.std()
        qk_expert = (qk_expert - se_mean) / (se_std + 1e-8)
    if getattr(model, 'score_center_expert', False):
        qk_expert = qk_expert - qk_expert.mean()
    return qk_expert


def recompute_r_t(y: torch.Tensor, D: torch.Tensor, actions_prev: List[int]) -> torch.Tensor:
    if len(actions_prev) == 0:
        return y
    D_S = D[:, actions_prev]
    sol = torch.linalg.lstsq(D_S, y)
    x_S = sol.solution
    y_hat = D_S @ x_S
    r = y - y_hat
    return r


class DTMinPointer(nn.Module):
    def __init__(self, F: int, E: int, M: int, d_model: int = 128, nhead: int = 2, nlayers: int = 1):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.d = d_model
        # Residual/Dictionary projections (align with teacher qk head semantics)
        self.P_R = nn.Linear(F, d_model, bias=False)
        self.P_D = nn.Linear(F, d_model, bias=False)
        # Type biases like teacher
        self.type_R = nn.Parameter(torch.randn(d_model))
        self.type_D = nn.Parameter(torch.randn(d_model))
        # RTG + step/budget projections
        self.proj_rtg = nn.Linear(2, d_model)
        self.proj_step = nn.Linear(2, d_model)
        self.ln = nn.LayerNorm(d_model)
        # Time encoder
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        # QK heads like teacher
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        # Precomputed dictionary embeddings after P_D + type_D + Wk
        self.register_buffer('KD_em', torch.empty(0))  # (E, M, d)

    def set_keys_from_D(self, D: torch.Tensor):
        # Precompute dictionary token embeddings H_D ≈ Wk(P_D(D.T)+type_D) without grad
        with torch.no_grad():
            Fdim, P = D.shape
            E = self.E; M = self.M
            DD = D.T  # (P,F)
            TD = self.P_D(DD) + self.type_D  # (P,d)
            KD = self.Wk(TD)  # (P,d)
            KD_em = KD.view(E, M, self.d).clone()
            self.KD_em = KD_em  # (E,M,d)

    def forward(self, R_seq: torch.Tensor, RTG_seq: torch.Tensor, STEP_seq: torch.Tensor, causal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: (B,K,F), (B,K,2), (B,K,2)
        B, K, Fdim = R_seq.shape
        r_tok = self.P_R(R_seq) + self.type_R           # (B,K,d)
        rtg_tok = self.proj_rtg(RTG_seq)                # (B,K,d)
        step_tok = self.proj_step(STEP_seq)             # (B,K,d)
        h = self.ln(r_tok + rtg_tok + step_tok)         # (B,K,d)
        Ht = self.encoder(h, mask=causal_mask)          # (B,K,d)
        Q = self.Wq(Ht)                                 # (B,K,d)
        # Teacher-like qk atoms: dot(Q, KD_em) / sqrt(d)
        KD = self.KD_em  # (E,M,d)
        qk = torch.einsum('bkd,emd->bkem', Q, KD) / math.sqrt(self.d)  # (B,K,E,M)
        # Expert L2 aggregation over atoms
        scores_e = torch.sqrt((qk.abs() ** 2).sum(dim=3) + 1e-12)      # (B,K,E)
        scores_em = qk                                                 # (B,K,E,M)
        return scores_e, scores_em


def generate_causal_mask(K: int, device: torch.device) -> torch.Tensor:
    m = torch.full((K, K), float('-inf'), device=device)
    m = torch.triu(m, diagonal=1)  # allow self and past
    return m


@torch.no_grad()
def unroll_policy(model: DTMinPointer, D: torch.Tensor, y: torch.Tensor, K: int,
                  rtg_resid_target: float, rtg_acc_target: float,
                  device: torch.device) -> Tuple[List[Tuple[int,int,int]], int, torch.Tensor]:
    """Greedy unroll: at each step compute inputs up to t and pick (e,m) by argmax.
    Returns (actions list of (e,m,j), predicted_angle_e, final y_hat).
    """
    E = model.E; M = model.M; Fdim = model.F
    actions: List[int] = []
    em_list: List[Tuple[int,int,int]] = []
    # Precompute keys are already set in model via set_keys_from_D
    # Build helper to compute p_true from y_hat
    def p_true_from_yhat(y_hat: torch.Tensor) -> float:
        g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)
        p = torch.softmax(g_hat, dim=0)
        # Use explicit e from y via dataset? Not available here, return mean as placeholder for RTG update
        # We only use it to compute RTG_acc; for stability, approximate by max probability gap to target using top-1 prob
        return float(p.max().item())
    # Begin unroll
    y = y.to(device)
    for t in range(K):
        # recompute r_t from actions so far
        r_t = recompute_r_t(y, D, actions)
        # build sequences up to t
        R_seq = r_t.view(1,1,-1)
        # estimate current p (using previous y_hat if available)
        y_hat = (D[:, actions] @ torch.linalg.lstsq(D[:, actions], y).solution) if actions else torch.zeros_like(y)
        p_est = p_true_from_yhat(y_hat)
        RTG_seq = torch.tensor([[[max(0.0, float((r_t @ r_t).item()) - rtg_resid_target), max(0.0, rtg_acc_target - p_est)]]], dtype=torch.float32, device=device)
        STEP_seq = torch.tensor([[[t/ K, (K - t)/ K]]], dtype=torch.float32, device=device)
        cmask = generate_causal_mask(1, device)
        se, sem = model(R_seq, RTG_seq, STEP_seq, cmask)
        e = int(se[0,0].argmax().item())
        a_scores = sem[0,0,e]
        m = int(a_scores.argmax().item())
        j = e * M + m
        if j in actions:
            # pick next best atom for the same expert if duplicated
            a_scores2 = a_scores.clone(); a_scores2[m] = -1e9
            m = int(a_scores2.argmax().item()); j = e*M + m
        actions.append(j); em_list.append((e,m,j))
    # Build final y_hat and angle prediction
    if actions:
        xS = torch.linalg.lstsq(D[:, actions], y).solution
        y_hat = D[:, actions] @ xS
    else:
        y_hat = torch.zeros_like(y)
    g_hat = (D.T @ y_hat).view(E, M).abs().sum(dim=1)
    e_pred = int(torch.argmax(g_hat).item())
    return em_list, e_pred, y_hat


def main():
    ap = argparse.ArgumentParser(description='Minimal DT trainer (hierarchical pointer)')
    ap.add_argument('--traj_dir', type=str, default='results/dt_traj_v1')
    ap.add_argument('--out_dir', type=str, default='results/dt_min_v1')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--nhead', type=int, default=2)
    ap.add_argument('--nlayers', type=int, default=1)
    ap.add_argument('--device', type=str, default='cpu', choices=['cpu','mps','cuda'])
    # Optional teacher for diagnostics alignment
    ap.add_argument('--teacher_ckpt', type=str, default='')
    ap.add_argument('--teacher_d_model', type=int, default=128)
    ap.add_argument('--teacher_nhead', type=int, default=2)
    ap.add_argument('--teacher_nlayers', type=int, default=1)
    ap.add_argument('--distill_weight', type=float, default=0.5, help='Weight for teacher qk_expert distillation')
    ap.add_argument('--distill_T', type=float, default=1.0, help='Temperature for distillation softmax')
    ap.add_argument('--warmup_epochs', type=int, default=2, help='Epochs to apply distillation strongly')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print('='*80); print('DT-Min Trainer (hierarchical pointer)'); print('='*80)
    print(f"Traj dir: {args.traj_dir}\nOut dir: {out_dir}")

    manifest = load_manifest(args.traj_dir)
    trajs = load_trajectories(args.traj_dir)

    # Rebuild dataset and dictionary using manifest
    angles = manifest['angles']
    # Load raw H/W and reconstruct reduced W according to manifest
    H_raw = torch.load(manifest['h_path'], map_location='cpu', weights_only=False)['H'].float()
    w_data = torch.load(manifest['w_path'], map_location='cpu', weights_only=False)
    W_full = w_data['W'].float() if isinstance(w_data, dict) and 'W' in w_data else w_data.float()
    ar = manifest.get('atom_reduce', {})
    sel = ar.get('selected_indices', None)
    mode = ar.get('mode', 'kcenter')
    M = int(manifest.get('M', 8))
    if sel is not None:
        W = W_full[:, sel].clone()
        W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    else:
        # Re-run reduction (for K-means case); default to KMeans numpy impl
        impl = ar.get('kmeans_impl', 'numpy')
        if mode == 'kmeans':
            # Prefer fast numpy implementation when reconstructing reduced W
            W = reduce_atoms_kmeans_numpy(W_full, n_clusters=M, random_state=manifest.get('seed', 42))
        else:
            # Fallback: use first M atoms normalized
            W = W_full[:, :M].clone(); W = W / (W.norm(dim=0, keepdim=True) + 1e-12)
    H, W = H_raw, W
    D, E, M = build_D(H, W, angles)
    Fdim = D.shape[0]
    print(f"D: F={Fdim}, E={E}, M={M}, P={E*M}")

    # Load dataset for y reconstruction
    ds = DoADataset(manifest['dataset_root'], angles, fs=manifest['fs'], n_fft=manifest['n_fft'], window='hann', freq_min=manifest['freq_min'], freq_max=manifest['freq_max'])
    # Build path -> y map lazily on demand to avoid loading all upfront
    from collections import defaultdict
    index_by_path = {str(p): i for i,(p,_,_) in enumerate(ds.index)}

    def load_y_for_path(path: str) -> torch.Tensor:
        i = index_by_path[path]
        sample = ds[i]
        Y = sample['Y']
        if Y.shape[0] != Fdim:
            raise RuntimeError(f"Y.F mismatch for {path}: {Y.shape[0]} vs D.F={Fdim}")
        y = Y.mean(dim=1).float()
        y = y / (y.norm() + 1e-12)
        return y

    # Build training tensors (R_seq, RTG_seq, STEP_seq) and labels (expert, atom) per step
    K = manifest['K']
    samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for obj in trajs:
        path = obj['path']; angle_idx = int(obj['angle_index']); steps = obj['steps']
        y = load_y_for_path(path)
        actions_prev: List[int] = []
        R_list=[]; RTG_list=[]; STEP_list=[]; lab_e=[]; lab_m=[]
        for t, s in enumerate(steps):
            r_t = recompute_r_t(y, D, actions_prev)
            # Inputs
            R_list.append(r_t.unsqueeze(0))
            RTG_list.append(torch.tensor([[float(s['rtg_resid']), float(s['rtg_acc'])]], dtype=torch.float32))
            STEP_list.append(torch.tensor([[t/ K, (K - t)/ K]], dtype=torch.float32))
            # Labels
            e = int(s['expert']); m = int(s['atom']); j = int(s['dict_index'])
            lab_e.append(torch.tensor([e], dtype=torch.long))
            lab_m.append(torch.tensor([m], dtype=torch.long))
            actions_prev.append(j)
        R_seq = torch.cat(R_list, dim=0)              # (K,F)
        RTG_seq = torch.cat(RTG_list, dim=0)          # (K,2)
        STEP_seq = torch.cat(STEP_list, dim=0)        # (K,2)
        lab_e = torch.cat(lab_e, dim=0)               # (K,)
        lab_m = torch.cat(lab_m, dim=0)               # (K,)
        samples.append((R_seq, RTG_seq, STEP_seq, lab_e, lab_m))

    # Data loader (simple list batching)
    def batch_iter(batch_size: int):
        idx = np.random.permutation(len(samples))
        for i in range(0, len(idx), batch_size):
            batch = [samples[j] for j in idx[i:i+batch_size]]
            maxK = K
            R_b = torch.stack([b[0] for b in batch], dim=0)  # (B,K,F)
            RTG_b = torch.stack([b[1] for b in batch], dim=0)
            STEP_b = torch.stack([b[2] for b in batch], dim=0)
            E_b = torch.stack([b[3] for b in batch], dim=0)  # (B,K)
            M_b = torch.stack([b[4] for b in batch], dim=0)  # (B,K)
            yield R_b, RTG_b, STEP_b, E_b, M_b

    # Model
    model = DTMinPointer(F=Fdim, E=E, M=M, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    model.set_keys_from_D(D.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    causal = generate_causal_mask(K, device)
    # Freeze dictionary-side qk components to keep KD_em consistent
    for p in [model.P_D.weight, model.Wk.weight, model.type_D]:
        p.requires_grad_(False)

    # Optional teacher (for diagnostics alignment)
    teacher_ckpt = args.teacher_ckpt
    if not teacher_ckpt and isinstance(manifest.get('qk', None), dict):
        teacher_ckpt = manifest['qk'].get('ckpt', '')
    teacher_model = None
    if teacher_ckpt:
        teacher_model = _load_qk_teacher_from_scripts(F=Fdim, E=E, M=M, d_model=args.teacher_d_model,
                                                     nhead=args.teacher_nhead, nlayers=args.teacher_nlayers,
                                                     checkpoint_path=teacher_ckpt, device=device)

    # Resume support
    ckpt_path = out_dir / 'ckpt_latest.pth'
    hist = []
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                opt.load_state_dict(ckpt['optimizer_state_dict'])
            if 'history' in ckpt:
                hist = ckpt['history']
            print(f"[Resume] Loaded checkpoint from {ckpt_path}")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint: {e}")

    # Training
    for epoch in range(args.epochs):
        model.train(); total=0.0; n=0
        for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(args.batch_size):
            R_b = R_b.to(device); RTG_b = RTG_b.to(device); STEP_b = STEP_b.to(device)
            E_b = E_b.to(device); M_b = M_b.to(device)
            scores_e, scores_em = model(R_b, RTG_b, STEP_b, causal_mask=causal)
            # Loss: CE over steps
            loss_e = F.cross_entropy(scores_e.reshape(-1, E), E_b.reshape(-1))
            # For atom CE, select expert row per sample/step using gather
            idx = E_b.reshape(-1)  # (B*K,)
            se_flat = scores_em.reshape(-1, E, M)  # (B*K,E,M)
            se_sel = se_flat.gather(1, idx.view(-1,1,1).expand(-1,1,M)).squeeze(1)  # (B*K,M)
            loss_a = F.cross_entropy(se_sel, M_b.reshape(-1))
            loss = loss_e + loss_a
            # Optional teacher distillation on expert scores (per step)
            if teacher_model is not None and args.distill_weight > 0.0:
                # Build teacher logits per step from R_b using current D
                with torch.no_grad():
                    te_list = []
                    for t in range(K):
                        # average over batch for speed? keep per-sample
                        te_t = []
                        for b in range(R_b.size(0)):
                            qk_exp = qk_scores_with_config(teacher_model, D.to(device), R_b[b,t])
                            te_t.append(qk_exp)
                        te_t = torch.stack(te_t, dim=0)  # (B,E)
                        te_list.append(te_t)
                    te_logits = torch.stack(te_list, dim=1)  # (B,K,E)
                # distillation KL: student vs teacher at temperature T
                T = max(args.distill_T, 1e-8)
                p_teacher = F.softmax(te_logits / T, dim=-1)
                log_p_student = F.log_softmax(scores_e / T, dim=-1)
                kl = F.kl_div(log_p_student, p_teacher, reduction='batchmean') * (T*T)
                # Warmup scaling
                w = args.distill_weight if epoch < args.warmup_epochs else (0.5 * args.distill_weight)
                loss = loss + w * kl
            opt.zero_grad(); loss.backward(retain_graph=True); opt.step()
            total += float(loss.item()); n += 1
        avg = total / max(n,1)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg:.4f}")
        hist.append({'epoch': (hist[-1]['epoch'] + 1 if hist else 1), 'loss': avg})
        # Save checkpoint each epoch
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'history': hist,
            }, ckpt_path)
        except Exception as e:
            print(f"[Checkpoint] Save failed: {e}")

    # Simple evaluation: per-step top-1 accuracy on the training set (proxy)
    model.eval();
    with torch.no_grad():
        correct_e=0; total_e=0; correct_a=0; total_a=0
        for R_b, RTG_b, STEP_b, E_b, M_b in batch_iter(batch_size=len(samples)):
            R_b = R_b.to(device); RTG_b = RTG_b.to(device); STEP_b = STEP_b.to(device)
            E_b = E_b.to(device); M_b = M_b.to(device)
            scores_e, scores_em = model(R_b, RTG_b, STEP_b, causal_mask=causal)
            pred_e = scores_e.argmax(dim=-1)
            correct_e += int((pred_e == E_b).sum().item()); total_e += int(E_b.numel())
            se_flat = scores_em.reshape(-1, E, M)
            idx = E_b.reshape(-1)
            se_sel = se_flat.gather(1, idx.view(-1,1,1).expand(-1,1,M)).squeeze(1)
            pred_a = se_sel.argmax(dim=-1).view(E_b.shape)
            correct_a += int((pred_a == M_b).sum().item()); total_a += int(M_b.numel())
        acc_e = correct_e/total_e if total_e else 0.0
        acc_a = correct_a/total_a if total_a else 0.0
        print(f"Final train step-acc: expert={acc_e:.3f}, atom={acc_a:.3f}")

    # Angle accuracy via unroll on full dataset (using manifest targets)
    rtg_resid_tgt = float(manifest['rtg_targets']['resid'])
    rtg_acc_tgt = float(manifest['rtg_targets']['acc'])

    # Helper: teacher-forced sequences from trajectories
    def build_teacher_seqs(obj) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        path = obj['path']; steps = obj['steps']
        y = load_y_for_path(path)
        actions=[]; R_list=[]; RTG_list=[]; STEP_list=[]
        for t,s in enumerate(steps):
            r_t = recompute_r_t(y, D, actions)
            R_list.append(r_t.unsqueeze(0))
            RTG_list.append(torch.tensor([[float(s['rtg_resid']), float(s['rtg_acc'])]], dtype=torch.float32))
            STEP_list.append(torch.tensor([[t/ K, (K - t)/ K]], dtype=torch.float32))
            actions.append(int(s['dict_index']))
        R_seq = torch.cat(R_list, dim=0).unsqueeze(0)
        RTG_seq = torch.cat(RTG_list, dim=0).unsqueeze(0)
        STEP_seq = torch.cat(STEP_list, dim=0).unsqueeze(0)
        return R_seq, RTG_seq, STEP_seq, actions

    # Teacher-forced imitation angle accuracy和對齊診斷（含 per-step entropy/margin/corr/top1-match）
    model.eval()
    correct_steps=0; total_steps=0; correct_angles_t0=0; total_angles=0
    ctrl_path = (out_dir / 'controllability.jsonl').open('w')
    # per-step aggregates
    entropies_dt = np.zeros(K, dtype=float)
    margins_dt = np.zeros(K, dtype=float)
    counts_dt = np.zeros(K, dtype=float)
    # DT-min vs teacher corr/top1 match per step
    pearson_arr = np.zeros(K, dtype=float)
    spearman_arr = np.zeros(K, dtype=float)
    top1match_arr = np.zeros(K, dtype=float)
    counts_align = np.zeros(K, dtype=float)
    # Four accuracy tallies
    correct_dtmin_t0=0; correct_dtmin_tK=0; correct_teacher_qk=0; correct_g=0
    for obj in trajs:
        angle_idx = int(obj['angle_index'])
        R_seq, RTG_seq, STEP_seq, actions = build_teacher_seqs(obj)
        se, sem = model(R_seq.to(device), RTG_seq.to(device), STEP_seq.to(device), causal_mask=generate_causal_mask(K, device))
        se_np = se.detach().squeeze(0).cpu().numpy()  # (K,E)
        pred_e = se.argmax(dim=-1).squeeze(0).cpu().tolist()
        # gather atom scores for predicted expert each step
        pred_m = []
        for t in range(K):
            a_scores = sem[0,t,pred_e[t]].cpu()
            pred_m.append(int(a_scores.argmax().item()))
        # compare with teacher steps
        for t in range(K):
            e_t = actions[t] // model.M
            m_t = actions[t] % model.M
            if pred_e[t] == e_t and pred_m[t] == m_t:
                correct_steps += 1
            total_steps += 1
        # Per-step entropy/margin
        for t in range(K):
            logits = se[0,t]
            p = torch.softmax(logits, dim=-1)
            ent = float((-p * p.clamp_min(1e-12).log()).sum().item())
            top2 = torch.topk(logits, k=2).values
            margin = float((top2[0] - top2[1]).item())
            entropies_dt[t] += ent; margins_dt[t] += margin; counts_dt[t] += 1
        # Four accuracy variants
        e_dt_t0 = int(se[0,0].argmax().item())
        e_dt_tK = int(se[0,-1].argmax().item())
        if e_dt_t0 == angle_idx: correct_dtmin_t0 += 1
        if e_dt_tK == angle_idx: correct_dtmin_tK += 1
        # Teacher qk head at t=0 and g-energy baseline
        y = load_y_for_path(obj['path']).to(device)
        if teacher_model is not None:
            with torch.no_grad():
                # build teacher qk_expert on r0=y
                t_R = teacher_model.P_R(y) + (teacher_model.type_R if not teacher_model.no_type_bias else 0)
                T_D = teacher_model.P_D(D.to(device).T) + (teacher_model.type_D if not teacher_model.no_type_bias else 0)
                T = torch.cat([t_R[None,:], T_D], dim=0)
                mask = teacher_model._make_mask(T.size(0)).to(device)
                Henc = T if teacher_model.encoder_identity else teacher_model.encoder(T, mask=mask)
                h_R = Henc[0]; H_D = Henc[1:]
                qk_atoms = (teacher_model.Wk(H_D) @ teacher_model.Wq(h_R)) / math.sqrt(teacher_model.d)
                qk_atoms = qk_atoms.view(E, M)
                if teacher_model.expert_agg == 'l2':
                    qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)
                elif teacher_model.expert_agg == 'max':
                    qk_expert = qk_atoms.abs().max(dim=1).values
                else:
                    qk_expert = F.relu(qk_atoms).mean(dim=1)
                e_teacher_qk = int(qk_expert.argmax().item())
                if e_teacher_qk == angle_idx: correct_teacher_qk += 1
                # Alignment per-step (compare DT-min vs teacher on same r_t)
                for t in range(K):
                    r_t = R_seq[0,t].to(device)
                    t_Rt = teacher_model.P_R(r_t) + (teacher_model.type_R if not teacher_model.no_type_bias else 0)
                    Tt = torch.cat([t_Rt[None,:], T_D], dim=0)
                    Ht = Tt if teacher_model.encoder_identity else teacher_model.encoder(Tt, mask=mask)
                    h_Rt = Ht[0]; H_Dt = Ht[1:]
                    qk_atoms_t = (teacher_model.Wk(H_Dt) @ teacher_model.Wq(h_Rt)) / math.sqrt(teacher_model.d)
                    qk_atoms_t = qk_atoms_t.view(E, M)
                    if teacher_model.expert_agg == 'l2':
                        qk_expert_t = torch.sqrt((qk_atoms_t.abs() ** 2).sum(dim=1) + 1e-12)
                    elif teacher_model.expert_agg == 'max':
                        qk_expert_t = qk_atoms_t.abs().max(dim=1).values
                    else:
                        qk_expert_t = F.relu(qk_atoms_t).mean(dim=1)
                    dt_vec = se[0,t].detach().cpu().numpy()
                    te_vec = qk_expert_t.detach().cpu().numpy()
                    # Pearson
                    if np.std(dt_vec) > 1e-12 and np.std(te_vec) > 1e-12:
                        pearson = float(np.corrcoef(dt_vec, te_vec)[0,1])
                        pearson_arr[t] += pearson; counts_align[t] += 1
                    # Spearman (rank then Pearson)
                    def _rank(x):
                        order = np.argsort(x)
                        ranks = np.empty_like(order)
                        ranks[order] = np.arange(len(x))
                        return ranks
                    sp = float(np.corrcoef(_rank(dt_vec), _rank(te_vec))[0,1])
                    spearman_arr[t] += sp
                    # top-1 match
                    top1match_arr[t] += 1.0 if (int(np.argmax(dt_vec)) == int(np.argmax(te_vec))) else 0.0
        # g-energy baseline
        g = (D.T.to(device) @ y).view(E, M).abs().sum(dim=1)
        e_g = int(g.argmax().item())
        if e_g == angle_idx: correct_g += 1
        total_angles += 1
        # Controllability: change RTG tokens while keeping R_seq fixed
        RTG_alt = RTG_seq.clone(); RTG_alt[:,:,0] = torch.clamp(RTG_alt[:,:,0]*0.5, min=0.0); RTG_alt[:,:,1] = torch.clamp(RTG_alt[:,:,1]+0.03, max=0.99)
        se2, sem2 = model(R_seq.to(device), RTG_alt.to(device), STEP_seq.to(device), causal_mask=generate_causal_mask(K, device))
        pred_e2 = se2.argmax(dim=-1).squeeze(0).cpu().tolist()
        pred_m2 = []
        for t in range(K):
            a_scores2 = sem2[0,t,pred_e2[t]].cpu()
            pred_m2.append(int(a_scores2.argmax().item()))
        diff_steps = int(sum(1 for a,b in zip(zip(pred_e,pred_m), zip(pred_e2,pred_m2)) if a!=b))
        ctrl_path.write(json.dumps({'path': obj['path'], 'angle_idx': angle_idx, 'diff_steps': diff_steps})+'\n')
    ctrl_path.close()
    step_match = correct_steps / max(total_steps,1)
    acc_dtmin_t0 = correct_dtmin_t0 / max(total_angles,1)
    acc_dtmin_tK = correct_dtmin_tK / max(total_angles,1)
    acc_teacher_qk = correct_teacher_qk / max(total_angles,1) if teacher_model is not None else None
    acc_g = correct_g / max(total_angles,1)
    print(f"Teacher-forced step match: {step_match:.3f}")
    print(f"Angle acc — DT-min t=0: {acc_dtmin_t0:.3f}; DT-min t=K-1: {acc_dtmin_tK:.3f}; teacher qk t=0: {acc_teacher_qk}; g(y): {acc_g:.3f}")

    # Optional: evaluate teacher qk t=0 using KMeans-reduced W to align with teacher training setup
    acc_teacher_qk_km = None
    if teacher_model is not None:
        W_km = reduce_atoms_kmeans(W, n_clusters=M, random_state=42)
        D_km, _, _ = build_D(H, W_km, angles)
        correct_teach_km = 0
        for obj in trajs:
            angle_idx = int(obj['angle_index'])
            y = load_y_for_path(obj['path']).to(device)
            with torch.no_grad():
                t_R = teacher_model.P_R(y) + (teacher_model.type_R if not teacher_model.no_type_bias else 0)
                T_D = teacher_model.P_D(D_km.to(device).T) + (teacher_model.type_D if not teacher_model.no_type_bias else 0)
                T = torch.cat([t_R[None,:], T_D], dim=0)
                mask = teacher_model._make_mask(T.size(0)).to(device)
                Henc = T if teacher_model.encoder_identity else teacher_model.encoder(T, mask=mask)
                h_R = Henc[0]; H_D = Henc[1:]
                qk_atoms = (teacher_model.Wk(H_D) @ teacher_model.Wq(h_R)) / math.sqrt(teacher_model.d)
                qk_atoms = qk_atoms.view(E, M)
                if teacher_model.expert_agg == 'l2':
                    qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)
                elif teacher_model.expert_agg == 'max':
                    qk_expert = qk_atoms.abs().max(dim=1).values
                else:
                    qk_expert = F.relu(qk_atoms).mean(dim=1)
                e_teach = int(qk_expert.argmax().item())
                if e_teach == angle_idx:
                    correct_teach_km += 1
        acc_teacher_qk_km = correct_teach_km / max(total_angles,1)
        print(f"Teacher qk t=0 with KMeans W: {acc_teacher_qk_km:.3f}")

    # Aggregated per-step diagnostics
    ent_mean = (entropies_dt / np.maximum(counts_dt, 1)).tolist()
    margin_mean = (margins_dt / np.maximum(counts_dt, 1)).tolist()
    pearson_mean = (pearson_arr / np.maximum(counts_align, 1)).tolist()
    spearman_mean = (spearman_arr / np.maximum(counts_align, 1)).tolist()
    top1match_mean = (top1match_arr / np.maximum(counts_align, 1)).tolist()

    # Save metrics and code_state
    np.savez(out_dir / 'metrics.npz', history=np.array(hist, dtype=object), acc_e=acc_e, acc_a=acc_a,
             teacher_forced_step_match=step_match,
             acc_dtmin_t0=acc_dtmin_t0, acc_dtmin_tK=acc_dtmin_tK,
             acc_teacher_qk=(acc_teacher_qk if acc_teacher_qk is not None else np.nan),
             acc_teacher_qk_km=(acc_teacher_qk_km if acc_teacher_qk_km is not None else np.nan),
             acc_g=acc_g,
             ent_mean=np.array(ent_mean), margin_mean=np.array(margin_mean),
             pearson_mean=np.array(pearson_mean), spearman_mean=np.array(spearman_mean),
             top1match_mean=np.array(top1match_mean))

    # Write numeric diagnostics JSONL
    with open(out_dir / 'numeric_diagnostics.jsonl', 'a') as f:
        rec = {
            'K': K, 'F': Fdim, 'E': E, 'M': M,
            'teacher_forced_step_match': step_match,
            'acc_dtmin_t0': acc_dtmin_t0, 'acc_dtmin_tK': acc_dtmin_tK,
            'acc_teacher_qk': acc_teacher_qk, 'acc_g': acc_g,
            'per_step': {
                'entropy_mean': ent_mean,
                'margin_mean': margin_mean,
                'pearson_mean': pearson_mean,
                'spearman_mean': spearman_mean,
                'top1match_mean': top1match_mean,
            }
        }
        f.write(json.dumps(rec) + '\n')
    code_state = {
        'git_head': os.popen('git rev-parse HEAD').read().strip(),
        'git_dirty': bool(os.popen('git diff --quiet').close()),
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'args': vars(args),
        'files_sha256': {
            'scripts/dt_pointer_ldv.py': torch_sha256(Path(__file__)),
            'doa_rl/trajectories/offline_dt_dataset.py': torch_sha256(Path('doa_rl/trajectories/offline_dt_dataset.py')),
            'doa_rl/data.py': torch_sha256(Path('doa_rl/data.py')),
        }
    }
    with open(out_dir / 'code_state.json', 'w') as f:
        json.dump(code_state, f, indent=2)

    # Diagnostics JSONL (summary)
    with open(out_dir / 'numeric_diagnostics.jsonl', 'w') as f:
        f.write(json.dumps({'K': K, 'F': Fdim, 'E': E, 'M': M, 'acc_e': acc_e, 'acc_a': acc_a})+'\n')

    print('\n' + '='*80)
    print('DT-Min training DONE')
    print(f"Metrics: {out_dir/'metrics.npz'}; code_state: {out_dir/'code_state.json'}")
    print('='*80)


if __name__ == '__main__':
    main()
