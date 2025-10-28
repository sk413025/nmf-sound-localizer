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
        # Residual projection
        self.P_R = nn.Linear(F, d_model, bias=False)
        # RTG + step/budget projections
        self.proj_rtg = nn.Linear(2, d_model)
        self.proj_step = nn.Linear(2, d_model)
        self.ln = nn.LayerNorm(d_model)
        # Time encoder
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)
        # Query for pointer
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        # Keys for experts/atoms (set at runtime via register_buffer)
        self.register_buffer('K_e', torch.empty(0))   # (E, d)
        self.register_buffer('K_em', torch.empty(0))  # (E, M, d)

    def set_keys_from_D(self, D: torch.Tensor):
        # Build expert and per-atom F-vectors, then project to d_model
        Fdim, P = D.shape
        E = self.E; M = self.M
        D_em = D.view(Fdim, E, M)
        # Expert prototype as L2 aggregation over atoms
        D_e = torch.linalg.norm(D_em, dim=2)  # (F,E)
        K_e = D_e.T @ self.P_R.weight.T  # (E,d) using same projection matrix weight^T
        # Atom keys
        K_em = torch.einsum('fem,df->emd', D_em, self.P_R.weight)  # (E,M,d)
        self.K_e = torch.nn.functional.normalize(K_e, dim=-1)
        self.K_em = torch.nn.functional.normalize(K_em, dim=-1)

    def forward(self, R_seq: torch.Tensor, RTG_seq: torch.Tensor, STEP_seq: torch.Tensor, causal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inputs: (B,K,F), (B,K,2), (B,K,2)
        B, K, Fdim = R_seq.shape
        r_tok = self.P_R(R_seq)                         # (B,K,d)
        rtg_tok = self.proj_rtg(RTG_seq)                # (B,K,d)
        step_tok = self.proj_step(STEP_seq)             # (B,K,d)
        h = self.ln(r_tok + rtg_tok + step_tok)         # (B,K,d)
        Ht = self.encoder(h, mask=causal_mask)          # (B,K,d)
        Q = self.Wq(Ht)                                 # (B,K,d)
        # Expert scores: dot(Q, K_e)
        scores_e = torch.einsum('bkd,ed->bke', Q, self.K_e)  # (B,K,E)
        # Atom scores within true expert will be selected by gather at loss time
        scores_em = torch.einsum('bkd,emd->bkem', Q, self.K_em).reshape(B, K, self.E, self.M)
        return scores_e, scores_em


def generate_causal_mask(K: int, device: torch.device) -> torch.Tensor:
    m = torch.full((K, K), float('-inf'), device=device)
    m = torch.triu(m, diagonal=1)  # allow self and past
    return m


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
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print('='*80); print('DT-Min Trainer (hierarchical pointer)'); print('='*80)
    print(f"Traj dir: {args.traj_dir}\nOut dir: {out_dir}")

    manifest = load_manifest(args.traj_dir)
    trajs = load_trajectories(args.traj_dir)

    # Rebuild dataset and dictionary using manifest
    angles = manifest['angles']
    H, W = load_H_W(manifest['h_path'], manifest['w_path'], manifest['atom_reduce']['selected_indices'], device='cpu')
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

    # Training
    hist = []
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
            opt.zero_grad(); loss.backward(retain_graph=True); opt.step()
            total += float(loss.item()); n += 1
        avg = total / max(n,1)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg:.4f}")
        hist.append({'epoch': epoch+1, 'loss': avg})

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

    # Save metrics and code_state
    np.savez(out_dir / 'metrics.npz', history=np.array(hist, dtype=object), acc_e=acc_e, acc_a=acc_a)
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
