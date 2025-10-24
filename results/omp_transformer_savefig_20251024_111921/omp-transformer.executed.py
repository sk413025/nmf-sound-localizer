#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainable Routed-Attn Soft-OMP (no solver), with soft routing (Gumbel-Softmax / entmax)
- Codebook (VQ) is FROZEN: we learn only continuous params (tau, eta, projections, Q/K)
- Train: soft routing (mixture weights) -> gradients flow; Eval: hard Top-K routing
- Loss: reconstruction + residual monotonicity (+ optional orthogonality)
"""

import math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from typing import List, Tuple

torch.manual_seed(0); np.random.seed(0)

# -------------------------- 0) Utilities: data & codebook --------------------------
def gen_W(F:int, M:int, seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); f=np.linspace(0,1,F); W=np.zeros((F,M), float)
    for k in range(M):
        n=rng.integers(2,4); spec=np.zeros(F,float)
        for _ in range(n):
            c=rng.uniform(0.1,0.9); w=rng.uniform(0.03,0.12); h=rng.uniform(0.6,1.6)
            spec += h*np.exp(-0.5*((f-c)/w)**2)
        spec=np.convolve(spec, np.ones(5)/5.0, mode="same"); spec=np.maximum(spec,0.0)
        W[:,k]=spec/(np.linalg.norm(spec)+1e-12)
    return W

def gen_H_from_angles(F:int, angles_deg:np.ndarray, seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); f=np.linspace(0,1,F); H=np.zeros((F,len(angles_deg)), float)
    for i, ang in enumerate(angles_deg):
        slope = np.cos(np.deg2rad(ang))*rng.uniform(-0.4,0.4); base  = 1.0 + slope*(f-0.5)
        c = (ang % 180)/180.0; w = 0.08
        bump = 0.6*np.exp(-0.5*((f-c)/w)**2)
        h = np.convolve(base+bump, np.ones(7)/7.0, mode="same")
        h = np.maximum(h, 0.05); H[:,i] = h/(np.linalg.norm(h)+1e-12)
    return H

def build_D(W:np.ndarray, H:np.ndarray)->Tuple[torch.Tensor, List[Tuple[int,int]]]:
    F,M=W.shape; _,T=H.shape; P=M*T
    D=np.zeros((F,P), float); idx2=[]
    j=0
    for th in range(T):
        for k in range(M):
            a=H[:,th]*W[:,k]; a=a/(np.linalg.norm(a)+1e-12)
            D[:,j]=a; idx2.append((th,k)); j+=1
    return torch.from_numpy(D).float(), idx2

def mixed_H_pool(F:int, spacings=[1,2,3,5], per_spacing=72, seed=0):
    rng=np.random.default_rng(seed); Hs=[]
    for sp in spacings:
        offset = rng.uniform(0, sp); angs=(np.arange(per_spacing)*sp + offset) % 360.0
        Hs.append(gen_H_from_angles(F, angs, seed+sp))
    return np.concatenate(Hs, axis=1)  # F x N

def kmeans(X:np.ndarray, K:int, iters:int=12, seed:int=0)->np.ndarray:
    rng=np.random.default_rng(seed); N,d=X.shape
    C=X[rng.choice(N,size=K,replace=False)].copy()
    for _ in range(iters):
        dots=X@C.T; dist=(X*X).sum(1,keepdims=True) + (C*C).sum(1)[None,:] - 2*dots
        assign=np.argmin(dist,axis=1)
        for k in range(K):
            sel=(assign==k); C[k]=X[sel].mean(0) if np.any(sel) else X[rng.integers(0,N)]
        C=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-12)
    return C

def build_vq_codebook(H_pool:np.ndarray, E:int=12, seed:int=0)->np.ndarray:
    C=kmeans(H_pool.T, E, iters=10, seed=seed)  # E x F
    return C.T                                  # F x E

# -------------------------- 1) Soft routing: Gumbel-Softmax / entmax --------------------------
def gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool=False) -> torch.Tensor:
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    y = F.softmax((logits + g)/max(tau,1e-8), dim=-1)
    if hard:
        y_hard = torch.zeros_like(y); y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y  # (..., C)

def sparsemax(logits: torch.Tensor, dim: int=-1) -> torch.Tensor:
    # Martins & Astudillo 2016
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim)+1, device=logits.device).view((1,)* (z.dim()-1) + (-1,))
    cumsum = z_sorted.cumsum(dim=dim)
    rho = torch.sum(z_sorted > (cumsum - 1)/k, dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, rho-1) - 1) / rho
    out = torch.clamp(z - tau, min=0.0)
    return out

def entmax15(logits: torch.Tensor, dim:int=-1, n_iter:int=50) -> torch.Tensor:
    # Peters & Martins 2019: a fixed-point implementation for alpha=1.5
    # This is a light implementation good enough for routing; for stability you can fall back to sparsemax.
    eps = 1e-12
    d = logits.size(dim)
    x = logits / 2  # init
    for _ in range(n_iter):
        p = torch.clamp(x, min=0)**2
        Z = p.sum(dim=dim, keepdim=True) + eps
        x = ( (logits + 2*p.sum(dim=dim, keepdim=True)/d) / 2 ).clamp(min=0)
    p = torch.clamp(x, min=0)**2
    return p / (p.sum(dim=dim, keepdim=True)+eps)

# -------------------------- 2) Trainable routed-attn soft-OMP (VQ frozen) --------------------------
class TrainableRoutedSoftOMP(nn.Module):
    """
    - Dictionary D (from VQ) is frozen outside the model.
    - Routing mode: 'gumbel' (Gumbel-Softmax) or 'entmax' (sparse entmax15).
    - Train: soft routing; Eval: hard Top-K on experts/atoms.
    """
    def __init__(self, F:int, E:int, M:int, steps:int=6, top_e:int=2, L:int=2,
                 tau_e:float=0.5, tau_a:float=0.2, eta:float=0.5, routing:str='gumbel'):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.steps, self.top_e, self.L = steps, top_e, L
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta   = nn.Parameter(torch.tensor(float(eta)))
        # light projections (learnable)
        d = F
        self.P_R = nn.Linear(F, d, bias=False)
        self.P_D = nn.Linear(F, d, bias=False)
        self.Wq  = nn.Linear(d, d, bias=False)
        self.Wk  = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.P_R.weight); nn.init.eye_(self.P_D.weight)
        nn.init.eye_(self.Wq.weight);  nn.init.eye_(self.Wk.weight)
        self.routing = routing

    def _pool_group_logits(self, D: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # smooth group score: l2 pooling over atoms in group (instead of max, to keep gradients)
        # D: [F, P=E*M]; r:[F]
        F, P = D.shape
        corr = (D.T @ r).view(self.E, self.M)         # [E, M]
        scores = torch.sqrt( (corr**2).sum(dim=1) + 1e-12 )  # [E]
        return scores

    def _soft_picker(self, logits: torch.Tensor, tau: torch.Tensor, mode:str, hard:bool):
        if mode == 'gumbel':
            return gumbel_softmax(logits, tau=float(tau.item()), hard=hard)
        elif mode == 'entmax':
            # entmax yields sparse probs; no hard option here (use top-k at eval)
            return entmax15(logits, dim=-1)
        else:
            return F.softmax(logits / max(float(tau.item()),1e-8), dim=-1)

    def forward(self, y: torch.Tensor, D: torch.Tensor, train_mode: bool=True):
        """
        y: [F]; D: [F, E*M] frozen; returns x, residual curve
        Train: soft routing; Eval: hard Top-K routing
        """
        F, P = D.shape; x = torch.zeros(P, device=D.device)
        r = y.clone()
        res = []
        # Pre-embed dictionary atoms for attention-like scoring (optional)
        # Apply linear projection along the feature (F) dimension for each atom (column)
        D_emb = self.P_D(D.T).T               # [F, P]
        D_emb = D_emb / (D_emb.norm(dim=0, keepdim=True) + 1e-12)  # unit-norm
        for _ in range(self.steps):
            # ---- Coarse (experts): get logits per expert ----
            # attention-like scoring using embeddings
            q = self.Wq( self.P_R(r) )            # [F]
            # Apply Wk along the feature dimension per atom
            K = self.Wk( D_emb.T ).T              # [F, P]
            scores_atoms = (K * q.view(-1,1)).sum(dim=0)   # [P]
            scores_expert = scores_atoms.view(self.E, self.M).abs().pow(2).sum(dim=1).sqrt()  # [E]

            if train_mode:
                # soft routing over experts
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)    # [E]
                # for each expert, soft routing over atoms
                scores_a = scores_atoms.view(self.E, self.M).abs()  # [E,M]
                w_a = []
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_a[e], self.tau_a, self.routing, hard=False)  # [M]
                    w_a.append(w_a_e * w_e[e])
                w_all = torch.stack(w_a, dim=0).reshape(-1)  # [E*M]
                # gradient-like update on all atoms with soft weights
                g = D.T @ r                                  # [P]
                x = x + self.eta * (w_all * g)
            else:
                # hard Top-K routing over experts then atoms
                kE = min(self.top_e, self.E)
                chosen_e = torch.topk(scores_expert, k=kE).indices.tolist()
                idx = []
                for e in chosen_e:
                    kL = min(self.L, self.M)
                    # pick atoms within expert e
                    sc_e = scores_atoms.view(self.E,self.M).abs()[e]
                    chosen_a = torch.topk(sc_e, k=kL).indices.tolist()
                    idx += [int(e)*self.M + int(a) for a in chosen_a]
                idx = list(dict.fromkeys(idx))
                if len(idx)>0:
                    g = D.T @ r
                    x[idx] = x[idx] + float(self.eta.item()) * g[idx]
            r = y - D @ x
            res.append( float(torch.norm(r)) )
        return x, res

# -------------------------- 3) Training / Eval loops --------------------------
def synth_batch(D_true: torch.Tensor, idx2: List[Tuple[int,int]], k_true:int=3, noise_std:float=0.02, B:int=16):
    F,P=D_true.shape
    Xb=[]; Yb=[]
    Ttheta_true = len({th for th,k in idx2})
    for _ in range(B):
        # sample k_true experts (groups), one atom per expert
        thetas = np.random.choice(Ttheta_true, size=k_true, replace=False)
        atoms=[]
        for th in thetas:
            idxs = [j for j,(t,k) in enumerate(idx2) if t==th]
            atoms.append(int(np.random.choice(idxs)))
        x_true = torch.zeros(P); x_true[atoms] = torch.from_numpy(0.6+0.8*np.random.rand(k_true)).float()
        y = D_true @ x_true + noise_std*torch.randn(F)
        Xb.append(x_true); Yb.append(y)
    return torch.stack(Yb,0), torch.stack(Xb,0)  # Y:[B,F], X:[B,P]

def train_epoch(model: TrainableRoutedSoftOMP, D: torch.Tensor, D_true: torch.Tensor, idx2_true: List[Tuple[int,int]],
                opt: torch.optim.Optimizer, batch_size:int=16, iters:int=200, device='cpu'):
    model.train()
    losses=[]
    for it in range(iters):
        yb, xb = synth_batch(D_true.to(device), idx2_true, B=batch_size)
        yb = yb.to(device)
        # forward (soft routing)
        rec_loss = 0.0; mono_loss = 0.0
        for b in range(yb.size(0)):
            x_hat, r_curve = model(yb[b], D.to(device), train_mode=True)
            y_hat = D.to(device) @ x_hat
            rec_loss = rec_loss + F.mse_loss(y_hat, yb[b])
            # monotonic residual
            rc = torch.tensor(r_curve, device=device)
            diffs = rc[1:] - rc[:-1]
            mono_loss = mono_loss + torch.relu(diffs).sum()
        rec_loss = rec_loss / yb.size(0)
        mono_loss = mono_loss / yb.size(0)
        loss = rec_loss + 0.2 * mono_loss

        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (it+1) % max(10, iters//5) == 0:
            print(f"[iter {it+1}/{iters}] loss={loss.item():.4f} rec={rec_loss.item():.4f} mono={mono_loss.item():.4f}  tau_e={float(model.tau_e.item()):.3f} tau_a={float(model.tau_a.item()):.3f} eta={float(model.eta.item()):.3f}")
    return losses

@torch.no_grad()
def eval_curves(model: TrainableRoutedSoftOMP, D: torch.Tensor, D_true: torch.Tensor, idx2_true: List[Tuple[int,int]],
                steps:int, n_eval:int=20, device='cpu'):
    model.eval()
    curves = np.zeros(steps)
    for _ in range(n_eval):
        yb, xb = synth_batch(D_true.to(device), idx2_true, B=1)
        x_hat, r_curve = model(yb[0].to(device), D.to(device), train_mode=False)  # hard routing
        curves += np.array(r_curve)
    return curves / n_eval

# -------------------------- 4) Main: build VQ dictionary, train & eval --------------------------
def main():
    parser = argparse.ArgumentParser(description="Trainable Routed-Attn Soft-OMP demo")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save outputs (figure, metrics)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: cpu|mps|cuda")
    args = parser.parse_args()

    device = args.device
    # Scales (kept modest for quick runs)
    F, M, E = 40, 8, 12
    steps, top_e, L = 6, 2, 2

    # Build mixed-angle H pool (1°,2°,3°,5°), learn VQ codebook (FROZEN)
    H_pool = mixed_H_pool(F, spacings=[1,2,3,5], per_spacing=72, seed=10)  # F x N
    H_vq   = build_vq_codebook(H_pool, E=E, seed=0)                        # F x E
    # Build dictionaries (ID true dict uses the same VQ H_vq here; OOD 可改 seed)
    W_id = gen_W(F,M,seed=20); D_id, idx2_id = build_D(W_id, H_vq)          # [F, E*M]
    W_od = gen_W(F,M,seed=21); D_od, idx2_od = build_D(W_od, H_vq)

    # Trainable model (soft routing during train)
    model = TrainableRoutedSoftOMP(F=F, E=E, M=M, steps=steps, top_e=top_e, L=L,
                                   tau_e=0.5, tau_a=0.2, eta=0.5, routing='gumbel').to(device)

    # Optimizer
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)

    # Train on ID (soft routing)
    print("== Training (soft routing) on ID ==")
    losses = train_epoch(model, D_id, D_id, idx2_id, opt, batch_size=12, iters=120, device=device)

    # Eval (hard routing curves)
    print("== Eval (hard routing) curves ==")
    curves_id  = eval_curves(model, D_id, D_id, idx2_id, steps=steps, n_eval=12, device=device)
    curves_ood = eval_curves(model, D_od, D_od, idx2_od, steps=steps, n_eval=12, device=device)

    # Plot loss & curves
    fig,axs = plt.subplots(1,3, figsize=(12,3))
    axs[0].plot(np.arange(1,len(losses)+1), losses); axs[0].set_title("Training loss"); axs[0].set_xlabel("iter"); axs[0].set_ylabel("loss")
    axs[1].plot(np.arange(1,steps+1), curves_id,  label="ID");  axs[1].set_title("Residual vs step (ID)");  axs[1].set_xlabel("step"); axs[1].set_ylabel("||r||"); axs[1].legend()
    axs[2].plot(np.arange(1,steps+1), curves_ood, label="OOD"); axs[2].set_title("Residual vs step (OOD)"); axs[2].set_xlabel("step"); axs[2].set_ylabel("||r||"); axs[2].legend()
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "curves.png")
        fig.savefig(fig_path, dpi=150)
        # also save basic metrics for reproducibility
        np.savez(os.path.join(args.out_dir, "metrics.npz"),
                 losses=np.array(losses),
                 curves_id=np.array(curves_id),
                 curves_ood=np.array(curves_ood))
        print(f"Saved figure to {fig_path}")
    else:
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
