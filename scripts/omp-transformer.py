#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-Transformer (B) Routed-Attn Soft-OMP (no solver) with explicit tokens+mask
- Dictionary D is frozen from a VQ codebook (H prototypes) and W (speech bases)
- Train: soft routing (Gumbel-Softmax / entmax), learn only continuous params (tau/eta/projections/QK/FFN)
- Eval: hard routing (Top-K experts, Top-L atoms)
- ICL: inference is no weight updates; behavior from Prompt=(D, r0=y, optional S0)

Run:
  python full_transformer_routed_softomp_vq.py           # train (soft routing) then eval (hard routing)
"""

import math, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
from typing import List, Tuple

torch.manual_seed(0); np.random.seed(0)

# ---------------- 0) Data and Codebook (VQ; frozen) ----------------
def gen_W(F:int, M:int, seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); f=np.linspace(0,1,F); W=np.zeros((F,M), float)
    for k in range(M):
        n=rng.integers(2,4); spec=np.zeros(F,float)
        for _ in range(n):
            c=rng.uniform(0.1,0.9); w=rng.uniform(0.03,0.12); h=rng.uniform(0.6,1.6)
            spec += h*np.exp(-0.5*((f-c)/w)**2)
        spec=np.convolve(spec, np.ones(5)/5.0, mode="same")
        spec=np.maximum(spec, 0.0); W[:,k]=spec/(np.linalg.norm(spec)+1e-12)
    return W

def gen_H_from_angles(F:int, angles_deg:np.ndarray, seed:int)->np.ndarray:
    rng=np.random.default_rng(seed); f=np.linspace(0,1,F); H=np.zeros((F, len(angles_deg)), float)
    for i, ang in enumerate(angles_deg):
        slope = np.cos(np.deg2rad(ang))*rng.uniform(-0.4,0.4); base = 1.0 + slope*(f-0.5)
        c=(ang % 180)/180.0; w=0.08; bump=0.6*np.exp(-0.5*((f-c)/w)**2)
        h=np.convolve(base+bump, np.ones(7)/7.0, mode="same")
        h=np.maximum(h,0.05); H[:,i]=h/(np.linalg.norm(h)+1e-12)
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
    return np.concatenate(Hs, axis=1)  # [F, N]

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
    C=kmeans(H_pool.T, E, iters=10, seed=seed)  # [E, F]
    return C.T                                  # [F, E]

# ---------------- 1) Soft routing ops ----------------
def gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool=False) -> torch.Tensor:
    g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
    y = F.softmax((logits + g)/max(tau,1e-8), dim=-1)
    if hard:
        y_hard = torch.zeros_like(y); y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

def sparsemax(logits: torch.Tensor, dim: int=-1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim)+1, device=logits.device).view((1,)* (z.dim()-1) + (-1,))
    cumsum = z_sorted.cumsum(dim=dim)
    rho = torch.sum(z_sorted > (cumsum - 1)/k, dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, rho-1) - 1) / rho
    out = torch.clamp(z - tau, min=0.0)
    return out

def entmax15(logits: torch.Tensor, dim:int=-1, n_iter:int=50) -> torch.Tensor:
    # fixed-point entmax ~1.5 (good enough for routing). For stability, sparsemax() is also fine.
    eps=1e-12; x = logits/2
    for _ in range(n_iter):
        p = torch.clamp(x, min=0)**2
        Z = p.sum(dim=dim, keepdim=True)+eps
        x = ((logits + 2*p.sum(dim=dim, keepdim=True)/p.size(dim)) / 2).clamp(min=0)
    p = torch.clamp(x, min=0)**2
    return p / (p.sum(dim=dim, keepdim=True)+eps)

# ---------------- 2) Full-Transformer Routed Soft-OMP ----------------
class FullTransformerRoutedSoftOMP(nn.Module):
    """
    - Explicit tokens: [Residual token; Dictionary tokens]
    - TransformerEncoder (1-2 layers) with custom mask (R can attend D+R; each D attends itself only)
    - Train: soft routing (Gumbel/entmax). Eval: hard Top-K experts, Top-L atoms.
    """
    def __init__(self, F:int, E:int, M:int, d:int=None, nhead:int=4, nlayers:int=1,
                 steps:int=6, top_e:int=2, L:int=2, tau_e:float=0.5, tau_a:float=0.2, eta:float=0.5,
                 routing:str='gumbel'):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.P = E*M
        self.d = d if d is not None else F
        self.steps, self.top_e, self.L = steps, top_e, L
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta   = nn.Parameter(torch.tensor(float(eta)))
        self.routing = routing

        # Token projections + type embeddings
        self.P_R = nn.Linear(F, self.d, bias=False)
        self.P_D = nn.Linear(F, self.d, bias=False)
        nn.init.eye_(self.P_R.weight) if self.d==F else nn.init.xavier_uniform_(self.P_R.weight)
        nn.init.eye_(self.P_D.weight) if self.d==F else nn.init.xavier_uniform_(self.P_D.weight)
        self.type_R = nn.Parameter(torch.randn(self.d))
        self.type_D = nn.Parameter(torch.randn(self.d))

        # TransformerEncoder
        enc_layer = nn.TransformerEncoderLayer(d_model=self.d, nhead=nhead, dim_feedforward=4*self.d, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        # Optional Q/K on top of encoder outputs (not strictly needed; we can use dot(h_R, H_D))
        self.Wq = nn.Linear(self.d, self.d, bias=False)
        self.Wk = nn.Linear(self.d, self.d, bias=False)

    def _build_tokens(self, r: torch.Tensor, D: torch.Tensor):
        """
        r: [F], D: [F, P]
        return tokens T:[1+P, d], index slices for convenience
        """
        t_R = self.P_R(r) + self.type_R                 # [d]
        # Apply projection along feature dimension per atom (operate on columns)
        # Result shape [P, d]; add type embedding per token
        T_D = self.P_D(D.T) + self.type_D                # [P, d]
        T   = torch.cat([t_R[None,:], T_D], dim=0)     # [1+P, d]
        return T

    def _make_mask(self, S:int):
        """
        Build attention mask (SxS) so that:
          - R (index 0) can attend to all (0..P)
          - each D_j (index j>=1) can attend ONLY to itself (j)
        nn.Transformer expects float mask with -inf for disallowed.
        """
        mask = torch.full((S, S), float('-inf'))
        # R attends to all
        mask[0, :] = 0.0
        # each D attends itself
        for j in range(1, S):
            mask[j, j] = 0.0
        return mask  # [S,S]

    def _soft_picker(self, logits: torch.Tensor, tau: torch.Tensor, mode:str, hard:bool):
        if mode=='gumbel':
            return gumbel_softmax(logits, tau=float(tau.item()), hard=hard)
        elif mode=='entmax':
            return entmax15(logits, dim=-1)
        else:
            return F.softmax(logits / max(float(tau.item()),1e-8), dim=-1)

    def forward(self, y: torch.Tensor, D: torch.Tensor, train_mode: bool=True):
        """
        y:[F], D:[F,P]  (P=E*M).  Return x:[P], residual curve list[steps].
        """
        F, P = D.shape
        x = torch.zeros(P, device=D.device)
        r = y.clone()
        res_curve = []

        for _ in range(self.steps):
            # ---- Tokens & Encoder ----
            T = self._build_tokens(r, D)          # [1+P, d]
            S = T.size(0)
            mask = self._make_mask(S).to(T.device)
            H = self.encoder(T, mask=mask)        # [1+P, d]
            h_R = H[0]                            # [d]
            H_D = H[1:]                           # [P,d]

            # Scores per atom (use encoder outputs, dot with h_R)
            scores_atoms = ( self.Wk(H_D) @ self.Wq(h_R) ) / math.sqrt(self.d)   # [P]
            scores_atoms = scores_atoms.reshape(self.E, self.M)                  # [E,M]
            # Coarse expert scores (l2 pooling over atoms)
            scores_expert = torch.sqrt( (scores_atoms.abs()**2).sum(dim=1) + 1e-12 )  # [E]

            if train_mode:
                # soft routing
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)  # [E]
                w_all = torch.zeros(self.E, self.M, device=D.device)
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_atoms[e].abs(), self.tau_a, self.routing, hard=False)  # [M]
                    w_all[e] = w_e[e] * w_a_e
                w_all = w_all.reshape(-1)  # [P]
                g = (D.T @ r)              # [P]  use geometric gradient-like signal
                x = x + float(self.eta.item()) * (w_all * g)
            else:
                # hard routing: Top-K experts then Top-L atoms each
                kE = min(self.top_e, self.E)
                chosen_e = torch.topk(scores_expert, k=kE).indices.tolist()
                chosen_idx=[]
                for e in chosen_e:
                    kL = min(self.L, self.M)
                    chosen_a = torch.topk(scores_atoms[e].abs(), k=kL).indices.tolist()
                    chosen_idx += [int(e)*self.M + int(a) for a in chosen_a]
                chosen_idx = list(dict.fromkeys(chosen_idx))
                if len(chosen_idx)>0:
                    g = (D.T @ r)
                    x[chosen_idx] = x[chosen_idx] + float(self.eta.item()) * g[chosen_idx]
            # residual
            r = y - D @ x
            res_curve.append( float(torch.norm(r)) )

        return x, res_curve

# -------------------------- 3) Train / Eval --------------------------
def synth_batch(D_true: torch.Tensor, idx2: List[Tuple[int,int]], k_true:int=3, noise_std:float=0.02, B:int=12):
    F,P=D_true.shape
    Xb=[]; Yb=[]
    Ttheta_true = len({th for th,k in idx2})
    for _ in range(B):
        thetas = np.random.choice(Ttheta_true, size=k_true, replace=False)
        atoms=[]
        for th in thetas:
            idxs=[j for j,(t,k) in enumerate(idx2) if t==th]
            atoms.append(int(np.random.choice(idxs)))
        x_true = torch.zeros(P); x_true[atoms] = torch.from_numpy(0.6+0.8*np.random.rand(k_true)).float()
        y = D_true @ x_true + noise_std*torch.randn(F)
        Xb.append(x_true); Yb.append(y)
    return torch.stack(Yb,0), torch.stack(Xb,0)

def train_epoch(model: FullTransformerRoutedSoftOMP, D: torch.Tensor, D_true: torch.Tensor, idx2_true: List[Tuple[int,int]],
                opt: torch.optim.Optimizer, batch_size:int=12, iters:int=120, device='cpu'):
    model.train()
    losses=[]
    for it in range(iters):
        yb, xb = synth_batch(D_true.to(device), idx2_true, B=batch_size)
        yb = yb.to(device)
        rec_loss=0.0; mono_loss=0.0
        for b in range(yb.size(0)):
            x_hat, r_curve = model(yb[b], D.to(device), train_mode=True)
            y_hat = D.to(device) @ x_hat
            rec_loss = rec_loss + F.mse_loss(y_hat, yb[b])
            rc = torch.tensor(r_curve, device=device)
            diffs = rc[1:] - rc[:-1]
            mono_loss = mono_loss + torch.relu(diffs).sum()
        rec_loss /= yb.size(0); mono_loss /= yb.size(0)
        loss = rec_loss + 0.2*mono_loss
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
        if (it+1) % max(10, iters//5) == 0:
            print(f"[iter {it+1}/{iters}] loss={loss.item():.4f} rec={rec_loss.item():.4f} mono={mono_loss.item():.4f}  tau_e={float(model.tau_e.item()):.3f} tau_a={float(model.tau_a.item()):.3f} eta={float(model.eta.item()):.3f}")
    return losses

@torch.no_grad()
def eval_curves(model: FullTransformerRoutedSoftOMP, D: torch.Tensor, D_true: torch.Tensor, idx2_true: List[Tuple[int,int]],
                steps:int, n_eval:int=12, device='cpu'):
    model.eval()
    curves=np.zeros(steps)
    for _ in range(n_eval):
        yb, xb = synth_batch(D_true.to(device), idx2_true, B=1)
        _, rc = model(yb[0].to(device), D.to(device), train_mode=False)  # hard routing
        curves += np.array(rc)
    return curves / n_eval

def mutual_coherence(D: torch.Tensor)->float:
    G=(D.T@D).cpu().numpy(); np.fill_diagonal(G,0); return float(np.max(np.abs(G)))

# -------------------------- 4) Main --------------------------
def main():
    parser = argparse.ArgumentParser(description='Full-Transformer Routed Soft-OMP demo')
    parser.add_argument('--out_dir', type=str, default=None, help='Directory to save curves.png and metrics.npz')
    parser.add_argument('--device', type=str, default='cpu', help='cpu|mps|cuda')
    args = parser.parse_args()

    device = args.device
    # Scales （可調）——保持 modest 以便快速跑
    F, M, E = 40, 8, 12
    steps, top_e, L = 6, 2, 2

    # 1) 混雜角度資料 -> VQ 碼本（凍結）
    H_pool = mixed_H_pool(F, spacings=[1,2,3,5], per_spacing=72, seed=10)  # F x N
    H_vq   = build_vq_codebook(H_pool, E=E, seed=0)                        # F x E

    # 2) 組 D（ID/OOD 可以把 W 換 seed 以模擬分佈差異）
    W_id = gen_W(F,M,seed=20); D_id, idx2_id = build_D(W_id, H_vq)          # [F, E*M]
    W_od = gen_W(F,M,seed=21); D_od, idx2_od = build_D(W_od, H_vq)

    print(f"Mutual coherence μ: ID={mutual_coherence(D_id):.3f}, OOD={mutual_coherence(D_od):.3f}")

    # 3) 建立 Full-Transformer 化的 routed soft-OMP（碼本凍結；只學連續小參數）
    model = FullTransformerRoutedSoftOMP(F=F, E=E, M=M, d=F, nhead=4, nlayers=1,
                                         steps=steps, top_e=top_e, L=L,
                                         tau_e=0.5, tau_a=0.2, eta=0.5,
                                         routing='gumbel').to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-3)

    # 4) 訓練（soft 路由）
    print("== Training (soft routing) on ID ==")
    losses = train_epoch(model, D_id, D_id, idx2_id, opt, batch_size=12, iters=120, device=device)

    # 5) 推論（硬路由），畫 ID/OOD 殘差曲線
    print("== Eval (hard routing) ==")
    curves_id  = eval_curves(model, D_id, D_id, idx2_id, steps=steps, n_eval=12, device=device)
    curves_ood = eval_curves(model, D_od, D_od, idx2_od, steps=steps, n_eval=12, device=device)

    fig,axs=plt.subplots(1,3,figsize=(12,3))
    axs[0].plot(np.arange(1,len(losses)+1), losses); axs[0].set_title("Training loss"); axs[0].set_xlabel("iter"); axs[0].set_ylabel("loss")
    axs[1].plot(np.arange(1,steps+1), curves_id, label="ID"); axs[1].set_title("Residual vs step (ID)"); axs[1].legend(); axs[1].set_xlabel("step"); axs[1].set_ylabel("||r||")
    axs[2].plot(np.arange(1,steps+1), curves_ood, label="OOD"); axs[2].set_title("Residual vs step (OOD)"); axs[2].legend(); axs[2].set_xlabel("step"); axs[2].set_ylabel("||r||")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, 'curves.png')
        fig.savefig(fig_path, dpi=150)
        np.savez(os.path.join(args.out_dir, 'metrics.npz'),
                 losses=np.array(losses),
                 curves_id=np.array(curves_id),
                 curves_ood=np.array(curves_ood))
        print(f'Saved figure to {fig_path}')
    else:
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
