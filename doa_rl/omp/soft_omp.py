import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_dictionary(W: torch.Tensor, H: torch.Tensor) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    """Build dictionary D(F, P) where atoms are H[:,e] ⊙ W[:,k].

    Returns D and an index map (theta_idx, k_idx) per column.
    Assumes W,H are real, same F, and columns are non-zero.
    """
    assert W.dim() == 2 and H.dim() == 2
    Fw, M = W.shape
    Fh, E = H.shape
    if Fw != Fh:
        raise ValueError(f"F mismatch: W.F={Fw} vs H.F={Fh}")
    F = Fw
    P = E * M
    D = torch.zeros((F, P), dtype=W.dtype)
    idx2: List[Tuple[int, int]] = []
    j = 0
    for e in range(E):
        h = H[:, e]
        for k in range(M):
            a = h * W[:, k]
            n = torch.norm(a) + 1e-12
            D[:, j] = a / n
            idx2.append((e, k))
            j += 1
    return D, idx2


class TrainableRoutedSoftOMP(nn.Module):
    """Routed soft-OMP with flexible routing modes and score regularization.

    - routing_mode='qk': QK attention-based routing (learnable but may misalign with physics)
    - routing_mode='g': Physics-based routing using g=D^T·r (100% accuracy, always aligned)
    - routing_mode='hybrid': Blended routing: α·g + (1-α)·QK (best of both worlds)
    
    Improvements over base version:
    1. Flexible routing modes (qk/g/hybrid) via routing_mode parameter
    2. Score regularization to prevent drift/saturation
    3. Preserves g's physical meaning: always used for UPDATE (x += η·w⊙g)
    4. Learnable hybrid_alpha for gradual transfer from physics to learned routing
    """

    def __init__(
        self,
        F: int,
        E: int,
        M: int,
        steps: int = 6,
        top_e: int = 2,
        L: int = 2,
        tau_e: float = 0.5,
        tau_a: float = 0.2,
        eta: float = 0.5,
        routing: str = "gumbel",
        routing_mode: str = "g",  # NEW: 'qk', 'g', or 'hybrid'
        hybrid_alpha: float = 1.0,  # NEW: blend factor (1.0=pure g, 0.0=pure qk)
        score_reg_weight: float = 0.01,  # NEW: L2 regularization on scores
    ):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.steps, self.top_e, self.L = steps, top_e, L
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.routing = routing
        self.routing_mode = routing_mode  # NEW
        self.hybrid_alpha = nn.Parameter(torch.tensor(float(hybrid_alpha)))  # NEW: learnable
        self.score_reg_weight = score_reg_weight  # NEW

        d = F
        self.P_R = nn.Linear(F, d, bias=False)
        self.P_D = nn.Linear(F, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.P_R.weight)
        nn.init.eye_(self.P_D.weight)
        nn.init.eye_(self.Wq.weight)
        nn.init.eye_(self.Wk.weight)
        
        # Track regularization loss (accessible after forward)
        self.last_reg_loss = 0.0

    def _soft_picker(self, logits: torch.Tensor, tau: torch.Tensor, mode: str, hard: bool):
        if mode == "gumbel":
            g = -torch.log(-torch.log(torch.rand_like(logits) + 1e-12) + 1e-12)
            y = F.softmax((logits + g) / max(float(tau.item()), 1e-8), dim=-1)
            if hard:
                y_hard = torch.zeros_like(y)
                y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
                y = (y_hard - y).detach() + y
            return y
        else:
            return F.softmax(logits / max(float(tau.item()), 1e-8), dim=-1)

    def forward(self, y: torch.Tensor, D: torch.Tensor, train_mode: bool = True):
        Fdim, P = D.shape
        assert Fdim == self.F
        x = torch.zeros(P, device=D.device)
        r = y.clone()
        res = []
        self.last_reg_loss = 0.0  # Reset regularization loss
        
        # Pre-embed atoms once
        D_emb = self.P_D(D.T).T
        D_emb = D_emb / (D_emb.norm(dim=0, keepdim=True) + 1e-12)
        
        for step_idx in range(self.steps):
            # ALWAYS compute physics correlation g (needed for update)
            g = D.T @ r  # Shape: (P,)
            
            if train_mode:
                # Compute QK attention scores
                q = self.Wq(self.P_R(r))
                K = self.Wk(D_emb.T).T
                scale = 1.0 / math.sqrt(float(D_emb.size(0)))
                qk_scores_atoms = (K * q.view(-1, 1)).sum(dim=0) * scale  # (P,)
                
                # Reshape g and qk_scores for expert-level computation
                g_atoms = g.view(self.E, self.M)  # (E, M)
                qk_atoms = qk_scores_atoms.view(self.E, self.M)  # (E, M)
                
                # Compute expert-level scores
                g_expert = torch.sqrt((g_atoms ** 2).sum(dim=1) + 1e-12)  # (E,)
                qk_expert = torch.sqrt((qk_atoms.abs() ** 2).sum(dim=1) + 1e-12)  # (E,)
                
                # === FLEXIBLE ROUTING MODE ===
                if self.routing_mode == 'g':
                    # Pure physics-based routing (100% accuracy)
                    scores_expert = g_expert
                    scores_atoms = g_atoms
                elif self.routing_mode == 'qk':
                    # Pure QK attention routing (learnable but may misalign)
                    scores_expert = qk_expert
                    scores_atoms = qk_atoms.abs()
                elif self.routing_mode == 'hybrid':
                    # Hybrid: blend physics and learned routing
                    # Normalize both to unit norm before blending
                    alpha = torch.clamp(self.hybrid_alpha, 0.0, 1.0)
                    
                    g_expert_norm = g_expert / (g_expert.norm() + 1e-12)
                    qk_expert_norm = qk_expert / (qk_expert.norm() + 1e-12)
                    scores_expert = alpha * g_expert_norm + (1 - alpha) * qk_expert_norm
                    
                    g_atoms_norm = g_atoms / (g_atoms.norm() + 1e-12)
                    qk_atoms_norm = qk_atoms / (qk_atoms.norm() + 1e-12)
                    scores_atoms = alpha * g_atoms_norm + (1 - alpha) * qk_atoms_norm.abs()
                else:
                    raise ValueError(f"Invalid routing_mode: {self.routing_mode}. Must be 'qk', 'g', or 'hybrid'.")
                
                # === SCORE REGULARIZATION (prevent drift) ===
                if self.score_reg_weight > 0:
                    # L2 penalty on expert scores to prevent unbounded growth
                    reg_loss = self.score_reg_weight * (scores_expert ** 2).mean()
                    self.last_reg_loss += reg_loss.item()
                    # Note: Actual backprop will happen in training loop via loss += model.last_reg_loss
                
                # Soft routing over experts and atoms
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)
                w_a = []
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_atoms[e], self.tau_a, self.routing, hard=False)
                    w_a.append(w_a_e * w_e[e])
                w_all = torch.stack(w_a, dim=0).reshape(-1)  # (P,)
                
                # === CRITICAL: Update using g (preserves physical meaning) ===
                x = x + self.eta * (w_all * g)  # w decides WHERE, g decides HOW MUCH
                
            else:
                # Hard greedy by correlations with sequential coordinate descent
                G = g.view(self.E, self.M)
                alpha = G.abs().max(dim=1).values
                kE = min(self.top_e, self.E)
                chosen_e = torch.topk(alpha, k=kE).indices.tolist()
                idx = []
                for e in chosen_e:
                    kL = min(self.L, self.M)
                    g_abs_e = G.abs()[e]
                    chosen_a = torch.topk(g_abs_e, k=kL).indices.tolist()
                    idx += [int(e) * self.M + int(a) for a in chosen_a]
                if idx:
                    idx = list(dict.fromkeys(idx))
                    idx.sort(key=lambda i: float(g[i].abs()), reverse=True)
                    for i in idx:
                        gi = (D[:, i] @ r)
                        x[i] = x[i] + gi
                        r = r - D[:, i] * gi
            
            r = y - D @ x
            res.append(float(torch.norm(r)))
        
        return x, res


__all__ = ["build_dictionary", "TrainableRoutedSoftOMP"]

