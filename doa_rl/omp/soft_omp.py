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
    """Routed soft-OMP with attention-style scoring for training and correlation-based hard eval.

    - Train: soft expert/atom routing with temperatures tau_e, tau_a, gradient-style update x += eta * (w ⊙ g).
    - Eval: greedy selection by correlations g=D^T r with sequential coordinate descent updates to ensure non-increasing residual.
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
    ):
        super().__init__()
        self.F, self.E, self.M = F, E, M
        self.steps, self.top_e, self.L = steps, top_e, L
        self.tau_e = nn.Parameter(torch.tensor(float(tau_e)))
        self.tau_a = nn.Parameter(torch.tensor(float(tau_a)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))

        d = F
        self.P_R = nn.Linear(F, d, bias=False)
        self.P_D = nn.Linear(F, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.P_R.weight)
        nn.init.eye_(self.P_D.weight)
        nn.init.eye_(self.Wq.weight)
        nn.init.eye_(self.Wk.weight)
        self.routing = routing

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
        # Pre-embed atoms once
        D_emb = self.P_D(D.T).T
        D_emb = D_emb / (D_emb.norm(dim=0, keepdim=True) + 1e-12)
        for _ in range(self.steps):
            # Attention-like scoring
            q = self.Wq(self.P_R(r))
            K = self.Wk(D_emb.T).T
            scale = 1.0 / math.sqrt(float(D_emb.size(0)))
            scores_atoms = (K * q.view(-1, 1)).sum(dim=0) * scale
            scores_expert = (
                scores_atoms.view(self.E, self.M).abs().pow(2).sum(dim=1).sqrt()
            )

            if train_mode:
                # Soft routing over experts and atoms
                w_e = self._soft_picker(scores_expert, self.tau_e, self.routing, hard=False)
                scores_a = scores_atoms.view(self.E, self.M).abs()
                w_a = []
                for e in range(self.E):
                    w_a_e = self._soft_picker(scores_a[e], self.tau_a, self.routing, hard=False)
                    w_a.append(w_a_e * w_e[e])
                w_all = torch.stack(w_a, dim=0).reshape(-1)
                g = D.T @ r
                x = x + self.eta * (w_all * g)
            else:
                # Hard greedy by correlations with sequential coordinate descent
                g = D.T @ r
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

