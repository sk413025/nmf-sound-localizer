from typing import Dict, Any
import torch

from nmf_localizer.core.localizer import NMFSoundLocalizer


def _is_z_update(Ybar: torch.Tensor, W: torch.Tensor, n_iter: int = 50, l1: float = 0.0) -> torch.Tensor:
    eps = 1e-8
    _, K = W.shape
    z = torch.full((K,), 1.0 / K, dtype=W.dtype)
    Ybar = torch.clamp(Ybar, min=eps)
    Wp = torch.clamp(W, min=eps)
    for _ in range(n_iter):
        Yhat = torch.clamp(Wp @ z, min=eps)
        num = (Wp.t() @ (Ybar / (Yhat ** 2)))
        den = (Wp.t() @ (1.0 / Yhat)) + l1
        z = z * torch.clamp(num / torch.clamp(den, min=eps), min=eps)
        z = z / torch.clamp(z.sum(), min=eps)
    return z


class AdvantageComputer:
    def __init__(self, localizer: NMFSoundLocalizer, W: torch.Tensor, H: torch.Tensor,
                 s_mode: str = "S1", nmf_iter: int = 50, nmf_l1: float = 0.0,
                 add_phys_feature: bool = True):
        self.localizer = localizer
        self.W = W.detach().cpu()
        self.H = H.detach().cpu()  # (F,D)
        self.s_mode = s_mode.upper()
        self.nmf_iter = nmf_iter
        self.nmf_l1 = nmf_l1
        self.add_phys_feature = add_phys_feature

    def __call__(self, Y: torch.Tensor) -> Dict[str, Any]:
        device = self.localizer.device
        Y = Y.to(device)
        X, info = self.localizer.factorize(Y)
        Y_hat = self.localizer.A @ X

        D = self.localizer.n_directions
        K = self.localizer.n_components
        group_norms = torch.zeros(D, device=device)
        for d in range(D):
            sidx = d * K
            eidx = (d + 1) * K
            X_d = X[sidx:eidx, :]
            group_norms[d] = torch.sum(torch.abs(X_d))

        Ybar = torch.mean(Y, dim=1).detach().cpu()
        if self.s_mode == "S2":
            Hbar = torch.clamp(self.H.mean(dim=1), min=1e-8)
            Ybar = torch.clamp(Ybar / Hbar, min=1e-8)
        z_hat = _is_z_update(Ybar, self.W, n_iter=self.nmf_iter, l1=self.nmf_l1)
        s_hat = self.W @ z_hat

        eps = 1e-8
        Y_hat_safe = torch.clamp(Y_hat, min=eps)
        base = (Y / (Y_hat_safe ** 2)) - (1.0 / Y_hat_safe)
        base_sum = torch.sum(base, dim=1).detach().cpu()

        H_cpu = self.H
        s_cpu = s_hat.detach().cpu()
        A = torch.zeros(H_cpu.shape[1], dtype=torch.float32)
        for d in range(H_cpu.shape[1]):
            Hd = H_cpu[:, d]
            contrib = (Hd * s_cpu) * base_sum
            A[d] = torch.sum(contrib)

        ratio = torch.clamp(Y / Y_hat_safe, min=eps)
        r_is = -torch.sum(ratio - torch.log(ratio) - 1.0).detach().cpu()

        phi_phys = None
        if self.add_phys_feature:
            phi_phys = (H_cpu * s_cpu.view(-1, 1)).sum(dim=0)

        phi = group_norms.detach().cpu()
        if phi_phys is not None:
            g = phi
            p = (phi_phys - phi_phys.mean()) / (phi_phys.std() + 1e-8)
            gp = (g - g.mean()) / (g.std() + 1e-8)
            phi = gp + p

        return {"phi": phi, "A": A, "r_is": r_is, "info": info, "s_hat": s_hat}

