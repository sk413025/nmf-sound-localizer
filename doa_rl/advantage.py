from typing import Dict, Any, Optional
import logging
import torch

from nmf_localizer.core.localizer import NMFSoundLocalizer


logger = logging.getLogger(__name__)


def _tensor_stats(x: torch.Tensor, name: str) -> str:
    x = x.detach().cpu().float().view(-1)
    if x.numel() == 0:
        return f"{name}: empty"
    return (
        f"{name}: shape={tuple(x.shape)} min={float(x.min()):.4g} "
        f"p5={float(torch.quantile(x, 0.05)): .4g} median={float(x.median()): .4g} "
        f"p95={float(torch.quantile(x, 0.95)): .4g} max={float(x.max()):.4g} "
        f"mean={float(x.mean()): .4g} std={float(x.std()): .4g}"
    )


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
                 add_phys_feature: bool = True, require_s_hat: bool = False):
        self.localizer = localizer
        self.W = W.detach().cpu()
        self.H = H.detach().cpu()  # (F,D)
        self.s_mode = s_mode.upper()
        self.nmf_iter = nmf_iter
        self.nmf_l1 = nmf_l1
        self.add_phys_feature = add_phys_feature
        self.require_s_hat = require_s_hat

    def __call__(self, Y: torch.Tensor, pi: torch.Tensor,
                 s_hat: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        device = self.localizer.device
        Y = Y.to(device)
        info: Dict[str, Any] = {}
        D = self.localizer.n_directions
        K = self.localizer.n_components
        Y_cpu = Y.detach().cpu()
        if pi is None:
            raise ValueError("AdvantageComputer requires policy distribution 'pi'")

        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("Advantage: input Y %s", _tensor_stats(Y, "Y(FxN)"))
            except Exception:
                pass

        z_hat: Optional[torch.Tensor] = None
        if s_hat is not None:
            s_hat_cpu = s_hat.detach().cpu().float().view(-1)
        else:
            if self.require_s_hat:
                raise ValueError("AdvantageComputer: s_hat is required but was not provided")
            Ybar = torch.mean(Y, dim=1).detach().cpu()
            if self.s_mode == "S2":
                Hbar = torch.clamp(self.H.mean(dim=1), min=1e-8)
                Ybar = torch.clamp(Ybar / Hbar, min=1e-8)
            z_hat = _is_z_update(Ybar, self.W, n_iter=self.nmf_iter, l1=self.nmf_l1)
            s_hat_cpu = (self.W @ z_hat).detach().cpu()

        if logger.isEnabledFor(logging.DEBUG):
            try:
                if s_hat is None:
                    logger.debug("Advantage: Ybar %s", _tensor_stats(Ybar, "Ybar(F)"))
                logger.debug("Advantage: W %s", _tensor_stats(self.W, "W(FxK)"))
                logger.debug("Advantage: H %s", _tensor_stats(self.H, "H(FxD)"))
                if z_hat is not None:
                    logger.debug("Advantage: z_hat %s", _tensor_stats(z_hat, "z_hat(K)"))
                logger.debug("Advantage: s_hat %s", _tensor_stats(s_hat_cpu, "s_hat(F)"))
            except Exception:
                pass

        eps = 1e-8
        # MD-consistent: policy-mixture reconstruction with cached ŝ
        pi_cpu = pi.detach().cpu().float().view(-1)
        H_cpu = self.H  # (F,D)
        s_cpu = s_hat_cpu.view(-1)  # (F,)
        Hs = H_cpu * s_cpu.view(-1, 1)  # (F,D)
        Y_mix = torch.matmul(Hs, pi_cpu)  # (F,)
        Y_mix = torch.clamp(Y_mix, min=eps)
        Fdim, N = Y.shape
        Y_hat_safe = Y_mix.view(-1, 1).expand(Fdim, N)
        base = (Y / (Y_hat_safe ** 2)) - (1.0 / Y_hat_safe)
        base_sum = torch.sum(base, dim=1).detach().cpu()
        # No group norms from factorization in this path
        group_norms = torch.zeros(D, device=device)
        Y_hat_cpu = Y_hat_safe.detach().cpu()
        logger.info(
            "Advantage: policy-mixture reconstruction shapes — pi=%s, Y_mix=%s, Y_hat=%s, base_sum=%s",
            tuple(pi_cpu.shape), tuple(Y_mix.shape), tuple(Y_hat_safe.shape), tuple(base_sum.shape)
        )
        try:
            logger.info(
                "Advantage: policy-mixture base_sum stats min=%.4g max=%.4g mean=%.4g",
                float(base_sum.min()), float(base_sum.max()), float(base_sum.mean())
            )
        except Exception:
            pass
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("Advantage: pi %s", _tensor_stats(pi_cpu, "pi(D)"))
                logger.debug("Advantage: Y_mix %s", _tensor_stats(Y_mix, "Y_mix(F)"))
                logger.debug("Advantage: base %s", _tensor_stats(base, "base(FxN)"))
                logger.debug(
                    "Advantage: base_sum(F) %s neg_frac=%.3f",
                    _tensor_stats(base_sum, "base_sum(F)"),
                    float((base_sum < 0).float().mean()),
                )
            except Exception:
                pass

        H_cpu = self.H
        s_cpu = s_hat_cpu
        A = torch.zeros(H_cpu.shape[1], dtype=torch.float32)
        for d in range(H_cpu.shape[1]):
            Hd = H_cpu[:, d]
            contrib = (Hd * s_cpu) * base_sum
            A[d] = torch.sum(contrib)

        # Importance-sampling style reward (always <= 0)
        ratio = torch.clamp(Y / Y_hat_safe, min=eps)
        r_is = -torch.sum(ratio - torch.log(ratio) - 1.0).detach().cpu()
        try:
            ratio_cpu = ratio.detach().cpu()
            logger.info(
                "Advantage: IS divergence summary — r_is=%.4g per_bin=%.4g ratio[min=%.4g max=%.4g mean=%.4g]",
                float(r_is), float(r_is) / max(float(Y.numel()), 1.0),
                float(ratio_cpu.min()), float(ratio_cpu.max()), float(ratio_cpu.mean())
            )
        except Exception:
            pass

        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug("Advantage: A %s neg_frac=%.3f", _tensor_stats(A, "A(D)"), float((A < 0).float().mean()))
                Fn = float(Y.numel())
                r_unit = float(r_is) / max(Fn, 1.0)
                logger.debug(
                    "Advantage: r_is=%.4g (per-bin=%.4g) ratio stats: %s",
                    float(r_is), r_unit, _tensor_stats(ratio, "ratio (Y/Y_hat)")
                )
            except Exception:
                pass

        phi_phys = None
        if self.add_phys_feature:
            phi_phys = (H_cpu * s_cpu.view(-1, 1)).sum(dim=0)

        phi = group_norms.detach().cpu()
        if phi_phys is not None:
            g = phi
            p = (phi_phys - phi_phys.mean()) / (phi_phys.std() + 1e-8)
            gp = (g - g.mean()) / (g.std() + 1e-8)
            phi = gp + p

        return {
            "phi": phi,
            "A": A,
            "r_is": r_is,
            "info": info,
            "s_hat": s_cpu.clone(),
            "Y_hat": Y_hat_cpu.clone(),
            "Y_orig": Y_cpu.clone(),
        }
