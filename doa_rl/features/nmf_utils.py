"""
Lightweight NMF helpers for content spectrum estimation (ŝ = W z).

This module is a self-contained copy of the minimal utilities we need to:
- Estimate z via IS multiplicative updates for a fixed dictionary W
- Compute ŝ = W @ z from a spectrogram segment Y
- Optionally train a small offline NMF dictionary (for completeness/tests)

Notes
- All functions are written in NumPy for portability; a small Torch twin is
  provided for numerical consistency tests with AdvantageComputer.
- Shapes follow: Y ∈ R_+^{F×N}, W ∈ R_+^{F×K}, z ∈ R_+^{K}, ŝ ∈ R_+^{F}
"""

from typing import Tuple, Optional, Union, List
import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor"]


def _safe_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.maximum(x, eps)


def normalize_W(W: np.ndarray) -> np.ndarray:
    """Column-normalize a non-negative dictionary W (F,K) by L1.

    Returns W where each column sums to 1 (up to eps).
    """
    Wp = _safe_np(W)
    scale = np.sum(Wp, axis=0, keepdims=True)  # (1,K)
    return Wp / _safe_np(scale)


def estimate_z_is(Ybar: np.ndarray, W: np.ndarray, n_iter: int = 50, l1: float = 0.0) -> np.ndarray:
    """IS-MU estimate of z given fixed W and Ybar (vector, F,).

    Implements Fevotte & Idier IS (β=0) multiplicative updates:
        z ← z ⊙ (W^T [Ybar / (Wz)^2]) / (W^T [1/(Wz)] + λ₁)
        then normalize z to sum to 1 to remove global scale.
    """
    F, K = W.shape
    z = np.ones(K, dtype=W.dtype) / max(K, 1)
    Wp = _safe_np(W)
    Yb = _safe_np(Ybar)
    for _ in range(max(int(n_iter), 0)):
        Yhat = _safe_np(Wp @ z)
        num = (Wp.T @ (Yb / (Yhat ** 2)))
        den = (Wp.T @ (1.0 / Yhat)) + float(l1)
        z *= _safe_np(num / _safe_np(den))
        z /= _safe_np(z.sum())
    return z


def estimate_s_hat(
    Y: np.ndarray,
    W: np.ndarray,
    mode: str = "S1",
    H: Optional[np.ndarray] = None,
    n_iter: int = 50,
    l1: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate ŝ(F,) from Y(F,N) and W(F,K) using IS-MU for z.

    Modes
    - S1: Ybar = mean_t(Y)
    - S2: Ybar = mean_t(Y) / mean_d(H_d)  (requires H with shape (F,D) or (D,F))
    Returns (s_hat, z_hat).
    """
    Ybar = _safe_np(Y.mean(axis=1))
    m = (mode or "S1").upper()
    if m == "S2":
        assert H is not None, "S2 requires H (transfer matrix)"
        Hm = H
        if Hm.shape[0] < Hm.shape[1]:  # (D,F) -> (F,D)
            Hm = Hm.T
        Hbar = _safe_np(Hm.mean(axis=1))
        Ybar = _safe_np(Ybar / Hbar)
    z = estimate_z_is(Ybar, W, n_iter=n_iter, l1=l1)
    s_hat = _safe_np(W @ z)
    return s_hat, z


def compute_s_hat_with_W(
    Y: ArrayLike,
    W: ArrayLike,
    mode: str = "S1",
    H: Optional[ArrayLike] = None,
    n_iter: int = 50,
    l1: float = 0.0,
) -> np.ndarray:
    """Convenience wrapper to compute ŝ = W ẑ from a precomputed W.

    Accepts NumPy arrays or Torch tensors; always returns NumPy array ŝ(F,).
    This is intended for inference/evaluation where W is trained offline.
    """
    try:
        import torch  # noqa: F401
        is_torch = True
    except Exception:
        is_torch = False

    def to_np(a: ArrayLike) -> Optional[np.ndarray]:
        if a is None:
            return None
        if is_torch and "torch" in str(type(a)):
            return a.detach().cpu().numpy()
        return np.asarray(a)

    Y_np = to_np(Y)
    W_np = to_np(W)
    H_np = to_np(H) if H is not None else None
    s_hat, _ = estimate_s_hat(Y_np, W_np, mode=mode, H=H_np, n_iter=n_iter, l1=l1)
    return s_hat


def train_nmf_is(Y_list: List[np.ndarray], K: int = 64, n_iter: int = 200, seed: int = 0) -> np.ndarray:
    """Simple offline IS-NMF dictionary learning over mean spectra Ybar.

    This is suitable for small tests. For production USM training, see
    nmf_localizer.core.usm_trainer.USMTrainer which uses a robust sklearn NMF implementation.
    """
    rng = np.random.RandomState(int(seed))
    Ybars = [np.maximum(Y.mean(axis=1), 1e-12) for Y in Y_list]
    F = int(Ybars[0].shape[0])
    Ycat = np.stack(Ybars, axis=1)  # (F, M)
    # Initialize W,Z
    W = np.maximum(rng.rand(F, int(K)).astype(np.float32), 1e-6)
    W = normalize_W(W)
    Z = np.maximum(rng.rand(int(K), Ycat.shape[1]).astype(np.float32), 1e-6)
    for _ in range(max(int(n_iter), 0)):
        Yhat = _safe_np(W @ Z)
        # Update Z (IS MU)
        Z *= (W.T @ (Ycat / (Yhat ** 2))) / _safe_np(W.T @ (1.0 / Yhat))
        Z /= _safe_np(Z.sum(axis=0, keepdims=True))
        # Update W (IS MU)
        Yhat = _safe_np(W @ Z)
        W *= ((Ycat / (Yhat ** 2)) @ Z.T) / _safe_np((1.0 / Yhat) @ Z.T)
        W = normalize_W(W)
    return _safe_np(W)


# Optional Torch twin for numerical checks against AdvantageComputer
def estimate_s_hat_torch(
    Y: "torch.Tensor",
    W: "torch.Tensor",
    mode: str = "S1",
    H: Optional["torch.Tensor"] = None,
    n_iter: int = 50,
    l1: float = 0.0,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Torch twin of estimate_s_hat using the same IS-MU logic for z.

    Returns (s_hat, z_hat) as torch tensors on CPU.
    """
    import torch
    eps = 1e-8
    Ybar = torch.clamp(Y.mean(dim=1), min=eps)
    if (mode or "S1").upper() == "S2":
        assert H is not None, "S2 requires H (transfer matrix)"
        Hm = H
        if Hm.shape[0] < Hm.shape[1]:
            Hm = Hm.t()
        Hbar = torch.clamp(Hm.mean(dim=1), min=eps)
        Ybar = torch.clamp(Ybar / Hbar, min=eps)
    F, K = W.shape
    z = torch.full((K,), 1.0 / max(int(K), 1), dtype=W.dtype)
    Wp = torch.clamp(W, min=eps)
    for _ in range(max(int(n_iter), 0)):
        Yhat = torch.clamp(Wp @ z, min=eps)
        num = (Wp.t() @ (Ybar / (Yhat ** 2)))
        den = (Wp.t() @ (1.0 / Yhat)) + float(l1)
        z = z * torch.clamp(num / torch.clamp(den, min=eps), min=eps)
        z = z / torch.clamp(z.sum(), min=eps)
    s_hat = torch.clamp(Wp @ z, min=eps)
    return s_hat.detach().cpu(), z.detach().cpu()


__all__ = [
    "normalize_W",
    "estimate_z_is",
    "estimate_s_hat",
    "compute_s_hat_with_W",
    "train_nmf_is",
    "estimate_s_hat_torch",
]
