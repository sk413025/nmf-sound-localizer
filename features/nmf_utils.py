import numpy as np


def _safe(x, eps=1e-12):
    return np.maximum(x, eps)


def normalize_W(W: np.ndarray) -> np.ndarray:
    W = _safe(W)
    scale = np.sum(W, axis=0, keepdims=True)  # (1,K)
    return W / _safe(scale)


def estimate_z_is(Ybar: np.ndarray, W: np.ndarray, n_iter: int = 50, l1: float = 0.0) -> np.ndarray:
    """IS-MU estimate of z given fixed W and Ybar (vector, F,).

    Geometry: beta=0 IS-Bregman; multiplicative updates using positive/negative
    gradient ratio (Fevotte & Idier; docs/code.md Eq. (15)-(16)).
    """
    F, K = W.shape
    z = np.ones(K, dtype=W.dtype) / K
    Wp = _safe(W)
    Yb = _safe(Ybar)
    for _ in range(n_iter):
        Yhat = _safe(Wp @ z)
        num = (Wp.T @ (Yb / (Yhat ** 2)))
        den = (Wp.T @ (1.0 / Yhat)) + l1
        z *= _safe(num / _safe(den))
        z /= _safe(z.sum())
    return z


def estimate_s_hat(Y: np.ndarray, W: np.ndarray, mode: str = "S1", H=None,
                   n_iter: int = 50, l1: float = 0.0):
    """Estimate s_hat(F,) via IS-MU on Y(F,N).

    mode S1: Ybar = mean_t |Y|
         S2: Ybar = mean_t |Y| / mean_d H_d
    Returns (s_hat, z_hat).
    """
    Ybar = _safe(Y.mean(axis=1))
    if mode.upper() == "S2":
        assert H is not None, "S2 requires H"
        if H.shape[0] < H.shape[1]:
            H = H.T
        Hbar = _safe(H.mean(axis=1))
        Ybar = _safe(Ybar / Hbar)
    z = estimate_z_is(Ybar, W, n_iter=n_iter, l1=l1)
    s_hat = _safe(W @ z)
    return s_hat, z


def train_nmf_is(Y_list: list, K: int = 256, n_iter: int = 200, seed: int = 0) -> np.ndarray:
    """Simple IS-NMF dictionary learning (offline) over concatenated Ybars.

    This util is provided for completeness with docs/code.md; in this repo we
    typically load pre-trained W from assets.
    """
    rng = np.random.RandomState(seed)
    # Build dataset of Ybar vectors
    Ybars = [np.maximum(Y.mean(axis=1), 1e-12) for Y in Y_list]
    F = Ybars[0].shape[0]
    Ycat = np.stack(Ybars, axis=1)  # (F, M)
    # Init W,z non-negatively
    W = np.maximum(rng.rand(F, K).astype(np.float32), 1e-6)
    W = normalize_W(W)
    Z = np.maximum(rng.rand(K, Ycat.shape[1]).astype(np.float32), 1e-6)
    for _ in range(n_iter):
        Yhat = _safe(W @ Z)
        # Update Z
        Z *= (W.T @ (Ycat / (Yhat ** 2))) / _safe(W.T @ (1.0 / Yhat))
        # Normalize per-sample
        Z /= _safe(Z.sum(axis=0, keepdims=True))
        # Update W
        Yhat = _safe(W @ Z)
        W *= ((Ycat / (Yhat ** 2)) @ Z.T) / _safe((1.0 / Yhat) @ Z.T)
        W = normalize_W(W)
    return _safe(W)

