import numpy as np
import torch
from typing import List, Optional, Tuple

from .nmf_utils import estimate_s_hat


class PatchTokenizer:
    def __init__(self, Fp=16, Np=10, n_levels=16):
        self.Fp, self.Np, self.n_levels = Fp, Np, n_levels

    def __call__(self, Y: np.ndarray) -> List[str]:
        F, N = Y.shape
        logY = np.log(np.maximum(Y, 1e-12))
        Lf, Lt = F // self.Fp, N // self.Np
        toks: List[str] = []
        for i in range(Lf):
            for j in range(Lt):
                v = float(logY[i * self.Fp:(i + 1) * self.Fp, j * self.Np:(j + 1) * self.Np].mean())
                lev = int(np.clip((v + 15) / 30 * (self.n_levels - 1), 0, self.n_levels - 1))
                toks.append(f"<P_{i}_{j}_{lev}>")
        return toks


class LeafTokenizer:
    def __init__(self, sr=16000, n_mels=64, n_levels=16):
        self.sr, self.n_mels, self.n_levels = sr, n_mels, n_levels
        try:
            import leaf_audio  # noqa: F401
            self.use_leaf = True
        except Exception:
            self.use_leaf = False
            import torchaudio
            self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=n_mels)

    def __call__(self, wav: np.ndarray) -> List[str]:
        x = torch.from_numpy(wav).float().unsqueeze(0)
        if self.use_leaf:
            from leaf_audio import Leaf, LogCompression, GaussianLowPass
            leaf = Leaf(self.sr, n_filters=self.n_mels)
            with torch.no_grad():
                feat = LogCompression()(GaussianLowPass(self.sr)(leaf(x)))  # (1,B,T)
        else:
            with torch.no_grad():
                S = self.mel(x)
                feat = torch.log(torch.clamp(S, min=1e-12))
        feat = feat.squeeze(0)
        B, T = feat.shape[0], feat.shape[1]
        toks: List[str] = []
        for t in range(T):
            col = feat[:, t]
            levf = ((col + 15) / 30 * (self.n_levels - 1)).clamp(0, self.n_levels - 1)
            lev = levf.to(torch.int64).cpu().numpy()
            col_np = col.cpu().numpy()
            top = np.argsort(-col_np)[:4]
            toks.append("<LEAF_" + "_".join([f"{i}:{int(lev[i])}" for i in top]) + ">")
        return toks


class ScatterTokenizer:
    def __init__(self, sr=16000, J=6, Q=8, T=2 ** 14, n_levels=16):
        try:
            from kymatio.torch import Scattering1D  # noqa: F401
            self.available = True
        except Exception:
            self.available = False
        self.sr = sr
        self.J = J
        self.Q = Q
        self.T = T
        self.n_levels = n_levels

    def __call__(self, wav: np.ndarray) -> List[str]:
        if not self.available:
            # Fallback: trivial token
            return ["<SC_UNAVAILABLE>"]
        from kymatio.torch import Scattering1D
        x = torch.zeros(self.T)
        w = torch.from_numpy(wav[: self.T]).float()
        x[: len(w)] = w
        with torch.no_grad():
            sc = Scattering1D(J=self.J, shape=self.T, Q=self.Q)
            Sx = sc(x.unsqueeze(0)).abs().mean(-1).squeeze(0).numpy()
        Sx = np.log(np.maximum(Sx, 1e-12))
        lev = np.clip(((Sx + 15) / 30 * (self.n_levels - 1)).astype(int), 0, self.n_levels - 1)
        return [f"<SC_{i}_{lev[i]}>" for i in range(len(lev))]


class NMFTokenizer:
    """Use offline W(F,K), estimate z_hat on Y(F,N) via IS-MU → s_hat = W z_hat.

    Returns atom tokens (top-k) and s_hat for use in (★).
    """

    def __init__(self, W: Optional[np.ndarray] = None, topk=6, n_iter=50, mode="S1", l1=0.0):
        self.W = W
        self.topk = topk
        self.n_iter = n_iter
        self.mode = mode
        self.l1 = l1

    def set_W(self, W: np.ndarray):
        self.W = W

    def __call__(self, Y: np.ndarray, H: Optional[np.ndarray] = None) -> Tuple[List[str], np.ndarray]:
        assert self.W is not None, "Please provide W(F,K) via set_W or constructor"
        s_hat, z = estimate_s_hat(Y, self.W, mode=self.mode, H=H, n_iter=self.n_iter, l1=self.l1)
        idx = np.argsort(-z)[: self.topk]
        toks = [f"<AT_{i}:{int(10 * np.log1p(z[i]))}>" for i in idx]
        return toks, s_hat


def direction_projection_tokens(YF: np.ndarray, H: np.ndarray, alpha=1.0, topM: Optional[int] = None) -> List[str]:
    # Ensure shapes (F,), (F,D)
    if H.shape[0] < H.shape[1]:
        H = H.T
    r = (np.maximum(H, 1e-12) ** alpha).T @ np.maximum(YF, 1e-12)  # (D,)
    r = np.log(np.maximum(r, 1e-12) / (np.mean(r) + 1e-12))
    idx = np.argsort(-r)[:topM] if topM else np.arange(len(r))
    return [f"<R_{int(d)}:{int(10 * r[int(d)])}>" for d in idx]

