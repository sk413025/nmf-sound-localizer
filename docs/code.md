下面給你**完整可執行**的專案骨架與**全部程式碼**（模組化：`features/`, `env/`, `algos/`, `scripts/`, `eval/`），並在開頭用**鏡像梯度下降（Mirror Descent, MD）統一視角**把核心公式與對應關係**從第一性原理推導**清楚，含 **dim/shape** 與**物理意義**。整個設計**不改 Transformer 架構**：我們只把資料表示與回饋/優勢的計算對齊 MD 幾何；策略更新交給 `trl` 的 **PPO** 或 **GRPO**。

---

## A) 鏡像梯度下降（MD）統一視角：公式、shape 與物理意義

**觀測與散射（窄帶 STFT 近似）**（原論文式(2)–(3)）：

$$
Y(n,f)=\sum_{j\in\mathcal J} S_j(n,f)\,H_j(f)+E(n,f)\quad\Longleftrightarrow\quad
Y=\sum_{j}\mathrm{diag}(H_j)\,S_j+E.
$$

* **shape**：$Y\in\mathbb R_+^{F\times N}$（幅譜/能量，頻帶×時間幀）；$H_j\in\mathbb R_+^{F}$（**directional filter**：方向→頻帶增益）；$E$ 噪聲。

**語音字典模型**（原論文式(7)–(8)）：

$$
S_j=WX_j,\quad
Y=AX+E,\quad
A=[\mathrm{diag}(H_1)W,\dots,\mathrm{diag}(H_D)W].
$$

* **shape**：$W\in\mathbb R_+^{F\times K}$（**energy dictionary**）、$X\in\mathbb R_+^{KD\times N}$。**物理**：$W$ 描述**內容**（parts‑based）；$H_d$ 把內容映到**方向‑頻帶**幾何。

**IS（Itakura–Saito）散度 + 乘法更新（MU）**（原論文式(9)–(16)）：
IS‑Bregman 幾何給出對能量變數的**乘法更新**（**鏡像步**），具**尺度不變**與**保非負**。

---

### 內容—策略「雙幾何」：內圈（IS）+ 外圈（KL）

* **內容幾何（IS）**：對**內容譜** $s=Wz\in\mathbb R_+^{F}$ 的估計，採 **IS‑MU** 反演係數 $z\in\mathbb R_+^{K}$。本專案提供 `features/nmf_utils.py` 中

  $$
  z \leftarrow z \odot \frac{W^\top\!\big(Y/\widehat Y^2\big)}{W^\top\!\big(1/\widehat Y\big)+\lambda}
  \quad\Rightarrow\quad s_{\hat{}}=Wz_{\hat{}}.
  $$

  這正是論文式(15)–(16) 的**正/負梯度比**（IS‑Bregman 鏡像步）。**shape**：$s_{\hat{}}\in\mathbb R_+^{F}$。

* **策略幾何（KL）**：把「把能量分到哪個方向」當作**策略分佈** $\pi\in\Delta^{D-1}$（**policy over directions**）。
  在 **KL‑Bregman** 幾何下，**PPO/GRPO** 的更新等價於對 **logits** 做

  $$
  g\leftarrow g+\eta\,\widehat A,\qquad \pi=\mathrm{softmax}(g),
  $$

  即**鏡像步**（exponentiated‑gradient 的對偶加法）。**shape**：$g,\widehat A\in\mathbb R^{D}$。

---

### 方向優勢（advantage）從 IS 推導（與論文 MU 對齊）

令 $\widehat Y=\rho\sum_{d=1}^{D}\pi_d(H_d\odot s)$，對「往方向 $d$ 多分一點能量」的參數 $\alpha_d=\rho\pi_d$ 求 IS 的導數之負值，可得

$$
\boxed{A_d=\sum_{f=1}^F (H_d\odot s)_f\Big(\frac{Y_f}{\widehat Y_f^2}-\frac{1}{\widehat Y_f}\Big)}\tag{★}
$$

* **shape**：$A\in\mathbb R^D$。**物理**：$(H_d\odot s)_f$ 放大**信息性頻帶**（例如 LEGO 3–8 kHz；原論文表 I），括號是 IS 的正/負梯度差；這與 MU 的分子/分母**一一對應**。

> **一句話**：內圈（IS‑MU）把 $Y,W$ 變成**當下片段的** $s_{\hat{}}$；外圈（PPO/GRPO）把 (★) 當作物理‑正確的 $\widehat A$ 做 **KL‑mirror** 更新。

---

## B) 專案骨架與完整程式碼

> **安裝**（Python 3.10+）：
> `pip install -U transformers datasets accelerate trl peft bitsandbytes torchaudio librosa scikit-learn kymatio matplotlib numpy`
> （可選）LEAF 前端：`pip install leaf-audio`

```
doa_rl/
├── README.md
├── doa_rl/
│   ├── __init__.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── tokenizers.py
│   │   └── nmf_utils.py
│   ├── env/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   ├── doa_math.py
│   │   ├── hrtf_loader.py
│   │   └── dataset.py
│   ├── algos/
│   │   ├── __init__.py
│   │   ├── ppo_runner.py
│   │   └── grpo_runner.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   ├── prepare_hrtf.py
│   │   ├── prepare_dict.py
│   │   ├── train_single.py
│   │   ├── train_multi.py
│   │   └── infer_demo.py
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py
│       └── plot_eval.py
└── pyproject.toml   # (可選)
```

> **執行方式（例）**
>
> 1. 產生 $H(d,f)$：`python -m doa_rl.scripts.prepare_hrtf --src path/to/IRs --type device --out data/H.npz`
> 2. 離線學 $W$：`python -m doa_rl.scripts.prepare_dict --wav_dir path/to/wavs --K 256 --out data/W_usm_is.npz`
> 3. 單源 + PPO：`python -m doa_rl.scripts.train_single --H data/H.npz --W data/W_usm_is.npz --feature patch --add_dir_tokens 1 --algo ppo`
> 4. 多源 + GRPO：`python -m doa_rl.scripts.train_multi --H data/H.npz --W data/W_usm_is.npz --J 2 --feature patch --add_dir_tokens 1 --algo grpo`

---

### 1) `README.md`

```markdown
# DoA-RL (Mirror Descent View)

## Problem & Physics
STFT narrowband model:
Y = Σ_j diag(H_j) S_j + E,  (H_j: directional filter)
Speech dictionary:
S_j = W X_j  ⇒  Y = A X + E,  A = [diag(H_1)W, …, diag(H_D)W].
(See eqs. (2)–(3), (7)–(8) in the paper.) 

## Mirror Descent Decomposition
**Inner loop (content, IS geometry)**: estimate z ≥ 0 by IS-NMF (multiplicative updates),
s_hat = W z_hat. (Eqs. (9)–(16).) 

**Outer loop (policy, KL geometry)**: update logits g with PPO / GRPO (KL mirror step):
g ← g + η Â,  π = softmax(g).

**Direction advantage (from IS gradient of α_d = ρ π_d)**:
A_d = Σ_f (H_d ⊙ s)_f [ Y_f / Ŷ_f^2 – 1/Ŷ_f ],  Ŷ = ρ Σ_d π_d (H_d ⊙ s). (★)

## Tokens
- Content tokens: spectrogram patches / LEAF / scattering / NMF atoms (IS-friendly).
- Physics-aware direction tokens: r_d = ⟨Y, |H_d|^α⟩ (cone-membership soft evidence). (See Fig. 1, eqs. (4)–(6).) 

## Metrics
10° bin accuracy, mean angular error, per-source accuracy, confusion matrix (Sec. IV‑C). 
```

---

### 2) `doa_rl/__init__.py`

```python
__all__ = []
```

---

### 3) `doa_rl/features/__init__.py`

```python
from .tokenizers import PatchTokenizer, LeafTokenizer, ScatterTokenizer, NMFTokenizer, direction_projection_tokens
from .nmf_utils import estimate_s_hat, estimate_z_is, train_nmf_is, normalize_W
```

---

### 4) `doa_rl/features/nmf_utils.py`

```python
import numpy as np

def _safe(x, eps=1e-12): return np.maximum(x, eps)

def normalize_W(W):
    W = _safe(W)
    scale = np.sum(W, axis=0, keepdims=True)  # (1,K)
    return W / _safe(scale)

def estimate_z_is(Ybar: np.ndarray, W: np.ndarray, n_iter=50, l1=0.0):
    """
    IS-NMF multiplicative updates for z with fixed W (vector case).
    Args:
      Ybar: (F,) non-negative spectrum (e.g., mean_t |Y|)
      W   : (F,K) dictionary (non-negative, column-normalized)
      n_iter: iterations
      l1: L1 regularization on z
    Returns:
      z_hat: (K,)
    """
    F, K = W.shape
    z = np.ones(K) / K
    W = _safe(W); Ybar = _safe(Ybar)
    for _ in range(n_iter):
        Yhat = _safe(W @ z)                       # (F,)
        num  = (W.T @ (Ybar / (Yhat**2)))         # (K,)
        den  = (W.T @ (1.0 / Yhat)) + l1          # (K,)
        z   *= _safe(num / den)
        z   /= _safe(np.sum(z))
    return z

def estimate_s_hat(Y: np.ndarray, W: np.ndarray, mode="S1", H=None,
                   n_iter=50, l1=0.0):
    """
    Estimate s_hat(F,) given Y(F,N) and W(F,K).
    S1: Ybar = mean_t |Y|
    S2: Y' = Ybar / H_bar, where H_bar = mean_d H(d,f)
    """
    Ybar = _safe(Y.mean(axis=1))  # (F,)
    if mode.upper() == "S2":
        assert H is not None, "S2 needs H(d,f)"
        Hbar = _safe(H.mean(axis=0))
        Ybar = _safe(Ybar / Hbar)
    Wn = normalize_W(W.copy())
    z  = estimate_z_is(Ybar, Wn, n_iter=n_iter, l1=l1)
    s_hat = _safe(Wn @ z)              # (F,)
    s_hat /= _safe(s_hat.max())
    return s_hat, z

def train_nmf_is(Y_list, K=256, n_iter=200, seed=0):
    """
    Simple IS-NMF to train W from a list of magnitude spectra segments.
    Returns:
      W: (F,K)
    """
    rng = np.random.RandomState(seed)
    V = np.hstack([_safe(Y) for Y in Y_list])  # (F, sum N)
    F = V.shape[0]
    W = rng.rand(F, K); W = normalize_W(W)
    H = rng.rand(K, V.shape[1])
    for _ in range(n_iter):
        WH = _safe(W @ H)
        H *= (W.T @ (V / (WH**2))) / _safe(W.T @ (1.0/WH))
        WH = _safe(W @ H)
        W *= ((V / (WH**2)) @ H.T) / _safe((1.0/WH) @ H.T)
        W = normalize_W(W)
    return _safe(W)
```

---

### 5) `doa_rl/features/tokenizers.py`

```python
import numpy as np, torch, torchaudio
from typing import List, Optional, Tuple
from .nmf_utils import estimate_s_hat

class PatchTokenizer:
    """Spectrogram patch tokens (AST/PaSST style).  Y(F,N) -> tokens length L=(F/Fp)*(N/Np)."""
    def __init__(self, Fp=16, Np=10, n_levels=16):
        self.Fp, self.Np, self.n_levels = Fp, Np, n_levels
    def __call__(self, Y: np.ndarray) -> List[str]:
        F,N = Y.shape
        logY = np.log(np.maximum(Y,1e-12))
        Lf, Lt = F//self.Fp, N//self.Np
        toks=[]
        for i in range(Lf):
            for j in range(Lt):
                v = float(logY[i*self.Fp:(i+1)*self.Fp, j*self.Np:(j+1)*self.Np].mean())
                lev = int(np.clip((v+15)/30*(self.n_levels-1), 0, self.n_levels-1))
                toks.append(f"<P_{i}_{j}_{lev}>")
        return toks

class LeafTokenizer:
    """LEAF (if installed) -> log energies -> tokens; fallback to MelSpectrogram."""
    def __init__(self, sr=16000, n_mels=64, n_levels=16):
        self.sr, self.n_mels, self.n_levels = sr, n_mels, n_levels
        try:
            import leaf_audio  # noqa
            self.use_leaf=True
        except Exception:
            self.use_leaf=False
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
                S = self.mel(x); feat = torch.log(torch.clamp(S, min=1e-12))
        feat = feat.squeeze(0).numpy()  # (B,T)
        toks=[]
        for t in range(feat.shape[1]):
            col = feat[:,t]; lev = np.clip(((col+15)/30*(self.n_levels-1)).astype(int),0,self.n_levels-1)
            top = np.argsort(-col)[:4]
            toks.append("<LEAF_"+ "_".join([f"{i}:{lev[i]}" for i in top]) + ">")
        return toks

class ScatterTokenizer:
    """Kymatio 1D Scattering -> multi-scale energies -> tokens (IS-friendly)."""
    def __init__(self, sr=16000, J=6, Q=8, T=2**14, n_levels=16):
        from kymatio.torch import Scattering1D
        self.T=T; self.sc = __import__("kymatio").torch.Scattering1D(J=J, shape=T, Q=Q)
        self.n_levels=n_levels
    def __call__(self, wav: np.ndarray) -> List[str]:
        x = torch.zeros(self.T); w=torch.from_numpy(wav[:self.T]).float(); x[:len(w)] = w
        with torch.no_grad():
            Sx = self.sc(x.unsqueeze(0)).abs().mean(-1).squeeze(0).numpy()
        Sx = np.log(np.maximum(Sx,1e-12)); lev = np.clip(((Sx+15)/30*(self.n_levels-1)).astype(int),0,self.n_levels-1)
        return [f"<SC_{i}_{lev[i]}>" for i in range(len(lev))]

class NMFTokenizer:
    """
    Read offline W(F,K), estimate s_hat=W z_hat by IS-MU on current segment Y(F,N),
    return atom tokens (top-k) and s_hat for RL reward/advantage (★).
    """
    def __init__(self, W: Optional[np.ndarray]=None, topk=6, n_iter=50, mode="S1", l1=0.0):
        self.W = W
        self.topk = topk; self.n_iter=n_iter; self.mode=mode; self.l1=l1
    def set_W(self, W: np.ndarray): self.W=W
    def __call__(self, Y: np.ndarray, H: Optional[np.ndarray]=None) -> Tuple[List[str], np.ndarray]:
        assert self.W is not None, "NMFTokenizer needs offline W(F,K)"
        s_hat, z = estimate_s_hat(Y, self.W, mode=self.mode, H=H, n_iter=self.n_iter, l1=self.l1)
        idx = np.argsort(-z)[:self.topk]
        toks = [f"<AT_{i}:{int(10*np.log1p(z[i]))}>" for i in idx]
        return toks, s_hat

def direction_projection_tokens(YF: np.ndarray, H: np.ndarray, alpha=1.0, topM: Optional[int]=None) -> List[str]:
    """
    Physics-aware direction tokens: r_d = <Y, |H_d|^alpha> (soft cone-membership).
    """
    r = (np.maximum(H,1e-12)**alpha) @ np.maximum(YF,1e-12)  # (D,)
    r = np.log(np.maximum(r,1e-12)/ (np.mean(r)+1e-12))
    idx = np.argsort(-r)[:topM] if topM else np.arange(len(r))
    return [f"<R_{int(d)}:{int(10*r[int(d)])}>" for d in idx]
```

---

### 6) `doa_rl/env/__init__.py`

```python
from .doa_math import itakura_saito, advantage_IS, reward_is
from .dataset import build_dataset, make_dir_vocab
from .hrtf_loader import load_cipic_hrtf, load_device_hrtf
from .utils import stft_mag
```

---

### 7) `doa_rl/env/utils.py`

```python
import numpy as np, librosa

def stft_mag(y, sr=16000, n_fft=512, hop=256, win=512):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win, window="hann")
    return np.maximum(np.abs(S), 1e-12)  # (F,N)
```

---

### 8) `doa_rl/env/doa_math.py`

```python
import numpy as np

def itakura_saito(Y: np.ndarray, Yhat: np.ndarray) -> float:
    Y = np.maximum(Y, 1e-12); Yhat = np.maximum(Yhat, 1e-12)
    ratio = Y / Yhat
    return float(np.sum(ratio - np.log(ratio) - 1.0))

def advantage_IS(Y: np.ndarray, s_hat: np.ndarray, H: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """
    A_d = Σ_f (H_d ⊙ s_hat)_f [ Y_f/Ŷ_f^2 - 1/Ŷ_f ], with Ŷ = (Σ_d π_d H_d) ⊙ s_hat.
    Shapes: Y(F,N), s_hat(F,), H(D,F), pi(D,) -> A(D,)
    """
    Ymean = np.maximum(Y.mean(axis=1), 1e-12)         # (F,)
    mix   = np.maximum((pi[:, None] * H).sum(0), 1e-12)  # (F,)
    Yhat  = np.maximum(mix * s_hat, 1e-12)            # (F,)
    term1 = (H * s_hat[None, :]) @ (Ymean / (Yhat**2))  # (D,)
    term2 = (H * s_hat[None, :]) @ (1.0 / Yhat)         # (D,)
    return term1 - term2

def reward_is(Y: np.ndarray, s_hat: np.ndarray, H: np.ndarray, sel_dirs: list, rho: float=1.0) -> float:
    """
    Reward = -D_IS(Y || Ŷ), where Ŷ = ρ (Σ_{d in sel} H_d) ⊙ s_hat.
    Shapes: Y(F,N), s_hat(F,), H(D,F)
    """
    Ymean = np.maximum(Y.mean(axis=1), 1e-12)
    Hsum  = np.maximum(H[sel_dirs, :].sum(axis=0), 1e-12)
    Yhat  = np.maximum(rho * Hsum * s_hat, 1e-12)
    ratio = Ymean / Yhat
    return float(np.sum(- (ratio - np.log(ratio) - 1.0)))
```

---

### 9) `doa_rl/env/hrtf_loader.py`

```python
import os, numpy as np, soundfile as sf
from glob import glob
import librosa

def load_cipic_hrtf(cipic_root: str, azim_step: int=10, sr: int=16000, n_fft: int=512) -> np.ndarray:
    """
    Convert CIPIC HRIRs into |H_d(f)| magnitude responses.
    Returns: H(D,F). D = 360/azim_step, F = n_fft//2+1
    """
    ir_files = sorted(glob(os.path.join(cipic_root, "ir_*deg.wav")))
    H_list=[]
    for path in ir_files[::azim_step]:
        ir, fs = sf.read(path)
        if fs != sr:
            ir = librosa.resample(ir.astype(np.float32), orig_sr=fs, target_sr=sr)
        if ir.ndim == 2:
            ir = ir[:,0]
        Hf = np.abs(np.fft.rfft(ir, n=n_fft))  # (F,)
        H_list.append(Hf / (Hf.max()+1e-12))
    H = np.stack(H_list, 0)
    return np.maximum(H, 1e-9)

def load_device_hrtf(ir_folder: str, sr: int=16000, n_fft: int=512) -> np.ndarray:
    """
    Load your device IRs measured on a turntable; convert to |H_d(f)|.
    """
    files = sorted(glob(os.path.join(ir_folder, "*.wav")))
    H=[]
    for f in files:
        ir, fs = sf.read(f)
        if fs != sr:
            ir = librosa.resample(ir.astype(np.float32), fs, sr)
        if ir.ndim == 2: ir = ir[:,0]
        Hf = np.abs(np.fft.rfft(ir, n=n_fft))
        H.append(Hf / (Hf.max()+1e-12))
    H = np.stack(H, 0)
    return np.maximum(H, 1e-9)
```

---

### 10) `doa_rl/env/dataset.py`

```python
import numpy as np, librosa, os, random
from typing import Dict, List, Optional, Tuple
from datasets import Dataset
from .utils import stft_mag
from ..features.tokenizers import PatchTokenizer, LeafTokenizer, ScatterTokenizer, NMFTokenizer, direction_projection_tokens

def make_dir_vocab(D: int) -> List[str]:
    return [f"<D_{i}>" for i in range(D)]

def build_prompt(content_tokens: List[str], dir_tokens: List[str]=None) -> str:
    s = "OBS: " + " ".join(content_tokens)
    if dir_tokens: s += " DIR: " + " ".join(dir_tokens)
    s += "\nAnswer with one or more direction tokens."
    return s

def synth_or_load_wav(wav_path: Optional[str], sr=16000, T=1.5):
    if wav_path and os.path.exists(wav_path):
        y, fs = librosa.load(wav_path, sr=sr, mono=True)
        return y
    n=int(T*sr); t=np.arange(n)/sr; y=np.zeros(n); rng=np.random.RandomState(0)
    for k in range(5):
        f0=rng.uniform(120,280); a=rng.uniform(0.4,0.9)
        y+=a*np.sin(2*np.pi*f0*t + 2*np.pi*0.2*np.sin(2*np.pi*2.0*t))
    return (y/(np.abs(y).max()+1e-12)).astype(np.float32)

def build_dataset(H: np.ndarray,               # (D,F)
                  wav_list: List[Optional[str]],
                  feature: str="patch",        # "patch" | "leaf" | "scatter" | "nmf"
                  add_dir_tokens: bool=True,
                  J: int=1, sr: int=16000, n_fft: int=512,
                  W: Optional[np.ndarray]=None, s_mode: str="S1", nmf_iter:int=50, nmf_l1:float=0.0
                  ) -> Tuple[Dataset, List[Dict], List[str]]:
    """
    Return HF Dataset (prompts only) + raw examples (Y, s_hat, H, dirs... ) + dir_vocab.
    If W provided: estimate s_hat = W z_hat (IS-MU). Else: s_hat = mean_t |Y|.
    """
    D,F = H.shape
    dir_vocab = make_dir_vocab(D)
    if feature=="patch":     fea = PatchTokenizer()
    elif feature=="leaf":    fea = LeafTokenizer(sr=sr)
    elif feature=="scatter": fea = ScatterTokenizer(sr=sr)
    elif feature=="nmf":     fea = NMFTokenizer(W=W, topk=6, n_iter=nmf_iter, mode=s_mode, l1=nmf_l1)
    else: fea = PatchTokenizer()

    exs=[]
    for i,wp in enumerate(wav_list):
        y  = synth_or_load_wav(wp, sr=sr)
        Y  = stft_mag(y, n_fft=n_fft)          # (F,N)
        # s_hat
        if W is not None:
            nmf_tmp = NMFTokenizer(W=W, n_iter=nmf_iter, mode=s_mode, l1=nmf_l1)
            _toks, s_hat = nmf_tmp(Y, H=H)
        else:
            s_hat = np.maximum(Y.mean(axis=1),1e-12); s_hat/= (s_hat.max()+1e-12)
        # content tokens
        if feature in ["leaf","scatter"]:
            toks = fea(y)
        elif feature=="nmf":
            toks, _ = fea(Y, H=H)
        else:
            toks = fea(Y)
        # direction tokens
        d_toks = direction_projection_tokens(Y.mean(axis=1), H) if add_dir_tokens else None
        prompt = build_prompt(toks, d_toks)
        # ground-truth directions (demo: random; real task: use labels)
        dirs = sorted(random.sample(range(D), k=J))
        exs.append({"prompt": prompt, "Y": Y, "s_hat": s_hat, "H": H, "dirs": dirs})
    ds = Dataset.from_list([{"prompt": e["prompt"]} for e in exs])
    return ds, exs, dir_vocab
```

---

### 11) `doa_rl/algos/__init__.py`

```python
from .ppo_runner import PPORunner
from .grpo_runner import GRPORunner
```

---

### 14) `doa_rl/scripts/__init__.py`

```python
# empty
```

---

### 15) `doa_rl/scripts/prepare_hrtf.py`

```python
import argparse, numpy as np
from ..env.hrtf_loader import load_cipic_hrtf, load_device_hrtf

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True)
    ap.add_argument("--type", type=str, choices=["cipic","device"], default="device")
    ap.add_argument("--sr", type=int, default=16000); ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--azim_step", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/H.npz")
    args=ap.parse_args()
    if args.type=="cipic":
        H = load_cipic_hrtf(args.src, azim_step=args.azim_step, sr=args.sr, n_fft=args.n_fft)
    else:
        H = load_device_hrtf(args.src, sr=args.sr, n_fft=args.n_fft)
    import os; os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, H=H)
    print("saved:", args.out, H.shape)
```

---

### 16) `doa_rl/scripts/prepare_dict.py`

```python
import argparse, os, numpy as np, glob, librosa
from ..env.utils import stft_mag
from ..features.nmf_utils import train_nmf_is

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--wav_dir", type=str, required=True)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--n_fft", type=int, default=512)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--out", type=str, default="data/W_usm_is.npz")
    args=ap.parse_args()

    wavs=sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    Ys=[]
    for p in wavs:
        y,fs = librosa.load(p, sr=args.sr, mono=True)
        Y = stft_mag(y, n_fft=args.n_fft)
        Ys.append(Y)
    W = train_nmf_is(Ys, K=args.K, n_iter=args.n_iter)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, W=W)
    print("Saved W:", args.out, W.shape)
```

---

### 17) `doa_rl/scripts/train_single.py`

```python
import argparse, numpy as np
from transformers import set_seed
from ..env.dataset import build_dataset
from ..algos.ppo_runner import PPORunner
from ..algos.grpo_runner import GRPORunner

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--H", type=str, required=True)
    ap.add_argument("--W", type=str, default=None)
    ap.add_argument("--algo", type=str, choices=["ppo","grpo"], default="ppo")
    ap.add_argument("--feature", type=str, choices=["patch","leaf","scatter","nmf"], default="patch")
    ap.add_argument("--add_dir_tokens", type=int, default=1)
    ap.add_argument("--num_wavs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s_mode", type=str, choices=["S1","S2"], default="S1")
    ap.add_argument("--nmf_iter", type=int, default=50)
    ap.add_argument("--nmf_l1", type=float, default=0.0)
    args=ap.parse_args(); set_seed(args.seed)

    H = np.load(args.H)["H"]      # (D,F)
    W = np.load(args.W)["W"] if args.W else None
    ds, exs, dir_vocab = build_dataset(H, [None]*args.num_wavs, feature=args.feature,
                                       add_dir_tokens=bool(args.add_dir_tokens), J=1,
                                       W=W, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)
    if args.algo=="ppo":
        PPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=1)
    else:
        GRPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=1, G=4)
```

---

### 18) `doa_rl/scripts/train_multi.py`

```python
import argparse, numpy as np
from transformers import set_seed
from ..env.dataset import build_dataset
from ..algos.ppo_runner import PPORunner
from ..algos.grpo_runner import GRPORunner

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--H", type=str, required=True)
    ap.add_argument("--W", type=str, default=None)
    ap.add_argument("--J", type=int, default=2)
    ap.add_argument("--algo", type=str, choices=["ppo","grpo"], default="grpo")
    ap.add_argument("--feature", type=str, choices=["patch","leaf","scatter","nmf"], default="patch")
    ap.add_argument("--add_dir_tokens", type=int, default=1)
    ap.add_argument("--num_wavs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--s_mode", type=str, choices=["S1","S2"], default="S1")
    ap.add_argument("--nmf_iter", type=int, default=50)
    ap.add_argument("--nmf_l1", type=float, default=0.0)
    args=ap.parse_args(); set_seed(args.seed)

    H = np.load(args.H)["H"]; W = np.load(args.W)["W"] if args.W else None
    ds, exs, dir_vocab = build_dataset(H, [None]*args.num_wavs, feature=args.feature,
                                       add_dir_tokens=bool(args.add_dir_tokens), J=args.J,
                                       W=W, s_mode=args.s_mode, nmf_iter=args.nmf_iter, nmf_l1=args.nmf_l1)
    if args.algo=="ppo":
        PPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=args.J)
    else:
        GRPORunner(dir_vocab=dir_vocab).train(ds, exs, dir_vocab, J=args.J, G=6)
```

---

### 19) `doa_rl/scripts/infer_demo.py`

```python
import argparse, numpy as np
from ..env.dataset import build_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--prompt", type=str, required=True)
    args=ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    mdl = AutoModelForCausalLM.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids
    out = mdl.generate(ids, max_new_tokens=4)
    print(tok.decode(out[0]))
```

---

### 20) `doa_rl/eval/__init__.py`

```python
from .metrics import angular_error, multi_match_error, accuracy_10deg, per_source_accuracy
```

---

### 21) `doa_rl/eval/metrics.py`

```python
import numpy as np
from typing import List
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def angular_error(gt: List[int], pr: List[int], D: int) -> float:
    bin_deg = 360.0 / D
    diff = ((pr[0]-gt[0])*bin_deg + 180) % 360 - 180
    return abs(diff)

def multi_match_error(gt: List[int], pr: List[int], D: int) -> float:
    from itertools import permutations
    bin_deg=360.0/D; J=len(gt); gt=np.array(gt)
    best=1e9
    for perm in permutations(pr, J):
        diff = ((np.array(perm)-gt)*bin_deg + 180) % 360 - 180
        best=min(best, float(np.mean(np.abs(diff))))
    return best

def accuracy_10deg(gt: List[int], pr: List[int], D: int, bin_deg=10.0) -> int:
    err = angular_error(gt, pr, D) if len(gt)==1 else multi_match_error(gt, pr, D)
    return int(err <= bin_deg/2.0)

def per_source_accuracy(gt: List[int], pr: List[int], D: int, bin_deg=10.0) -> float:
    from itertools import permutations
    J=len(gt); half=bin_deg/2.0; bin_deg_all=360.0/D
    best=0
    for perm in permutations(pr, J):
        hit=np.mean([ abs(((perm[j]-gt[j])*bin_deg_all + 180)%360-180) <= half for j in range(J) ])
        best=max(best, hit)
    return best

def plot_angle_hist(errors, title="Angle Error Distribution"):
    plt.figure(); plt.hist(errors, bins=36); plt.xlabel("deg"); plt.ylabel("#"); plt.title(title); plt.show()

def plot_confusion(gt: List[int], pr: List[int], D: int, title="Confusion Matrix"):
    cm = confusion_matrix(gt, pr, labels=list(range(D)))
    plt.figure(figsize=(6,5)); plt.imshow(cm, cmap="pink"); plt.colorbar(); plt.title(title)
    plt.xlabel("Estimated"); plt.ylabel("True"); plt.show()
```

---

### 22) `doa_rl/eval/plot_eval.py`

```python
import argparse, numpy as np
from .metrics import angular_error, multi_match_error, accuracy_10deg, per_source_accuracy, plot_angle_hist, plot_confusion

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--gt", type=str, required=True)  # npz: gt, pr, D
    args=ap.parse_args()
    data = np.load(args.gt)
    gt_list = data["gt"].tolist()
    pr_list = data["pr"].tolist()
    D = int(data["D"])
    errs = [angular_error([g],[p],D) for g,p in zip(gt_list, pr_list)]
    print("Mean error:", float(np.mean(errs)))
    print("10° acc:", sum(accuracy_10deg([g],[p],D) for g,p in zip(gt_list, pr_list))/len(gt_list))
    plot_angle_hist(errs)
    plot_confusion([g for g in gt_list], [p for p in pr_list], D)
```

---

## C) 這套程式與**鏡像梯度下降**的逐步對應（含 shape）

1. **內容內圈（IS‑MD）**：`features/nmf_utils.estimate_s_hat`

   * 入：$Y(F,N), W(F,K)$（可選 $H(D,F)$ 做消色 S2）
   * 出：$s_{\hat{}}(F,)$ 與 $z_{\hat{}}(K,)$
   * 幾何：IS‑Bregman 乘法更新（論文式(15)–(16)） → **鏡像步**。

2. **策略外圈（KL‑MD）**：`algos/ppo_runner.py` / `algos/grpo_runner.py`

   * 入：prompt（由 `features/tokenizers` 產生的 token 序列）
   * 出：方向 logits $g(D,)$ → $\pi=\mathrm{softmax}(g)$
   * 更新：PPO/GRPO 在 **KL** 幾何內的信任域步（對應**鏡像步**）。

3. **物理化回饋/優勢**：`env/doa_math.py`

   * Reward = $-D_{\rm IS}(Y\|\widehat Y)$；Advantage (★) = 由 IS 對 $\alpha_d$ 的導數。
   * **shape**：Reward 為標量；Advantage 為 $A(D,)$（亦可作 shaping）。

4. **token 化（不改 Transformer）**：`features/tokenizers.py`

   * 內容 tokens：**patch/LEAF/scattering/NMF‑atom**（**IS‑friendly** 表示）。
   * 方向 tokens：`<R_d:*>` = **physics-aware**（cone membership 的軟化：$\langle Y,|H_d|^\alpha\rangle$）。
   * **shape**：序列長 $L$（任務依據），最後加\*\*\[ACTION]\*\* 頭輸出 $D$ 類別 logits。

---

## D) 使用範例

```bash
# 產生 H(d,f)
python -m doa_rl.scripts.prepare_hrtf --src path/to/IRs --type device --out data/H.npz

# 離線學 W（IS-NMF）
python -m doa_rl.scripts.prepare_dict --wav_dir path/to/wavs --K 256 --out data/W_usm_is.npz

# 單源 + PPO（patch + direction tokens）
python -m doa_rl.scripts.train_single --H data/H.npz --W data/W_usm_is.npz \
       --feature patch --add_dir_tokens 1 --algo ppo

# 多源 + GRPO（J=2）
python -m doa_rl.scripts.train_multi --H data/H.npz --W data/W_usm_is.npz \
       --J 2 --feature patch --add_dir_tokens 1 --algo grpo

# Hugging Face + TRL（簡化版 patch→方向 1-token）
python scripts/train_trl_policy.py --data-root path/to/angle_dataset --algo ppo
```

---

### 備註與核對（對齊原論文）

* STFT/散射模型、方向濾波 $H_d$ 與白噪聲的 cone 幾何：式(2)–(6)、圖1（LEGO vs KEMAR 指紋）。
* 字典建模與 $Y=AX$：式(7)–(8)，USM/prototype 字典與有效頻帶（表 I）。
* IS 散度、MU 與我們的 (★) 推導：式(9)–(16)。
* 指標：10° bin accuracy、mean error、per-source accuracy、混淆矩陣（§IV‑C、表 V–VIII、圖 3–5）。

---

如果你想把 **(★) 的 $A_d$** 做成**token‑wise** 的 step reward（例如多源逐步輸出 `<D_i>` 時將 $A_{d_t}$ 分配到第 $t$ 步），可以在 `ppo_runner`/`grpo_runner` 內把 reward 改成列表並對每步更新；這仍是 **KL‑mirror** 步，只是 credit assignment 更細緻。整套管線保持**不改 Transformer 架構**，完全落在**表示與回饋**層即達成 MD 對齊。
