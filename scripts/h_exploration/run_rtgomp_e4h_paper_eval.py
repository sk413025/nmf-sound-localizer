import argparse
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.io import wavfile

from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

CAPTURE_TOL = 1e-6


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def load_wav_float32(path: Path, *, expected_fs: int) -> np.ndarray:
    fs, wav = wavfile.read(path)
    if int(fs) != int(expected_fs):
        raise SystemExit(f"WAV fs mismatch for {path}: expected {expected_fs}, got {fs}")
    if wav.ndim != 1:
        raise SystemExit(f"Unsupported WAV shape for {path}: expected mono (N,), got {wav.shape}")

    if wav.dtype == np.int16:
        x = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        x = wav.astype(np.float32) / 2147483648.0
    elif wav.dtype == np.float32:
        x = wav
    elif wav.dtype == np.float64:
        x = wav.astype(np.float32)
    else:
        raise SystemExit(f"Unsupported WAV dtype for {path}: {wav.dtype}")

    max_abs = float(np.max(np.abs(x)))
    if max_abs > 1e-9:
        x = x / max_abs
    return x.astype(np.float32, copy=False)


def design_bandpass(fs: int, *, f_lo: float, f_hi: float) -> Tuple[np.ndarray, np.ndarray]:
    nyq = 0.5 * float(fs)
    lo = float(f_lo) / nyq
    hi = float(f_hi) / nyq
    if not (0.0 < lo < hi < 1.0):
        raise SystemExit(f"Invalid bandpass normalized frequencies: lo={lo}, hi={hi}")
    b, a = signal.butter(4, [lo, hi], btype="bandpass")
    return b, a


def gcc_phat_delay(
    x: np.ndarray,
    y: np.ndarray,
    *,
    fs: int,
    tau_center_samples: int,
    search_radius_samples: int,
    b: np.ndarray,
    a: np.ndarray,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Return (tau_hat_samples, tau_hat_ms, psr, boundary_hit)."""
    if x.size == 0 or y.size == 0:
        return None, None, None, None
    if x.size != y.size:
        raise RuntimeError(f"GCC-PHAT requires equal-length segments; got {x.size} and {y.size}")

    # Match STFT band: remove DC and bandpass (zero-phase).
    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32, copy=False))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32, copy=False))

    # Choose FFT length large enough to cover the search window around tau_center.
    base_nfft = next_pow2(2 * int(x_f.size))
    need_half = int(abs(tau_center_samples)) + int(search_radius_samples) + 1
    nfft = base_nfft if need_half <= base_nfft // 2 else next_pow2(4 * int(x_f.size))

    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)
    # Sign convention: tau_hat > 0 means y lags x (y[t] ~= x[t - tau_hat]).
    # For this convention we need the cross-power spectrum conj(X) * Y.
    R = np.conj(X) * Y
    R /= (np.abs(R) + eps)
    cc = np.fft.irfft(R, n=nfft)
    cc = np.concatenate([cc[-(nfft // 2) :], cc[: nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)

    lo = int(tau_center_samples) - int(search_radius_samples)
    hi = int(tau_center_samples) + int(search_radius_samples)
    sel = (lags >= lo) & (lags <= hi)
    if not np.any(sel):
        return None, None, None, None

    cc_sel = cc[sel]
    lags_sel = lags[sel]
    abs_cc = np.abs(cc_sel)
    peak_i = int(np.argmax(abs_cc))
    peak_lag = int(lags_sel[peak_i])
    peak_abs = float(abs_cc[peak_i])
    boundary_hit = bool(peak_lag == lo or peak_lag == hi)

    # Parabolic sub-sample refinement on |cc| (if interior).
    tau_hat = float(peak_lag)
    if 0 < peak_i < abs_cc.size - 1:
        y1, y2, y3 = float(abs_cc[peak_i - 1]), float(abs_cc[peak_i]), float(abs_cc[peak_i + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-12:
            offset = 0.5 * (y1 - y3) / denom
            if -1.0 <= offset <= 1.0:
                tau_hat = tau_hat + float(offset)

    # PSR: peak / median sidelobe within the search window (excluding a small neighborhood).
    exc = int(max(1, exclusion_radius_samples))
    mask = np.ones(abs_cc.size, dtype=bool)
    mask[max(0, peak_i - exc) : min(abs_cc.size, peak_i + exc + 1)] = False
    sidelobe = abs_cc[mask]
    if sidelobe.size == 0:
        psr = None
    else:
        psr = float(peak_abs / (float(np.median(sidelobe)) + eps))

    tau_ms = float(tau_hat) * 1000.0 / float(fs)
    return float(tau_hat), float(tau_ms), psr, boundary_hit


def gcc_phat_abs_cc(
    x: np.ndarray,
    y: np.ndarray,
    *,
    b: np.ndarray,
    a: np.ndarray,
    nfft: int,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lags[int], abs(cc)[float]) for GCC-PHAT correlation."""
    if x.size == 0 or y.size == 0:
        raise RuntimeError("GCC-PHAT requires non-empty segments.")
    if x.size != y.size:
        raise RuntimeError(f"GCC-PHAT requires equal-length segments; got {x.size} and {y.size}")
    if nfft <= 0:
        raise RuntimeError(f"Invalid nfft={nfft}")

    x_f = signal.filtfilt(b, a, (x - np.mean(x)).astype(np.float32, copy=False))
    y_f = signal.filtfilt(b, a, (y - np.mean(y)).astype(np.float32, copy=False))

    X = np.fft.rfft(x_f, n=nfft)
    Y = np.fft.rfft(y_f, n=nfft)
    # Sign convention: tau_hat > 0 means y lags x (y[t] ~= x[t - tau_hat]).
    R = np.conj(X) * Y
    R /= (np.abs(R) + eps)
    cc = np.fft.irfft(R, n=nfft)
    cc = np.concatenate([cc[-(nfft // 2) :], cc[: nfft // 2]])
    lags = np.arange(-(nfft // 2), nfft // 2, dtype=np.int64)
    return lags, np.abs(cc)


def gcc_phat_pick_peak(
    lags: np.ndarray,
    abs_cc: np.ndarray,
    *,
    fs: int,
    tau_center_samples: int,
    search_radius_samples: int,
    exclusion_radius_samples: int = 16,
    eps: float = 1e-12,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[bool]]:
    """Pick the GCC-PHAT peak within [tau_center - r, tau_center + r]."""
    if lags.size == 0 or abs_cc.size == 0:
        return None, None, None, None
    if lags.size != abs_cc.size:
        raise RuntimeError(f"lags/abs_cc size mismatch: {lags.size} vs {abs_cc.size}")

    lo = int(tau_center_samples) - int(search_radius_samples)
    hi = int(tau_center_samples) + int(search_radius_samples)
    sel = (lags >= lo) & (lags <= hi)
    if not np.any(sel):
        return None, None, None, None

    abs_cc_sel = abs_cc[sel]
    lags_sel = lags[sel]
    peak_i = int(np.argmax(abs_cc_sel))
    peak_lag = int(lags_sel[peak_i])
    peak_abs = float(abs_cc_sel[peak_i])
    boundary_hit = bool(peak_lag == lo or peak_lag == hi)

    tau_hat = float(peak_lag)
    if 0 < peak_i < abs_cc_sel.size - 1:
        y1, y2, y3 = float(abs_cc_sel[peak_i - 1]), float(abs_cc_sel[peak_i]), float(abs_cc_sel[peak_i + 1])
        denom = (y1 - 2.0 * y2 + y3)
        if abs(denom) > 1e-12:
            offset = 0.5 * (y1 - y3) / denom
            if -1.0 <= offset <= 1.0:
                tau_hat = tau_hat + float(offset)

    exc = int(max(1, exclusion_radius_samples))
    mask = np.ones(abs_cc_sel.size, dtype=bool)
    mask[max(0, peak_i - exc) : min(abs_cc_sel.size, peak_i + exc + 1)] = False
    sidelobe = abs_cc_sel[mask]
    if sidelobe.size == 0:
        psr = None
    else:
        psr = float(peak_abs / (float(np.median(sidelobe)) + eps))

    tau_ms = float(tau_hat) * 1000.0 / float(fs)
    return float(tau_hat), float(tau_ms), psr, boundary_hit


@dataclass
class RandomCaptureResult:
    capture_mean: torch.Tensor
    capture_trials: List[torch.Tensor]
    rand_ids_trials: List[np.ndarray]
    duplicate_count: int
    duplicate_rate: float


class SeqDT_FreqAware(nn.Module):
    def __init__(
        self,
        M_lags: int = 16,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        max_freq: int = 1025,
        rtg_dim: int = 2,
        action_dim: Optional[int] = None,
    ):
        super().__init__()
        if rtg_dim not in (1, 2):
            raise ValueError("rtg_dim must be 1 or 2")
        self.rtg_dim = rtg_dim

        self.rtg_embed = nn.Linear(rtg_dim, d_model)
        self.state_embed = nn.Linear(M_lags, d_model)
        self.freq_embed = nn.Embedding(max_freq, d_model)
        self.corr_norm = nn.LayerNorm(M_lags)
        self.layer_norm = nn.LayerNorm(d_model)
        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.1,
        )
        out_dim = M_lags if action_dim is None else int(action_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, x, rtg, freq_idx):
        # x: (B, K, M)
        B, K, M = x.shape
        x = self.corr_norm(x)
        s_emb = self.state_embed(x)
        if rtg.dim() == 2:
            rtg_in = rtg.unsqueeze(-1)
        else:
            rtg_in = rtg
        r_emb = self.rtg_embed(rtg_in)
        f_emb = self.freq_embed(freq_idx).unsqueeze(1).expand(-1, K, -1)
        h = self.layer_norm(s_emb + r_emb + f_emb)
        out, _ = self.rnn(h)
        logits = self.head(out)
        return logits


def manual_complex_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    return torch.sqrt(
        tensor.real.pow(2).sum(dim=dim, keepdim=keepdim)
        + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim)
        + 1e-10
    )


def gather_atoms(dict_atoms, ids_till_k):
    # dict_atoms: (F, Tw, M); ids_till_k: (F, k)
    _, tw, _ = dict_atoms.shape
    ids_exp = ids_till_k.unsqueeze(1).expand(-1, tw, -1)
    return torch.gather(dict_atoms, 2, ids_exp)


def normalize_logc(c: float, logc_min: float, logc_max: float) -> float:
    if logc_max == logc_min:
        return 0.0
    return float(np.clip((np.log10(c) - logc_min) / (logc_max - logc_min), 0.0, 1.0))


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_pairing_mode(dataset: DoALagDataset, pairing_mode: str) -> None:
    """Modify dataset.clips in-place for guardrail diagnostics (still real data)."""
    if pairing_mode == "paired":
        return
    if pairing_mode != "mispair_shift1":
        raise SystemExit(f"Unsupported pairing_mode: {pairing_mode}")
    if len(dataset.clips) < 2:
        raise SystemExit("mispair_shift1 requires dataset length >= 2")

    # Keep MIC order, shift LDV by +1 with cyclic wrap. This should break time/content alignment.
    clips = list(dataset.clips)
    shifted = []
    n = len(clips)
    for i, (mic_path, _ldv_path) in enumerate(clips):
        shifted_ldv = clips[(i + 1) % n][1]
        shifted.append((mic_path, shifted_ldv))
    dataset.clips = shifted


def median_and_mad(values: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if values.size == 0:
        return None, None
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def generate_subset_manifest(
    dataset: DoALagDataset,
    mic_root: str,
    ldv_root: str,
    num_pairs: int,
    out_path: Path,
) -> Dict[str, object]:
    if num_pairs > len(dataset):
        raise SystemExit(f"Dataset too small: len(dataset)={len(dataset)} < num_pairs={num_pairs}")

    file_hashes: List[Dict[str, str]] = []
    for idx in range(num_pairs):
        mic_path, ldv_path = dataset.clips[idx]
        mic_path = Path(mic_path)
        ldv_path = Path(ldv_path)
        if not mic_path.exists():
            raise SystemExit(f"Missing mic file: {mic_path}")
        if not ldv_path.exists():
            raise SystemExit(f"Missing ldv file: {ldv_path}")
        file_hashes.append({"path": str(mic_path), "md5": md5_file(mic_path)})
        file_hashes.append({"path": str(ldv_path), "md5": md5_file(ldv_path)})

    lines = sorted([f'{r["md5"]}  {r["path"]}' for r in file_hashes])
    fingerprint_md5 = hashlib.md5("\n".join(lines).encode("utf-8")).hexdigest()

    manifest = {
        "mic_root": mic_root,
        "ldv_root": ldv_root,
        "num_pairs": int(num_pairs),
        "num_files": int(len(file_hashes)),
        "selection": f"first {num_pairs} clip pairs in dataset order (all angles)",
        "file_hashes": file_hashes,
        "fingerprint_md5": fingerprint_md5,
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_manifest(dataset: DoALagDataset, manifest: Dict[str, object]) -> Tuple[int, int]:
    missing_files = 0
    md5_mismatches = 0
    file_hashes = manifest["file_hashes"]
    expected_paths = [entry["path"] for entry in file_hashes]

    actual_paths = []
    for idx in range(int(manifest["num_pairs"])):
        mic_path, ldv_path = dataset.clips[idx]
        actual_paths.extend([str(mic_path), str(ldv_path)])

    if expected_paths != actual_paths:
        raise SystemExit("Subset mismatch: manifest order does not match dataset order.")

    for entry in file_hashes:
        path = Path(entry["path"])
        if not path.exists():
            missing_files += 1
            continue
        md5_now = md5_file(path)
        if md5_now != entry["md5"]:
            md5_mismatches += 1
    return missing_files, md5_mismatches


class RunningStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")

    def update(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.sumsq += value * value
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def mean(self) -> Optional[float]:
        if self.count == 0:
            return None
        return self.sum / self.count

    def std(self) -> Optional[float]:
        if self.count == 0:
            return None
        mean = self.sum / self.count
        var = max(self.sumsq / self.count - mean * mean, 0.0)
        return float(np.sqrt(var))


class P2Quantile:
    def __init__(self, q: float):
        self.q = q
        self.n = 0
        self.initial: List[float] = []
        self.qv: List[float] = []
        self.npos: List[float] = []
        self.np: List[float] = []
        self.dn: List[float] = []

    def update(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if self.n < 5:
            self.initial.append(x)
            self.n += 1
            if self.n == 5:
                self.initial.sort()
                self.qv = [float(v) for v in self.initial]
                self.npos = [1, 2, 3, 4, 5]
                self.np = [1, 1 + 2 * self.q, 1 + 4 * self.q, 3 + 2 * self.q, 5]
                self.dn = [0.0, self.q / 2.0, self.q, (1 + self.q) / 2.0, 1.0]
            return

        self.n += 1
        if x < self.qv[0]:
            self.qv[0] = x
            k = 0
        elif x < self.qv[1]:
            k = 0
        elif x < self.qv[2]:
            k = 1
        elif x < self.qv[3]:
            k = 2
        elif x <= self.qv[4]:
            k = 3
        else:
            self.qv[4] = x
            k = 3

        for i in range(k + 1, 5):
            self.npos[i] += 1
        for i in range(5):
            self.np[i] += self.dn[i]

        for i in range(1, 4):
            d = self.np[i] - self.npos[i]
            if (d >= 1 and self.npos[i + 1] - self.npos[i] > 1) or (d <= -1 and self.npos[i] - self.npos[i - 1] > 1):
                s = 1 if d >= 0 else -1
                qv_new = self.qv[i] + (s / (self.npos[i + 1] - self.npos[i - 1])) * (
                    (self.npos[i] - self.npos[i - 1] + s) * (self.qv[i + 1] - self.qv[i]) / (self.npos[i + 1] - self.npos[i])
                    + (self.npos[i + 1] - self.npos[i] - s) * (self.qv[i] - self.qv[i - 1]) / (self.npos[i] - self.npos[i - 1])
                )
                if self.qv[i - 1] < qv_new < self.qv[i + 1]:
                    self.qv[i] = qv_new
                else:
                    self.qv[i] = self.qv[i] + s * (self.qv[i + s] - self.qv[i]) / (self.npos[i + s] - self.npos[i])
                self.npos[i] += s

    def result(self) -> Optional[float]:
        if self.n == 0:
            return None
        if self.n <= 5:
            return float(np.median(self.initial))
        return float(self.qv[2])


def rankdata(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=float)
    sorted_vals = arr[order]
    n = len(arr)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def spearmanr(values_x: List[float], values_y: List[float]) -> Optional[float]:
    if len(values_x) == 0 or len(values_x) != len(values_y):
        return None
    rx = rankdata(values_x)
    ry = rankdata(values_y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return None
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr)


def update_capture_integrity(
    *,
    method: str,
    capture_val: float,
    e0: float,
    e_res: float,
    eval_context: str,
    lambda_c: Optional[float],
    k_selected: Optional[int],
    steps_decision: Optional[int],
    clip_idx: int,
    window_idx: int,
    freq_idx: int,
    random_trial: Optional[int],
    unique_count: Optional[int],
    duplicate_count: Optional[int],
    capture_minmax: Dict[str, Dict[str, float]],
    out_of_range_counts: Dict[str, int],
    worst_violation: Dict[str, object],
    diag_fp,
    capture_tol: float,
) -> None:
    minmax = capture_minmax[method]
    if capture_val < minmax["min"]:
        minmax["min"] = capture_val
    if capture_val > minmax["max"]:
        minmax["max"] = capture_val

    if capture_val < -capture_tol or capture_val > 1.0 + capture_tol:
        out_of_range_counts[method] += 1
        violation = max(-capture_val, capture_val - 1.0)
        if violation > worst_violation["abs"]:
            worst_violation["abs"] = violation
            worst_violation["details"] = {
                "method": method,
                "eval_context": eval_context,
                "capture": capture_val,
                "violation": violation,
                "lambda_c": lambda_c,
                "k_selected": k_selected,
                "steps_decision": steps_decision,
                "clip_idx": int(clip_idx),
                "window_idx": int(window_idx),
                "freq_idx": int(freq_idx),
                "random_trial": random_trial,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count,
                "E0": e0,
                "E_res": e_res,
            }
        if diag_fp is not None:
            row = {
                "method": method,
                "eval_context": eval_context,
                "capture": capture_val,
                "lambda_c": lambda_c,
                "k_selected": k_selected,
                "steps_decision": steps_decision,
                "clip_idx": int(clip_idx),
                "window_idx": int(window_idx),
                "freq_idx": int(freq_idx),
                "random_trial": random_trial,
                "unique_count": unique_count,
                "duplicate_count": duplicate_count,
                "E0": e0,
                "E_res": e_res,
            }
            diag_fp.write(json.dumps(row) + "\n")


def compute_omp_capture_by_k(
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    eps_energy: float,
    tol: float,
) -> Tuple[torch.Tensor, int]:
    F, _, M_lags = D.shape
    res = Y.clone()
    chosen = torch.zeros(F, 0, dtype=torch.long)
    captures = torch.zeros(F, max_k)
    prev_res_energy = None
    violations = 0

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        if chosen.numel() > 0:
            abs_corrs.scatter_(1, chosen, -1.0)
        act = torch.argmax(abs_corrs, dim=1).unsqueeze(1)
        chosen = torch.cat([chosen, act], dim=1)

        active = gather_atoms(D, chosen)
        sol = torch.linalg.lstsq(active, Y).solution
        recon = active @ sol
        res = Y - recon
        res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2

        if prev_res_energy is not None:
            if torch.any(res_energy > prev_res_energy + tol):
                violations += int(torch.sum(res_energy > prev_res_energy + tol).item())
        prev_res_energy = res_energy

        capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
        captures[:, k] = capture

    return captures, violations


def compute_random_capture_by_k(
    D: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    random_trials: int,
    rng: np.random.Generator,
    eps_energy: float,
    random_sampling: str,
) -> RandomCaptureResult:
    F, _, M_lags = D.shape
    if random_sampling not in ("without_replacement", "with_replacement"):
        raise ValueError(f"Unsupported random_sampling: {random_sampling}")
    if random_sampling == "without_replacement" and max_k > M_lags:
        raise ValueError(f"max_k={max_k} exceeds M_lags={M_lags} for without-replacement sampling.")

    capture_trials: List[torch.Tensor] = []
    rand_ids_trials: List[np.ndarray] = []
    captures_sum = torch.zeros(F, max_k)
    duplicate_count = 0
    duplicate_possible = 0

    for _ in range(random_trials):
        if random_sampling == "with_replacement":
            rand_ids_np = rng.integers(0, M_lags, size=(F, max_k))
            unique_counts = np.apply_along_axis(lambda row: np.unique(row).size, 1, rand_ids_np)
            duplicate_count += int(np.sum(max_k - unique_counts))
            duplicate_possible += int(F * max_k)
        else:
            rand_ids_np = np.empty((F, max_k), dtype=np.int64)
            for f in range(F):
                rand_ids_np[f] = rng.choice(M_lags, size=max_k, replace=False)
            duplicate_possible += int(F * max_k)

        rand_ids_trials.append(rand_ids_np)
        rand_ids = torch.from_numpy(rand_ids_np).long()
        captures = torch.zeros(F, max_k)
        for k in range(max_k):
            active = gather_atoms(D, rand_ids[:, : k + 1])
            sol = torch.linalg.lstsq(active, Y).solution
            recon = active @ sol
            res_energy = manual_complex_norm((Y - recon).squeeze(), dim=1) ** 2
            capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
            captures[:, k] = capture

        capture_trials.append(captures)
        captures_sum += captures

    captures_mean = captures_sum / float(random_trials)
    duplicate_rate = float(duplicate_count) / float(duplicate_possible) if duplicate_possible > 0 else 0.0
    return RandomCaptureResult(
        capture_mean=captures_mean,
        capture_trials=capture_trials,
        rand_ids_trials=rand_ids_trials,
        duplicate_count=duplicate_count,
        duplicate_rate=duplicate_rate,
    )


def compute_dt_forced_k_capture(
    model: nn.Module,
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    max_k: int,
    rtg0: float,
    rtg_dim: int,
    stop_id: int,
    eps_energy: float,
    freq_ids: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    F, _, M_lags = D.shape
    device = D.device
    D_cpu = D.cpu()
    Y_cpu = Y.cpu()
    res_cpu = Y_cpu.clone()
    res = res_cpu.to(device)
    mask = torch.zeros(F, M_lags, dtype=torch.bool, device=device)
    indices = torch.full((F, max_k), -1, dtype=torch.long, device=device)
    captures = torch.zeros(F, max_k, device=device)
    duplicates = 0

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        abs_corrs[mask] = -1.0

        if rtg_dim == 2:
            remaining = float(max_k - k) / max(float(max_k), 1.0)
            rtg_in = torch.stack(
                [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)],
                dim=-1,
            )
        else:
            rtg_in = torch.full((F,), rtg0, device=device)

        logits = model(abs_corrs.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
        logits[:, stop_id] = -float("inf")
        logits[:, :M_lags][mask] = -float("inf")
        act = torch.argmax(logits, dim=1)

        row_idx = torch.arange(F, device=device)
        dup_mask = mask[row_idx, act]
        if dup_mask.any():
            duplicates += int(torch.sum(dup_mask).item())
        mask[row_idx, act] = True
        indices[:, k] = act

        if (indices[:, : k + 1] < 0).any():
            raise RuntimeError("DT forced-K has invalid indices; duplicate masking failed.")
        active = gather_atoms(D_cpu, indices[:, : k + 1].cpu())
        sol = torch.linalg.lstsq(active, Y_cpu).solution
        recon = active @ sol
        res_cpu = Y_cpu - recon
        res = res_cpu.to(device)
        res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2
        captures[:, k] = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))

    return captures, duplicates


def compute_dt_free_rollout(
    model: nn.Module,
    D: torch.Tensor,
    D_norm: torch.Tensor,
    Y: torch.Tensor,
    initial_energy: torch.Tensor,
    *,
    lambda_c: float,
    logc_min: float,
    logc_max: float,
    max_k: int,
    rtg_dim: int,
    stop_id: int,
    eps_energy: float,
    freq_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    F, _, M_lags = D.shape
    device = D.device
    action_dim = M_lags + 1
    D_cpu = D.cpu()
    Y_cpu = Y.cpu()
    res_cpu = Y_cpu.clone()
    res = res_cpu.to(device)
    mask = torch.zeros(F, action_dim, dtype=torch.bool, device=device)
    indices = torch.full((F, max_k), -1, dtype=torch.long, device=device)
    stop_mask = torch.zeros(F, dtype=torch.bool, device=device)
    steps_decision = torch.full((F,), -1, dtype=torch.long, device=device)

    rtg0 = normalize_logc(lambda_c, logc_min, logc_max)

    for k in range(max_k):
        corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
        abs_corrs = torch.abs(corrs)
        abs_corrs[mask[:, :M_lags]] = -1.0

        if rtg_dim == 2:
            remaining = float(max_k - k) / max(float(max_k), 1.0)
            rtg_in = torch.stack(
                [torch.full((F,), rtg0, device=device), torch.full((F,), remaining, device=device)],
                dim=-1,
            )
        else:
            rtg_in = torch.full((F,), rtg0, device=device)

        logits = model(abs_corrs.unsqueeze(1), rtg_in.unsqueeze(1), freq_ids).squeeze(1)
        logits[mask] = -float("inf")
        act = torch.argmax(logits, dim=1)

        for f in range(F):
            if stop_mask[f]:
                continue
            a = int(act[f].item())
            if a == stop_id:
                stop_mask[f] = True
                steps_decision[f] = k + 1
                continue
            mask[f, a] = True
            indices[f, k] = a

        if bool(stop_mask.all()):
            break

        active = ~stop_mask
        if bool(active.any()):
            active_cpu = active.cpu()
            active_ids = indices[active, : k + 1].cpu()
            if (active_ids < 0).any():
                raise RuntimeError("DT free rollout has invalid indices for active frequencies.")
            active_dict = gather_atoms(D_cpu[active_cpu], active_ids)
            active_y = Y_cpu[active_cpu]
            sol = torch.linalg.lstsq(active_dict, active_y).solution
            recon = active_dict @ sol
            res_cpu[active_cpu] = active_y - recon
            res = res_cpu.to(device)

    for f in range(F):
        if steps_decision[f] < 0:
            steps_decision[f] = max_k

    res_energy = manual_complex_norm(res.squeeze(), dim=1) ** 2
    capture = 1.0 - (res_energy / torch.clamp(initial_energy, min=eps_energy))
    k_selected = torch.sum(indices >= 0, dim=1)
    first_ids = indices[:, 0].clone()
    return k_selected, steps_decision, capture, res_energy, first_ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--subset_manifest", type=str, default=None)
    p.add_argument("--mode", type=str, choices=["scale_check_subset", "full_dataset", "smoke"], required=True)
    p.add_argument("--num_pairs", type=int, default=None)
    p.add_argument("--hop_length", type=int, default=160)
    p.add_argument("--fs", type=int, default=16000)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--freq_min", type=float, default=300.0)
    p.add_argument("--freq_max", type=float, default=3000.0)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument("--rtg_dim", type=int, default=2)
    p.add_argument("--eps_energy", type=float, default=1e-12)
    p.add_argument("--lambda_c_values", type=str, required=True)
    p.add_argument("--random_trials", type=int, default=3)
    p.add_argument(
        "--random_sampling",
        type=str,
        choices=["without_replacement", "with_replacement"],
        default="without_replacement",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write_per_sample", type=int, default=0)
    p.add_argument(
        "--pairing_mode",
        type=str,
        default="paired",
        choices=["paired", "mispair_shift1"],
        help="Dataset pairing mode. mispair_shift1 is a guardrail diagnostic (still real WAV).",
    )
    p.add_argument(
        "--require_wav_only",
        type=int,
        default=0,
        help="If 1, fail fast if any selected dataset path is not a .wav (guards against running on .npy).",
    )
    p.add_argument(
        "--write_delay_diagnostics",
        type=int,
        default=0,
        help="If 1, write window-level lag/delay diagnostics to delay_diagnostics.jsonl and a summary JSON.",
    )
    p.add_argument(
        "--write_subsample_delay_diagnostics",
        type=int,
        default=0,
        help="If 1, write GCC-PHAT sub-sample delay diagnostics to subsample_delay_diagnostics.jsonl + summary JSON.",
    )
    p.add_argument(
        "--subsample_method",
        type=str,
        default="gcc_phat",
        help="Comma-separated list: gcc_phat[,phase_slope]. Only gcc_phat is implemented in this evaluator.",
    )
    p.add_argument(
        "--search_radius_frames",
        type=int,
        default=2,
        help="Search radius in STFT frames for sub-sample delay refinement (default: 2 frames => +/-20ms).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "auto"],
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary").mkdir(parents=True, exist_ok=True)

    mic_root = Path(args.mic_root)
    ldv_root = Path(args.ldv_root)
    if not mic_root.exists():
        raise SystemExit(f"Missing mic_root: {mic_root}")
    if not ldv_root.exists():
        raise SystemExit(f"Missing ldv_root: {ldv_root}")

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Requested --device mps, but MPS is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("Using device: %s", device)

    dataset = DoALagDataset(
        str(mic_root),
        str(ldv_root),
        angle=None,
        hop_length=int(args.hop_length),
    )
    apply_pairing_mode(dataset, str(args.pairing_mode))
    logger.info("Pairing mode: %s", args.pairing_mode)

    if args.mode == "scale_check_subset":
        expected_pairs = 48
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit(f"scale_check_subset requires num_pairs={expected_pairs}")
        num_pairs = expected_pairs
    elif args.mode == "full_dataset":
        expected_pairs = len(dataset)
        if args.num_pairs is not None and int(args.num_pairs) != expected_pairs:
            raise SystemExit("full_dataset requires num_pairs=len(dataset)")
        num_pairs = expected_pairs
    else:
        num_pairs = 3 if args.num_pairs is None else int(args.num_pairs)
        if num_pairs <= 0:
            raise SystemExit("smoke mode requires num_pairs > 0")
        if num_pairs > len(dataset):
            raise SystemExit(f"smoke num_pairs={num_pairs} exceeds dataset length {len(dataset)}")

    if int(args.require_wav_only) == 1:
        for idx in range(int(num_pairs)):
            mic_path, ldv_path = dataset.clips[idx]
            if not str(mic_path).endswith(".wav") or not str(ldv_path).endswith(".wav"):
                raise SystemExit(
                    "require_wav_only=1 but found non-wav path in dataset subset: "
                    f"{mic_path} , {ldv_path}"
                )

    manifest_path = Path(args.subset_manifest) if args.subset_manifest else (out_dir / "subset_manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = generate_subset_manifest(
            dataset,
            str(mic_root),
            str(ldv_root),
            num_pairs,
            manifest_path,
        )
        logger.info("Wrote subset manifest: %s", manifest_path)

    missing_files, md5_mismatches = validate_manifest(dataset, manifest)
    if missing_files > 0:
        raise SystemExit(f"Missing files in manifest: {missing_files}")

    if md5_mismatches > 0:
        raise SystemExit(f"MD5 mismatches in manifest: {md5_mismatches}")

    lambda_c_values = [float(x) for x in str(args.lambda_c_values).split(",") if x.strip() != ""]
    if not lambda_c_values:
        raise SystemExit("--lambda_c_values must be non-empty.")
    logc_vals = [np.log10(c) for c in lambda_c_values]
    logc_min, logc_max = min(logc_vals), max(logc_vals)

    subsample_methods = [m.strip() for m in str(args.subsample_method).split(",") if m.strip() != ""]
    if int(args.write_subsample_delay_diagnostics) == 1:
        if subsample_methods != ["gcc_phat"]:
            raise SystemExit(
                "Only subsample_method=gcc_phat is implemented. "
                f"Got subsample_method={args.subsample_method!r} (parsed={subsample_methods})."
            )
        if int(args.search_radius_frames) <= 0:
            raise SystemExit("--search_radius_frames must be > 0 for sub-sample diagnostics.")

    M_lags = int(args.max_lag) * 2 + 1
    action_dim = M_lags + 1
    state = torch.load(args.ckpt_path, map_location="cpu")
    state_mlags = int(state["state_embed.weight"].shape[1])
    head_out = int(state["head.2.weight"].shape[0])
    if state_mlags != M_lags:
        raise SystemExit(f"Checkpoint state_embed M_lags mismatch. Expected {M_lags}, got {state_mlags}.")
    if head_out != action_dim:
        raise SystemExit(f"Checkpoint head_out mismatch. Expected {action_dim}, got {head_out}.")

    model = SeqDT_FreqAware(M_lags=M_lags, rtg_dim=int(args.rtg_dim), action_dim=action_dim).to(device)
    model.load_state_dict(state)
    model.eval()

    tol = 1e-7
    forced_k_values = [1, 2, 4, 8, 16]
    forced_k_values = [k for k in forced_k_values if k <= int(args.max_k)]

    per_sample_path = out_dir / "per_sample.jsonl"
    per_sample_fp = per_sample_path.open("w", encoding="utf-8") if int(args.write_per_sample) == 1 else None
    diagnostics_path = out_dir / "integrity_diagnostics.jsonl"
    diagnostics_fp = diagnostics_path.open("w", encoding="utf-8")
    delay_path = out_dir / "delay_diagnostics.jsonl"
    delay_fp = delay_path.open("w", encoding="utf-8") if int(args.write_delay_diagnostics) == 1 else None
    subsample_path = out_dir / "subsample_delay_diagnostics.jsonl"
    subsample_fp = (
        subsample_path.open("w", encoding="utf-8") if int(args.write_subsample_delay_diagnostics) == 1 else None
    )
    bp_b = bp_a = None
    if subsample_fp is not None:
        bp_b, bp_a = design_bandpass(
            int(args.fs),
            f_lo=float(args.freq_min),
            f_hi=float(args.freq_max),
        )

    stats_by_lambda = {}
    steps_decision_stats = {}
    k_selected_stats = {}
    k_selected_quantiles = {}
    k_selected_hist = {}
    medians_by_lambda = {}
    delay_counts_by_lambda = {}
    delay_stats_by_lambda = {}
    delay_mad_quantiles_by_lambda = {}
    subsample_counts_by_lambda = {}
    subsample_tau_quantiles_by_lambda = {}
    subsample_psr_quantiles_by_lambda = {}
    subsample_boundary_rate_by_lambda = {}
    subsample_clip_taus_ms_by_lambda = {}
    for lambda_c in lambda_c_values:
        stats_by_lambda[lambda_c] = {
            "dt": RunningStats(),
            "omp": RunningStats(),
            "random": RunningStats(),
        }
        steps_decision_stats[lambda_c] = RunningStats()
        k_selected_stats[lambda_c] = RunningStats()
        k_selected_quantiles[lambda_c] = {
            "p50": P2Quantile(0.5),
            "p90": P2Quantile(0.9),
            "p99": P2Quantile(0.99),
        }
        k_selected_hist[lambda_c] = np.zeros(int(args.max_k) + 1, dtype=np.int64)
        medians_by_lambda[lambda_c] = {
            "dt": P2Quantile(0.5),
            "omp": P2Quantile(0.5),
            "random": P2Quantile(0.5),
        }
        delay_counts_by_lambda[lambda_c] = {
            "num_windows": 0,
            "num_windows_dt_first_lag_undefined": 0,
            "num_windows_dt_match_undefined": 0,
        }
        delay_stats_by_lambda[lambda_c] = {
            "dt_stop0_frac": RunningStats(),
            "dt_first_lag_mad_frames": RunningStats(),
            "dt_first_lag_median_frames": RunningStats(),
            "dt_first_lag_abs_le_1_frac": RunningStats(),
            "dt_first_lag_abs_le_2_frac": RunningStats(),
            "dt_k_selected_mean": RunningStats(),
            "dt_steps_decision_mean": RunningStats(),
            "dt_capture_mean": RunningStats(),
            "omp_first_lag_mad_frames": RunningStats(),
            "omp_first_lag_median_frames": RunningStats(),
            "dt_vs_omp_first_lag_match_frac": RunningStats(),
        }
        delay_mad_quantiles_by_lambda[lambda_c] = {
            "p50": P2Quantile(0.5),
            "p90": P2Quantile(0.9),
        }
        if subsample_fp is not None:
            subsample_counts_by_lambda[lambda_c] = {
                "dt": {"num_windows": 0, "num_defined": 0, "num_segment_oob": 0, "num_coarse_undefined": 0},
                "omp": {"num_windows": 0, "num_defined": 0, "num_segment_oob": 0, "num_coarse_undefined": 0},
            }
            subsample_tau_quantiles_by_lambda[lambda_c] = {
                "dt": {"p50": P2Quantile(0.5), "p90": P2Quantile(0.9)},
                "omp": {"p50": P2Quantile(0.5), "p90": P2Quantile(0.9)},
            }
            subsample_psr_quantiles_by_lambda[lambda_c] = {
                "dt": {"p50": P2Quantile(0.5), "p90": P2Quantile(0.9)},
                "omp": {"p50": P2Quantile(0.5), "p90": P2Quantile(0.9)},
            }
            subsample_boundary_rate_by_lambda[lambda_c] = {"dt": RunningStats(), "omp": RunningStats()}
            subsample_clip_taus_ms_by_lambda[lambda_c] = {
                "dt": [[] for _ in range(int(num_pairs))],
                "omp": [[] for _ in range(int(num_pairs))],
            }

    forced_stats = {}
    for k in forced_k_values:
        forced_stats[k] = {
            "dt": RunningStats(),
            "omp": RunningStats(),
            "random": RunningStats(),
        }

    num_samples_total = 0
    num_samples_used = 0
    num_nan_or_inf = 0
    num_capture_out_of_range_dt = 0
    num_capture_out_of_range_omp = 0
    num_capture_out_of_range_random = 0
    num_omp_monotonicity_violations = 0
    num_dt_duplicate_actions_forced_k = 0
    capture_minmax = {
        "dt": {"min": float("inf"), "max": float("-inf")},
        "omp": {"min": float("inf"), "max": float("-inf")},
        "random": {"min": float("inf"), "max": float("-inf")},
    }
    out_of_range_counts = {
        "dt": 0,
        "omp": 0,
        "random": 0,
    }
    worst_violation = {"abs": float("-inf"), "details": None}
    random_duplicate_total = 0
    random_duplicate_possible = 0

    rng = np.random.default_rng(int(args.seed))
    freq_ids = None
    freq_idx_cpu = None
    log_every = 1

    for clip_idx in range(num_pairs):
        mic_path, ldv_path = dataset.clips[clip_idx]
        item = dataset[clip_idx]
        mic_stft = item["mic_stft"]
        ldv_stft = item["ldv_stft"]

        if float(args.gain) != 1.0:
            mic_stft = mic_stft * float(args.gain)
            ldv_stft = ldv_stft * float(args.gain)

        T_total, F = ldv_stft.shape
        if freq_ids is None:
            expected_bins = int(args.n_fft) // 2 + 1
            if F != expected_bins:
                raise SystemExit(f"STFT F mismatch: expected {expected_bins}, got {F}")
            freqs = np.linspace(0.0, float(args.fs) / 2.0, expected_bins)
            idx = np.where((freqs >= float(args.freq_min)) & (freqs <= float(args.freq_max)))[0]
            if len(idx) == 0:
                raise SystemExit("Frequency band selection is empty; check freq_min/freq_max.")
            freq_idx_cpu = torch.from_numpy(idx.astype(np.int64))
            freq_ids_full = torch.arange(F, device=device)
            freq_ids = freq_ids_full[freq_idx_cpu.to(device)]

        Lag_Min = -int(args.max_lag)
        Lag_Max = int(args.max_lag)
        Lags = torch.arange(Lag_Min, Lag_Max + 1)

        start_limit = Lag_Max
        end_limit = T_total - int(args.tw) + Lag_Min
        if end_limit <= start_limit:
            continue

        valid_starts = list(range(start_limit, end_limit, int(args.tw)))
        if len(valid_starts) == 0:
            continue

        mic_wav_pad = ldv_wav_pad = None
        segment_len_samples = None
        if subsample_fp is not None:
            mic_wav = load_wav_float32(Path(mic_path), expected_fs=int(args.fs))
            ldv_wav = load_wav_float32(Path(ldv_path), expected_fs=int(args.fs))

            # Match scipy.signal.stft boundary='zeros' by pre-padding n_fft//2.
            pad_left = int(args.n_fft) // 2
            segment_len_samples = (int(args.tw) - 1) * int(args.hop_length) + int(args.n_fft)
            max_seg_end = int(max(valid_starts)) * int(args.hop_length) + int(segment_len_samples)

            mic_pad_right = max(pad_left, int(max_seg_end) - (pad_left + int(mic_wav.size)))
            ldv_pad_right = max(pad_left, int(max_seg_end) - (pad_left + int(ldv_wav.size)))
            mic_wav_pad = np.pad(mic_wav, (pad_left, mic_pad_right), mode="constant")
            ldv_wav_pad = np.pad(ldv_wav, (pad_left, ldv_pad_right), mode="constant")

        for w_idx, start_t in enumerate(valid_starts):
            end_t = start_t + int(args.tw)
            y_block = ldv_stft[start_t:end_t, :].T

            D = torch.zeros(F, int(args.tw), M_lags, dtype=mic_stft.dtype)
            for m, k_lag in enumerate(Lags):
                s = int(start_t - k_lag)
                e = int(s + int(args.tw))
                D[:, :, m] = mic_stft[s:e, :].T

            if freq_idx_cpu is not None:
                D = D[freq_idx_cpu]
                y_block = y_block[freq_idx_cpu]
            norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
            D_norm = D / norms
            Y = y_block.unsqueeze(2)
            initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2

            D_device = D.to(device)
            D_norm_device = D_norm.to(device)
            Y_device = Y.to(device)
            initial_energy_device = initial_energy.to(device)

            omp_capture_by_k, violations = compute_omp_capture_by_k(
                D,
                D_norm,
                Y,
                initial_energy,
                max_k=int(args.max_k),
                eps_energy=float(args.eps_energy),
                tol=tol * max(float(initial_energy.max().item()), 1.0),
            )
            num_omp_monotonicity_violations += int(violations)

            rand_seed = int(args.seed) + clip_idx * 100000 + w_idx
            rng = np.random.default_rng(rand_seed)
            random_result = compute_random_capture_by_k(
                D,
                Y,
                initial_energy,
                max_k=int(args.max_k),
                random_trials=int(args.random_trials),
                rng=rng,
                eps_energy=float(args.eps_energy),
                random_sampling=str(args.random_sampling),
            )

            rtg0_forced = normalize_logc(lambda_c_values[0], logc_min, logc_max)
            dt_forced_capture_by_k, dupes = compute_dt_forced_k_capture(
                model,
                D_device,
                D_norm_device,
                Y_device,
                initial_energy_device,
                max_k=int(args.max_k),
                rtg0=rtg0_forced,
                rtg_dim=int(args.rtg_dim),
                stop_id=M_lags,
                eps_energy=float(args.eps_energy),
                freq_ids=freq_ids,
            )
            num_dt_duplicate_actions_forced_k += int(dupes)

            F_band = int(Y.shape[0])
            random_duplicate_total += int(random_result.duplicate_count)
            random_duplicate_possible += int(F_band * int(args.max_k) * int(args.random_trials))
            for k in forced_k_values:
                idx_k = k - 1
                dt_vals = dt_forced_capture_by_k[:, idx_k]
                omp_vals = omp_capture_by_k[:, idx_k]
                rnd_vals = random_result.capture_mean[:, idx_k]
                for f in range(F_band):
                    dt_val = float(dt_vals[f].item())
                    omp_val = float(omp_vals[f].item())
                    rnd_val = float(rnd_vals[f].item())
                    e0 = float(initial_energy[f].item())
                    e0_clamped = max(e0, float(args.eps_energy))

                    for trial_idx in range(int(args.random_trials)):
                        trial_val = float(random_result.capture_trials[trial_idx][f, idx_k].item())
                        if not np.isfinite(trial_val):
                            num_nan_or_inf += 1
                            continue
                        if str(args.random_sampling) == "with_replacement":
                            ids = random_result.rand_ids_trials[trial_idx][f, :k]
                            unique_count = int(np.unique(ids).size)
                            duplicate_count = int(k - unique_count)
                        else:
                            unique_count = int(k)
                            duplicate_count = 0
                        e_res = float((1.0 - trial_val) * e0_clamped)
                        update_capture_integrity(
                            method="random",
                            capture_val=trial_val,
                            e0=e0,
                            e_res=e_res,
                            eval_context="forced_k",
                            lambda_c=None,
                            k_selected=int(k),
                            steps_decision=None,
                            clip_idx=clip_idx,
                            window_idx=w_idx,
                            freq_idx=f,
                            random_trial=trial_idx,
                            unique_count=unique_count,
                            duplicate_count=duplicate_count,
                            capture_minmax=capture_minmax,
                            out_of_range_counts=out_of_range_counts,
                            worst_violation=worst_violation,
                            diag_fp=diagnostics_fp,
                            capture_tol=CAPTURE_TOL,
                        )

                    if not np.isfinite(dt_val) or not np.isfinite(omp_val) or not np.isfinite(rnd_val):
                        num_nan_or_inf += 1
                        continue
                    update_capture_integrity(
                        method="dt",
                        capture_val=dt_val,
                        e0=e0,
                        e_res=float((1.0 - dt_val) * e0_clamped),
                        eval_context="forced_k",
                        lambda_c=None,
                        k_selected=int(k),
                        steps_decision=None,
                        clip_idx=clip_idx,
                        window_idx=w_idx,
                        freq_idx=f,
                        random_trial=None,
                        unique_count=None,
                        duplicate_count=None,
                        capture_minmax=capture_minmax,
                        out_of_range_counts=out_of_range_counts,
                        worst_violation=worst_violation,
                        diag_fp=diagnostics_fp,
                        capture_tol=CAPTURE_TOL,
                    )
                    update_capture_integrity(
                        method="omp",
                        capture_val=omp_val,
                        e0=e0,
                        e_res=float((1.0 - omp_val) * e0_clamped),
                        eval_context="forced_k",
                        lambda_c=None,
                        k_selected=int(k),
                        steps_decision=None,
                        clip_idx=clip_idx,
                        window_idx=w_idx,
                        freq_idx=f,
                        random_trial=None,
                        unique_count=None,
                        duplicate_count=None,
                        capture_minmax=capture_minmax,
                        out_of_range_counts=out_of_range_counts,
                        worst_violation=worst_violation,
                        diag_fp=diagnostics_fp,
                        capture_tol=CAPTURE_TOL,
                    )
                    forced_stats[k]["dt"].update(dt_val)
                    forced_stats[k]["omp"].update(omp_val)
                    forced_stats[k]["random"].update(rnd_val)

            need_coarse = (delay_fp is not None) or (subsample_fp is not None)
            omp_first_ids_cpu = None
            omp_med = None
            omp_mad = None
            if need_coarse:
                with torch.no_grad():
                    corrs0 = torch.bmm(D_norm_device.conj().transpose(1, 2), Y_device).squeeze(2)
                    omp_first_ids = torch.argmax(torch.abs(corrs0), dim=1)
                omp_first_ids_cpu = omp_first_ids.detach().cpu().numpy().astype(np.int64)
                omp_first_lags = omp_first_ids_cpu + int(Lag_Min)
                omp_med, omp_mad = median_and_mad(omp_first_lags.astype(float))

            dt_med_by_lambda: Optional[Dict[float, Optional[float]]] = {} if subsample_fp is not None else None
            for lambda_c in lambda_c_values:
                k_selected, steps_decision, dt_capture, dt_res_energy, dt_first_ids = compute_dt_free_rollout(
                    model,
                    D_device,
                    D_norm_device,
                    Y_device,
                    initial_energy_device,
                    lambda_c=lambda_c,
                    logc_min=logc_min,
                    logc_max=logc_max,
                    max_k=int(args.max_k),
                    rtg_dim=int(args.rtg_dim),
                    stop_id=M_lags,
                    eps_energy=float(args.eps_energy),
                    freq_ids=freq_ids,
                )
                if need_coarse:
                    dt_first_ids_cpu = dt_first_ids.detach().cpu().numpy()
                    dt_defined_mask = dt_first_ids_cpu >= 0
                    dt_stop0_frac = float(np.mean(~dt_defined_mask))
                    dt_first_lags = dt_first_ids_cpu[dt_defined_mask].astype(np.int64) + int(Lag_Min)
                    dt_med, dt_mad = median_and_mad(dt_first_lags.astype(float))
                    if dt_first_lags.size == 0:
                        dt_abs_le_1 = None
                        dt_abs_le_2 = None
                    else:
                        dt_abs_le_1 = float(np.mean(np.abs(dt_first_lags) <= 1))
                        dt_abs_le_2 = float(np.mean(np.abs(dt_first_lags) <= 2))

                    if dt_defined_mask.any():
                        match_frac = float(
                            np.mean(dt_first_ids_cpu[dt_defined_mask] == omp_first_ids_cpu[dt_defined_mask])
                        )
                    else:
                        match_frac = None

                    dt_k_mean = float(k_selected.float().mean().item())
                    dt_steps_mean = float(steps_decision.float().mean().item())
                    dt_cap_mean = float(dt_capture.float().mean().item())
                    if delay_fp is not None:
                        delay_counts_by_lambda[lambda_c]["num_windows"] += 1
                        delay_stats_by_lambda[lambda_c]["dt_stop0_frac"].update(dt_stop0_frac)
                        delay_stats_by_lambda[lambda_c]["dt_k_selected_mean"].update(dt_k_mean)
                        delay_stats_by_lambda[lambda_c]["dt_steps_decision_mean"].update(dt_steps_mean)
                        delay_stats_by_lambda[lambda_c]["dt_capture_mean"].update(dt_cap_mean)
                        delay_stats_by_lambda[lambda_c]["omp_first_lag_median_frames"].update(float(omp_med))
                        delay_stats_by_lambda[lambda_c]["omp_first_lag_mad_frames"].update(float(omp_mad))

                        if dt_med is None or dt_mad is None:
                            delay_counts_by_lambda[lambda_c]["num_windows_dt_first_lag_undefined"] += 1
                        else:
                            delay_stats_by_lambda[lambda_c]["dt_first_lag_median_frames"].update(float(dt_med))
                            delay_stats_by_lambda[lambda_c]["dt_first_lag_mad_frames"].update(float(dt_mad))
                            delay_mad_quantiles_by_lambda[lambda_c]["p50"].update(float(dt_mad))
                            delay_mad_quantiles_by_lambda[lambda_c]["p90"].update(float(dt_mad))
                        if dt_abs_le_1 is not None:
                            delay_stats_by_lambda[lambda_c]["dt_first_lag_abs_le_1_frac"].update(float(dt_abs_le_1))
                        if dt_abs_le_2 is not None:
                            delay_stats_by_lambda[lambda_c]["dt_first_lag_abs_le_2_frac"].update(float(dt_abs_le_2))
                        if match_frac is None:
                            delay_counts_by_lambda[lambda_c]["num_windows_dt_match_undefined"] += 1
                        else:
                            delay_stats_by_lambda[lambda_c]["dt_vs_omp_first_lag_match_frac"].update(float(match_frac))

                        row = {
                            "pairing_mode": str(args.pairing_mode),
                            "clip_idx": int(clip_idx),
                            "window_idx": int(w_idx),
                            "start_t": int(start_t),
                            "lambda_c": float(lambda_c),
                            "fs": int(args.fs),
                            "hop_length": int(args.hop_length),
                            "n_fft": int(args.n_fft),
                            "freq_min": float(args.freq_min),
                            "freq_max": float(args.freq_max),
                            "max_lag": int(args.max_lag),
                            "tw": int(args.tw),
                            "max_k": int(args.max_k),
                            "F_band": int(F_band),
                            "dt_stop0_frac": dt_stop0_frac,
                            "dt_first_lag_defined_frac": float(1.0 - dt_stop0_frac),
                            "dt_first_lag_median_frames": dt_med,
                            "dt_first_lag_mad_frames": dt_mad,
                            "dt_first_lag_abs_le_1_frac": dt_abs_le_1,
                            "dt_first_lag_abs_le_2_frac": dt_abs_le_2,
                            "dt_k_selected_mean": dt_k_mean,
                            "dt_steps_decision_mean": dt_steps_mean,
                            "dt_capture_mean": dt_cap_mean,
                            "omp_first_lag_median_frames": omp_med,
                            "omp_first_lag_mad_frames": omp_mad,
                            "dt_vs_omp_first_lag_match_frac": match_frac,
                        }
                        delay_fp.write(json.dumps(row) + "\n")

                    if dt_med_by_lambda is not None:
                        dt_med_by_lambda[lambda_c] = dt_med
                for f in range(F_band):
                    k_sel = int(k_selected[f].item())
                    steps = int(steps_decision[f].item())
                    dt_val = float(dt_capture[f].item())
                    if k_sel < 0 or k_sel > int(args.max_k):
                        raise RuntimeError(f"Invalid k_selected={k_sel} (max_k={args.max_k}).")
                    if steps < 1 or steps > int(args.max_k):
                        raise RuntimeError(f"Invalid steps_decision={steps} (max_k={args.max_k}).")
                    if k_sel == 0:
                        omp_val = 0.0
                        rnd_val = 0.0
                    else:
                        omp_val = float(omp_capture_by_k[f, k_sel - 1].item())
                        rnd_val = float(random_result.capture_mean[f, k_sel - 1].item())
                    if not np.isfinite(dt_val) or not np.isfinite(omp_val) or not np.isfinite(rnd_val):
                        num_nan_or_inf += 1
                        continue

                    e0 = float(initial_energy[f].item())
                    e0_clamped = max(e0, float(args.eps_energy))
                    update_capture_integrity(
                        method="dt",
                        capture_val=dt_val,
                        e0=e0,
                        e_res=float(dt_res_energy[f].item()),
                        eval_context="compute_matched",
                        lambda_c=float(lambda_c),
                        k_selected=k_sel,
                        steps_decision=steps,
                        clip_idx=clip_idx,
                        window_idx=w_idx,
                        freq_idx=f,
                        random_trial=None,
                        unique_count=None,
                        duplicate_count=None,
                        capture_minmax=capture_minmax,
                        out_of_range_counts=out_of_range_counts,
                        worst_violation=worst_violation,
                        diag_fp=diagnostics_fp,
                        capture_tol=CAPTURE_TOL,
                    )
                    update_capture_integrity(
                        method="omp",
                        capture_val=omp_val,
                        e0=e0,
                        e_res=float((1.0 - omp_val) * e0_clamped),
                        eval_context="compute_matched",
                        lambda_c=float(lambda_c),
                        k_selected=k_sel,
                        steps_decision=steps,
                        clip_idx=clip_idx,
                        window_idx=w_idx,
                        freq_idx=f,
                        random_trial=None,
                        unique_count=None,
                        duplicate_count=None,
                        capture_minmax=capture_minmax,
                        out_of_range_counts=out_of_range_counts,
                        worst_violation=worst_violation,
                        diag_fp=diagnostics_fp,
                        capture_tol=CAPTURE_TOL,
                    )

                    for trial_idx in range(int(args.random_trials)):
                        if k_sel == 0:
                            trial_val = 0.0
                        else:
                            trial_val = float(random_result.capture_trials[trial_idx][f, k_sel - 1].item())
                        if not np.isfinite(trial_val):
                            num_nan_or_inf += 1
                            continue
                        if str(args.random_sampling) == "with_replacement":
                            ids = random_result.rand_ids_trials[trial_idx][f, :k_sel]
                            unique_count = int(np.unique(ids).size)
                            duplicate_count = int(k_sel - unique_count)
                        else:
                            unique_count = int(k_sel)
                            duplicate_count = 0
                        e_res = float((1.0 - trial_val) * e0_clamped)
                        update_capture_integrity(
                            method="random",
                            capture_val=trial_val,
                            e0=e0,
                            e_res=e_res,
                            eval_context="compute_matched",
                            lambda_c=float(lambda_c),
                            k_selected=k_sel,
                            steps_decision=steps,
                            clip_idx=clip_idx,
                            window_idx=w_idx,
                            freq_idx=f,
                            random_trial=trial_idx,
                            unique_count=unique_count,
                            duplicate_count=duplicate_count,
                            capture_minmax=capture_minmax,
                            out_of_range_counts=out_of_range_counts,
                            worst_violation=worst_violation,
                            diag_fp=diagnostics_fp,
                            capture_tol=CAPTURE_TOL,
                        )

                    stats_by_lambda[lambda_c]["dt"].update(dt_val)
                    stats_by_lambda[lambda_c]["omp"].update(omp_val)
                    stats_by_lambda[lambda_c]["random"].update(rnd_val)
                    medians_by_lambda[lambda_c]["dt"].update(dt_val)
                    medians_by_lambda[lambda_c]["omp"].update(omp_val)
                    medians_by_lambda[lambda_c]["random"].update(rnd_val)
                    steps_decision_stats[lambda_c].update(float(steps))
                    k_selected_stats[lambda_c].update(float(k_sel))
                    k_selected_quantiles[lambda_c]["p50"].update(float(k_sel))
                    k_selected_quantiles[lambda_c]["p90"].update(float(k_sel))
                    k_selected_quantiles[lambda_c]["p99"].update(float(k_sel))
                    k_selected_hist[lambda_c][k_sel] += 1

                    if per_sample_fp is not None:
                        row = {
                            "clip_idx": int(clip_idx),
                            "window_idx": int(w_idx),
                            "freq_idx": int(f),
                            "lambda_c": float(lambda_c),
                            "k_selected": int(k_sel),
                            "steps_decision": int(steps),
                            "dt_capture": dt_val,
                            "omp_capture": omp_val,
                            "random_capture": rnd_val,
                        }
                        per_sample_fp.write(json.dumps(row) + "\n")

            if subsample_fp is not None:
                if mic_wav_pad is None or ldv_wav_pad is None or segment_len_samples is None:
                    raise RuntimeError("subsample enabled but waveform padding is not initialized.")
                if bp_b is None or bp_a is None:
                    raise RuntimeError("subsample enabled but bandpass filter is not initialized.")
                if dt_med_by_lambda is None:
                    raise RuntimeError("subsample enabled but dt_med_by_lambda is missing.")

                seg_start = int(start_t) * int(args.hop_length)
                seg_end = int(seg_start) + int(segment_len_samples)
                segment_oob = (
                    seg_start < 0
                    or seg_end > int(mic_wav_pad.size)
                    or seg_end > int(ldv_wav_pad.size)
                    or seg_end <= seg_start
                )
                search_radius_samples = int(args.search_radius_frames) * int(args.hop_length)

                if segment_oob:
                    for lambda_c in lambda_c_values:
                        for coarse_source, coarse_lag_frames in (("dt", dt_med_by_lambda[lambda_c]), ("omp", omp_med)):
                            subsample_counts_by_lambda[lambda_c][coarse_source]["num_windows"] += 1
                            subsample_counts_by_lambda[lambda_c][coarse_source]["num_segment_oob"] += 1
                            row = {
                                "pairing_mode": str(args.pairing_mode),
                                "clip_idx": int(clip_idx),
                                "window_idx": int(w_idx),
                                "start_t": int(start_t),
                                "lambda_c": float(lambda_c),
                                "method": "gcc_phat",
                                "coarse_source": str(coarse_source),
                                "coarse_lag_frames": coarse_lag_frames,
                                "coarse_delay_samples": None
                                if coarse_lag_frames is None
                                else int(round(float(coarse_lag_frames) * float(args.hop_length))),
                                "search_radius_samples": int(search_radius_samples),
                                "gcc_phat_tau_hat_samples": None,
                                "gcc_phat_tau_hat_ms": None,
                                "gcc_phat_psr": None,
                                "gcc_phat_boundary_hit": None,
                                "undefined_reason": "segment_oob",
                            }
                            subsample_fp.write(json.dumps(row) + "\n")
                else:
                    x_seg = mic_wav_pad[seg_start:seg_end]
                    y_seg = ldv_wav_pad[seg_start:seg_end]
                    if x_seg.size != y_seg.size:
                        raise RuntimeError(
                            f"Segment length mismatch at clip={clip_idx} window={w_idx}: {x_seg.size} vs {y_seg.size}"
                        )

                    tau_centers = []
                    for lambda_c in lambda_c_values:
                        dt_frames = dt_med_by_lambda[lambda_c]
                        if dt_frames is not None:
                            tau_centers.append(int(round(float(dt_frames) * float(args.hop_length))))
                    if omp_med is not None:
                        tau_centers.append(int(round(float(omp_med) * float(args.hop_length))))
                    max_abs_tau = max((abs(int(t)) for t in tau_centers), default=0)
                    need_half = int(max_abs_tau) + int(search_radius_samples) + 1
                    base_nfft = next_pow2(2 * int(x_seg.size))
                    nfft = base_nfft if need_half <= base_nfft // 2 else next_pow2(4 * int(x_seg.size))

                    lags, abs_cc = gcc_phat_abs_cc(x_seg, y_seg, b=bp_b, a=bp_a, nfft=int(nfft))

                    for lambda_c in lambda_c_values:
                        for coarse_source, coarse_lag_frames in (("dt", dt_med_by_lambda[lambda_c]), ("omp", omp_med)):
                            subsample_counts_by_lambda[lambda_c][coarse_source]["num_windows"] += 1
                            if coarse_lag_frames is None:
                                subsample_counts_by_lambda[lambda_c][coarse_source]["num_coarse_undefined"] += 1
                                row = {
                                    "pairing_mode": str(args.pairing_mode),
                                    "clip_idx": int(clip_idx),
                                    "window_idx": int(w_idx),
                                    "start_t": int(start_t),
                                    "lambda_c": float(lambda_c),
                                    "method": "gcc_phat",
                                    "coarse_source": str(coarse_source),
                                    "coarse_lag_frames": None,
                                    "coarse_delay_samples": None,
                                    "search_radius_samples": int(search_radius_samples),
                                    "gcc_phat_tau_hat_samples": None,
                                    "gcc_phat_tau_hat_ms": None,
                                    "gcc_phat_psr": None,
                                    "gcc_phat_boundary_hit": None,
                                    "undefined_reason": "coarse_undefined",
                                }
                                subsample_fp.write(json.dumps(row) + "\n")
                                continue

                            coarse_delay_samples = int(round(float(coarse_lag_frames) * float(args.hop_length)))
                            tau_hat_s, tau_hat_ms, psr, bhit = gcc_phat_pick_peak(
                                lags,
                                abs_cc,
                                fs=int(args.fs),
                                tau_center_samples=int(coarse_delay_samples),
                                search_radius_samples=int(search_radius_samples),
                            )

                            if tau_hat_s is None or tau_hat_ms is None:
                                row = {
                                    "pairing_mode": str(args.pairing_mode),
                                    "clip_idx": int(clip_idx),
                                    "window_idx": int(w_idx),
                                    "start_t": int(start_t),
                                    "lambda_c": float(lambda_c),
                                    "method": "gcc_phat",
                                    "coarse_source": str(coarse_source),
                                    "coarse_lag_frames": float(coarse_lag_frames),
                                    "coarse_delay_samples": int(coarse_delay_samples),
                                    "search_radius_samples": int(search_radius_samples),
                                    "gcc_phat_tau_hat_samples": None,
                                    "gcc_phat_tau_hat_ms": None,
                                    "gcc_phat_psr": None,
                                    "gcc_phat_boundary_hit": None,
                                    "undefined_reason": "gcc_phat_undefined",
                                }
                                subsample_fp.write(json.dumps(row) + "\n")
                                continue

                            subsample_counts_by_lambda[lambda_c][coarse_source]["num_defined"] += 1
                            subsample_tau_quantiles_by_lambda[lambda_c][coarse_source]["p50"].update(float(tau_hat_ms))
                            subsample_tau_quantiles_by_lambda[lambda_c][coarse_source]["p90"].update(float(tau_hat_ms))
                            if psr is not None:
                                subsample_psr_quantiles_by_lambda[lambda_c][coarse_source]["p50"].update(float(psr))
                                subsample_psr_quantiles_by_lambda[lambda_c][coarse_source]["p90"].update(float(psr))
                            if bhit is not None:
                                subsample_boundary_rate_by_lambda[lambda_c][coarse_source].update(
                                    1.0 if bool(bhit) else 0.0
                                )
                            subsample_clip_taus_ms_by_lambda[lambda_c][coarse_source][clip_idx].append(float(tau_hat_ms))

                            row = {
                                "pairing_mode": str(args.pairing_mode),
                                "clip_idx": int(clip_idx),
                                "window_idx": int(w_idx),
                                "start_t": int(start_t),
                                "lambda_c": float(lambda_c),
                                "method": "gcc_phat",
                                "coarse_source": str(coarse_source),
                                "coarse_lag_frames": float(coarse_lag_frames),
                                "coarse_delay_samples": int(coarse_delay_samples),
                                "search_radius_samples": int(search_radius_samples),
                                "gcc_phat_tau_hat_samples": float(tau_hat_s),
                                "gcc_phat_tau_hat_ms": float(tau_hat_ms),
                                "gcc_phat_psr": None if psr is None else float(psr),
                                "gcc_phat_boundary_hit": None if bhit is None else bool(bhit),
                            }
                            subsample_fp.write(json.dumps(row) + "\n")

            num_samples_total += int(F_band)

        if (clip_idx + 1) % log_every == 0:
            logger.info("Processed clip %d / %d", clip_idx + 1, num_pairs)

    if per_sample_fp is not None:
        per_sample_fp.close()
    diagnostics_fp.close()
    if delay_fp is not None:
        delay_fp.close()
    if subsample_fp is not None:
        subsample_fp.close()

    compute_rows = []
    k_selected_means = []
    steps_decision_means = []
    for lambda_c in lambda_c_values:
        dt_mean = stats_by_lambda[lambda_c]["dt"].mean()
        omp_mean = stats_by_lambda[lambda_c]["omp"].mean()
        rnd_mean = stats_by_lambda[lambda_c]["random"].mean()
        dt_med = medians_by_lambda[lambda_c]["dt"].result()
        omp_med = medians_by_lambda[lambda_c]["omp"].result()
        rnd_med = medians_by_lambda[lambda_c]["random"].result()
        k_sel_mean = k_selected_stats[lambda_c].mean()
        k_sel_std = k_selected_stats[lambda_c].std()
        steps_mean = steps_decision_stats[lambda_c].mean()
        steps_std = steps_decision_stats[lambda_c].std()
        k_selected_means.append(k_sel_mean)
        steps_decision_means.append(steps_mean)
        compute_rows.append(
            {
                "lambda_c": float(lambda_c),
                "dt_capture_mean": dt_mean,
                "omp_capture_mean": omp_mean,
                "random_capture_mean": rnd_mean,
                "dt_capture_median": dt_med,
                "omp_capture_median": omp_med,
                "random_capture_median": rnd_med,
                "dt_over_omp_mean": None if dt_mean is None or omp_mean is None else float(dt_mean / (omp_mean + 1e-12)),
                "dt_minus_random_mean": None if dt_mean is None or rnd_mean is None else float(dt_mean - rnd_mean),
                "k_selected_mean": k_sel_mean,
                "k_selected_std": k_sel_std,
                "steps_decision_mean": steps_mean,
                "steps_decision_std": steps_std,
            }
        )

    forced_rows = []
    for k in forced_k_values:
        dt_mean = forced_stats[k]["dt"].mean()
        omp_mean = forced_stats[k]["omp"].mean()
        rnd_mean = forced_stats[k]["random"].mean()
        forced_rows.append(
            {
                "k": int(k),
                "dt_capture_mean": dt_mean,
                "omp_capture_mean": omp_mean,
                "random_capture_mean": rnd_mean,
                "dt_over_omp_mean": None if dt_mean is None or omp_mean is None else float(dt_mean / (omp_mean + 1e-12)),
                "dt_minus_random_mean": None if dt_mean is None or rnd_mean is None else float(dt_mean - rnd_mean),
            }
        )

    if any(s is None for s in k_selected_means):
        spearman_k = None
    else:
        spearman_k = spearmanr(lambda_c_values, k_selected_means)
    if any(s is None for s in steps_decision_means):
        spearman_steps = None
    else:
        spearman_steps = spearmanr(lambda_c_values, steps_decision_means)

    k_selected_range = None
    if k_selected_means and all(s is not None for s in k_selected_means):
        k_selected_range = float(max(k_selected_means) - min(k_selected_means))
    steps_decision_range = None
    if steps_decision_means and all(s is not None for s in steps_decision_means):
        steps_decision_range = float(max(steps_decision_means) - min(steps_decision_means))

    num_samples_used = num_samples_total
    for method, stats in capture_minmax.items():
        if not np.isfinite(stats["min"]):
            stats["min"] = None
        if not np.isfinite(stats["max"]):
            stats["max"] = None
    num_capture_out_of_range_dt = out_of_range_counts["dt"]
    num_capture_out_of_range_omp = out_of_range_counts["omp"]
    num_capture_out_of_range_random = out_of_range_counts["random"]
    num_capture_out_of_range_total = (
        num_capture_out_of_range_dt + num_capture_out_of_range_omp + num_capture_out_of_range_random
    )
    random_duplicate_rate = (
        float(random_duplicate_total) / float(random_duplicate_possible)
        if random_duplicate_possible > 0
        else 0.0
    )
    integrity = {
        "num_samples_total": int(num_samples_total),
        "num_samples_used": int(num_samples_used),
        "num_missing_files": int(missing_files),
        "num_md5_mismatches": int(md5_mismatches),
        "num_nan_or_inf": int(num_nan_or_inf),
        "num_capture_out_of_range_total": int(num_capture_out_of_range_total),
        "num_capture_out_of_range_dt": int(num_capture_out_of_range_dt),
        "num_capture_out_of_range_omp": int(num_capture_out_of_range_omp),
        "num_capture_out_of_range_random": int(num_capture_out_of_range_random),
        "num_omp_monotonicity_violations": int(num_omp_monotonicity_violations),
        "num_dt_duplicate_actions_forced_k": int(num_dt_duplicate_actions_forced_k),
        "capture_minmax": capture_minmax,
        "worst_violation": worst_violation["details"],
        "random_duplicate_count": int(random_duplicate_total),
        "random_duplicate_possible": int(random_duplicate_possible),
        "random_duplicate_rate": float(random_duplicate_rate),
    }

    compute_summary = {
        "config": {
            "mic_root": str(mic_root),
            "ldv_root": str(ldv_root),
            "ckpt_path": str(args.ckpt_path),
            "subset_manifest": str(manifest_path),
            "mode": args.mode,
            "pairing_mode": str(args.pairing_mode),
            "require_wav_only": bool(int(args.require_wav_only)),
            "num_pairs": int(num_pairs),
            "hop_length": int(args.hop_length),
            "fs": int(args.fs),
            "n_fft": int(args.n_fft),
            "freq_min": float(args.freq_min),
            "freq_max": float(args.freq_max),
            "max_lag": int(args.max_lag),
            "max_k": int(args.max_k),
            "tw": int(args.tw),
            "gain": float(args.gain),
            "rtg_dim": int(args.rtg_dim),
            "eps_energy": float(args.eps_energy),
            "lambda_c_values": lambda_c_values,
            "random_trials": int(args.random_trials),
            "random_sampling": str(args.random_sampling),
            "forced_k_rtg0_lambda_c": float(lambda_c_values[0]),
            "device": str(device),
            "write_per_sample": bool(int(args.write_per_sample)),
            "write_delay_diagnostics": bool(int(args.write_delay_diagnostics)),
            "write_subsample_delay_diagnostics": bool(int(args.write_subsample_delay_diagnostics)),
            "subsample_method": str(args.subsample_method),
            "search_radius_frames": int(args.search_radius_frames),
        },
        "integrity": integrity,
        "rows": compute_rows,
    }

    forced_summary = {
        "config": compute_summary["config"],
        "integrity": integrity,
        "rows": forced_rows,
    }

    controllability_summary = {
        "lambda_c_values": lambda_c_values,
        "k_selected_mean": k_selected_means,
        "k_selected_std": [row["k_selected_std"] for row in compute_rows],
        "k_selected_range": k_selected_range,
        "k_selected_quantiles": {
            "p50": [k_selected_quantiles[l]["p50"].result() for l in lambda_c_values],
            "p90": [k_selected_quantiles[l]["p90"].result() for l in lambda_c_values],
            "p99": [k_selected_quantiles[l]["p99"].result() for l in lambda_c_values],
        },
        "k_selected_histogram": {
            str(l): [int(v) for v in k_selected_hist[l].tolist()] for l in lambda_c_values
        },
        "p_k_selected_lt_max_k": [
            float(np.sum(k_selected_hist[l][:-1]) / max(int(np.sum(k_selected_hist[l])), 1))
            for l in lambda_c_values
        ],
        "p_k_selected_leq_12": [
            float(np.sum(k_selected_hist[l][: min(12, int(args.max_k)) + 1]) / max(int(np.sum(k_selected_hist[l])), 1))
            for l in lambda_c_values
        ],
        "steps_decision_mean": steps_decision_means,
        "steps_decision_std": [row["steps_decision_std"] for row in compute_rows],
        "steps_decision_range": steps_decision_range,
        "spearman_lambda_k_selected": spearman_k,
        "spearman_lambda_steps_decision": spearman_steps,
    }

    (out_dir / "summary" / "compute_matched_summary.json").write_text(
        json.dumps(compute_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "summary" / "forced_k_summary.json").write_text(
        json.dumps(forced_summary, indent=2), encoding="utf-8"
    )
    (out_dir / "summary" / "rtg_controllability_summary.json").write_text(
        json.dumps(controllability_summary, indent=2), encoding="utf-8"
    )

    if int(args.write_delay_diagnostics) == 1:
        delay_rows = []
        num_windows_total = None
        for lambda_c in lambda_c_values:
            num_w = int(delay_counts_by_lambda[lambda_c]["num_windows"])
            if num_windows_total is None:
                num_windows_total = num_w
            elif num_w != num_windows_total:
                raise RuntimeError(
                    "delay_diagnostics window count mismatch across lambdas: "
                    f"expected {num_windows_total}, got {num_w} at lambda_c={lambda_c}"
                )

            dt_mad_p50 = delay_mad_quantiles_by_lambda[lambda_c]["p50"].result()
            dt_mad_p90 = delay_mad_quantiles_by_lambda[lambda_c]["p90"].result()
            delay_rows.append(
                {
                    "lambda_c": float(lambda_c),
                    "num_windows": num_w,
                    "num_windows_dt_first_lag_undefined": int(
                        delay_counts_by_lambda[lambda_c]["num_windows_dt_first_lag_undefined"]
                    ),
                    "num_windows_dt_match_undefined": int(
                        delay_counts_by_lambda[lambda_c]["num_windows_dt_match_undefined"]
                    ),
                    "dt_stop0_frac_mean": delay_stats_by_lambda[lambda_c]["dt_stop0_frac"].mean(),
                    "dt_stop0_frac_std": delay_stats_by_lambda[lambda_c]["dt_stop0_frac"].std(),
                    "dt_first_lag_median_frames_mean": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_median_frames"
                    ].mean(),
                    "dt_first_lag_median_frames_std": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_median_frames"
                    ].std(),
                    "dt_first_lag_mad_frames_mean": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_mad_frames"
                    ].mean(),
                    "dt_first_lag_mad_frames_std": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_mad_frames"
                    ].std(),
                    "dt_first_lag_mad_frames_p50": dt_mad_p50,
                    "dt_first_lag_mad_frames_p90": dt_mad_p90,
                    "dt_first_lag_abs_le_1_frac_mean": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_abs_le_1_frac"
                    ].mean(),
                    "dt_first_lag_abs_le_2_frac_mean": delay_stats_by_lambda[lambda_c][
                        "dt_first_lag_abs_le_2_frac"
                    ].mean(),
                    "dt_k_selected_mean_window_mean": delay_stats_by_lambda[lambda_c]["dt_k_selected_mean"].mean(),
                    "dt_steps_decision_mean_window_mean": delay_stats_by_lambda[lambda_c]["dt_steps_decision_mean"].mean(),
                    "dt_capture_mean_window_mean": delay_stats_by_lambda[lambda_c]["dt_capture_mean"].mean(),
                    "omp_first_lag_median_frames_mean": delay_stats_by_lambda[lambda_c][
                        "omp_first_lag_median_frames"
                    ].mean(),
                    "omp_first_lag_mad_frames_mean": delay_stats_by_lambda[lambda_c]["omp_first_lag_mad_frames"].mean(),
                    "dt_vs_omp_first_lag_match_frac_mean": delay_stats_by_lambda[lambda_c][
                        "dt_vs_omp_first_lag_match_frac"
                    ].mean(),
                }
            )

        delay_integrity = {
            "num_windows_total": int(num_windows_total) if num_windows_total is not None else 0,
            "num_rows_total": int((num_windows_total or 0) * len(lambda_c_values)),
        }
        delay_summary = {
            "config": compute_summary["config"],
            "integrity": delay_integrity,
            "rows": delay_rows,
        }
        (out_dir / "summary" / "delay_diagnostics_summary.json").write_text(
            json.dumps(delay_summary, indent=2), encoding="utf-8"
        )

    if int(args.write_subsample_delay_diagnostics) == 1:
        subsample_rows = []
        num_windows_total = None
        for lambda_c in lambda_c_values:
            num_w_dt = int(subsample_counts_by_lambda[lambda_c]["dt"]["num_windows"])
            num_w_omp = int(subsample_counts_by_lambda[lambda_c]["omp"]["num_windows"])
            if num_w_dt != num_w_omp:
                raise RuntimeError(
                    "subsample_delay window count mismatch across coarse sources: "
                    f"dt={num_w_dt}, omp={num_w_omp} at lambda_c={lambda_c}"
                )
            if num_windows_total is None:
                num_windows_total = num_w_dt
            elif num_w_dt != num_windows_total:
                raise RuntimeError(
                    "subsample_delay window count mismatch across lambdas: "
                    f"expected {num_windows_total}, got {num_w_dt} at lambda_c={lambda_c}"
                )

            def _source_row(source: str) -> Dict[str, object]:
                counts = subsample_counts_by_lambda[lambda_c][source]
                num_windows = int(counts["num_windows"])
                num_defined = int(counts["num_defined"])
                frac_defined = float(num_defined / num_windows) if num_windows > 0 else None
                boundary_hit_rate = subsample_boundary_rate_by_lambda[lambda_c][source].mean()

                tau_p50 = subsample_tau_quantiles_by_lambda[lambda_c][source]["p50"].result()
                tau_p90 = subsample_tau_quantiles_by_lambda[lambda_c][source]["p90"].result()
                psr_p50 = subsample_psr_quantiles_by_lambda[lambda_c][source]["p50"].result()
                psr_p90 = subsample_psr_quantiles_by_lambda[lambda_c][source]["p90"].result()

                clip_mads = []
                for clip_i in range(int(num_pairs)):
                    taus = subsample_clip_taus_ms_by_lambda[lambda_c][source][clip_i]
                    if len(taus) == 0:
                        continue
                    med = float(np.median(taus))
                    mad = float(np.median(np.abs(np.asarray(taus, dtype=np.float64) - med)))
                    clip_mads.append(mad)
                within_p50 = float(np.percentile(clip_mads, 50)) if clip_mads else None
                within_p90 = float(np.percentile(clip_mads, 90)) if clip_mads else None

                return {
                    "num_windows": num_windows,
                    "num_defined": num_defined,
                    "num_segment_oob": int(counts["num_segment_oob"]),
                    "num_coarse_undefined": int(counts["num_coarse_undefined"]),
                    "fraction_defined": frac_defined,
                    "boundary_hit_rate": boundary_hit_rate,
                    "tau_hat_ms_p50": tau_p50,
                    "tau_hat_ms_p90": tau_p90,
                    "psr_p50": psr_p50,
                    "psr_p90": psr_p90,
                    "within_clip_tau_mad_ms_p50": within_p50,
                    "within_clip_tau_mad_ms_p90": within_p90,
                }

            subsample_rows.append(
                {
                    "lambda_c": float(lambda_c),
                    "dt": _source_row("dt"),
                    "omp": _source_row("omp"),
                }
            )

        subsample_integrity = {
            "num_windows_total": int(num_windows_total) if num_windows_total is not None else 0,
            "num_rows_total": int((num_windows_total or 0) * len(lambda_c_values) * 2),
        }
        subsample_summary = {
            "config": compute_summary["config"],
            "integrity": subsample_integrity,
            "rows": subsample_rows,
        }
        (out_dir / "summary" / "subsample_delay_diagnostics_summary.json").write_text(
            json.dumps(subsample_summary, indent=2), encoding="utf-8"
        )

    logger.info("Wrote summaries to %s", out_dir / "summary")


if __name__ == "__main__":
    main()
