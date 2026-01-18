import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import scipy.signal
import torch

from scripts.h_exploration.dataset_lag import DoALagDataset


@dataclass(frozen=True)
class Band:
    name: str
    hz_min: float
    hz_max: float


def hz_to_bin(hz: float, *, fs: int = 16000, n_fft: int = 2048) -> int:
    # STFT bin spacing = fs/n_fft
    return int(round(hz / (fs / n_fft)))


def _sample_bins(b0: int, b1: int, n: int) -> np.ndarray:
    if b1 <= b0:
        return np.array([], dtype=int)
    if n >= (b1 - b0):
        return np.arange(b0, b1, dtype=int)
    return np.linspace(b0, b1 - 1, n, dtype=int)


def band_coherence(
    mic_stft: np.ndarray,
    ldv_stft: np.ndarray,
    bins: np.ndarray,
    *,
    fs: int,
    nperseg: int,
) -> float:
    # Coherence over time index (STFT frames). Note: inputs are complex.
    # This is *not* waveform coherence, but is consistent with the STFT-lag modeling pipeline.
    T = mic_stft.shape[0]
    nperseg = min(nperseg, T)
    vals: List[float] = []
    for b in bins:
        _, cxy = scipy.signal.coherence(mic_stft[:, b], ldv_stft[:, b], fs=fs, nperseg=nperseg)
        vals.append(float(np.nanmean(cxy)))
    return float(np.mean(vals)) if vals else float("nan")


def run_omp_reduction(
    X_chunk: torch.Tensor,
    Y_chunk: torch.Tensor,
    *,
    k_list: List[int],
) -> Dict[int, np.ndarray]:
    """OMP reduction for a chunk.

    X_chunk: (MaxLag+Tw, F) complex
    Y_chunk: (Tw, F) complex

    Returns: dict K -> reductions array (F,)
    """
    assert X_chunk.ndim == 2 and Y_chunk.ndim == 2

    Tw, F = Y_chunk.shape
    MaxLag = X_chunk.shape[0] - Tw
    assert MaxLag > 0

    device = X_chunk.device

    # Build dictionary: (F, Tw, MaxLag)
    Dict_tensor = torch.zeros(F, Tw, MaxLag, dtype=X_chunk.dtype, device=device)
    for k in range(MaxLag):
        start_row = MaxLag - k
        end_row = MaxLag + Tw - k
        Dict_tensor[:, :, k] = X_chunk[start_row:end_row, :].T

    Targets = Y_chunk.T.unsqueeze(2)  # (F, Tw, 1)

    # Normalize dict atoms (per F, per lag)
    d_norms = torch.linalg.vector_norm(Dict_tensor.abs(), dim=1, keepdim=True) + 1e-8
    Dict_norm = Dict_tensor / d_norms

    max_k = max(k_list)
    Active = torch.full((F, max_k), -1, dtype=torch.long, device=device)
    Residuals = Targets.clone()

    init_energy = torch.linalg.vector_norm(Targets, dim=1).squeeze() ** 2  # (F,)

    out: Dict[int, np.ndarray] = {}

    for step in range(max_k):
        Corrs = torch.bmm(Dict_norm.conj().transpose(1, 2), Residuals)  # (F, MaxLag, 1)
        Abs = torch.abs(Corrs).squeeze(2)  # (F, MaxLag)

        if step > 0:
            for f in range(F):
                prev = Active[f, :step]
                prev = prev[prev >= 0]
                if len(prev) > 0:
                    Abs[f, prev] = -1.0

        best = torch.argmax(Abs, dim=1)
        Active[:, step] = best

        # LS update per frequency bin
        for f in range(F):
            idx = Active[f, : step + 1]
            A = Dict_tensor[f, :, idx]  # (Tw, k)
            y = Targets[f, :, 0]
            h = torch.linalg.lstsq(A, y).solution.flatten()
            recon = A @ h
            Residuals[f, :, 0] = y - recon

        k_now = step + 1
        if k_now in k_list:
            cur_energy = torch.linalg.vector_norm(Residuals, dim=1).squeeze() ** 2
            mask = init_energy > 1e-9
            red = torch.zeros_like(init_energy)
            red[mask] = (init_energy[mask] - cur_energy[mask]) / init_energy[mask]
            out[k_now] = red.detach().cpu().numpy()

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_original_16k_no_edge_sync_vad_normalized")
    ap.add_argument("--ldv_root", type=str, default="/Users/sbplab/LDV-data-processed/speech260_box_16k_no_edge_sync_vad_normalized")
    ap.add_argument("--angle", type=float, default=90.0)
    ap.add_argument("--clip_ids", type=str, default="0,50,100", help="Comma-separated clip indices")
    ap.add_argument("--bins_per_band", type=int, default=64)
    ap.add_argument("--coh_nperseg", type=int, default=32)
    ap.add_argument("--max_lag", type=int, default=50)
    ap.add_argument("--tw", type=int, default=16)
    ap.add_argument("--k_list", type=str, default="16,50")
    ap.add_argument("--start_t", type=int, default=None, help="If None, uses max_lag + 16")
    args = ap.parse_args()

    device = torch.device("cpu")

    bands = [
        Band("LowMids_250_500", 250, 500),
        Band("Mids_500_2000", 500, 2000),
        Band("UpperMids_2000_4000", 2000, 4000),
        Band("Presence_4000_6000", 4000, 6000),
        Band("Highs_6000_8000", 6000, 8000),
    ]

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]

    ds = DoALagDataset(args.mic_root, args.ldv_root, angle=args.angle)

    clip_ids = [int(x) for x in args.clip_ids.split(",") if x.strip()]
    clip_ids = [i for i in clip_ids if 0 <= i < len(ds)]

    start_t = args.start_t if args.start_t is not None else (args.max_lag + 16)

    # Aggregate
    coh_agg: Dict[str, List[float]] = {b.name: [] for b in bands}
    red_agg: Dict[str, Dict[int, List[float]]] = {b.name: {k: [] for k in k_list} for b in bands}

    for ci in clip_ids:
        item = ds[ci]
        mic = item["mic_stft"].to(device)
        ldv = item["ldv_stft"].to(device)

        T, F = mic.shape
        if start_t + args.tw >= T:
            continue

        # Prepare chunk for OMP
        MaxLag = args.max_lag
        Tw = args.tw
        X_chunk = mic[start_t - MaxLag : start_t + Tw]
        Y_chunk = ldv[start_t : start_t + Tw]

        # Convert for coherence
        mic_np = mic.cpu().numpy()
        ldv_np = ldv.cpu().numpy()

        for band in bands:
            b0 = hz_to_bin(band.hz_min)
            b1 = hz_to_bin(band.hz_max)
            b0 = max(0, min(F - 1, b0))
            b1 = max(0, min(F, b1))

            chosen = _sample_bins(b0, b1, args.bins_per_band)
            if chosen.size == 0:
                continue

            # Coherence
            coh = band_coherence(mic_np, ldv_np, chosen, fs=16000, nperseg=args.coh_nperseg)
            coh_agg[band.name].append(coh)

            # OMP reductions on the same chosen bins only
            Xb = X_chunk[:, chosen]
            Yb = Y_chunk[:, chosen]
            reds = run_omp_reduction(Xb, Yb, k_list=k_list)
            for k in k_list:
                red_agg[band.name][k].append(float(np.mean(reds[k])))

    print("Band diagnostic (STFT-domain coherence + OMP ceiling by K)")
    print(f"Clips used: {clip_ids}")
    print(f"STFT assumed: fs=16000, n_fft=2048, default overlap=75% (hop=512 samples = 32ms)")
    print(f"OMP chunk: max_lag={args.max_lag} frames, tw={args.tw} frames, start_t={start_t}")
    print(f"Bins per band sampled: {args.bins_per_band}")
    print("")

    for band in bands:
        coh_vals = coh_agg[band.name]
        if coh_vals:
            coh_mean = float(np.mean(coh_vals))
        else:
            coh_mean = float("nan")

        parts = [f"{band.name:>20}: coh_mean={coh_mean:.4f}"]

        for k in k_list:
            vals = red_agg[band.name][k]
            m = float(np.mean(vals)) if vals else float("nan")
            parts.append(f"K={k}: red={m:.2%}")

        if len(k_list) >= 2:
            k0, k1 = k_list[0], k_list[1]
            v0 = float(np.mean(red_agg[band.name][k0])) if red_agg[band.name][k0] else float("nan")
            v1 = float(np.mean(red_agg[band.name][k1])) if red_agg[band.name][k1] else float("nan")
            parts.append(f"delta({k0}->{k1})={v1 - v0:+.2%}")

        print("  " + " | ".join(parts))


if __name__ == "__main__":
    main()
