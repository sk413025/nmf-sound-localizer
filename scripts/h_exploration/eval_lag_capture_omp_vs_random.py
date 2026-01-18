import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from scripts.h_exploration.dataset_lag import DoALagDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def manual_complex_norm(tensor, dim=None, keepdim=False):
    if dim is None:
        return torch.sqrt(tensor.real.pow(2) + tensor.imag.pow(2) + 1e-10)
    return torch.sqrt(
        tensor.real.pow(2).sum(dim=dim, keepdim=keepdim)
        + tensor.imag.pow(2).sum(dim=dim, keepdim=keepdim)
        + 1e-10
    )


def gather_atoms(Dict, Ids_till_k):
    # Dict: (F, Tw, M)
    # Ids: (F, k)
    F, Tw, M = Dict.shape
    K_curr = Ids_till_k.shape[1]
    ids_exp = Ids_till_k.unsqueeze(1).expand(-1, Tw, -1)
    return torch.gather(Dict, 2, ids_exp)


@torch.no_grad()
def eval_one_clip(
    dataset,
    idx: int,
    device: torch.device,
    *,
    max_lag: int,
    max_k: int,
    tw: int,
    gain: float,
    stride: int,
    random_trials: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    item = dataset[idx]
    mic_stft = item["mic_stft"].to(device)
    ldv_stft = item["ldv_stft"].to(device)

    if gain != 1.0:
        mic_stft = mic_stft * float(gain)
        ldv_stft = ldv_stft * float(gain)

    T_total, F = ldv_stft.shape

    Lag_Min = -max_lag
    Lag_Max = max_lag
    Lags = torch.arange(Lag_Min, Lag_Max + 1, device=device)
    M_lags = len(Lags)

    start_limit = Lag_Max
    end_limit = T_total - tw + Lag_Min
    if end_limit <= start_limit:
        return None

    valid_starts = list(range(start_limit, end_limit, stride))
    if len(valid_starts) == 0:
        return None

    # Accumulate final energy reduction per window (flattened across freq)
    final_red_omp = []
    final_red_rnd = []

    for start_t in valid_starts:
        end_t = start_t + tw
        y_block = ldv_stft[start_t:end_t, :].T  # (F, Tw)

        # Build dictionary: (F, Tw, M)
        D = torch.zeros(F, tw, M_lags, dtype=mic_stft.dtype, device=device)
        for m, k_lag in enumerate(Lags):
            s = int(start_t - k_lag)
            e = int(s + tw)
            D[:, :, m] = mic_stft[s:e, :].T

        # Normalize dictionary for correlations only
        norms = manual_complex_norm(D, dim=1).unsqueeze(1) + 1e-8
        D_norm = D / norms

        Y = y_block.unsqueeze(2)  # (F, Tw, 1)
        initial_energy = manual_complex_norm(Y.squeeze(), dim=1) ** 2  # (F,)

        # === OMP ===
        res = Y.clone()
        chosen = torch.zeros(F, 0, dtype=torch.long, device=device)
        for _k in range(max_k):
            corrs = torch.bmm(D_norm.conj().transpose(1, 2), res).squeeze(2)
            abs_corrs = torch.abs(corrs)
            if chosen.numel() > 0:
                abs_corrs.scatter_(1, chosen, -1.0)
            act = torch.argmax(abs_corrs, dim=1).unsqueeze(1)
            chosen = torch.cat([chosen, act], dim=1)

            # Projection (CPU for stability on complex)
            active = gather_atoms(D.cpu(), chosen.cpu())
            Y_cpu = Y.cpu()
            sol = torch.linalg.lstsq(active, Y_cpu).solution
            recon = active @ sol
            res = (Y_cpu - recon).to(device)

        res_energy = manual_complex_norm((Y_cpu - recon).squeeze(), dim=1) ** 2
        red = (initial_energy.cpu() - res_energy) / (initial_energy.cpu() + 1e-6)
        final_red_omp.append(red)

        # === Random ===
        # For each trial, sample a random set of K unique lags per frequency, project, and score.
        # Report the mean across trials.
        trial_reds = []
        for _t in range(random_trials):
            # Per-frequency random permutation of [0..M-1], take first K
            rand_ids = torch.empty(F, max_k, dtype=torch.long)
            for f in range(F):
                perm = rng.permutation(M_lags)[:max_k]
                rand_ids[f] = torch.from_numpy(perm)

            active_r = gather_atoms(D.cpu(), rand_ids)
            Y_cpu = Y.cpu()
            sol_r = torch.linalg.lstsq(active_r, Y_cpu).solution
            recon_r = active_r @ sol_r

            res_energy_r = manual_complex_norm((Y_cpu - recon_r).squeeze(), dim=1) ** 2
            red_r = (initial_energy.cpu() - res_energy_r) / (initial_energy.cpu() + 1e-6)
            trial_reds.append(red_r)

        mean_red_r = torch.stack(trial_reds, dim=0).mean(dim=0)
        final_red_rnd.append(mean_red_r)

    omp_all = torch.cat(final_red_omp, dim=0)
    rnd_all = torch.cat(final_red_rnd, dim=0)

    return {
        "omp_mean": float(omp_all.mean().item()),
        "rnd_mean": float(rnd_all.mean().item()),
        "gap_mean": float((omp_all.mean() - rnd_all.mean()).item()),
        "n_windows": int(len(valid_starts)),
        "F": int(F),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mic_root", type=str, required=True)
    p.add_argument("--ldv_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--angle", type=float, default=90.0)
    p.add_argument("--hop_length", type=int, default=160, help="STFT hop length (samples)")
    p.add_argument("--max_items", type=int, default=3)
    p.add_argument("--max_lag", type=int, default=50)
    p.add_argument("--max_k", type=int, default=16)
    p.add_argument("--tw", type=int, default=32)
    p.add_argument("--gain", type=float, default=100.0)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--random_trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset = DoALagDataset(
        args.mic_root, args.ldv_root, angle=float(args.angle), hop_length=int(args.hop_length)
    )
    n = min(int(args.max_items), len(dataset))

    results = {
        "config": {
            "mic_root": args.mic_root,
            "ldv_root": args.ldv_root,
            "angle": float(args.angle),
            "max_items": int(n),
            "max_lag": int(args.max_lag),
            "max_k": int(args.max_k),
            "tw": int(args.tw),
            "gain": float(args.gain),
            "stride": int(args.stride),
            "random_trials": int(args.random_trials),
            "seed": int(args.seed),
        },
        "per_clip": [],
    }

    for i in tqdm(range(n), desc="OMP vs Random"):
        r = eval_one_clip(
            dataset,
            i,
            device,
            max_lag=int(args.max_lag),
            max_k=int(args.max_k),
            tw=int(args.tw),
            gain=float(args.gain),
            stride=int(args.stride),
            random_trials=int(args.random_trials),
            seed=int(args.seed) + i,
        )
        if r is None:
            continue
        r["clip_idx"] = int(i)
        results["per_clip"].append(r)

    if len(results["per_clip"]) == 0:
        raise RuntimeError("No valid windows found; check Tw/max_lag/stride and dataset lengths")

    omp_mean = float(np.mean([c["omp_mean"] for c in results["per_clip"]]))
    rnd_mean = float(np.mean([c["rnd_mean"] for c in results["per_clip"]]))
    gap_mean = float(np.mean([c["gap_mean"] for c in results["per_clip"]]))

    results["summary"] = {
        "omp_mean": omp_mean,
        "rnd_mean": rnd_mean,
        "gap_mean": gap_mean,
        "n_clips_used": int(len(results["per_clip"])),
    }

    out_path = out_dir / "omp_vs_random_summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote {out_path}")
    logger.info(f"Summary: OMP={omp_mean:.4f}, Random={rnd_mean:.4f}, Gap={gap_mean:.4f}")


if __name__ == "__main__":
    main()
